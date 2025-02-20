import argparse
import ast
import csv
import glob
import logging
import os
import pdb
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Literal

import h5py
import numpy as np
import pandas as pd

try:
    from disscode.files import exit_if_output_is_newer
except ImportError:
    # For when this script is copied into the RNBERT repo where disscode is not
    # available
    def exit_if_output_is_newer(*args, **kwargs):
        pass


try:
    from disscode.script_helpers import normalize_string
except ImportError:
    # For when this script is copied into the RNBERT repo where disscode is not
    # available
    import unicodedata

    def normalize_string(s):
        # Annoying special case found in When-in-Rome data:
        s = s.replace("’", "'")
        s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode()
        return s


from tqdm import tqdm

from music_df import quantize_df, read_csv
from music_df.add_feature import concatenate_features
from music_df.salami_slice import (
    appears_salami_sliced,
    get_unique_salami_slices,
)
from music_df.script_helpers import get_csv_path, get_itos, get_stoi, read_config_oc
from music_df.sync_df import get_unique_from_array_by_df

try:
    from disscode.utils.moving_average import moving_average

    DISSCODE_UNAVAILABLE = False
except ImportError:
    DISSCODE_UNAVAILABLE = True

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_OUTPUT = os.path.expanduser(os.path.join("~", "output", "metrics"))


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type is not KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


@dataclass
class Config:
    # metadata: path to metadata csv containing at least the following columns:
    # csv_path, df_indices. Rows should be in one-to-one correspondance with
    # predictions.
    metadata: str
    # predictions: h5 file containing predicted tokens. Rows
    # should be in one-to-one correspondance with metadata.
    predictions: str
    # We allow dictionary_folder to be a sequence for the case where
    #   we are getting harmony onsets from a separate run
    dictionary_folder: tuple[str, ...] | str | None = None

    # In the event where we get predictions on a fairseq dataset that was separately
    #   binarized, the counts of token types may differ, and (since fairseq) sorts
    #   vocabularies by frequency, this means the dictionaries will differ. In that
    #   case we also need the "original" dictionary the model was trained with
    #   so we can translate from one to the other.
    original_dictionary_folder: str | None = None

    debug: bool = False
    # When predicting tokens we need to subtract the number of specials
    n_specials: int = 4
    data_has_start_and_stop_tokens: bool = False
    features: tuple[str, ...] = (
        "harmony_onset",
        "primary_degree",
        "primary_alteration",
        "secondary_degree",
        "secondary_alteration",
        "inversion",
        "mode",
        "quality",
        "key_pc",
        "tonicization",
        "scale_degree",
        "root_pc",
    )
    concat_features: tuple[tuple[str, ...], ...] = ()
    ignore_features: tuple[str, ...] = ()

    csv_prefix_to_strip: None | str = None
    csv_prefix_to_add: None | str = None
    column_types: dict[str, str] = field(default_factory=lambda: {})
    output_folder: str = DEFAULT_OUTPUT
    uniform_step: None | int = 8
    smoothing: None | int = None
    collated: bool = False

    average: Literal["logits", "probs"] = "logits"

    @property
    def concat_feature_names(self):
        return ["_".join(f) for f in self.concat_features]

    def __post_init__(self):
        if isinstance(self.ignore_features, str):
            self.ignore_features = (self.ignore_features,)

        if self.dictionary_folder is not None:
            if isinstance(self.dictionary_folder, str):
                self.dictionary_folder = (self.dictionary_folder,)
            self.dictionary_folder = tuple(self.dictionary_folder)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=False)
    # remaining passed through to omegaconf
    parser.add_argument("remaining", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def softmax(x: np.ndarray, axis: int = -1):
    exponentiated = np.exp(x)
    summed = np.sum(exponentiated, axis=axis, keepdims=True)
    out = exponentiated / summed
    return out


def process_h5(
    h5_path,
    metadata_df,
    config: Config,
    feature_name,
    stoi,
    concat_feature,
    index_translators,
):
    h5file = h5py.File(h5_path, mode="r")

    output_path = os.path.join(config.output_folder, f"{feature_name}.csv")
    outf = open(output_path, "w", newline="")
    writer = csv.writer(outf)
    writer.writerow(
        ["path", "indices", "uniform_steps", "labels", "predicted", "note_indices"]
    )

    output_h5_path = os.path.join(config.output_folder, f"{feature_name}.h5")
    output_h5 = h5py.File(output_h5_path, mode="w")

    assert len(metadata_df) >= len(h5file)
    prev_csv_path: None | str = None
    music_df: pd.DataFrame | None = None

    for i in tqdm(range(len(h5file))):
        metadata_row = metadata_df.iloc[i]
        try:
            logits: np.ndarray = (h5file[f"logits_{i}"])[:]  # type:ignore
        except KeyError:
            warnings.warn(f"Key 'logits_{i}' not found in {h5file}, skipping")
            continue

        if config.data_has_start_and_stop_tokens:
            logits = logits[1:-1]

        if prev_csv_path is None or metadata_row.csv_path != prev_csv_path:
            prev_csv_path = metadata_row.csv_path
            assert isinstance(prev_csv_path, str)
            if not os.path.exists(prev_csv_path):
                prev_csv_path = normalize_string(prev_csv_path)

            # LOGGER.info(f"Reading {get_csv_path(prev_csv_path, config)}")
            music_df = read_csv(get_csv_path(prev_csv_path, config))
            if music_df is None:
                continue
            assert appears_salami_sliced(music_df)

        if music_df is None:
            continue

        df_indices = metadata_row.df_indices
        if isinstance(df_indices, str):
            df_indices = ast.literal_eval(df_indices)

        if not config.collated and min(df_indices) != df_indices[0]:
            LOGGER.warning("skipping a file because min(df_indices) != df_indices[0]")

        # This former strategy for cropping led to incorrect results sometimes:
        # cropped_df = crop_df(music_df, start_i=min(df_indices), end_i=max(df_indices))
        cropped_df = music_df.loc[df_indices]
        assert cropped_df.type.unique().tolist() == ["note"]

        notes_df = cropped_df.reset_index(drop=True)

        if concat_feature:
            feature_i = config.concat_feature_names.index(feature_name)
            features = config.concat_features[feature_i]
            notes_df = concatenate_features(notes_df, features)

        unique_slices = get_unique_salami_slices(notes_df)
        labels = unique_slices[feature_name]
        # If we have an unknown token here, I'm not sure what the best procedure is.
        #   For now, returning the length of stoi.
        label_indices = [stoi.get(str(label), len(stoi)) for label in labels]

        # Trim pad tokens
        logits = logits[: len(notes_df)]

        if len(logits) < len(notes_df):
            LOGGER.error(f"{metadata_row.csv_path} length of logits < length of notes")
            continue

        # For fair comparison with AugmentedNet, we "sync" the logits (averaging the
        #   values across all logits of each salami slice) and then take just one
        #   value per salami slice
        if config.average == "logits":
            logits, note_indices = get_unique_from_array_by_df(
                logits,
                notes_df,
                unique_col_name_or_names="onset",
                sync_col_name_or_names="onset",
                return_indices=True,
            )

            if config.smoothing:
                assert not DISSCODE_UNAVAILABLE
                # Adjacent logits might not be on the same scale, so we take probabilities
                #   first so that smoothing will not favor larger magnitude logits
                probs = softmax(logits)
                assert probs.ndim == 2
                smoothed = moving_average(probs, axis=0, n=config.smoothing)  # type:ignore
                predicted_indices = smoothed.argmax(axis=-1)
            else:
                predicted_indices = logits.argmax(axis=-1)
        elif config.average == "probs":
            if config.smoothing:
                raise NotImplementedError

            probs = softmax(logits)
            probs, note_indices = get_unique_from_array_by_df(
                probs,
                notes_df,
                unique_col_name_or_names="onset",
                sync_col_name_or_names="onset",
                return_indices=True,
            )
            predicted_indices = probs.argmax(axis=-1)
        else:
            raise ValueError

        predicted_indices -= config.n_specials
        if predicted_indices.min() < 0:
            LOGGER.warning(
                f"Predicted at least one special token in {metadata_row.csv_path}; "
                "replacing with 0"
            )
            predicted_indices[predicted_indices < 0] = 0

        if config.uniform_step:
            # notes_df["predicted_indices"] = predicted_indices
            unique_slices = quantize_df(
                unique_slices,
                tpq=config.uniform_step,
                ticks_out=True,
                zero_dur_action="preserve",
            )
            uniform_steps = (unique_slices["release"] - unique_slices["onset"]).tolist()
            assert len(uniform_steps) == len(label_indices) == len(predicted_indices)
        else:
            uniform_steps = None
            assert len(label_indices) == len(predicted_indices)

        if index_translators:
            predicted_indices = np.array(
                [index_translators[i] for i in predicted_indices]
            )

        output_h5.create_dataset(
            f"predictions_{i}", data=np.array(predicted_indices, dtype=np.int32)
        )

        writer.writerow(
            [
                prev_csv_path,
                df_indices,
                uniform_steps,
                label_indices,
                predicted_indices.tolist(),
                note_indices.tolist(),
            ]
        )
    print(f"Wrote {output_path}")
    print(f"Wrote {output_h5_path}")


def main():
    args = parse_args()
    config = read_config_oc(args.config_file, args.remaining, Config)

    metadata_df = pd.read_csv(config.metadata)
    os.makedirs(config.output_folder, exist_ok=True)
    if os.path.isdir(config.predictions):
        if config.dictionary_folder is None:
            raise ValueError
        else:
            assert isinstance(config.dictionary_folder, tuple)
            dictionary_paths = []
            for f in config.dictionary_folder:
                dictionary_paths += glob.glob(os.path.join(f, "*_dictionary.txt"))
            predictions_paths = glob.glob(os.path.join(config.predictions, "*.h5"))
            exit_if_output_is_newer(
                predictions_paths + dictionary_paths + [config.metadata],
                glob.glob(
                    os.path.join(config.output_folder, "**", "*"), recursive=True
                ),
            )

            stoi_vocabs = get_stoi(dictionary_paths)
            if config.original_dictionary_folder is not None:
                orig_dictionary_paths = glob.glob(
                    os.path.join(config.original_dictionary_folder, "*_dictionary.txt")
                )
                orig_itos_vocabs = get_itos(orig_dictionary_paths)
                index_translators = {}
                for key in orig_itos_vocabs:
                    itos = orig_itos_vocabs[key]
                    stoi = stoi_vocabs[key]
                    index_translators[key] = [stoi[itos[i]] for i in range(len(itos))]
            else:
                index_translators = None

            if not predictions_paths:
                raise ValueError(f"No h5 files found in {config.predictions}")
            for predictions_path in predictions_paths:
                this_feature_name = os.path.basename(
                    os.path.splitext(predictions_path)[0]
                )
                if this_feature_name in config.ignore_features:
                    continue
                concat_feature = this_feature_name in config.concat_feature_names
                if (this_feature_name not in config.features) and not concat_feature:
                    continue
                # if (
                #     config.feature_names
                #     and this_feature_name not in config.feature_names
                # ):
                #     continue
                process_h5(
                    predictions_path,
                    metadata_df,
                    config,
                    this_feature_name,
                    stoi_vocabs[this_feature_name],
                    concat_feature,
                    (
                        index_translators[this_feature_name]
                        if index_translators is not None
                        else None
                    ),
                )
    else:
        raise ValueError(f"{config.predictions} does not exist or is not a directory")


if __name__ == "__main__":
    main()
