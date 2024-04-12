import argparse
import ast
import glob
import logging
import os
from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from music_df import read_csv
from music_df.salami_slice import appears_salami_sliced
from music_df.script_helpers import get_stoi, read_config_oc
from music_df.sync_df import get_unique_from_array_by_df

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_OUTPUT = os.path.expanduser(os.path.join("~", "output", "metrics"))


@dataclass
class Config:
    # metadata: path to metadata csv containing at least the following columns:
    # csv_path, df_indices. Rows should be in one-to-one correspondance with
    # predictions.
    metadata: str
    # predictions: h5 file containing predicted tokens. Rows
    # should be in one-to-one correspondance with metadata.
    predictions: str
    dictionary_folder: str | None = None
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
    )
    concat_features: tuple[tuple[str, ...], ...] = ()
    csv_prefix_to_strip: None | str = None
    csv_prefix_to_add: None | str = None
    column_types: dict[str, str] = field(default_factory=lambda: {})
    output_folder: str = DEFAULT_OUTPUT
    uniform_step: None | int = 8
    smoothing: None | int = None
    collated: bool = False

    @property
    def concat_feature_names(self):
        return ["_".join(f) for f in self.concat_features]


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


def handle_metadata(metadata_rows, reference_df: pd.DataFrame | None, config: Config):
    out_metadata_df = pd.DataFrame(metadata_rows)
    if reference_df is None:
        df_path = os.path.join(config.output_folder, os.path.basename(config.metadata))
        out_metadata_df.to_csv(df_path)
        print(f"Wrote {df_path}")
        return out_metadata_df
    else:
        assert out_metadata_df.equals(reference_df)
        return reference_df


def process_h5(
    h5_path, metadata_df, config: Config, feature_name, stoi, concat_feature
):
    h5file = h5py.File(h5_path, mode="r")

    output_path = os.path.join(config.output_folder, f"{feature_name}.h5")
    outf = open(output_path, "w", newline="")
    assert len(metadata_df) >= len(h5file)

    output_metadata_rows = []

    with h5py.File(output_path, mode="w") as outf:

        prev_csv_path: None | str = None
        music_df: pd.DataFrame | None = None

        for i in tqdm(range(len(h5file))):
            metadata_row = metadata_df.iloc[i]
            logits: np.ndarray = (h5file[f"logits_{i}"])[:]  # type:ignore

            if config.data_has_start_and_stop_tokens:
                logits = logits[1:-1]

            if prev_csv_path is None or metadata_row.csv_path != prev_csv_path:
                prev_csv_path = metadata_row.csv_path
                assert isinstance(prev_csv_path, str)
                music_df = read_csv(prev_csv_path)
                assert music_df is not None
                assert appears_salami_sliced(music_df)

            assert music_df is not None

            df_indices = metadata_row.df_indices
            if isinstance(df_indices, str):
                df_indices = ast.literal_eval(df_indices)

            if not config.collated and min(df_indices) != df_indices[0]:
                LOGGER.warning(
                    f"skipping a file because min(df_indices) != df_indices[0]"
                )

            # This former strategy for cropping led to incorrect results sometimes:
            # cropped_df = crop_df(music_df, start_i=min(df_indices), end_i=max(df_indices))
            cropped_df = music_df.loc[df_indices]
            assert cropped_df.type.unique().tolist() == ["note"]

            notes_df = cropped_df.reset_index(names="raw_indices")

            # if concat_feature:
            #     feature_i = config.concat_feature_names.index(feature_name)
            #     features = config.concat_features[feature_i]
            #     notes_df = concatenate_features(notes_df, features)

            # Trim pad tokens
            logits = logits[: len(notes_df)]

            if len(logits) < len(notes_df):
                LOGGER.error(
                    f"{metadata_row.csv_path} length of logits < length of notes"
                )
                continue

            # For fair comparison with AugmentedNet, we "sync" the logits (averaging the
            #   values across all logits of each salami slice) and then take just one
            #   value per salami slice
            logits, note_indices = get_unique_from_array_by_df(
                logits,
                notes_df,
                unique_col_name_or_names="onset",
                sync_col_name_or_names="onset",
                return_indices=True,
            )

            per_salami_slice_df_indices = notes_df.loc[note_indices, "raw_indices"]

            output_metadata_row = metadata_row.copy()
            output_metadata_row["df_indices"] = per_salami_slice_df_indices.tolist()
            output_metadata_rows.append(output_metadata_row)
            # if config.smoothing:
            #     # Adjacent logits might not be on the same scale, so we take probabilities
            #     #   first so that smoothing will not favor larger magnitude logits
            #     # probs = softmax(logits)
            #     # assert probs.ndim == 2
            #     # smoothed = moving_average(probs, axis=0, n=config.smoothing)
            #     # predicted_indices = smoothed.argmax(axis=-1)
            # else:
            # predicted_indices = logits.argmax(axis=-1)

            # predicted_indices -= config.n_specials
            # if predicted_indices.min() < 0:
            #     breakpoint()
            #     LOGGER.warning(
            #         f"Predicted at least one special token in {metadata_row.csv_path}; "
            #         "replacing with 0"
            #     )
            #     predicted_indices[predicted_indices < 0] = 0
            outf.create_dataset(f"logits_{i}", data=logits)
            # outf.create_dataset(f"indices_{i}", data=note_indices)

    print("\n")
    print(f"Wrote {output_path}")
    return output_metadata_rows


def main():
    args = parse_args()
    config = read_config_oc(args.config_file, args.remaining, Config)
    metadata_df = pd.read_csv(config.metadata)
    reference_out_metadata_df = None
    os.makedirs(config.output_folder, exist_ok=True)
    if os.path.isdir(config.predictions):
        if config.dictionary_folder is None:
            raise ValueError
        else:
            dictionary_paths = glob.glob(
                os.path.join(config.dictionary_folder, "*_dictionary.txt")
            )
            stoi_vocabs = get_stoi(dictionary_paths)
            predictions_paths = glob.glob(os.path.join(config.predictions, "*.h5"))
            if not predictions_paths:
                raise ValueError(f"No h5 files found in {config.predictions}")
            for predictions_path in predictions_paths:
                this_feature_name = os.path.basename(
                    os.path.splitext(predictions_path)[0]
                )
                concat_feature = this_feature_name in config.concat_feature_names
                if (this_feature_name not in config.features) and not concat_feature:
                    continue
                # if (
                #     config.feature_names
                #     and this_feature_name not in config.feature_names
                # ):
                #     continue
                output_metadata_rows = process_h5(
                    predictions_path,
                    metadata_df,
                    config,
                    this_feature_name,
                    stoi_vocabs[this_feature_name],
                    concat_feature,
                )
                reference_out_metadata_df = handle_metadata(
                    output_metadata_rows, reference_out_metadata_df, config
                )
    else:
        raise ValueError(f"{config.predictions} does not exist or is not a directory")


if __name__ == "__main__":
    main()
