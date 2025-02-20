import argparse
import ast
import glob
import logging
import multiprocessing
import os
import pdb
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from functools import partial

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
from music_df.salami_slice import appears_salami_sliced
from music_df.script_helpers import get_stoi, read_config_oc
from music_df.sync_df import get_unique_from_array_by_df


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type is not KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook

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
    # We allow dictionary_folder to be a sequence for the case where
    #   we are getting harmony onsets from a separate run
    dictionary_folder: tuple[str, ...] | str | None = None
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
    ignore_features: tuple[str, ...] = ()

    csv_prefix_to_strip: None | str = None
    csv_prefix_to_add: None | str = None
    column_types: dict[str, str] = field(default_factory=lambda: {})
    output_folder: str = DEFAULT_OUTPUT
    uniform_step: int = 8
    smoothing: None | int = None
    collated: bool = False
    num_workers: int = 8
    # If take_only_distinct_salami_slices is False, then we take each salami slice
    #   regardless of whether there is any pitch change.
    take_only_distinct_salami_slices: bool = True

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


def handle_metadata(metadata_rows, reference_df: pd.DataFrame | None, config: Config):
    out_metadata_df = pd.DataFrame(metadata_rows).sort_index()
    if reference_df is None:
        df_path = os.path.join(config.output_folder, os.path.basename(config.metadata))
        out_metadata_df.to_csv(df_path)
        print(f"Wrote {df_path}")
        return out_metadata_df
    else:
        assert out_metadata_df.equals(reference_df)
        return reference_df


def item_handler(
    config: Config,
    h5_path,
    output_path,
    metadata_df,
    h5out_lock,
    feature_name,
    metadata_i,
):
    metadata_row = metadata_df.loc[metadata_i]

    logits_i = metadata_row.get(f"{feature_name}_map", metadata_row.name)

    try:
        with h5py.File(h5_path, mode="r") as h5file:
            logits: np.ndarray = (h5file[f"logits_{logits_i}"])[:]  # type:ignore
    except KeyError:
        warnings.warn(f"Key 'logits_{logits_i}' not found in {h5_path}, skipping")
        return

    if config.data_has_start_and_stop_tokens:
        logits = logits[1:-1]

    logits = logits[:, config.n_specials :]

    prev_csv_path = metadata_row.csv_path
    assert isinstance(prev_csv_path, str)
    if not os.path.exists(prev_csv_path):
        # At a certain point I started normalizing unicode to ASCII
        #   characters, but some paths will be saved differently
        prev_csv_path = normalize_string(prev_csv_path)

    music_df = read_csv(prev_csv_path)
    assert music_df is not None, (
        f"Error reading '{prev_csv_path}', maybe it doesn't exist?"
    )

    if config.take_only_distinct_salami_slices:
        assert "distinct_slice_id" in music_df.columns
    else:
        assert appears_salami_sliced(music_df)

    df_indices = metadata_row.df_indices
    if isinstance(df_indices, str):
        df_indices = ast.literal_eval(df_indices)

    if not config.collated and min(df_indices) != df_indices[0]:
        LOGGER.warning(f"skipping a file because min(df_indices) != df_indices[0]")

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

    if "note_count" in metadata_row:
        try:
            assert logits.shape[0] == metadata_row["note_count"]
        except AssertionError:
            warnings.warn(
                f"Logits shape and note count don't match for 'logits_{logits_i}' "
                f"and '{metadata_row['csv_path']}', skipping"
            )
            return

    if len(logits) < len(notes_df):
        LOGGER.error(f"{metadata_row.csv_path} length of logits < length of notes")
        return

    # For fair comparison with AugmentedNet, we "sync" the logits (averaging the
    #   values across all logits of each salami slice) and then take just one
    #   value per salami slice

    if config.take_only_distinct_salami_slices:
        col_name = "distinct_slice_id"
    else:
        col_name = "onset"

    logits, note_indices = get_unique_from_array_by_df(
        logits,
        notes_df,
        unique_col_name_or_names=col_name,
        sync_col_name_or_names=col_name,
        return_indices=True,
    )

    per_salami_slice_df_indices = notes_df.loc[note_indices, "raw_indices"]

    # Get uniform_steps
    uniform_df = quantize_df(
        notes_df.loc[note_indices],
        tpq=config.uniform_step,
        ticks_out=True,
        zero_dur_action="preserve",
    )
    uniform_steps = uniform_df["release"] - uniform_df["onset"]

    output_metadata_row = metadata_row.copy()
    output_metadata_row["df_indices"] = per_salami_slice_df_indices.tolist()
    output_metadata_row["uniform_steps"] = uniform_steps.tolist()

    with h5out_lock:
        with h5py.File(output_path, mode="a") as outf:
            outf.create_dataset(f"logits_{logits_i}", data=logits)

    return output_metadata_row


class NullContextManager:
    # Used in place of multiprocessing.Lock
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


def handle_existing(output_path):
    if os.path.exists(output_path):
        print(f"Warning: found existing {output_path} file, removing")
        os.remove(output_path)


def process_h5(
    h5_path, metadata_df, config: Config, feature_name, stoi, concat_feature
):
    output_path = os.path.join(config.output_folder, f"{feature_name}.h5")
    handle_existing(output_path)
    # outf = open(output_path, "w", newline="")
    # assert len(metadata_df) >= len(h5file)

    output_metadata_rows = []

    # prev_csv_path: None | str = None
    # music_df: pd.DataFrame | None = None

    if config.num_workers > 1:
        manager = multiprocessing.Manager()
        h5out_lock = manager.Lock()
    else:
        h5out_lock = NullContextManager()

    partial_handler = partial(
        item_handler,
        config,
        h5_path,
        output_path,
        metadata_df,
        h5out_lock,
        feature_name,
    )

    row_indices = metadata_df.index

    if config.num_workers > 1:
        with multiprocessing.Pool(config.num_workers) as pool:
            output_metadata_rows = list(
                tqdm(
                    pool.imap_unordered(partial_handler, row_indices, chunksize=32),
                    total=len(row_indices),
                )
            )
    else:
        for i in tqdm(row_indices, total=len(row_indices)):
            output_metadata_rows.append(partial_handler(i))

    # Filter None values from output rows
    output_metadata_rows = [r for r in output_metadata_rows if r is not None]
    print("\n")
    print(f"Wrote {output_path}")

    return output_metadata_rows


def main():
    args = parse_args()
    config = read_config_oc(args.config_file, args.remaining, Config)
    assert isinstance(config.dictionary_folder, tuple)
    dictionary_paths = []
    for f in config.dictionary_folder:
        dictionary_paths += glob.glob(os.path.join(f, "*_dictionary.txt"))

    predictions_paths = glob.glob(os.path.join(config.predictions, "*.h5"))
    exit_if_output_is_newer(
        dictionary_paths + predictions_paths + [config.metadata],
        glob.glob(os.path.join(config.output_folder, "**", "*"), recursive=True),
    )

    metadata_df = pd.read_csv(config.metadata, index_col=0)
    reference_out_metadata_df = None
    os.makedirs(config.output_folder, exist_ok=True)
    if os.path.isdir(config.predictions):
        if config.dictionary_folder is None:
            raise ValueError
        else:
            stoi_vocabs = get_stoi(dictionary_paths)
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
