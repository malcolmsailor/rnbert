import ast
import csv
import os
import pdb
import sys
import traceback
import warnings
from dataclasses import dataclass

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


from omegaconf import OmegaConf
from tqdm import tqdm

from music_df import read_csv
from music_df.add_feature import concatenate_features
from music_df.script_helpers import get_single_stoi


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type is not KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


@dataclass
class Config:
    metadata: str
    output_path: str
    dictionary_path: str | None = None
    feature_name: str | None = None
    concat_feature: tuple[str, str] | None = None
    predictions_h5: str | None = None
    predictions_txt: str | None = None

    uncollated: bool = False


def read_h5_row(h5file, i):
    try:
        preds: np.ndarray = (h5file[f"predictions_{i}"])[:]  # type:ignore
    except KeyError:
        try:
            logits: np.ndarray = (h5file[f"logits_{i}"])[:]  # type:# type:ignore
        except KeyError:
            return None
        else:
            preds = logits.argmax(-1)
    return preds


def read_txt_row(txtfile, i):
    raise NotImplementedError


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore

    exit_if_output_is_newer(
        [config.metadata, config.predictions_h5]
        + ([config.dictionary_path] if config.dictionary_path else []),
        [config.output_path],
    )

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)

    outf = open(config.output_path, "w", newline="")
    writer = csv.writer(outf)
    col_names = [
        "",
        "path",
        "indices",
        "uniform_steps",
        "labels",
        "predicted",
        "note_indices",
    ]
    if not config.feature_name:
        col_names.remove("labels")
    writer.writerow(col_names)

    metadata_df = pd.read_csv(config.metadata, index_col=0)

    # assert (metadata_df.index == pd.RangeIndex(len(metadata_df))).all()
    # TODO: (Malcolm 2024-06-27) I'm not sure why only 1 row per score would be supported
    #    It seems it should work fine on uncollated data.
    # assert len(metadata_df) == len(
    #     metadata_df.csv_path.unique()
    # ), "Only 1 row per score is supported"

    assert config.predictions_h5 is not None or config.predictions_txt is not None

    if config.feature_name:
        assert config.dictionary_path
        stoi = get_single_stoi(config.dictionary_path)
    else:
        stoi = None

    if config.predictions_h5:
        inf = h5py.File(config.predictions_h5, "r")
        read_row = read_h5_row
    else:
        raise NotImplementedError
        read_row = read_txt_row

    for row_i, metadata_row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        predicted_indices = read_row(inf, row_i)
        if predicted_indices is None:
            warnings.warn(f"Key {row_i} not found in predictions, skipping")
            continue

        df_indices = metadata_row.df_indices
        if isinstance(df_indices, str):
            df_indices = ast.literal_eval(df_indices)

        csv_path = metadata_row.csv_path
        if not os.path.exists(csv_path):
            csv_path = normalize_string(csv_path)

        if config.feature_name:
            music_df = read_csv(csv_path)
            assert music_df is not None

            cropped_df = music_df.loc[df_indices]
            assert cropped_df.type.unique().tolist() == ["note"]
            assert len(cropped_df) == len(predicted_indices)
            if config.concat_feature:
                cropped_df = concatenate_features(cropped_df, config.concat_feature)

            assert stoi is not None
            label_indices = [
                stoi.get(str(label), len(stoi))
                for label in cropped_df[config.feature_name]
            ]
            writer.writerow(
                [
                    row_i,
                    csv_path,
                    df_indices,
                    metadata_row.get("uniform_steps", ""),
                    label_indices,
                    predicted_indices.tolist(),
                    df_indices,
                ]
            )
        else:
            writer.writerow(
                [
                    row_i,
                    csv_path,
                    df_indices,
                    metadata_row.get("uniform_steps", ""),
                    predicted_indices.tolist(),
                    df_indices,
                ]
            )
    outf.close()
    print(f"Wrote {config.output_path}")


if __name__ == "__main__":
    main()
