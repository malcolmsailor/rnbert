import os
import sys
from dataclasses import dataclass
from omegaconf import OmegaConf
import h5py
import pandas as pd
import ast
from music_df import read_csv

from music_df.script_helpers import get_single_stoi
import csv
from music_df.add_feature import concatenate_features

import traceback, pdb, sys


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type != KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


@dataclass
class Config:
    metadata: str
    output_path: str
    feature_name: str
    dictionary_path: str
    concat_feature: tuple[str, str] | None = None
    predictions_h5: str | None = None
    predictions_txt: str | None = None


def read_h5_row(h5file, i):
    preds: np.ndarray = (h5file[f"predictions_{i}"])[:]  # type:ignore
    return preds


def read_txt_row(txtfile, i):
    raise NotImplementedError


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)

    outf = open(config.output_path, "w", newline="")
    writer = csv.writer(outf)
    writer.writerow(
        ["path", "indices", "uniform_steps", "labels", "predicted", "note_indices"]
    )

    metadata_df = pd.read_csv(config.metadata)
    assert (metadata_df.index == pd.RangeIndex(len(metadata_df))).all()
    assert len(metadata_df) == len(
        metadata_df.csv_path.unique()
    ), "Only 1 row per score is supported"

    assert config.predictions_h5 is not None or config.predictions_txt is not None

    stoi = get_single_stoi(config.dictionary_path)

    if config.predictions_h5:
        inf = h5py.File(config.predictions_h5, "r")
        assert len(metadata_df) == len(inf)
        read_row = read_h5_row
    else:
        raise NotImplementedError
        read_row = read_txt_row

    for i, metadata_row in metadata_df.iterrows():
        predicted_indices = read_row(inf, i)

        df_indices = metadata_row.df_indices
        if isinstance(df_indices, str):
            df_indices = ast.literal_eval(df_indices)

        music_df = read_csv(metadata_row.csv_path)
        assert music_df is not None

        cropped_df = music_df.loc[df_indices]
        assert cropped_df.type.unique().tolist() == ["note"]
        assert len(cropped_df) == len(predicted_indices)
        if config.concat_feature:
            cropped_df = concatenate_features(cropped_df, config.concat_feature)

        label_indices = [stoi[str(label)] for label in cropped_df[config.feature_name]]
        writer.writerow(
            [
                metadata_row.csv_path,
                df_indices,
                "",
                label_indices,
                predicted_indices.tolist(),
                df_indices,
            ]
        )
    outf.close()
    print(f"Wrote {config.output_path}")


if __name__ == "__main__":
    main()
