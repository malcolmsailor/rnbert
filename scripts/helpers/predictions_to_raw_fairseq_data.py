"""
input_csv: "predictions csv" output of one of my scripts containing "predicted" column
    whose values are lists of integer indices
dictionary_path: path to fairseq dictionary for this token type
output_path: each example becomes a space-separated line of tokens in this file, 
    prepended by the start token
uncollated metadata: to move from collated/per-salami-sliced predictions back to
    uncollated predictions, we need the uncollated metadata file
comparison path: 
"""

from ast import literal_eval
import math
import os
import sys
from dataclasses import dataclass
from omegaconf import OmegaConf
import pandas as pd
from tqdm import tqdm
from typing import Optional

import traceback, pdb, sys


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type != KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


@dataclass
class Config:
    input_csv: str
    dictionary_path: str
    output_path: str
    uncollated_metadata: str
    # To check the input, we see how it compares with the labeled data
    comparison_path: Optional[str] = None
    start_token: str = "<s>"


def get_single_itos(dictionary_path: str):
    # copied this function here rather than doing
    #   `from music_df.script_helpers import get_single_itos`
    # to avoid music_df dependency for this script
    feature_name = os.path.basename(dictionary_path).rsplit("_", maxsplit=1)[0]
    with open(dictionary_path) as inf:
        data = inf.readlines()
    return [
        line.split(" ", maxsplit=1)[0]
        for line in data
        if line and not line.startswith("madeupword")
    ]


def get_indices(row, uncollated_metadata):
    # the indices are both collated and then per-salami-sliced, so we
    #   need to invert both of these operations to get our dataset
    #   of predictions

    uncollated_rows = uncollated_metadata[uncollated_metadata["csv_path"] == row.path]

    # indices1 = set(literal_eval(row.indices))
    indices1 = {indx: i for (i, indx) in enumerate(literal_eval(row.indices))}
    indices2 = set()
    for _, row2 in uncollated_rows.iterrows():
        indices2 |= set(literal_eval(row2.df_indices))
    assert all(i in indices2 for i in indices1)
    index_df = pd.DataFrame({"uncollated": sorted(indices2), "collated": float("nan")})
    index_df = index_df.set_index("uncollated", drop=True)
    sorted_indices1 = sorted(indices1)
    index_df.loc[sorted_indices1, "collated"] = sorted_indices1

    # In the (unusual) case where there are nans at the start of index_df["collated"]
    #   we just take the first non-nan token
    if math.isnan(index_df["collated"].iloc[0]):
        index_df["collated"].iloc[0] = sorted_indices1[0]

    index_df["collated"] = index_df["collated"].ffill().astype(int)
    index_df["out"] = [indices1[x] for x in index_df["collated"]]

    out = []
    for _, uncollated_row in uncollated_rows.iterrows():
        idxs = literal_eval(uncollated_row.df_indices)
        mapped_indices = index_df.loc[idxs, "out"]
        out.append(mapped_indices)

    return out


def check_indices(uncollated_metadata, data):
    paths1 = sorted(data.path.unique())
    paths2 = sorted(uncollated_metadata.csv_path.unique())
    assert paths1 == paths2
    assert len(paths1) == len(data)


def compare_output(config):
    with open(config.output_path) as inf:
        output_lines = [l.strip().split() for l in inf.readlines()]
    with open(config.comparison_path) as inf:
        comparison_lines = [l.strip().split() for l in inf.readlines()]
    assert all(len(x) == len(y) for (x, y) in zip(output_lines, comparison_lines))
    assert all(x[0] == config.start_token for x in output_lines)
    assert all(x[0] == config.start_token for x in comparison_lines)
    print(f"Output passed assertions")
    n_agree = 0
    n_total = 0
    for x, y in zip(output_lines, comparison_lines):
        n_total += len(x)
        n_agree += sum([xx == yy for (xx, yy) in zip(x, y)])
    print("Agreement with comparison:", n_agree / n_total)


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore

    uncollated_metadata = pd.read_csv(config.uncollated_metadata)
    data = pd.read_csv(config.input_csv)
    itos = get_single_itos(config.dictionary_path)

    check_indices(uncollated_metadata, data)

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)

    with open(config.output_path, "w") as outf:
        for _, row in tqdm(data.iterrows(), total=len(data)):
            pred_indices = literal_eval(row["predicted"])
            pred_tokens = [itos[i] for i in pred_indices]
            df_indices = get_indices(row, uncollated_metadata)
            for row in df_indices:
                these_tokens = [pred_tokens[i] for i in row]
                outf.write(" ".join([config.start_token] + these_tokens))
                outf.write("\n")

    print(f"Wrote {config.output_path}")

    if config.comparison_path:
        compare_output(config)
    else:
        print(
            "Warning: skipping output checks because `config.comparison_path` was not provided"
        )


if __name__ == "__main__":
    main()
