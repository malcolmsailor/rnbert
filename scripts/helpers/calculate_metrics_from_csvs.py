import argparse
import ast

import numpy as np
import pandas as pd
import sklearn.metrics

DEBUG = True
if DEBUG:
    import pdb
    import sys
    import traceback

    def custom_excepthook(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)

    sys.excepthook = custom_excepthook


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", nargs="+")
    parser.add_argument("--uniform-steps", action="store_true")
    parser.add_argument(
        "--uniform-steps-csv",
        type=str,
        help="""
the csv_paths output by scripts/get_csv_of_predictions_and_labels.py
don't contain uniform_steps, but we can get them from a csv file output
by scripts/get_per_salami_slice_preds.py""",
    )
    parser.add_argument(
        "--output-file", default=None, help="We will append a row to this csv file"
    )
    parser.add_argument(
        "--key",
        default=None,
        help="If present, first cell in output csv row (i.e., an index name)",
    )
    parser.add_argument(
        "--filter-labels",
        type=str,
        default=None,
        help=(
            "For scale degree alteration and possibly other tasks, "
            "we only want to look at the predictions for certain labels. A comma-"
            "separated list of integers, e.g. '1,2' or '1,2,4'"
        ),
    )
    parser.add_argument("--per-class", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if len(args.input_files) == 1:
        df = pd.read_csv(args.input_files[0])
        paths_df_indices = df[["path", "indices"]]
        df = df.drop(["path", "indices"], axis=1)
        y_true = np.concatenate([np.array(ast.literal_eval(x)) for x in df["labels"]])
        y_pred = np.concatenate(
            [np.array(ast.literal_eval(x)) for x in df["predicted"]]
        )
    else:
        df = pd.read_csv(args.input_files[0])
        paths_df_indices = df[["path", "indices"]]
        df = df.drop(["path", "indices"], axis=1)
        for i, input_file in enumerate(args.input_files[1:], start=1):
            new_df = pd.read_csv(input_file).drop(["path", "indices"], axis=1)
            df = df.merge(
                new_df,
                left_index=True,
                right_index=True,
                suffixes=("", f"_{i}"),
                how="outer",
            )
        df = df.rename({"predicted": "predicted_0", "labels": "labels_0"}, axis=1)
        y_trues = [
            np.concatenate([np.array(ast.literal_eval(x)) for x in df[f"labels_{i}"]])
            for i in range(len(args.input_files))
        ]
        y_preds = [
            np.concatenate(
                [np.array(ast.literal_eval(x)) for x in df[f"predicted_{i}"]]
            )
            for i in range(len(args.input_files))
        ]

        y_true = ["_".join(str(x) for x in xs) for xs in zip(*y_trues)]
        y_pred = ["_".join(str(x) for x in xs) for xs in zip(*y_preds)]

    if args.uniform_steps:
        if args.uniform_steps_csv:
            uniform_step_df = pd.read_csv(args.uniform_steps_csv)
            assert (paths_df_indices["path"] == uniform_step_df["path"]).all()
            assert (paths_df_indices["indices"] == uniform_step_df["indices"]).all()
            uniform_step_col = uniform_step_df["uniform_steps"]

        else:
            if "uniform_steps" not in df.columns:
                raise ValueError(f"No 'uniform_steps' column in {df.columns=}")
            uniform_step_col = df["uniform_steps"]

        repeats = np.concatenate(
            [np.array(ast.literal_eval(x)) for x in uniform_step_col]
        )
        y_true = np.repeat(y_true, repeats)
        y_pred = np.repeat(y_pred, repeats)
    del df

    assert len(y_true) == len(y_pred)

    if args.filter_labels:
        filter_labels = np.array(list(int(x) for x in args.filter_labels.split(",")))
        mask = np.isin(y_true, filter_labels)
        y_true = y_true[mask]  # type:ignore
        y_pred = y_pred[mask]  # type:ignore

    unique_labels = sorted(set(y_true) | set(y_pred))

    # precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(
    #     y_true, y_pred, average="weighted"
    # )
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    print(f"{accuracy=:.5}")
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    print(f"{balanced_accuracy=:.5}")
    # confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

    if args.per_class:
        precision, recall, fscore, support = (
            sklearn.metrics.precision_recall_fscore_support(
                y_true, y_pred, labels=unique_labels
            )
        )
        print("|Class | Precision | Recall | F-score | Support |")
        print("|------|-----------|--------|---------|---------|")
        for i, (prec, rec, fsc, supp) in enumerate(
            zip(precision, recall, fscore, support)  # type:ignore
        ):
            # print(
            #     f"Class {unique_labels[i]}: Precision={prec:.2f}, Recall={rec:.2f}, F-score={fsc:.2f}, Support={supp}"
            # )
            print(f"| {unique_labels[i]}|{prec:.2f} | {rec:.2f} | {fsc:.2f} | {supp} |")
    if args.output_file is not None:
        with open(args.output_file, "a") as outf:
            if args.key is None:
                outf.write(f"{accuracy:.5},{balanced_accuracy:.5}\n")
            else:
                outf.write(f"{args.key},{accuracy:.5},{balanced_accuracy:.5}\n")
        print(f"Appended to {args.output_file}")


if __name__ == "__main__":
    main()
