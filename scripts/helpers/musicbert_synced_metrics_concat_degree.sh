#!/bin/bash

# The way this script interfaces with the calculate_metrics_from_csvs.py script
#   to produce a csv file is a big mess

set -e
# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

input_folder="$1"
output_csv="$2"
shift
shift

xx=(
    "key"
    "degree"
    "quality"
    "inversion"
    "dqi"
    "kdqi"
)
names=(
    "key_pc mode"
    "primary_alteration_primary_degree_secondary_alteration_secondary_degree"
    "quality"
    "inversion"
    "primary_alteration_primary_degree_secondary_alteration_secondary_degree quality inversion"
    "key_pc mode primary_alteration_primary_degree_secondary_alteration_secondary_degree quality inversion"
)

if [ ${#xx[@]} -ne ${#names[@]} ]; then
    echo "Arrays are not of the same length."
    exit 1
fi

echo ,accuracy,balanced_accuracy >"${output_csv}"

for ((i = 0; i < ${#xx[@]}; i++)); do
    x="${xx[i]}"
    array="${names[i]}"
    # Convert string to an array
    IFS=' ' read -r -a inner_array <<<"$array"
    new_array=()
    for element in "${inner_array[@]}"; do
        this_csv="${input_folder}/${element}.csv"
        new_array+=("${this_csv}")
        if [ ${#inner_array[@]} -gt 1 ]; then
            echo "${element}"
            python "${SCRIPT_DIR}"/calculate_metrics_from_csvs.py \
                "${this_csv}" --key "${element}" --output-file "${output_csv}" "${@}"
            echo
        fi
    done
    echo "${x^^}"
    python "${SCRIPT_DIR}"/calculate_metrics_from_csvs.py \
        "${new_array[@]}" --key "${x^^}" --output-file "${output_csv}" "${@}"
    echo

done

cat "${output_csv}"
