#!/bin/bash

RNBERT_DIR=$(dirname $0)/..
ZIP_FILE="${RNBERT_DIR}/dataset.zip"
EXTRACT_DIR=$(mktemp -d)

RNDATA_ROOT="${RNDATA_ROOT-${HOME}/datasets}"

SEQS_DIR="${SEQS_DIR-${RNDATA_ROOT}/rnbert_seqs}"
FAIRSEQ_ABSTRACT_RAW="${FAIRSEQ_ABSTRACT_RAW-${RNDATA_ROOT}/rnbert_abstract_data_raw}"

# Function to clean up the extracted contents
cleanup() {
    echo "Cleaning up zip contents..."
    rm -rf "$EXTRACT_DIR"
}

# Ensure cleanup runs on script exit, error, or interrupt
trap cleanup EXIT

# Silently unzip the file
echo Unzipping "$ZIP_FILE"
unzip -qq "$ZIP_FILE" -d "$EXTRACT_DIR"

set -e
set -x
python -m write_seqs \
    --src-data-dir ${EXTRACT_DIR} \
    --data-settings ${RNBERT_DIR}/write_seqs/configs/oct_data_abstract.yaml \
    --output-dir ${SEQS_DIR} \
    --input-paths-dir ${RNBERT_DIR}/data_splits \
    --num-workers 8 \
    --overwrite

python ~/code/write_seqs/scripts/to_fair_seq_abstract.py \
    --input-dir ${SEQS_DIR} \
    --output-dir ${FAIRSEQ_ABSTRACT_RAW}
set +x
