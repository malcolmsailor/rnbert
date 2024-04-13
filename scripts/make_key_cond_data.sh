set -e
set -x

HELPER_SCRIPTS_DIR=$(dirname $0)/helpers
MUSICBERT_FORK_DIR=$(dirname $0)/../musicbert_fork
KEY_RUN_ID=$1
VITERBI_ALPHA=${2-7.0}

if [ -z "$KEY_RUN_ID" ]; then
    echo "Error: one positional argument KEY_RUN_ID required."
    exit 1
fi

RNDATA_ROOT="${RNDATA_ROOT-${HOME}/datasets}"

FAIRSEQ_RN_COND_TEST_BIN="${FAIRSEQ_RN_COND_TEST_BIN-${RNDATA_ROOT}/rnbert_rn_cond_test_data_bin}"

# get key predictions

python "${HELPER_SCRIPTS_DIR}"/predictions_to_raw_fairseq_data.py \
    input_csv="${RN_PREDS}/viterbi_predictions/${KEY_RUN_ID}_collated/test/key_pc_mode-alpha=${alpha}.csv" \
    dictionary_path="${RN_PREDS}/${KEY_RUN_ID}/test/key_pc_mode_dictionary.txt" \
    uncollated_metadata="${RN_PREDS}/${KEY_RUN_ID}/test/metadata_test.txt" \
    output_path="${RNDATA_ROOT}/predicted/${KEY_RUN_ID}_predicted_key_pc_mode_test.txt"

# binarize key predictions with fairseq

fairseq-preprocess --only-source \
    --testpref "${RNDATA_ROOT}/predicted/${KEY_RUN_ID}_predicted_key_pc_mode_test.txt" \
    --srcdict "${RNDATA_ROOT}/rnbert_abstract_data_bin/key_pc_mode/dict.txt" \
    --destdir "${RNDATA_ROOT}/predicted/${KEY_RUN_ID}_predicted_key_pc_mode_bin" \
    --workers 16

# instantiate dataset
python "${MUSICBERT_FORK_DIR}/misc_scripts/instantiate_abstract_dataset.py" \
    input_folder="${RNDATA_ROOT}/rnbert_abstract_data_bin" \
    feature_names='[chord_factors,chord_tone,harmony_onset,bass_pcs,primary_alteration_primary_degree_secondary_alteration_secondary_degree,inversion,quality]' \
    external_conditioning="${RNDATA_ROOT}/predicted/${KEY_RUN_ID}_predicted_key_pc_mode_bin" \
    output_folder="${FAIRSEQ_RN_COND_TEST_BIN}"

set +x
