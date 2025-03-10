HELPER_SCRIPTS_DIR=$(dirname $0)/helpers
RN_RUN_ID=$1
KEY_RUN_ID=$2

if [ -z "$KEY_RUN_ID" ]; then
    echo "Error: two positional arguments (RN_RUN_ID and KEY_RUN_ID) required."
    exit 1
fi

set -x
set -e

# Check if the local file exists, and set the metadata basenam accordingly
if [[ -f "${RN_PREDS}/${RN_RUN_ID}/test/metadata_test_local.csv" ]]; then
    METADATA_BASENAME="metadata_test_local.csv"
else
    METADATA_BASENAME="metadata_test.txt"
fi

# collate logits

python "${HELPER_SCRIPTS_DIR}/collate_predictions.py" \
    metadata="${RN_PREDS}/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}/test/${METADATA_BASENAME}" \
    predictions="${RN_PREDS}/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}/test/predictions/" \
    prediction_file_type=both \
    output_folder="${RN_PREDS}/collated_predictions/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}" \
    overwrite=True error_if_exists=False n_specials_to_ignore=0

# get per-salami-slice preds

## RN
python "${HELPER_SCRIPTS_DIR}/get_per_salami_slice_preds.py" \
    column_types.inversion=float \
    metadata="${RN_PREDS}/collated_predictions/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}/${METADATA_BASENAME}" \
    predictions="${RN_PREDS}/collated_predictions/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}/predictions" \
    dictionary_folder="${RN_PREDS}/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}/test" \
    output_folder="${RN_PREDS}/per_salami_slice_predictions/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}_collated/test" \
    concat_features='[[primary_alteration,primary_degree,secondary_alteration,secondary_degree]]' \
    n_specials=0 collated=True

## Key

python "${HELPER_SCRIPTS_DIR}/get_per_salami_slice_preds.py" \
    column_types.inversion=float \
    metadata="${RN_PREDS}/collated_predictions/${KEY_RUN_ID}/${METADATA_BASENAME}" \
    predictions="${RN_PREDS}/collated_predictions/${KEY_RUN_ID}/predictions" \
    dictionary_folder="${RN_PREDS}/${KEY_RUN_ID}/test" \
    output_folder="${RN_PREDS}/per_salami_slice_predictions/${KEY_RUN_ID}_collated/test" \
    concat_features='[[key_pc,mode]]' n_specials=0 collated=True

# calculate metrics
bash "${HELPER_SCRIPTS_DIR}/musicbert_synced_metrics_concat_degree_with_conditioning.sh" \
    "${RN_PREDS}/per_salami_slice_predictions/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}_collated/test" \
    "${RN_PREDS}/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}_metrics.csv" --uniform-steps

# get K+D+Q+I score

python "${HELPER_SCRIPTS_DIR}/calculate_metrics_from_csvs.py" \
    "${RN_PREDS}/per_salami_slice_predictions/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}_collated/test/primary_alteration_primary_degree_secondary_alteration_secondary_degree.csv" \
    "${RN_PREDS}/per_salami_slice_predictions/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}_collated/test/quality.csv" \
    "${RN_PREDS}/per_salami_slice_predictions/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}_collated/test/inversion.csv" \
    "${RN_PREDS}/per_salami_slice_predictions/${KEY_RUN_ID}_collated/test/key_pc_mode.csv"

if [ -f "${RN_PREDS}/per_salami_slice_predictions/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}_collated/test/root_pc.csv" ]; then
    # get K+R+D+Q+I score
    python "${HELPER_SCRIPTS_DIR}/calculate_metrics_from_csvs.py" \
        "${RN_PREDS}/per_salami_slice_predictions/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}_collated/test/primary_alteration_primary_degree_secondary_alteration_secondary_degree.csv" \
        "${RN_PREDS}/per_salami_slice_predictions/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}_collated/test/quality.csv" \
        "${RN_PREDS}/per_salami_slice_predictions/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}_collated/test/inversion.csv" \
        "${RN_PREDS}/per_salami_slice_predictions/${RN_RUN_ID}_cond_on_${KEY_RUN_ID}_collated/test/root_pc.csv" \
        "${RN_PREDS}/per_salami_slice_predictions/${KEY_RUN_ID}_collated/test/key_pc_mode.csv"
fi

set +x
