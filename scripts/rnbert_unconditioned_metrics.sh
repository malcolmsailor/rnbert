HELPER_SCRIPTS_DIR=$(dirname $0)/helpers
SLURM_ID=$1

set -e
set -x

# Check if the local file exists, and set the metadata basenam accordingly
if [[ -f "${RN_PREDS}/${RN_RUN_ID}/test/metadata_test_local.csv" ]]; then
    METADATA_BASENAME="metadata_test_local.csv"
else
    METADATA_BASENAME="metadata_test.txt"
fi

# collate logits

python ${HELPER_SCRIPTS_DIR}/collate_predictions.py \
    metadata=${RN_PREDS}/${SLURM_ID}/test/${METADATA_BASENAME} \
    predictions=${RN_PREDS}/${SLURM_ID}/test/predictions/ \
    prediction_file_type=both output_folder=${RN_PREDS}/collated_predictions/${SLURM_ID} \
    overwrite=True error_if_exists=False n_specials_to_ignore=0

# get per-salami-slice preds

python ${HELPER_SCRIPTS_DIR}/get_per_salami_slice_preds.py \
    column_types.inversion=float \
    metadata=${RN_PREDS}/collated_predictions/${SLURM_ID}/${METADATA_BASENAME} \
    predictions=${RN_PREDS}/collated_predictions/${SLURM_ID}/predictions \
    dictionary_folder=${RN_PREDS}/${SLURM_ID}/test \
    output_folder=${RN_PREDS}/per_salami_slice_predictions/${SLURM_ID}_collated/test \
    concat_features='[[primary_alteration,primary_degree,secondary_alteration,secondary_degree],[key_pc,mode]]' \
    n_specials=0 collated=True

# calculate metrics

bash ${HELPER_SCRIPTS_DIR}/musicbert_synced_metrics_concat_degree.sh \
    ${RN_PREDS}/per_salami_slice_predictions/${SLURM_ID}_collated/test \
    ${RN_PREDS}/${SLURM_ID}_synced_metrics.csv --uniform-steps

set +x
