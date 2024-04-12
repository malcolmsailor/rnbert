HELPER_SCRIPTS_DIR=$(dirname $0)
SLURM_ID=$1

# collate logits

python ${HELPER_SCRIPTS_DIR}/collate_predictions.py \
    metadata=${SAVED_PREDICTIONS_DIR}/${SLURM_ID}/test/metadata_test.txt \
    predictions=${SAVED_PREDICTIONS_DIR}/${SLURM_ID}/test/predictions/ \
    prediction_file_type=both output_folder=${SAVED_PREDICTIONS_DIR}/collated_predictions/${SLURM_ID} \
    overwrite=True error_if_exists=False n_specials_to_ignore=0

# get per-salami-slice preds

python ${HELPER_SCRIPTS_DIR}/get_per_salami_slice_preds.py \
    column_types.inversion=float \
    metadata=${SAVED_PREDICTIONS_DIR}/collated_predictions/${SLURM_ID}/metadata_test.txt \
    predictions=${SAVED_PREDICTIONS_DIR}/collated_predictions/${SLURM_ID}/predictions \
    dictionary_folder=${SAVED_PREDICTIONS_DIR}/${SLURM_ID}/test \
    output_folder=${SAVED_PREDICTIONS_DIR}/per_salami_slice_predictions/${SLURM_ID}_collated/test \
    concat_features='[[primary_alteration,primary_degree,secondary_alteration,secondary_degree],[key_pc,mode]]' \
    n_specials=0 collated=True

# calculate metrics

bash ${HELPER_SCRIPTS_DIR}/musicbert_synced_metrics_concat_degree.sh \
    ${SAVED_PREDICTIONS_DIR}/per_salami_slice_predictions/${SLURM_ID}_collated/test \
    ${SAVED_PREDICTIONS_DIR}/${SLURM_ID}_synced_metrics.csv --uniform-steps
