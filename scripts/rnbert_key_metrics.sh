set -e
set -x

HELPER_SCRIPTS_DIR=$(dirname $0)/helpers
SLURM_ID=$1
VITERBI_ALPHA=${2-7.0}

collate logits

python ${HELPER_SCRIPTS_DIR}/collate_predictions.py \
    metadata=${RN_PREDS}/${SLURM_ID}/test/metadata_test.txt \
    predictions=${RN_PREDS}/${SLURM_ID}/test/predictions/ \
    prediction_file_type=both \
    output_folder=${RN_PREDS}/collated_predictions/${SLURM_ID} \
    overwrite=True error_if_exists=False n_specials_to_ignore=4

# Get per-salami-slice logits
python ${HELPER_SCRIPTS_DIR}/get_per_salami_slice_logits.py \
    column_types.inversion=float \
    metadata=${RN_PREDS}/collated_predictions/${SLURM_ID}/metadata_test.txt \
    predictions=${RN_PREDS}/collated_predictions/${SLURM_ID}/predictions \
    dictionary_folder=${RN_PREDS}/${SLURM_ID}/test \
    output_folder=${RN_PREDS}/per_salami_slice_logits/${SLURM_ID}_collated/test \
    concat_features='[[key_pc,mode]]' n_specials=0 collated=True


# Run Viterbi decoding
python ${HELPER_SCRIPTS_DIR}/predict_keys_with_viterbi.py \
    logits_h5=${RN_PREDS}/per_salami_slice_logits/${SLURM_ID}_collated/test/key_pc_mode.h5 \
    output_h5=${RN_PREDS}/viterbi_predictions/${SLURM_ID}_collated/test/key_pc_mode-alpha=${alpha}.h5 \
    sticky_viterbi_alpha=${VITERBI_ALPHA}

# Get predictions
python ${HELPER_SCRIPTS_DIR}/get_csv_of_predictions_and_labels.py \
    metadata=${RN_PREDS}/per_salami_slice_logits/${SLURM_ID}_collated/test/metadata_test.txt \
    predictions_h5=${RN_PREDS}/viterbi_predictions/${SLURM_ID}_collated/test/key_pc_mode-alpha=${alpha}.h5 \
    output_path=${RN_PREDS}/viterbi_predictions/${SLURM_ID}_collated/test/key_pc_mode-alpha=${alpha}.csv \
    feature_name=key_pc_mode \
    concat_feature='[key_pc,mode]' \
    dictionary_path=${RN_PREDS}/${SLURM_ID}/test/key_pc_mode_dictionary.txt

# Calculate metrics
python ${HELPER_SCRIPTS_DIR}/calculate_metrics_from_csvs.py \
    ${RN_PREDS}/viterbi_predictions/${SLURM_ID}_collated/test/key_pc_mode-alpha=${alpha}.csv \
    --output-file ${RN_PREDS}/${SLURM_ID}_synced_metrics.csv

set +x
