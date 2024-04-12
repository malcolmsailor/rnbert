set -e
set -x

HELPER_SCRIPTS_DIR=$(dirname $0)
SLURM_ID=$1
VITERBI_ALPHA={$2-7.0}

# TODO: (Malcolm 2024-04-12) clean up this file

# collate logits

python ${HELPER_SCRIPTS_DIR}/collate_predictions.py \
    metadata=${SAVED_PREDICTIONS_DIR}/${SLURM_ID}/test/metadata_test.txt \
    predictions=${SAVED_PREDICTIONS_DIR}/${SLURM_ID}/test/predictions/ \
    prediction_file_type=both output_folder=${SAVED_PREDICTIONS_DIR}/collated_predictions${SLURM_ID} \
    overwrite=True error_if_exists=False n_specials_to_ignore=4

# # get per-salami-slice preds

# python ${HELPER_SCRIPTS_DIR}/get_per_salami_slice_preds.py \
#     column_types.inversion=float \
#     metadata=${SAVED_PREDICTIONS_DIR}/collated_predictions${SLURM_ID}/metadata_test.txt \
#     predictions=${SAVED_PREDICTIONS_DIR}/collated_predictions${SLURM_ID}/predictions \
#     dictionary_folder=${SAVED_PREDICTIONS_DIR}/${SLURM_ID}/test \
#     output_folder=${SAVED_PREDICTIONS_DIR}/per_salami_slice_predictions/${SLURM_ID}_collated/test \
#     concat_features='[[key_pc,mode]]' n_specials=0 collated=True

# Get per-salami-slice logits
python ${HELPER_SCRIPTS_DIR}/get_per_salami_slice_logits.py \
    column_types.inversion=float \
    metadata=${SAVED_PREDICTIONS_DIR}/collated_predictions${SLURM_ID}/metadata_test.txt \
    predictions=${SAVED_PREDICTIONS_DIR}/collated_predictions${SLURM_ID}/predictions \
    dictionary_folder=${SAVED_PREDICTIONS_DIR}/${SLURM_ID}/test \
    output_folder=${SAVED_PREDICTIONS_DIR}/per_salami_slice_logits/${SLURM_ID}_collated/test \
    concat_features='[[key_pc,mode]]' n_specials=0 collated=True

# calculate metrics on collated per-salami-slice predictions

# python ${HELPER_SCRIPTS_DIR}/calculate_metrics_from_csvs.py \
#     ${SAVED_PREDICTIONS_DIR}/per_salami_slice_predictions/${SLURM_ID}_collated/test/key_pc_mode.csv --uniform-steps

# Run Viterbi decoding
python ${HELPER_SCRIPTS_DIR}/predict_keys_with_viterbi.py \
    logits_h5=${SAVED_PREDICTIONS_DIR}/per_salami_slice_logits/${SLURM_ID}_collated/test/key_pc_mode.h5 \
    output_h5=${SAVED_PREDICTIONS_DIR}/viterbi_predictions/${SLURM_ID}_collated/test/key_pc_mode-alpha=${alpha}.h5 \
    sticky_viterbi_alpha=${VITERBI_ALPHA}

# Get predictions
python ${HELPER_SCRIPTS_DIR}/get_csv_of_predictions_and_labels.py \
    metadata=${SAVED_PREDICTIONS_DIR}/per_salami_slice_logits/${SLURM_ID}_collated/test/metadata_test.txt \
    predictions_h5=${SAVED_PREDICTIONS_DIR}/viterbi_predictions/${SLURM_ID}_collated/test/key_pc_mode-alpha=${alpha}.h5 \
    output_path=${SAVED_PREDICTIONS_DIR}/viterbi_predictions/${SLURM_ID}_collated/test/key_pc_mode-alpha=${alpha}.csv \
    feature_name=key_pc_mode \
    concat_feature='[key_pc,mode]' \
    dictionary_path=~/output/musicbert/saved_predictions/${SLURM_ID}/test/key_pc_mode_dictionary.txt

# Calculate metrics
python ${HELPER_SCRIPTS_DIR}/calculate_metrics_from_csvs.py \
    ${SAVED_PREDICTIONS_DIR}/viterbi_predictions/${SLURM_ID}_collated/test/key_pc_mode-alpha=${alpha}.csv \
    --output-file ${SAVED_PREDICTIONS_DIR}/${SLURM_ID}_synced_metrics.csv

# sync logits

# python scripts/sync_logits.py \
#     metadata=${SAVED_PREDICTIONS_DIR}/collated_predictions${SLURM_ID}/metadata_test.txt \
#     input_folder=${SAVED_PREDICTIONS_DIR}/collated_predictions${SLURM_ID}/predictions \
#     output_folder=${SAVED_PREDICTIONS_DIR}/synced_predictions/${SLURM_ID} \
#     features_to_sync=[key_pc_mode]

# See histograms of sequence lengths TODO

# See top-K accuracy

# # Then calculate top-K accuracy (also makes a confusion matrix)

# python ${HELPER_SCRIPTS_DIR}/calculate_top_k.py \
#     metadata=${SAVED_PREDICTIONS_DIR}/per_salami_slice_logits/${SLURM_ID}_collated/test/metadata_test.txt \
#     logits_folder=${SAVED_PREDICTIONS_DIR}/per_salami_slice_logits/${SLURM_ID}_collated/test \
#     dictionary_folder=${SAVED_PREDICTIONS_DIR}/${SLURM_ID}/test \
#     concat_features='[[key_pc,mode]]'

# # Get per-score metrics

# python scripts/calculate_per_score_metrics.py \
#     input_csv=${SAVED_PREDICTIONS_DIR}/per_salami_slice_predictions/${SLURM_ID}_collated/test/key_pc_mode.csv \
#     output_csv=${SAVED_PREDICTIONS_DIR}/per_salami_slice_predictions/${SLURM_ID}_collated/test/key_pc_mode_per_score_metrics.csv

# # Per-score scatterplot

# python data_analysis/visualize_per_score_metric.py \
#     input_csv=/Users/malcolm/output/musicbert/per_salami_slice_predictions/${SLURM_ID}_collated/test/key_pc_mode_per_score_metrics.csv \
#     metric=micro_fscore

# Do viterbi experiment

# bash shell_scripts/viterbi_keys.sh ${SLURM_ID}

# Display score

set +x
