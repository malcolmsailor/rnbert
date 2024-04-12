#!/bin/bash

RNBERT_DIR=$(dirname $0)/..
RNDATA_ROOT="${RNDATA_ROOT-${HOME}/datasets}"

FAIRSEQ_ABSTRACT_RAW="${FAIRSEQ_ABSTRACT_RAW-${RNDATA_ROOT}/rnbert_abstract_data_raw}"
FAIRSEQ_ABSTRACT_BIN="${FAIRSEQ_ABSTRACT_RAW%_bin})_raw"
FAIRSEQ_KEY_BIN="${FAIRSEQ_KEY_BIN-${RNDATA_ROOT}/rnbert_key_data_bin}"
FAIRSEQ_RN_COND_BIN="${FAIRSEQ_KEY_BIN-${RNDATA_ROOT}/rnbert_rn_cond_data_bin}"
FAIRSEQ_RN_UNCOND_BIN="${FAIRSEQ_KEY_BIN-${RNDATA_ROOT}/rnbert_rn_uncond_data_bin}"

set -e
set -x

python ${RNBERT_DIR}/musicbert_fork/binarize_scripts/binarize_abstract_folder.py \
    input_folder=${FAIRSEQ_ABSTRACT_RAW} workers=16

python ${RNBERT_DIR}/musicbert_fork/misc_scripts/instantiate_abstract_dataset.py \
    input_folder=${FAIRSEQ_ABSTRACT_BIN} \
    feature_names='[key_pc_mode]' \
    output_folder=${FAIRSEQ_KEY_BIN}

python ${RNBERT_DIR}/musicbert_fork/misc_scripts/instantiate_abstract_dataset.py \
    input_folder=${FAIRSEQ_ABSTRACT_BIN} \
    feature_names='[chord_factors,chord_tone,harmony_onset,bass_pcs,primary_alteration_primary_degree_secondary_alteration_secondary_degree,inversion,quality]' \
    output_folder=${FAIRSEQ_RN_COND_BIN} \
    conditioning=key_pc_mode

python ${RNBERT_DIR}/musicbert_fork/misc_scripts/instantiate_abstract_dataset.py \
    input_folder=${FAIRSEQ_ABSTRACT_BIN} \
    feature_names='[chord_factors,chord_tone,harmony_onset,bass_pcs,primary_alteration_primary_degree_secondary_alteration_secondary_degree,inversion,quality,key_pc,mode]' \
    output_folder=${FAIRSEQ_RN_UNCOND_BIN} \
    conditioning=key_pc_mode

set +x
