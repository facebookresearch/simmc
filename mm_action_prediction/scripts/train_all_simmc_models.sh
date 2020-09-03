#!/bin/bash
GPU_ID=0
# DOMAIN="furniture"
DOMAIN="fashion"
ROOT="../data/simmc_${DOMAIN}/"


# Input files.
TRAIN_JSON_FILE="${ROOT}${DOMAIN}_train_dials.json"
DEV_JSON_FILE="${ROOT}${DOMAIN}_dev_dials.json"
# Output files.
METADATA_EMBEDS="${ROOT}${DOMAIN}_asset_embeds.npy"
ATTR_VOCAB_FILE="${ROOT}${DOMAIN}_attribute_vocabulary.json"
MODEL_METAINFO="models/${DOMAIN}_model_metainfo.json"
CHECKPOINT_PATH="checkpoints"
LOG_PATH="logs/"


COMMON_FLAGS="
    --train_data_path=${TRAIN_JSON_FILE/.json/_mm_inputs.npy} \
    --eval_data_path=${DEV_JSON_FILE/.json/_mm_inputs.npy} \
    --asset_embed_path=${METADATA_EMBEDS} \
    --metainfo_path=${MODEL_METAINFO} \
    --attr_vocab_path=${ATTR_VOCAB_FILE} \
    --learning_rate=0.0001 --gpu_id=$GPU_ID --use_action_attention \
    --num_epochs=100 --eval_every_epoch=5 --batch_size=20 \
    --save_every_epoch=5 --word_embed_size=256 --num_layers=2 \
    --hidden_size=512 \
    --use_multimodal_state --use_action_output --use_bahdanau_attention \
    --skip_bleu_evaluation --domain=${DOMAIN}"


# History-agnostic model.
function history_agnostic () {
    python -u train_simmc_agent.py $COMMON_FLAGS \
        --encoder="history_agnostic" --text_encoder="lstm" \
        --snapshot_path="${CHECKPOINT_PATH}/$1/hae/" &> "${LOG_PATH}/$1/hae.log" &
}
# Hierarchical recurrent encoder model.
function hierarchical_recurrent () {
    python -u train_simmc_agent.py $COMMON_FLAGS \
        --encoder="hierarchical_recurrent" --text_encoder="lstm" \
        --snapshot_path="${CHECKPOINT_PATH}/$1/hre/" &> "${LOG_PATH}/$1/hre.log" &
}
# Memory encoder model.
function memory_network () {
    python -u train_simmc_agent.py $COMMON_FLAGS \
        --encoder="memory_network" --text_encoder="lstm" \
        --snapshot_path="${CHECKPOINT_PATH}/$1/mn/" &> "${LOG_PATH}/$1/mn.log" &
}
# TF-IDF model.
function tf_idf () {
    python -u train_simmc_agent.py $COMMON_FLAGS \
        --encoder="tf_idf" --text_encoder="lstm" \
        --snapshot_path="${CHECKPOINT_PATH}/$1/tf_idf/" &> "${LOG_PATH}/$1/tf_idf.log" &
}
# Transformer model.
function transformer () {
    python -u train_simmc_agent.py $COMMON_FLAGS \
        --encoder="history_agnostic" \
        --text_encoder="transformer" \
        --num_heads_transformer=4 --num_layers_transformer=4 \
        --hidden_size_transformer=2048 --hidden_size=256\
        --snapshot_path="${CHECKPOINT_PATH}/$1/transf/" &> "${LOG_PATH}/$1/transf.log" &
}


# Train all models on a domain Save checkpoints and logs with unique label.
UNIQ_LABEL="${DOMAIN}_dstc_split"
CUR_TIME=$(date +"_%m_%d_%Y_%H_%M_%S")
UNIQ_LABEL+=$CUR_TIME
mkdir "${LOG_PATH}${UNIQ_LABEL}"

history_agnostic "$UNIQ_LABEL"
hierarchical_recurrent "$UNIQ_LABEL"
memory_network "$UNIQ_LABEL"
tf_idf "$UNIQ_LABEL"
transformer "$UNIQ_LABEL"
