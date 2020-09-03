#!/bin/bash

GPU_ID=0
DOMAIN="furniture"
# DOMAIN="fashion"
ROOT="../data/simmc_${DOMAIN}/"


# Input files.
TRAIN_JSON_FILE="${ROOT}${DOMAIN}_train_dials.json"
DEV_JSON_FILE="${ROOT}${DOMAIN}_dev_dials.json"
DEVTEST_JSON_FILE="${ROOT}${DOMAIN}_devtest_dials.json"


# Output files.
METADATA_EMBEDS="${ROOT}${DOMAIN}_asset_embeds.npy"
ATTR_VOCAB_FILE="${ROOT}${DOMAIN}_attribute_vocabulary.json"
MODEL_METAINFO="models/${DOMAIN}_model_metainfo.json"


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


# Train history-agnostic model.
# For other models, please look at scripts/train_all_simmc_models.sh
python -u train_simmc_agent.py $COMMON_FLAGS \
    --encoder="history_agnostic" \
    --text_encoder="lstm"


# Evaluate a trained model checkpoint.
CHECKPOINT_PATH="checkpoints/hae/epoch_20.tar"
python -u eval_simmc_agent.py \
    --eval_data_path=${DEVTEST_JSON_FILE/.json/_mm_inputs.npy} \
    --checkpoint="$CHECKPOINT_PATH" --gpu_id=0 --batch_size=50 \
    --domain="$DOMAIN"
