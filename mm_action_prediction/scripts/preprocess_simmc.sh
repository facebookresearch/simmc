#!/bin/bash

DOMAIN="furniture"
# DOMAIN="fashion"
ROOT="../data/simmc_${DOMAIN}/"

# Input files.
TRAIN_JSON_FILE="${ROOT}${DOMAIN}_train_dials.json"
DEV_JSON_FILE="${ROOT}${DOMAIN}_dev_dials.json"
DEVTEST_JSON_FILE="${ROOT}${DOMAIN}_devtest_dials.json"


if [ "$DOMAIN" == "furniture" ]; then
    METADATA_FILE="${ROOT}furniture_metadata.csv"
elif [ "$DOMAIN" == "fashion" ]; then
    METADATA_FILE="${ROOT}fashion_metadata.json"
else
    echo "Invalid domain!"
    exit 0
fi


# Output files.
VOCAB_FILE="${ROOT}${DOMAIN}_vocabulary.json"
METADATA_EMBEDS="${ROOT}${DOMAIN}_asset_embeds.npy"
ATTR_VOCAB_FILE="${ROOT}${DOMAIN}_attribute_vocabulary.json"


# Step 1: Extract assistant API.
INPUT_FILES="${TRAIN_JSON_FILE} ${DEV_JSON_FILE} ${DEVTEST_JSON_FILE}"
# If statement.
if [ "$DOMAIN" == "furniture" ]; then
    python tools/extract_actions.py \
        --json_path="${INPUT_FILES}" \
        --save_root="${ROOT}" \
        --metadata_path="${METADATA_FILE}"
elif [ "$DOMAIN" == "fashion" ]; then
    python tools/extract_actions_fashion.py \
        --json_path="${INPUT_FILES}" \
        --save_root="${ROOT}" \
        --metadata_path="${METADATA_FILE}"
else
    echo "Invalid domain!"
    exit 0
fi


# Step 2: Extract vocabulary from train.
python tools/extract_vocabulary.py \
    --train_json_path="${TRAIN_JSON_FILE}" \
    --vocab_save_path="${VOCAB_FILE}" \
    --threshold_count=5


# Step 3: Read and embed shopping assets.
if [ "$DOMAIN" == "furniture" ]; then
    python tools/embed_furniture_assets.py \
        --input_csv_file="${METADATA_FILE}" \
        --embed_path="${METADATA_EMBEDS}"
elif [ "$DOMAIN" == "fashion" ]; then
    python tools/embed_fashion_assets.py \
        --input_asset_file="${METADATA_FILE}" \
        --embed_path="${METADATA_EMBEDS}"
else
    echo "Invalid domain!"
    exit 0
fi


# Step 4: Convert all the splits into npy files for dataloader.
SPLIT_JSON_FILES=("${TRAIN_JSON_FILE}" "${DEV_JSON_FILE}" "${DEVTEST_JSON_FILE}")
for SPLIT_JSON_FILE in "${SPLIT_JSON_FILES[@]}" ; do
    python tools/build_multimodal_inputs.py \
        --json_path="${SPLIT_JSON_FILE}" \
        --vocab_file="${VOCAB_FILE}" \
        --save_path="$ROOT" \
        --action_json_path="${SPLIT_JSON_FILE/.json/_api_calls.json}" \
        --retrieval_candidate_file="${SPLIT_JSON_FILE/.json/_retrieval_candidates.json}" \
        --domain="${DOMAIN}"
done


# Step 5: Extract vocabulary for attributes from train npy file.
python tools/extract_attribute_vocabulary.py \
    --train_npy_path="${TRAIN_JSON_FILE/.json/_mm_inputs.npy}" \
    --vocab_save_path="${ATTR_VOCAB_FILE}" \
    --domain="${DOMAIN}"
