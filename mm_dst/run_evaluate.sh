#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
    PATH_DATA_DIR=$(realpath ../data)
else
    PATH_DIR=$(realpath "$1")
    PATH_DATA_DIR=$(realpath "$2")
fi

# Evaluate (Example)
python -m utils.evaluate_dst \
    --input_path_target="${PATH_DATA_DIR}"/simmc_fashion/fashion_devtest_dials.json \
    --input_path_predicted="${PATH_DIR}"/fashion_devtest_pred_dials.json \
    --output_path_report="${PATH_DIR}"/fashion_report.json

python -m utils.evaluate_dst \
    --input_path_target="${PATH_DATA_DIR}"/simmc_furniture/furniture_devtest_dials.json \
    --input_path_predicted="${PATH_DIR}"/furniture_devtest_pred_dials.json \
    --output_path_report="${PATH_DIR}"/furniture_report.json