#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Train (furniture, text-only)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/furniture_to \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture_to/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/furniture_to/furniture_train_dials_target.txt \
    --do_eval \
    --eval_data_file="${PATH_DIR}"/gpt2_dst/data/furniture_to/furniture_dev_dials_target.txt \
    --num_train_epochs=1 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=4

# Train (furniture, multi-modal)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/furniture \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_train_dials_target.txt \
    --do_eval \
    --eval_data_file="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_dev_dials_target.txt \
    --num_train_epochs=1 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=4

# Train (Fashion, text-only)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/fashion_to \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion_to/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/fashion_to/fashion_train_dials_target.txt \
    --do_eval \
    --eval_data_file="${PATH_DIR}"/gpt2_dst/data/fashion_to/fashion_dev_dials_target.txt \
    --num_train_epochs=1 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=4

# Train (Fashion, multi-modal)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/fashion \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_train_dials_target.txt \
    --do_eval \
    --eval_data_file="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_dev_dials_target.txt \
    --num_train_epochs=1 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=4
