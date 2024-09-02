#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,4
WORKSPACE_DIR=/data/fcl/fcl/workspace/2024_35/240528_35_xiaorongshiyan/llama3/numerical_enhancement/multi/1err
MODEL_DIR=/data/fcl/fcl/workspace/model/llama3_8b_hf/Meta-Llama-3-8B
BATCH_SIZE=16
NUM_EPOCHS=60

CURRENT_DATETIME=$(date +"%Y%m%d_%H%M")
# CURRENT_DATE=$(date +"%Y%m%d")
OUTPUT_DIR=${WORKSPACE_DIR}/model${CURRENT_DATETIME}/lora
CACHE_DIR=${WORKSPACE_DIR}/cache
LOG_FILE=${WORKSPACE_DIR}/${CURRENT_DATETIME}.log
NUM_LOG_FILE=${WORKSPACE_DIR}/${CURRENT_DATETIME}_numerical.log
TRAIN_FILE_DIR=${WORKSPACE_DIR}/dataset

nohup torchrun --nproc_per_node 4 ${WORKSPACE_DIR}/supervised_finetuning.py \
    --model_type llama \
    --model_name_or_path ${MODEL_DIR} \
    --train_file_dir ${TRAIN_FILE_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --numerical_log_file_path ${NUM_LOG_FILE} \
    --number_of_numbers 3 \
    --do_train \
    --do_eval \
    --template_name vicuna \
    --use_peft True \
    --max_train_samples -1 \
    --max_eval_samples -1 \
    --model_max_length 8192 \
    --num_train_epochs ${NUM_EPOCHS} \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 20 \
    --eval_steps 40 \
    --evaluation_strategy steps \
    --save_steps 40 \
    --save_strategy steps \
    --save_total_limit 1000 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 256 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ${CACHE_DIR} > ${LOG_FILE} 2>&1 &
