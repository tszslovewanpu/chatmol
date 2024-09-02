#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
WORKSPACE_DIR=/data/fcl/fcl/workspace/2024_35/240528_35_xiaorongshiyan/llama3/multi_properties_llama3/multi_properties_brio_training/1err/brio_training_3/inference/inference_9700/inference_2_10
BASE_MODEL_DIR=/data/fcl/fcl/workspace/2024_35/240528_35_xiaorongshiyan/llama3/multi_properties_llama3/multi_properties_brio_training/1err/brio_training_3/model20240718/merged_9700

CURRENT_DATETIME=$(date +"%Y%m%d_%H%M")
DATA_FILE=${WORKSPACE_DIR}/inference_prompt.txt
OUTPUT_FILE=${WORKSPACE_DIR}/prediction_result.jsonl
LOG_FILE=${WORKSPACE_DIR}/${CURRENT_DATETIME}.log

nohup python ${WORKSPACE_DIR}/inference.py \
    --model_type llama \
    --template_name vicuna \
    --base_model ${BASE_MODEL_DIR} \
    --tokenizer_path ${BASE_MODEL_DIR} \
    --data_file ${DATA_FILE} \
    --eval_batch_size 76 \
    --max_new_tokens 512 \
    --repetition_penalty 1.0 \
    --temperature 1.0 \
    --output_file ${OUTPUT_FILE} > ${LOG_FILE} 2>&1 &