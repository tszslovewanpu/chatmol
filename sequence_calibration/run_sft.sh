#!/bin/bash
### llama3 128138
export CUDA_VISIBLE_DEVICES=0,1,2,7
WORKSPACE_DIR=/data/fcl/fcl/workspace/2024_35/240528_35_xiaorongshiyan/llama3/multi_properties_llama3/multi_properties_brio_training/1err/brio_training_3
MODEL_DIR=/data/fcl/fcl/workspace/2024_35/240528_35_xiaorongshiyan/llama3/multi_properties_llama3/multi_properties_sft_training/1err/sft_training/model20240624/merged_760
BATCH_SIZE=1
NUM_EPOCHS=31

CURRENT_DATETIME=$(date +"%Y%m%d_%H%M")
CURRENT_DATE=$(date +"%Y%m%d")
OUTPUT_DIR=${WORKSPACE_DIR}/model${CURRENT_DATE}/lora
CACHE_DIR=${WORKSPACE_DIR}/cache
LOG_FILE=${WORKSPACE_DIR}/${CURRENT_DATETIME}.log
TRAIN_FILE_DIR=${WORKSPACE_DIR}/dataset
BRIO_LOGGER_PATH=${WORKSPACE_DIR}/${CURRENT_DATETIME}_BRIO.log

nohup torchrun --nproc_per_node 4 ${WORKSPACE_DIR}/supervised_finetuning_240329_brio_240719_one_forward.py \
    --model_type llama \
    --model_name_or_path ${MODEL_DIR} \
    --train_file_dir ${TRAIN_FILE_DIR} \
    --brio_log_file_path ${BRIO_LOGGER_PATH} \
    --brio_candidate_labels_pad_id 128138 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
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
    --eval_steps 50 \
    --evaluation_strategy steps \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 10000 \
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



