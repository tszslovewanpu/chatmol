#!/bin/bash

# Set CUDA devices for each process
export CUDA_VISIBLE_DEVICES="0,2,3,4"
BASE_MODEL_DIR="/data/fcl/fcl/workspace/2024_35/240528_35_xiaorongshiyan/llama3/numerical_enhancement/multi/1err/model20240805_1031/merged_final"
TEMPERATURE_VALUE=1.0
BATCH_SIZE=36
PENALTY=1.0
MAX_NEW=512
NUM_NUMBERS=3

# Define workspace directory as the script's own directory
WORKSPACE_DIR="$(dirname "$0")"
CURRENT_DATETIME=$(date +"%Y%m%d_%H%M")
DATA_FILE="${WORKSPACE_DIR}/inference_prompt.txt"
OUTPUT_FILE="${WORKSPACE_DIR}/result/prediction_result.jsonl"
LOG_DIR="${WORKSPACE_DIR}/log"

# Create necessary directories
mkdir -p "${WORKSPACE_DIR}/split_data"
mkdir -p "${LOG_DIR}"
mkdir -p "${WORKSPACE_DIR}/result"

# Get available GPU IDs and calculate split size
GPU_IDS=($(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '))
NUM_GPUS=${#GPU_IDS[@]}
SPLIT_SIZE="l/$NUM_GPUS"

# Split the data file into parts based on the number of available GPUs
echo "Splitting data file into $NUM_GPUS parts..."
split -n "$SPLIT_SIZE" "${DATA_FILE}" "${WORKSPACE_DIR}/split_data/inference_prompt_part_"

########################
# Get available GPU IDs
AVAILABLE_GPUS=($(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '))
# Find all split files
SPLIT_FILES=(${WORKSPACE_DIR}/split_data/inference_prompt_part_*)
# Create an array to map GPU IDs to split files
declare -A GPU_TO_FILE
# Assign each GPU a split file
for i in "${!AVAILABLE_GPUS[@]}"; do
    GPU_ID=${AVAILABLE_GPUS[$i]}
    SPLIT_FILE=${SPLIT_FILES[$i]}
    GPU_TO_FILE[$GPU_ID]=$SPLIT_FILE
done
#######################

# Run inference on each GPU separately
echo "Starting inference on each GPU..."
for GPU_ID in "${AVAILABLE_GPUS[@]}"; do
    DATA_FILE=${GPU_TO_FILE[$GPU_ID]}
    CUDA_VISIBLE_DEVICES="$GPU_ID" nohup python ${WORKSPACE_DIR}/inference.py \
        --model_type llama \
        --template_name vicuna \
        --base_model ${BASE_MODEL_DIR} \
        --tokenizer_path ${BASE_MODEL_DIR} \
        --number_of_numbers ${NUM_NUMBERS} \
        --data_file ${DATA_FILE} \
        --eval_batch_size ${BATCH_SIZE} \
        --max_new_tokens ${MAX_NEW} \
        --repetition_penalty ${PENALTY} \
        --temperature ${TEMPERATURE_VALUE} \
        --output_file ${OUTPUT_FILE}_${GPU_ID}.jsonl > ${LOG_DIR}/${CURRENT_DATETIME}_${GPU_ID}.log 2>&1 &
done

# Wait for all background jobs to complete
wait

# Merge the output files into one
echo "Merging output files..."
cat ${OUTPUT_FILE}_*.jsonl > ${OUTPUT_FILE}

# Clean up individual output files
echo "Cleaning up individual output files..."
rm ${OUTPUT_FILE}_*.jsonl

echo "Inference complete. Results saved to ${OUTPUT_FILE}."