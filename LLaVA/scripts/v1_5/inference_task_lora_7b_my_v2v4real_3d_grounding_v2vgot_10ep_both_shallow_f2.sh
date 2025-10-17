#!/bin/bash

#cd /ocean/projects/cis230055p/hchiu1/V2V-GoT/LLaVA
conda activate llava
nvidia-smi
#export PYTHONPATH=/ocean/projects/cis230055p/hchiu1/V2V-GoT/LLaVA:$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
echo $PYTHONPATH

MODEL="v2vgot_10ep_both_shallow_f2"
CKPT_NUMBER=$1
DATA=$2 # 'nq1sm3w0d', or others
DATA_SOURCE=$3 # 'graph' or 'gt'
GRAPH=$4 # 'full'

NUM_LATENCY_FRAMES=0
echo $NUM_LATENCY_FRAMES
POSITIONAL_ERROR_10_STD=0
echo $POSITIONAL_ERROR_10_STD


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
SPLIT='val'

for CKPT_ID in $CKPT_NUMBER
do

    EXP="v2v4real_3d_grounding_${MODEL}_${CKPT_ID}_${GRAPH}_${DATA}"
    echo $EXP
    MODEL_PATH="./checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_${MODEL}/checkpoint-${CKPT_ID}"
    echo $MODEL_PATH

    if [ "$DATA_SOURCE" == "graph" ]; then
      QUESTION_FILE="./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/${DATA}.json"
    else
      QUESTION_FILE="../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_${DATA}.json"
    fi
    echo $QUESTION_FILE

    for IDX in $(seq 0 $((CHUNKS-1))); do
        ANSWER_FILE="./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl"
        echo $ANSWER_FILE

        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path $MODEL_PATH \
            --mm_scene_projector_input_size 3072 \
            --scene_level_only False \
            --object_level_only False \
            --scene_feature_mode shallow \
            --object_feature_mode shallow \
            --num_input_frames 2 \
            --ego_only False \
            --feature_source no_fusion_keep_all \
            --num_latency_frames $NUM_LATENCY_FRAMES \
	    --positional_error_10_std $POSITIONAL_ERROR_10_STD \
            --model-base liuhaotian/llava-v1.5-7b \
            --question-file $QUESTION_FILE \
            --answers-file $ANSWER_FILE \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done

    wait

    output_file=./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

done
