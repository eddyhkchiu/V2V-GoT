#!/bin/bash

#cd /ocean/projects/cis230055p/hchiu1/V2V-GoT/LLaVA
cd LLaVA
conda activate llava
nvidia-smi
#export PYTHONPATH=/ocean/projects/cis230055p/hchiu1/V2V-GoT/LLaVA:$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
echo $PYTHONPATH

MODEL="v2vllmq5_10ep_both_shallow_f2_cobevt"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
SPLIT='val'

for CKPT_ID in 490
do
    EXP="v2v4real_3d_grounding_${MODEL}_${CKPT_ID}"
    echo $EXP

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path ./checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_${MODEL}/checkpoint-$CKPT_ID \
            --mm_scene_projector_input_size 3072 \
            --scene_level_only False \
            --object_level_only False \
            --scene_feature_mode shallow \
            --object_feature_mode shallow \
            --num_input_frames 2 \
            --ego_only True \
            --feature_source cobevt \
            --model-base liuhaotian/llava-v1.5-7b \
            --question-file ../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v2vllmq5.json \
            --answers-file ./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done

    wait

    output_file=./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done


done

