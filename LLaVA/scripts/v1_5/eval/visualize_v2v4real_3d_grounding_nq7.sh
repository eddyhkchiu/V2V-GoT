#!/bin/bash

cd DMSTrack/DMSTrack
source dmstrack_init_env.sh
cd ../../LLaVA

nvidia-smi

MODEL=$1
DATA=$2
GRAPH=$3
CKPT_ID_INPUT=$4

VISUALIZATION_DOUBLE=$5
VISUALIZATION_OUTPUT_FOLDER=$6

for CKPT_ID in $CKPT_ID_INPUT
do
    python scripts/eval_v2v4real_3d_grounding.py \
	    --visualization-only \
	    --visualization-double ${VISUALIZATION_DOUBLE} \
            --visualization-output-folder ${VISUALIZATION_OUTPUT_FOLDER} \
            --simplified \
            --multiple-output \
            --max-num-answer-objects 3 \
            --qa-type-id 17 \
            --num-future-waypoints 1 \
            --npy-save-path ../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy \
            --answers-file ./playground/data/eval/v2v4real_3d_grounding_${MODEL}_${CKPT_ID}_${GRAPH}_${DATA}/answers/val/llava-v1.5-7b/merge.jsonl
done

cd ../
