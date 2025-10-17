#!/bin/bash

cd DMSTrack/DMSTrack
source dmstrack_init_env.sh
cd ../../LLaVA

MODEL=$1
VISUALIZATION_DOUBLE=$2
VISUALIZATION_OUTPUT_FOLDER=$3

for CKPT_ID in 490
do
        python scripts/eval_v2v4real_3d_grounding.py \
	    --visualization-only \
            --visualization-double ${VISUALIZATION_DOUBLE} \
	    --visualization-output-folder ${VISUALIZATION_OUTPUT_FOLDER} \
            --simplified \
            --multiple-output \
            --max-num-answer-objects 3 \
            --qa-type-id 5 \
            --num-future-waypoints 6 \
            --npy-save-path ../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy \
            --answers-file ./playground/data/eval/v2v4real_3d_grounding_${MODEL}_${CKPT_ID}/answers/val/llava-v1.5-7b/merge.jsonl
done

cd ../
