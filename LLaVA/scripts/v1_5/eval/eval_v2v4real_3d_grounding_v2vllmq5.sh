#!/bin/bash

cd DMSTrack/DMSTrack
source dmstrack_init_env.sh
cd ../../LLaVA

MODEL=$1

echo $MODEL > results/${MODEL}.txt

for CKPT_ID in 490
do
        echo $CKPT_ID >> results/${MODEL}.txt	
        python scripts/eval_v2v4real_3d_grounding.py \
            --simplified \
            --multiple-output \
            --max-num-answer-objects 3 \
            --qa-type-id 5 \
            --num-future-waypoints 6 \
            --npy-save-path ../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy \
            --answers-file ./playground/data/eval/v2v4real_3d_grounding_${MODEL}_${CKPT_ID}/answers/val/llava-v1.5-7b/merge.jsonl \
	    >> results/${MODEL}.txt
done

cd ../
