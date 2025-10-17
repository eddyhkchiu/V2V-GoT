#!/bin/bash

echo "nq1 inference"
conda activate llava
cd LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh 4330 nq1sm3w0d gt full

echo "nq2 inference"
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh 4330 nq2sm3w0d gt full
 
echo "nq3 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt 4330 --input_qa_dataset nq2sm3w0d --output_qa_dataset nq3sm3w0dc --graph full

echo "nq3 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh 4330 nq3sm3w0dc graph full

echo "nq4 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt 4330 --input_qa_dataset nq1sm3w0d nq3sm3w0dc --output_qa_dataset nq4sm3w0dc --graph full

echo "nq4 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh 4330 nq4sm3w0dc graph full

echo "nq5 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt 4330 --input_qa_dataset nq4sm3w0dc --output_qa_dataset nq5sm3w1dc --graph full

echo "nq5 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh 4330 nq5sm3w1dc graph full

echo "nq6 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt 4330 --input_qa_dataset nq4sm3w0dc --output_qa_dataset nq6sm3w1dc --graph full

echo "nq6 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh 4330 nq6sm3w1dc graph full


echo "nq7 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt 4330 --input_qa_dataset nq5sm3w1dc nq6sm3w1dc  --output_qa_dataset nq7sm3w1dc --graph full

echo "nq7 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh 4330 nq7sm3w1dc graph full

echo "nq8 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt 4330 --input_qa_dataset nq7sm3w1dc  --output_qa_dataset nq8sm3w6dc --graph full

echo "nq8 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh 4330 nq8sm3w6dc graph full

echo "nq9 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt 4330 --input_qa_dataset nq8sm3w6dc  --output_qa_dataset nq9sm3w6dc --graph full

echo "nq9 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh 4330 nq9sm3w6dc graph full
