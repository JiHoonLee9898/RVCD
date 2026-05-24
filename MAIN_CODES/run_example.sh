#!/bin/bash

COMMON_DATA_PATH="/home/jihoon/jihoon/DATASETS/coco2014/val2014"
OUT_DIR="./generated_captions/"
NUM_SAMPLES=300
SEED=42
GPU_ID=0
MAX_NEW_TOKENS=64
CHAIR_CACHE_PATH="eval/CHAIR_CACHE/chair.pkl"

### RVCD for CHAIR/BLEU ###
CUDA_VISIBLE_DEVICES=${GPU_ID} python rvcd_generation_chair_bleu.py \
    --model llava-1.5 \
    --ref_folder_path DB_single_concept_images_flux_generated/generated_images \
    --data_path ${COMMON_DATA_PATH} \
    --chair_cache_path ${CHAIR_CACHE_PATH} \
    --yolo_version yolov8x.pt \
    --num_samples ${NUM_SAMPLES} \
    --seed ${SEED} \
    --gpu-id ${GPU_ID} \
    --output_dir ${OUT_DIR} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --rvcd_alpha 1 \
    --rvcd_beta 0.1 \
 


### greedy for CHAIR/BLEU ###
CUDA_VISIBLE_DEVICES=${GPU_ID} python prior_decodings/prior_generation_chair_bleu.py \
    --model not_rvcd_llava \
    --data_path ${COMMON_DATA_PATH} \
    -d greedy \
    --num_samples ${NUM_SAMPLES} \
    --seed ${SEED} \
    --gpu-id ${GPU_ID} \
    --output_dir ${OUT_DIR} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --chair_cache_path ${CHAIR_CACHE_PATH}


### dola for CHAIR/BLEU ###
CUDA_VISIBLE_DEVICES=${GPU_ID} python prior_decodings/prior_generation_chair_bleu.py \
    --model not_rvcd_llava \
    --data_path ${COMMON_DATA_PATH} \
    -d dola \
    --num_samples ${NUM_SAMPLES} \
    --seed ${SEED} \
    --gpu-id ${GPU_ID} \
    --output_dir ${OUT_DIR} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --chair_cache_path ${CHAIR_CACHE_PATH}


### halc for CHAIR/BLEU ###
CUDA_VISIBLE_DEVICES=${GPU_ID} python prior_decodings/prior_generation_chair_bleu.py \
    --model not_rvcd_llava \
    --data_path ${COMMON_DATA_PATH} \
    -d halc \
    --num_samples ${NUM_SAMPLES} \
    --seed ${SEED} \
    --gpu-id ${GPU_ID} \
    --output_dir ${OUT_DIR} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --chair_cache_path ${CHAIR_CACHE_PATH} \
    --beam 3 \
    --k-candidate-num 4 \
    --expand-ratio 0.6 \
    --detector dino \
    --box_threshold 0.4


### opera for CHAIR/BLEU ###
CUDA_VISIBLE_DEVICES=${GPU_ID} python prior_decodings/prior_generation_chair_bleu.py \
    --model not_rvcd_llava \
    --data_path ${COMMON_DATA_PATH} \
    -d opera \
    --num_samples ${NUM_SAMPLES} \
    --seed ${SEED} \
    --gpu-id ${GPU_ID} \
    --output_dir ${OUT_DIR} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --chair_cache_path ${CHAIR_CACHE_PATH} \
    --beam 3 \
    --scale_factor 50 \
    --threshold 15 \
    --num_attn_candidates 5 \
    --penalty_weights 1.0


### vcd for CHAIR/BLEU ###
CUDA_VISIBLE_DEVICES=${GPU_ID} python prior_decodings/prior_generation_chair_bleu.py \
    --model not_rvcd_llava \
    --data_path ${COMMON_DATA_PATH} \
    -d vcd \
    --num_samples ${NUM_SAMPLES} \
    --seed ${SEED} \
    --gpu-id ${GPU_ID} \
    --output_dir ${OUT_DIR} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --chair_cache_path ${CHAIR_CACHE_PATH} \
    --cd_alpha 1 \
    --cd_beta 0.1 \
    --noise_step 500


### beam for CHAIR/BLEU ###
CUDA_VISIBLE_DEVICES=${GPU_ID} python prior_decodings/prior_generation_chair_bleu.py \
    --model not_rvcd_llava \
    --data_path ${COMMON_DATA_PATH} \
    -d beam \
    --num_samples ${NUM_SAMPLES} \
    --seed ${SEED} \
    --gpu-id ${GPU_ID} \
    --output_dir ${OUT_DIR} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --chair_cache_path ${CHAIR_CACHE_PATH} \
    --beam 3











# ### RVCD for POPE ###
# CUDA_VISIBLE_DEVICES=0 python rvcd_generation_pope.py \
#     --model llava-1.5 \
#     --pope_type random \
#     --ref_folder_path /RVCD/DB_single_concept_images_flux_generated/generated_images/ \
#     --data_path /coco2014/ \
#     --chair_cache_path /RVCD/MAIN_CODES/eval/CHAIR_CACHE/chair.pkl \
#     --yolo_version yolov8x.pt \
#     --num_images 3 \
#     --seed 42 \
#     --gpu-id 0 \
#     --output_dir ./generated_captions/ \
#     --rvcd_alpha 1 \
#     --rvcd_beta 0.1

# ### RVCD for MME ###
# CUDA_VISIBLE_DEVICES=0 python rvcd_generation_mme.py \
#     --model llava-1.5 \
#     --ref_folder_path /RVCD/DB_single_concept_images_flux_generated/generated_images/ \
#     --coco_path /coco2014/ \
#     --data_paths /MME/MME_Benchmark_release_version/MME_Benchmark/ \
#     --chair_cache_path /RVCD/MAIN_CODES/eval/CHAIR_CACHE/chair.pkl \
#     --yolo_version yolov8x.pt \
#     --seed 42 \
#     --gpu-id 0 \
#     --output_dir ./generated_captions/ \
#     --rvcd_alpha 1 \
#     --rvcd_beta 0.1 

### Prior SOTA methods for CHAIR/BLEU. Additional arguments described at README.md ###
# CUDA_VISIBLE_DEVICES=0 python prior_decodings/prior_generation_chair_bleu.py \
#     --model not_rvcd_llava \
#     --data_path /home/work/jihoon_wombat_storage/COCO_DIR \
#     -d dola \
#     --num_samples 500 \
#     --seed 43 \
#     --gpu-id 0 \
#     --output_dir ./generated_captions/ 

# ### Prior SOTA methods for POPE. Additional arguments described at README.md ###
# CUDA_VISIBLE_DEVICES=0 python prior_decodings/prior_generation_pope.py \
#     --model not_rvcd_llava \
#     --pope_type random \
#     --data_path /coco2014/ \
#     -d greedy \
#     --num_images 3 \
#     --seed 42 \
#     --gpu-id 0 \
#     --output_dir ./generated_captions/ 

# ### Prior SOTA methods for MME. Additional arguments described at README.md ###
# CUDA_VISIBLE_DEVICES=0 python prior_decodings/prior_generation_mme.py \
#     --model not_rvcd_llava \
#     --data_paths /MME/MME_Benchmark_release_version/MME_Benchmark/ \
#     -d greedy \
#     --seed 42 \
#     --gpu-id 0 \
#     --output_dir ./generated_captions/ 