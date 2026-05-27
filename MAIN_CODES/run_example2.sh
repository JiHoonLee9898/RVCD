#!/usr/bin/env bash
set -euo pipefail

COMMON_DATA_PATH="/home/jihoon/jihoon/DATASETS/coco2014/val2014"
OUT_DIR="./generated_captions_ervcd_grid_modes_OPTIMIZED/"
NUM_SAMPLES=300
GPU_ID=0
MAX_NEW_TOKENS=64
CHAIR_CACHE_PATH="eval/CHAIR_CACHE/chair.pkl"
REF_FOLDER_PATH="DB_single_concept_images_flux_generated/generated_images"

SEEDS=(42 43 44)
MODES=(black_back black_front repeat repeat_front repeat_last)

for MODE in "${MODES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "============================================================"
    echo "Running eRVCD | mode=${MODE} | seed=${SEED}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python ervcd_generation_chair_bleu_optimized.py \
      --model llava-1.5 \
      --data_path "${COMMON_DATA_PATH}" \
      --ref_folder_path "${REF_FOLDER_PATH}" \
      --chair_cache_path "${CHAIR_CACHE_PATH}" \
      --num_samples "${NUM_SAMPLES}" \
      --seed "${SEED}" \
      --gpu-id "${GPU_ID}" \
      --output_dir "${OUT_DIR}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --rvcd_alpha 1 \
      --rvcd_beta 0.1 \
      --ervcd_grid_fill_mode "${MODE}" \
      --ervcd_logit_scale_mode presence
  done
done