#!/bin/bash

### RVCD for CHAIR/BLEU ###
CUDA_VISIBLE_DEVICES=0 python /home/work/jihoon_wombat_storage/RVCD/MAIN_CODES/test.py \
    --model mplug-owl2 \
    --ref_folder_path /home/work/jihoon_wombat_storage/RVCD/DB_single_concept_images_flux_generated/generated_images \
    --data_path /home/work/jihoon_wombat_storage/COCO_DIR \
    --chair_cache_path /home/work/jihoon_wombat_storage/RVCD/MAIN_CODES/eval/CHAIR_CACHE/chair.pkl \
    --yolo_version yolov8x.pt \
    --num_samples 1 \
    --seed 42 \
    --gpu-id 0 \
    --output_dir ./generated_captions/ \
    --rvcd_alpha 1 \
    --rvcd_beta 0 \
    --kv_cache_faster True
