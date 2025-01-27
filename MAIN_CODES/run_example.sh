#!/bin/bash

### RVCD for CHAIR/BLEU ###
CUDA_VISIBLE_DEVICES=0 python rvcd_generation_chair_bleu.py \
    --model llava-1.5 \
    --ref_folder_path /home/donut2024/JIHOON/RVCD/DB_single_concept_images_flux_generated/generated_images \
    --data_path /home/donut2024/coco2014/ \
    --chair_cache_path /home/donut2024/JIHOON/RVCD/MAIN_CODES/eval/CHAIR_CACHE/chair.pkl \
    --yolo_version yolov8x.pt \
    --num_samples 5 \
    --seed 42 \
    --gpu-id 0 \
    --output_dir ./generated_captions/ \
    --rvcd_alpha 1 \
    --rvcd_beta 0.1 

### RVCD for POPE ###
CUDA_VISIBLE_DEVICES=0 python rvcd_generation_pope.py \
    --model llava-1.5 \
    --pope_type random \
    --ref_folder_path /home/donut2024/JIHOON/RVCD/DB_single_concept_images_flux_generated/generated_images \
    --data_path /home/donut2024/coco2014/ \
    --chair_cache_path /home/donut2024/JIHOON/RVCD/MAIN_CODES/eval/CHAIR_CACHE/chair.pkl \
    --yolo_version yolov8x.pt \
    --num_images 5 \
    --seed 42 \
    --gpu-id 0 \
    --output_dir ./generated_captions/ \
    --rvcd_alpha 1 \
    --rvcd_beta 0.1 

### RVCD for MME ###
CUDA_VISIBLE_DEVICES=0 python rvcd_generation_mme.py \
    --model llava-1.5 \
    --ref_folder_path /home/donut2024/JIHOON/RVCD/DB_single_concept_images_flux_generated/generated_images \
    --coco_path /home/donut2024/coco2014 \
    --data_paths /home/donut2024/JIHOON/MME_Benchmark_release_version/MME_Benchmark \
    --chair_cache_path /home/donut2024/JIHOON/RVCD/MAIN_CODES/eval/CHAIR_CACHE/chair.pkl \
    --yolo_version yolov8x.pt \
    --seed 42 \
    --gpu-id 0 \
    --output_dir ./generated_captions/ \
    --rvcd_alpha 1 \
    --rvcd_beta 0.1 

#########################################################


### --model : not_rvcd_llava, not_rvcd_mini_gpt4, not_rvcd_mplug_owl2
### Prior SOTA methods for CHAIR/BLEU ###
CUDA_VISIBLE_DEVICES=0 python prior_decodings/prior_generation_chair_bleu.py \
    --model not_rvcd_llava \
    --data_path /home/donut2024/coco2014/ \
    -d greedy \
    --num_samples 5 \
    --seed 42 \
    --gpu-id 0 \
    --output_dir ./generated_captions/ 

### Prior SOTA methods for POPE ###
CUDA_VISIBLE_DEVICES=0 python prior_decodings/prior_generation_pope.py \
    --model not_rvcd_llava \
    --pope_type random \
    --data_path /home/donut2024/coco2014 \
    -d greedy \
    --num_images 5 \
    --seed 42 \
    --gpu-id 0 \
    --output_dir ./generated_captions/ 

### Prior SOTA methods for MME ###
CUDA_VISIBLE_DEVICES=0 python prior_decodings/prior_generation_mme.py \
    --model not_rvcd_llava \
    --data_paths /home/donut2024/JIHOON/MME_Benchmark_release_version/MME_Benchmark \
    -d greedy \
    --num_samples 5 \
    --seed 42 \
    --gpu-id 0 \
    --output_dir ./generated_captions/ 