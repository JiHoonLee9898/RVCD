#!/bin/bash
NOW=$(date +"%Y%m%d_%H%M%S")

python /home/work/jihoon_wombat_storage/RVCD/MAIN_CODES/eval/chair.py \
--cap_file /home/work/jihoon_wombat_storage/RVCD/MAIN_CODES/generated_captions/chair/llava-1.5/a1.0_b0.0_202505201304_seed_42_samples_500_maxtokens_64_ablation_None/rvcd_llava-1.5_a1.0_b0.0_202505201304_seed_42_samples_500_maxtokens_64_ablation_None_generated_captions.jsonl \
--image_id_key image_id --caption_key caption \
--coco_path /home/work/jihoon_wombat_storage/COCO_DIR/annotations/ \
--save_path /home/work/jihoon_wombat_storage/eyetrack/${NOW}_chair.json \
--cache /home/work/jihoon_wombat_storage/CODES/chair.pkl