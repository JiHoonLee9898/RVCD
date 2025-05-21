#!/bin/bash
NOW=$(date +"%Y%m%d_%H%M%S")

python /home/work/jihoon_wombat_storage/RVCD/MAIN_CODES/eval/chair.py \
--cap_file /home/work/jihoon_wombat_storage/RVCD/MAIN_CODES/generated_captions/chair/not_rvcd_llava/halc_202505201327_seed_42_samples_500_maxtokens_64/not_rvcd_llava_halc_dino_box_0.4_beams_1_k_4_coco_expand_ratio_0.6_seed_42_max_tokens_64_samples_500_skip_0_generated_captions.json \
--image_id_key image_id --caption_key caption \
--coco_path /home/work/jihoon_wombat_storage/COCO_DIR/annotations/ \
--save_path /home/work/jihoon_wombat_storage/eyetrack/${NOW}_chair.json \
--cache /home/work/jihoon_wombat_storage/CODES/chair.pkl