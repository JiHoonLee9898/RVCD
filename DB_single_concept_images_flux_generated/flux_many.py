import os, pickle

# 사용할 GPU를 설정
able_gpus = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = able_gpus  # 예: 첫 번째 GPU만 사용
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 예: 첫 번째와 두 번째 GPU 사용
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 예: GPU 사용하지 않음(CPU로 강제)

import torch

# GPU 디바이스 확인
print("Available GPUs:", able_gpus)
print("GPU count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

import sys
import numpy as np
from transformers import MllamaForConditionalGeneration, AutoProcessor
import requests, random
import matplotlib.cm as cm
import matplotlib.image as mpimg
import cv2
import json
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, LlavaForConditionalGeneration
import matplotlib.pyplot as plt
import shutil, time
import re
from diffusers import FluxPipeline
sys.path.append("../MAIN_CODES/eval")
import chair
from chair import CHAIR  

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('--vlm_path', type=str, default='/datasets2/llava-1.5-7b-hf', required=True, help="")
parser.add_argument('--img_gen_model_path', type=str, default='/datasets2/FLUX.1-dev', required=True, help="")
parser.add_argument('--coco_path', type=str, default='/home/donut2024/coco2014/annotations', required=True, help="")
parser.add_argument('--chair_cache_path', type=str, default='/home/donut2024/JIHOON/RVCD/MAIN_CODES/eval/CHAIR_CACHE/chair.pkl', required=True, help="")
parser.add_argument('--img_db_path', type=str, default='/home/donut2024/JIHOON/RVCD/DB_single_concept_images_flux_generated/generated_images', required=True, help="")

args = parser.parse_args()

chair_cache_path = args.chair_cache_path
coco_path = args.coco_path
vlm_path = args.vlm_path
img_gen_model_path = args.img_gen_model_path
img_db_path = args.img_db_path

# vlm load
##########################
model_id = vlm_path
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    attn_implementation="eager",  
    return_dict_in_generate=True,
).to(0)
processor = AutoProcessor.from_pretrained(model_id)
############################

# img generation model load
############################
pipe = FluxPipeline.from_pretrained(img_gen_model_path, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() 
############################


coco_all_word_include_synonyms_list = [word.strip() for line in chair.synonyms_txt.splitlines() for word in line.split(',') if word.strip()]

global_chair_evaluator = None
def get_chair_evaluator(chair_cache_path=chair_cache_path, coco_path=coco_path):
    global global_chair_evaluator
    if global_chair_evaluator is None:
        if chair_cache_path and os.path.exists(chair_cache_path):
            global_chair_evaluator = pickle.load(open(chair_cache_path, 'rb'))
            print(f"Loaded evaluator from cache: {chair_cache_path}")
        else:
            print("Cache not set or not exist, initializing evaluator...")
            global_chair_evaluator = CHAIR(coco_path)
            pickle.dump(global_chair_evaluator, open(chair_cache_path, 'wb'))
            print(f"Evaluator cached to: {chair_cache_path}")

    return global_chair_evaluator

def chair_change_synonym_to_cocofirst_word(word):
    evaluator = get_chair_evaluator(chair_cache_path, coco_path)
    words, node_words, _, double_words = evaluator.caption_to_words(word)
    print(words, node_words, double_words)
    if len(node_words) == 1:
        return node_words[0]
    else: # 정상처리되지 않은 단어. 나중에 chair 파일에 추가하고
        # 캐시를 새로 초기화해야 chair.pkl이 업데이트됨. 
        return 'chair_add_'+' '.join(double_words)
    
seeds = [i for i in range(100)]
guidance_scale = [i*0.1 for i in range(100)]

for scale in guidance_scale:
    for seed in seeds:
        already_made_img_list = []
        not_yet_list = []

        for root, dirs, files in os.walk(img_db_path):
            for file in files:
                entity = os.path.join(root, file).split('.png')[0].split('/')[-1]
                already_made_img_list.append(entity)

        for word in coco_all_word_include_synonyms_list:
            if word not in already_made_img_list:
                not_yet_list.append(word)

        print(f'아직 이미지가 없는 단어 개수 : {len(not_yet_list)}')
        print(f'coco전체 모든 단어 개수 : {len(coco_all_word_include_synonyms_list)}')

        for word in not_yet_list:
            entity = word
            synonym_firstford_tuple = (chair_change_synonym_to_cocofirst_word(entity), entity)
            print(synonym_firstford_tuple)

            prompt = f"A {entity}({synonym_firstford_tuple[0]}), white background"
            print(prompt)
            print('*'*30)
            print(f'entity : {entity}, seed : {seed}')

            image = pipe(
                prompt,
                height=336,
                width=336,
                guidance_scale=scale,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(seed)
            ).images[0]
            user_input_text = f'Please describe this image in detail.'
            conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input_text},
                    {"type": "image"},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            initial_inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
            draft_caption = model.generate(**initial_inputs, 
                                    max_new_tokens=32, 
                                    do_sample=False,
                                    )
            draft_caption_nl = processor.decode(draft_caption['sequences'][0], skip_special_tokens=True).split('ASSISTANT: ')[-1]
            entity_lower = entity.lower()
            draft_caption_nl_lower = draft_caption_nl.lower()
            print(f"vlm이 {entity}이미지를 보고 만든 캡션.lower() : {draft_caption_nl_lower}")
        
            if entity_lower in draft_caption_nl_lower:
                print(f'아직 이미지가 없는 단어 개수 : {len(not_yet_list)}')
                print(f'coco전체 모든 단어 개수 : {len(coco_all_word_include_synonyms_list)}')
                print(f'저장합니다. {entity} 이미지가 아직 없고, vlm도 이 이미지에서 {entity}가 있다고 합니다.')
                if " " in entity:
                    image.save(os.path.join(img_db_path, f'{entity}.png'))
                    entity = entity.replace(" ", "")
                    image.save(os.path.join(img_db_path, f'{entity}.png'))
                else: 
                    image.save(os.path.join(img_db_path, f'{entity}.png'))
                continue
     
            ##############################################################################
            ## VQA MODE
            # user_input_text = f'Is there {entity}({synonym_firstford_tuple[0]}) in this image? Answer yes or no.'
            # print(user_input_text)
            # conversation = [
            #     {
            #     "role": "user",
            #     "content": [
            #         {"type": "text", "text": user_input_text},
            #         {"type": "image"},
            #         ],
            #     },
            # ]
            # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            # initial_inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
            # draft_caption = model.generate(**initial_inputs, 
            #                         max_new_tokens=32, 
            #                         do_sample=False,
            #                         )
            # draft_caption_nl = processor.decode(draft_caption['sequences'][0], skip_special_tokens=True).split('ASSISTANT: ')[-1]
            # entity_lower = entity.lower()
            # draft_caption_nl_lower = draft_caption_nl.lower()
            # print(f"vlm이 {entity}이미지를 보고 만든 캡션.lower() : {draft_caption_nl_lower}")
            
            # if 'yes' in draft_caption_nl_lower:
            #     print(f'아직 이미지가 없는 단어 개수 : {len(not_yet_list)}')
            #     print(f'coco전체 모든 단어 개수 : {len(coco_all_word_include_synonyms_list)}')
            #     print(f'저장합니다. {entity} 이미지가 아직 없고, vlm도 이 이미지에서 {entity}의 대표어가 있다고 합니다.')
            #     if " " in entity:
            #         image.save(os.path.join(img_db_path, f'{entity}.png'))
            #         entity = entity.replace(" ", "")
            #         image.save(os.path.join(img_db_path, f'{entity}.png'))
            #     else: 
            #         image.save(os.path.join(img_db_path, f'{entity}.png'))
            #     continue
            #######################################

already_made_img_list = []
not_yet_list = []

for root, dirs, files in os.walk(img_db_path):
    for file in files:
        entity = os.path.join(root, file).split('.png')[0].split('/')[-1]
        already_made_img_list.append(entity)

for word in coco_all_word_include_synonyms_list:
    if word not in already_made_img_list:
        not_yet_list.append(word)

print(f'아직 이미지가 없는 단어 개수 : {len(not_yet_list)}')
print(f'coco전체 모든 단어 개수 : {len(coco_all_word_include_synonyms_list)}')