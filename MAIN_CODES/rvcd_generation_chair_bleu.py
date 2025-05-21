
import argparse
import os
import random
import sys
sys.path.append("mPLUG-Owl/mPLUG-Owl2")
sys.path.append("mPLUG-Owl/mPLUG-Owl2")
sys.path.append("./")
sys.path.append("../")
sys.path.append("./eval")
sys.path.append("./YOLO")
import yolo
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms
from minigpt4.models import load_preprocess
from minigpt4.common.config import Config
from minigpt4.common.registry import registry


from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from PIL import Image
import json
# from decoder_zoo.HALC.context_density.halc import halc_assistant
from pycocotools.coco import COCO
from collections import defaultdict
import torch
from PIL import Image
from transformers import TextStreamer
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
import pickle
from chair import CHAIR  # 위의 코드를 저장한 파일명을 your_module로 변경
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    # "instructblip": "eval_configs/instructblip_eval.yaml",
    # "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    # "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "mplug-owl2": "eval_configs/mplug-owl2_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    # "instructblip": "<ImageHere><question>",
    # "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    # "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "mplug-owl2": "USER: <|image|><question> ASSISTANT:",
}

def setup_seeds(config, seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, default="llava-1.5", help="model")
parser.add_argument(
        "--ref_folder_path",
        type=str,
        default="/home/donut2024/JIHOON/RVCD/DB_single_concept_images_flux_generated/generated_images",
        help="single concept AI image DB path.",
    )
parser.add_argument("-g", "--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
parser.add_argument("--dataset_name",type=str,default="coco",help="Name of the dataset. Default is 'coco'.",)
parser.add_argument("--data_path",type=str,default="/home/donut2024/coco2014",help="data path",)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("-n", "--num_samples", type=int, default=500)
parser.add_argument("-m", "--max_new_tokens", type=int, default=64)
parser.add_argument("--output_dir",type=str,default="./generated_chair_inputs/",help="Output ditectory for saving test results. Default is './generated_chair_inputs/'.",)
parser.add_argument("--options",nargs="+",help="override some settings in the used config, the key-value pair ""in xxx=yyy format will be merged into config file (deprecate), ""change to --cfg-options instead.",)
parser.add_argument("--chair_cache_path",type=str,default="/home/donut2024/JIHOON/RVCD/MAIN_CODES/eval/CHAIR_CACHE/chair.pkl",help="chair_pickle_path",)
###############################

############ RVCD ############# 
parser.add_argument("--yolo_version",type=str,default="yolov8x.pt",help="yolo")
parser.add_argument("--check_draft_chair", type=str2bool, default=True, help="기본 True, draft캡션을 평가. 평가 결과와 무관하게 RVCD 수행")
parser.add_argument("--ablation_rvcd_all", type=str2bool, default=False, help="기본 False, True로 바꾸면 nvcd에서 draft의 모든 객체를 제거")
parser.add_argument("--ablation_rvcd_gt", type=str2bool, default=False, help="기본 False, True로 바꾸면 nvcd에서 draft의 gt를 제거, gt는 chair를 매 draft마다 체크해서 산출")
parser.add_argument("--ablation_rvcd_hal", type=str2bool, default=False, help="기본 False, True로 바꾸면 nvcd에서 draft의 hal를 제거, hal는 chair를 매 draft마다 체크해서 산출")
parser.add_argument("--rvcd_alpha", type=float, default=1, help='기본 1, rvcd의 negative logits 규제율') 
parser.add_argument("--rvcd_beta", type=float, default=0.1, help='기본 0.1, rvcd의 positive logits 회복률') 
parser.add_argument("--rvcd_gamma", type=float, default=0, help='선행 연구들에서 제시하는 패널티 term. 이 연구에서는 0') 
parser.add_argument("--kv_cache_faster", type=str2bool, default=True, help='generate kv cache.') 

################################
args = parser.parse_known_args()[0]
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)

###################################
yolo_version = args.yolo_version
model_name = args.model
decoding_strategy = 'rvcd'
seed = args.seed
num_samples = args.num_samples
dataset_name = args.dataset_name
data_path = args.data_path
chair_cache_path = args.chair_cache_path
output_dir = args.output_dir
num_beams = 1
batch_size = 1
max_new_tokens = args.max_new_tokens


check_draft_chair = args.check_draft_chair
ablation_rvcd_all = args.ablation_rvcd_all
ablation_rvcd_gt = args.ablation_rvcd_gt
ablation_rvcd_hal = args.ablation_rvcd_hal

true_flags = sum([
    ablation_rvcd_all,
    ablation_rvcd_gt,
    ablation_rvcd_hal
])
# 조건: 셋 중 하나만 True이거나, 모두 False여야 함
if true_flags > 1:
    sys.exit("Error: At most one of --ablation_rvcd_all, --ablation_rvcd_gt, or --ablation_rvcd_hal can be True.")

if not check_draft_chair: 
    sys.exit("Error: check_draft_chair는 평가를 위해 항상 True여야 합니다.")

######################################
setup_seeds(cfg, seed)
device = (torch.device(f"cuda:{int(args.gpu_id)}") if torch.cuda.is_available() else "cpu")

# ========================================
#             Model Initialization
# ========================================
print("Initializing Model")

# print("cfg", cfg)
# input()
model_config = cfg.model_cfg 
print(f'model_config : {model_config}')
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()

###############
def load_tokenizer(model_config):
    if model_name == 'llava-1.5' or model_name == 'mplug-owl2':
        tokenizer_path = 'merged_ckpt'
    elif model_name == 'minigpt4':
        tokenizer_path = 'llama_model'
    tokenizer = AutoTokenizer.from_pretrained(model_config[tokenizer_path], use_fast=False)
    return tokenizer
model_tokenizer = load_tokenizer(model_config)
################


processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
    vis_processor_cfg
)

print(f"\033[42m####### Current Decoding Strategy: {decoding_strategy} #######\033[0m")

# HALC (https://arxiv.org/abs/2403.00425) 
# (https://github.com/BillChan226/HALC)
# 에서 제시하는 정규화 term.
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)


annotation_file_path = os.path.join(args.data_path, 'annotations/instances_val2014.json')
# annotation_file_path = "/home/donut2024/coco2014/annotations/instances_val2014.json"
caption_file_path = os.path.join(args.data_path, 'annotations/captions_val2014.json')
# caption_file_path = "/home/donut2024/coco2014/annotations/captions_val2014.json"

with open(annotation_file_path, "r") as f: lines = f.readlines()
coco_anns = json.loads(lines[0])
coco = COCO(caption_file_path)
img_ids = coco.getImgIds()

sampled_img_ids = random.sample(img_ids, num_samples)
print("sampled_img_ids", len(sampled_img_ids))

img_files = []
for cur_img_id in sampled_img_ids:
    cur_img = coco.loadImgs(cur_img_id)[0]
    cur_img_path = cur_img["file_name"]
    img_files.append(cur_img_path)

img_dict = {}

categories = coco_anns["categories"]
category_names = [c["name"] for c in categories]
category_dict = {int(c["id"]): c["name"] for c in categories}

for img_info in coco_anns["images"]:
    img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

for ann_info in coco_anns["annotations"]:
    img_dict[ann_info["image_id"]]["anns"].append(
        category_dict[ann_info["category_id"]]
    )

base_dir = os.path.join(output_dir, "chair", args.model) # outputdir/chair/llava-1.5 형태
if not os.path.exists(base_dir):
    os.makedirs(base_dir)


#############################################################

def process_before_norm(img_path):
        raw_image = Image.open(img_path).convert('RGB')
        if model_name == "mplug-owl2":
            max_edge = max(raw_image.size) # We recommand you to resize to squared image for BEST performance.
            image = raw_image.resize((max_edge, max_edge))
            image_tensor = process_images([image], model.image_processor)
            image = image_tensor.to(device, dtype=torch.float16)
        else:
            image = vis_processors["eval"](raw_image).unsqueeze(0)  # 얘가 이미지 프로세서 
            image = image.to(device)
        return image


global_chair_evaluator = None
coco_path = os.path.join(args.data_path, 'annotations')
def get_chair_evaluator(chair_cache_path=chair_cache_path, coco_path=coco_path):
    """
    Load or initialize the global CHAIR evaluator object.

    Args:
        chair_cache_path (str): Path to the cached CHAIR evaluator.
        coco_path (str): Path to the COCO dataset annotations.

    Returns:
        CHAIR: A CHAIR evaluator object.
    """
    global global_chair_evaluator

    if global_chair_evaluator is None:
        # Load from cache or initialize
        if chair_cache_path and os.path.exists(chair_cache_path):
            global_chair_evaluator = pickle.load(open(chair_cache_path, 'rb'))
            print(f"Loaded evaluator from cache: {chair_cache_path}")
        else:
            print("Cache not set or not exist, initializing evaluator...")
            global_chair_evaluator = CHAIR(coco_path)
            pickle.dump(global_chair_evaluator, open(chair_cache_path, 'wb'))
            print(f"Evaluator cached to: {chair_cache_path}")

    return global_chair_evaluator

def evaluate_sentence(sentence, image_id, chair_cache_path=chair_cache_path, coco_path=coco_path):
    """
    Given a sentence, evaluates if the objects in the sentence are ground-truth or hallucinated.

    Args:
        sentence (str): The input sentence to evaluate.
        chair_cache_path (str): Path to the cached CHAIR evaluator.
        coco_path (str): Path to the COCO dataset annotations.

    Returns:
        dict: A dictionary indicating which objects are ground-truth or hallucinated.
    """
    evaluator = get_chair_evaluator(chair_cache_path, coco_path)

    # Use CHAIR's `caption_to_words` method to analyze the sentence
    words, node_words, _, _ = evaluator.caption_to_words(sentence)
    gt_objects = evaluator.imid_to_objects.get(image_id, set())  # Ground-truth objects (for all images, for context)


    results = {"ground_truth": [], "hallucinated": []}
    for word, node_word in zip(words, node_words):
        if node_word in gt_objects:  # Check if the word is in the ground-truth objects
            results["ground_truth"].append((node_word,word))
        else:
            results["hallucinated"].append((node_word,word))

    return results

def chair_change_synonym_to_cocofirst_word(word):
    evaluator = get_chair_evaluator(chair_cache_path, coco_path)
    words, node_words, _, double_words = evaluator.caption_to_words(word)
    print(words, node_words, double_words)
    if len(node_words) == 1:
        return node_words[0]
    else: # 정상처리되지 않은 단어. 나중에 chair 파일에 추가하고
        # 캐시를 새로 초기화해야함. 잇던 캐시 지우기
        return 'chair_add_'+' '.join(double_words)
       
    
def calculate_metrics(chair1_detect1, chair1_detect0, chair0_detect1, chair0_detect0):
    """
    입력값으로 각 감지 결과를 받아 비율, Accuracy, Recall, Precision을 계산하여 출력하는 함수.
    
    Args:
        chair1_detect1 (int): Chair1 -> Detect1 (True Positive, TP)
        chair1_detect0 (int): Chair1 -> Detect0 (False Negative, FN)
        chair0_detect1 (int): Chair0 -> Detect1 (False Positive, FP)
        chair0_detect0 (int): Chair0 -> Detect0 (True Negative, TN)
    """
    # 전체 데이터 합계
    accumulated_total_to_now = chair1_detect1 + chair1_detect0 + chair0_detect1 + chair0_detect0
    if accumulated_total_to_now == 0:
        print("입력된 모든 데이터가 0입니다. 계산을 수행할 수 없습니다. Ablation을 진행하셨다면 정상적인 상황입니다.")
        return {
            "accuracy": None,
            "recall_chair1": None,
            "recall_chair0": None,
            "precision_chair1": None,
            "precision_chair0": None,
            "ratios": None
        }
    
    # Accuracy 계산
    accuracy = (chair1_detect1 + chair0_detect0) / accumulated_total_to_now
    
    # Recall 계산
    recall_chair1 = chair1_detect1 / (chair1_detect1 + chair1_detect0) if (chair1_detect1 + chair1_detect0) != 0 else 0
    recall_chair0 = chair0_detect0 / (chair0_detect0 + chair0_detect1) if (chair0_detect0 + chair0_detect1) != 0 else 0
    
    # Precision 계산
    precision_chair1 = chair1_detect1 / (chair1_detect1 + chair0_detect1) if (chair1_detect1 + chair0_detect1) != 0 else 0
    precision_chair0 = chair0_detect0 / (chair0_detect0 + chair1_detect0) if (chair0_detect0 + chair1_detect0) != 0 else 0
    
    return {
        "Accuracy": accuracy,
        "Recall (Chair GT)": recall_chair1,
        "Recall (Chair HAL)": recall_chair0,
        "Precision (Chair GT)": precision_chair1,
        "Precision (Chair HAL)": precision_chair0
    }

#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

true_flag_name = "ablation_None"
if ablation_rvcd_all:
    true_flag_name = "ablation_rvcd_all"
elif ablation_rvcd_gt:
    true_flag_name = "ablation_rvcd_gt"
elif ablation_rvcd_hal:
    true_flag_name = "ablation_rvcd_hal"

from datetime import datetime
current_time = datetime.now()
formatted_time = current_time.strftime("%Y%m%d%H%M")
result_dir = os.path.join(base_dir, f'a{args.rvcd_alpha}_b{args.rvcd_beta}_{formatted_time}_seed_{seed}_samples_{num_samples}_maxtokens_{max_new_tokens}_{true_flag_name}')
if not os.path.exists(result_dir): os.makedirs(result_dir)

global_all_info = {
    'model_name' : model_name,
    'decoding_strategy' : 'rvcd',
    'seed' : seed,
    'num_samples' : num_samples,
    'max_new_tokens' : max_new_tokens,
    'dataset_name' : dataset_name,
    'data_path' : data_path,
    'output_dir' : output_dir,
    'num_beams' : num_beams,
    'batch_size' : batch_size,
    'ref_not_exist' : [],
    'chair1_detect1' : 0,
    'chair0_detect0' : 0,
    'chair1_detect0' : 0,
    'chair0_detect1' : 0,
    'total_detector_score' : [],
    'chair_not_yet_doublewords' : [],
    'latency' : 0,
    'total_generated_tokens' : 0,
    'latency_per_token' : 0,
}




####### CHAIR/BLEU seed check #########
seed_valid_check = []
for path in img_files:
    img_id = int(path.split(".jpg")[0][-6:])
    seed_valid_check.append(img_id)
seed_valid_check = sorted(seed_valid_check)
print(f'시드 : {seed} / 샘플링된 이미지들 : {seed_valid_check[:20]}')
import time
time.sleep(5)
start_time = time.time()
#######################################
for idx, img_id in tqdm(enumerate(range(len(img_files))), total=len(img_files)):


    img_file = img_files[img_id]
    img_id = int(img_file.split(".jpg")[0][-6:])
    img_info = img_dict[img_id]
    assert img_info["name"] == img_file
    img_anns = set(img_info["anns"])
    img_save = {}
    img_save["image_id"] = img_id
    image_path = os.path.join(args.data_path, img_file)
    image = process_before_norm(image_path)

    # CHAIR, BLEU default image captioning propmt. 
    qu = "Please describe this image in detail."
    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)

    # DRAFT caption generation 
    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": norm(image), "prompt":qu, "img_path": image_path},
                use_nucleus_sampling=args.sample,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True, 
                output_attentions=True,
                return_dict_in_generate=True,
                nvcd=False,
                nvcd_previous_last_ids_list=[], 
            )

    ####################
    # print(f'out.keys() : {out.keys()}') # llava.py 참고고
    # attentions = out['attentions']
    # print(len(attentions))
    # print(len(attentions[0]))
    # print(attentions[0][0].shape)

    all_nl_tokens = [model_tokenizer.convert_ids_to_tokens(seq) for seq in out["sequences"].tolist()][0]

    # minigpt4는 예외, input_nl_tokens에도 output_nl_tokens가 들어있음
    # MAIN_CODES/minigpt4/models/mini_gpt4.py 참고

    input_nl_tokens = [model_tokenizer.convert_ids_to_tokens(seq) for seq in out["input_token_ids"].tolist()][0]
    output_nl_tokens = [model_tokenizer.convert_ids_to_tokens(seq) for seq in out["output_token_ids"].tolist()][0]
    print('-'*30)
    print(f'input_nl_tokens : {len(input_nl_tokens)}, {input_nl_tokens}')
    print('-'*30)
    print(f'output_nl_tokens : {len(output_nl_tokens)}, {output_nl_tokens}')

    all_tokens_to_text = model_tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)[0]
    draft_output_text = all_tokens_to_text

    if model_name == 'minigpt4':
        draft_output_text = draft_output_text.split('###')[0].split('Assistant:')[-1].strip()
    else:
        draft_output_text = draft_output_text.split('ASSISTANT: ')[-1]

    token_count = len(output_nl_tokens)
    print(token_count)
    #######################################

    # draft의 chair 정답 객체를 알아야 하는 경우
    # draft에 chair를 돌리고, gt와 hal을 찾아냄. 이게 정답지 역할.
    # 이후 all, gt, hal ablation 중 하나가 true이면 이 정답지를 기반으로 수행.

    # draft_chair_answer_dict = None
    # if check_draft_chair: #draft마다 chair check
    #     draft_chair_answer_dict = evaluate_sentence(draft_output_text, img_id)
    #     # print(f'draft_chair_answer_dict : {draft_chair_answer_dict}')
    #     #{"ground_truth": [], "hallucinated": []} 안에 (firstword, synonym) 들이 들어감
    #     chair_answer_dict = {(cocofirst, cocosynonym): 1 for cocofirst, cocosynonym in draft_chair_answer_dict["ground_truth"]}
    #     chair_answer_dict.update({(cocofirst, cocosynonym): 0 for cocofirst, cocosynonym in draft_chair_answer_dict["hallucinated"]})
    #     draft_chair_answer_dict = chair_answer_dict 
    #     #{("dog", "hound"): 1, ("cat", "feline"): 0, ("traffic light", "signal"): 1, ("chasing", "pursue"): 0} 형태로 변경.
    #     # 중복 제거된 상태이지만, 첫값은 같고 뒤값은 다른 키는 중복제거 안함

    # if check_draft_chair and draft_chair_answer_dict is not None: # draft의 chair를 체크하는 플래그가 켜지면. 항상 켜야함!

    if True:
        # DETECTOR ABLATION ON. 
        ###########################
        # ablation_rvcd는 셋 중 하나만 true여야 함
        # coco first word빼지말고 synonym을빼야하므로, key[1]을 뺌

        #draft_chair_answer_dicts는 ->
        #{("dog", "hound"): 1, ("cat", "feline"): 0, ("traffic light", "signal"): 1, ("chasing", "pursue"): 0} 형태
        # chair_answer_synonym_gt_list = list(set([key[1] for key, value in draft_chair_answer_dict.items() if value == 1]))
        # chair_answer_synonym_hal_list = list(set([key[1] for key, value in draft_chair_answer_dict.items() if value == 0]))

        # if ablation_rvcd_all: # rvcd를, draft에서 모든 탐지된 객체에 적용. 
            
        #     hal_detected = chair_answer_synonym_gt_list + chair_answer_synonym_hal_list
        #     gt_detected = []

        # elif ablation_rvcd_gt: # rvcd를, draft에서 모든 탐지된 gt에 적용. 즉 detector 정확도가 0%

        #     hal_detected = chair_answer_synonym_gt_list
        #     gt_detected = chair_answer_synonym_hal_list

        # elif ablation_rvcd_hal: # rvcd를, draft에서 모든 탐지된 hal에 적용. 즉 detector 정확도가 100%

        #     hal_detected = chair_answer_synonym_hal_list
        #     gt_detected = chair_answer_synonym_gt_list
        # else:
        if True: # rvcd ablation 안하는경우 (일반적인 rvcd)

            number = img_id
            
            input_image_path = os.path.join(args.data_path, f'COCO_val2014_{int(number):012d}.jpg')
            yolo_detected_entity_prob = yolo.main(input_image_path, yolo_version)
            yolo_detected_entity_list = []
            for entity,prob in yolo_detected_entity_prob:
                yolo_detected_entity_list.append(entity)

            for i in range(len(yolo_detected_entity_list)):
                # yolo가 detect한 entity명이 chair사전의 단어와 호환불가한 경우 기록.
                cocofirst_or_notyetword = chair_change_synonym_to_cocofirst_word(yolo_detected_entity_list[i])
                if cocofirst_or_notyetword.startswith("chair_add_"):
                    # 처리불가단어. yolo_detected_entity_list[i] 그대로 둠
                    # 나중에 MAIN_CODES/eval/chair.py 에서 CHAIR(object)__init__에서 추가해야함
                    # 추가하고 MAIN_CODES/eval/CHAIR_CACHE/ 의 캐시 초기화 필요.
                    global_all_info['chair_not_yet_doublewords'].append(cocofirst_or_notyetword.split("chair_add_")[-1])
                else: # chair사전의 단어와 호환가능하면 대표어로 변환 
                    yolo_detected_entity_list[i] = cocofirst_or_notyetword

            detected_info = {}
            draft_synonyms = global_chair_evaluator.process_sentence_get_coco_synonyms(draft_output_text)
            for synonym in draft_synonyms:
                # synonym = (cocofirstword, synonymfromdraft) 
                if synonym[0] in yolo_detected_entity_list: detected_info[synonym] = 1
                else: detected_info[synonym] = 0

            print(f'detected_info : {detected_info}') 

            #{("dog", "hound"): 1, ("cat", "feline"): 0, ("traffic light", "signal"): 1, ("chasing", "pursue"): 0}
            # print(f'draft_chair_answer_dict : {draft_chair_answer_dict}')
            #{"ground_truth": [("dog","웰시코기"), ("traffic light", "신호등")], "hallucinated": [("chasing", "pursue"]}

            # for chair_key, infer_value in draft_chair_answer_dict.items():
            #     chair_first = chair_key[0]  # draft_chair_answer_dict의 키의 첫 번째 값
            #     for detected_key, gt_value in detected_info.items():
            #         detected_first = detected_key[0]  # detected_info의 키의 첫 번째 값
            #         # 첫 번째 값(대표어) 동일한 경우만 기록
            #         if chair_first == detected_first:
            #             if gt_value == 1 and infer_value == 1:
            #                 global_all_info['chair1_detect1'] += 1
            #             elif gt_value == 1 and infer_value == 0:
            #                 global_all_info['chair1_detect0'] += 1
            #             elif gt_value == 0 and infer_value == 1:
            #                 global_all_info['chair0_detect1'] += 1
            #             elif gt_value == 0 and infer_value == 0:
            #                 global_all_info['chair0_detect0'] += 1
            # accumulated_detector_score = calculate_metrics(global_all_info['chair1_detect1'], 
            #                                                 global_all_info['chair1_detect0'], 
            #                                                 global_all_info['chair0_detect1'],
            #                                                 global_all_info['chair0_detect0'])
            # print(f'accumulated_detector_score : {accumulated_detector_score}')

            hal_detected = []
            for key, value in detected_info.items():
                if value == 0:  # 값이 0인 경우
                    hal_detected.append(key[1]) # coco first word로 바뀌기 전의 synonym들을 저장
           
            gt_detected = []
            for key, value in detected_info.items():
                if value == 1:  # 값이 1인 경우
                    gt_detected.append(key[1]) # coco first word로 바뀌기 전의 synonym들을 저장

    ###############################
    

    hall_ref_list = []
    if len(hal_detected) > 0: # draft 캡션에서 지워야하는 객체가 있다면
        ref_folder = args.ref_folder_path
        hall_ref_list = [os.path.join(ref_folder,f'{synonym}.png') for synonym in hal_detected]
        for i in range(len(hall_ref_list) - 1, -1, -1):  # 역순 순회
            if not os.path.exists(hall_ref_list[i]):
                global_all_info['ref_not_exist'].append(hall_ref_list[i]) # 맨 위에 정의한 글로벌 리스트에 없는거 기록
                hall_ref_list.pop(i)  # 존재하지 않는 경로 제거
                

    gt_ref_list = []
    if len(gt_detected) > 0: # gt가 감지되었다면
        ref_folder = args.ref_folder_path
        gt_ref_list = [os.path.join(ref_folder,f'{synonym}.png') for synonym in gt_detected]
        for i in range(len(gt_ref_list) - 1, -1, -1):  # 역순 순회
            if not os.path.exists(gt_ref_list[i]):
                global_all_info['ref_not_exist'].append(gt_ref_list[i]) # 맨 위에 정의한 글로벌 리스트에 없는거 기록
                gt_ref_list.pop(i)  # 존재하지 않는 경로 제거
                

    # 모든 처리 후에도 없앨 ref이미지 경로가 존재한다면 rvcd.
    # 존재하지 않는다면 negative logit을 만들 수 없으므로 draft 캡션을 그대로 return.
    if len(hall_ref_list) > 0:
        nvcd_operate = True
    else: 
        nvcd_operate = False

    print(f'hall_ref_list : {hall_ref_list}')
    print(f'gt_ref_list : {gt_ref_list}')
   
    now_datapoint_draft_caption = None
    now_datapoint_final_caption = None

    ################################################
    # RVCD
    ################################################

    if nvcd_operate:
        image_kv_cache = {} 
        past_key_values = None 
        output_tokens = []
        
        # 모델의 vocab head. (입력 텐서의 차원 크기, 출력 사전의 모든 토큰 수) 형태의 2차원 매트릭스.
        if model_name == 'mplug-owl2': lm_head_matrix = model.model.lm_head.weight
        else: lm_head_matrix = model.llama_model.lm_head.weight

        for output_index in range(max_new_tokens):
        
            original_img_path = image_path
            negative_img_path = hall_ref_list
            positive_img_path = gt_ref_list


            if args.rvcd_beta == 0 : positive_img_path = []

            original_logit = None
            negative_logits = []
            positive_logits = []

            if len(output_tokens) == 0: #최초토큰생성
                nvcd = False 
                # False이지만, llava.py와 같은 모델 정의 파일에서 확인가능하듯
                # 첫 토큰 포함한 모든 디코딩 스텝에서 RVCD 수행
            else:
                nvcd = True

            # 원본 이미지 v, negative image N 안의 모든 이미지에 대해
            for path in [original_img_path]+negative_img_path:
                
                image = process_before_norm(path) #원본 이미지와 N 이미지들.
                kv_cache = image_kv_cache.get(path, None)

                ##############################################################
                output = model.generate(
                    {"image": norm(image), "prompt": qu, "img_path": path},
                    use_nucleus_sampling=args.sample,
                    num_beams=num_beams,
                    max_new_tokens=1,
                    output_hidden_states=True, 
                    output_attentions=True,
                    return_dict_in_generate=True,
                    nvcd=True,
                    nvcd_previous_last_ids_list=output_tokens, 
                    past_key_values=kv_cache,
                )   
                last_logit = output['hidden_states'][-1][-1][:, -1, :]
                
                ##############################################################

                if model_name == 'mplug-owl2':
                    last_logit = last_logit.clone()
                    lm_head_matrix = lm_head_matrix.clone()
                last_logit = torch.matmul(last_logit, lm_head_matrix.T)

                if path == original_img_path : 
                    original_logit = last_logit
                    original_mature_logit = original_logit.clone()
                else: 
                    negative_logits.append(last_logit)

                if args.kv_cache_faster:
                    image_kv_cache[path] = output['past_key_values']

            
            # Positive image P 안의 모든 이미지에 대해
            ##############################################################
            for path in positive_img_path:
                image = process_before_norm(path) #P 이미지들.
                kv_cache = image_kv_cache.get(path, None)

                ##############################################################
                output = model.generate(
                    {"image": norm(image), "prompt": qu, "img_path": path},
                    use_nucleus_sampling=args.sample,
                    num_beams=num_beams,
                    max_new_tokens=1,
                    output_hidden_states=True, 
                    output_attentions=True,
                    return_dict_in_generate=True,
                    nvcd=True,
                    nvcd_previous_last_ids_list=output_tokens, 
                    past_key_values=kv_cache,

                )   
                last_logit = output['hidden_states'][-1][-1][:, -1, :]
                ##############################################################
                if model_name == 'mplug-owl2':
                    last_logit = last_logit.clone()
                    lm_head_matrix = lm_head_matrix.clone()
                last_logit = torch.matmul(last_logit, lm_head_matrix.T)
                if path == original_img_path : 
                    original_logit = last_logit
                    original_mature_logit = original_logit.clone()
                else: 
                    positive_logits.append(last_logit)
                
                if args.kv_cache_faster:
                    image_kv_cache[path] = output['past_key_values']

            print('-'*50)
            print(f'image_count : {idx}')
            print(f"negative_logits count : {len(negative_logits)}")
            print(f"positive_logits count : {len(positive_logits)}")
            print(f"hal_detected_synonym: {hal_detected}")
            print(f"gt_detected_synonym: {gt_detected}")

            alpha = args.rvcd_alpha
            beta = args.rvcd_beta
            gamma = args.rvcd_gamma # 0, 0.00000001?
            
            print(f'alpha, beta, gamma : {alpha, beta, gamma}')
            
            negative_logits_count = len(negative_logits)
            positive_logits_count = len(positive_logits)

            sum_negative_logits = sum(negative_logits)
            sum_positive_logits = sum(positive_logits)
        
            adjusted_logits = (1 + (alpha * negative_logits_count) - (beta * positive_logits_count)) \
                * original_logit - (alpha * sum_negative_logits - beta * sum_positive_logits)

            original_probabilities = F.softmax(original_mature_logit, dim=-1)
            probabilities = F.softmax(adjusted_logits, dim=-1)

            # 선행 연구들의 아이디어 : 원본 로짓의 최대확률 * gamma보다 낮은 확률을 갖는 토큰은 못나오게 규제
            # 이 연구에서는 큰 효과가 없었음.. 추가적인 하이퍼파라미터 도입을 배제하기 위해 제거. 
            # abnormal_threshold = gamma * torch.max(original_probabilities)
            # low_prob_indices = torch.where(original_probabilities < abnormal_threshold)[0]
            # probabilities[low_prob_indices] = 0

            max_index = torch.argmax(probabilities, dim=-1)

            output_first_token_index = max_index
            output_first_token_name = model_tokenizer.convert_ids_to_tokens([output_first_token_index], skip_special_tokens=False)[-1]
            print(f'output token, index : {output_first_token_name}, {output_index}')

            output_tokens.append(output_first_token_index.squeeze(0))

            if output_first_token_index == model_tokenizer.eos_token_id :
                break
        
        token_count = len(output_tokens)
        nnvcd_caption_nl = model_tokenizer.decode(output_tokens, skip_special_tokens=True)
        
        if model_name == 'minigpt4':
            nnvcd_caption_nl = nnvcd_caption_nl.split('###')[0].split('Assistant:')[-1].strip()
        else:
            nnvcd_caption_nl = nnvcd_caption_nl.split('ASSISTANT: ')[-1]

        print('-'*30)
        print(f"draft_caption : \n{draft_output_text}")
        # print(f"coco first objects : {global_chair_evaluator.process_sentence_get_coco_objects(draft_output_text)}")
        print('-'*30)
        print(f"nnvcd_caption_nl : \n{nnvcd_caption_nl}")
        # print(f"coco first objects : {global_chair_evaluator.process_sentence_get_coco_objects(nnvcd_caption_nl)}")
        print('-'*30)
        # print(f'ablation_rvcd_all, ablation_rvcd_gt, ablation_rvcd_hal')
        # print(f'{ablation_rvcd_all, ablation_rvcd_gt, ablation_rvcd_hal}')
        print(f"hal_detected_synonym: {hal_detected}")
        print(f"gt_detected_synonym: {gt_detected}")
        
        now_datapoint_draft_caption = draft_output_text
        now_datapoint_final_caption = nnvcd_caption_nl

    else:
        print(f'detector가 negative object를 정의하지 않고 있습니다. rvcd할 수 없는 데이터포인트입니다. draft캡션을 출력합니다.')
        print(f"draft_caption : \n{draft_output_text}")

        now_datapoint_draft_caption = draft_output_text
        now_datapoint_final_caption = draft_output_text

    # 아래 두개가 ouput 캡션들.
    # now_datapoint_draft_caption
    # now_datapoint_final_caption

    now_draft_result = {"image_id": int(img_id),"caption": now_datapoint_draft_caption}
    draft_captions_path = os.path.join(result_dir,f"rvcd_{model_name}_{formatted_time}_DRAFT_generated_captions.jsonl")
    with open(draft_captions_path, "a") as f:
        json.dump(now_draft_result, f)
        f.write("\n")

    now_nvcd_result = {"image_id": int(img_id),"caption": now_datapoint_final_caption,"tokens": token_count}
    global_all_info['total_generated_tokens'] += token_count
    nvcd_captions_path = os.path.join(result_dir,f'rvcd_{model_name}_a{args.rvcd_alpha}_b{args.rvcd_beta}_{formatted_time}_seed_{seed}_samples_{num_samples}_maxtokens_{max_new_tokens}_{true_flag_name}_generated_captions.jsonl')
    with open(nvcd_captions_path, "a") as f:
        json.dump(now_nvcd_result, f)
        f.write("\n")

if check_draft_chair:
    total_detector_score = calculate_metrics(global_all_info['chair1_detect1'], 
                                            global_all_info['chair1_detect0'], 
                                            global_all_info['chair0_detect1'],
                                            global_all_info['chair0_detect0'])
    global_all_info['total_detector_score'].append(total_detector_score) 
    global_all_info['latency'] = time.time()-start_time
    global_all_info['latency_per_token'] = global_all_info['latency'] / global_all_info['total_generated_tokens']


global_info_save_path = os.path.join(result_dir,f"rvcd_{model_name}_a{args.rvcd_alpha}_b{args.rvcd_beta}_{formatted_time}_seed_{seed}_samples_{num_samples}_maxtokens_{max_new_tokens}_{true_flag_name}_DETECTOR_info.json")
with open(global_info_save_path, 'w', encoding='utf-8') as json_file:
    json.dump(global_all_info, json_file, indent=4, ensure_ascii=False)


# CUDA_VISIBLE_DEVICES=0 \
# python rvcd_generation_chair_bleu.py \
# --model llava-1.5 \
# --data_path /home/donut2024/coco2014 \
# --num_samples 500 \
# --seed 42 \
# --gpu-id 0 \
# --output_dir ./generated_captions/  

