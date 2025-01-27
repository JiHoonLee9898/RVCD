
import os
import argparse
import random
import sys
sys.path.append("MAIN_CODES/mPLUG-Owl/mPLUG-Owl2")
sys.path.append("mPLUG-Owl/mPLUG-Owl2")
sys.path.append("./")
sys.path.append("../")
sys.path.append("./eval")
sys.path.append("./YOLO")
import yolo
import numpy as np
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
from types import SimpleNamespace
# from decoder_zoo.Woodpecker.vis_corrector import Corrector
# from decoder_zoo.Woodpecker.config import woodpecker_args_dict
# from decoder_zoo.HALC.context_density.halc import halc_assistant
# from decoder_zoo.VCD.vcd_utils.vcd_add_noise import add_diffusion_noise
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict
import torch
from PIL import Image
from transformers import TextStreamer
import torch.nn.functional as F
import numpy as np
import pickle
from chair import CHAIR 
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


parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, default="llava-1.5", help="model")
parser.add_argument(
    "-d",
    "--decoder",
    type=str,
    default="rvcd",
    help="Decoding strategy",
)
parser.add_argument(
    "-g", "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
)
parser.add_argument(
        "--ref_folder_path",
        type=str,
        default="/home/donut2024/JIHOON/RVCD/DB_single_concept_images_flux_generated/generated_images",
        help="ref_folder_path",
    )
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="mme",
    help="Name of the dataset. Default is 'mme'.",
)
parser.add_argument(
    "--data_paths",
    type=str,
    default="/home/donut2024/JIHOON/MME_Benchmark_release_version/MME_Benchmark",
    help="data path",
)
parser.add_argument("--coco_path",type=str,default="/home/donut2024/coco2014",help="coco path",)
parser.add_argument("--chair_cache_path",type=str,default="/home/donut2024/JIHOON/RVCD/MAIN_CODES/eval/CHAIR_CACHE/chair.pkl",help="chair_pickle_path",)

parser.add_argument("--sample", action="store_true")

parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("-n", "--num_samples", type=int, default=120)
parser.add_argument("-m", "--max_new_tokens", type=int, default=128)
parser.add_argument("-v", "--verbosity", action="store_false", dest="verbosity", default=True, help="Verbosity. Default: True.",)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./generated_chair_inputs/",
    help="Output ditectory for saving test results. Default is './generated_chair_inputs/'.",
)

parser.add_argument(
    "--gt_seg_path",
    type=str,
    default="pope_coco/coco_ground_truth_segmentation.json",
    help="Input json file that contains ground truth objects in the image.",
)
parser.add_argument(
    "--generate_pope",
    action="store_true",
    default=False,
    help="Whether to generate POPE questions.",
)
parser.add_argument("--skip_num", type=int, default=0, help="Skip the first skip_num samples.")
parser.add_argument("--yolo_version",type=str,default="yolov8x.pt",help="yolo")
parser.add_argument("--rvcd_alpha", type=float, default=1, help='') 
parser.add_argument("--rvcd_beta", type=float, default=0.1, help='') 
parser.add_argument("--rvcd_gamma", type=float, default=0, help='') 


args = parser.parse_known_args()[0]

# print("args.gpu_id", args.gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)

model_name = args.model
decoding_strategy = 'rvcd'

seed = args.seed
setup_seeds(cfg, seed)

device = (
    torch.device(f"cuda:{int(args.gpu_id)}") if torch.cuda.is_available() else "cpu"
)
# device = "cpu"
yolo_version = args.yolo_version
verbosity = args.verbosity
num_samples = args.num_samples
dataset_name = args.dataset_name
data_paths = args.data_paths
coco_path = args.coco_path
chair_cache_path = args.chair_cache_path
output_dir = args.output_dir
num_beams = 1
batch_size = 1
max_new_tokens = args.max_new_tokens
gt_seg_path = args.gt_seg_path
generate_pope = args.generate_pope
skip_num = args.skip_num

##########################################################

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
coco_ann_path = os.path.join(args.coco_path, 'annotations')
def get_chair_evaluator(chair_cache_path=chair_cache_path, coco_path=coco_ann_path):
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

def chair_change_synonym_to_cocofirst_word(word):
    evaluator = get_chair_evaluator(chair_cache_path, coco_path)
    words, node_words, _, double_words = evaluator.caption_to_words(word)
    print(words, node_words, double_words)
    if len(node_words) == 1:
        return node_words[0]
    else: # 정상처리되지 않은 단어. 나중에 chair 파일에 추가하고
        # 캐시를 새로 초기화해야함. 잇던 캐시 지우기
        return 'chair_add_'+' '.join(double_words)
    
###########################################

from datetime import datetime
current_time = datetime.now()
formatted_time = current_time.strftime("%Y%m%d%H%M")

# ========================================
#             Model Initialization
# ========================================
print("Initializing Model")

# print("cfg", cfg)
# input()
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()
print("model device", model.device)

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

if verbosity:
    print("\ndecoding strategy: ", decoding_strategy)
    print("backbone model_name: ", args.model)
    print("dataset_name: ", dataset_name)
    print("data_paths: ", data_paths)
    print("output_dir: ", output_dir)
    print("num_samples: ", num_samples)
    print("num_beams: ", num_beams)
    print("seed: ", seed)
    print(vis_processors["eval"].transform)


data_paths_folders = []

for folder in os.listdir(data_paths):
    full_path = os.path.join(data_paths, folder)
    if os.path.isdir(full_path):  # 폴더인지 확인
        for mme_type in ["existence", 'count', 'position', 'color']:
            if mme_type in full_path:
                data_paths_folders.append(full_path + '/')

###### MME seed check #######
# MME는 타겟 데이터포인트 수가 고정되어 있음
# sampling 하지 않음.
print(f'시드 : {seed}')
import time
time.sleep(5)
##############################
for data_path in data_paths_folders:

    img_files = []

    # read in all the images in a folder
    for file in os.listdir(data_path):
        if file.endswith(".jpg"):
            img_files.append(file)

    print("img_files", len(img_files))

    base_dir = os.path.join(output_dir, "mme", args.model)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


    offlight = True

    iterations = 2*len(img_files)

    result_txt = []


    ################################################################
    ################################################################
    ################################################################

    check_draft_chair = True
    for idx in tqdm(range(iterations)):
        new_line = ""
        img_file = img_files[int(idx/2)]
        # if idx <= 23:
        #     continue
        # img_file = img_files[img_id]
        new_line += img_file + "\t"
        print("img_file", img_file)
        txt_file = img_file.replace(".jpg", ".txt")
        # get the first line of the txt file
        if idx % 2 == 0:
            with open(data_path + txt_file, "r") as f:
                qu = f.readlines()[0]
                # token_num = len(qu.split(" "))
                # print("qu.split(" ")", qu.split(" "))
                # input()
                # qu = " ".join(qu.split(" ")[:-1])
                if "Yes" in qu:
                    gt = "Yes"
                else:
                    gt = "No"
                qu = qu.replace("Yes", "")
                qu = qu.replace("No", "")

            print("idx % 2 == 0", qu)
        else:
            # get the second line of the txt file
            with open(data_path + txt_file, "r") as f:
                qu = f.readlines()[1]
                # token_num = len(qu.split(" "))
                # qu = " ".join(qu.split(" ")[:-1])
                if "Yes" in qu:
                    gt = "Yes"
                else:
                    gt = "No"
                qu = qu.replace("Yes", "")
                qu = qu.replace("No", "")
                # gt = qu.split(" ")[-1]
            print("idx % 2 == 1", qu)

        # qu = str(qu)

        new_line += qu + "\t" + gt + "\t"

        img_id = int(img_file.split(".jpg")[0][-6:])

        img_save = {}
        img_save["image_id"] = img_id

        image_path = data_path + img_file
        raw_image = Image.open(image_path).convert('RGB')

        if model_name == "mplug-owl2":
            max_edge = max(raw_image.size) # We recommand you to resize to squared image for BEST performance.
            image = raw_image.resize((max_edge, max_edge))
            image_tensor = process_images([image], model.image_processor)
            image = image_tensor.to(device, dtype=torch.float16)
        else:
            image = vis_processors["eval"](raw_image).unsqueeze(0)
            image = image.to(device)

        # print("image device", norm(image).device)

        # qu = "Please describe this image in detail."
        # # qu = "Please provide a very detailed description of the image."
        # # qu = "Please provide a very long and detailed description of the image."
        # # qu = "Generate a one sentence caption of the image."
        # # qu = "Generate a short caption of the image."

        #################33
        # Is there a train in this image? Please answer yes or no.	
        original_qu = qu[:]
        print(f"MME의 원본 qu : {original_qu}")
        qu = f"Describe this image and then answer: {original_qu.split(' Please')[0]}"
        print(f"MME의 qu : {qu}")
        #################

        template = INSTRUCTION_TEMPLATE[args.model]
        qu = template.replace("<question>", qu)

        ####################################################################
        print(f'qu : {qu}')
        with torch.inference_mode():
            with torch.no_grad():
                out = model.generate(
                    {"image": image, "prompt": qu},
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
        print(f'out.keys() : {out.keys()}')
        attentions = out['attentions']
        print(len(attentions))
        print(len(attentions[0]))
        print(attentions[0][0].shape)

        all_nl_tokens = [model_tokenizer.convert_ids_to_tokens(seq) for seq in out["sequences"].tolist()][0]
        #minigpt4는 그냥 input_nl_tokens에도 output_nl_tokens들어있음
        #MAIN_CODES/minigpt4/models/mini_gpt4.py 참고
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

        ############
        input_image_path = image_path
        yolo_detected_entity_prob = yolo.main(input_image_path, yolo_version)
        yolo_detected_entity_list = []
        for entity,prob in yolo_detected_entity_prob:
            yolo_detected_entity_list.append(entity)

        for i in range(len(yolo_detected_entity_list)):
            cocofirst_or_notyetword = chair_change_synonym_to_cocofirst_word(yolo_detected_entity_list[i])
            if cocofirst_or_notyetword.startswith("chair_add_"):
                pass
                #이럼 아직 처리불가단어니까, 그냥 있는그대로씀 yolo_detected_entity_list[i] 그대로 둠둠
                #나중에 MAIN_CODES/eval/chair.py 에서 CHAIR(object)__init__에서 추가해야함
                #추가하고 캐시 지우기 MAIN_CODES/eval/CHAIR_CACHE 이안에 있는거 지우기.
                # global_all_info['chair_not_yet_doublewords'].append(cocofirst_or_notyetword.split("chair_add_")[-1])
            else: #정상변환됐으면 대표어로 변환 
                yolo_detected_entity_list[i] = cocofirst_or_notyetword


        detected_info = {}
        draft_synonyms = global_chair_evaluator.process_sentence_get_coco_synonyms(draft_output_text) 
        
        for synonym in draft_synonyms:
            # synonym = (cocofirstword, synonymfromdraft) 
            if synonym[0] in yolo_detected_entity_list: detected_info[synonym] = 1
            else: detected_info[synonym] = 0

        print(f'detected_info : {detected_info}') 

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
        if len(hal_detected) > 0: #draft 캡션에서 지워야하는 객체가 있다면
            ref_folder = args.ref_folder_path
            hall_ref_list = [os.path.join(ref_folder,f'{synonym}.png') for synonym in hal_detected]
            for i in range(len(hall_ref_list) - 1, -1, -1):  # 역순 순회
                if not os.path.exists(hall_ref_list[i]):
                    hall_ref_list.pop(i)  # 존재하지 않는 경로 제거
                    # global_all_info['ref_not_exist'].append(hall_ref_list[i]) # 맨 위에 정의한 글로벌 리스트에 없는거 기록

        gt_ref_list = []
        if len(gt_detected) > 0: #gt가 감지되었다면
            ref_folder = args.ref_folder_path
            gt_ref_list = [os.path.join(ref_folder,f'{synonym}.png') for synonym in gt_detected]
            for i in range(len(gt_ref_list) - 1, -1, -1):  # 역순 순회
                if not os.path.exists(gt_ref_list[i]):
                    gt_ref_list.pop(i)  # 존재하지 않는 경로 제거
                    # global_all_info['ref_not_exist'].append(gt_ref_list[i]) # 맨 위에 정의한 글로벌 리스트에 없는거 기록

        if len(hall_ref_list) > 0:
            nvcd_operate = True
        else: 
            nvcd_operate = False


        print(f'hall_ref_list : {hall_ref_list}')
        print(f'gt_ref_list : {gt_ref_list}')


        now_datapoint_draft_caption = None
        now_datapoint_final_caption = None
        ################################################
        # nvcd
        ################################################
        if nvcd_operate:

            output_tokens = []

            # 모델의 vocab head. 
            # mplug_owl이면 model.llama_model.lm_head.weight 안됨
            if model_name == 'mplug-owl2': lm_head_matrix = model.model.lm_head.weight
            else: lm_head_matrix = model.llama_model.lm_head.weight
            token_count = 0
            for output_index in range(max_new_tokens):
                token_count += 1
                original_img_path = image_path
                negative_img_path = hall_ref_list
                positive_img_path = gt_ref_list

                original_logit = None
                negative_logits = []
                positive_logits = []


                if len(output_tokens) == 0: #최초토큰생성성
                    nvcd = False
                else:
                    nvcd = True

                final_qu = f"Describe this image and then answer: {original_qu.split(' Please')[0]}"
                template = INSTRUCTION_TEMPLATE[args.model]
                final_qu = template.replace("<question>", final_qu)


                print(f'final_qu : {final_qu}')


                for path in [original_img_path]+negative_img_path:
                    image = process_before_norm(path) #원본 이미지와 ref이미지들.
                    output = model.generate(
                            {"image": image, "prompt":final_qu, "img_path": path},
                            use_nucleus_sampling=args.sample,
                            num_beams=num_beams,
                            max_new_tokens=1,
                            output_hidden_states=True, 
                            output_attentions=True,
                            return_dict_in_generate=True,
                            
                            nvcd=nvcd,
                            nvcd_previous_last_ids_list=output_tokens, 
                        )   
                    hidden_states = output['hidden_states']
                    last_logit = hidden_states[-1][-1][:, -1, :] 
                    if model_name == 'mplug-owl2':
                        last_logit = last_logit.clone()
                        lm_head_matrix = lm_head_matrix.clone()
                    last_logit = torch.matmul(last_logit, lm_head_matrix.T)
                    if path == original_img_path : 
                        original_logit = last_logit
                        original_mature_logit = original_logit.clone()
                    else: 
                        negative_logits.append(last_logit)
                ##############################################################
                for path in positive_img_path:
                    image = process_before_norm(path) #ref이미지들.
                    output = model.generate(
                            {"image": image, "prompt":final_qu, "img_path": path},
                            use_nucleus_sampling=args.sample,
                            num_beams=num_beams,
                            max_new_tokens=1,
                            output_hidden_states=True, 
                            output_attentions=True,
                            return_dict_in_generate=True,
                            
                            nvcd=nvcd,
                            nvcd_previous_last_ids_list=output_tokens, 
                        )   
                    hidden_states = output['hidden_states']
                    last_logit = hidden_states[-1][-1][:, -1, :] 
                    if model_name == 'mplug-owl2':
                        last_logit = last_logit.clone()
                        lm_head_matrix = lm_head_matrix.clone()
                    last_logit = torch.matmul(last_logit, lm_head_matrix.T)
                    if path == original_img_path : 
                        original_logit = last_logit
                        original_mature_logit = original_logit.clone()
                    else: 
                        positive_logits.append(last_logit)

                print('-'*50)
                print(f'image_count : {idx}')
                print(f"negative_logits count : {len(negative_logits)}")
                print(f"positive_logits count : {len(positive_logits)}")
                print(f"hal_detected_synonym: {hal_detected}")
                print(f"gt_detected_synonym: {gt_detected}")

                alpha = args.rvcd_alpha
                beta = args.rvcd_beta
                gamma = args.rvcd_gamma # 0, 0.00000001
                print(f'alpha, beta, gamma : {alpha, beta, gamma}')

                negative_logits_count = len(negative_logits)
                positive_logits_count = len(positive_logits)

                sum_negative_logits = sum(negative_logits)
                sum_positive_logits = sum(positive_logits)
                
                
                adjusted_logits = (1 + (alpha * negative_logits_count) - (beta * positive_logits_count)) \
                * original_logit - (alpha * sum_negative_logits - beta * sum_positive_logits)

                original_probabilities = F.softmax(original_mature_logit, dim=-1)
                probabilities = F.softmax(adjusted_logits, dim=-1)

                #원본 로짓의 최대확률 * beta보다 낮은 확률을 갖는 토큰은 못나오게 규제
                #이 연구에서는 0이 안정적
                abnormal_threshold = gamma * torch.max(original_probabilities)
                low_prob_indices = torch.where(original_probabilities < abnormal_threshold)[0]
                probabilities[low_prob_indices] = 0

                max_index = torch.argmax(probabilities, dim=-1)

                output_first_token_index = max_index
                output_first_token_name = model_tokenizer.convert_ids_to_tokens([output_first_token_index], skip_special_tokens=False)[-1]
                print(f'output token, index : {output_first_token_name}, {output_index}')

                output_tokens.append(output_first_token_index.squeeze(0))
                print(len(output_tokens))

                
                if output_first_token_index == model_tokenizer.eos_token_id :
                    break
               

            nnvcd_caption_nl = model_tokenizer.decode(output_tokens, skip_special_tokens=True)
            
            if model_name == 'minigpt4':
                # nnvcd_caption_nl = nnvcd_caption_nl.split('Assistant:')[-1].replace('###', '')
                nnvcd_caption_nl = nnvcd_caption_nl.split('###')[0].split('Assistant:')[-1].strip()
            else:
                nnvcd_caption_nl = nnvcd_caption_nl.split('ASSISTANT: ')[-1]

            print('-'*30)
            print(f"draft_caption : \n{draft_output_text}")
            print(f"coco first objects : {global_chair_evaluator.process_sentence_get_coco_objects(draft_output_text)}")
            print('-'*30)
            print(f"nnvcd_caption_nl : \n{nnvcd_caption_nl}")
            print(f"coco first objects : {global_chair_evaluator.process_sentence_get_coco_objects(nnvcd_caption_nl)}")
            print('-'*30)
            print(f"hal_detected_synonym: {hal_detected}")
            print(f"gt_detected_synonym: {gt_detected}")

            now_datapoint_draft_caption = draft_output_text
            now_datapoint_final_caption = nnvcd_caption_nl

        else:

            print(f'detector가 negative object를 정의하지 않고 있습니다. rvcd할 수 없는 데이터포인트입니다. draft캡션을 출력합니다.')
            print(f"draft_caption : \n{draft_output_text}")

            now_datapoint_draft_caption = draft_output_text
            now_datapoint_final_caption = draft_output_text


        ###########################################
        #문제마다 output_text 발생
        ###########################################

        output_text = now_datapoint_final_caption
        print("FINAL output text: ", output_text)

        ########################################
        NEG_WORDS = ["No", "not", "no", "NO"]
        output_text = output_text.replace(".", "").replace(",", "").replace("\n", " ").strip()
        words = output_text.split(" ")
        if any(word in NEG_WORDS for word in words) or any(
            word.endswith("n't") for word in words
        ):
            output_text = 'no'
            print(f'FINAL words : {words}\nOUTPUT_text : no')
        else:
            output_text = 'yes'
            print(f'FINAL words : {words}\nOUTPUT_text : yes')
        ########################################

        img_save["caption"] = output_text

        # print("img_id: ", img_id)
        print("image_path: ", image_path)
        print("caption: ", output_text)

        new_line += output_text
        # input()

        # dump metric file
        # if skip_num == 0:
        #     generated_captions_path = os.path.join(
        #         base_dir,
        #         f"{model_name}_{decoding_strategy}_{detector_type}_box_{box_threshold}_beams_{num_beams}_k_{k_candidate_num}_{dataset_name}_expand_ratio_{expand_ratio}_seed_{seed}_max_tokens_{max_new_tokens}_samples_{num_samples}_generated_captions.json",
        #     )
        # else:

        data_type = None
        types = ['existence', 'count', 'position', 'color']
        for type in types:
            if type in data_path:
                data_type = type


        result_dir = os.path.join(base_dir, formatted_time)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        generated_captions_path = os.path.join(
            result_dir,
            f"a{args.rvcd_alpha}_b{args.rvcd_beta}_{formatted_time}_{model_name}_{data_type}_{decoding_strategy}_seed_{seed}_max_tokens_{max_new_tokens}_generated_captions.json"
        )

        with open(generated_captions_path, "a") as f:
            json.dump(img_save, f)
            f.write("\n")
        
       