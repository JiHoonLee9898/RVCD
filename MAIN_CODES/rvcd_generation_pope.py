
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


import json
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image
from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from types import SimpleNamespace
# from decoder_zoo.Woodpecker.vis_corrector import Corrector
# from decoder_zoo.Woodpecker.config import woodpecker_args_dict
# from decoder_zoo.HALC.context_density.halc import halc_assistant
# from decoder_zoo.VCD.vcd_utils.vcd_add_noise import add_diffusion_noise
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict
from eval.pope_metrics.utils import generate_ground_truth_objects, pope




MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    # "instructblip": "eval_configs/instructblip_eval.yaml",
    # "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    # "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "mplug-owl2": "eval_configs/mplug-owl2_eval.yaml",
}

POPE_PATH = {
    "random": "pope_coco/coco_pope_random.json",
    "popular": "pope_coco/coco_pope_popular.json",
    "adversarial": "pope_coco/coco_pope_adversarial.json",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    # "instructblip": "<ImageHere><question>",
    # "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    # "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "mplug-owl2": "USER: <|image|><question> ASSISTANT:",

}


def parse_args():
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--pope_type", type=str, help="model")
    # parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
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
        default="coco",
        help="Name of the dataset. Default is 'coco'.",
    )
    parser.add_argument("--data_path",type=str,default="/home/donut2024/coco2014",help="data path",)
    parser.add_argument("--chair_cache_path",type=str,default="/home/donut2024/JIHOON/RVCD/MAIN_CODES/eval/CHAIR_CACHE/chair.pkl",help="chair_pickle_path",)
    
    parser.add_argument("--scale_factor", type=float, default=50)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num_attn_candidates", type=int, default=5)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-m", "--max_new_tokens", type=int, default=16)
    parser.add_argument(
        "-v",
        "--verbosity",
        action="store_false",
        dest="verbosity",
        default=True,
        help="Verbosity. Default: True.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./paper_result/",
        help="Output ditectory for saving test results. Default is './generated_chair_inputs/'.",
    )
  
    parser.add_argument(
        "--gt_seg_path",
        type=str,
        default="pope_coco/coco_ground_truth_segmentation.json",
        help="Input json file that contains ground truth objects in the image.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of images to build POPE questions. Default is 500.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of positive/negative objects to be sampled. Default is 3.",
    )
    parser.add_argument(
        "--question_template",
        type=str,
        # default="Is there a {} in the image? ",
        # default="Is there a XXX in the image? There is no XXX in the image, so the answer is No. Is there a YYY in the image? There is 2 YYY in the image, so the answer is Yes. Is there a {} in the image? ",
        default="Find evidence first and then answer: is there a {} in the image?",
        # default="Is there a {} in the image?",  # for llava-1.5
        help="Prompt template. Default is 'Is there a {} in the image?'.",
    )
    parser.add_argument("--yolo_version",type=str,default="yolov8x.pt",help="yolo")
    parser.add_argument("--rvcd_alpha", type=float, default=1, help='') 
    parser.add_argument("--rvcd_beta", type=float, default=0.1, help='') 
    parser.add_argument("--rvcd_gamma", type=float, default=0, help='') 

    args = parser.parse_args()
    return args



#############################################3
def setup_seeds(config, seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def print_acc(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print("TP\tFP\tTN\tFN\t")
    print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print("Accuracy: {}".format(acc))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 score: {}".format(f1))
    print("Yes ratio: {}".format(yes_ratio))

    return acc, precision, recall, f1


def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:
        line = line.replace(".", "")
        line = line.replace(",", "")
        words = line.split(" ")
        if any(word in NEG_WORDS for word in words) or any(
            word.endswith("n't") for word in words
        ):
            pred_list.append(0)
        else:
            pred_list.append(1)

    return pred_list



#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
global_chair_evaluator = None

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    args.pope_path = POPE_PATH[args.pope_type]
    cfg = Config(args)

    decoding_strategy = 'rvcd'
    seed = args.seed
    setup_seeds(cfg, seed)
    pope_type = args.pope_type
    device = (
        torch.device(f"cuda:{int(args.gpu_id)}") if torch.cuda.is_available() else "cpu"
    )
    yolo_version = args.yolo_version
    model_name = args.model
    num_samples = args.num_samples
    num_images = args.num_images
    dataset_name = args.dataset_name
    data_path = args.data_path
    chair_cache_path = args.chair_cache_path
    output_dir = args.output_dir
    num_beams = 1
    batch_size = 1
    max_new_tokens = args.max_new_tokens
    gt_seg_path = args.gt_seg_path
    question_template = args.question_template
    verbosity = args.verbosity
    # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

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

    def chair_change_synonym_to_cocofirst_word(word):
        evaluator = get_chair_evaluator(chair_cache_path, coco_path)
        words, node_words, _, double_words = evaluator.caption_to_words(word)
        print(words, node_words, double_words)
        if len(node_words) == 1:
            return node_words[0]
        else: # 정상처리되지 않은 단어. 나중에 chair 파일에 추가하고
            # 캐시를 새로 초기화해야함. 잇던 캐시 지우기
            return 'chair_add_'+' '.join(double_words)
        
    #########################

    # ========================================
    #             Model Initialization
    # ========================================
    print("Initializing Model")
    
    model_config = cfg.model_cfg
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

    vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
        vis_processor_cfg
    )
    # vis_processors.do_normalize = False
    print(vis_processors["eval"].transform)

    decoding_strategy = 'nvcd'
    print("decoding_strategy", decoding_strategy)
 
    if verbosity:
        print("\ndecoding strategy: ", decoding_strategy)
        print("backbone model_name: ", args.model)
        print("data_path: ", data_path)
        print("output_dir: ", output_dir)
        print("num_samples: ", num_samples)
        print("num_images: ", num_images)
        print("num_beams: ", num_beams)
        print("seed: ", seed)
        print(vis_processors["eval"].transform)

    print("Done!")

    if verbosity:
        print(f"\nGenerating {pope_type} POPE questions")

    # generate pope questions
    question_dir = os.path.join(output_dir, "pope")
    if not os.path.exists(question_dir):
        os.makedirs(question_dir)
    question_path = os.path.join(
        question_dir,
        f"_num_images_{num_images}_num_samples_{num_samples}_pope_{pope_type}_questions.json",
    )
    # load ground truth segmentation results.
    # Must include (other keys such as image_id can exist):
    # {"image": "COCO_val2014_000000131089.jpg", "objects": ["person", "baseball bat"]}
    segment_results = [json.loads(q) for q in open(gt_seg_path, "r")]
    if verbosity:
        print(
            f"\nGround truth segmentation results loaded successfully, contains {len(segment_results)} classes."
        )

    # process segmentation ground truth
    processed_segment_results = []
    # Sample images which contain more than sample_num objects
    for cur_image in segment_results:
        if len(cur_image["objects"]) >= num_samples:
            processed_segment_results.append(cur_image)

    assert (
        len(processed_segment_results) >= num_images
    ), f"The number of images that contain more than {num_samples} objects is less than {num_images}."

    # Randomly sample num_images images
    processed_segment_results = random.sample(processed_segment_results, num_images)

    # Organize the ground truth objects and their co-occurring frequency
    question_name = f"_num_images_{num_images}_num_samples_{num_samples}"
    # ground truth object summary
    ground_truth_objects = generate_ground_truth_objects(
        processed_segment_results,
        question_dir,
        question_name,
        verbosity,
    )

    # Generate POPE questions and save to local file
    if pope_type is None:
        for cur_type in ["random", "popular", "adversarial"]:
            pope(
                ground_truth_objects=ground_truth_objects,
                segment_results=processed_segment_results,
                num_samples=num_samples,
                template=question_template,
                neg_strategy=cur_type,
                output_dir=question_dir,
                dataset_name=question_name,
                verbosity=verbosity,
            )
    else:
        pope(
            ground_truth_objects=ground_truth_objects,
            segment_results=processed_segment_results,
            num_samples=num_samples,
            template=question_template,
            neg_strategy=pope_type,
            output_dir=question_dir,
            dataset_name=question_name,
            verbosity=verbosity,
        )

    # load all the POPE questions
    all_pope_questions = [json.loads(q) for q in open(question_path, "r")]
    if verbosity:
        print(
            f"\nLoaded {len(all_pope_questions)} POPE questions from {question_path}."
        )
    # sanity check
    if len(all_pope_questions) != num_images * num_samples * 2:
        raise ValueError(
            f"Number of POPE questions loaded from {question_path} is not equal to {num_images * num_samples * 2}."
        )

    # print("all_pope_questions", all_pope_questions)
    # save all the POPE questions to local file
    # if not os.path.exists(question_dir):
    #     os.makedirs(pope_question_dir)
    # pope_question_path = os.path.join(
    #     pope_question_dir,
    #     f"_num_images_{num_images}_num_samples_{num_samples}_pope_{pope_type}_questions.json",
    # )
    # input()

    # load pope data
    pope_dataset = POPEDataSet(
        pope_path=question_path, data_path=args.data_path, trans=vis_processors["eval"]
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=args.num_workers,
        drop_last=False,
    )

    print("load data finished")

    base_dir = os.path.join(output_dir, "pope", args.model)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

   
    print("Start eval...")
    pred_list, pred_list_s, label_list = [], [], []
    
    from datetime import datetime
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M")
    
################################################################
################################################################
################################################################
    
    ###### POPE seed check #######
    seed_valid_check = []
    for data in pope_loader:
        image_path = data["image_path"]
        image_id = int(image_path[0].split("/")[-1].split(".")[0].split("_")[-1].lstrip("0"))
        seed_valid_check.append(image_id)
    seed_valid_check = sorted(seed_valid_check)
    print(f'시드 : {seed} / 샘플링된 이미지들 : {seed_valid_check[:20]}')
    import time
    time.sleep(5)
    ##############################
    
    check_draft_chair = True
    idx = 0
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        idx += 1
        image = data["image"]
        qu = data["query"]
        label = data["label"]
        image_path = data["image_path"] #여기서 리스트에 담긴 형태
        print(f'pope image_path : {image_path}')

        if model_name == "mplug-owl2":
            image_path = image_path[0]
        else:
            image_path = image_path[0]

        image = process_before_norm(image_path)
        image_id = image_path[0].split("/")[-1].split(".")[0].split("_")[-1].lstrip("0")
        label_list = label_list + list(label)

        template = INSTRUCTION_TEMPLATE[args.model]
        qu = [template.replace("<question>", q) for q in qu][0]

        # image = image.to(device)
        label = torch.Tensor(label).to(device)

        image_cd = None

        #################################
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
            #yolo가 detect한 entity명이 1단어로 변경불가하면 에러 일으킴
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
        draft_synonyms = global_chair_evaluator.process_sentence_get_coco_synonyms(qu) ###!!! pope에서는 질문의 객체를 rvcd
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
                    # global_all_info['ref_not_exist'].append(hall_ref_list[i]) # 맨 위에 정의한 글로벌 리스트에 없는거 기록
                    hall_ref_list.pop(i)  # 존재하지 않는 경로 제거
                    

        gt_ref_list = []
        if len(gt_detected) > 0: #gt가 감지되었다면
            ref_folder = args.ref_folder_path
            gt_ref_list = [os.path.join(ref_folder,f'{synonym}.png') for synonym in gt_detected]
            for i in range(len(gt_ref_list) - 1, -1, -1):  # 역순 순회
                if not os.path.exists(gt_ref_list[i]):
                    # global_all_info['ref_not_exist'].append(gt_ref_list[i]) # 맨 위에 정의한 글로벌 리스트에 없는거 기록
                    gt_ref_list.pop(i)  # 존재하지 않는 경로 제거
                    
        # 결과는 hal이 하나라도 존재할때만 하는게 나았음
        if len(hall_ref_list) > 0:
            nvcd_operate = True
        else: 
            nvcd_operate = False


        print(f'hall_ref_list : {hall_ref_list}')
        print(f'gt_ref_list : {gt_ref_list}')
   

        now_datapoint_draft_caption = None
        now_datapoint_final_caption = None
        ################################################
        # rvcd
        ################################################
        if nvcd_operate:
            
            output_tokens = []

            # 모델의 vocab head. 
            if model_name == 'mplug-owl2': lm_head_matrix = model.model.lm_head.weight
            else: lm_head_matrix = model.llama_model.lm_head.weight

            for output_index in range(max_new_tokens):
            
                original_img_path = image_path
                negative_img_path = hall_ref_list
                positive_img_path = gt_ref_list

                original_logit = None
                negative_logits = []
                positive_logits = []


                if len(output_tokens) == 0: #최초토큰생성
                    nvcd = False
                else:
                    nvcd = True

                for path in [original_img_path]+negative_img_path:
                    image = process_before_norm(path) #원본 이미지와 ref이미지들.
                    output = model.generate(
                            {"image": image, "prompt":qu, "img_path": path},
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
                            {"image": image, "prompt":qu, "img_path": path},
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

##############################################################


        #디코더의 최종출력캡션 = out
        out = [now_datapoint_final_caption]
        pred_list = recorder(out, pred_list)
        for line in out:
            print(line)

        output_text = out[0]
        cur_generated_answer = {
            "image_id": image_id,
            "question": " ".join(qu[0].split(" ")[2:]).split("?")[0] + "?",
            "answer": output_text,
        }

        

        # dump metric file
        generated_captions_path = os.path.join(
            base_dir,
            f"a{args.rvcd_alpha}_b{args.rvcd_beta}_{pope_type}_{formatted_time}_{model_name}_{decoding_strategy}_seed_{seed}_max_tokens_{max_new_tokens}_samples_{num_images}_generated_captions.json",
        )
        with open(generated_captions_path, "a") as f:
            json.dump(cur_generated_answer, f)
            f.write("\n")

    print(
        "[{}, {}]===============================================".format(
            args.scale_factor, args.num_attn_candidates
        )
    )
    if len(pred_list) != 0:
        acc, precision, recall, f1 = print_acc(pred_list, label_list)
    if len(pred_list_s) != 0:
        acc, precision, recall, f1 = print_acc(pred_list_s, label_list)

    result = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }


    metrics_path = os.path.join(
        base_dir,
        f"a{args.rvcd_alpha}_b{args.rvcd_beta}_{pope_type}_{formatted_time}_{model_name}_{decoding_strategy}_seed_{seed}_max_tokens_{max_new_tokens}_samples_{num_images}_results.json",
    )
    with open(metrics_path, "w") as f:
        json.dump(result, f)
        f.write("\n")


if __name__ == "__main__":
    main()
