import argparse
import os
import random
import sys
import json
import pickle
import time
from datetime import datetime

sys.path.append("mPLUG-Owl/mPLUG-Owl2")
sys.path.append("./")
sys.path.append("../")
sys.path.append("./eval")
sys.path.append("./YOLO")

import yolo
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
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

from chair import CHAIR
from mplug_owl2.mm_utils import process_images


"""
promptban_generation_chair_bleu.py

기존 eRVCD/RVCD generation 코드에서 다음 부분만 남긴 prompt-level baseline 스크립트.

1. COCO image sampling
2. LVLM draft caption generation
3. draft caption에서 COCO object synonym 추출
4. YOLO detection 결과와 비교해서
   - gt_detected  : draft에 등장했고 YOLO도 이미지에서 찾은 object synonym
   - hal_detected : draft에 등장했지만 YOLO가 이미지에서 찾지 못한 object synonym
5. 위 정보를 prompt에 넣어 contrastive decoding 없이 일반 generate로 final caption 생성

promptban_mode:
- both          : 존재하는 객체 + 존재하지 않는 객체 둘 다 prompt에 제공
- positive_only : 존재하는 객체만 hint로 제공
- negative_only : 존재하지 않는 객체만 ban/avoid 대상으로 제공

주의:
- 여기서 "존재하지 않는 객체"는 전체 COCO class 중 absent class가 아니라,
  draft caption에 언급되었지만 YOLO가 탐지하지 못한 객체 후보를 의미한다.
- YOLO false negative가 있으면 실제 존재 객체도 hal_detected로 들어갈 수 있다.
"""


MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "mplug-owl2": "eval_configs/mplug-owl2_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "mplug-owl2": "USER: <|image|><question> ASSISTANT:",
}


def setup_seeds(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def unique_preserve_order(items):
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def format_object_list(items):
    items = unique_preserve_order([str(x).strip() for x in items if str(x).strip()])
    if len(items) == 0:
        return "none"
    return ", ".join(items)


def build_promptban_question(base_question, gt_detected, hal_detected, mode):
    """
    YOLO 기반 object hint/ban 정보를 base captioning question에 추가한다.

    Args:
        base_question (str): 기본 image captioning prompt.
        gt_detected (list[str]): draft object 중 YOLO가 있다고 판단한 synonym들.
        hal_detected (list[str]): draft object 중 YOLO가 없다고 판단한 synonym들.
        mode (str): both / positive_only / negative_only
    """
    present = format_object_list(gt_detected)
    absent = format_object_list(hal_detected)

    if mode == "both":
        return (
            f"{base_question}\n\n"
            "Object guidance from an external detector:\n"
            f"- Objects likely present in the image: {present}.\n"
            f"- Objects likely absent from the image: {absent}.\n\n"
            "Describe the image accurately and in detail. "
            "Use the likely-present objects only as visual hints. "
            "Do not mention the likely-absent objects unless they are clearly visible in the image."
        )

    if mode == "positive_only":
        return (
            f"{base_question}\n\n"
            "Object guidance from an external detector:\n"
            f"- Objects likely present in the image: {present}.\n\n"
            "Describe the image accurately and in detail. "
            "Use these objects only as visual hints, and do not force any object into the caption if it is not visible."
        )

    if mode == "negative_only":
        return (
            f"{base_question}\n\n"
            "Object guidance from an external detector:\n"
            f"- Objects likely absent from the image: {absent}.\n\n"
            "Describe the image accurately and in detail. "
            "Avoid mentioning the likely-absent objects unless they are clearly visible in the image."
        )

    raise ValueError(f"Unsupported promptban_mode: {mode}")


def clean_generated_text(decoded_text, model_name):
    if model_name == "minigpt4":
        return decoded_text.split("###")[0].split("Assistant:")[-1].strip()
    return decoded_text.split("ASSISTANT: ")[-1].strip()


def parse_args():
    parser = argparse.ArgumentParser(description="PromptBan CHAIR/BLEU generation on LVLMs.")

    parser.add_argument("--model", type=str, default="llava-1.5", choices=list(MODEL_EVAL_CONFIG_PATH.keys()), help="model")
    parser.add_argument("-g", "--gpu-id", type=int, default=0, help="specify the gpu to load the model")
    parser.add_argument("--dataset_name", type=str, default="coco", help="Name of the dataset. Default is 'coco'.")
    parser.add_argument("--data_path", type=str, default="/home/donut2024/coco2014", help="COCO data path")
    parser.add_argument("--sample", action="store_true", help="use nucleus sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-n", "--num_samples", type=int, default=500)
    parser.add_argument("-m", "--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_chair_inputs/",
        help="Output directory for saving test results.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help=(
            "override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead."
        ),
    )
    parser.add_argument(
        "--chair_cache_path",
        type=str,
        default="/home/donut2024/JIHOON/RVCD/MAIN_CODES/eval/CHAIR_CACHE/chair.pkl",
        help="chair pickle cache path",
    )

    # Detector / PromptBan options
    parser.add_argument("--yolo_version", type=str, default="yolov8x.pt", help="YOLO model path/name")
    parser.add_argument(
        "--check_draft_chair",
        type=str2bool,
        default=True,
        help="If True, evaluate draft objects with CHAIR and save detector score info.",
    )
    parser.add_argument(
        "--promptban_mode",
        type=str,
        default="both",
        choices=["both", "positive_only", "negative_only"],
        help="How to inject YOLO-based object information into the final generation prompt.",
    )
    parser.add_argument(
        "--base_question",
        type=str,
        default="Please describe this image in detail.",
        help="Base image captioning question before PromptBan guidance is appended.",
    )
    parser.add_argument(
        "--save_prompt_records",
        type=str2bool,
        default=True,
        help="Save per-image PromptBan prompt/object records in DETECTOR_info.json.",
    )

    args = parser.parse_known_args()[0]
    return args


def load_tokenizer(model_config, model_name):
    if model_name in ["llava-1.5", "mplug-owl2"]:
        tokenizer_path = "merged_ckpt"
    elif model_name == "minigpt4":
        tokenizer_path = "llama_model"
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return AutoTokenizer.from_pretrained(model_config[tokenizer_path], use_fast=False)


def generate_caption(
    model,
    tokenizer,
    model_name,
    image,
    prompt,
    image_path,
    use_nucleus_sampling=False,
    num_beams=1,
    max_new_tokens=64,
):
    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": image, "prompt": prompt, "img_path": image_path},
                use_nucleus_sampling=use_nucleus_sampling,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
                nvcd=False,
                nvcd_previous_last_ids_list=[],
            )

    all_tokens_to_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)[0]
    caption = clean_generated_text(all_tokens_to_text, model_name)

    output_tokens = out.get("output_token_ids", None)
    if output_tokens is not None:
        token_count = int(output_tokens.shape[-1])
    else:
        token_count = int(out["sequences"].shape[-1])

    try:
        input_nl_tokens = [tokenizer.convert_ids_to_tokens(seq) for seq in out["input_token_ids"].tolist()][0]
        output_nl_tokens = [tokenizer.convert_ids_to_tokens(seq) for seq in out["output_token_ids"].tolist()][0]
    except Exception:
        input_nl_tokens = []
        output_nl_tokens = []

    return caption, token_count, input_nl_tokens, output_nl_tokens


def calculate_metrics(chair1_detect1, chair1_detect0, chair0_detect1, chair0_detect0):
    accumulated_total_to_now = chair1_detect1 + chair1_detect0 + chair0_detect1 + chair0_detect0
    if accumulated_total_to_now == 0:
        return {
            "Accuracy": None,
            "Recall (Chair GT)": None,
            "Recall (Chair HAL)": None,
            "Precision (Chair GT)": None,
            "Precision (Chair HAL)": None,
        }

    accuracy = (chair1_detect1 + chair0_detect0) / accumulated_total_to_now
    recall_chair1 = chair1_detect1 / (chair1_detect1 + chair1_detect0) if (chair1_detect1 + chair1_detect0) != 0 else 0
    recall_chair0 = chair0_detect0 / (chair0_detect0 + chair0_detect1) if (chair0_detect0 + chair0_detect1) != 0 else 0
    precision_chair1 = chair1_detect1 / (chair1_detect1 + chair0_detect1) if (chair1_detect1 + chair0_detect1) != 0 else 0
    precision_chair0 = chair0_detect0 / (chair0_detect0 + chair1_detect0) if (chair0_detect0 + chair1_detect0) != 0 else 0

    return {
        "Accuracy": accuracy,
        "Recall (Chair GT)": recall_chair1,
        "Recall (Chair HAL)": recall_chair0,
        "Precision (Chair GT)": precision_chair1,
        "Precision (Chair HAL)": precision_chair0,
    }


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    cfg = Config(args)

    yolo_version = args.yolo_version
    model_name = args.model
    decoding_strategy = "promptban"
    seed = args.seed
    num_samples = args.num_samples
    dataset_name = args.dataset_name
    data_path = args.data_path
    chair_cache_path = args.chair_cache_path
    output_dir = args.output_dir
    num_beams = 1
    batch_size = 1
    max_new_tokens = args.max_new_tokens

    setup_seeds(cfg, seed)
    device = torch.device(f"cuda:{int(args.gpu_id)}") if torch.cuda.is_available() else "cpu"

    print("Initializing Model")
    model_config = cfg.model_cfg
    print(f"model_config : {model_config}")
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    model_tokenizer = load_tokenizer(model_config, model_name)

    processor_cfg = cfg.get_config().preprocess
    processor_cfg.vis_processor.eval.do_normalize = False
    vis_processors, txt_processors = load_preprocess(processor_cfg)

    print(f"\033[42m####### Current Decoding Strategy: {decoding_strategy} / mode={args.promptban_mode} #######\033[0m")

    # HALC/RVCD 코드와 동일한 normalize setting 유지
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    norm = transforms.Normalize(mean, std)

    def process_before_norm(img_path):
        raw_image = Image.open(img_path).convert("RGB")
        if model_name == "mplug-owl2":
            max_edge = max(raw_image.size)
            image = raw_image.resize((max_edge, max_edge))
            image_tensor = process_images([image], model.image_processor)
            image = image_tensor.to(device, dtype=torch.float16)
        else:
            image = vis_processors["eval"](raw_image).unsqueeze(0)
            image = image.to(device)
        return image

    def run_yolov8_once_loaded(yolo_model, image_path):
        bounding_boxes, probabilities, entity_names, _ = yolo.run_inference(yolo_model, image_path)
        unique_items = {}
        for name, prob in zip(entity_names, probabilities):
            if name not in unique_items or prob > unique_items[name]:
                unique_items[name] = prob
        return [(entity, probability) for entity, probability in unique_items.items()]

    # YOLOv8x는 이미지마다 새로 로드하지 않고, 스크립트 시작 시 1회만 로드한다.
    yolo_model = None
    if yolo_version == "yolov8x.pt":
        print(f"Loading YOLO once: {yolo_version}")
        yolo_model = yolo.load_yolo_model(yolo_version)

    annotation_file_path = os.path.join(args.data_path, "annotations/instances_val2014.json")
    caption_file_path = os.path.join(args.data_path, "annotations/captions_val2014.json")

    with open(annotation_file_path, "r") as f:
        coco_anns = json.loads(f.readlines()[0])

    coco = COCO(caption_file_path)
    img_ids = coco.getImgIds()

    if num_samples > len(img_ids):
        raise ValueError(f"num_samples={num_samples} is larger than available images={len(img_ids)}")

    sampled_img_ids = random.sample(img_ids, num_samples)
    print("sampled_img_ids", len(sampled_img_ids))

    img_files = []
    for cur_img_id in sampled_img_ids:
        cur_img = coco.loadImgs(cur_img_id)[0]
        img_files.append(cur_img["file_name"])

    img_dict = {}
    categories = coco_anns["categories"]
    category_dict = {int(c["id"]): c["name"] for c in categories}

    for img_info in coco_anns["images"]:
        img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

    for ann_info in coco_anns["annotations"]:
        img_dict[ann_info["image_id"]]["anns"].append(category_dict[ann_info["category_id"]])

    base_dir = os.path.join(output_dir, "chair", args.model)
    os.makedirs(base_dir, exist_ok=True)

    global_chair_evaluator = None
    coco_path = os.path.join(args.data_path, "annotations")

    def get_chair_evaluator(chair_cache_path=chair_cache_path, coco_path=coco_path):
        nonlocal global_chair_evaluator
        if global_chair_evaluator is None:
            if chair_cache_path and os.path.exists(chair_cache_path):
                global_chair_evaluator = pickle.load(open(chair_cache_path, "rb"))
                print(f"Loaded evaluator from cache: {chair_cache_path}")
            else:
                print("Cache not set or not exist, initializing evaluator...")
                global_chair_evaluator = CHAIR(coco_path)
                os.makedirs(os.path.dirname(chair_cache_path), exist_ok=True)
                pickle.dump(global_chair_evaluator, open(chair_cache_path, "wb"))
                print(f"Evaluator cached to: {chair_cache_path}")
        return global_chair_evaluator

    def evaluate_sentence(sentence, image_id, chair_cache_path=chair_cache_path, coco_path=coco_path):
        evaluator = get_chair_evaluator(chair_cache_path, coco_path)
        words, node_words, _, _ = evaluator.caption_to_words(sentence)
        gt_objects = evaluator.imid_to_objects.get(image_id, set())

        results = {"ground_truth": [], "hallucinated": []}
        for word, node_word in zip(words, node_words):
            if node_word in gt_objects:
                results["ground_truth"].append((node_word, word))
            else:
                results["hallucinated"].append((node_word, word))
        return results

    def chair_change_synonym_to_cocofirst_word(word):
        evaluator = get_chair_evaluator(chair_cache_path, coco_path)
        words, node_words, _, double_words = evaluator.caption_to_words(word)
        print(words, node_words, double_words)
        if len(node_words) == 1:
            return node_words[0]
        # 정상 처리되지 않은 단어. chair.py에 synonym을 추가하고 CHAIR_CACHE를 초기화하면 개선 가능.
        return "chair_add_" + " ".join(double_words)

    formatted_time = datetime.now().strftime("%Y%m%d%H%M")
    result_dir = os.path.join(
        base_dir,
        f"promptban_{args.promptban_mode}_{formatted_time}_seed_{seed}_samples_{num_samples}_maxtokens_{max_new_tokens}",
    )
    os.makedirs(result_dir, exist_ok=True)

    global_all_info = {
        "model_name": model_name,
        "decoding_strategy": decoding_strategy,
        "promptban_mode": args.promptban_mode,
        "base_question": args.base_question,
        "seed": seed,
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "dataset_name": dataset_name,
        "data_path": data_path,
        "output_dir": output_dir,
        "num_beams": num_beams,
        "batch_size": batch_size,
        "chair1_detect1": 0,
        "chair0_detect0": 0,
        "chair1_detect0": 0,
        "chair0_detect1": 0,
        "total_detector_score": [],
        "chair_not_yet_doublewords": [],
        "promptban_records": [],
        "latency": 0,
        "total_generated_tokens": 0,
        "latency_per_token": 0,
    }

    seed_valid_check = []
    for path in img_files:
        img_id = int(path.split(".jpg")[0][-6:])
        seed_valid_check.append(img_id)
    seed_valid_check = sorted(seed_valid_check)
    print(f"시드 : {seed} / 샘플링된 이미지들 : {seed_valid_check[:20]}")

    start_time = time.time()

    draft_captions_path = os.path.join(
        result_dir,
        f"promptban_{model_name}_{formatted_time}_DRAFT_generated_captions.jsonl",
    )
    promptban_captions_path = os.path.join(
        result_dir,
        f"promptban_{model_name}_{args.promptban_mode}_{formatted_time}_seed_{seed}_samples_{num_samples}_maxtokens_{max_new_tokens}_generated_captions.jsonl",
    )

    for idx, img_file in tqdm(enumerate(img_files), total=len(img_files)):
        img_id = int(img_file.split(".jpg")[0][-6:])
        img_info = img_dict[img_id]
        assert img_info["name"] == img_file

        image_path = os.path.join(args.data_path, img_file)
        image = norm(process_before_norm(image_path))

        # 1) Draft caption generation
        draft_question = args.base_question
        template = INSTRUCTION_TEMPLATE[args.model]
        draft_prompt = template.replace("<question>", draft_question)

        draft_output_text, draft_token_count, input_nl_tokens, output_nl_tokens = generate_caption(
            model=model,
            tokenizer=model_tokenizer,
            model_name=model_name,
            image=image,
            prompt=draft_prompt,
            image_path=image_path,
            use_nucleus_sampling=args.sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )

        print("-" * 30)
        print(f"image_count : {idx}")
        print(f"image_id : {img_id}")
        print(f"input_nl_tokens : {len(input_nl_tokens)}, {input_nl_tokens}")
        print("-" * 30)
        print(f"output_nl_tokens : {len(output_nl_tokens)}, {output_nl_tokens}")
        print("-" * 30)
        print(f"draft_caption :\n{draft_output_text}")

        # 2) YOLO detection + CHAIR synonym compatibility conversion
        if yolo_model is not None:
            yolo_detected_entity_prob = run_yolov8_once_loaded(yolo_model, image_path)
        else:
            yolo_detected_entity_prob = yolo.main(image_path, yolo_version)

        yolo_detected_entity_list = [entity for entity, prob in yolo_detected_entity_prob]

        for i in range(len(yolo_detected_entity_list)):
            cocofirst_or_notyetword = chair_change_synonym_to_cocofirst_word(yolo_detected_entity_list[i])
            if cocofirst_or_notyetword.startswith("chair_add_"):
                global_all_info["chair_not_yet_doublewords"].append(
                    cocofirst_or_notyetword.split("chair_add_")[-1]
                )
            else:
                yolo_detected_entity_list[i] = cocofirst_or_notyetword

        yolo_detected_entity_list = unique_preserve_order(yolo_detected_entity_list)

        # 3) draft caption objects 중 YOLO가 찾은 것 / 못 찾은 것 분리
        detected_info = {}
        chair_evaluator = get_chair_evaluator(chair_cache_path, coco_path)
        draft_synonyms = chair_evaluator.process_sentence_get_coco_synonyms(draft_output_text)
        for synonym in draft_synonyms:
            # synonym = (coco_first_word, synonym_from_draft)
            if synonym[0] in yolo_detected_entity_list:
                detected_info[synonym] = 1
            else:
                detected_info[synonym] = 0

        print(f"yolo_detected_entity_list : {yolo_detected_entity_list}")
        print(f"detected_info : {detected_info}")

        # Optional detector score against CHAIR ground-truth/hallucination labels.
        draft_chair_answer_dict = None
        if args.check_draft_chair:
            draft_chair_answer = evaluate_sentence(draft_output_text, img_id)
            draft_chair_answer_dict = {
                (cocofirst, cocosynonym): 1
                for cocofirst, cocosynonym in draft_chair_answer["ground_truth"]
            }
            draft_chair_answer_dict.update(
                {
                    (cocofirst, cocosynonym): 0
                    for cocofirst, cocosynonym in draft_chair_answer["hallucinated"]
                }
            )

            for chair_key, infer_value in draft_chair_answer_dict.items():
                chair_first = chair_key[0]
                for detected_key, gt_value in detected_info.items():
                    detected_first = detected_key[0]
                    if chair_first == detected_first:
                        if gt_value == 1 and infer_value == 1:
                            global_all_info["chair1_detect1"] += 1
                        elif gt_value == 1 and infer_value == 0:
                            global_all_info["chair1_detect0"] += 1
                        elif gt_value == 0 and infer_value == 1:
                            global_all_info["chair0_detect1"] += 1
                        elif gt_value == 0 and infer_value == 0:
                            global_all_info["chair0_detect0"] += 1

            accumulated_detector_score = calculate_metrics(
                global_all_info["chair1_detect1"],
                global_all_info["chair1_detect0"],
                global_all_info["chair0_detect1"],
                global_all_info["chair0_detect0"],
            )
            print(f"accumulated_detector_score : {accumulated_detector_score}")

        hal_detected = []
        gt_detected = []
        for key, value in detected_info.items():
            if value == 0:
                hal_detected.append(key[1])
            elif value == 1:
                gt_detected.append(key[1])

        hal_detected = unique_preserve_order(hal_detected)
        gt_detected = unique_preserve_order(gt_detected)

        print(f"hal_detected_synonym : {hal_detected}")
        print(f"gt_detected_synonym : {gt_detected}")

        # 4) PromptBan prompt로 final caption 생성. Contrastive decoding 없음.
        promptban_question = build_promptban_question(
            base_question=args.base_question,
            gt_detected=gt_detected,
            hal_detected=hal_detected,
            mode=args.promptban_mode,
        )
        promptban_prompt = template.replace("<question>", promptban_question)

        promptban_output_text, promptban_token_count, _, _ = generate_caption(
            model=model,
            tokenizer=model_tokenizer,
            model_name=model_name,
            image=image,
            prompt=promptban_prompt,
            image_path=image_path,
            use_nucleus_sampling=args.sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )

        print("-" * 30)
        print(f"promptban_prompt :\n{promptban_question}")
        print("-" * 30)
        print(f"promptban_caption :\n{promptban_output_text}")
        print("-" * 30)

        # 5) Save outputs
        now_draft_result = {"image_id": int(img_id), "caption": draft_output_text}
        with open(draft_captions_path, "a", encoding="utf-8") as f:
            json.dump(now_draft_result, f, ensure_ascii=False)
            f.write("\n")

        now_promptban_result = {
            "image_id": int(img_id),
            "caption": promptban_output_text,
            "tokens": promptban_token_count,
        }
        global_all_info["total_generated_tokens"] += promptban_token_count
        with open(promptban_captions_path, "a", encoding="utf-8") as f:
            json.dump(now_promptban_result, f, ensure_ascii=False)
            f.write("\n")

        if args.save_prompt_records:
            record = {
                "image_id": int(img_id),
                "image_path": image_path,
                "draft_caption": draft_output_text,
                "promptban_caption": promptban_output_text,
                "promptban_mode": args.promptban_mode,
                "promptban_question": promptban_question,
                "yolo_detected_entity_list": yolo_detected_entity_list,
                "draft_synonyms": [list(x) for x in draft_synonyms],
                "detected_info": [
                    {
                        "coco_first_word": key[0],
                        "draft_synonym": key[1],
                        "yolo_detected": int(value),
                    }
                    for key, value in detected_info.items()
                ],
                "gt_detected": gt_detected,
                "hal_detected": hal_detected,
            }
            if draft_chair_answer_dict is not None:
                record["draft_chair_answer_dict"] = [
                    {
                        "coco_first_word": key[0],
                        "draft_synonym": key[1],
                        "chair_gt": int(value),
                    }
                    for key, value in draft_chair_answer_dict.items()
                ]
            global_all_info["promptban_records"].append(record)

    if args.check_draft_chair:
        total_detector_score = calculate_metrics(
            global_all_info["chair1_detect1"],
            global_all_info["chair1_detect0"],
            global_all_info["chair0_detect1"],
            global_all_info["chair0_detect0"],
        )
        global_all_info["total_detector_score"].append(total_detector_score)

    global_all_info["latency"] = time.time() - start_time
    if global_all_info["total_generated_tokens"] > 0 and model_name != "minigpt4":
        global_all_info["latency_per_token"] = (
            global_all_info["latency"] / global_all_info["total_generated_tokens"]
        )

    global_info_save_path = os.path.join(
        result_dir,
        f"promptban_{model_name}_{args.promptban_mode}_{formatted_time}_seed_{seed}_samples_{num_samples}_maxtokens_{max_new_tokens}_DETECTOR_info.json",
    )
    with open(global_info_save_path, "w", encoding="utf-8") as json_file:
        json.dump(global_all_info, json_file, indent=4, ensure_ascii=False)

    print("=" * 80)
    print(f"Saved draft captions to      : {draft_captions_path}")
    print(f"Saved promptban captions to  : {promptban_captions_path}")
    print(f"Saved detector info to       : {global_info_save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()


# Example runs
#
# CUDA_VISIBLE_DEVICES=0 \
# python promptban_generation_chair_bleu.py \
# --model llava-1.5 \
# --data_path /home/jihoon/jihoon/DATASETS/coco2014/val2014 \
# --chair_cache_path eval/CHAIR_CACHE/chair.pkl \
# --num_samples 300 \
# --seed 42 \
# --gpu-id 0 \
# --output_dir ./generated_captions_promptban_both/ \
# --promptban_mode both

# CUDA_VISIBLE_DEVICES=0 \
# python promptban_generation_chair_bleu.py \
# --model llava-1.5 \
# --data_path /home/jihoon/jihoon/DATASETS/coco2014/val2014 \
# --chair_cache_path eval/CHAIR_CACHE/chair.pkl \
# --num_samples 300 \
# --seed 42 \
# --gpu-id 0 \
# --output_dir ./generated_captions_promptban_positive/ \
# --promptban_mode positive_only

# CUDA_VISIBLE_DEVICES=0 \
# python promptban_generation_chair_bleu.py \
# --model llava-1.5 \
# --data_path /home/jihoon/jihoon/DATASETS/coco2014/val2014 \
# --chair_cache_path eval/CHAIR_CACHE/chair.pkl \
# --num_samples 300 \
# --seed 42 \
# --gpu-id 0 \
# --output_dir ./generated_captions_promptban_negative/ \
# --promptban_mode negative_only
