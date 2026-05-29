#!/usr/bin/env python3
"""
refgrid_slot_permutation_probe.py

Slot permutation probe for eRVCD negative reference grids.

Purpose:
  Test whether the VLM first-token distribution changes depending on where
  the same reference images are placed in a 2x2 grid.

What it does:
  1. Takes one or more object sets, e.g.
       apple,orange,carrot
       desk,mouse,table,keyboard
  2. Finds {object}.png/jpg/webp under --ref_folder_path.
  3. Creates all 2x2 slot permutations:
       - 3 refs: P(4,3)=24 layouts, one black slot varies.
       - 4 refs: 4!=24 layouts.
       - 2 refs: P(4,2)=12 layouts.
       - 1 ref : 4 layouts.
  4. For each layout:
       - saves the actual grid image
       - probes first-token top-N distribution
       - forces each top-N first token and greedily continues
       - saves one wide image: grid + bar plot
  5. Writes CSV/JSON summaries:
       - layout_summary.csv
       - object_slot_results.csv
       - position_summary.csv
       - result.json

Example:
  CUDA_VISIBLE_DEVICES=0 python refgrid_slot_permutation_probe.py \
    --model llava-1.5 \
    --ref_folder_path DB_single_concept_images_flux_generated/generated_images \
    --object_sets "apple,orange,carrot;desk,mouse,table,keyboard" \
    --top_n 10 \
    --continuation_tokens 10 \
    --seed 42
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.append("mPLUG-Owl/mPLUG-Owl2")
sys.path.append("./")
sys.path.append("../")
sys.path.append("./eval")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import AutoTokenizer

from minigpt4.models import load_preprocess
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from mplug_owl2.mm_utils import process_images


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

SLOT_NAMES = ["top-left", "top-right", "bottom-left", "bottom-right"]

# Lightweight aliases for COCO-ish labels and common VLM outputs.
OBJECT_ALIASES: Dict[str, List[str]] = {
    "airplane": ["plane", "aircraft", "jet"],
    "bicycle": ["bike", "cycle"],
    "motorcycle": ["motorbike", "bike"],
    "traffic light": ["traffic", "light", "signal"],
    "fire hydrant": ["hydrant"],
    "stop sign": ["stop", "sign"],
    "parking meter": ["meter"],
    "bench": ["seat"],
    "bird": ["birds"],
    "cat": ["kitty"],
    "dog": ["puppy"],
    "horse": ["pony"],
    "sheep": ["lamb"],
    "cow": ["cattle"],
    "elephant": ["elephants"],
    "bear": ["teddy"],
    "zebra": ["zebras"],
    "giraffe": ["giraffes"],
    "backpack": ["bag", "rucksack"],
    "umbrella": ["parasol"],
    "handbag": ["purse", "bag"],
    "tie": ["necktie"],
    "suitcase": ["luggage"],
    "frisbee": ["disc"],
    "skis": ["ski"],
    "snowboard": ["board"],
    "sports ball": ["ball", "basketball", "baseball", "soccer", "football", "tennis"],
    "kite": ["kites"],
    "baseball bat": ["bat"],
    "baseball glove": ["glove"],
    "skateboard": ["skate"],
    "surfboard": ["surf"],
    "tennis racket": ["racket", "racquet"],
    "wine glass": ["glass", "wineglass"],
    "cup": ["mug", "glass"],
    "fork": ["utensil"],
    "knife": ["utensil"],
    "spoon": ["utensil"],
    "bowl": ["dish"],
    "banana": ["bananas"],
    "apple": ["apples"],
    "sandwich": ["burger", "hamburger", "sub", "bread"],
    "orange": ["pumpkin", "fruit"],
    "broccoli": ["vegetable"],
    "carrot": ["vegetable"],
    "hot dog": ["hotdog", "sausage"],
    "pizza": ["pie"],
    "donut": ["doughnut"],
    "doughnut": ["donut"],
    "cake": ["dessert"],
    "chair": ["seat"],
    "couch": ["sofa"],
    "potted plant": ["plant", "pot", "planter"],
    "bed": ["mattress"],
    "dining table": ["table", "desk"],
    "table": ["desk", "dining table"],
    "desk": ["table", "dining table"],
    "toilet": ["bathroom"],
    "tv": ["television", "monitor", "screen"],
    "laptop": ["computer", "notebook"],
    "mouse": ["computer mouse"],
    "remote": ["controller"],
    "keyboard": ["keypad"],
    "cell phone": ["phone", "mobile", "smartphone"],
    "microwave": ["oven"],
    "oven": ["stove"],
    "toaster": ["toast"],
    "sink": ["basin"],
    "refrigerator": ["fridge", "refriger"],
    "book": ["books"],
    "clock": ["watch"],
    "vase": ["jar"],
    "scissors": ["shears"],
    "teddy bear": ["teddy", "bear"],
    "hair drier": ["dryer", "drier"],
    "toothbrush": ["brush"],
}


DEFAULT_OBJECT_SETS = [
    ["sandwich", "cake", "bowl", "table"],
    ["desk", "mouse", "table", "keyboard"],
    ["microwave", "toaster", "cup", "sink"],
    ["bowl", "pancake", "table"],
    ["apple", "orange", "carrot"],
    ["table", "orange", "apple"],
]


Json = Dict[str, Any]


def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "y", "1"):
        return True
    if str(v).lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def safe_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9가-힣._-]+", "_", text).strip("_")
    return text or "unnamed"


def parse_object_sets(value: str) -> List[List[str]]:
    if not value.strip():
        return DEFAULT_OBJECT_SETS
    sets: List[List[str]] = []
    for chunk in value.split(";"):
        names = [x.strip() for x in chunk.split(",") if x.strip()]
        if names:
            sets.append(names)
    return sets


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("▁", " ").replace("Ġ", " ")
    text = re.sub(r"[^a-z0-9가-힣\s]+", " ", text)
    return " ".join(text.split())


def object_aliases(name: str) -> List[str]:
    norm = normalize_text(name.replace("_", " "))
    aliases = {norm, norm.replace(" ", "")}
    for part in norm.split():
        aliases.add(part)
    for alias in OBJECT_ALIASES.get(norm, []):
        a = normalize_text(alias)
        aliases.add(a)
        aliases.add(a.replace(" ", ""))
        for part in a.split():
            aliases.add(part)
    return sorted(x for x in aliases if x)


def text_matches_object(text: str, obj_name: str) -> bool:
    norm_text = normalize_text(text)
    compact_text = norm_text.replace(" ", "")
    if not norm_text:
        return False
    for alias in object_aliases(obj_name):
        alias_n = normalize_text(alias)
        alias_c = alias_n.replace(" ", "")
        if not alias_n:
            continue
        # Exact/word boundary match for single words, substring for compact multi-token aliases.
        if alias_n in norm_text.split():
            return True
        if alias_n and re.search(rf"(?<![a-z0-9]){re.escape(alias_n)}(?![a-z0-9])", norm_text):
            return True
        if len(alias_c) >= 4 and alias_c in compact_text:
            return True
    return False


def pil_resample_lanczos():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def load_tokenizer(model_config, model_name: str):
    if model_name in ["llava-1.5", "mplug-owl2"]:
        tokenizer_path = "merged_ckpt"
    elif model_name == "minigpt4":
        tokenizer_path = "llama_model"
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return AutoTokenizer.from_pretrained(model_config[tokenizer_path], use_fast=False)


def token_display(tokenizer, token_id: int) -> Json:
    raw_token = tokenizer.convert_ids_to_tokens([int(token_id)], skip_special_tokens=False)[0]
    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return {
        "token_id": int(token_id),
        "token": str(raw_token),
        "decoded": str(decoded),
        "decoded_clean": str(decoded).replace("\n", "\\n"),
    }


def is_special_id(tokenizer, token_id: int) -> bool:
    special_ids = set()
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
        val = getattr(tokenizer, attr, None)
        if val is not None:
            special_ids.add(int(val))
    extra = getattr(tokenizer, "all_special_ids", None)
    if extra is not None:
        special_ids.update(int(x) for x in extra)
    return int(token_id) in special_ids


def get_lm_head_matrix(model, model_name: str):
    if model_name == "mplug-owl2":
        return model.model.lm_head.weight
    return model.llama_model.lm_head.weight


def next_token_logits(
    *,
    model,
    model_name: str,
    image,
    prompt: str,
    image_path: str,
    prev_tokens: Sequence[torch.Tensor],
    use_nucleus_sampling: bool = False,
    num_beams: int = 1,
):
    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": image, "prompt": prompt, "img_path": str(image_path)},
                use_nucleus_sampling=use_nucleus_sampling,
                num_beams=num_beams,
                max_new_tokens=1,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
                nvcd=True,
                nvcd_previous_last_ids_list=list(prev_tokens),
            )

    last_hidden = out["hidden_states"][-1][-1][:, -1, :].detach().clone()
    lm_head = get_lm_head_matrix(model, model_name).detach()
    with torch.no_grad():
        logits = torch.matmul(last_hidden, lm_head.T)
    return logits.detach()


def top_n_from_logits(logits, tokenizer, top_n: int, skip_special_tokens: bool) -> List[Json]:
    probs = F.softmax(logits.float(), dim=-1)[0]
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)

    results: List[Json] = []
    for prob, token_id in zip(sorted_probs.tolist(), sorted_ids.tolist()):
        token_id = int(token_id)
        if skip_special_tokens and is_special_id(tokenizer, token_id):
            continue
        info = token_display(tokenizer, token_id)
        info["probability"] = float(prob)
        info["log_probability"] = float(math.log(max(float(prob), 1e-45)))
        info["rank"] = len(results) + 1
        results.append(info)
        if len(results) >= int(top_n):
            break
    return results


def greedy_continue_from_first_token(
    *,
    model,
    tokenizer,
    model_name: str,
    image,
    prompt: str,
    image_path: str,
    first_token_id: int,
    total_tokens: int,
    use_nucleus_sampling: bool = False,
    num_beams: int = 1,
) -> Json:
    device = image.device
    output_tokens = [torch.tensor(int(first_token_id), device=device)]

    for _ in range(max(0, int(total_tokens) - 1)):
        logits = next_token_logits(
            model=model,
            model_name=model_name,
            image=image,
            prompt=prompt,
            image_path=image_path,
            prev_tokens=output_tokens,
            use_nucleus_sampling=use_nucleus_sampling,
            num_beams=num_beams,
        )
        next_id = int(torch.argmax(F.softmax(logits.float(), dim=-1), dim=-1).item())
        output_tokens.append(torch.tensor(next_id, device=device))
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and next_id == int(eos_id):
            break

    token_ids = [int(t.item()) for t in output_tokens]
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    continuation_token_ids = token_ids[1:]
    continuation_text = tokenizer.decode(continuation_token_ids, skip_special_tokens=True).strip()
    return {
        "forced_first_token": token_display(tokenizer, first_token_id),
        "generated_token_ids": token_ids,
        "generated_tokens": [token_display(tokenizer, tid) for tid in token_ids],
        "generated_text": decoded,
        "continuation_token_ids": continuation_token_ids,
        "continuation_text": continuation_text,
    }


def discover_ref_image(ref_folder: Path, object_name: str) -> Path:
    suffixes = [".png", ".jpg", ".jpeg", ".webp"]
    candidates = []
    # Exact.
    for suffix in suffixes:
        candidates.append(ref_folder / f"{object_name}{suffix}")
    # Common safe filename alternatives.
    safe = object_name.replace(" ", "_")
    for suffix in suffixes:
        candidates.append(ref_folder / f"{safe}{suffix}")
    hyphen = object_name.replace(" ", "-")
    for suffix in suffixes:
        candidates.append(ref_folder / f"{hyphen}{suffix}")

    for p in candidates:
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(f"Could not find reference image for object '{object_name}' under {ref_folder}")


def make_manual_2x2_grid(
    *,
    slot_paths: Sequence[Optional[Path]],
    slot_names_for_labels: Sequence[Optional[str]],
    save_path: Path,
    canvas_size: int = 336,
    background_color=(0, 0, 0),
    draw_labels: bool = False,
) -> Image.Image:
    if len(slot_paths) != 4:
        raise ValueError("slot_paths must have length 4 for a 2x2 grid.")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (canvas_size, canvas_size), background_color)
    slot_size = canvas_size // 2
    resample = pil_resample_lanczos()

    for idx, ref_path in enumerate(slot_paths):
        row, col = divmod(idx, 2)
        left = col * slot_size
        top = row * slot_size
        right = canvas_size if col == 1 else (col + 1) * slot_size
        bottom = canvas_size if row == 1 else (row + 1) * slot_size
        slot_w = right - left
        slot_h = bottom - top

        slot = Image.new("RGB", (slot_w, slot_h), background_color)
        if ref_path is not None:
            ref_img = Image.open(ref_path).convert("RGB")
            ref_img.thumbnail((slot_w, slot_h), resample)
            paste_x = (slot_w - ref_img.width) // 2
            paste_y = (slot_h - ref_img.height) // 2
            slot.paste(ref_img, (paste_x, paste_y))

        if draw_labels:
            draw = ImageDraw.Draw(slot)
            label = slot_names_for_labels[idx] or "black"
            # Semi-transparent is harder on RGB, draw a black rectangle and white text.
            text = f"{idx}:{SLOT_NAMES[idx]}\n{label}"
            bbox = draw.multiline_textbbox((4, 4), text)
            draw.rectangle((bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2), fill=(0, 0, 0))
            draw.multiline_text((4, 4), text, fill=(255, 255, 255))

        canvas.paste(slot, (left, top))

    canvas.save(save_path)
    return canvas


def generate_slot_permutations(object_names: Sequence[str]) -> List[List[Optional[str]]]:
    """Return layouts of length 4. Each slot contains object name or None."""
    names = list(object_names)
    n = len(names)
    if n < 1 or n > 4:
        raise ValueError("This script supports 1 to 4 objects per set for a fixed 2x2 grid.")

    layouts: List[List[Optional[str]]] = []
    for slots in itertools.permutations(range(4), n):
        layout: List[Optional[str]] = [None, None, None, None]
        for obj, slot in zip(names, slots):
            layout[slot] = obj
        layouts.append(layout)
    return layouts


def wrap_truncate_text(text: str, width: int = 18, max_lines: int = 3) -> str:
    text = str(text).replace("\n", " ").strip()
    if not text:
        return ""
    wrapped = textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False)
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        if len(wrapped[-1]) > width - 1:
            wrapped[-1] = wrapped[-1][: width - 1] + "…"
    return "\n".join(wrapped)


def save_combined_figure(
    *,
    grid_image_path: Path,
    topn: List[Json],
    continuations: List[Json],
    save_path: Path,
    title: str,
    layout: Sequence[Optional[str]],
) -> None:
    grid_img = Image.open(grid_image_path).convert("RGB")
    labels = []
    values = []

    for item, cont in zip(topn, continuations):
        prefix = item["decoded"].replace("\n", " ").strip()
        if not prefix:
            prefix = item["token"]
        labels.append({
            "rank": item["rank"],
            "prefix": prefix,
            "continuation": cont.get("continuation_text", ""),
        })
        values.append(float(item["probability"]))

    fig_w = max(17, len(labels) * 1.65 + 6)
    fig_h = 7.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, max(2.0, len(labels) * 0.58)], wspace=0.18)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(grid_img)
    layout_lines = [f"{i}:{SLOT_NAMES[i]}={layout[i] or 'black'}" for i in range(4)]
    ax_img.set_title("2x2 slot permutation\n" + "\n".join(layout_lines), fontsize=10)
    ax_img.axis("off")

    ax_bar = fig.add_subplot(gs[0, 1])
    x = np.arange(len(labels))
    bars = ax_bar.bar(x, values)
    ax_bar.set_ylabel("Probability")
    ax_bar.set_xlabel("Top-N first-token candidates (prefix black, continuation blue)")
    ax_bar.set_title(title, fontsize=11)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([""] * len(labels))
    ax_bar.set_xlim(-0.6, len(labels) - 0.4)

    ymax = max(values) if values else 1.0
    ax_bar.set_ylim(0.0, ymax * 1.20 if ymax > 0 else 1.0)

    for bar, item in zip(bars, labels):
        height = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(ymax * 0.02, 1e-6),
            f"#{item['rank']}\n{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    trans = ax_bar.get_xaxis_transform()
    for xi, item in zip(x, labels):
        prefix_text = wrap_truncate_text(item["prefix"], width=12, max_lines=2)
        continuation_text = wrap_truncate_text(item["continuation"], width=18, max_lines=3)
        ax_bar.text(
            xi,
            -0.09,
            prefix_text,
            ha="center",
            va="top",
            fontsize=9,
            color="black",
            transform=trans,
            clip_on=False,
        )
        if continuation_text:
            ax_bar.text(
                xi,
                -0.22,
                continuation_text,
                ha="center",
                va="top",
                fontsize=8,
                color="blue",
                transform=trans,
                clip_on=False,
            )

    fig.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.36, wspace=0.16)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def infer_object_hits(topn: List[Json], continuations: List[Json], object_names: Sequence[str]) -> Dict[str, Json]:
    hits: Dict[str, Json] = {}

    for obj in object_names:
        best_rank = None
        best_prob = None
        best_text = ""
        best_prefix = ""

        for item, cont in zip(topn, continuations):
            prefix = item["decoded"]
            full = cont.get("generated_text", "")
            # Check both prefix and forced continuation text.
            if text_matches_object(prefix, obj) or text_matches_object(full, obj):
                rank = int(item["rank"])
                prob = float(item["probability"])
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_prob = prob
                    best_text = full
                    best_prefix = prefix

        hits[obj] = {
            "object": obj,
            "hit_topn": best_rank is not None,
            "best_rank": best_rank,
            "best_probability": best_prob,
            "best_prefix": best_prefix,
            "best_generated_text": best_text,
            "aliases": object_aliases(obj),
        }

    return hits


def aggregate_position_summary(object_rows: List[Json]) -> List[Json]:
    by_slot: Dict[int, List[Json]] = {}
    for row in object_rows:
        by_slot.setdefault(int(row["slot_index"]), []).append(row)

    out = []
    for slot_idx in range(4):
        rows = by_slot.get(slot_idx, [])
        n = len(rows)
        if n == 0:
            out.append({
                "slot_index": slot_idx,
                "slot_name": SLOT_NAMES[slot_idx],
                "n": 0,
                "top10_hit_rate": None,
                "top1_rate": None,
                "avg_rank_if_hit": None,
                "avg_prob_if_hit": None,
            })
            continue

        hit_rows = [r for r in rows if str(r["hit_topn"]).lower() == "true" or r["hit_topn"] is True]
        top1_rows = [r for r in rows if r.get("best_rank") == 1]
        ranks = [float(r["best_rank"]) for r in hit_rows if r.get("best_rank") is not None]
        probs = [float(r["best_probability"]) for r in hit_rows if r.get("best_probability") is not None]
        out.append({
            "slot_index": slot_idx,
            "slot_name": SLOT_NAMES[slot_idx],
            "n": n,
            "top10_hit_rate": len(hit_rows) / n,
            "top1_rate": len(top1_rows) / n,
            "avg_rank_if_hit": sum(ranks) / len(ranks) if ranks else None,
            "avg_prob_if_hit": sum(probs) / len(probs) if probs else None,
        })
    return out


def write_csv(path: Path, rows: List[Json]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # union keys preserving first-row order
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            out = {}
            for k in keys:
                v = row.get(k)
                if isinstance(v, (list, dict)):
                    out[k] = json.dumps(v, ensure_ascii=False)
                else:
                    out[k] = v
            w.writerow(out)


def parse_args():
    parser = argparse.ArgumentParser(description="2x2 slot permutation probe for reference-grid VLM distributions.")
    parser.add_argument("--model", type=str, default="llava-1.5", choices=list(MODEL_EVAL_CONFIG_PATH.keys()))
    parser.add_argument("-g", "--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref_folder_path", type=str, default="DB_single_concept_images_flux_generated/generated_images")
    parser.add_argument(
        "--object_sets",
        type=str,
        default="",
        help=(
            "Semicolon-separated object sets. Each set is comma-separated. "
            "Example: 'apple,orange,carrot;desk,mouse,table,keyboard'. "
            "If empty, uses the 3+/4-ref examples discussed in the analysis."
        ),
    )
    parser.add_argument("--grid_canvas_size", type=int, default=336)
    parser.add_argument("--draw_slot_labels", type=str2bool, default=False, help="Draw slot labels directly on saved grid images.")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--continuation_tokens", type=int, default=10)
    parser.add_argument("--question", type=str, default="What object is shown in this image? Answer with one word.")
    parser.add_argument("--output_dir", type=str, default="./refgrid_slot_permutation_probe_outputs")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--skip_special_tokens", type=str2bool, default=True)
    parser.add_argument("--max_layouts_per_set", type=int, default=0, help="0 means run all layouts. Useful for quick debugging.")
    parser.add_argument("--options", nargs="+", help="Config override passthrough for MiniGPT4 Config.")
    return parser.parse_known_args()[0]


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    setup_seeds(args.seed)

    device = torch.device(f"cuda:{int(args.gpu_id)}") if torch.cuda.is_available() else "cpu"
    model_name = args.model

    cfg = Config(args)
    print("Initializing Model")
    model_config = cfg.model_cfg
    print(f"model_config : {model_config}")
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    tokenizer = load_tokenizer(model_config, model_name)

    processor_cfg = cfg.get_config().preprocess
    processor_cfg.vis_processor.eval.do_normalize = False
    vis_processors, _ = load_preprocess(processor_cfg)

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    norm = transforms.Normalize(mean, std)

    def process_pil_before_norm(raw_image):
        raw_image = raw_image.convert("RGB")
        if model_name == "mplug-owl2":
            max_edge = max(raw_image.size)
            image = raw_image.resize((max_edge, max_edge))
            image_tensor = process_images([image], model.image_processor)
            image = image_tensor.to(device, dtype=torch.float16)
        else:
            image = vis_processors["eval"](raw_image).unsqueeze(0)
            image = image.to(device)
        return image

    ref_folder = Path(args.ref_folder_path).expanduser().resolve()
    object_sets = parse_object_sets(args.object_sets)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    root_name = f"slotperm_{model_name}_seed_{args.seed}_topn_{args.top_n}_{timestamp}"
    root_dir = Path(args.output_dir).expanduser().resolve() / safe_filename(root_name)
    root_dir.mkdir(parents=True, exist_ok=True)

    template = INSTRUCTION_TEMPLATE[model_name]
    prompt = template.replace("<question>", args.question)

    layout_rows: List[Json] = []
    object_rows: List[Json] = []
    all_results: List[Json] = []

    for set_idx, object_names in enumerate(object_sets, start=1):
        if not (1 <= len(object_names) <= 4):
            print(f"[SKIP] set {set_idx}: only 1-4 objects are supported: {object_names}")
            continue

        ref_paths_by_obj = {obj: discover_ref_image(ref_folder, obj) for obj in object_names}
        layouts = generate_slot_permutations(object_names)
        if args.max_layouts_per_set and args.max_layouts_per_set > 0:
            layouts = layouts[: args.max_layouts_per_set]

        set_name = safe_filename("_".join(object_names))
        set_dir = root_dir / f"set_{set_idx:02d}_{set_name}"
        set_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print(f"Object set {set_idx}: {object_names}")
        print(f"Layouts: {len(layouts)}")
        print("=" * 80)

        for layout_idx, layout in enumerate(layouts, start=1):
            layout_dir = set_dir / f"layout_{layout_idx:03d}"
            layout_dir.mkdir(parents=True, exist_ok=True)

            slot_paths: List[Optional[Path]] = [
                None if obj is None else ref_paths_by_obj[obj]
                for obj in layout
            ]
            grid_path = layout_dir / "reference_grid.png"
            grid_canvas = make_manual_2x2_grid(
                slot_paths=slot_paths,
                slot_names_for_labels=layout,
                save_path=grid_path,
                canvas_size=args.grid_canvas_size,
                draw_labels=args.draw_slot_labels,
            )

            image = norm(process_pil_before_norm(grid_canvas))

            first_logits = next_token_logits(
                model=model,
                model_name=model_name,
                image=image,
                prompt=prompt,
                image_path=str(grid_path),
                prev_tokens=[],
                use_nucleus_sampling=args.sample,
                num_beams=1,
            )

            topn = top_n_from_logits(
                logits=first_logits,
                tokenizer=tokenizer,
                top_n=args.top_n,
                skip_special_tokens=args.skip_special_tokens,
            )

            continuations: List[Json] = []
            for item in topn:
                cont = greedy_continue_from_first_token(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    image=image,
                    prompt=prompt,
                    image_path=str(grid_path),
                    first_token_id=int(item["token_id"]),
                    total_tokens=args.continuation_tokens,
                    use_nucleus_sampling=args.sample,
                    num_beams=1,
                )
                cont["rank"] = item["rank"]
                cont["first_token_probability"] = item["probability"]
                continuations.append(cont)

            hits = infer_object_hits(topn, continuations, object_names)

            top1_text = continuations[0]["generated_text"] if continuations else ""
            top1_decoded = topn[0]["decoded"] if topn else ""
            top1_prob = topn[0]["probability"] if topn else None

            top1_objects = [
                obj for obj in object_names
                if hits[obj]["best_rank"] == 1
            ]

            combined_path = layout_dir / "slot_permutation_distribution.png"
            save_combined_figure(
                grid_image_path=grid_path,
                topn=topn,
                continuations=continuations,
                save_path=combined_path,
                title=f"Set {set_idx} layout {layout_idx}: 2x2 slot permutation",
                layout=layout,
            )

            result = {
                "set_index": set_idx,
                "layout_index": layout_idx,
                "object_names": object_names,
                "layout": list(layout),
                "slot_names": SLOT_NAMES,
                "question": args.question,
                "prompt": prompt,
                "grid_path": str(grid_path),
                "combined_figure_path": str(combined_path),
                "top1_decoded": top1_decoded,
                "top1_generated_text": top1_text,
                "top1_probability": top1_prob,
                "top1_objects": top1_objects,
                "topn_first_token_probs": topn,
                "forced_continuations": continuations,
                "object_hits": hits,
            }

            (layout_dir / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            all_results.append(result)

            layout_rows.append({
                "set_index": set_idx,
                "layout_index": layout_idx,
                "object_names": object_names,
                "layout_top_left": layout[0] or "",
                "layout_top_right": layout[1] or "",
                "layout_bottom_left": layout[2] or "",
                "layout_bottom_right": layout[3] or "",
                "top1_decoded": top1_decoded,
                "top1_generated_text": top1_text,
                "top1_probability": top1_prob,
                "top1_objects": top1_objects,
                "grid_path": str(grid_path),
                "combined_figure_path": str(combined_path),
            })

            for slot_idx, obj in enumerate(layout):
                if obj is None:
                    continue
                h = hits[obj]
                object_rows.append({
                    "set_index": set_idx,
                    "layout_index": layout_idx,
                    "object": obj,
                    "slot_index": slot_idx,
                    "slot_name": SLOT_NAMES[slot_idx],
                    "row": slot_idx // 2,
                    "col": slot_idx % 2,
                    "hit_topn": bool(h["hit_topn"]),
                    "best_rank": h["best_rank"],
                    "best_probability": h["best_probability"],
                    "best_prefix": h["best_prefix"],
                    "best_generated_text": h["best_generated_text"],
                    "aliases": h["aliases"],
                    "layout_top_left": layout[0] or "",
                    "layout_top_right": layout[1] or "",
                    "layout_bottom_left": layout[2] or "",
                    "layout_bottom_right": layout[3] or "",
                    "combined_figure_path": str(combined_path),
                })

            print(
                f"set={set_idx:02d} layout={layout_idx:03d} "
                f"layout={layout} top1={top1_decoded!r} prob={top1_prob}"
            )

    position_summary = aggregate_position_summary(object_rows)

    write_csv(root_dir / "layout_summary.csv", layout_rows)
    write_csv(root_dir / "object_slot_results.csv", object_rows)
    write_csv(root_dir / "position_summary.csv", position_summary)

    summary = {
        "model_name": model_name,
        "seed": args.seed,
        "ref_folder_path": str(ref_folder),
        "object_sets": object_sets,
        "question": args.question,
        "top_n": args.top_n,
        "continuation_tokens": args.continuation_tokens,
        "num_layouts": len(layout_rows),
        "layout_summary_csv": str(root_dir / "layout_summary.csv"),
        "object_slot_results_csv": str(root_dir / "object_slot_results.csv"),
        "position_summary_csv": str(root_dir / "position_summary.csv"),
        "position_summary": position_summary,
        "results": all_results,
    }
    (root_dir / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Slot Permutation Probe Summary",
        "",
        f"- model: `{model_name}`",
        f"- object sets: `{object_sets}`",
        f"- layouts: `{len(layout_rows)}`",
        f"- top_n: `{args.top_n}`",
        f"- continuation_tokens: `{args.continuation_tokens}`",
        "",
        "## Position summary",
        "",
        "| slot | n | top10 hit | top1 rate | avg rank if hit | avg prob if hit |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in position_summary:
        def fmt(x):
            if x is None:
                return ""
            if isinstance(x, float):
                return f"{x:.4f}"
            return str(x)
        md_lines.append(
            f"| {row['slot_name']} | {row['n']} | {fmt(row['top10_hit_rate'])} | "
            f"{fmt(row['top1_rate'])} | {fmt(row['avg_rank_if_hit'])} | {fmt(row['avg_prob_if_hit'])} |"
        )
    md_lines += [
        "",
        "## Files",
        "",
        f"- layout_summary_csv: `{root_dir / 'layout_summary.csv'}`",
        f"- object_slot_results_csv: `{root_dir / 'object_slot_results.csv'}`",
        f"- position_summary_csv: `{root_dir / 'position_summary.csv'}`",
        f"- result_json: `{root_dir / 'result.json'}`",
    ]
    (root_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("=" * 80)
    print(f"Saved root dir       : {root_dir}")
    print(f"layout_summary.csv   : {root_dir / 'layout_summary.csv'}")
    print(f"object_slot_results  : {root_dir / 'object_slot_results.csv'}")
    print(f"position_summary.csv : {root_dir / 'position_summary.csv'}")
    print(f"report.md            : {root_dir / 'report.md'}")
    print("=" * 80)


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=0 \
# python refgrid_slot_permutation_probe.py \
#   --model llava-1.5 \
#   --ref_folder_path DB_single_concept_images_flux_generated/generated_images \
#   --top_n 10 \
#   --continuation_tokens 10 \
#   --seed 42 \
#   --output_dir ./refgrid_slot_permutation_outputs