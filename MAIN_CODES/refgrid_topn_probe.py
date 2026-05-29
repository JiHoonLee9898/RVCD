import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
import textwrap

sys.path.append("mPLUG-Owl/mPLUG-Owl2")
sys.path.append("./")
sys.path.append("../")
sys.path.append("./eval")

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
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


"""
refgrid_topn_probe.py

Diagnostic script for eRVCD-style reference grids.

What it does:
1. Randomly samples several single-concept reference images, e.g. dog.png, car.png,
   from --ref_folder_path, or uses --object_names if provided.
2. Merges them into one square grid image using black_front by default.
   black_front means: black empty cells first, reference images in the later slots.
3. Feeds the grid image to an LVLM with a one-word classification prompt.
4. Extracts the first decoding step top-N token probabilities.
5. For each top-N first token, greedily continues generation up to --continuation_tokens
   total tokens, with that first token forced.
6. Saves:
   - the grid image
   - one wide combined figure: input grid + top-N probability bars + color-coded labels
   - JSON/Markdown summaries

This is meant to inspect whether a negative reference grid pushes the model's first-token
probability mass toward the expected object names.
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

GRID_FILL_MODES = ["black_back", "black_front", "repeat", "repeat_front", "repeat_last"]


def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_object_names(value: str):
    if not value:
        return []
    # Prefer comma-separated because COCO classes can contain spaces, e.g. traffic light.
    if "," in value:
        return [x.strip() for x in value.split(",") if x.strip()]
    return [x.strip() for x in value.split() if x.strip()]


def safe_filename(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    name = "".join(keep).strip("_")
    return name or "unnamed"


def pil_resample_lanczos():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def build_grid_slots(ref_paths, total_slots, fill_mode):
    ref_paths = list(ref_paths)
    num_refs = len(ref_paths)
    num_empty = total_slots - num_refs

    if num_refs == 0:
        return [None] * total_slots
    if num_empty <= 0:
        return ref_paths[:total_slots]
    if fill_mode == "black_back":
        return ref_paths + [None] * num_empty
    if fill_mode == "black_front":
        return [None] * num_empty + ref_paths
    if fill_mode == "repeat":
        return ref_paths + [ref_paths[i % num_refs] for i in range(num_empty)]
    if fill_mode == "repeat_front":
        return [ref_paths[i % num_refs] for i in range(num_empty)] + ref_paths
    if fill_mode == "repeat_last":
        return ref_paths + [ref_paths[-1]] * num_empty
    raise ValueError(f"Unsupported grid fill mode: {fill_mode}")


def make_reference_grid_image(ref_paths, save_path, canvas_size=336, fill_mode="black_front", background_color=(0, 0, 0)):
    ref_paths = [Path(p) for p in ref_paths]
    num_refs = len(ref_paths)
    if num_refs == 0:
        raise ValueError("No reference images were provided.")

    grid_side = int(math.ceil(math.sqrt(num_refs)))
    total_slots = grid_side * grid_side
    slot_paths = build_grid_slots(ref_paths, total_slots, fill_mode)

    canvas = Image.new("RGB", (canvas_size, canvas_size), background_color)
    resample = pil_resample_lanczos()

    for slot_idx, ref_path in enumerate(slot_paths):
        if ref_path is None:
            continue
        row, col = divmod(slot_idx, grid_side)
        left = int(round(col * canvas_size / grid_side))
        right = int(round((col + 1) * canvas_size / grid_side))
        top = int(round(row * canvas_size / grid_side))
        bottom = int(round((row + 1) * canvas_size / grid_side))
        slot_w = max(1, right - left)
        slot_h = max(1, bottom - top)

        ref_img = Image.open(ref_path).convert("RGB")
        ref_img.thumbnail((slot_w, slot_h), resample)
        slot = Image.new("RGB", (slot_w, slot_h), background_color)
        paste_x = (slot_w - ref_img.width) // 2
        paste_y = (slot_h - ref_img.height) // 2
        slot.paste(ref_img, (paste_x, paste_y))
        canvas.paste(slot, (left, top))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)

    return canvas, {
        "num_refs": num_refs,
        "grid_side": grid_side,
        "total_slots": total_slots,
        "num_empty_slots": total_slots - num_refs,
        "fill_mode": fill_mode,
        "canvas_size": canvas_size,
        "save_path": str(save_path),
        "slot_paths": [None if p is None else str(p) for p in slot_paths],
    }


def discover_reference_images(ref_folder: Path):
    out = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        out.extend(ref_folder.glob(ext))
    return sorted({p.resolve() for p in out if p.is_file()})


def choose_reference_images(ref_folder: Path, object_names, num_refs: int, seed: int):
    if object_names:
        paths = []
        missing = []
        suffixes = [".png", ".jpg", ".jpeg", ".webp"]
        for name in object_names:
            found = None
            for suffix in suffixes:
                candidate = ref_folder / f"{name}{suffix}"
                if candidate.exists():
                    found = candidate.resolve()
                    break
            if found is None:
                missing.append(name)
            else:
                paths.append(found)
        if missing:
            raise FileNotFoundError(
                "Missing reference image(s): " + ", ".join(missing) + f" under {ref_folder}"
            )
        return paths

    all_refs = discover_reference_images(ref_folder)
    if not all_refs:
        raise FileNotFoundError(f"No reference images found under {ref_folder}")
    if num_refs > len(all_refs):
        raise ValueError(f"num_refs={num_refs} is larger than available refs={len(all_refs)}")

    rng = random.Random(seed)
    return rng.sample(all_refs, num_refs)


def load_tokenizer(model_config, model_name):
    if model_name in ["llava-1.5", "mplug-owl2"]:
        tokenizer_path = "merged_ckpt"
    elif model_name == "minigpt4":
        tokenizer_path = "llama_model"
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return AutoTokenizer.from_pretrained(model_config[tokenizer_path], use_fast=False)


def clean_generated_text(decoded_text, model_name):
    if model_name == "minigpt4":
        return decoded_text.split("###")[0].split("Assistant:")[-1].strip()
    return decoded_text.split("ASSISTANT: ")[-1].strip()


def token_display(tokenizer, token_id: int):
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


def get_lm_head_matrix(model, model_name):
    if model_name == "mplug-owl2":
        return model.model.lm_head.weight
    return model.llama_model.lm_head.weight


def next_token_logits(model, model_name, image, prompt, image_path, prev_tokens, use_nucleus_sampling=False, num_beams=1):
    """Return vocab logits for the next token, matching the custom RVCD-style generation path."""
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
                nvcd_previous_last_ids_list=prev_tokens,
            )

    # The custom LLaVA generate path returns tensors created inside torch.inference_mode().
    # If we multiply an inference tensor with a trainable lm_head outside that context,
    # PyTorch may try to save the inference tensor for backward and raise:
    # "Inference tensors cannot be saved for backward".
    # Clone/detach the hidden state and run the projection under no_grad.
    last_hidden = out["hidden_states"][-1][-1][:, -1, :].detach().clone()
    lm_head = get_lm_head_matrix(model, model_name).detach()
    with torch.no_grad():
        logits = torch.matmul(last_hidden, lm_head.T)
    return logits.detach()


def top_n_from_logits(logits, tokenizer, top_n: int, skip_special_tokens: bool):
    probs = F.softmax(logits.float(), dim=-1)[0]
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)

    results = []
    for prob, token_id in zip(sorted_probs.tolist(), sorted_ids.tolist()):
        token_id = int(token_id)
        if skip_special_tokens and is_special_id(tokenizer, token_id):
            continue
        info = token_display(tokenizer, token_id)
        info["probability"] = float(prob)
        info["log_probability"] = float(math.log(max(float(prob), 1e-45)))
        info["rank"] = len(results) + 1
        results.append(info)
        if len(results) >= top_n:
            break
    return results


def greedy_continue_from_first_token(
    model,
    tokenizer,
    model_name,
    image,
    prompt,
    image_path,
    first_token_id: int,
    total_tokens: int,
    use_nucleus_sampling=False,
    num_beams=1,
):
    """Force the first token, then greedily continue until total_tokens or EOS."""
    device = image.device
    output_tokens = [torch.tensor(int(first_token_id), device=device)]

    for _ in range(max(0, total_tokens - 1)):
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


def shorten_text(text: str, max_chars: int) -> str:
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1] + "…"


def wrap_truncate_text(text: str, width: int = 18, max_lines: int = 3) -> str:
    text = str(text).replace("\n", " ").strip()
    if not text:
        return ""
    wrapped = textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False)
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = shorten_text(wrapped[-1], max(3, width))
    return "\n".join(wrapped)


def save_combined_probe_figure(grid_image_path, topn, continuations, save_path, title):
    import matplotlib.pyplot as plt

    grid_img = Image.open(grid_image_path).convert("RGB")

    labels = []
    values = []
    for item, cont in zip(topn, continuations):
        prefix = item["decoded"].replace("\n", " ").strip()
        if not prefix:
            prefix = item["token"]
        continuation = cont.get("continuation_text", "")
        labels.append({
            "rank": item["rank"],
            "prefix": prefix,
            "continuation": continuation,
        })
        values.append(item["probability"])

    fig_w = max(16, len(labels) * 1.6 + 6)
    fig_h = 6.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, max(2.0, len(labels) * 0.55)], wspace=0.18)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(grid_img)
    ax_img.set_title("Input reference grid", fontsize=12)
    ax_img.axis("off")

    ax_bar = fig.add_subplot(gs[0, 1])
    x = np.arange(len(labels))
    bars = ax_bar.bar(x, values)
    ax_bar.set_ylabel("Probability")
    ax_bar.set_xlabel("Top-N first-token candidates (prefix black, continuation blue)")
    ax_bar.set_title(title, fontsize=12)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([""] * len(labels))
    ax_bar.set_xlim(-0.6, len(labels) - 0.4)

    ymax = max(values) if values else 1.0
    ax_bar.set_ylim(0.0, ymax * 1.18 if ymax > 0 else 1.0)

    for bar, item in zip(bars, labels):
        height = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(ymax * 0.02, 1e-6),
            f"#{item['rank']}\n{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
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
            fontsize=10,
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
                fontsize=9,
                color="blue",
                transform=trans,
                clip_on=False,
            )

    fig.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.34, wspace=0.16)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)

def write_markdown_report(path: Path, result: dict):
    lines = []
    lines.append("# Reference Grid Top-N Probe")
    lines.append("")
    lines.append(f"- model: `{result['model_name']}`")
    lines.append(f"- prompt: `{result['question']}`")
    lines.append(f"- grid image: `{result['grid_meta']['save_path']}`")
    if result.get('combined_figure_path'):
        lines.append(f"- combined figure: `{result['combined_figure_path']}`")
    lines.append(f"- fill mode: `{result['grid_meta']['fill_mode']}`")
    lines.append(f"- sampled objects: {', '.join(result['sampled_object_names'])}")
    lines.append("")
    lines.append("## First-token top-N probabilities")
    lines.append("")
    lines.append("| rank | token id | token | decoded | probability |")
    lines.append("|---:|---:|---|---|---:|")
    for item in result["topn_first_token_probs"]:
        token = str(item["token"]).replace("|", r"\|")
        decoded = str(item["decoded_clean"]).replace("|", r"\|")
        lines.append(
            f"| {item['rank']} | {item['token_id']} | `{token}` | `{decoded}` | {item['probability']:.6f} |"
        )
    lines.append("")
    lines.append("## Forced-first-token continuations")
    lines.append("")
    lines.append("| rank | forced decoded token | generated text |")
    lines.append("|---:|---|---|")
    for item in result["forced_continuations"]:
        rank = item["rank"]
        forced = item["forced_first_token"]["decoded_clean"].replace("|", r"\|")
        text = item["generated_text"].replace("\n", " ").replace("|", r"\|")
        lines.append(f"| {rank} | `{forced}` | {text} |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Probe first-token top-N logits on black_front reference grids.")
    parser.add_argument("--model", type=str, default="llava-1.5", choices=list(MODEL_EVAL_CONFIG_PATH.keys()))
    parser.add_argument("-g", "--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref_folder_path", type=str, default="DB_single_concept_images_flux_generated/generated_images")
    parser.add_argument("--object_names", type=str, default="", help="Comma-separated refs, e.g. 'dog,cat,traffic light'. If empty, random refs are sampled.")
    parser.add_argument("--num_refs", type=int, default=5, help="Number of random reference images if --object_names is empty.")
    parser.add_argument("--grid_fill_mode", type=str, default="black_front", choices=GRID_FILL_MODES)
    parser.add_argument("--grid_canvas_size", type=int, default=336)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--continuation_tokens", type=int, default=10, help="Total generated tokens per forced first-token continuation.")
    parser.add_argument("--question", type=str, default="What object is shown in this image? Answer with one word.")
    parser.add_argument("--output_dir", type=str, default="./refgrid_topn_probe_outputs")
    parser.add_argument("--sample", action="store_true", help="Pass use_nucleus_sampling=True to model.generate. Top-N extraction itself is deterministic.")
    parser.add_argument("--skip_special_tokens", type=str2bool, default=True)
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
    object_names = parse_object_names(args.object_names)
    ref_paths = choose_reference_images(
        ref_folder=ref_folder,
        object_names=object_names,
        num_refs=args.num_refs,
        seed=args.seed,
    )
    sampled_object_names = [p.stem for p in ref_paths]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = (
        f"refgrid_topn_{model_name}_{args.grid_fill_mode}_seed_{args.seed}_"
        f"refs_{len(ref_paths)}_topn_{args.top_n}_{timestamp}"
    )
    out_dir = Path(args.output_dir).expanduser().resolve() / safe_filename(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_path = out_dir / "reference_grid.png"
    grid_canvas, grid_meta = make_reference_grid_image(
        ref_paths=ref_paths,
        save_path=grid_path,
        canvas_size=args.grid_canvas_size,
        fill_mode=args.grid_fill_mode,
    )

    image = norm(process_pil_before_norm(grid_canvas))
    template = INSTRUCTION_TEMPLATE[model_name]
    prompt = template.replace("<question>", args.question)

    print("=" * 80)
    print(f"Sampled objects: {sampled_object_names}")
    print(f"Grid saved to   : {grid_path}")
    print(f"Question        : {args.question}")
    print("=" * 80)

    first_logits = next_token_logits(
        model=model,
        model_name=model_name,
        image=image,
        prompt=prompt,
        image_path=grid_path,
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

    continuations = []
    for item in topn:
        cont = greedy_continue_from_first_token(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            image=image,
            prompt=prompt,
            image_path=grid_path,
            first_token_id=item["token_id"],
            total_tokens=args.continuation_tokens,
            use_nucleus_sampling=args.sample,
            num_beams=1,
        )
        cont["rank"] = item["rank"]
        cont["first_token_probability"] = item["probability"]
        continuations.append(cont)

    combined_figure_path = out_dir / "combined_probe_figure.png"
    save_combined_probe_figure(
        grid_image_path=grid_path,
        topn=topn,
        continuations=continuations,
        save_path=combined_figure_path,
        title=f"Top-{args.top_n} first-token probabilities on {args.grid_fill_mode} reference grid",
    )

    result = {
        "model_name": model_name,
        "seed": args.seed,
        "ref_folder_path": str(ref_folder),
        "sampled_object_names": sampled_object_names,
        "sampled_ref_paths": [str(p) for p in ref_paths],
        "question": args.question,
        "prompt": prompt,
        "top_n": args.top_n,
        "continuation_tokens": args.continuation_tokens,
        "skip_special_tokens": args.skip_special_tokens,
        "grid_meta": grid_meta,
        "topn_first_token_probs": topn,
        "forced_continuations": continuations,
        "combined_figure_path": str(combined_figure_path),
    }

    json_path = out_dir / "result.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(md_path, result)

    print("Top-N first-token probabilities")
    for item in topn:
        print(
            f"{item['rank']:>2}. id={item['token_id']:<6} "
            f"token={item['token']!r:<16} decoded={item['decoded']!r:<12} "
            f"prob={item['probability']:.6f}"
        )

    print("\nForced-first-token continuations")
    for cont in continuations:
        forced = cont["forced_first_token"]["decoded"]
        print(f"{cont['rank']:>2}. first={forced!r:<12} -> {cont['generated_text']}")

    print("=" * 80)
    print(f"Saved JSON report : {json_path}")
    print(f"Saved MD report   : {md_path}")
    print(f"Saved combined figure : {combined_figure_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()


# Example:
# CUDA_VISIBLE_DEVICES=0 \
# python refgrid_topn_probe.py \
#   --model llava-1.5 \
#   --ref_folder_path DB_single_concept_images_flux_generated/generated_images \
#   --num_refs 5 \
#   --grid_fill_mode black_front \
#   --top_n 10 \
#   --continuation_tokens 10 \
#   --seed 42 \
#   --output_dir ./refgrid_topn_probe_outputs
#
# With explicit object names:
# CUDA_VISIBLE_DEVICES=0 \
# python refgrid_topn_probe.py \
#   --model llava-1.5 \
#   --ref_folder_path DB_single_concept_images_flux_generated/generated_images \
#   --object_names "dog,cat,traffic light,bicycle,dining table" \
#   --grid_fill_mode black_front \
#   --top_n 10 \
#   --continuation_tokens 10 \
#   --seed 42
