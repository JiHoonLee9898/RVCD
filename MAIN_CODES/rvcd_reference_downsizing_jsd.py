#!/usr/bin/env python3
"""
Measure first-token distribution stability of RVCD single-concept images
under spatial downsizing with white-background padding.

This script is designed to be placed in, or executed from, RVCD/MAIN_CODES.
It follows the repository's LLaVA-1.5 loading and image preprocessing path:

  1. Sample N images from the RVCD reference-image database.
  2. Convert each source image to a 336x336 white-background base canvas.
  3. Downsize that full base image by factors such as 1.5, 2.0, ..., 4.0.
  4. Center the downsized image on a white 336x336 canvas.
  5. Run LLaVA-1.5 with the same one-word prompt at every scale.
  6. Extract the *actual vocabulary logits* at the first answer-token position.
  7. Compute JSD against the scale=1.0 base distribution.
  8. Save per-image CSV results, summary statistics, and plots.

Example (run from RVCD/MAIN_CODES):

python rvcd_reference_downsizing_jsd.py \
  --ref-folder-path ./DB_single_concept_images_flux_generated/generated_images \
  --cfg-path ./eval_configs/llava-1.5_eval.yaml \
  --num-samples 50 \
  --gpu-id 0 \
  --output-dir ./reference_downsizing_results

To override the model path written in llava-1.5_eval.yaml:

python rvcd_reference_downsizing_jsd.py \
  --ref-folder-path ./DB_single_concept_images_flux_generated/generated_images \
  --model-path /absolute/path/to/llava-v1.5-7b

Notes:
- LLaVA still receives a 336x336 tensor and therefore still uses 576 visual tokens.
- The manipulation reduces the reference's spatial occupancy and effective detail;
  it does not reduce LLaVA's visual-token count or inference FLOPs.
- JSD is reported both in nats and bits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm


IMAGE_TOKEN_INDEX = -200
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    LANCZOS = Image.LANCZOS


class ChannelNormalize:
    """Torchvision-free channel normalization for CHW or BCHW tensors."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        if len(mean) != len(std):
            raise ValueError("mean and std must have the same length")
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            shape = (-1, 1, 1)
        elif tensor.ndim == 4:
            shape = (1, -1, 1, 1)
        else:
            raise ValueError(f"Expected CHW or BCHW image tensor, got shape {tuple(tensor.shape)}")
        mean = torch.as_tensor(self.mean, device=tensor.device, dtype=tensor.dtype).view(*shape)
        std = torch.as_tensor(self.std, device=tensor.device, dtype=tensor.dtype).view(*shape)
        return (tensor - mean) / std


PER_IMAGE_FIELDS = [
    "image_index",
    "image_path",
    "image_name",
    "original_width",
    "original_height",
    "scale",
    "inner_width",
    "inner_height",
    "jsd_nats",
    "jsd_bits",
    "base_top1_id",
    "base_top1_token",
    "base_top1_text",
    "scaled_top1_id",
    "scaled_top1_token",
    "scaled_top1_text",
    "top1_agreement",
    "base_top1_probability",
    "scaled_probability_of_base_top1",
    "base_top1_probability_retention",
    "topk",
    "topk_overlap_ratio",
]

SUMMARY_FIELDS = [
    "scale",
    "inner_width",
    "inner_height",
    "count",
    "mean_jsd_nats",
    "std_jsd_nats",
    "median_jsd_nats",
    "min_jsd_nats",
    "max_jsd_nats",
    "ci95_jsd_nats",
    "mean_jsd_bits",
    "top1_agreement_rate",
    "mean_base_top1_probability_retention",
    "mean_topk_overlap_ratio",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LLaVA-1.5 first-token logit stability for RVCD reference "
            "images downsized and centered on a white 336x336 canvas."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ref-folder-path",
        type=str,
        default="./DB_single_concept_images_flux_generated/generated_images",
        help="Folder containing RVCD single-concept reference images.",
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="./eval_configs/llava-1.5_eval.yaml",
        help="RVCD LLaVA-1.5 evaluation YAML.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional override for model.merged_ckpt in the YAML.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llava-1.5",
        choices=["llava-1.5"],
        help="This experiment currently targets the RVCD LLaVA-1.5 wrapper.",
    )
    parser.add_argument("-g", "--gpu-id", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=50,
        help="Number of images to sample. Use 0 or a negative value for all images.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        help="Downsizing factors. Scale 1.0 is always added as the base condition.",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=336,
        help="Square white canvas size supplied to the RVCD image processor.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Describe the main object in this image using exactly one English word."
        ),
        help="Text inserted after <ImageHere> in the LLaVA conversation prompt.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="K used for top-k vocabulary-set overlap.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help=(
            "Number of scale variants inferred together. Keep 1 for maximum "
            "compatibility; increase if the local RVCD build has enough VRAM."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reference_downsizing_results",
        help="Directory for CSV, JSON, plots, and optional probe images.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search reference images recursively.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse the saved sample list and skip fully completed images.",
    )
    parser.add_argument(
        "--save-probes",
        action="store_true",
        help="Save every generated white-padded scale variant for inspection.",
    )
    parser.add_argument(
        "--strict-source-size",
        action="store_true",
        help="Reject source images that are not exactly canvas_size x canvas_size.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first unreadable image or inference failure.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        default=None,
        help="Compatibility option consumed by RVCD's Config class.",
    )
    return parser


def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def normalize_scales(scales: Sequence[float]) -> List[float]:
    cleaned: List[float] = []
    for scale in scales:
        value = float(scale)
        if not math.isfinite(value) or value < 1.0:
            raise ValueError(f"Every scale must be finite and >= 1.0; received {scale!r}.")
        if not any(math.isclose(value, old, rel_tol=0.0, abs_tol=1e-9) for old in cleaned):
            cleaned.append(value)
    if not any(math.isclose(1.0, old, rel_tol=0.0, abs_tol=1e-9) for old in cleaned):
        cleaned.append(1.0)
    return sorted(cleaned)


def discover_images(folder: Path, recursive: bool) -> List[Path]:
    iterator: Iterable[Path] = folder.rglob("*") if recursive else folder.glob("*")
    paths = [
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(paths, key=lambda item: str(item).lower())


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def choose_images(
    all_images: Sequence[Path],
    num_samples: int,
    seed: int,
    sample_file: Path,
    ref_root: Path,
    resume: bool,
) -> List[Path]:
    if resume and sample_file.exists():
        saved: List[Path] = []
        for raw_line in sample_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = ref_root / candidate
            candidate = candidate.resolve()
            if candidate.exists():
                saved.append(candidate)
            else:
                print(f"[warning] Saved sample no longer exists: {candidate}", file=sys.stderr)
        if saved:
            return saved

    if num_samples <= 0 or num_samples >= len(all_images):
        selected = list(all_images)
    else:
        rng = random.Random(seed)
        selected = rng.sample(list(all_images), num_samples)
        selected.sort(key=lambda item: str(item).lower())

    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text(
        "\n".join(relative_or_absolute(path, ref_root) for path in selected) + "\n",
        encoding="utf-8",
    )
    return selected


def fit_to_white_canvas(image: Image.Image, canvas_size: int) -> Image.Image:
    """Fit a source image inside a square white canvas while preserving aspect ratio."""
    image = image.convert("RGB")
    if image.size == (canvas_size, canvas_size):
        return image.copy()

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid source image size: {image.size}")

    ratio = min(canvas_size / width, canvas_size / height)
    new_width = max(1, min(canvas_size, round(width * ratio)))
    new_height = max(1, min(canvas_size, round(height * ratio)))
    resized = image.resize((new_width, new_height), resample=LANCZOS)

    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    left = (canvas_size - new_width) // 2
    top = (canvas_size - new_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def downsize_and_white_pad(
    base_canvas: Image.Image,
    scale: float,
    canvas_size: int,
) -> Tuple[Image.Image, int, int]:
    """Downsize a full base canvas, then center it on a white square canvas."""
    if scale < 1.0:
        raise ValueError("scale must be >= 1.0")
    if base_canvas.size != (canvas_size, canvas_size):
        raise ValueError(
            f"base_canvas must be {canvas_size}x{canvas_size}; got {base_canvas.size}."
        )

    inner_size = max(1, min(canvas_size, round(canvas_size / scale)))
    if math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-9):
        return base_canvas.copy(), canvas_size, canvas_size

    resized = base_canvas.resize((inner_size, inner_size), resample=LANCZOS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    left = (canvas_size - inner_size) // 2
    top = (canvas_size - inner_size) // 2
    canvas.paste(resized, (left, top))
    return canvas, inner_size, inner_size


def prompt_with_image_placeholder(user_prompt: str) -> str:
    if "<ImageHere>" in user_prompt:
        return user_prompt
    return f"USER: <ImageHere> {user_prompt.strip()} ASSISTANT:"


def resolve_cfg_path(cfg_path: str, script_dir: Path) -> Path:
    candidate = Path(cfg_path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    candidate_from_script = script_dir / candidate
    if candidate_from_script.exists():
        return candidate_from_script.resolve()
    raise FileNotFoundError(f"Could not find cfg file: {cfg_path}")


def load_rvcd_llava(
    args: argparse.Namespace,
    device: torch.device,
    script_dir: Path,
) -> Tuple[Any, Any, "ChannelNormalize", Any, Any]:
    """Load the same LLaVA-1.5 wrapper and preprocessing path used by RVCD."""
    for path in (script_dir, script_dir.parent):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.models import load_preprocess

    # Importing the package registers LLaVA and the processors in RVCD's registry.
    import minigpt4.models  # noqa: F401
    import minigpt4.processors  # noqa: F401

    args.cfg_path = str(resolve_cfg_path(args.cfg_path, script_dir))
    cfg = Config(args)
    model_config = cfg.model_cfg

    if args.model_path:
        model_config.merged_ckpt = str(Path(args.model_path).expanduser().resolve())

    # Kept for compatibility with the repository's existing generation scripts.
    model_config.device_8bit = args.gpu_id

    model_cls = registry.get_model_class(model_config.arch)
    if model_cls is None:
        raise RuntimeError(f"RVCD registry has no model class for arch={model_config.arch!r}.")

    print(f"Loading RVCD model arch={model_config.arch} from {model_config.merged_ckpt}")
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    processor_cfg = cfg.get_config().preprocess
    processor_cfg.vis_processor.eval.do_normalize = False
    vis_processors, _ = load_preprocess(processor_cfg)
    vis_processor = vis_processors["eval"]
    normalizer = ChannelNormalize(CLIP_MEAN, CLIP_STD)

    tokenizer = model.llama_tokenizer
    tokenizer.padding_side = "left"
    return model, tokenizer, normalizer, vis_processor, cfg


class LlavaFirstTokenProbe:
    """Directly obtains first-answer-token vocabulary logits from RVCD LLaVA."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        vis_processor: Any,
        normalizer: "ChannelNormalize",
        device: torch.device,
        prompt: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.vis_processor = vis_processor
        self.normalizer = normalizer
        self.device = device
        self.prompt = prompt_with_image_placeholder(prompt)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        tensor = self.vis_processor(image.convert("RGB")).unsqueeze(0)
        tensor = self.normalizer(tensor)
        return tensor.to(self.device)

    def build_input_ids(self, batch_size: int) -> torch.Tensor:
        instruction = self.model.system_message + self.prompt
        if instruction.count("<ImageHere>") != 1:
            raise ValueError(
                "The final prompt must contain exactly one <ImageHere> placeholder; "
                f"received: {instruction!r}"
            )
        chunk_before, chunk_after = instruction.split("<ImageHere>")

        before_ids = self.tokenizer(
            [chunk_before] * batch_size,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).input_ids.to(self.device)
        after_ids = self.tokenizer(
            [chunk_after] * batch_size,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).input_ids.to(self.device)

        bos = torch.full(
            (batch_size, 1),
            int(self.tokenizer.bos_token_id),
            dtype=torch.long,
            device=self.device,
        )
        image_token = torch.full(
            (batch_size, 1),
            IMAGE_TOKEN_INDEX,
            dtype=torch.long,
            device=self.device,
        )
        return torch.cat([bos, before_ids, image_token, after_ids], dim=1)

    @torch.inference_mode()
    def logits_for_images(
        self,
        images: Sequence[Image.Image],
        inference_batch_size: int,
    ) -> torch.Tensor:
        if not images:
            raise ValueError("At least one image is required.")
        if inference_batch_size < 1:
            raise ValueError("inference_batch_size must be >= 1.")

        output_batches: List[torch.Tensor] = []
        for start in range(0, len(images), inference_batch_size):
            chunk = images[start : start + inference_batch_size]
            image_batch = torch.cat([self.preprocess_image(image) for image in chunk], dim=0)
            input_ids = self.build_input_ids(len(chunk))

            with self.model.maybe_autocast():
                outputs = self.model.llama_model(
                    input_ids=input_ids,
                    images=image_batch,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

            # The final prompt position predicts the first assistant answer token.
            first_token_logits = outputs.logits[:, -1, :].float().cpu()
            output_batches.append(first_token_logits)

            del outputs, image_batch, input_ids

        return torch.cat(output_batches, dim=0)


def js_divergence_from_logits(
    base_logits: torch.Tensor,
    other_logits: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """Return Jensen-Shannon divergence in nats and bits."""
    base_log_prob = F.log_softmax(base_logits.float(), dim=-1)
    other_log_prob = F.log_softmax(other_logits.float(), dim=-1)
    base_prob = base_log_prob.exp()
    other_prob = other_log_prob.exp()
    mixture = 0.5 * (base_prob + other_prob)
    log_mixture = mixture.clamp_min(eps).log()

    kl_base = torch.sum(base_prob * (base_log_prob - log_mixture), dim=-1)
    kl_other = torch.sum(other_prob * (other_log_prob - log_mixture), dim=-1)
    jsd_nats = 0.5 * (kl_base + kl_other)
    jsd_nats_value = float(jsd_nats.item())
    return jsd_nats_value, jsd_nats_value / math.log(2.0)


def token_descriptions(tokenizer: Any, token_id: int) -> Tuple[str, str]:
    token = str(tokenizer.convert_ids_to_tokens(int(token_id)))
    text = str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    return token, text


def topk_overlap_ratio(base_logits: torch.Tensor, other_logits: torch.Tensor, k: int) -> float:
    vocab_size = int(base_logits.shape[-1])
    actual_k = max(1, min(int(k), vocab_size))
    base_ids = set(torch.topk(base_logits, k=actual_k, dim=-1).indices.tolist())
    other_ids = set(torch.topk(other_logits, k=actual_k, dim=-1).indices.tolist())
    return len(base_ids.intersection(other_ids)) / actual_k


def rows_for_image(
    image_index: int,
    image_path: Path,
    original_size: Tuple[int, int],
    scales: Sequence[float],
    inner_sizes: Mapping[float, Tuple[int, int]],
    logits_by_scale: Mapping[float, torch.Tensor],
    tokenizer: Any,
    topk: int,
) -> List[Dict[str, Any]]:
    base_scale = 1.0
    base_logits = logits_by_scale[base_scale]
    base_prob = F.softmax(base_logits.float(), dim=-1)
    base_top1_id = int(torch.argmax(base_logits, dim=-1).item())
    base_top1_probability = float(base_prob[base_top1_id].item())
    base_top1_token, base_top1_text = token_descriptions(tokenizer, base_top1_id)

    rows: List[Dict[str, Any]] = []
    for scale in scales:
        current_logits = logits_by_scale[scale]
        current_prob = F.softmax(current_logits.float(), dim=-1)
        current_top1_id = int(torch.argmax(current_logits, dim=-1).item())
        current_top1_token, current_top1_text = token_descriptions(tokenizer, current_top1_id)

        if math.isclose(scale, base_scale, rel_tol=0.0, abs_tol=1e-9):
            jsd_nats, jsd_bits = 0.0, 0.0
        else:
            jsd_nats, jsd_bits = js_divergence_from_logits(base_logits, current_logits)

        scaled_probability_of_base = float(current_prob[base_top1_id].item())
        retention = (
            scaled_probability_of_base / base_top1_probability
            if base_top1_probability > 0.0
            else float("nan")
        )
        inner_width, inner_height = inner_sizes[scale]

        rows.append(
            {
                "image_index": image_index,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "original_width": original_size[0],
                "original_height": original_size[1],
                "scale": float(scale),
                "inner_width": inner_width,
                "inner_height": inner_height,
                "jsd_nats": jsd_nats,
                "jsd_bits": jsd_bits,
                "base_top1_id": base_top1_id,
                "base_top1_token": base_top1_token,
                "base_top1_text": base_top1_text,
                "scaled_top1_id": current_top1_id,
                "scaled_top1_token": current_top1_token,
                "scaled_top1_text": current_top1_text,
                "top1_agreement": int(current_top1_id == base_top1_id),
                "base_top1_probability": base_top1_probability,
                "scaled_probability_of_base_top1": scaled_probability_of_base,
                "base_top1_probability_retention": retention,
                "topk": int(topk),
                "topk_overlap_ratio": topk_overlap_ratio(base_logits, current_logits, topk),
            }
        )
    return rows


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def completed_image_paths(
    rows: Sequence[Mapping[str, Any]],
    expected_scales: Sequence[float],
) -> set[str]:
    scales_by_image: MutableMapping[str, set[float]] = defaultdict(set)
    for row in rows:
        try:
            scales_by_image[str(row["image_path"])].add(float(row["scale"]))
        except (KeyError, TypeError, ValueError):
            continue

    completed: set[str] = set()
    for image_path, existing_scales in scales_by_image.items():
        if all(
            any(math.isclose(scale, value, rel_tol=0.0, abs_tol=1e-8) for value in existing_scales)
            for scale in expected_scales
        ):
            completed.add(image_path)
    return completed


def numeric_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    float_fields = {
        "scale",
        "jsd_nats",
        "jsd_bits",
        "base_top1_probability",
        "scaled_probability_of_base_top1",
        "base_top1_probability_retention",
        "topk_overlap_ratio",
    }
    int_fields = {
        "image_index",
        "original_width",
        "original_height",
        "inner_width",
        "inner_height",
        "base_top1_id",
        "scaled_top1_id",
        "top1_agreement",
        "topk",
    }
    for source in rows:
        row = dict(source)
        for field in float_fields:
            if field in row:
                try:
                    row[field] = float(row[field])
                except (TypeError, ValueError):
                    row[field] = float("nan")
        for field in int_fields:
            if field in row:
                try:
                    row[field] = int(float(row[field]))
                except (TypeError, ValueError):
                    row[field] = 0
        converted.append(row)
    return converted


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: MutableMapping[float, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["scale"])].append(row)

    summaries: List[Dict[str, Any]] = []
    for scale in sorted(grouped):
        group = grouped[scale]
        jsd_nats = np.asarray([float(row["jsd_nats"]) for row in group], dtype=np.float64)
        jsd_bits = np.asarray([float(row["jsd_bits"]) for row in group], dtype=np.float64)
        agreements = np.asarray([float(row["top1_agreement"]) for row in group], dtype=np.float64)
        retentions = np.asarray(
            [float(row["base_top1_probability_retention"]) for row in group],
            dtype=np.float64,
        )
        overlaps = np.asarray([float(row["topk_overlap_ratio"]) for row in group], dtype=np.float64)
        count = int(jsd_nats.size)
        std = float(np.std(jsd_nats, ddof=1)) if count > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(count) if count > 0 else float("nan")

        summaries.append(
            {
                "scale": scale,
                "inner_width": int(group[0]["inner_width"]),
                "inner_height": int(group[0]["inner_height"]),
                "count": count,
                "mean_jsd_nats": float(np.mean(jsd_nats)),
                "std_jsd_nats": std,
                "median_jsd_nats": float(np.median(jsd_nats)),
                "min_jsd_nats": float(np.min(jsd_nats)),
                "max_jsd_nats": float(np.max(jsd_nats)),
                "ci95_jsd_nats": ci95,
                "mean_jsd_bits": float(np.mean(jsd_bits)),
                "top1_agreement_rate": float(np.mean(agreements)),
                "mean_base_top1_probability_retention": float(np.nanmean(retentions)),
                "mean_topk_overlap_ratio": float(np.mean(overlaps)),
            }
        )
    return summaries


def make_plots(
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped_by_image: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_by_image[str(row["image_path"])].append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    show_labels = len(grouped_by_image) <= 15
    for image_path, image_rows in sorted(grouped_by_image.items()):
        ordered = sorted(image_rows, key=lambda item: float(item["scale"]))
        label = Path(image_path).name if show_labels else None
        ax.plot(
            [float(row["scale"]) for row in ordered],
            [float(row["jsd_nats"]) for row in ordered],
            marker="o",
            linewidth=1.0,
            markersize=2.5,
            alpha=0.45,
            label=label,
        )
    ax.set_xlabel("Downsizing factor")
    ax.set_ylabel("Jensen-Shannon divergence (nats)")
    ax.set_title(f"Per-image first-token distribution shift (N={len(grouped_by_image)})")
    ax.grid(True, alpha=0.25)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    if show_labels:
        ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "per_image_jsd_curves.png", dpi=220)
    fig.savefig(output_dir / "per_image_jsd_curves.pdf")
    plt.close(fig)

    ordered_summary = sorted(summaries, key=lambda item: float(item["scale"]))
    x = np.asarray([float(row["scale"]) for row in ordered_summary], dtype=np.float64)
    mean = np.asarray([float(row["mean_jsd_nats"]) for row in ordered_summary], dtype=np.float64)
    ci95 = np.asarray([float(row["ci95_jsd_nats"]) for row in ordered_summary], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(x, mean, marker="o", linewidth=2.0, label="Mean JSD")
    ax.fill_between(x, np.maximum(0.0, mean - ci95), mean + ci95, alpha=0.2, label="95% CI")
    ax.set_xlabel("Downsizing factor")
    ax.set_ylabel("Jensen-Shannon divergence (nats)")
    ax.set_title("Mean first-token distribution shift")
    ax.grid(True, alpha=0.25)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_jsd_curve.png", dpi=220)
    fig.savefig(output_dir / "mean_jsd_curve.pdf")
    plt.close(fig)

    agreement = np.asarray(
        [float(row["top1_agreement_rate"]) for row in ordered_summary], dtype=np.float64
    )
    retention = np.asarray(
        [float(row["mean_base_top1_probability_retention"]) for row in ordered_summary],
        dtype=np.float64,
    )
    overlap = np.asarray(
        [float(row["mean_topk_overlap_ratio"]) for row in ordered_summary], dtype=np.float64
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(x, agreement, marker="o", label="Top-1 agreement")
    ax.plot(x, retention, marker="s", label="Base-token probability retention")
    ax.plot(x, overlap, marker="^", label="Top-k overlap")
    ax.set_xlabel("Downsizing factor")
    ax.set_ylabel("Ratio")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Complementary first-token stability metrics")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "complementary_stability_metrics.png", dpi=220)
    fig.savefig(output_dir / "complementary_stability_metrics.pdf")
    plt.close(fig)


def save_probe_image(
    probe: Image.Image,
    output_dir: Path,
    image_index: int,
    image_path: Path,
    scale: float,
) -> None:
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:8]
    folder = output_dir / "probe_images" / f"{image_index:04d}_{image_path.stem}_{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    probe.save(folder / f"scale_{scale:g}.png")


def git_commit(script_dir: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=script_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_run_config(
    args: argparse.Namespace,
    output_dir: Path,
    selected_images: Sequence[Path],
    scales: Sequence[float],
    script_dir: Path,
) -> None:
    config = vars(args).copy()
    config.update(
        {
            "normalized_scales": list(scales),
            "selected_image_count": len(selected_images),
            "rvcd_git_commit": git_commit(script_dir),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "jsd_log_base": {"jsd_nats": "e", "jsd_bits": "2"},
            "manipulation": "full-image downsizing + centered white padding",
            "created_unix_time": time.time(),
        }
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def append_failure(output_dir: Path, image_path: Path, exc: BaseException) -> None:
    record = {
        "image_path": str(image_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "time": time.time(),
    }
    with (output_dir / "failures.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    scales = normalize_scales(args.scales)

    if args.canvas_size <= 0:
        parser.error("--canvas-size must be positive.")
    if args.topk <= 0:
        parser.error("--topk must be positive.")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive.")

    script_dir = Path(__file__).resolve().parent
    ref_root = Path(args.ref_folder_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.resume:
        stale_files = [
            "per_image_jsd.csv",
            "summary_statistics.csv",
            "failures.jsonl",
            "per_image_jsd_curves.png",
            "per_image_jsd_curves.pdf",
            "mean_jsd_curve.png",
            "mean_jsd_curve.pdf",
            "complementary_stability_metrics.png",
            "complementary_stability_metrics.pdf",
        ]
        for filename in stale_files:
            stale_path = output_dir / filename
            if stale_path.exists():
                stale_path.unlink()

    if not ref_root.is_dir():
        raise FileNotFoundError(f"Reference folder does not exist: {ref_root}")

    all_images = discover_images(ref_root, args.recursive)
    if not all_images:
        raise RuntimeError(f"No supported image files found under {ref_root}")

    sample_file = output_dir / "sampled_images.txt"
    selected_images = choose_images(
        all_images=all_images,
        num_samples=args.num_samples,
        seed=args.seed,
        sample_file=sample_file,
        ref_root=ref_root,
        resume=args.resume,
    )
    print(f"Discovered {len(all_images)} images; selected {len(selected_images)}.")
    print(f"Scales: {scales}")

    setup_seeds(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Loading LLaVA-1.5 7B on CPU is not supported by "
            "this experiment script."
        )
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"--gpu-id {args.gpu_id} is invalid; visible CUDA device count is "
            f"{torch.cuda.device_count()}."
        )
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    write_run_config(args, output_dir, selected_images, scales, script_dir)
    model, tokenizer, normalizer, vis_processor, _ = load_rvcd_llava(
        args=args,
        device=device,
        script_dir=script_dir,
    )
    probe_runner = LlavaFirstTokenProbe(
        model=model,
        tokenizer=tokenizer,
        vis_processor=vis_processor,
        normalizer=normalizer,
        device=device,
        prompt=args.prompt,
    )

    per_image_csv = output_dir / "per_image_jsd.csv"
    existing_rows = read_csv_rows(per_image_csv) if args.resume else []
    completed = completed_image_paths(existing_rows, scales)
    all_rows: List[Mapping[str, Any]] = list(existing_rows)

    progress = tqdm(enumerate(selected_images), total=len(selected_images), desc="Reference images")
    for image_index, image_path in progress:
        canonical_path = str(image_path)
        if canonical_path in completed:
            progress.set_postfix_str(f"skip {image_path.name}")
            continue

        try:
            with Image.open(image_path) as opened:
                source = opened.convert("RGB")
                original_size = source.size
                if args.strict_source_size and source.size != (args.canvas_size, args.canvas_size):
                    raise ValueError(
                        f"Expected {args.canvas_size}x{args.canvas_size}, got {source.size}."
                    )
                base_canvas = fit_to_white_canvas(source, args.canvas_size)

            probe_images: List[Image.Image] = []
            inner_sizes: Dict[float, Tuple[int, int]] = {}
            for scale in scales:
                probe_image, inner_width, inner_height = downsize_and_white_pad(
                    base_canvas=base_canvas,
                    scale=scale,
                    canvas_size=args.canvas_size,
                )
                probe_images.append(probe_image)
                inner_sizes[scale] = (inner_width, inner_height)
                if args.save_probes:
                    save_probe_image(
                        probe_image,
                        output_dir,
                        image_index,
                        image_path,
                        scale,
                    )

            logits = probe_runner.logits_for_images(
                probe_images,
                inference_batch_size=args.inference_batch_size,
            )
            logits_by_scale = {scale: logits[index] for index, scale in enumerate(scales)}
            new_rows = rows_for_image(
                image_index=image_index,
                image_path=image_path,
                original_size=original_size,
                scales=scales,
                inner_sizes=inner_sizes,
                logits_by_scale=logits_by_scale,
                tokenizer=tokenizer,
                topk=args.topk,
            )

            # Replace any partial rows for this image, then checkpoint immediately.
            all_rows = [row for row in all_rows if str(row.get("image_path", "")) != canonical_path]
            all_rows.extend(new_rows)
            all_rows = sorted(
                all_rows,
                key=lambda row: (int(float(row["image_index"])), float(row["scale"])),
            )
            atomic_write_csv(per_image_csv, all_rows, PER_IMAGE_FIELDS)
            completed.add(canonical_path)

            max_scale_row = max(new_rows, key=lambda row: float(row["scale"]))
            progress.set_postfix(
                image=image_path.name,
                jsd_max=f"{float(max_scale_row['jsd_nats']):.3e}",
                top1=max_scale_row["scaled_top1_text"].strip(),
            )

            del logits, logits_by_scale, probe_images

        except Exception as exc:  # Continue long experiments while recording failures.
            append_failure(output_dir, image_path, exc)
            print(f"\n[error] {image_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise

    final_rows = numeric_rows(read_csv_rows(per_image_csv))
    if not final_rows:
        raise RuntimeError("No successful image results were produced.")

    summaries = summarize_rows(final_rows)
    atomic_write_csv(output_dir / "summary_statistics.csv", summaries, SUMMARY_FIELDS)
    make_plots(final_rows, summaries, output_dir)

    print("\nExperiment complete.")
    print(f"Per-image results: {per_image_csv}")
    print(f"Summary:           {output_dir / 'summary_statistics.csv'}")
    print(f"Curves:            {output_dir / 'per_image_jsd_curves.png'}")
    print(f"Mean curve:        {output_dir / 'mean_jsd_curve.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# python rvcd_reference_downsizing_jsd.py \
#   --ref-folder-path ./DB_single_concept_images_flux_generated/generated_images \
#   --cfg-path ./eval_configs/llava-1.5_eval.yaml \
#   --num-samples 50 \
#   --gpu-id 0 \
#   --output-dir ./reference_downsizing_results