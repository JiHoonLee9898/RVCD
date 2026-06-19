#!/usr/bin/env python3
"""
RVCD 90%-nucleus stability analysis under white-padding downsizing.

For each reference image:
1. Run RVCD/LLaVA-1.5 on the original image and obtain first-token logits.
2. Convert logits to probabilities.
3. Define S_p as the smallest token set whose cumulative probability mass
   reaches at least threshold p (default p = 0.90).
4. For each downsized white-padded version of the same image, compute:

   (A) Original nucleus mass retention:
       Sum of downsized probabilities assigned to the original S_p tokens.

   (B) Original nucleus set overlap:
       Overlap between original S_p and the downsized top-k_p token set,
       where k_p is the original nucleus size for that image.

This script generates one compact main figure:

1) nucleus90_mass_retention_compact.png

Each figure contains:
- one faint line per image
- median curve
- mean curve
- 10-90 percentile band
- summary box at the final downsizing factor
- area-retained labels on the x-axis

Also saves:
- per_image_scale_metrics.csv
- per_scale_summary.csv
- summary_metrics.json
- successful_images.txt
- failures.json (if any failures occur)

Example
-------
CUDA_VISIBLE_DEVICES=1 python rvcd_nucleus90_retention_and_overlap.py \
  --ref-folder-path ./DB_single_concept_images_flux_generated/generated_images \
  --cfg-path ./eval_configs/llava-1.5_eval.yaml \
  --num-samples 0 \
  --scales 1 1.5 2 2.5 3 \
  --mass-threshold 0.90 \
  --gpu-id 0 \
  --inference-batch-size 1 \
  --curve-alpha 0.10 \
  --curve-line-width 0.45 \
  --output-dir ./rvcd_nucleus90_retention_and_overlap
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
}
IMAGE_TOKEN_INDEX = -200
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ChannelNormalize:
    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            shape = (-1, 1, 1)
        elif tensor.ndim == 4:
            shape = (1, -1, 1, 1)
        else:
            raise ValueError(f"Unexpected tensor shape: {tuple(tensor.shape)}")

        mean = torch.as_tensor(
            self.mean,
            dtype=tensor.dtype,
            device=tensor.device,
        ).view(*shape)
        std = torch.as_tensor(
            self.std,
            dtype=tensor.dtype,
            device=tensor.device,
        ).view(*shape)
        return (tensor - mean) / std


def parse_float_list(values: Sequence[str]) -> List[float]:
    result: List[float] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                result.append(float(item))
    return result


def normalize_scales(values: Sequence[float]) -> List[float]:
    scales = sorted(set(float(v) for v in values))
    if not scales:
        raise ValueError("At least one scale is required.")
    if any(scale < 1.0 for scale in scales):
        raise ValueError("Every downsizing factor must be >= 1.")
    if not any(math.isclose(scale, 1.0, abs_tol=1e-9) for scale in scales):
        scales.insert(0, 1.0)
    return sorted(set(scales))


def area_percentage(scale: float) -> float:
    return 100.0 / (float(scale) ** 2)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_images(root: Path, recursive: bool) -> List[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def choose_images(
    paths: Sequence[Path],
    num_samples: int,
    seed: int,
) -> List[Path]:
    if num_samples <= 0 or num_samples >= len(paths):
        return list(paths)
    chosen = list(paths)
    random.Random(seed).shuffle(chosen)
    return sorted(chosen[:num_samples])


def fit_to_white_canvas(
    image: Image.Image,
    canvas_size: int,
) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size

    if width == canvas_size and height == canvas_size:
        return image.copy()

    ratio = min(canvas_size / width, canvas_size / height)
    resized_width = max(1, int(round(width * ratio)))
    resized_height = max(1, int(round(height * ratio)))

    resized = image.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new(
        "RGB",
        (canvas_size, canvas_size),
        color=(255, 255, 255),
    )
    left = (canvas_size - resized_width) // 2
    top = (canvas_size - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def downsize_and_white_pad(
    original_canvas: Image.Image,
    scale: float,
    canvas_size: int,
) -> Tuple[Image.Image, int]:
    if math.isclose(scale, 1.0, abs_tol=1e-9):
        return original_canvas.copy(), canvas_size

    inner_size = max(1, int(round(canvas_size / scale)))
    resized = original_canvas.resize(
        (inner_size, inner_size),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (canvas_size, canvas_size),
        color=(255, 255, 255),
    )
    left = (canvas_size - inner_size) // 2
    top = (canvas_size - inner_size) // 2
    canvas.paste(resized, (left, top))
    return canvas, inner_size


def prompt_with_image_placeholder(prompt: str) -> str:
    if "<ImageHere>" in prompt:
        return prompt
    return f"USER: <ImageHere> {prompt.strip()} ASSISTANT:"


def resolve_cfg_path(cfg_path: str, script_dir: Path) -> Path:
    candidate = Path(cfg_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    candidate = script_dir / candidate
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"Could not find cfg file: {cfg_path}")


def load_rvcd_llava(
    args: argparse.Namespace,
    device: torch.device,
    script_dir: Path,
):
    # Avoid importing minigpt4.conversation.conversation.py because that may
    # require optional context_density modules not needed here.
    for path in (script_dir, script_dir.parent):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.models import load_preprocess

    import minigpt4.models  # noqa: F401
    import minigpt4.processors  # noqa: F401

    args.cfg_path = str(resolve_cfg_path(args.cfg_path, script_dir))
    cfg = Config(args)
    model_config = cfg.model_cfg

    if args.model_path:
        model_config.merged_ckpt = str(
            Path(args.model_path).expanduser().resolve()
        )

    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    if model_cls is None:
        raise RuntimeError(
            f"No registered RVCD model for arch={model_config.arch!r}."
        )

    print(
        f"Loading arch={model_config.arch} "
        f"from {model_config.merged_ckpt}"
    )
    model = model_cls.from_config(model_config)
    model.eval()

    processor_cfg = cfg.get_config().preprocess
    processor_cfg.vis_processor.eval.do_normalize = False
    vis_processors, _ = load_preprocess(processor_cfg)
    vis_processor = vis_processors["eval"]
    normalizer = ChannelNormalize(CLIP_MEAN, CLIP_STD)

    tokenizer = model.llama_tokenizer
    tokenizer.padding_side = "left"
    return model, tokenizer, vis_processor, normalizer


class FirstTokenProbe:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        vis_processor: Any,
        normalizer: ChannelNormalize,
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
            raise ValueError("Prompt must contain exactly one <ImageHere>.")

        before, after = instruction.split("<ImageHere>")
        before_ids = self.tokenizer(
            [before] * batch_size,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).input_ids.to(self.device)
        after_ids = self.tokenizer(
            [after] * batch_size,
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
        batch_size: int,
    ) -> torch.Tensor:
        outputs: List[torch.Tensor] = []

        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            image_batch = torch.cat(
                [self.preprocess_image(image) for image in chunk],
                dim=0,
            )
            input_ids = self.build_input_ids(len(chunk))

            with self.model.maybe_autocast():
                model_outputs = self.model.llama_model(
                    input_ids=input_ids,
                    images=image_batch,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

            outputs.append(model_outputs.logits[:, -1, :].float().cpu())
            del image_batch, input_ids, model_outputs

        return torch.cat(outputs, dim=0)


def nucleus_from_probs(
    probabilities: torch.Tensor,
    mass_threshold: float,
) -> Tuple[List[int], int, float]:
    if not 0.0 < mass_threshold <= 1.0:
        raise ValueError("mass_threshold must be in (0, 1].")

    sorted_probs, sorted_ids = torch.sort(
        probabilities.float(),
        descending=True,
    )
    cumulative = torch.cumsum(sorted_probs, dim=0)
    k = int(torch.searchsorted(cumulative, torch.tensor(mass_threshold), right=False).item()) + 1
    token_ids = sorted_ids[:k].tolist()
    realized_mass = float(sorted_probs[:k].sum().item())
    return token_ids, k, realized_mass


def original_nucleus_mass_retention_percent(
    original_nucleus_ids: Sequence[int],
    downsized_probabilities: torch.Tensor,
) -> float:
    if len(original_nucleus_ids) == 0:
        return 0.0
    index_tensor = torch.tensor(
        list(original_nucleus_ids),
        dtype=torch.long,
    )
    retained = downsized_probabilities[index_tensor].sum().item()
    return float(retained * 100.0)


def nucleus_set_overlap_percent(
    original_nucleus_ids: Sequence[int],
    downsized_probabilities: torch.Tensor,
    original_k: int,
) -> float:
    if original_k <= 0:
        return 0.0

    downsized_topk_ids = torch.topk(
        downsized_probabilities.float(),
        k=original_k,
        largest=True,
        sorted=False,
    ).indices.tolist()

    overlap_count = len(
        set(int(i) for i in original_nucleus_ids).intersection(
            set(int(i) for i in downsized_topk_ids)
        )
    )
    return 100.0 * overlap_count / float(original_k)


def colors_from_colormap(
    values_at_final_scale: np.ndarray,
    colormap: str = "viridis",
) -> np.ndarray:
    cmap = plt.get_cmap(colormap)
    count = len(values_at_final_scale)
    order = np.argsort(values_at_final_scale)

    positions_sorted = (
        np.asarray([0.5])
        if count == 1
        else np.linspace(0.05, 0.95, count)
    )
    colors_sorted = cmap(positions_sorted)

    colors = np.empty_like(colors_sorted)
    colors[order] = colors_sorted
    return colors


def x_tick_labels(scales: np.ndarray) -> List[str]:
    return [
        f"{scale:g}×\n{area_percentage(scale):.1f}% area"
        for scale in scales
    ]


def add_curve_summary(
    axis: plt.Axes,
    scales: np.ndarray,
    curves: np.ndarray,
    curve_alpha: float,
    curve_line_width: float,
    colormap: str,
    ylabel: str,
    title: str,
    y_limits: Tuple[float, float] | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    colors = colors_from_colormap(curves[:, -1], colormap)

    for curve, color in zip(curves, colors):
        axis.plot(
            scales,
            curve,
            color=color,
            alpha=curve_alpha,
            linewidth=curve_line_width,
            zorder=2,
        )

    median_curve = np.nanmedian(curves, axis=0)
    mean_curve = np.nanmean(curves, axis=0)
    p10_curve = np.nanpercentile(curves, 10, axis=0)
    p90_curve = np.nanpercentile(curves, 90, axis=0)

    axis.fill_between(
        scales,
        p10_curve,
        p90_curve,
        color="gray",
        alpha=0.6,
        label="10–90 percentile",
        zorder=1,
    )
    axis.plot(
        scales,
        median_curve,
        color="black",
        linewidth=4.0,
        marker="o",
        markersize=5.2,
        label="Median",
        zorder=6,
    )
    axis.plot(
        scales,
        mean_curve,
        color="goldenrod",
        linewidth=2.6,
        linestyle="--",
        marker="s",
        markersize=4.2,
        label="Mean",
        zorder=6,
    )

    axis.set_title(title)
    axis.set_xlabel(
        "Spatial downsizing factor\n"
        "(second line shows remaining image area)"
    )
    axis.set_ylabel(ylabel)
    axis.set_xticks(scales)
    axis.set_xticklabels(x_tick_labels(scales))
    axis.grid(alpha=0.22)

    if y_limits is not None:
        axis.set_ylim(*y_limits)

    return median_curve, mean_curve, p10_curve, p90_curve


def plot_mass_retention_figure(
    scales: np.ndarray,
    curves: np.ndarray,
    mass_threshold: float,
    k_values: np.ndarray,
    output_path: Path,
    curve_alpha: float,
    curve_line_width: float,
    colormap: str,
    dpi: int,
) -> None:
    """Draw one compact mass-retention figure without a side panel."""
    figure, axis = plt.subplots(figsize=(10.5, 6.3))

    median_curve, mean_curve, _, _ = add_curve_summary(
        axis=axis,
        scales=scales,
        curves=curves,
        curve_alpha=curve_alpha,
        curve_line_width=curve_line_width,
        colormap=colormap,
        ylabel="Retained probability mass (%)",
        title=(
            "Retention of Original First Forward Probability Mass under Downsizing"
        ),
        y_limits=(0.0, 100.0),
    )

    # reference_percent = mass_threshold * 100.0
    # axis.axhline(
    #     reference_percent,
    #     color="steelblue",
    #     linewidth=1.5,
    #     linestyle=":",
    #     label=f"Original nucleus target ({reference_percent:.0f}%)",
    #     zorder=3,
    # )

    # Keep only the essential visual elements: individual curves,
    # percentile band, median, mean, and the 90% reference line.
    axis.set_xlabel("Spatial downsizing factor")
    axis.set_xticks(scales)
    axis.set_xticklabels(
        [
            f"{scale:g}×\n({area_percentage(scale):.1f}% area)"
            for scale in scales
        ]
    )
    axis.legend(loc="lower left", fontsize=9)

    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)



def plot_nucleus_overlap_figure(
    scales: np.ndarray,
    curves: np.ndarray,
    mass_threshold: float,
    k_values: np.ndarray,
    output_path: Path,
    curve_alpha: float,
    curve_line_width: float,
    colormap: str,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(13.0, 7.0))
    figure.subplots_adjust(right=0.75)

    median_curve, mean_curve, p10_curve, _ = add_curve_summary(
        axis=axis,
        scales=scales,
        curves=curves,
        curve_alpha=curve_alpha,
        curve_line_width=curve_line_width,
        colormap=colormap,
        ylabel=(
            f"Overlap with original {int(round(mass_threshold * 100))}%-"
            f"nucleus token set (%)"
        ),
        title=(
            f"Original {int(round(mass_threshold * 100))}%-nucleus token set "
            "remains largely represented after downsizing\n"
            f"N={len(curves)} reference images"
        ),
        y_limits=(0.0, 100.0),
    )

    final_values = curves[:, -1]
    final_scale = float(scales[-1])
    final_median = float(np.median(final_values))
    final_mean = float(np.mean(final_values))
    final_p10 = float(np.percentile(final_values, 10))

    k_median = float(np.median(k_values))
    k_mean = float(np.mean(k_values))
    k_p10 = float(np.percentile(k_values, 10))
    k_p90 = float(np.percentile(k_values, 90))
    mean_shared_tokens = float(np.mean(final_values / 100.0 * k_values))

    threshold_070 = float(np.mean(final_values >= 70.0) * 100.0)
    threshold_080 = float(np.mean(final_values >= 80.0) * 100.0)
    threshold_090 = float(np.mean(final_values >= 90.0) * 100.0)

    summary_text = (
        f"At {final_scale:g}× downsizing\n"
        f"({area_percentage(final_scale):.1f}% area retained)\n\n"
        f"Median overlap: {final_median:.1f}%\n"
        f"Mean overlap:   {final_mean:.1f}%\n"
        f"10th pct.:      {final_p10:.1f}%\n"
        f"Mean shared tokens: {mean_shared_tokens:.1f}\n\n"
        f"Original nucleus size k_p\n"
        f"Median: {k_median:.1f}\n"
        f"Mean:   {k_mean:.1f}\n"
        f"10–90 pct.: {k_p10:.1f}–{k_p90:.1f}\n\n"
        f"Refs with overlap ≥ 70%: {threshold_070:.1f}%\n"
        f"Refs with overlap ≥ 80%: {threshold_080:.1f}%\n"
        f"Refs with overlap ≥ 90%: {threshold_090:.1f}%"
    )

    axis.text(
        1.03,
        0.98,
        summary_text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.96,
        },
    )

    axis.annotate(
        f"{final_scale:g}× median\n{final_median:.1f}%",
        xy=(final_scale, median_curve[-1]),
        xytext=(-120, 26),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        fontsize=9,
    )

    axis.legend(
        bbox_to_anchor=(1.03, 0.35),
        loc="upper left",
        borderaxespad=0,
        fontsize=9,
    )

    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one compact RVCD downsizing figure using original "
            "90%-nucleus probability-mass retention."
        )
    )
    parser.add_argument("--ref-folder-path", type=Path, required=True)
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="./eval_configs/llava-1.5_eval.yaml",
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="0 means all discovered reference images.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scales",
        nargs="+",
        default=["1", "1.5", "2", "2.5", "3"],
    )
    parser.add_argument("--canvas-size", type=int, default=336)
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--strict-source-size",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the main object in this image using exactly one word.",
    )
    parser.add_argument(
        "--mass-threshold",
        type=float,
        default=0.90,
        help="Cumulative probability threshold for original nucleus. Default: 0.90",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--curve-alpha", type=float, default=0.10)
    parser.add_argument("--curve-line-width", type=float, default=0.45)
    parser.add_argument("--colormap", type=str, default="viridis")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./rvcd_nucleus90_mass_retention_compact"),
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--options",
        nargs="+",
        default=None,
        help="Compatibility argument consumed by RVCD Config.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.canvas_size <= 0:
        raise ValueError("--canvas-size must be positive.")
    if not 0.0 < args.mass_threshold <= 1.0:
        raise ValueError("--mass-threshold must be in (0, 1].")
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive.")
    if not 0.0 < args.curve_alpha <= 1.0:
        raise ValueError("--curve-alpha must be in (0, 1].")
    if args.curve_line_width <= 0:
        raise ValueError("--curve-line-width must be positive.")

    scales = normalize_scales(parse_float_list(args.scales))
    set_global_seed(args.seed)

    reference_root = args.ref_folder_path.expanduser().resolve()
    if not reference_root.is_dir():
        raise FileNotFoundError(reference_root)

    all_images = discover_images(reference_root, args.recursive)
    if not all_images:
        raise RuntimeError(
            f"No supported images were found under {reference_root}."
        )

    selected_images = choose_images(all_images, args.num_samples, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid --gpu-id {args.gpu_id}; visible devices: "
            f"{torch.cuda.device_count()}."
        )

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    model, tokenizer, vis_processor, normalizer = load_rvcd_llava(
        args=args,
        device=device,
        script_dir=script_dir,
    )
    probe = FirstTokenProbe(
        model=model,
        tokenizer=tokenizer,
        vis_processor=vis_processor,
        normalizer=normalizer,
        device=device,
        prompt=args.prompt,
    )

    mass_curves: List[List[float]] = []
    overlap_curves: List[List[float]] = []
    k_values: List[int] = []
    realized_original_masses: List[float] = []
    successful_images: List[str] = []
    per_image_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    print(f"References selected: {len(selected_images)}")
    print(f"Downsizing factors: {scales}")
    print(
        f"Metric A: retained mass on original {args.mass_threshold:.2f}-nucleus"
    )
    print(
        f"Metric B: overlap with original {args.mass_threshold:.2f}-nucleus token set"
    )

    progress = tqdm(
        enumerate(selected_images),
        total=len(selected_images),
        desc="RVCD nucleus retention + overlap",
    )

    for image_index, image_path in progress:
        try:
            with Image.open(image_path) as opened:
                source = opened.convert("RGB")

                if (
                    args.strict_source_size
                    and source.size != (args.canvas_size, args.canvas_size)
                ):
                    raise ValueError(
                        f"{image_path.name}: expected "
                        f"{args.canvas_size}x{args.canvas_size}, got {source.size}."
                    )

                original = fit_to_white_canvas(source, args.canvas_size)

            probe_images: List[Image.Image] = []
            inner_sizes: List[int] = []

            for scale in scales:
                probe_image, inner_size = downsize_and_white_pad(
                    original,
                    scale,
                    args.canvas_size,
                )
                probe_images.append(probe_image)
                inner_sizes.append(inner_size)

            logits = probe.logits_for_images(
                probe_images,
                batch_size=args.inference_batch_size,
            )
            original_logits = logits[0]
            original_probs = torch.softmax(original_logits.float(), dim=-1)

            original_nucleus_ids, original_k, realized_mass = nucleus_from_probs(
                original_probs,
                args.mass_threshold,
            )

            image_mass_curve: List[float] = []
            image_overlap_curve: List[float] = []

            for scale_index, scale in enumerate(scales):
                current_logits = logits[scale_index]
                current_probs = torch.softmax(current_logits.float(), dim=-1)

                retained_mass = original_nucleus_mass_retention_percent(
                    original_nucleus_ids,
                    current_probs,
                )
                overlap_percent = nucleus_set_overlap_percent(
                    original_nucleus_ids,
                    current_probs,
                    original_k,
                )

                image_mass_curve.append(retained_mass)
                image_overlap_curve.append(overlap_percent)

                per_image_rows.append({
                    "image_index": image_index,
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "scale": float(scale),
                    "inner_size": int(inner_sizes[scale_index]),
                    "area_percent": area_percentage(scale),
                    "mass_threshold": float(args.mass_threshold),
                    "original_k": int(original_k),
                    "original_realized_mass_percent": float(realized_mass * 100.0),
                    "retained_mass_percent": float(retained_mass),
                    "nucleus_set_overlap_percent": float(overlap_percent),
                    "shared_token_count": float(overlap_percent / 100.0 * original_k),
                })

            mass_curves.append(image_mass_curve)
            overlap_curves.append(image_overlap_curve)
            k_values.append(original_k)
            realized_original_masses.append(realized_mass * 100.0)
            successful_images.append(str(image_path))

            progress.set_postfix(
                image=image_path.name,
                final_mass=f"{image_mass_curve[-1]:.1f}%",
                final_overlap=f"{image_overlap_curve[-1]:.1f}%",
                k=f"{original_k}",
            )

            del logits, original_logits, original_probs

        except Exception as error:
            failure = {
                "image_index": image_index,
                "image_path": str(image_path),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            print(
                f"\n[warning] {image_path}: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            if args.fail_fast:
                raise

    if not mass_curves:
        raise RuntimeError("No image completed successfully.")

    mass_array = np.asarray(mass_curves, dtype=float)
    overlap_array = np.asarray(overlap_curves, dtype=float)
    k_array = np.asarray(k_values, dtype=float)
    realized_mass_array = np.asarray(realized_original_masses, dtype=float)
    scale_array = np.asarray(scales, dtype=float)

    per_scale_rows: List[Dict[str, Any]] = []
    for scale_index, scale in enumerate(scales):
        retained_values = mass_array[:, scale_index]
        overlap_values = overlap_array[:, scale_index]

        per_scale_rows.append({
            "scale": float(scale),
            "inner_size": int(round(args.canvas_size / scale)),
            "area_percent": area_percentage(scale),
            "num_images": int(len(retained_values)),
            "mass_threshold": float(args.mass_threshold),
            "original_k_mean": float(np.mean(k_array)),
            "original_k_median": float(np.median(k_array)),
            "original_k_p10": float(np.percentile(k_array, 10)),
            "original_k_p90": float(np.percentile(k_array, 90)),
            "original_realized_mass_mean_percent": float(np.mean(realized_mass_array)),
            "retained_mass_mean_percent": float(np.mean(retained_values)),
            "retained_mass_median_percent": float(np.median(retained_values)),
            "retained_mass_p10_percent": float(np.percentile(retained_values, 10)),
            "retained_mass_p90_percent": float(np.percentile(retained_values, 90)),
            "retained_mass_ge_80_percent": float(np.mean(retained_values >= 80.0) * 100.0),
            "retained_mass_ge_90_percent": float(np.mean(retained_values >= 90.0) * 100.0),
            "retained_mass_ge_95_percent": float(np.mean(retained_values >= 95.0) * 100.0),
            "nucleus_overlap_mean_percent": float(np.mean(overlap_values)),
            "nucleus_overlap_median_percent": float(np.median(overlap_values)),
            "nucleus_overlap_p10_percent": float(np.percentile(overlap_values, 10)),
            "nucleus_overlap_p90_percent": float(np.percentile(overlap_values, 90)),
            "nucleus_overlap_ge_70_percent": float(np.mean(overlap_values >= 70.0) * 100.0),
            "nucleus_overlap_ge_80_percent": float(np.mean(overlap_values >= 80.0) * 100.0),
            "nucleus_overlap_ge_90_percent": float(np.mean(overlap_values >= 90.0) * 100.0),
        })

    mass_figure = (
        args.output_dir / "nucleus90_mass_retention_compact.png"
    )

    plot_mass_retention_figure(
        scales=scale_array,
        curves=mass_array,
        mass_threshold=args.mass_threshold,
        k_values=k_array,
        output_path=mass_figure,
        curve_alpha=args.curve_alpha,
        curve_line_width=args.curve_line_width,
        colormap=args.colormap,
        dpi=args.dpi,
    )

    write_csv(
        args.output_dir / "per_image_scale_metrics.csv",
        per_image_rows,
        [
            "image_index",
            "image_path",
            "image_name",
            "scale",
            "inner_size",
            "area_percent",
            "mass_threshold",
            "original_k",
            "original_realized_mass_percent",
            "retained_mass_percent",
            "nucleus_set_overlap_percent",
            "shared_token_count",
        ],
    )

    write_csv(
        args.output_dir / "per_scale_summary.csv",
        per_scale_rows,
        list(per_scale_rows[0].keys()),
    )

    final_scale = float(scales[-1])
    final_mass = mass_array[:, -1]
    final_overlap = overlap_array[:, -1]

    summary = {
        "num_successful_images": int(len(mass_array)),
        "num_failed_images": int(len(failures)),
        "scales": scales,
        "mass_threshold": float(args.mass_threshold),
        "final_scale": final_scale,
        "final_inner_size": int(round(args.canvas_size / final_scale)),
        "final_area_percent": area_percentage(final_scale),
        "original_k_mean": float(np.mean(k_array)),
        "original_k_median": float(np.median(k_array)),
        "original_k_p10": float(np.percentile(k_array, 10)),
        "original_k_p90": float(np.percentile(k_array, 90)),
        "original_realized_mass_mean_percent": float(np.mean(realized_mass_array)),
        "original_realized_mass_median_percent": float(np.median(realized_mass_array)),
        "final_retained_mass_mean_percent": float(np.mean(final_mass)),
        "final_retained_mass_median_percent": float(np.median(final_mass)),
        "final_retained_mass_p10_percent": float(np.percentile(final_mass, 10)),
        "final_retained_mass_ge_80_percent": float(np.mean(final_mass >= 80.0) * 100.0),
        "final_retained_mass_ge_90_percent": float(np.mean(final_mass >= 90.0) * 100.0),
        "final_retained_mass_ge_95_percent": float(np.mean(final_mass >= 95.0) * 100.0),
        "final_nucleus_overlap_mean_percent": float(np.mean(final_overlap)),
        "final_nucleus_overlap_median_percent": float(np.median(final_overlap)),
        "final_nucleus_overlap_p10_percent": float(np.percentile(final_overlap, 10)),
        "final_nucleus_overlap_ge_70_percent": float(np.mean(final_overlap >= 70.0) * 100.0),
        "final_nucleus_overlap_ge_80_percent": float(np.mean(final_overlap >= 80.0) * 100.0),
        "final_nucleus_overlap_ge_90_percent": float(np.mean(final_overlap >= 90.0) * 100.0),
    }

    (args.output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (args.output_dir / "successful_images.txt").write_text(
        "\n".join(successful_images),
        encoding="utf-8",
    )

    if failures:
        (args.output_dir / "failures.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("\nFinished.")
    print(f"Successful references: {len(mass_array)}")
    print(f"Mass-retention figure: {mass_figure}")
    print(
        f"At {final_scale:g}× ({area_percentage(final_scale):.1f}% area): "
        f"median retained mass={np.median(final_mass):.2f}%, "
        f"median nucleus overlap={np.median(final_overlap):.2f}%."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# CUDA_VISIBLE_DEVICES=0 python rvcd_nucleus90_mass_retention_compact.py \
#   --ref-folder-path \
#   ./DB_single_concept_images_flux_generated/generated_images \
#   --cfg-path ./eval_configs/llava-1.5_eval.yaml \
#   --num-samples 0 \
#   --scales 1 1.5 2 2.5 3 \
#   --mass-threshold 0.90 \
#   --gpu-id 0 \
#   --inference-batch-size 1 \
#   --curve-alpha 0.25 \
#   --curve-line-width 0.45 \
#   --colormap viridis \
#   --output-dir ./rvcd_nucleus90_mass_retention_compact