
import argparse
import math
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


# ============================================================
# Negative-signal online probe and visualization utilities
# - embedded so this script can run generation + analysis in one file
# ============================================================
import argparse
import json
import math
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - allows plotting mode without torch
    torch = None
    F = None


Json = Dict[str, Any]


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _append_jsonl(path: str, row: Json) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")


def _write_json(path: str, obj: Json) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_jsonl(path: str) -> List[Json]:
    rows: List[Json] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _slug(text: Any, max_len: int = 80) -> str:
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9가-힣._-]+", "_", text).strip("_")
    return text[:max_len] if text else "item"


def _norm_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    text_n = _norm_text(text)
    phrase_n = _norm_text(phrase)
    if not phrase_n:
        return False
    # Word-ish boundary for Latin phrases; substring fallback for other scripts.
    if re.search(r"[a-zA-Z]", phrase_n):
        return re.search(rf"(?<![a-zA-Z]){re.escape(phrase_n)}(?![a-zA-Z])", text_n) is not None
    return phrase_n in text_n


def _tensor_2d(logit: Any):
    if torch is None:
        raise RuntimeError("torch is required for online probing")
    if logit is None:
        return None
    if isinstance(logit, (list, tuple)):
        raise TypeError("Expected a tensor, got list/tuple")
    if logit.dim() == 1:
        logit = logit.unsqueeze(0)
    return logit.detach()


def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _token_id_to_text(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        try:
            return tokenizer.convert_ids_to_tokens([int(token_id)], skip_special_tokens=False)[0]
        except Exception:
            return str(token_id)


def _encode_variants(tokenizer: Any, surface: str) -> List[int]:
    """Return plausible first-token IDs for an object surface form.

    Most LLaMA/LLaVA-style tokenizers encode object mentions differently with
    and without leading whitespace. We keep both first tokens and deduplicate.
    """
    ids: List[int] = []
    for variant in (surface, " " + surface):
        try:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer(variant, add_special_tokens=False).input_ids
        except Exception:
            encoded = []
        if encoded:
            ids.append(int(encoded[0]))
    out: List[int] = []
    seen = set()
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _encode_variant_sequences(tokenizer: Any, surface: str) -> List[List[int]]:
    """Return token-id sequences for surface and leading-space surface variants."""
    seqs: List[List[int]] = []
    for variant in (surface, " " + surface):
        try:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer(variant, add_special_tokens=False).input_ids
        except Exception:
            encoded = []
        encoded = [int(x) for x in encoded]
        if encoded and encoded not in seqs:
            seqs.append(encoded)
    return seqs


def _normalize_hal_object(obj: Union[str, Tuple[str, str], Json]) -> Json:
    if isinstance(obj, dict):
        coco = obj.get("coco") or obj.get("coco_first") or obj.get("object") or obj.get("surface")
        surface = obj.get("surface") or obj.get("synonym") or obj.get("word") or coco
        return {"coco": str(coco), "surface": str(surface)}
    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        return {"coco": str(obj[0]), "surface": str(obj[1])}
    return {"coco": str(obj), "surface": str(obj)}


def _safe_softmax_prob(logit_2d: Any, token_id: int) -> float:
    probs = F.softmax(logit_2d.float(), dim=-1)
    return _to_float(probs[0, int(token_id)])


def _safe_logit_value(logit_2d: Any, token_id: int) -> float:
    return _to_float(logit_2d.float()[0, int(token_id)])


def _topk(logit_2d: Any, tokenizer: Any, k: int = 20) -> List[Json]:
    if k <= 0:
        return []
    probs = F.softmax(logit_2d.float(), dim=-1)
    values, indices = torch.topk(probs[0], k=min(k, probs.shape[-1]))
    rows: List[Json] = []
    for rank, (prob, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        idx = int(idx)
        rows.append(
            {
                "rank": rank,
                "token_id": idx,
                "token": _token_id_to_text(tokenizer, idx),
                "prob": float(prob),
                "logit": _safe_logit_value(logit_2d, idx),
            }
        )
    return rows


def _rank_of_token(logit_2d: Any, token_id: int) -> Optional[int]:
    # Rank 1 means highest logit. This is O(vocab) but fine for probing.
    vals = logit_2d.float()[0]
    token_val = vals[int(token_id)]
    return int((vals > token_val).sum().item()) + 1


@dataclass
class StepRecord:
    image_id: int
    step: int
    selected_token_id: int
    selected_token: str
    negative_grid_path: Optional[str]
    negative_grid_meta: Optional[Json]
    selected_stats: Json
    hal_object_stats: Dict[str, Json]
    topk: Optional[Json]
    prefix_token_ids: List[int]


class NegativeSignalProbe:
    """Online probe for eRVCD negative-signal analysis.

    This class should be called inside the generation loop because logits are
    not recoverable from final caption JSONL files alone.
    """

    def __init__(
        self,
        out_dir: str,
        tokenizer: Any,
        chair_evaluator: Optional[Any] = None,
        top_k: int = 20,
        save_all_steps_jsonl: bool = True,
    ) -> None:
        self.root = _ensure_dir(os.path.join(out_dir, "negative_signal_probe"))
        self.fail_dir = _ensure_dir(os.path.join(self.root, "fail_cases"))
        self.steps_jsonl = os.path.join(self.root, "step_trace.jsonl")
        self.fail_jsonl = os.path.join(self.root, "fail_cases.jsonl")
        self.tokenizer = tokenizer
        self.chair_evaluator = chair_evaluator
        self.top_k = int(top_k)
        self.save_all_steps_jsonl = bool(save_all_steps_jsonl)
        self._steps_by_image: Dict[int, List[StepRecord]] = defaultdict(list)

    def _aggregate_negative_logits(
        self,
        negative_logits: Sequence[Any],
        negative_logits_count: int,
    ) -> Optional[Any]:
        if not negative_logits:
            return None
        negs = [_tensor_2d(x).float() for x in negative_logits]
        if len(negs) == 1:
            # eRVCD normally has one merged negative-grid logit. In count mode,
            # negative_logits_count becomes the raw reference count, so scale the
            # single merged logit to match aggregate_ervcd_logits() in the main code.
            return negs[0] * max(1, int(negative_logits_count))
        # In the original RVCD case this is a sum.
        return torch.stack(negs, dim=0).sum(dim=0)

    def log_step(
        self,
        *,
        image_id: int,
        step: int,
        selected_token_id: int,
        original_logit: Any,
        negative_logits: Sequence[Any],
        adjusted_logits: Any,
        hal_objects: Sequence[Union[str, Tuple[str, str], Json]],
        negative_grid_path: Optional[str],
        negative_grid_meta: Optional[Json],
        alpha: float,
        negative_logits_count: int,
        prefix_token_ids: Optional[Sequence[int]] = None,
        keep_topk: bool = True,
    ) -> StepRecord:
        """Record one decoding step.

        Parameters are intentionally close to the variable names in your
        generation script.
        """
        if torch is None:
            raise RuntimeError("torch is required for NegativeSignalProbe.log_step")

        image_id = int(image_id)
        step = int(step)
        selected_token_id = int(selected_token_id)
        prefix_token_ids = [int(x) for x in (prefix_token_ids or [])]

        O = _tensor_2d(original_logit).float()
        A = _tensor_2d(adjusted_logits).float()
        N = self._aggregate_negative_logits(negative_logits, negative_logits_count)

        if N is None:
            # No negative signal. Store a dummy copy so downstream fields exist.
            N = torch.zeros_like(O)
            A_neg_only = O.clone()
        else:
            # Isolated negative-only adjusted logit.
            A_neg_only = (1.0 + float(alpha) * int(negative_logits_count)) * O - float(alpha) * N

        def stats_for_token(token_id: int) -> Json:
            token_id = int(token_id)
            p_o = _safe_softmax_prob(O, token_id)
            p_n = _safe_softmax_prob(N, token_id)
            p_a_neg = _safe_softmax_prob(A_neg_only, token_id)
            p_a = _safe_softmax_prob(A, token_id)
            l_o = _safe_logit_value(O, token_id)
            l_n = _safe_logit_value(N, token_id)
            l_a_neg = _safe_logit_value(A_neg_only, token_id)
            l_a = _safe_logit_value(A, token_id)
            return {
                "token_id": token_id,
                "token": _token_id_to_text(self.tokenizer, token_id),
                "prob_original": p_o,
                "prob_negative_agg": p_n,
                "prob_adjusted_neg_only": p_a_neg,
                "prob_adjusted_actual": p_a,
                "prob_suppression_neg_only": p_o - p_a_neg,
                "prob_suppression_actual": p_o - p_a,
                "logit_original": l_o,
                "logit_negative_agg": l_n,
                "logit_adjusted_neg_only": l_a_neg,
                "logit_adjusted_actual": l_a,
                "negative_pressure_logit": l_n - l_o,
                "logit_suppression_neg_only": l_o - l_a_neg,
                "logit_suppression_actual": l_o - l_a,
                "rank_original": _rank_of_token(O, token_id),
                "rank_negative_agg": _rank_of_token(N, token_id),
                "rank_adjusted_neg_only": _rank_of_token(A_neg_only, token_id),
                "rank_adjusted_actual": _rank_of_token(A, token_id),
            }

        selected_stats = stats_for_token(selected_token_id)

        hal_object_stats: Dict[str, Json] = {}
        for raw_obj in hal_objects:
            obj = _normalize_hal_object(raw_obj)
            key = f"{obj['coco']}::{obj['surface']}"
            first_token_ids = _encode_variants(self.tokenizer, obj["surface"])
            token_rows = [stats_for_token(tid) for tid in first_token_ids]
            hal_object_stats[key] = {
                "coco": obj["coco"],
                "surface": obj["surface"],
                "first_token_ids": first_token_ids,
                "first_token_stats": token_rows,
            }

        topk_blob: Optional[Json] = None
        if keep_topk:
            suppress_delta = N - O  # high values are tokens pushed by N more than O, thus suppressed by contrast.
            topk_blob = {
                "original": _topk(O, self.tokenizer, self.top_k),
                "negative_agg": _topk(N, self.tokenizer, self.top_k),
                "adjusted_neg_only": _topk(A_neg_only, self.tokenizer, self.top_k),
                "adjusted_actual": _topk(A, self.tokenizer, self.top_k),
                "suppressed_by_negative_topk": _topk(suppress_delta, self.tokenizer, self.top_k),
            }

        rec = StepRecord(
            image_id=image_id,
            step=step,
            selected_token_id=selected_token_id,
            selected_token=_token_id_to_text(self.tokenizer, selected_token_id),
            negative_grid_path=negative_grid_path,
            negative_grid_meta=negative_grid_meta,
            selected_stats=selected_stats,
            hal_object_stats=hal_object_stats,
            topk=topk_blob,
            prefix_token_ids=prefix_token_ids,
        )
        self._steps_by_image[image_id].append(rec)

        if self.save_all_steps_jsonl:
            _append_jsonl(
                self.steps_jsonl,
                {
                    "image_id": rec.image_id,
                    "step": rec.step,
                    "selected_token_id": rec.selected_token_id,
                    "selected_token": rec.selected_token,
                    "negative_grid_path": rec.negative_grid_path,
                    "negative_grid_meta": rec.negative_grid_meta,
                    "selected_stats": rec.selected_stats,
                    "hal_object_stats": rec.hal_object_stats,
                    "topk": rec.topk,
                    "prefix_token_ids": rec.prefix_token_ids,
                },
            )
        return rec

    def _final_caption_objects(self, caption: str) -> List[Json]:
        if self.chair_evaluator is None:
            return []
        try:
            pairs = self.chair_evaluator.process_sentence_get_coco_synonyms(caption)
            return [{"coco": str(c), "surface": str(s)} for c, s in pairs]
        except Exception:
            return []

    def _find_first_mention_step(self, output_token_ids: Sequence[int], mention_surface: str) -> Optional[int]:
        """Approximate first generation step where mention_surface appears.

        This is tokenizer-agnostic and works by incrementally decoding prefixes.
        For subword tokenizers, the step returned is the first step at which the
        full surface form becomes visible. That is often the last sub-token of
        the word, not always the first sub-token. To compensate, finalize_datapoint
        also stores selected-token stats for that step and object first-token stats
        logged at every step.
        """
        prev = ""
        for i in range(len(output_token_ids)):
            try:
                cur = self.tokenizer.decode(list(map(int, output_token_ids[: i + 1])), skip_special_tokens=True)
            except Exception:
                cur = "".join(_token_id_to_text(self.tokenizer, tid) for tid in output_token_ids[: i + 1])
            if _contains_phrase(cur, mention_surface) and not _contains_phrase(prev, mention_surface):
                return i
            prev = cur
        return None


    def _find_first_token_step(self, output_token_ids: Sequence[int], mention_surface: str) -> Optional[int]:
        """Find the decoding step of the first token of a surface mention.

        This first tries exact token-sequence matching for both `surface` and
        `" " + surface`. If exact matching fails, it falls back to the first
        occurrence of any plausible first-token id. This is the metric target
        requested for fail cases: the probability of the first token that starts
        the surviving object expression.
        """
        output_ids = [int(x) for x in output_token_ids]
        seqs = _encode_variant_sequences(self.tokenizer, mention_surface)
        for i in range(len(output_ids)):
            for seq in seqs:
                if output_ids[i : i + len(seq)] == seq:
                    return i

        first_ids = {seq[0] for seq in seqs if seq}
        for i, tid in enumerate(output_ids):
            if tid in first_ids:
                return i
        return None

    def finalize_datapoint(
        self,
        *,
        image_id: int,
        draft_caption: str,
        final_caption: str,
        output_token_ids: Sequence[int],
        hal_objects: Sequence[Union[str, Tuple[str, str], Json]],
        negative_grid_path: Optional[str],
        negative_grid_meta: Optional[Json],
        extra: Optional[Json] = None,
    ) -> List[Json]:
        """Check whether suppressed objects survived in the final caption.

        Returns the fail-case records saved for this image.
        """
        image_id = int(image_id)
        hal_norm = [_normalize_hal_object(x) for x in hal_objects]
        final_objs = self._final_caption_objects(final_caption)

        failures: List[Json] = []
        steps = self._steps_by_image.get(image_id, [])

        for obj in hal_norm:
            # Prefer CHAIR's final object extraction. Fall back to surface substring.
            matching_final_mentions = [x for x in final_objs if x.get("coco") == obj.get("coco")]
            if not matching_final_mentions and _contains_phrase(final_caption, obj["surface"]):
                matching_final_mentions = [obj]

            if not matching_final_mentions:
                continue

            # Pick the first detected final surface mention for this COCO object.
            final_mention = matching_final_mentions[0]
            mention_surface = final_mention.get("surface") or obj["surface"]
            full_mention_step = self._find_first_mention_step(output_token_ids, mention_surface)
            first_token_step = self._find_first_token_step(output_token_ids, mention_surface)
            if first_token_step is None and obj.get("surface") != mention_surface:
                first_token_step = self._find_first_token_step(output_token_ids, obj["surface"])

            # Main analysis target: step of the first token that starts the surviving object expression.
            mention_step = first_token_step if first_token_step is not None else full_mention_step

            step_rec: Optional[StepRecord] = None
            if mention_step is not None:
                for rec in steps:
                    if rec.step == int(mention_step):
                        step_rec = rec
                        break

            # If token matching failed, store the closest info we have.
            if step_rec is None and steps:
                fallback_step = full_mention_step if full_mention_step is not None else len(output_token_ids) - 1
                step_rec = steps[min(len(steps) - 1, max(0, int(fallback_step)))]

            obj_key = f"{obj['coco']}::{obj['surface']}"
            object_step_stats = step_rec.hal_object_stats.get(obj_key) if step_rec else None
            selected_stats = step_rec.selected_stats if step_rec else None

            case_id = f"image_{image_id}_obj_{_slug(obj['coco'])}_{_slug(obj['surface'])}"
            case_dir = _ensure_dir(os.path.join(self.fail_dir, case_id))

            copied_negative_grid = None
            if negative_grid_path and os.path.exists(negative_grid_path):
                ext = os.path.splitext(negative_grid_path)[1] or ".png"
                copied_negative_grid = os.path.join(case_dir, "negative_grid" + ext)
                try:
                    shutil.copy2(negative_grid_path, copied_negative_grid)
                except Exception:
                    copied_negative_grid = negative_grid_path

            row: Json = {
                "case_id": case_id,
                "image_id": image_id,
                "failed_object": obj,
                "final_mention": final_mention,
                "mention_surface_used_for_step_search": mention_surface,
                "mention_step": mention_step,
                "first_token_step": first_token_step,
                "full_mention_step": full_mention_step,
                "step_metric_target": "object_first_token_step" if first_token_step is not None else "full_mention_step_or_fallback",
                "draft_caption": draft_caption,
                "final_caption": final_caption,
                "negative_grid_path_original": negative_grid_path,
                "negative_grid_path_copied": copied_negative_grid,
                "negative_grid_meta": negative_grid_meta,
                "selected_token_stats_at_mention_step": selected_stats,
                "selected_token_stats_at_object_first_token_step": selected_stats,
                "object_first_token_stats_at_mention_step": object_step_stats,
                "extra": extra or {},
            }

            _write_json(os.path.join(case_dir, "mapping.json"), row)
            _append_jsonl(self.fail_jsonl, row)
            failures.append(row)

        return failures


def _metric_from_case(row: Json, metric: str) -> float:
    stats = row.get("selected_token_stats_at_mention_step") or {}
    value = stats.get(metric)
    try:
        if value is None or math.isnan(float(value)):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def visualize_fail_cases(fail_jsonl: str, out_dir: str, top_n: int = 30) -> Json:
    """Create simple visualizations for failed negative-signal cases."""
    rows = _read_jsonl(fail_jsonl)
    _ensure_dir(out_dir)

    # Sort by weak negative-only probability suppression first.
    rows_sorted = sorted(rows, key=lambda r: _metric_from_case(r, "prob_suppression_neg_only"))
    top_rows = rows_sorted[: int(top_n)]

    summary_path = os.path.join(out_dir, f"top{top_n}_weakest_suppression_fail_cases.json")
    _write_json(summary_path, {"fail_jsonl": fail_jsonl, "top_n": top_n, "cases": top_rows})

    if not top_rows:
        return {"num_fail_cases": 0, "summary_path": summary_path}

    labels = [
        f"{r.get('image_id')}:{(r.get('failed_object') or {}).get('surface')}"
        for r in top_rows
    ]
    y = list(range(len(top_rows)))
    suppression = [_metric_from_case(r, "prob_suppression_neg_only") for r in top_rows]
    negative_pressure = [_metric_from_case(r, "negative_pressure_logit") for r in top_rows]
    p_orig = [_metric_from_case(r, "prob_original") for r in top_rows]
    p_adj = [_metric_from_case(r, "prob_adjusted_neg_only") for r in top_rows]

    # Plot 1: weakest suppression among failures.
    plt.figure(figsize=(11, max(4, 0.38 * len(top_rows))))
    plt.barh(y, suppression)
    plt.yticks(y, labels)
    plt.xlabel("P_original(token) - P_negative_only_adjusted(token)")
    plt.ylabel("failed case: image_id:object")
    plt.title(f"Top-{top_n} failed cases sorted by weakest negative-only probability suppression")
    plt.tight_layout()
    bar_path = os.path.join(out_dir, f"top{top_n}_weakest_suppression_bar.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()

    # Plot 2: original vs negative-only adjusted probability for surviving object token.
    plt.figure(figsize=(7, 6))
    plt.scatter(p_orig, p_adj)
    lim_max = max(max(p_orig), max(p_adj), 1e-8)
    plt.plot([0, lim_max], [0, lim_max])
    plt.xlabel("P_original(surviving object token)")
    plt.ylabel("P_negative_only_adjusted(surviving object token)")
    plt.title("Failed mentions: original vs negative-only adjusted token probability")
    plt.tight_layout()
    scatter_path = os.path.join(out_dir, f"top{top_n}_original_vs_negonly_prob.png")
    plt.savefig(scatter_path, dpi=200)
    plt.close()

    # Plot 3: negative pressure logit for surviving token.
    plt.figure(figsize=(11, max(4, 0.38 * len(top_rows))))
    plt.barh(y, negative_pressure)
    plt.yticks(y, labels)
    plt.xlabel("N_logit(token) - O_logit(token)")
    plt.ylabel("failed case: image_id:object")
    plt.title(f"Top-{top_n} failed cases: negative pressure on surviving object token")
    plt.tight_layout()
    pressure_path = os.path.join(out_dir, f"top{top_n}_negative_pressure_bar.png")
    plt.savefig(pressure_path, dpi=200)
    plt.close()

    return {
        "num_fail_cases": len(rows),
        "num_visualized": len(top_rows),
        "summary_path": summary_path,
        "bar_path": bar_path,
        "scatter_path": scatter_path,
        "pressure_path": pressure_path,
    }

# ============================================================
# End negative-signal probe utilities
# ============================================================


# ============================================================
# Actual eRVCD grid VLM probe utilities
# - probes the exact negative reference grid image used by eRVCD
# ============================================================

def _grid_probe_token_display(tokenizer: Any, token_id: int) -> Json:
    raw_token = tokenizer.convert_ids_to_tokens([int(token_id)], skip_special_tokens=False)[0]
    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return {
        "token_id": int(token_id),
        "token": str(raw_token),
        "decoded": str(decoded),
        "decoded_clean": str(decoded).replace("\n", "\\n"),
    }


def _grid_probe_is_special_id(tokenizer: Any, token_id: int) -> bool:
    special_ids = set()
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
        val = getattr(tokenizer, attr, None)
        if val is not None:
            special_ids.add(int(val))
    extra = getattr(tokenizer, "all_special_ids", None)
    if extra is not None:
        special_ids.update(int(x) for x in extra)
    return int(token_id) in special_ids


def _grid_probe_get_lm_head_matrix(model: Any, model_name: str):
    if model_name == "mplug-owl2":
        return model.model.lm_head.weight
    return model.llama_model.lm_head.weight


def _grid_probe_next_token_logits(
    *,
    model: Any,
    model_name: str,
    image: Any,
    prompt: str,
    image_path: str,
    prev_tokens: Sequence[Any],
    use_nucleus_sampling: bool = False,
    num_beams: int = 1,
):
    """Return vocab logits for the next token on the given grid image.

    This follows the custom LLaVA/MiniGPT generation path used in the main eRVCD code.
    output_attentions=True is intentionally kept because the local llava.py expects it.
    """
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
    lm_head = _grid_probe_get_lm_head_matrix(model, model_name).detach()
    with torch.no_grad():
        logits = torch.matmul(last_hidden, lm_head.T)
    return logits.detach()


def _grid_probe_topn_from_logits(logits: Any, tokenizer: Any, top_n: int, skip_special_tokens: bool) -> List[Json]:
    probs = F.softmax(logits.float(), dim=-1)[0]
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)

    results: List[Json] = []
    for prob, token_id in zip(sorted_probs.tolist(), sorted_ids.tolist()):
        token_id = int(token_id)
        if skip_special_tokens and _grid_probe_is_special_id(tokenizer, token_id):
            continue
        info = _grid_probe_token_display(tokenizer, token_id)
        info["probability"] = float(prob)
        info["log_probability"] = float(math.log(max(float(prob), 1e-45)))
        info["rank"] = len(results) + 1
        results.append(info)
        if len(results) >= int(top_n):
            break
    return results


def _grid_probe_greedy_continue_from_first_token(
    *,
    model: Any,
    tokenizer: Any,
    model_name: str,
    image: Any,
    prompt: str,
    image_path: str,
    first_token_id: int,
    total_tokens: int,
    use_nucleus_sampling: bool = False,
    num_beams: int = 1,
) -> Json:
    """Force the first token, then greedily continue until total_tokens or EOS."""
    device = image.device
    output_tokens = [torch.tensor(int(first_token_id), device=device)]

    for _ in range(max(0, int(total_tokens) - 1)):
        logits = _grid_probe_next_token_logits(
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
        "forced_first_token": _grid_probe_token_display(tokenizer, first_token_id),
        "generated_token_ids": token_ids,
        "generated_tokens": [_grid_probe_token_display(tokenizer, tid) for tid in token_ids],
        "generated_text": decoded,
        "continuation_token_ids": continuation_token_ids,
        "continuation_text": continuation_text,
    }


def _grid_probe_shorten_text(text: str, max_chars: int) -> str:
    text = str(text).replace("\n", " ").strip()
    if len(text) <= int(max_chars):
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: int(max_chars) - 1] + "…"


def _grid_probe_wrap_text(text: str, width: int = 18, max_lines: int = 3) -> str:
    text = str(text).replace("\n", " ").strip()
    if not text:
        return ""
    words = text.split()
    lines: List[str] = []
    cur = ""
    for word in words:
        trial = (cur + " " + word).strip()
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if not lines:
        lines = [text]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _grid_probe_shorten_text(lines[-1], max(3, width))
    return "\n".join(lines)


def _grid_probe_save_combined_figure(
    *,
    grid_image_path: str,
    topn: List[Json],
    continuations: List[Json],
    save_path: str,
    title: str,
    ref_names: Sequence[str],
) -> None:
    grid_img = Image.open(grid_image_path).convert("RGB")

    labels: List[Json] = []
    values: List[float] = []
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
    fig_h = 7.1
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, max(2.0, len(labels) * 0.58)], wspace=0.18)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(grid_img)
    ref_title = "Actual negative ref grid"
    if ref_names:
        ref_title += "\n" + _grid_probe_wrap_text(", ".join(ref_names), width=38, max_lines=3)
    ax_img.set_title(ref_title, fontsize=11)
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
        prefix_text = _grid_probe_wrap_text(item["prefix"], width=12, max_lines=2)
        continuation_text = _grid_probe_wrap_text(item["continuation"], width=18, max_lines=3)
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
    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def run_actual_grid_vlm_probe(
    *,
    model: Any,
    tokenizer: Any,
    model_name: str,
    image: Any,
    grid_image_path: str,
    prompt: str,
    question: str,
    out_dir: str,
    image_id: int,
    ref_names: Sequence[str],
    grid_meta: Optional[Json],
    top_n: int,
    continuation_tokens: int,
    skip_special_tokens: bool,
    use_nucleus_sampling: bool,
    num_beams: int = 1,
) -> Json:
    """Probe the exact eRVCD negative grid image and save JSON + combined figure."""
    case_dir = _ensure_dir(os.path.join(out_dir, f"image_{int(image_id)}"))
    copied_grid_path = os.path.join(case_dir, "actual_negative_grid.png")
    try:
        shutil.copy2(grid_image_path, copied_grid_path)
    except Exception:
        copied_grid_path = grid_image_path

    first_logits = _grid_probe_next_token_logits(
        model=model,
        model_name=model_name,
        image=image,
        prompt=prompt,
        image_path=grid_image_path,
        prev_tokens=[],
        use_nucleus_sampling=use_nucleus_sampling,
        num_beams=num_beams,
    )

    topn = _grid_probe_topn_from_logits(
        logits=first_logits,
        tokenizer=tokenizer,
        top_n=top_n,
        skip_special_tokens=skip_special_tokens,
    )

    continuations: List[Json] = []
    for item in topn:
        cont = _grid_probe_greedy_continue_from_first_token(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            image=image,
            prompt=prompt,
            image_path=grid_image_path,
            first_token_id=int(item["token_id"]),
            total_tokens=continuation_tokens,
            use_nucleus_sampling=use_nucleus_sampling,
            num_beams=num_beams,
        )
        cont["rank"] = item["rank"]
        cont["first_token_probability"] = item["probability"]
        continuations.append(cont)

    combined_figure_path = os.path.join(case_dir, "actual_negative_grid_vlm_distribution.png")
    _grid_probe_save_combined_figure(
        grid_image_path=copied_grid_path,
        topn=topn,
        continuations=continuations,
        save_path=combined_figure_path,
        title=f"VLM first-token distribution for actual negative grid / image_id={int(image_id)}",
        ref_names=ref_names,
    )

    result: Json = {
        "image_id": int(image_id),
        "model_name": model_name,
        "question": question,
        "prompt": prompt,
        "grid_image_path_original": grid_image_path,
        "grid_image_path_copied": copied_grid_path,
        "combined_figure_path": combined_figure_path,
        "ref_names": list(ref_names),
        "grid_meta": grid_meta,
        "top_n": int(top_n),
        "continuation_tokens": int(continuation_tokens),
        "topn_first_token_probs": topn,
        "forced_continuations": continuations,
    }

    json_path = os.path.join(case_dir, "actual_negative_grid_vlm_probe.json")
    _write_json(json_path, result)
    result["json_path"] = json_path

    _append_jsonl(os.path.join(out_dir, "actual_grid_vlm_probe_records.jsonl"), result)
    return result


# ============================================================
# End actual eRVCD grid VLM probe utilities
# ============================================================



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

############ eRVCD grid options #############
parser.add_argument(
    "--ervcd_grid_fill_mode",
    type=str,
    default="black_back",
    choices=["black_back", "black_front", "repeat", "repeat_front", "repeat_last"],
    help=(
        "eRVCD reference grid empty-cell fill mode. "
        "black_back: reference images first, black cells last. "
        "black_front: black cells first, reference images last. "
        "repeat: reference images first, then cycle references to fill. "
        "repeat_front: cycle references first, then reference images. "
        "repeat_last: reference images first, then repeat the last reference."
    ),
)
parser.add_argument(
    "--ervcd_grid_canvas_size",
    type=int,
    default=336,
    help="Final square canvas size for merged reference grid images. Default: 336.",
)
parser.add_argument(
    "--ervcd_logit_scale_mode",
    type=str,
    default="presence",
    choices=["presence", "count"],
    help=(
        "How to scale aggregated N/P logits. "
        "presence: merged N and merged P each count as one logit. "
        "count: approximate original count scaling by multiplying the merged logit by the number of original refs."
    ),
)

############ negative-signal probe options #############
parser.add_argument("--negative_probe_enabled", type=str2bool, default=True, help="Run online negative-signal fail-case analysis while generating captions.")
parser.add_argument("--negative_probe_top_k", type=int, default=20, help="Top-k tokens to store per decoding step in the negative-signal probe.")
parser.add_argument("--negative_probe_plot_top_n", type=int, default=30, help="Number of fail cases to visualize after generation finishes.")
parser.add_argument("--negative_probe_save_all_steps", type=str2bool, default=True, help="Save step_trace.jsonl for every decoding step. Set False to save less disk space.")
parser.add_argument("--negative_probe_no_plots", type=str2bool, default=False, help="If True, skip automatic plot generation at the end.")

############ actual negative-grid VLM probe options #############
parser.add_argument("--grid_vlm_probe_enabled", type=str2bool, default=True, help="Probe the exact negative grid image used by eRVCD and save a combined grid+distribution figure.")
parser.add_argument("--grid_vlm_probe_max_images", type=int, default=20, help="Maximum number of negative grids to probe. Use 0 or negative for unlimited.")
parser.add_argument("--grid_vlm_probe_top_n", type=int, default=10, help="Top-N first-token candidates to visualize for the actual negative grid VLM probe.")
parser.add_argument("--grid_vlm_probe_continuation_tokens", type=int, default=10, help="Total generated tokens per forced-first-token continuation in the actual negative grid VLM probe.")
parser.add_argument("--grid_vlm_probe_question", type=str, default="What object is shown in this image? Answer with one word.", help="Question used when probing the actual negative grid image.")
parser.add_argument("--grid_vlm_probe_skip_special_tokens", type=str2bool, default=True, help="Skip special tokens in the actual negative grid VLM probe top-N.")


################################
args = parser.parse_known_args()[0]
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)

###################################
yolo_version = args.yolo_version
model_name = args.model
decoding_strategy = 'ervcd'
seed = args.seed
num_samples = args.num_samples
dataset_name = args.dataset_name
data_path = args.data_path
chair_cache_path = args.chair_cache_path
output_dir = args.output_dir
num_beams = 1
batch_size = 1
max_new_tokens = args.max_new_tokens

# YOLOv8x는 이미지마다 새로 로드하지 않고, 스크립트 시작 시 1회만 로드한다.
yolo_model = None
if yolo_version == 'yolov8x.pt':
    print(f"Loading YOLO once: {yolo_version}")
    yolo_model = yolo.load_yolo_model(yolo_version)


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


def process_pil_before_norm(raw_image):
        """이미 디스크에 저장하지 않은 PIL Image를 바로 모델 입력 tensor로 변환한다."""
        raw_image = raw_image.convert('RGB')
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
    """yolo.py의 run_inference를 사용하되, YOLO 모델은 루프 밖에서 1회만 로드한다."""
    bounding_boxes, probabilities, entity_names, _ = yolo.run_inference(
        yolo_model,
        image_path,
    )
    unique_items = {}
    for name, prob in zip(entity_names, probabilities):
        if name not in unique_items or prob > unique_items[name]:
            unique_items[name] = prob
    return [(entity, probability) for entity, probability in unique_items.items()]


def _pil_resample_lanczos():
    """Pillow version compatibility helper."""
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _build_ervcd_grid_slots(ref_paths, total_slots, fill_mode):
    """
    ref_paths를 grid slot 수에 맞게 배치한다.
    None은 검은 칸을 의미한다.
    """
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

    raise ValueError(f"Unsupported eRVCD grid fill mode: {fill_mode}")


def make_ervcd_reference_grid_image(
    ref_paths,
    save_path,
    canvas_size=336,
    fill_mode="black_back",
    background_color=(0, 0, 0),
):
    """
    여러 single-concept reference image를 하나의 square grid image로 합친다.

    예:
    - ref 3개 -> 2x2 grid
    - ref 5개 -> 3x3 grid

    반환:
    - save_path: 저장된 grid image path. ref_paths가 비어있으면 None.
    - meta: grid 구성 정보 dict.
    - canvas: 디스크 재로딩 없이 바로 전처리할 수 있는 PIL Image.
    """
    ref_paths = list(ref_paths)
    num_refs = len(ref_paths)

    if num_refs == 0:
        return None, {
            "num_refs": 0,
            "grid_side": 0,
            "total_slots": 0,
            "fill_mode": fill_mode,
            "canvas_size": canvas_size,
            "save_path": None,
        }, None

    grid_side = int(math.ceil(math.sqrt(num_refs)))
    total_slots = grid_side * grid_side
    slot_paths = _build_ervcd_grid_slots(ref_paths, total_slots, fill_mode)

    canvas = Image.new("RGB", (canvas_size, canvas_size), background_color)
    resample = _pil_resample_lanczos()

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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)

    return save_path, {
        "num_refs": num_refs,
        "grid_side": grid_side,
        "total_slots": total_slots,
        "num_empty_slots": total_slots - num_refs,
        "fill_mode": fill_mode,
        "canvas_size": canvas_size,
        "save_path": save_path,
        "slot_paths": slot_paths,
    }, canvas


def aggregate_ervcd_logits(logits, raw_ref_count, logit_scale_mode):
    """
    eRVCD에서는 N/P를 각각 하나의 grid image로 합치므로 logits 길이는 보통 0 또는 1이다.

    presence: grid logit을 하나의 통합 logit으로 그대로 사용한다.
    count: grid logit 하나를 raw reference 개수만큼 곱해 원래 RVCD의 count scaling을 근사한다.
    """
    if len(logits) == 0:
        return 0, 0

    if logit_scale_mode == "presence":
        return logits[0], 1

    if logit_scale_mode == "count":
        effective_count = max(1, int(raw_ref_count))
        return logits[0] * effective_count, effective_count

    raise ValueError(f"Unsupported eRVCD logit scale mode: {logit_scale_mode}")


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
result_dir = os.path.join(
    base_dir,
    f'ervcd_a{args.rvcd_alpha}_b{args.rvcd_beta}_grid_{args.ervcd_grid_fill_mode}_scale_{args.ervcd_logit_scale_mode}_{formatted_time}_seed_{seed}_samples_{num_samples}_maxtokens_{max_new_tokens}_{true_flag_name}'
)
if not os.path.exists(result_dir): os.makedirs(result_dir)

probe = None
if args.negative_probe_enabled:
    probe = NegativeSignalProbe(
        out_dir=result_dir,
        tokenizer=model_tokenizer,
        chair_evaluator=get_chair_evaluator(chair_cache_path, coco_path),
        top_k=args.negative_probe_top_k,
        save_all_steps_jsonl=args.negative_probe_save_all_steps,
    )
    print(f"[NegativeSignalProbe] enabled. Output dir: {probe.root}")

grid_vlm_probe_saved_count = 0

global_all_info = {
    'model_name' : model_name,
    'decoding_strategy' : 'ervcd',
    'seed' : seed,
    'num_samples' : num_samples,
    'max_new_tokens' : max_new_tokens,
    'dataset_name' : dataset_name,
    'data_path' : data_path,
    'output_dir' : output_dir,
    'num_beams' : num_beams,
    'batch_size' : batch_size,
    'ervcd_grid_fill_mode' : args.ervcd_grid_fill_mode,
    'ervcd_grid_canvas_size' : args.ervcd_grid_canvas_size,
    'ervcd_logit_scale_mode' : args.ervcd_logit_scale_mode,
    'grid_vlm_probe_enabled' : args.grid_vlm_probe_enabled,
    'grid_vlm_probe_max_images' : args.grid_vlm_probe_max_images,
    'grid_vlm_probe_top_n' : args.grid_vlm_probe_top_n,
    'grid_vlm_probe_continuation_tokens' : args.grid_vlm_probe_continuation_tokens,
    'grid_vlm_probe_question' : args.grid_vlm_probe_question,
    'grid_vlm_probe_records' : [],
    'ervcd_grid_records' : [],
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

    # NegativeSignalProbe에서 fail-case 매핑에 사용할 객체 pair 정보.
    hal_detected_pairs = []
    gt_detected_pairs = []

    # path별 normalized image tensor cache.
    # 원본/N-grid/P-grid 이미지를 토큰마다 다시 Image.open + preprocess 하지 않도록 한다.
    image_tensor_cache = {}

    def get_cached_norm_image(path, pil_image=None):
        if path not in image_tensor_cache:
            if pil_image is None:
                image_tensor_cache[path] = norm(process_before_norm(path))
            else:
                image_tensor_cache[path] = norm(process_pil_before_norm(pil_image))
        return image_tensor_cache[path]

    image = get_cached_norm_image(image_path)

    # CHAIR, BLEU default image captioning propmt. 
    qu = "Please describe this image in detail."
    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)

    # DRAFT caption generation 
    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": image, "prompt":qu, "img_path": image_path},
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
            if yolo_model is not None:
                yolo_detected_entity_prob = run_yolov8_once_loaded(yolo_model, input_image_path)
            else:
                # fallback: 기존 yolo.main 경로 유지
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
            chair_evaluator = get_chair_evaluator(chair_cache_path, coco_path)
            draft_synonyms = chair_evaluator.process_sentence_get_coco_synonyms(draft_output_text)
            for synonym in draft_synonyms:
                # synonym = (cocofirstword, synonymfromdraft) 
                if synonym[0] in yolo_detected_entity_list: detected_info[synonym] = 1
                else: detected_info[synonym] = 0

            print(f'detected_info : {detected_info}') 

            # NegativeSignalProbe용 pair 정보: key == (COCO 대표어, draft caption surface form)
            hal_detected_pairs = []
            gt_detected_pairs = []
            for key, value in detected_info.items():
                item = {"coco": key[0], "surface": key[1]}
                if value == 0:
                    hal_detected_pairs.append(item)
                else:
                    gt_detected_pairs.append(item)

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

    # eRVCD: 여러 reference image를 각각 하나의 N-grid/P-grid 이미지로 합친다.
    # 원래 RVCD는 N/P reference 개수만큼 매 디코딩 스텝 forward하지만,
    # eRVCD는 negative grid 1회, positive grid 1회만 forward한다.
    ervcd_grid_dir = os.path.join(result_dir, "ervcd_grid_refs")
    negative_grid_path = None
    positive_grid_path = None
    negative_grid_meta = None
    positive_grid_meta = None
    negative_grid_canvas = None
    positive_grid_canvas = None

    if len(hall_ref_list) > 0:
        negative_grid_path, negative_grid_meta, negative_grid_canvas = make_ervcd_reference_grid_image(
            hall_ref_list,
            os.path.join(ervcd_grid_dir, f"image_{int(img_id)}_negative_N_grid.png"),
            canvas_size=args.ervcd_grid_canvas_size,
            fill_mode=args.ervcd_grid_fill_mode,
        )

    if len(gt_ref_list) > 0 and args.rvcd_beta != 0:
        positive_grid_path, positive_grid_meta, positive_grid_canvas = make_ervcd_reference_grid_image(
            gt_ref_list,
            os.path.join(ervcd_grid_dir, f"image_{int(img_id)}_positive_P_grid.png"),
            canvas_size=args.ervcd_grid_canvas_size,
            fill_mode=args.ervcd_grid_fill_mode,
        )

    global_all_info['ervcd_grid_records'].append({
        "image_id": int(img_id),
        "negative_raw_ref_count": len(hall_ref_list),
        "positive_raw_ref_count": len(gt_ref_list) if args.rvcd_beta != 0 else 0,
        "negative_grid": negative_grid_meta,
        "positive_grid": positive_grid_meta,
    })

    print(f'negative_grid_path : {negative_grid_path}')
    print(f'positive_grid_path : {positive_grid_path}')

    # grid는 저장은 하되, 모델 입력은 방금 만든 PIL canvas에서 바로 전처리해 cache에 넣는다.
    # 따라서 eRVCD loop에서 grid png를 다시 Image.open 하지 않는다.
    if negative_grid_path is not None and negative_grid_canvas is not None:
        get_cached_norm_image(negative_grid_path, negative_grid_canvas)

        # Actual-grid VLM probe:
        # Probe the exact negative reference grid image used by eRVCD and save
        # a wide figure with the grid on the left and first-token top-N distribution on the right.
        if args.grid_vlm_probe_enabled and (
            args.grid_vlm_probe_max_images <= 0 or grid_vlm_probe_saved_count < args.grid_vlm_probe_max_images
        ):
            try:
                grid_vlm_probe_dir = os.path.join(result_dir, "actual_negative_grid_vlm_probe")
                grid_probe_prompt = template.replace("<question>", args.grid_vlm_probe_question)
                grid_probe_ref_names = [os.path.splitext(os.path.basename(p))[0] for p in hall_ref_list]
                grid_probe_image = get_cached_norm_image(negative_grid_path, negative_grid_canvas)
                grid_probe_record = run_actual_grid_vlm_probe(
                    model=model,
                    tokenizer=model_tokenizer,
                    model_name=model_name,
                    image=grid_probe_image,
                    grid_image_path=negative_grid_path,
                    prompt=grid_probe_prompt,
                    question=args.grid_vlm_probe_question,
                    out_dir=grid_vlm_probe_dir,
                    image_id=int(img_id),
                    ref_names=grid_probe_ref_names,
                    grid_meta=negative_grid_meta,
                    top_n=args.grid_vlm_probe_top_n,
                    continuation_tokens=args.grid_vlm_probe_continuation_tokens,
                    skip_special_tokens=args.grid_vlm_probe_skip_special_tokens,
                    use_nucleus_sampling=args.sample,
                    num_beams=num_beams,
                )
                global_all_info['grid_vlm_probe_records'].append({
                    "image_id": int(img_id),
                    "ref_names": grid_probe_ref_names,
                    "json_path": grid_probe_record.get("json_path"),
                    "combined_figure_path": grid_probe_record.get("combined_figure_path"),
                    "top1_decoded": grid_probe_record["topn_first_token_probs"][0]["decoded"] if grid_probe_record.get("topn_first_token_probs") else "",
                    "top1_probability": grid_probe_record["topn_first_token_probs"][0]["probability"] if grid_probe_record.get("topn_first_token_probs") else None,
                })
                grid_vlm_probe_saved_count += 1
                print(f"[GridVLMProbe] saved actual negative-grid distribution: {grid_probe_record.get('combined_figure_path')}")
            except Exception as e:
                print(f"[GridVLMProbe][WARN] failed for image_id={img_id}: {repr(e)}")
    if positive_grid_path is not None and positive_grid_canvas is not None:
        get_cached_norm_image(positive_grid_path, positive_grid_canvas)
   
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
            negative_img_path = [negative_grid_path] if negative_grid_path is not None else []
            positive_img_path = [positive_grid_path] if positive_grid_path is not None else []


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

            # 원본 이미지 v와 eRVCD negative grid image N_agg에 대해
            for path in [original_img_path]+negative_img_path:
                
                image = get_cached_norm_image(path) #원본 이미지와 N 이미지들.
                kv_cache = image_kv_cache.get(path, None)

                ##############################################################
                output = model.generate(
                    {"image": image, "prompt": qu, "img_path": path},
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

            
            # eRVCD positive grid image P_agg에 대해
            ##############################################################
            for path in positive_img_path:
                image = get_cached_norm_image(path) #P 이미지들.
                kv_cache = image_kv_cache.get(path, None)

                ##############################################################
                output = model.generate(
                    {"image": image, "prompt": qu, "img_path": path},
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
            
            # eRVCD 핵심 변경점:
            # 기존 RVCD: N/P reference image 각각의 logits를 모두 더함.
            # eRVCD: N/P reference image들을 하나의 grid image로 합쳤으므로,
            #        negative_logits와 positive_logits는 각각 최대 1개만 가진다.
            sum_negative_logits, negative_logits_count = aggregate_ervcd_logits(
                negative_logits,
                raw_ref_count=len(hall_ref_list),
                logit_scale_mode=args.ervcd_logit_scale_mode,
            )
            sum_positive_logits, positive_logits_count = aggregate_ervcd_logits(
                positive_logits,
                raw_ref_count=len(gt_ref_list) if args.rvcd_beta != 0 else 0,
                logit_scale_mode=args.ervcd_logit_scale_mode,
            )

            print(f"eRVCD raw N/P ref count : {len(hall_ref_list)}, {len(gt_ref_list) if args.rvcd_beta != 0 else 0}")
            print(f"eRVCD effective N/P logit count : {negative_logits_count}, {positive_logits_count}")
        
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

            if probe is not None:
                probe.log_step(
                    image_id=int(img_id),
                    step=int(output_index),
                    selected_token_id=int(max_index.squeeze().item()),
                    original_logit=original_logit,
                    negative_logits=negative_logits,
                    adjusted_logits=adjusted_logits,
                    hal_objects=hal_detected_pairs,
                    negative_grid_path=negative_grid_path,
                    negative_grid_meta=negative_grid_meta,
                    alpha=float(alpha),
                    negative_logits_count=int(negative_logits_count),
                    prefix_token_ids=[
                        int(x.item()) if hasattr(x, "item") else int(x)
                        for x in output_tokens
                    ],
                    keep_topk=True,
                )

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

    if probe is not None and nvcd_operate:
        probe.finalize_datapoint(
            image_id=int(img_id),
            draft_caption=now_datapoint_draft_caption,
            final_caption=now_datapoint_final_caption,
            output_token_ids=[
                int(x.item()) if hasattr(x, "item") else int(x)
                for x in output_tokens
            ],
            hal_objects=hal_detected_pairs,
            negative_grid_path=negative_grid_path,
            negative_grid_meta=negative_grid_meta,
            extra={
                "image_path": image_path,
                "gt_detected": gt_detected_pairs,
                "hal_detected": hal_detected_pairs,
                "grid_fill_mode": args.ervcd_grid_fill_mode,
                "logit_scale_mode": args.ervcd_logit_scale_mode,
                "rvcd_alpha": args.rvcd_alpha,
                "rvcd_beta": args.rvcd_beta,
            },
        )

    now_draft_result = {"image_id": int(img_id),"caption": now_datapoint_draft_caption}
    draft_captions_path = os.path.join(result_dir,f"ervcd_{model_name}_{formatted_time}_DRAFT_generated_captions.jsonl")
    with open(draft_captions_path, "a") as f:
        json.dump(now_draft_result, f)
        f.write("\n")

    now_nvcd_result = {"image_id": int(img_id),"caption": now_datapoint_final_caption,"tokens": token_count}
    global_all_info['total_generated_tokens'] += token_count
    nvcd_captions_path = os.path.join(result_dir,f'ervcd_{model_name}_a{args.rvcd_alpha}_b{args.rvcd_beta}_grid_{args.ervcd_grid_fill_mode}_scale_{args.ervcd_logit_scale_mode}_{formatted_time}_seed_{seed}_samples_{num_samples}_maxtokens_{max_new_tokens}_{true_flag_name}_generated_captions.jsonl')
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

    if model_name != 'minigpt4':
        global_all_info['latency_per_token'] = global_all_info['latency'] / global_all_info['total_generated_tokens']


global_info_save_path = os.path.join(result_dir,f"ervcd_{model_name}_a{args.rvcd_alpha}_b{args.rvcd_beta}_grid_{args.ervcd_grid_fill_mode}_scale_{args.ervcd_logit_scale_mode}_{formatted_time}_seed_{seed}_samples_{num_samples}_maxtokens_{max_new_tokens}_{true_flag_name}_DETECTOR_info.json")
with open(global_info_save_path, 'w', encoding='utf-8') as json_file:
    json.dump(global_all_info, json_file, indent=4, ensure_ascii=False)

if probe is not None:
    fail_jsonl_path = os.path.join(probe.root, "fail_cases.jsonl")
    plots_dir = os.path.join(probe.root, "plots")
    if os.path.exists(fail_jsonl_path):
        num_fail_lines = sum(1 for _ in open(fail_jsonl_path, "r", encoding="utf-8"))
        print(f"[NegativeSignalProbe] fail cases saved: {num_fail_lines} -> {fail_jsonl_path}")
        if not args.negative_probe_no_plots:
            viz_result = visualize_fail_cases(
                fail_jsonl=fail_jsonl_path,
                out_dir=plots_dir,
                top_n=args.negative_probe_plot_top_n,
            )
            print("[NegativeSignalProbe] visualization result:")
            print(json.dumps(viz_result, ensure_ascii=False, indent=2))
    else:
        print(f"[NegativeSignalProbe] no fail cases found. Expected path not created: {fail_jsonl_path}")


# CUDA_VISIBLE_DEVICES=0 \
# python ervcd_generation_chair_bleu.py \
# --model llava-1.5 \
# --data_path /home/jihoon/jihoon/DATASETS/coco2014/val2014 \
# --ref_folder_path DB_single_concept_images_flux_generated/generated_images \
# --chair_cache_path eval/CHAIR_CACHE/chair.pkl \
# --num_samples 300 \
# --seed 42 \
# --gpu-id 0 \
# --output_dir ./generated_captions_test_blackfront/ \
# --rvcd_alpha 1 \
# --rvcd_beta 0.1 \
# --ervcd_grid_fill_mode black_front \
# --ervcd_logit_scale_mode presence

# CUDA_VISIBLE_DEVICES=0 \
# python ervcd_generation_chair_bleu.py \
# --model llava-1.5 \
# --data_path /home/jihoon/jihoon/DATASETS/coco2014/val2014 \
# --ref_folder_path DB_single_concept_images_flux_generated/generated_images \
# --chair_cache_path eval/CHAIR_CACHE/chair.pkl \
# --num_samples 300 \
# --seed 42 \
# --gpu-id 0 \
# --output_dir ./generated_captions_test_repeatlast/ \
# --rvcd_alpha 1 \
# --rvcd_beta 0.1 \
# --ervcd_grid_fill_mode repeat_last \
# --ervcd_logit_scale_mode presence



# black_back    : reference 이미지들을 앞에서부터 넣고, 남는 칸은 뒤쪽 검은 칸
# black_front   : 남는 검은 칸을 앞에 두고, reference 이미지들을 뒤쪽에 배치
# repeat        : reference 이미지들을 먼저 넣고, 남는 칸은 앞 이미지부터 반복
# repeat_front  : 반복 이미지들을 앞에 채우고, reference 이미지들을 뒤쪽에 배치
# repeat_last   : reference 이미지들을 먼저 넣고, 남는 칸은 마지막 reference 이미지 반복




# CUDA_VISIBLE_DEVICES=0 \
# python ervcd_generation_chair_bleu_with_actual_grid_probe.py \
#   --model llava-1.5 \
#   --data_path /home/jihoon/jihoon/DATASETS/coco2014/val2014 \
#   --ref_folder_path DB_single_concept_images_flux_generated/generated_images \
#   --chair_cache_path eval/CHAIR_CACHE/chair.pkl \
#   --num_samples 300 \
#   --seed 42 \
#   --gpu-id 0 \
#   --output_dir ./generated_captions_probe_test_260529/ \
#   --rvcd_alpha 1 \
#   --rvcd_beta 0 \
#   --ervcd_grid_fill_mode black_front \
#   --ervcd_logit_scale_mode presence \
#   --grid_vlm_probe_max_images 0 \
#   --grid_vlm_probe_enabled true \
#   --grid_vlm_probe_top_n 10 \
#   --grid_vlm_probe_continuation_tokens 3