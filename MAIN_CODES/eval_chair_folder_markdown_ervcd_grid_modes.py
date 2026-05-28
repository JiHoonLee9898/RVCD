#!/usr/bin/env python3
"""
eval_chair_folder_markdown_ervcd.py

Compact RVCD CHAIR evaluator with copy-friendly Markdown table output.

Input:
  --folder can be a top-level folder such as:
    /home/jihoon/jihoon/RVCD/MAIN_CODES/generated_captions

Behavior:
  1. Recursively finds non-DRAFT:
       *_generated_captions.jsonl
       *_generated_captions.json
  2. If the corresponding *_chair.json is missing, runs:
       eval/caption_to_chair2.py --gt-caption-path ... -c <that folder>
  3. Runs:
       eval/eval_hallucination.py -v --metric chair --chair_input_path ...
  4. Prints compact output:
       run/name
       chairs: 14.00% | chairi: 3.51% | bleu: 15.81% | latency: 1.23 sec

Defaults:
  - DRAFT files are ignored.
  - No wrapper logs/csv/json are written unless --save-csv is provided.
  - eval_hallucination output_dir uses a temporary directory by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from statistics import mean, stdev
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


GENERATED_RE = re.compile(r"generated_captions\.(?:jsonl|json)$", re.IGNORECASE)
INFO_RE = re.compile(r"(?:info|INFO)\.json$", re.IGNORECASE)

SCRIPT_VERSION = "2026-05-26-v6-ervcd-grid-mode-rows"

# eRVCD grid fill modes used by ervcd_generation_chair_bleu.py.
# Keep longer repeat_* names before repeat when regex matching.
ERVCD_GRID_MODES = ["black_back", "black_front", "repeat", "repeat_front", "repeat_last"]
ERVCD_MODE_METHODS = [f"eRVCD, {mode}" for mode in ERVCD_GRID_MODES]

DEFAULT_MD_METHODS = ["Greedy", "Beam Search", "DoLA", "OPERA", "VCD", "RVCD", "eRVCD"]
DEFAULT_ERVCD_MD_METHODS = ERVCD_MODE_METHODS


def is_draft(path: Path) -> bool:
    return "DRAFT" in path.name.upper()


def resolve_main_codes(user_value: Optional[str]) -> Path:
    candidates: List[Path] = []
    if user_value:
        candidates.append(Path(user_value).expanduser())

    here = Path(__file__).resolve()
    cwd = Path.cwd()
    candidates.extend([
        cwd,
        cwd / "MAIN_CODES",
        here.parent,
        here.parent.parent,
        Path("/home/jihoon/jihoon/RVCD/MAIN_CODES"),
    ])

    for c in candidates:
        c = c.resolve()
        if (c / "eval" / "caption_to_chair2.py").exists() and (c / "eval" / "eval_hallucination.py").exists():
            return c

    raise FileNotFoundError("Could not locate RVCD MAIN_CODES. Pass --main-codes /path/to/RVCD/MAIN_CODES")


def run_capture(cmd: List[str], cwd: Path, verbose: bool = False) -> Tuple[int, str]:
    if verbose:
        print("\n[CMD]", " ".join(map(str, cmd)))
        print("[CWD]", cwd)

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out = proc.stdout or ""

    if verbose and out.strip():
        print(out)

    return proc.returncode, out


def discover_generated_files(root: Path, include_draft: bool) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jsonl", ".json"}:
            continue
        if not GENERATED_RE.search(p.name):
            continue
        if not include_draft and is_draft(p):
            continue
        files.append(p.resolve())
    return sorted(files)


def group_by_parent(paths: Iterable[Path]) -> Dict[Path, List[Path]]:
    grouped: Dict[Path, List[Path]] = {}
    for p in paths:
        grouped.setdefault(p.parent, []).append(p)
    return dict(sorted(grouped.items(), key=lambda kv: str(kv[0])))


def expected_chair_path(generated_path: Path) -> Path:
    name = generated_path.name
    if name.endswith("_generated_captions.jsonl"):
        chair_name = name[: -len("_generated_captions.jsonl")] + "_chair.json"
    elif name.endswith("_generated_captions.json"):
        chair_name = name[: -len("_generated_captions.json")] + "_chair.json"
    elif name.endswith(".jsonl"):
        chair_name = name[: -len(".jsonl")] + "_chair.json"
    elif name.endswith(".json"):
        chair_name = name[: -len(".json")] + "_chair.json"
    else:
        chair_name = generated_path.stem + "_chair.json"
    return generated_path.with_name(chair_name).resolve()


def json_to_jsonl_alias_if_needed(generated_path: Path) -> Optional[Path]:
    """
    caption_to_chair2.py often scans for *_generated_captions.jsonl.
    If the model output is *_generated_captions.json, create a temporary sibling
    .jsonl file with the same records, then remove it after conversion.
    """
    if generated_path.suffix.lower() == ".jsonl":
        return None

    alias = generated_path.with_suffix(".jsonl")
    if alias.exists():
        return None

    text = generated_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"Empty generated captions file: {generated_path}")

    records: List[Any]
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            records = obj
        elif isinstance(obj, dict):
            records = []
            for key in ("annotations", "results", "captions", "data", "samples", "outputs"):
                val = obj.get(key)
                if isinstance(val, list):
                    records = val
                    break
            if not records:
                records = [obj]
        else:
            records = [obj]
    except json.JSONDecodeError:
        # It may already be JSONL but saved with .json extension.
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    with alias.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return alias.resolve()


def common_prefix_score(a: str, b: str) -> float:
    a = a.lower()
    b = b.lower()
    ratio = SequenceMatcher(None, a, b).ratio()
    prefix_len = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix_len += 1
        else:
            break
    return ratio + min(prefix_len / max(len(a), 1), 1.0)


def find_non_draft_chair_files(folder: Path, include_draft: bool) -> List[Path]:
    out = []
    for p in folder.glob("*_chair.json"):
        if not p.is_file():
            continue
        if not include_draft and is_draft(p):
            continue
        out.append(p.resolve())
    return sorted(out)


def find_info_jsons(folder: Path) -> List[Path]:
    return sorted([p.resolve() for p in folder.glob("*.json") if p.is_file() and INFO_RE.search(p.name)])


def pick_best_info(chair_path: Path, info_files: List[Path]) -> Optional[Path]:
    if not info_files:
        return None
    chair_stem = chair_path.stem.replace("_chair", "")
    return max(info_files, key=lambda p: common_prefix_score(chair_stem, p.stem))


def load_json_maybe(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        vals = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    vals.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return vals if vals else None


def count_records_from_json(path: Path) -> Optional[int]:
    try:
        obj = load_json_maybe(path)
    except Exception:
        return None

    if isinstance(obj, list):
        return len(obj)

    if isinstance(obj, dict):
        for key in ("annotations", "results", "captions", "data", "samples", "images"):
            val = obj.get(key)
            if isinstance(val, list):
                return len(val)

    return None


def flatten_latency_candidates(obj: Any) -> List[Tuple[str, float, str]]:
    candidates: List[Tuple[str, float, str]] = []

    include_terms = (
        "latency", "time", "sec", "second", "duration", "elapsed", "runtime",
        "inference", "generate", "generation", "detector", "dino",
    )
    exclude_terms = (
        "image_id", "img_id", "id", "seed", "token", "tokens", "max_tokens",
        "sample", "samples", "count", "num", "index", "idx",
    )

    def interesting(path: str) -> bool:
        p = path.lower()
        return any(t in p for t in include_terms) and not any(t in p for t in exclude_terms)

    def walk(x: Any, path: str) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(x, list):
            nums = [v for v in x if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if nums and interesting(path):
                candidates.append((path, float(sum(nums) / len(nums)), f"list_mean_n={len(nums)}"))
                candidates.append((path, float(sum(nums)), f"list_sum_n={len(nums)}"))
            for i, v in enumerate(x[:2000]):
                if isinstance(v, (dict, list)):
                    walk(v, f"{path}[{i}]")
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            if interesting(path):
                candidates.append((path, float(x), "scalar"))

    walk(obj, "")

    def rank(item: Tuple[str, float, str]) -> Tuple[int, int, str]:
        path, _, kind = item
        p = path.lower()
        score = 100
        if "avg" in p or "mean" in p:
            score -= 50
        if "latency" in p:
            score -= 40
        if kind.startswith("list_mean"):
            score -= 30
        if "per" in p:
            score -= 15
        if "total" in p or "sum" in p:
            score += 20
        if "time" in p:
            score -= 5
        return (score, len(path), path)

    seen = set()
    out = []
    for c in sorted(candidates, key=rank):
        key = (c[0], c[2])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def latency_from_info(info_path: Optional[Path], chair_path: Path) -> Optional[float]:
    if not info_path:
        return None
    try:
        obj = load_json_maybe(info_path)
    except Exception:
        return None

    candidates = flatten_latency_candidates(obj)
    if not candidates:
        return None

    path, value, kind = candidates[0]

    # Convert obvious total latency to per-sample latency if possible.
    n = count_records_from_json(chair_path)
    if n and n > 0 and ("total" in path.lower() or kind.startswith("list_sum")):
        return float(value) / float(n)

    return float(value)


def parse_eval_metrics(output: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    names = [
        "CHAIRs", "CHAIRi", "CHAIR", "chairs", "chairi", "bleu", "hallucinate_sum",
        "Recall", "Len", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4",
        "METEOR", "ROUGE_L", "CIDEr", "SPICE",
    ]

    for name in names:
        patterns = [
            rf"{re.escape(name)}\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            rf"['\"]{re.escape(name)}['\"]\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        ]
        for pat in patterns:
            m = re.search(pat, output)
            if m:
                try:
                    metrics[name] = float(m.group(1))
                except Exception:
                    pass
                break

    for m in re.finditer(
        r"['\"]([^'\"]*(?:CHAIR|chair|Bleu|bleu|Recall|METEOR|ROUGE|CIDEr|SPICE|Len)[^'\"]*)['\"]\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        output,
    ):
        try:
            metrics.setdefault(m.group(1), float(m.group(2)))
        except Exception:
            pass

    return metrics


def pick_metric(metrics: Dict[str, float], *names: str) -> Optional[float]:
    lower = {str(k).lower(): v for k, v in metrics.items()}
    for name in names:
        val = lower.get(name.lower())
        if val is not None:
            return float(val)
    return None


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    pct = value * 100.0 if abs(value) <= 1.0 else value
    return f"{pct:.2f}%"


def fmt_sec(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f} sec"





def fmt_pct_md(value: Optional[float], blank: str = "", percent_symbol: bool = True) -> str:
    if value is None:
        return blank
    pct = value * 100.0 if abs(value) <= 1.0 else value
    return f"{pct:.2f}%" if percent_symbol else f"{pct:.2f}"


def parse_csv_items(value: Optional[str]) -> List[str]:
    """Parse user-specified Markdown row order.

    The old script used comma-separated values only, but eRVCD mode labels are
    intentionally formatted as e.g. "eRVCD, black_back". Therefore semicolon is
    preferred whenever labels themselves contain commas.
    """
    if not value:
        return []
    sep = ";" if ";" in value else ","
    return [x.strip() for x in value.split(sep) if x.strip()]


def markdown_escape_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    return text.replace("|", r"\|").strip()


def compact_identifier(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


METHOD_ALIASES: List[Tuple[str, Tuple[str, ...]]] = [
    ("Greedy", ("greedy", "vanilla")),
    ("Beam Search", ("beam", "beam_search", "beam-search", "beamsearch")),
    ("DoLA", ("dola",)),
    ("OPERA", ("opera",)),
    ("VCD", ("vcd",)),
    ("RVCD", ("rvcd",)),
    ("eRVCD", ("ervcd",)),
]

ALIAS_TO_METHOD: Dict[str, str] = {}
for _display, _aliases in METHOD_ALIASES:
    for _alias in _aliases:
        ALIAS_TO_METHOD[re.sub(r"[-\s]+", "_", _alias.lower())] = _display

# This is intentionally strict: it only matches the decoding name when it is
# immediately followed by a timestamp. This avoids false matches from parent
# folders such as "not_rvcd_llava".
RUN_PREFIX_RE = re.compile(
    r"^(?P<method>greedy|beam(?:_search)?|beamsearch|dola|opera|vcd|rvcd|ervcd)[_-](?P<stamp>20\d{6,12})(?:[_-].*)?$",
    re.IGNORECASE,
)

# RVCD runs in your folder may be named like:
#   a1.0_b0.1_202605241357_seed_42_samples_300_maxtokens_64_ablation_None
# without a literal "rvcd" token.
RVCD_AB_RE = re.compile(
    r"^a\d+(?:\.\d+)?_b\d+(?:\.\d+)?[_-]20\d{6,12}(?:[_-].*)?$",
    re.IGNORECASE,
)

# eRVCD runs may be named like:
#   ervcd_a1.0_b0.1_grid_black_back_scale_presence_202605251530_seed_42_...
#   ervcd_llava-1.5_a1.0_b0.1_grid_black_back_scale_presence_202605251530_...
# The timestamp does not have to appear immediately after the method name.
ERVCD_AB_RE = re.compile(
    r"^ervcd[_-].*?a\d+(?:\.\d+)?[_-]b\d+(?:\.\d+)?.*?[_-]20\d{6,12}(?:[_-].*)?$",
    re.IGNORECASE,
)

SUFFIX_RE = re.compile(r"_(?:generated_captions|chair|info)$", re.IGNORECASE)


def normalize_run_component(token: str) -> str:
    raw = str(token).strip().lower()
    # Keep dots inside run names such as a1.0_b0.1_...; only remove file suffixes.
    raw = re.sub(r"\.(?:jsonl|json)$", "", raw, flags=re.IGNORECASE)
    raw = SUFFIX_RE.sub("", raw)
    raw = re.sub(r"[-\s]+", "_", raw)
    return raw.strip("_")


ERVCD_GRID_MODE_RE = re.compile(
    # Order matters: repeat_front / repeat_last must be tried before repeat.
    r"(?:^|_)grid_(?P<mode>black_back|black_front|repeat_front|repeat_last|repeat)(?:_|$)",
    re.IGNORECASE,
)


def ervcd_label_from_norm(norm: str) -> str:
    """Return a method label such as 'eRVCD, black_back' from a run component."""
    m = ERVCD_GRID_MODE_RE.search(norm)
    if m:
        return f"eRVCD, {m.group('mode').lower()}"
    return "eRVCD"


def is_ervcd_run_path(path: Path) -> bool:
    """True if this generated-caption path belongs to an eRVCD run.

    This intentionally checks the leaf/run-like components only. It avoids
    treating a top-level folder such as generated_captions_ervcd_grid_modes as
    proof that every child run is eRVCD.
    """
    candidates: List[str] = [path.name, path.stem, path.parent.name]
    try:
        candidates.extend(path.parts[-4:])
    except Exception:
        pass

    for comp in candidates:
        norm = normalize_run_component(comp)
        if not norm:
            continue
        if norm == "ervcd" or norm.startswith("ervcd_") or ERVCD_AB_RE.match(norm):
            return True
    return False


def method_from_run_component(token: str) -> Optional[str]:
    norm = normalize_run_component(token)
    if not norm:
        return None

    if ERVCD_AB_RE.match(norm):
        return ervcd_label_from_norm(norm)

    if RVCD_AB_RE.match(norm):
        return "RVCD"

    m = RUN_PREFIX_RE.match(norm)
    if not m:
        return None

    method = m.group("method").lower()
    if method in {"beam", "beam_search", "beamsearch"}:
        return "Beam Search"
    if method == "ervcd":
        return ervcd_label_from_norm(norm)
    return ALIAS_TO_METHOD.get(method)


def method_from_exact_component(token: str) -> Optional[str]:
    """Fallback for clean component names such as 'greedy' or 'opera'."""
    norm = normalize_run_component(token)
    if not norm:
        return None

    # Avoid false positives from project/container names.
    if norm in {
        "not_rvcd", "not_rvcd_llava", "non_rvcd", "non_rvcd_llava",
        "not_ervcd", "not_ervcd_llava", "non_ervcd", "non_ervcd_llava",
    }:
        return None
    if norm == "rvcd":
        return None
    if norm == "ervcd":
        return "eRVCD"

    return ALIAS_TO_METHOD.get(norm)


def path_components_for_method(row: Dict[str, Any]) -> List[str]:
    """Candidate path components; leaf run names are checked before parents."""
    comps: List[str] = []

    def add_component(part: str) -> None:
        part = str(part).strip()
        if not part or part in {".", ".."}:
            return
        comps.append(part)
        # Add a simple file stem after the original. For a1.0_b0.1_... this
        # original component is still checked first, so the dots are preserved.
        stem = Path(part).stem
        if stem and stem != part:
            comps.append(stem)

    def add_pathlike(raw: str, reverse: bool = False) -> None:
        if not raw:
            return
        parts = [x for x in re.split(r"[\\/]+", str(raw)) if x and x not in {".", ".."}]
        if reverse:
            parts = parts[::-1]
        for part in parts:
            add_component(part)

    # label is usually: chair/not_rvcd_llava/dola_2026...
    # Reverse it so the leaf folder is checked before not_rvcd_llava.
    add_pathlike(str(row.get("label") or ""), reverse=True)

    for key in ("generated_path", "chair_path", "info_path"):
        raw = str(row.get(key) or "")
        if not raw:
            continue
        try:
            path = Path(raw)
            add_component(path.name)
            add_component(path.stem)
            add_component(path.parent.name)
        except Exception:
            add_pathlike(raw, reverse=True)

    seen = set()
    out: List[str] = []
    for comp in comps:
        if comp not in seen:
            seen.add(comp)
            out.append(comp)
    return out


def infer_method(row: Dict[str, Any]) -> str:
    components = path_components_for_method(row)

    # 1) Strong match: method immediately before timestamp.
    for comp in components:
        method = method_from_run_component(comp)
        if method:
            return method

    # 2) Exact fallback for clean names.
    for comp in components:
        method = method_from_exact_component(comp)
        if method:
            return method

    label = str(row.get("label") or "").strip()
    return label if label else "Unknown"



SEED_RE = re.compile(r"(?:^|[_-])seed[_-]?(?P<seed>\d+)(?:[_-]|$)", re.IGNORECASE)


def seed_from_component(token: str) -> Optional[int]:
    norm = normalize_run_component(token)
    m = SEED_RE.search(norm)
    if not m:
        return None
    try:
        return int(m.group("seed"))
    except Exception:
        return None


def infer_seed(row: Dict[str, Any]) -> Optional[int]:
    for comp in path_components_for_method(row):
        seed = seed_from_component(comp)
        if seed is not None:
            return seed
    return None


def pct_number(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    return v * 100.0 if abs(v) <= 1.0 else v


def fmt_metric_aggregate(
    values: List[Optional[float]],
    blank: str = "",
    percent_symbol: bool = True,
    aggregate: bool = False,
) -> str:
    nums = [pct_number(v) for v in values if v is not None]
    nums = [v for v in nums if v is not None]
    if not nums:
        return blank
    if aggregate and len(nums) >= 2:
        mu = mean(nums)
        sd = stdev(nums)
        return f"{mu:.2f} ± {sd:.2f}%" if percent_symbol else f"{mu:.2f} ± {sd:.2f}"
    v = nums[0]
    return f"{v:.2f}%" if percent_symbol else f"{v:.2f}"


def unique_seed_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return one row per seed, plus duplicate-same-seed rows.

    Aggregation is meant for independent seed runs. If the same seed appears
    multiple times for a method, the first row is used and later same-seed rows
    are treated as duplicates instead of silently changing the mean.
    """
    by_seed: Dict[int, Dict[str, Any]] = {}
    no_seed: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []

    for row in rows:
        seed = infer_seed(row)
        row["seed"] = seed
        if seed is None:
            no_seed.append(row)
        elif seed in by_seed:
            duplicates.append(row)
        else:
            by_seed[seed] = row

    ordered_seed_rows = [by_seed[k] for k in sorted(by_seed)]
    return ordered_seed_rows + no_seed, duplicates


def summarize_method_rows(
    method: str,
    rows: List[Dict[str, Any]],
    blank: str,
    include_latency: bool,
    percent_symbol: bool,
    aggregate_seeds: bool,
    show_n: bool,
) -> List[str]:
    if not rows:
        vals = [method, blank, blank, blank]
        if include_latency:
            vals.append(blank)
        return vals

    seed_rows, _duplicates = unique_seed_rows(rows)
    distinct_seeds = sorted({r.get("seed") for r in seed_rows if r.get("seed") is not None})
    do_agg = bool(aggregate_seeds and len(distinct_seeds) >= 2)

    label = method
    if show_n and do_agg:
        label = f"{method} (n={len(distinct_seeds)})"

    if do_agg:
        used_rows = [r for r in seed_rows if r.get("seed") is not None]
        vals = [
            label,
            fmt_metric_aggregate([r.get("chairs") for r in used_rows], blank=blank, percent_symbol=percent_symbol, aggregate=True),
            fmt_metric_aggregate([r.get("chairi") for r in used_rows], blank=blank, percent_symbol=percent_symbol, aggregate=True),
            fmt_metric_aggregate([r.get("bleu") for r in used_rows], blank=blank, percent_symbol=percent_symbol, aggregate=True),
        ]
        if include_latency:
            lat_nums = [r.get("latency_sec") for r in used_rows if r.get("latency_sec") is not None]
            if len(lat_nums) >= 2:
                vals.append(f"{mean(lat_nums):.2f} ± {stdev(lat_nums):.2f} sec")
            elif len(lat_nums) == 1:
                vals.append(fmt_sec(lat_nums[0]))
            else:
                vals.append(blank)
        return vals

    # No multi-seed aggregation: keep the first result for backward-compatible output.
    row = seed_rows[0]
    vals = [
        label,
        fmt_metric_aggregate([row.get("chairs")], blank=blank, percent_symbol=percent_symbol, aggregate=False),
        fmt_metric_aggregate([row.get("chairi")], blank=blank, percent_symbol=percent_symbol, aggregate=False),
        fmt_metric_aggregate([row.get("bleu")], blank=blank, percent_symbol=percent_symbol, aggregate=False),
    ]
    if include_latency:
        vals.append(fmt_sec(row.get("latency_sec")) if row.get("latency_sec") is not None else blank)
    return vals


def rows_to_markdown_table(
    rows: List[Dict[str, Any]],
    title: str,
    methods: List[str],
    blank: str = "",
    append_extra: bool = False,
    include_latency: bool = False,
    percent_symbol: bool = True,
    aggregate_seeds: bool = True,
    show_n: bool = False,
) -> str:
    headers = [title, "Chair S", "Chair I", "BLEU"]
    if include_latency:
        headers.append("Latency")

    method_order = methods or DEFAULT_MD_METHODS
    canonical = {compact_identifier(m): m for m in method_order}

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        inferred = infer_method(row)
        key = compact_identifier(inferred)
        method = canonical.get(key, inferred)
        by_method.setdefault(method, []).append(row)

    table_rows: List[List[str]] = []
    used = set()

    for method in method_order:
        used.add(method)
        table_rows.append(
            summarize_method_rows(
                method=method,
                rows=by_method.get(method, []),
                blank=blank,
                include_latency=include_latency,
                percent_symbol=percent_symbol,
                aggregate_seeds=aggregate_seeds,
                show_n=show_n,
            )
        )

    if append_extra:
        for method, method_rows in by_method.items():
            if method in used:
                # For known methods, append duplicate same-seed rows only.
                _seed_rows, duplicate_rows = unique_seed_rows(method_rows)
                for row in duplicate_rows:
                    seed = infer_seed(row)
                    label = str(row.get("label") or method)
                    if seed is not None:
                        label = f"{label} [duplicate seed={seed}]"
                    vals = [
                        label,
                        fmt_metric_aggregate([row.get("chairs")], blank=blank, percent_symbol=percent_symbol),
                        fmt_metric_aggregate([row.get("chairi")], blank=blank, percent_symbol=percent_symbol),
                        fmt_metric_aggregate([row.get("bleu")], blank=blank, percent_symbol=percent_symbol),
                    ]
                    if include_latency:
                        vals.append(fmt_sec(row.get("latency_sec")) if row.get("latency_sec") is not None else blank)
                    table_rows.append(vals)
                continue

            table_rows.append(
                summarize_method_rows(
                    method=method,
                    rows=method_rows,
                    blank=blank,
                    include_latency=include_latency,
                    percent_symbol=percent_symbol,
                    aggregate_seeds=aggregate_seeds,
                    show_n=show_n,
                )
            )

    escaped_headers = [markdown_escape_cell(h) for h in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for vals in table_rows:
        lines.append("| " + " | ".join(markdown_escape_cell(v) for v in vals) + " |")

    return "\n".join(lines)

def debug_method_matches(rows: List[Dict[str, Any]]) -> str:
    lines = ["METHOD/SEED MATCH DEBUG"]
    for row in rows:
        label = str(row.get("label") or "")
        seed = infer_seed(row)
        seed_text = "N/A" if seed is None else str(seed)
        lines.append(f"{infer_method(row)}\tseed={seed_text}\t<-\t{label}")
    return "\n".join(lines)


def write_text(rows_path: Path, text: str) -> None:
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def result_label(generated_path: Path, root: Path) -> str:
    try:
        rel_parent = generated_path.parent.relative_to(root)
        if str(rel_parent) != ".":
            return str(rel_parent)
    except Exception:
        pass
    return generated_path.parent.name


def print_result(label: str, chairs: Optional[float], chairi: Optional[float], bleu: Optional[float], latency: Optional[float]) -> None:
    width = min(max(len(label), 60), 120)
    print("\n" + "─" * width)
    print(label)
    print(f"chairs: {fmt_pct(chairs)} | chairi: {fmt_pct(chairi)} | bleu: {fmt_pct(bleu)} | latency: {fmt_sec(latency)}")


def ensure_chairs_for_folder(
    folder: Path,
    generated_files: List[Path],
    gt_caption_path: Path,
    main_codes: Path,
    force_convert: bool,
    verbose: bool,
) -> None:
    caption_to_chair = main_codes / "eval" / "caption_to_chair2.py"

    missing = [expected_chair_path(g) for g in generated_files if not expected_chair_path(g).exists()]
    if not force_convert and not missing:
        return

    aliases: List[Path] = []
    try:
        for g in generated_files:
            alias = json_to_jsonl_alias_if_needed(g)
            if alias is not None:
                aliases.append(alias)
                if verbose:
                    print(f"[ALIAS] {g.name} -> {alias.name}")

        cmd = [
            sys.executable,
            str(caption_to_chair),
            "--gt-caption-path",
            str(gt_caption_path),
            "-c",
            str(folder),
        ]
        rc, out = run_capture(cmd, cwd=main_codes, verbose=verbose)
        if rc != 0:
            print(f"\n[ERROR] caption_to_chair2.py failed for: {folder}", file=sys.stderr)
            print(out[-5000:], file=sys.stderr)
            raise RuntimeError(f"caption_to_chair2.py failed for {folder}")
    finally:
        for alias in aliases:
            try:
                alias.unlink()
                if verbose:
                    print(f"[CLEAN] removed temporary alias {alias.name}")
            except FileNotFoundError:
                pass


def resolve_chair_for_generated(generated_path: Path, include_draft: bool) -> Path:
    expected = expected_chair_path(generated_path)
    if expected.exists():
        return expected

    candidates = find_non_draft_chair_files(generated_path.parent, include_draft=include_draft)
    if candidates:
        return max(candidates, key=lambda p: common_prefix_score(generated_path.stem, p.stem)).resolve()

    raise FileNotFoundError(f"Chair file missing for {generated_path}")


def eval_one(
    generated_path: Path,
    root: Path,
    main_codes: Path,
    data_dir: Path,
    eval_output_dir: Path,
    include_draft: bool,
    verbose: bool,
    print_compact: bool = True,
) -> Dict[str, Any]:
    chair_path = resolve_chair_for_generated(generated_path, include_draft=include_draft)
    info = pick_best_info(chair_path, find_info_jsons(chair_path.parent))
    latency = latency_from_info(info, chair_path)

    eval_py = main_codes / "eval" / "eval_hallucination.py"
    cmd = [
        sys.executable,
        str(eval_py),
        "-v",
        "--metric",
        "chair",
        "--chair_input_path",
        str(chair_path),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(eval_output_dir),
    ]
    rc, out = run_capture(cmd, cwd=main_codes, verbose=verbose)
    if rc != 0:
        print(f"\n[ERROR] eval_hallucination.py failed for: {chair_path}", file=sys.stderr)
        print(out[-5000:], file=sys.stderr)
        raise RuntimeError(f"eval_hallucination.py failed for {chair_path}")

    metrics = parse_eval_metrics(out)
    chairs = pick_metric(metrics, "chairs", "CHAIRs")
    chairi = pick_metric(metrics, "chairi", "CHAIRi")
    bleu = pick_metric(metrics, "bleu", "Bleu_4", "BLEU")

    label = result_label(generated_path, root)
    if print_compact:
        print_result(label, chairs, chairi, bleu, latency)

    return {
        "label": label,
        "chairs": chairs,
        "chairi": chairi,
        "bleu": bleu,
        "latency_sec": latency,
        "generated_path": str(generated_path),
        "chair_path": str(chair_path),
        "info_path": str(info) if info else "",
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Save raw per-run results.

    This is intentionally raw, not aggregated. method/seed columns are added so
    you can also pivot/group the CSV manually if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "method",
        "seed",
        "label",
        "chairs",
        "chairi",
        "bleu",
        "latency_sec",
        "generated_path",
        "chair_path",
        "info_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["method"] = infer_method(row)
            out["seed"] = infer_seed(row)
            writer.writerow({k: out.get(k, "") for k in keys})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact RVCD CHAIR evaluator for generated_captions folder trees.")
    parser.add_argument("--folder", "-f", required=True, help="Top-level folder, e.g. generated_captions or generated_captions/chair")
    parser.add_argument("--gt-caption-path", required=True, help="Absolute path to captions_val2014.json")
    parser.add_argument("--data-dir", default=None, help="COCO val2014 dir. Default: parent of annotations dir")
    parser.add_argument("--eval-output-dir", default=None, help="Writable output dir for eval_hallucination. Default: temp dir")
    parser.add_argument("--main-codes", default=None, help="Path to RVCD/MAIN_CODES. Default: auto-detect")
    parser.add_argument("--include-draft", action="store_true", help="Include DRAFT generated files")
    parser.add_argument("--only-ervcd", action="store_true", help="Only evaluate eRVCD generated caption files")
    parser.add_argument("--force-convert", action="store_true", help="Regenerate chair files even if present")
    parser.add_argument("--verbose", action="store_true", help="Show underlying command output")
    parser.add_argument("--save-csv", default=None, help="Optional path to save compact CSV")
    parser.add_argument("--save-md", default=None, help="Optional path to save the final Markdown table")
    parser.add_argument("--md-title", default="llava 1.5 7b instruct, COCO", help="First Markdown table header cell")
    parser.add_argument(
        "--md-methods",
        default=None,
        help="Markdown row order. Use semicolons if labels contain commas, e.g. 'eRVCD, black_back;eRVCD, black_front'",
    )
    parser.add_argument("--md-empty", default="", help="Cell text for missing Markdown values. Default: empty cell")
    parser.add_argument("--md-append-extra", action="store_true", help="Append unmatched or duplicate runs below the fixed method rows")
    parser.add_argument("--no-md-latency", action="store_true", help="Do not add a Latency column to the Markdown table")
    parser.add_argument("--md-include-latency", action="store_true", help=argparse.SUPPRESS)  # backward-compatible no-op; latency is on by default
    parser.add_argument("--md-no-percent-symbol", action="store_true", help="Print metric numbers without percent signs in the Markdown table")
    parser.add_argument("--no-md-aggregate-seeds", action="store_true", help="Disable automatic mean ± std aggregation when a method has 2+ distinct seeds")
    parser.add_argument("--md-show-n", action="store_true", help="Append n=<seed count> to aggregated method labels")
    parser.add_argument("--no-print-md", action="store_true", help="Do not print the Markdown table to stdout")
    parser.add_argument("--no-final-summary", action="store_true", help="Do not print the FINAL SUMMARY compact block")
    parser.add_argument("--only-md", action="store_true", help="Only print the final Markdown table, not per-run compact output or FINAL SUMMARY")
    parser.add_argument("--md-debug-match", action="store_true", help="Print how each run label was mapped to a Markdown method row")
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    args = parser.parse_args()

    root = Path(args.folder).expanduser().resolve()
    gt = Path(args.gt_caption_path).expanduser().resolve()
    main_codes = resolve_main_codes(args.main_codes)

    if not root.exists():
        raise FileNotFoundError(f"--folder does not exist: {root}")
    if not gt.exists():
        raise FileNotFoundError(f"--gt-caption-path does not exist: {gt}")

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else gt.parent.parent

    if args.eval_output_dir:
        eval_output_dir = Path(args.eval_output_dir).expanduser().resolve()
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="rvcd_chair_eval_")
        eval_output_dir = Path(temp_ctx.name).resolve()

    generated_files = discover_generated_files(root, include_draft=args.include_draft)
    if args.only_ervcd:
        generated_files = [p for p in generated_files if is_ervcd_run_path(p)]
    if not generated_files:
        kind = "eRVCD " if args.only_ervcd else ""
        raise SystemExit(f"No non-DRAFT {kind}*generated_captions.json/jsonl files found under: {root}")

    if not args.only_md:
        only_text = " eRVCD" if args.only_ervcd else ""
        print(f"Found {len(generated_files)}{only_text} generated caption file(s).")
        print(f"Root: {root}")

    rows: List[Dict[str, Any]] = []
    for folder, group in group_by_parent(generated_files).items():
        ensure_chairs_for_folder(
            folder=folder,
            generated_files=group,
            gt_caption_path=gt,
            main_codes=main_codes,
            force_convert=args.force_convert,
            verbose=args.verbose,
        )
        for g in group:
            rows.append(
                eval_one(
                    generated_path=g,
                    root=root,
                    main_codes=main_codes,
                    data_dir=data_dir,
                    eval_output_dir=eval_output_dir,
                    include_draft=args.include_draft,
                    verbose=args.verbose,
                    print_compact=not args.only_md,
                )
            )

    if args.md_debug_match:
        if not args.only_md:
            print("\n" + debug_method_matches(rows))
        else:
            print(debug_method_matches(rows), file=sys.stderr)

    if (not args.no_final_summary) and (not args.only_md):
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        for row in rows:
            print_result(
                row["label"],
                row.get("chairs"),
                row.get("chairi"),
                row.get("bleu"),
                row.get("latency_sec"),
            )

    if args.md_methods:
        md_methods = parse_csv_items(args.md_methods)
    elif args.only_ervcd:
        md_methods = DEFAULT_ERVCD_MD_METHODS
    else:
        md_methods = DEFAULT_MD_METHODS

    md_table = rows_to_markdown_table(
        rows=rows,
        title=args.md_title,
        methods=md_methods,
        blank=args.md_empty,
        append_extra=args.md_append_extra,
        include_latency=not args.no_md_latency,
        percent_symbol=not args.md_no_percent_symbol,
        aggregate_seeds=not args.no_md_aggregate_seeds,
        show_n=args.md_show_n,
    )

    if not args.no_print_md:
        if not args.only_md:
            print("\n" + "=" * 80)
            print("MARKDOWN TABLE - copy below")
            print("=" * 80)
        print(md_table)

    if args.save_md:
        out = Path(args.save_md).expanduser().resolve()
        write_text(out, md_table)
        if not args.only_md:
            print(f"\nMarkdown saved: {out}")

    if args.save_csv:
        out = Path(args.save_csv).expanduser().resolve()
        write_csv(rows, out)
        if not args.only_md:
            print(f"\nCSV saved: {out}")

    if temp_ctx is not None:
        temp_ctx.cleanup()


if __name__ == "__main__":
    main()


# python eval_chair_folder_markdown_ervcd.py \
#   --gt-caption-path /home/jihoon/jihoon/DATASETS/coco2014/val2014/annotations/captions_val2014.json \
#   --folder /home/jihoon/jihoon/RVCD/MAIN_CODES/generated_captions \
#   --only-md \
#   --md-show-n

#   python eval_chair_folder_markdown_ervcd.py \
#   --gt-caption-path /home/jihoon/jihoon/DATASETS/coco2014/val2014/annotations/captions_val2014.json \
#   --folder /home/jihoon/jihoon/RVCD/MAIN_CODES/generated_captions_test_blackback \
#   --only-md \
#   --md-show-n

# python eval_chair_folder_markdown_ervcd.py \
#   --gt-caption-path /home/jihoon/jihoon/DATASETS/coco2014/val2014/annotations/captions_val2014.json \
#   --folder /home/jihoon/jihoon/RVCD/MAIN_CODES/generated_captions_test_blackfront \
#   --only-md \
#   --md-show-n

#   python eval_chair_folder_markdown_ervcd.py \
#   --gt-caption-path /home/jihoon/jihoon/DATASETS/coco2014/val2014/annotations/captions_val2014.json \
#   --folder /home/jihoon/jihoon/RVCD/MAIN_CODES/generated_captions_test_repeat \
#   --only-md \
#   --md-show-n

#   python eval_chair_folder_markdown_ervcd.py \
#   --gt-caption-path /home/jihoon/jihoon/DATASETS/coco2014/val2014/annotations/captions_val2014.json \
#   --folder /home/jihoon/jihoon/RVCD/MAIN_CODES/generated_captions_test_repeatfront \
#   --only-md \
#   --md-show-n

#   python eval_chair_folder_markdown_ervcd.py \
#   --gt-caption-path /home/jihoon/jihoon/DATASETS/coco2014/val2014/annotations/captions_val2014.json \
#   --folder /home/jihoon/jihoon/RVCD/MAIN_CODES/generated_captions_test_repeatlast \
#   --only-md \
#   --md-show-n


# python eval_chair_folder_markdown_ervcd.py \
#   --gt-caption-path /root/DATASETS/coco2014/val2014/annotations/captions_val2014.json \
#   --folder /root/RVCD/MAIN_CODES/generated_captions \
#   --only-md \
#   --md-show-n




# python eval_chair_folder_markdown_ervcd_grid_modes.py \
#   --folder /root/RVCD/MAIN_CODES/generated_captions_ervcd_grid_modes_OPTIMIZED_no_beta \
#   --gt-caption-path /home/jihoon/jihoon/DATASETS/coco2014/val2014/annotations/captions_val2014.json \
#   --only-ervcd \
#   --only-md \
#   --md-show-n \
#   --save-md ./generated_captions_ervcd_grid_modes/ervcd_grid_mode_results.md \
#   --save-csv ./generated_captions_ervcd_grid_modes/ervcd_grid_mode_results.csv