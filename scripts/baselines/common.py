"""Shared paths, constants and data loaders for the Part B baselines.

Plan: usher/stage1_partB_plan.html (approved 2026-10-07).
Inputs come from the Part A pipeline: data/issues.jsonl.gz (all pulled issues) and
data/sets.json (per-judgment train / tune / test sets with ground truth).
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ISSUES = DATA / "issues.jsonl.gz"
SETS = DATA / "sets.json"
CACHE = DATA / "baselines"            # gitignored: embedding vectors and other caches
OUT = ROOT / "outputs" / "baselines"  # tracked: predictions, tuning records, metrics (no issue text)

JUDGMENTS = ("type", "module", "duplicate")
SPLITS = ("train", "tune", "test")

SEED = 20261007
N_BOOT = 10_000


def load_issues() -> dict[int, dict]:
    """Every pulled issue (2023-01..2026-09), keyed by number."""
    with gzip.open(ISSUES, "rt", encoding="utf-8") as f:
        return {r["number"]: r for r in map(json.loads, f)}


def load_sets(judgment: str) -> dict[str, list[dict]]:
    """{split: [{"number", "created_at", "truth", ...}]} for one judgment.

    truth is a type label (type), a sorted list of module labels (module) or the
    canonical issue number (duplicate).
    """
    with open(SETS, encoding="utf-8") as f:
        sets = json.load(f)[judgment]
    return {s: sets[s] for s in SPLITS}


def issue_text(r: dict) -> str:
    """Title + raw body, the input of every method. HTML comments are kept on purpose:
    some record the issue template the reporter chose, which is known at creation time."""
    return (r["title"] or "") + "\n\n" + (r["body"] or "")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, path)
