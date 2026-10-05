"""Shared paths, constants and helpers for the type / assignee classification experiment."""
from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FEAS_DATA = ROOT.parent / "feasibility" / "data"        # read-only: metadata (labels, assignees)
DEDUP_DATA = ROOT.parent / "dedup_experiment" / "data"  # read-only: issue text + cached embeddings

REPO = "microsoft/vscode"
SEED = 20261004
N_TEST = 300
N_BOOT = 1000
MAX_BODY_CHARS = 2000

# Splits by creation month (UTC), from poc.html section 6.
TRAIN_MONTHS = {"2026-01", "2026-02", "2026-03", "2026-04", "2026-05"}
TUNE_MONTHS = {"2026-06"}
TEST_MONTHS = {"2026-09"}

# The 10 type labels from the vscode wiki (same list as feasibility/scripts/common.py).
TYPE_LABELS = [
    "bug", "feature-request", "under-discussion", "debt", "*question",
    "upstream", "upstream-issue-linked", "electron", "engineering", "polish",
]

MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
TRIAGE_BOT = "vs-code-engineering[bot]"

# Same bot-author rule as dedup_experiment (GraphQL drops the "[bot]" suffix).
BOT_AUTHORS = {"vs-code-engineering", "copilot-swe-agent", "github-actions", "Copilot", "dependabot"}


def is_bot_login(login: str | None) -> bool:
    return bool(login) and (login in BOT_AUTHORS or login.endswith("[bot]"))


_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def issue_text(title: str, body: str) -> str:
    """Title + body; HTML comments removed, body truncated to MAX_BODY_CHARS
    (identical to dedup_experiment/scripts/common.py, so its embeddings can be reused)."""
    body = _HTML_COMMENT.sub("", body or "").strip()
    return (title or "").strip() + "\n\n" + body[:MAX_BODY_CHARS]


def load_meta() -> dict[int, dict]:
    with gzip.open(FEAS_DATA / "d1_issues.jsonl.gz", "rt") as f:
        return {r["number"]: r for r in map(json.loads, f)}


def load_text_corpus() -> list[dict]:
    """Issue text rows sorted by (created_at, number): the row order of dedup's embeddings.npy."""
    with gzip.open(DEDUP_DATA / "issues_text.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    rows.sort(key=lambda r: (r["created_at"], r["number"]))
    return rows


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
