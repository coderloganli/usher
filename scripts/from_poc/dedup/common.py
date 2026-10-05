"""Shared paths, constants and text helpers for the dedup experiment."""
from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FEAS = ROOT.parent / "feasibility" / "data"  # read-only inputs from the feasibility step

REPO = "microsoft/vscode"
QUERY_START = "2026-07-01T00:00:00Z"
QUERY_END = "2026-10-01T00:00:00Z"  # exclusive
SEED = 20261004
MAX_BODY_CHARS = 2000

MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

# Accounts that open issues automatically (GraphQL drops the "[bot]" suffix).
BOT_AUTHORS = {"vs-code-engineering", "copilot-swe-agent", "github-actions", "Copilot", "dependabot"}


def is_bot_login(login: str | None) -> bool:
    return bool(login) and (login in BOT_AUTHORS or login.endswith("[bot]"))


_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def issue_text(title: str, body: str) -> str:
    """Title + body. HTML comments (issue-template boilerplate) are removed,
    then the body is truncated to MAX_BODY_CHARS characters."""
    body = _HTML_COMMENT.sub("", body or "").strip()
    return (title or "").strip() + "\n\n" + body[:MAX_BODY_CHARS]


def load_corpus() -> list[dict]:
    """All issues with text, sorted by (created_at, number)."""
    with gzip.open(DATA / "issues_text.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    rows.sort(key=lambda r: (r["created_at"], r["number"]))
    return rows


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
