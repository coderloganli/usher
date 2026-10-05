"""Shared constants, paths and helpers for the stage-1 data pipeline (spec A5-A9).

`is_bot_login`, `issue_text` and TYPE_LABELS are imported unchanged from the
PoC code in scripts/from_poc so that the rules stay identical to the PoC.
All GitHub access goes through the authenticated `gh` CLI (no token in code).
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MANIFEST = ROOT / "manifest"
REVIEW = ROOT / "review"
FROM_POC = ROOT / "scripts" / "from_poc"

REPO = "microsoft/vscode"
TRIAGE_BOT = "vs-code-engineering[bot]"
SEED = 20261004

# Pull range (A5): issues created 2023-01-01 .. 2026-09-30 (UTC), END exclusive.
PULL_START = "2023-01-01T00:00:00Z"
PULL_END = "2026-10-01T00:00:00Z"

# Splits by createdAt (A8), [start, end) in UTC ISO strings.
SPLITS = {
    "train": ("2025-01-01T00:00:00Z", "2026-06-01T00:00:00Z"),
    "tune": ("2026-06-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    "test": ("2026-07-01T00:00:00Z", "2026-10-01T00:00:00Z"),
}
JUDGMENTS = ["type", "assignee", "duplicate"]

# Raw / intermediate data (gitignored).
PAGES = DATA / "raw" / "issues_pages"
TIMELINES = DATA / "raw" / "timelines"
ISSUES = DATA / "issues.jsonl.gz"
SETS = DATA / "sets.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_poc_dedup = _load("poc_dedup_common", FROM_POC / "dedup" / "common.py")
_poc_feas = _load("poc_feas_common", FROM_POC / "feasibility" / "common.py")
is_bot_login = _poc_dedup.is_bot_login
issue_text = _poc_dedup.issue_text
TYPE_LABELS = list(_poc_feas.TYPE_LABELS)


def normalized_text(title: str, body: str) -> str:
    """`issue_text` with whitespace collapsed and lower-cased; used for exact-duplicate flags."""
    return " ".join(issue_text(title, body).split()).lower()


def split_of(created_at: str) -> str | None:
    for name, (a, b) in SPLITS.items():
        if a <= created_at < b:
            return name
    return None


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", file=sys.stderr, flush=True)


def write_json(path: Path, obj, indent: int | None = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def write_gz(path: Path, text: str) -> None:
    """Deterministic gzip (mtime=0, no file name) so identical content gives identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as g:
            g.write(text.encode("utf-8"))
    os.replace(tmp, path)


def read_gz(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.read()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_issues() -> dict[int, dict]:
    with gzip.open(ISSUES, "rt", encoding="utf-8") as f:
        return {r["number"]: r for r in map(json.loads, f)}


def is_rate_error(msg: str) -> bool:
    low = msg.lower()
    return any(k in low for k in ("rate limit", "secondary", "abuse", "retry-after"))


def gh(args: list[str], max_tries: int = 12) -> str:
    """Run `gh` with exponential backoff; long waits on primary/secondary rate limits."""
    delay = 10
    for attempt in range(max_tries):
        p = subprocess.run(["gh", *args], capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout
        msg = (p.stderr or p.stdout).strip()[:300]
        if "404" in msg or "Not Found" in msg or "410" in msg:
            raise FileNotFoundError(msg)
        wait = max(65, delay) if is_rate_error(msg) else delay
        log(f"gh try {attempt + 1} failed, sleeping {wait}s: {msg}")
        time.sleep(wait)
        delay = min(delay * 2, 300)
    raise RuntimeError(f"gh failed after {max_tries} tries: {args[:3]}")


def core_rate() -> tuple[int, int]:
    d = json.loads(gh(["api", "rate_limit"]))["resources"]["core"]
    return d["remaining"], d["reset"]
