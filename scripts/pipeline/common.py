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
# 2026-10-07 (Part B plan): tune widened from 2026-06 to 2026-04..06, the same length as test;
# train ends 2026-03-31 instead of 2026-05-31. Test is unchanged.
SPLITS = {
    "train": ("2025-01-01T00:00:00Z", "2026-04-01T00:00:00Z"),
    "tune": ("2026-04-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    "test": ("2026-07-01T00:00:00Z", "2026-10-01T00:00:00Z"),
}
# "module" replaced "assignee" on 2026-10-07 (spec revision). The assignee judgment is still
# built so its outputs stay on record, but it is discontinued and not evaluated further.
JUDGMENTS = ["type", "module", "duplicate", "assignee"]
ACTIVE_JUDGMENTS = ["type", "module", "duplicate"]
DISCONTINUED = {"assignee": {"date": "2026-10-07",
                             "reason": "who should own an issue depends on internal team allocation and "
                                       "cannot be checked from public content, and many final assignees "
                                       "in the test set never acted on the issue (see "
                                       "manifest/assignee_check.json). Replaced by module."}}


# Raw / intermediate data (gitignored).
PAGES = DATA / "raw" / "issues_pages"
TIMELINES = DATA / "raw" / "timelines"
ISSUES = DATA / "issues.jsonl.gz"
MODULE_LABELS_FILE = MANIFEST / "module_labels.json"
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


def gh(args: list[str], max_tries: int = 12, max_rate_wait: int = 5400) -> str:
    """Run `gh` with exponential backoff. Rate-limit errors (primary or secondary) do not
    count as tries; they are waited out for up to `max_rate_wait` seconds in total."""
    delay, tries, rate_waited = 10, 0, 0
    while True:
        p = subprocess.run(["gh", *args], capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout
        msg = (p.stderr or p.stdout).strip()[:300]
        if "HTTP 404" in msg or "HTTP 410" in msg:
            raise FileNotFoundError(msg)
        if is_rate_error(msg) and rate_waited < max_rate_wait:
            log(f"gh rate limited, sleeping 120s: {msg}")
            time.sleep(120)
            rate_waited += 120
            continue
        tries += 1
        if tries >= max_tries:
            raise RuntimeError(f"gh failed after {max_tries} tries: {args[:3]}: {msg}")
        log(f"gh try {tries} failed, sleeping {delay}s: {msg}")
        time.sleep(delay)
        delay = min(delay * 2, 300)


def core_rate() -> tuple[int, int]:
    """(remaining, reset epoch) of the REST core budget, read from the headers of a real
    1-request call. The /rate_limit endpoint returned stale values (always 5000 left) for
    this token, so it is not used."""
    out = gh(["api", "-i", f"repos/{REPO}/issues/1/labels"])
    hdr = {}
    for line in out.splitlines():
        if not line.strip():
            break
        if ":" in line:
            k, v = line.split(":", 1)
            hdr[k.strip().lower()] = v.strip()
    return int(hdr["x-ratelimit-remaining"]), int(hdr["x-ratelimit-reset"])
