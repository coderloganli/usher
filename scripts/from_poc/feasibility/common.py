"""Shared helpers for the Usher Step 1 feasibility scripts.

All GitHub access goes through the authenticated `gh` CLI so that no token is
handled in this code. Every helper sleeps / backs off on rate limits.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPO = "microsoft/vscode"
BOT_LOGINS = {"vs-code-engineering", "vs-code-engineering[bot]"}

# Type labels as defined in the vscode wiki page "Issues-Triaging",
# section "Categorizing Issues" (first table, the labels that play a role in
# triage). `needs more info` (now `info-needed`) is listed there too but is
# handled as its own output, not as a type.
TYPE_LABELS = [
    "bug", "feature-request", "under-discussion", "debt", "*question",
    "upstream", "upstream-issue-linked", "electron", "engineering", "polish",
]
# The two type labels that carry most of the triage decision.
TYPE_CORE = ["bug", "feature-request"]

# Labels that close an issue as a duplicate (commands.json) or mark it.
DUP_LABEL = "*duplicate"
INFO_LABEL = "info-needed"
# Command labels / comment commands that make the bot add `info-needed`
# (from .github/commands.json, fetched 2026-10-03, saved in data/commands.json).
INFO_TRIGGER_LABELS = {"~info-needed", "~version-info-needed", "~confirmation-needed"}
INFO_TRIGGER_COMMENTS = {"needsmoreinfo", "needsperfinfo", "jsdebuglogs", "gifplease", "confirmplease"}
DUP_TRIGGER_COMMENTS = {"duplicate"}


def area_labels() -> set[str]:
    """Area labels = keys of .github/classifier.json that exist as repo labels."""
    cls = json.load(open(DATA / "classifier.json"))
    repo_labels = {x["name"] for x in json.load(open(DATA / "labels.json"))}
    return set(cls["labels"]) & repo_labels


# Broad area set (a heuristic, not an official list): classifier.json areas
# plus every repo label with the light-blue feature-area colours c5def5 /
# bfd4f2 (295 of the 298 classifier area labels are c5def5). Added because
# classifier.json lacks most 2025-2026 chat/agent labels. Two automation
# labels with the same colour are excluded.
AREA_COLOURS = {"c5def5", "bfd4f2"}
AREA_COLOUR_EXCLUDE = {"cherry-pick-artifact", "copilot-sdk-historical-replay"}


def area_labels_broad() -> set[str]:
    labels = json.load(open(DATA / "labels.json"))
    colour = {x["name"] for x in labels if x["color"].lower() in AREA_COLOURS} - AREA_COLOUR_EXCLUDE
    return area_labels() | (colour - set(TYPE_LABELS))


def is_bot(login: str | None, typename: str | None = None) -> bool:
    if login is None:
        return False
    if typename == "Bot":
        return True
    return login in BOT_LOGINS or login.endswith("[bot]")


def gh(args: list[str], max_tries: int = 8) -> str:
    """Run `gh` with retries and exponential backoff. Returns stdout."""
    delay = 5
    for attempt in range(max_tries):
        p = subprocess.run(["gh", *args], capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout
        err = p.stderr.lower()
        if "rate limit" in err or "secondary" in err or "abuse" in err:
            wait = 65 if "secondary" in err or "abuse" in err else delay * 4
            print(f"[gh] rate limited, sleeping {wait}s: {p.stderr.strip()[:200]}", file=sys.stderr)
            time.sleep(wait)
        else:
            print(f"[gh] error (try {attempt+1}): {p.stderr.strip()[:300]}", file=sys.stderr)
            time.sleep(delay)
        delay = min(delay * 2, 120)
    raise RuntimeError(f"gh failed after {max_tries} tries: {args[:4]}")


def core_remaining() -> tuple[int, int]:
    d = json.loads(gh(["api", "rate_limit"]))["resources"]["core"]
    return d["remaining"], d["reset"]


def wait_for_core(min_left: int = 200) -> None:
    left, reset = core_remaining()
    if left < min_left:
        wait = max(0, reset - int(time.time())) + 5
        print(f"[core] {left} left, sleeping {wait}s until reset", file=sys.stderr)
        time.sleep(wait)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


COMMAND_RE = re.compile(r"^\s*[/\\](\w+)", re.M)


def comment_commands(body: str | None) -> set[str]:
    return {m.lower() for m in COMMAND_RE.findall(body or "")}


_TARGET = r"(?:#(\d+)|https://github\.com/([\w.-]+/[\w.-]+)/issues/(\d+))"
# "Duplicate of #N" / "duplicate of <issue url>" anywhere in a comment.
DUP_OF_RE = re.compile(r"duplicate of\s+" + _TARGET, re.I)
# Bot comment command "/duplicate", optionally followed by "of" and a target.
DUP_CMD_RE = re.compile(r"^\s*[/\\]duplicate\b(?:\s+of)?\s*" + _TARGET + "?", re.I | re.M)


def dup_targets(body: str | None) -> list[tuple[str, int, str]]:
    """Return [(repo, number, form)] for duplicate targets named in a comment."""
    out = []
    for rx, form in ((DUP_CMD_RE, "command"), (DUP_OF_RE, "duplicate_of")):
        for m in rx.finditer(body or ""):
            if m.group(1):
                out.append(("microsoft/vscode", int(m.group(1)), form))
            elif m.group(3):
                out.append((m.group(2).lower(), int(m.group(3)), form))
    return out
