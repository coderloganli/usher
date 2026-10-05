"""Fetch title + body for every microsoft/vscode issue created 2023-01-01..2026-09-30.

Uses the GraphQL `repository.issues` connection ordered by CREATED_AT DESC,
100 issues per page, through the authenticated `gh` CLI (no token in code).
Resumable: raw pages go to data/text_pages/page_NNNNN.json.gz and the cursor
is kept in data/text_pages/state.json. Backs off on errors and rate limits.

`--flatten` writes data/issues_text.jsonl.gz ({number, created_at, title, body})
restricted to the date range, plus data/pull_meta.json (pull times, counts).

Usage:
  python scripts/fetch_text.py            # fetch (resumable)
  python scripts/fetch_text.py --flatten
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "text_pages"
START = "2023-01-01T00:00:00Z"
END = "2026-10-01T00:00:00Z"  # exclusive

QUERY = """
query($cursor:String){ rateLimit{cost remaining resetAt}
 repository(owner:"microsoft",name:"vscode"){ issues(first:100, orderBy:{field:CREATED_AT,direction:DESC}, after:$cursor){
  pageInfo{hasNextPage endCursor}
  nodes{ number createdAt title body } }}}
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def gh_graphql(cursor: str | None) -> dict:
    """One GraphQL call with exponential backoff on any failure."""
    delay = 10
    for attempt in range(12):
        args = ["gh", "api", "graphql", "-f", f"query={QUERY}"]
        if cursor:
            args += ["-f", f"cursor={cursor}"]
        p = subprocess.run(args, capture_output=True, text=True)
        if p.returncode == 0:
            d = json.loads(p.stdout)
            if not d.get("errors"):
                return d
            msg = str(d["errors"])[:300]
        else:
            msg = p.stderr.strip()[:300]
        low = msg.lower()
        wait = 65 if ("secondary" in low or "abuse" in low) else delay
        print(f"[{now()}] try {attempt + 1} failed, sleeping {wait}s: {msg}", file=sys.stderr, flush=True)
        time.sleep(wait)
        delay = min(delay * 2, 300)
    raise RuntimeError("gh graphql failed repeatedly")


def fetch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state_f = OUT / "state.json"
    state = json.load(open(state_f)) if state_f.exists() else {
        "page": 0, "cursor": None, "done": False, "started_at": now(), "finished_at": None}
    while not state["done"]:
        d = gh_graphql(state["cursor"])
        with gzip.open(OUT / f"page_{state['page']:05d}.json.gz", "wt") as f:
            json.dump(d, f)
        conn = d["data"]["repository"]["issues"]
        oldest = conn["nodes"][-1]["createdAt"] if conn["nodes"] else START
        rl = d["data"]["rateLimit"]
        state["page"] += 1
        state["cursor"] = conn["pageInfo"]["endCursor"]
        state["done"] = (not conn["pageInfo"]["hasNextPage"]) or oldest < START
        if state["done"]:
            state["finished_at"] = now()
        write_json(state_f, state)
        if state["page"] % 25 == 0:
            print(f"[{now()}] page {state['page']} oldest={oldest} rl={rl}", flush=True)
        if rl["remaining"] < 100:
            reset = datetime.strptime(rl["resetAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            time.sleep(max(0, (reset - datetime.now(timezone.utc)).total_seconds()) + 5)
        time.sleep(1.0)  # stay well clear of secondary limits
    print("done", state)


def flatten() -> None:
    rows = {}
    for p in sorted(OUT.glob("page_*.json.gz")):
        d = json.load(gzip.open(p, "rt"))
        for n in d["data"]["repository"]["issues"]["nodes"]:
            if START <= n["createdAt"] < END:
                rows[n["number"]] = {"number": n["number"], "created_at": n["createdAt"],
                                     "title": n["title"] or "", "body": n["body"] or ""}
    with gzip.open(DATA / "issues_text.jsonl.gz", "wt") as f:
        for k in sorted(rows):
            f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    state = json.load(open(OUT / "state.json"))
    write_json(DATA / "pull_meta.json", {
        "repo": "microsoft/vscode", "range": [START, END], "issues": len(rows),
        "pages": state["page"], "pull_started_at": state["started_at"],
        "pull_finished_at": state["finished_at"]})
    print(len(rows), "issues written")


if __name__ == "__main__":
    flatten() if "--flatten" in sys.argv else fetch()
