"""A5 (part 1): bulk-pull every microsoft/vscode issue created 2023-01-01..2026-09-30.

One GraphQL pass over `repository.issues` ordered by CREATED_AT DESC, 100 per
page, through `gh api graphql`. Per issue: number, createdAt, title, body,
author (login + typename), final labels, final assignees, and `duplicateOf`
(number, createdAt, repository). `duplicateOf` is used because the
MARKED_AS_DUPLICATE event only appears on the canonical issue's timeline.

Resumable: raw pages go to data/raw/issues_pages/page_NNNNN.json.gz, the cursor
and pull times to data/raw/issues_pages/state.json. Backs off on errors and on
primary / secondary rate limits.

`--flatten` writes data/issues.jsonl.gz (one row per issue in the range, sorted
by number) and data/pull_meta.json.

Usage:
  .venv/bin/python scripts/pipeline/pull_issues.py            # fetch (resumable)
  .venv/bin/python scripts/pipeline/pull_issues.py --flatten
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

from common import (DATA, ISSUES, PAGES, PULL_END, PULL_START, gh, log, now,
                    read_gz, write_gz, write_json)

PAGE_SIZE = 100
LABELS_MAX = 50
ASSIGNEES_MAX = 20

QUERY = """
query($cursor:String){ rateLimit{cost remaining resetAt}
 repository(owner:"microsoft",name:"vscode"){ issues(first:%d, orderBy:{field:CREATED_AT,direction:DESC}, after:$cursor){
  totalCount pageInfo{hasNextPage endCursor}
  nodes{ number createdAt title body author{__typename login}
   labels(first:%d){totalCount nodes{name}}
   assignees(first:%d){totalCount nodes{login}}
   duplicateOf{ number createdAt repository{nameWithOwner} } } }}}
""" % (PAGE_SIZE, LABELS_MAX, ASSIGNEES_MAX)


def fetch_page(cursor: str | None) -> tuple[str, dict]:
    for attempt in range(20):
        args = ["api", "graphql", "-f", f"query={QUERY}"]
        if cursor:
            args += ["-f", f"cursor={cursor}"]
        try:
            raw = gh(args)
        except RuntimeError as e:
            log(f"page fetch failed ({e}); sleeping 300s")
            time.sleep(300)
            continue
        d = json.loads(raw)
        if not d.get("errors"):
            return raw, d
        log(f"graphql errors (try {attempt + 1}): {str(d['errors'])[:300]}")
        time.sleep(65 if "rate" in str(d["errors"]).lower() else 20)
    raise RuntimeError("graphql page failed repeatedly")


def fetch() -> None:
    PAGES.mkdir(parents=True, exist_ok=True)
    sf = PAGES / "state.json"
    st = json.load(open(sf)) if sf.exists() else {
        "page": 0, "cursor": None, "done": False, "started_at": now(), "finished_at": None,
        "query": QUERY}
    while not st["done"]:
        raw, d = fetch_page(st["cursor"])
        write_gz(PAGES / f"page_{st['page']:05d}.json.gz", raw)
        conn = d["data"]["repository"]["issues"]
        oldest = conn["nodes"][-1]["createdAt"] if conn["nodes"] else PULL_START
        rl = d["data"]["rateLimit"]
        st["page"] += 1
        st["cursor"] = conn["pageInfo"]["endCursor"]
        st["done"] = (not conn["pageInfo"]["hasNextPage"]) or oldest < PULL_START
        if st["done"]:
            st["finished_at"] = now()
        write_json(sf, st)
        if st["page"] % 25 == 0:
            log(f"page {st['page']} oldest={oldest} rate={rl}")
        if rl["remaining"] < 100:
            reset = datetime.strptime(rl["resetAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            wait = max(0, (reset - datetime.now(timezone.utc)).total_seconds()) + 10
            log(f"graphql budget low, sleeping {wait:.0f}s")
            time.sleep(wait)
        time.sleep(1.0)  # stay clear of secondary limits
    log(f"fetch done: {st['page']} pages, {st['started_at']} .. {st['finished_at']}")


def flatten() -> None:
    rows: dict[int, dict] = {}
    trunc = {"labels": 0, "assignees": 0}
    pages = sorted(PAGES.glob("page_*.json.gz"))
    for p in pages:
        d = json.loads(read_gz(p))
        for n in d["data"]["repository"]["issues"]["nodes"]:
            if not (PULL_START <= n["createdAt"] < PULL_END):
                continue
            if n["labels"]["totalCount"] > len(n["labels"]["nodes"]):
                trunc["labels"] += 1
            if n["assignees"]["totalCount"] > len(n["assignees"]["nodes"]):
                trunc["assignees"] += 1
            dup = n["duplicateOf"]
            rows[n["number"]] = {
                "number": n["number"],
                "created_at": n["createdAt"],
                "title": n["title"] or "",
                "body": n["body"] or "",
                "author": (n["author"] or {}).get("login"),
                "author_type": (n["author"] or {}).get("__typename"),
                "labels": sorted(x["name"] for x in n["labels"]["nodes"]),
                "assignees": sorted(x["login"] for x in n["assignees"]["nodes"]),
                "duplicate_of": None if not dup else {
                    "number": dup["number"], "created_at": dup["createdAt"],
                    "repo": dup["repository"]["nameWithOwner"]},
            }
    text = "".join(json.dumps(rows[k], ensure_ascii=False, sort_keys=True) + "\n" for k in sorted(rows))
    write_gz(ISSUES, text)
    st = json.load(open(PAGES / "state.json"))
    meta = {"repo": "microsoft/vscode", "range": [PULL_START, PULL_END], "issues": len(rows),
            "pages": len(pages), "pull_started_at": st["started_at"], "pull_finished_at": st["finished_at"],
            "pull_done": st["done"], "truncated_connections": trunc,
            "limits": {"page_size": PAGE_SIZE, "labels": LABELS_MAX, "assignees": ASSIGNEES_MAX}}
    write_json(DATA / "pull_meta.json", meta)
    log(f"flatten: {len(rows)} issues from {len(pages)} pages; truncated={trunc}")


if __name__ == "__main__":
    flatten() if "--flatten" in sys.argv else fetch()
