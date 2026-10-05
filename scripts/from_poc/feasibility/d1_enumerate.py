"""D1 (part A): enumerate every issue of microsoft/vscode created 2023-01-01..2026-09-30.

Uses the GraphQL `repository.issues` connection (issues only, never PRs),
ordered by CREATED_AT descending, 100 per page, until createdAt < 2023-01-01.
For each issue it records final labels, final assignees, close reason, the
MARKED/UNMARKED_AS_DUPLICATE events and the LABELED/UNLABELED events.

Raw pages are kept gzipped in data/d1_graphql/page_NNNNN.json.gz; the run is
resumable (cursor stored in data/d1_graphql/state.json). A flattened table is
written by `--flatten` to data/d1_issues.jsonl.gz.

Usage:
  python3 scripts/d1_enumerate.py          # fetch (resumable)
  python3 scripts/d1_enumerate.py --flatten
"""
from __future__ import annotations

import gzip
import json
import sys
import time

from common import DATA, gh, write_json

OUT = DATA / "d1_graphql"
START = "2023-01-01T00:00:00Z"
END = "2026-10-01T00:00:00Z"  # exclusive
PAGE = 100

QUERY = """
query($cursor:String){ rateLimit{cost remaining resetAt}
 repository(owner:"microsoft",name:"vscode"){ issues(first:%d, orderBy:{field:CREATED_AT,direction:DESC}, after:$cursor){ totalCount pageInfo{hasNextPage endCursor}
  nodes{ number createdAt closedAt state stateReason author{__typename login}
   labels(first:50){totalCount nodes{name}}
   assignees(first:10){totalCount nodes{login}}
   dup: timelineItems(first:10,itemTypes:[MARKED_AS_DUPLICATE_EVENT,UNMARKED_AS_DUPLICATE_EVENT]){totalCount nodes{__typename
     ... on MarkedAsDuplicateEvent{createdAt actor{__typename login} canonical{... on Issue{number} ... on PullRequest{number}} duplicate{... on Issue{number} ... on PullRequest{number}}}
     ... on UnmarkedAsDuplicateEvent{createdAt actor{__typename login} canonical{... on Issue{number} ... on PullRequest{number}} duplicate{... on Issue{number} ... on PullRequest{number}}}}}
   lab: timelineItems(first:60,itemTypes:[LABELED_EVENT,UNLABELED_EVENT]){totalCount nodes{__typename
     ... on LabeledEvent{createdAt actor{__typename login} label{name}}
     ... on UnlabeledEvent{createdAt actor{__typename login} label{name}}}}
 }}}}
""" % PAGE


def fetch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state_f = OUT / "state.json"
    state = json.load(open(state_f)) if state_f.exists() else {"page": 0, "cursor": None, "done": False}
    while not state["done"]:
        args = ["api", "graphql", "-f", f"query={QUERY}"]
        if state["cursor"]:
            args += ["-f", f"cursor={state['cursor']}"]
        raw = gh(args)
        d = json.loads(raw)
        if d.get("errors"):
            print("errors:", d["errors"], file=sys.stderr)
            time.sleep(30)
            continue
        with gzip.open(OUT / f"page_{state['page']:05d}.json.gz", "wt") as f:
            f.write(raw)
        conn = d["data"]["repository"]["issues"]
        oldest = conn["nodes"][-1]["createdAt"] if conn["nodes"] else START
        rl = d["data"]["rateLimit"]
        state["page"] += 1
        state["cursor"] = conn["pageInfo"]["endCursor"]
        state["done"] = (not conn["pageInfo"]["hasNextPage"]) or oldest < START
        write_json(state_f, state)
        if state["page"] % 20 == 0:
            print(f"page {state['page']} oldest={oldest} rl={rl}", flush=True)
        if rl["remaining"] < 100:
            time.sleep(600)
        time.sleep(0.5)
    print("done", state)


def flatten() -> None:
    rows = {}
    trunc = {"labels": 0, "lab_events": 0, "dup_events": 0, "assignees": 0}
    for p in sorted(OUT.glob("page_*.json.gz")):
        d = json.load(gzip.open(p, "rt"))
        for n in d["data"]["repository"]["issues"]["nodes"]:
            if not (START <= n["createdAt"] < END):
                continue
            for key, conn in (("labels", n["labels"]), ("lab_events", n["lab"]), ("dup_events", n["dup"]), ("assignees", n["assignees"])):
                if conn["totalCount"] > len(conn["nodes"]):
                    trunc[key] += 1
            rows[n["number"]] = {
                "number": n["number"],
                "created_at": n["createdAt"],
                "closed_at": n["closedAt"],
                "state": n["state"],
                "state_reason": n["stateReason"],
                "author": (n["author"] or {}).get("login"),
                "labels": [x["name"] for x in n["labels"]["nodes"]],
                "assignees": [x["login"] for x in n["assignees"]["nodes"]],
                "dup_events": [
                    {"type": e["__typename"], "at": e["createdAt"],
                     "actor": (e["actor"] or {}).get("login"), "actor_type": (e["actor"] or {}).get("__typename"),
                     "canonical": (e.get("canonical") or {}).get("number"),
                     "duplicate": (e.get("duplicate") or {}).get("number")}
                    for e in n["dup"]["nodes"]],
                "dup_events_total": n["dup"]["totalCount"],
                "label_events": [
                    {"type": e["__typename"], "at": e["createdAt"],
                     "actor": (e["actor"] or {}).get("login"), "actor_type": (e["actor"] or {}).get("__typename"),
                     "label": (e.get("label") or {}).get("name")}
                    for e in n["lab"]["nodes"]],
                "label_events_total": n["lab"]["totalCount"],
            }
    with gzip.open(DATA / "d1_issues.jsonl.gz", "wt") as f:
        for k in sorted(rows):
            f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    write_json(DATA / "d1_flatten_meta.json", {"issues": len(rows), "truncated_connections": trunc})
    print(len(rows), "issues;", "truncated:", trunc)


if __name__ == "__main__":
    flatten() if "--flatten" in sys.argv else fetch()
