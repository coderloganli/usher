"""Pull the timeline events that create or remove a duplicate link, for every snapshot issue
that has a duplicateOf link, so that each link has the time it was created.

A link is created either by a 'marked as duplicate' event or by closing the issue as a
duplicate. GitHub GraphQL (schema introspection, 2026-10-08): MarkedAsDuplicateEvent and
UnmarkedAsDuplicateEvent carry createdAt and canonical (the original); ClosedEvent carries
createdAt, stateReason (DUPLICATE when closed as a duplicate) and duplicateOf; ReopenedEvent
carries createdAt. Each request asks for up to BATCH issues through aliases; every raw response
is cached under data/raw/duplicate_events/, so an interrupted run resumes.

Output: data/duplicate_events.json   {issue number: [{"event", "created_at", "actor", "actor_type", "state_reason",
                                                      "canonical", "canonical_repo"}]}
        data/duplicate_events_meta.json   pull time and counts
Usage:  .venv/bin/python scripts/stage2/pull_duplicate_events.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
from common import DATA, gh, load_issues, log, now, write_json  # noqa: E402

RAW = DATA / "raw" / "duplicate_events"
BATCH = 50
EVENTS = ("timelineItems(first: 50, itemTypes: [MARKED_AS_DUPLICATE_EVENT, UNMARKED_AS_DUPLICATE_EVENT, "
          "CLOSED_EVENT, REOPENED_EVENT])")
ACTOR = "actor { __typename login }"
TARGET = "... on Issue { number repository { nameWithOwner } } ... on PullRequest { number repository { nameWithOwner } }"
FIELDS = (f"number {EVENTS} {{ totalCount nodes {{ __typename "
          f"... on MarkedAsDuplicateEvent {{ createdAt {ACTOR} canonical {{ {TARGET} }} }} "
          f"... on UnmarkedAsDuplicateEvent {{ createdAt {ACTOR} canonical {{ {TARGET} }} }} "
          f"... on ClosedEvent {{ createdAt {ACTOR} stateReason duplicateOf {{ {TARGET} }} }} "
          f"... on ReopenedEvent {{ createdAt {ACTOR} }} }} }}")
KIND = {"MarkedAsDuplicateEvent": "marked", "UnmarkedAsDuplicateEvent": "unmarked",
        "ClosedEvent": "closed", "ReopenedEvent": "reopened"}


def query(numbers: list[int]) -> str:
    body = " ".join(f"i{n}: issue(number: {n}) {{ {FIELDS} }}" for n in numbers)
    return f'{{ repository(owner: "microsoft", name: "vscode") {{ {body} }} }}'


def main() -> None:
    issues = load_issues()
    numbers = sorted(n for n, r in issues.items() if r["duplicate_of"])
    RAW.mkdir(parents=True, exist_ok=True)
    started = now()
    for b in range(0, len(numbers), BATCH):
        path = RAW / f"batch_{b // BATCH:04d}.json.gz"
        if path.exists():
            continue
        out = gh(["api", "graphql", "-f", f"query={query(numbers[b:b + BATCH])}"])
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(out)
        log(f"{min(b + BATCH, len(numbers))}/{len(numbers)} issues")

    events, truncated, missing = {}, [], []
    for path in sorted(RAW.glob("batch_*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            repo = json.load(f)["data"]["repository"]
        for node in repo.values():
            if node is None:
                continue
            t = node["timelineItems"]
            if t["totalCount"] > len(t["nodes"]):
                truncated.append(node["number"])
            rows = []
            for e in t["nodes"]:
                target = e.get("canonical") or e.get("duplicateOf") or {}
                actor = e.get("actor") or {}
                rows.append({"event": KIND[e["__typename"]], "created_at": e["createdAt"],
                             "actor": actor.get("login"), "actor_type": actor.get("__typename"),
                             "state_reason": e.get("stateReason"), "canonical": target.get("number"),
                             "canonical_repo": (target.get("repository") or {}).get("nameWithOwner")})
            events[node["number"]] = rows
    missing = [n for n in numbers if n not in events]
    write_json(DATA / "duplicate_events.json", {str(n): events[n] for n in sorted(events)})
    write_json(DATA / "duplicate_events_meta.json", {
        "pulled_started_at": started, "pulled_finished_at": now(), "issues_requested": len(numbers),
        "issues_returned": len(events), "missing": missing, "truncated": truncated,
        "issues_without_any_event": sum(1 for v in events.values() if not v)})
    log(f"done: {len(events)} issues, {sum(len(v) for v in events.values())} events, "
        f"{len(missing)} missing, {len(truncated)} truncated")


if __name__ == "__main__":
    main()
