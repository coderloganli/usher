"""Shared helpers for stage 2: duplicate links with their creation time, and recall@k
under the sibling rule.

Sibling rule (decided 2026-10-08): a query counts as found at k if, among its first k
candidates, there is the true original, or a "sibling": an issue whose duplicate link to the
same original was created before the query issue was created.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
from common import DATA, ROOT as _ROOT  # noqa: E402,F401

OUT2 = ROOT / "outputs" / "stage2"
CACHE2 = DATA / "stage2"


def duplicate_links(issues: dict[int, dict]) -> dict[int, tuple[int, str]]:
    """{duplicate issue: (original, time the link was created)} for every snapshot link whose
    creation time is known: the earliest 'marked as duplicate' event or 'closed as duplicate'
    event pointing to the snapshot's original (data/duplicate_events.json)."""
    with open(DATA / "duplicate_events.json", encoding="utf-8") as f:
        events = json.load(f)
    links = {}
    for n, r in issues.items():
        original = (r["duplicate_of"] or {}).get("number")
        if not original:
            continue
        times = [e["created_at"] for e in events.get(str(n), [])
                 if e["canonical"] == original
                 and (e["event"] == "marked" or (e["event"] == "closed" and e["state_reason"] == "DUPLICATE"))]
        if times:
            links[n] = (original, min(times))
    return links


def found_at(k: int, query: int, original: int, original_rank: int, top: list[int],
             links: dict[int, tuple[int, str]], query_created: str) -> bool:
    """True if the original is ranked within k, or a sibling linked before the query is in top[:k]."""
    if original_rank <= k:
        return True
    for c in top[:k]:
        link = links.get(c)
        if c != query and link and link[0] == original and link[1] < query_created:
            return True
    return False
