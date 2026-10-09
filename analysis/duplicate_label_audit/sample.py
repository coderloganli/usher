"""Sample high-ranked duplicate candidates that the ground truth calls wrong, for an audit of
duplicates the maintainers may have missed.

Source: the 1,205 development queries scored with embedding retrieval on cleaned text
(outputs/stage2/duplicate_clean_dev.json). A candidate is eligible when it is in the first
3 positions and is neither the original nor a sibling linked before the query (sibling rule).
Two groups by how much earlier the candidate was created than the query:
  old    more than 30 days earlier
  recent at most 7 days earlier
At most one candidate per query per group (the highest-ranked eligible one); 30 pairs are
drawn at random per group with a fixed seed.

Output: analysis/duplicate_label_audit/sample.json (issue numbers and numbers only)
Usage:  .venv/bin/python analysis/duplicate_label_audit/sample.py
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "stage2"))
from common2 import OUT2, duplicate_links  # noqa: E402
from common import load_issues, write_json  # noqa: E402

SEED = 20261009
PER_GROUP = 30
TOP = 3


def days(a: str, b: str) -> float:
    p = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (p(a) - p(b)).total_seconds() / 86400


def main() -> None:
    issues = load_issues()
    links = duplicate_links(issues)
    rows = json.load(open(OUT2 / "duplicate_clean_dev.json"))["queries"]
    groups = {"old": [], "recent": []}
    for r in rows:
        qt = issues[r["query"]]["created_at"]
        taken = set()
        for pos, c in enumerate(r["embedding_top10"][:TOP], 1):
            link = links.get(c)
            if c == r["original"] or (link and link[0] == r["original"] and link[1] < qt):
                continue
            age = days(qt, issues[c]["created_at"])
            g = "old" if age > 30 else "recent" if age <= 7 else None
            if g and g not in taken:
                taken.add(g)
                groups[g].append({"query": r["query"], "candidate": c, "position": pos,
                                  "candidate_age_days": round(age, 1), "labelled_original": r["original"]})
    rng = random.Random(SEED)
    sample = {g: sorted(rng.sample(v, PER_GROUP), key=lambda x: x["query"]) for g, v in groups.items()}
    write_json(HERE / "sample.json", {"seed": SEED, "eligible": {g: len(v) for g, v in groups.items()},
                                      "sample": sample})
    print({g: len(v) for g, v in groups.items()}, "eligible; sampled", PER_GROUP, "per group")


if __name__ == "__main__":
    main()
