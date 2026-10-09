"""Summarise why stage-1 retrieval missed duplicates (development queries, raw text).

categories.csv holds the reading of 40 sampled misses by an AI model (Claude Opus 5.5),
not human-verified. Categories:
  R  different symptoms, same root cause (both are user reports)
  N  near miss: the original is ranked 11-100 by embedding retrieval
  P  same symptom, very different wording
  U  the original is a maintainer's technical note or tracking item, not a user's symptom
  C  other reports of the same problem ("siblings") outrank the original
  L  query and original are in different languages
  T  little text (mostly an image or video)
  H  the body is pasted markup
  J  junk query
Across all 706 misses this script also counts, from the snapshot's final duplicateOf links,
how many have a sibling in a top 10 (a candidate linked to the same original; the time of
the link is not checked here), and how far down the originals are ranked.

Output: analysis/duplicate_miss_diagnosis/summary.json
Usage:  python3 analysis/duplicate_miss_diagnosis/summary.py
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    with open(HERE / "categories.csv", encoding="utf-8") as f:
        cats = Counter(r["category"] for r in csv.DictReader(f))
    dup_of = {}
    with gzip.open(ROOT / "data" / "issues.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            dup_of[r["number"]] = (r["duplicate_of"] or {}).get("number")
    rows = json.load(open(ROOT / "outputs" / "baselines" / "duplicate_dev.json"))["queries"]
    miss = [r for r in rows if r["bm25_rank"] > 10 and r["embedding_rank"] > 10]
    sib = lambda r, m: any(dup_of.get(c) == r["original"] and c != r["query"] for c in r[f"{m}_top10"])
    ranks = [r["embedding_rank"] for r in miss]
    out = {
        "reviewer": "AI model (Claude Opus 5.5), not human-verified",
        "sample_categories": dict(cats.most_common()),
        "misses": len(miss),
        "misses_with_sibling_in_top10_final_state": {
            "embedding": sum(sib(r, "embedding") for r in miss), "bm25": sum(sib(r, "bm25") for r in miss),
            "either": sum(sib(r, "embedding") or sib(r, "bm25") for r in miss)},
        "embedding_rank_of_missed_original": {
            "11-50": sum(10 < x <= 50 for x in ranks), "51-200": sum(50 < x <= 200 for x in ranks),
            "over_200": sum(x > 200 for x in ranks)},
    }
    with open(HERE / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
