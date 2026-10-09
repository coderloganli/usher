"""Sample development queries that both stage-1 retrievers missed, for a reading of why.

Source: stage-1 development run on raw text (outputs/baselines/duplicate_dev.json).
A query is a miss when neither BM25 nor embedding retrieval ranks the original within 10.
40 misses are drawn with a fixed seed.

Output: analysis/duplicate_miss_diagnosis/sample.json (issue numbers and ranks only)
Usage:  python3 analysis/duplicate_miss_diagnosis/sample.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 20261008
N = 40


def main() -> None:
    rows = json.load(open(ROOT / "outputs" / "baselines" / "duplicate_dev.json"))["queries"]
    miss = [r for r in rows if r["bm25_rank"] > 10 and r["embedding_rank"] > 10]
    random.seed(SEED)
    sample = sorted(random.sample(miss, N), key=lambda r: r["query"])
    out = [{"query": r["query"], "original": r["original"], "bm25_rank": r["bm25_rank"],
            "embedding_rank": r["embedding_rank"]} for r in sample]
    with open(HERE / "sample.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "misses": len(miss), "sample": out}, f, indent=1)
        f.write("\n")
    print(len(miss), "misses; sampled", N)


if __name__ == "__main__":
    main()
