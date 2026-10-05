"""Compute recall@1/5/10 per method, 95% bootstrap CIs for recall@5, and a
paired bootstrap CI for recall@5(embedding) - recall@5(BM25).

Bootstrap: 1,000 resamples of queries with replacement, fixed seed,
percentile intervals (2.5th, 97.5th). The paired bootstrap resamples the
same query indices for both methods.

Output: results/metrics.json
Usage:  python scripts/evaluate.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from common import RESULTS, SEED, write_json

N_BOOT = 1000


def main() -> None:
    hits = defaultdict(dict)
    with open(RESULTS / "per_query.csv") as f:
        for r in csv.DictReader(f):
            hits[r["method"]][int(r["query"])] = {k: int(r[f"hit@{k}"]) for k in (1, 5, 10)}
    queries = sorted(hits["bm25"])
    assert queries == sorted(hits["embedding"])
    n = len(queries)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))

    res = {"n_queries": n, "n_boot": N_BOOT, "seed": SEED, "methods": {}}
    h5 = {}
    for m in ("bm25", "embedding"):
        arr = {k: np.array([hits[m][q][k] for q in queries]) for k in (1, 5, 10)}
        h5[m] = arr[5]
        boot = arr[5][idx].mean(axis=1)
        res["methods"][m] = {f"recall@{k}": float(arr[k].mean()) for k in (1, 5, 10)}
        res["methods"][m].update({f"hits@{k}": int(arr[k].sum()) for k in (1, 5, 10)})
        res["methods"][m]["recall@5_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    diff = h5["embedding"] - h5["bm25"]
    boot_d = diff[idx].mean(axis=1)
    res["diff_recall@5"] = {
        "point": float(diff.mean()),
        "ci95": [float(np.percentile(boot_d, 2.5)), float(np.percentile(boot_d, 97.5))],
        "embedding_only_hits": int(((h5["embedding"] == 1) & (h5["bm25"] == 0)).sum()),
        "bm25_only_hits": int(((h5["embedding"] == 0) & (h5["bm25"] == 1)).sum()),
        "both_hit": int(((h5["embedding"] == 1) & (h5["bm25"] == 1)).sum()),
        "neither": int(((h5["embedding"] == 0) & (h5["bm25"] == 0)).sum()),
    }
    # Sensitivity check: drop queries opened by bot accounts.
    tests = {t["query"]: t for t in json.load(open(RESULTS / "testset.json"))["queries"]}
    human = np.array([not tests[q]["author_is_bot"] for q in queries])
    hb, he = h5["bm25"][human], h5["embedding"][human]
    idx_h = rng.integers(0, len(hb), size=(N_BOOT, len(hb)))
    boot_h = (he - hb)[idx_h].mean(axis=1)
    res["human_authored_only"] = {
        "n_queries": int(human.sum()), "n_bot_authored": int((~human).sum()),
        "bm25_recall@5": float(hb.mean()), "embedding_recall@5": float(he.mean()),
        "diff_recall@5": float((he - hb).mean()),
        "diff_ci95": [float(np.percentile(boot_h, 2.5)), float(np.percentile(boot_h, 97.5))]}
    write_json(RESULTS / "metrics.json", res)
    for m, v in res["methods"].items():
        print(m, {k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()})
    print("diff recall@5", res["diff_recall@5"])
    print("human-authored only", res["human_authored_only"])


if __name__ == "__main__":
    main()
