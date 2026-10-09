"""Reciprocal rank fusion (RRF) of BM25 and embedding retrieval on cleaned text,
compared with embedding retrieval alone on the development queries.

Both retrievers use the cleaned text of scripts/stage2/dedup_clean.py; RRF is the stage-1
definition (k = 60, competition ranks). Scoring uses the sibling rule and reports recall@k
overall and by how much older the original is (common2.AGE_GROUPS).

Decision rule fixed in advance (2026-10-09): RRF is adopted only if
  (a) RRF minus embedding recall@10 has a 95% paired bootstrap interval entirely above 0, and
  (b) on the over-30-days group, that interval is not entirely below 0.
Test queries are not used.

Output: outputs/stage2/duplicate_fusion_dev.json
Usage:  .venv/bin/python scripts/stage2/dedup_fusion.py
"""
from __future__ import annotations

import json

import numpy as np

from common2 import AGE_GROUPS, OUT2, age_group, duplicate_links, found_at
from common import N_BOOT, SEED, load_issues, load_sets, write_json
from dedup import BM25, Corpus, retrieve
from dedup_clean import clean_text, embed_clean
from stats import mean_boot_values, paired_difference, resample_indices, summary

KS = (1, 5, 10)
METHODS = ("bm25", "embedding", "rrf")


def main() -> None:
    issues = load_issues()
    links = duplicate_links(issues)
    sets = load_sets("duplicate")
    dev = sorted(sets["train"] + sets["tune"], key=lambda x: (x["created_at"], x["number"]))

    corpus = Corpus(issues)
    corpus.texts = [clean_text(issues[n]) for n in corpus.numbers]
    rows = retrieve(dev, corpus, BM25(corpus.texts), embed_clean(corpus), METHODS)
    groups = np.array([age_group(issues, r["query"], r["original"]) for r in rows])

    hit = {(m, k): np.array([float(found_at(k, r["query"], r["original"], r[f"{m}_rank"], r[f"{m}_top10"],
                                            links, issues[r["query"]]["created_at"])) for r in rows])
           for m in METHODS for k in KS}
    res = {"n_queries": len(rows), "subsets": {}}
    for sub in ("all", *AGE_GROUPS):
        pos = np.arange(len(rows)) if sub == "all" else np.where(groups == sub)[0]
        idx = resample_indices(len(pos), N_BOOT, SEED)
        b = {m: {k: (hit[m, k][pos].mean(), mean_boot_values(hit[m, k][pos], idx)) for k in KS} for m in METHODS}
        res["subsets"][sub] = {
            "n": len(pos),
            "methods": {m: {f"recall@{k}": summary(*b[m][k]) for k in KS} for m in METHODS},
            "rrf_minus_embedding": {f"recall@{k}": paired_difference(*b["rrf"][k], *b["embedding"][k]) for k in KS},
        }
    d_all = res["subsets"]["all"]["rrf_minus_embedding"]["recall@10"]["ci95"]
    d_old = res["subsets"]["over_30_days"]["rrf_minus_embedding"]["recall@10"]["ci95"]
    res["adopt_rrf"] = bool(d_all[0] > 0 and not d_old[1] < 0)
    write_json(OUT2 / "duplicate_fusion_dev.json", {"metrics": res, "queries": rows})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
