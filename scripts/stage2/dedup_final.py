"""Score the frozen stage-2 duplicate pipeline and the stage-1 duplicate baseline on one split.

Final pipeline (frozen 2026-10-10): cleaned text (dedup_clean.clean_text) -> embedding retrieval
with BAAI/bge-small-en-v1.5, top 50 -> rerank with Alibaba-NLP/gte-reranker-modernbert-base
(dedup_crossencoder.py settings) -> top 10.
Stage-1 baseline: embedding retrieval with the same model on raw title + body (stage-1 cache).

Both are scored on the same queries with the sibling rule (common2.found_at), overall and by
how much older the original is; the original-only recall@10 is also reported so the stage-1 test
number can be traced. Differences (final minus baseline) use the paired bootstrap. There is no
adoption decision here: the pipeline is frozen and this only measures it. The test split is
meant to be run once; cross-encoder scores are cached, so a rerun reproduces the same numbers.

Output: outputs/stage2/duplicate_final_<split>.json (issue numbers and numbers only)
Usage:  .venv/bin/python scripts/stage2/dedup_final.py --split {tune,test}
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common2 import AGE_GROUPS, OUT2, age_group, duplicate_links, found_at
from common import N_BOOT, SEED, load_issues, load_sets, write_json
from dedup import Corpus, embed
from dedup_clean import clean_text, embed_clean
from dedup_crossencoder import score, top_candidates
from stats import mean_boot_values, paired_difference, resample_indices, summary

KS = (1, 5, 10)


def ranked_lists(queries, issues, split: str):
    raw = Corpus(issues)
    raw_top = top_candidates(queries, raw, embed(raw))
    clean = Corpus(issues)
    clean.texts = [clean_text(issues[n]) for n in clean.numbers]
    cands = top_candidates(queries, clean, embed_clean(clean))
    sc, secs = score("gte_modernbert", split, queries, cands, lambda n: clean.texts[clean.pos[n]])
    final = [[c for _, c in sorted(zip(-s, cs), key=lambda t: t[0])] for s, cs in zip(sc, cands)]
    return {"stage1_baseline": raw_top, "final_pipeline": final}, secs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "test"], required=True)
    a = ap.parse_args()
    issues = load_issues()
    links = duplicate_links(issues)
    queries = sorted(load_sets("duplicate")[a.split], key=lambda x: (x["created_at"], x["number"]))
    lists, secs = ranked_lists(queries, issues, a.split)
    groups = np.array([age_group(issues, q["number"], q["truth"]) for q in queries])

    def hits(method: str, k: int, siblings: bool = True) -> np.ndarray:
        out = []
        for q, r in zip(queries, lists[method]):
            rank = r.index(q["truth"]) + 1 if q["truth"] in r else 10 ** 6
            out.append(float(found_at(k, q["number"], q["truth"], rank, r, links if siblings else {},
                                      issues[q["number"]]["created_at"])))
        return np.array(out)

    h = {(m, k): hits(m, k) for m in lists for k in KS}
    res = {"split": a.split, "n": len(queries), "crossencoder_seconds": secs, "subsets": {},
           "original_only_recall@10": {m: float(hits(m, 10, siblings=False).mean()) for m in lists}}
    for sub in ("all", *AGE_GROUPS):
        pos = np.arange(len(queries)) if sub == "all" else np.where(groups == sub)[0]
        idx = resample_indices(len(pos), N_BOOT, SEED)
        b = {(m, k): (h[m, k][pos].mean(), mean_boot_values(h[m, k][pos], idx)) for m in lists for k in KS}
        res["subsets"][sub] = {
            "n": len(pos),
            "methods": {m: {f"recall@{k}": summary(*b[m, k]) for k in KS} for m in lists},
            "final_minus_stage1": {f"recall@{k}": paired_difference(*b["final_pipeline", k], *b["stage1_baseline", k])
                                   for k in KS},
        }
    per_query = {q["number"]: {"original": q["truth"], **{f"{m}_top10": lists[m][i][:10] for m in lists}}
                 for i, q in enumerate(queries)}
    write_json(OUT2 / f"duplicate_final_{a.split}.json", {"metrics": res, "queries": per_query})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
