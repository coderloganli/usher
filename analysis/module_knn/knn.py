"""Similar-issue vote for the module judgment, compared with the stage-1 LR (no LLM).

Agreed with the user on 2026-10-10, step 1 of the plan "LLM makes the final module judgment with
reference material": how good, and how complementary to LR, is a vote of the most similar labelled
issues? The answer decides whether similar issues go into the LLM's reference material.

Fixed before the run:
- Embeddings: the cached bge-small embeddings of cleaned text (scripts/stage2/dedup_clean.py).
- Neighbours of a tune issue: the 10 most similar train-period module issues by cosine
  (all of them were created before every tune issue).
- Vote: each neighbour adds its cosine to every module label it has; modules are ranked by total.
- LR: stage-1 module LR (TF-IDF defaults, one-vs-rest, C = 10, balanced), refitted here.
- Groups by how many train issues carry the issue's best-represented true module
  (1-19, 20-99, 100+), as in the rarity check of 2026-10-10.
- Paired bootstrap, 10,000 resamples, fixed seed.

Output: analysis/module_knn/summary.json (no issue text)
Usage:  .venv/bin/python analysis/module_knn/knn.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(ROOT / "analysis" / "module_repo_evidence"))
from common import N_BOOT, SEED, load_issues, load_sets, write_json  # noqa: E402
from evidence import lr_top10  # noqa: E402
from stats import mean_boot_values, paired_difference, resample_indices, summary  # noqa: E402

EMB = ROOT / "data" / "stage2" / "embeddings_clean.npy"
K = 10


def main() -> None:
    issues = load_issues()
    order = [r["number"] for r in sorted(issues.values(), key=lambda r: (r["created_at"], r["number"]))]
    pos = {n: i for i, n in enumerate(order)}
    emb = np.load(EMB)
    assert emb.shape[0] == len(order)

    sets = load_sets("module")
    tr, tu = sets["train"], sets["tune"]
    Etr = emb[[pos[x["number"]] for x in tr]]
    Etu = emb[[pos[x["number"]] for x in tu]]
    sims = Etu @ Etr.T
    lr = lr_top10(issues)
    freq = Counter(m for x in tr for m in x["truth"])

    hit = lambda truth, ranked, k: float(bool(set(ranked[:k]) & set(truth)))
    knn1, knn3, knn10, lr1, lr3, groups = [], [], [], [], [], []
    for x, s in zip(tu, sims):
        nb = np.argsort(-s, kind="stable")[:K]
        score = defaultdict(float)
        for j in nb:
            for m in tr[j]["truth"]:
                score[m] += float(s[j])
        ranked = [m for m, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]
        t = x["truth"]
        knn1.append(hit(t, ranked, 1)); knn3.append(hit(t, ranked, 3)); knn10.append(hit(t, ranked, 10))
        lr1.append(hit(t, lr[x["number"]], 1)); lr3.append(hit(t, lr[x["number"]], 3))
        f = max(freq[m] for m in t)
        groups.append("0" if f == 0 else "1-19" if f < 20 else "20-99" if f < 100 else "100+")
    knn1, knn3, knn10, lr1, lr3 = map(np.array, (knn1, knn3, knn10, lr1, lr3))
    groups = np.array(groups)

    def compare(mask: np.ndarray) -> dict:
        a, b = knn1[mask], lr1[mask]
        idx = resample_indices(len(a), N_BOOT, SEED)
        ba, bb = mean_boot_values(a, idx), mean_boot_values(b, idx)
        return {"n": int(mask.sum()),
                "knn_top1": summary(a.mean(), ba), "lr_top1": summary(b.mean(), bb),
                "knn_minus_lr_top1": paired_difference(a.mean(), ba, b.mean(), bb),
                "knn_top3": float(knn3[mask].mean()), "lr_top3": float(lr3[mask].mean()),
                "both_right": int(((a == 1) & (b == 1)).sum()), "only_knn_right": int(((a == 1) & (b == 0)).sum()),
                "only_lr_right": int(((a == 0) & (b == 1)).sum()), "both_wrong": int(((a == 0) & (b == 0)).sum()),
                "either_right": float(np.maximum(a, b).mean())}

    out = {"k": K, "embedding": "bge-small-en-v1.5 on cleaned text (stage-2 cache)", "tune_issues": len(tu),
           "all": {**compare(np.ones(len(tu), bool)), "knn_top10": float(knn10.mean())},
           "by_train_frequency_of_true_module": {g: compare(groups == g) for g in ("1-19", "20-99", "100+")
                                                 if (groups == g).any()}}
    write_json(HERE / "summary.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
