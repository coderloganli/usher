"""Compute headline metrics, 95% bootstrap CIs and paired bootstrap differences.

Bootstrap: N_BOOT resamples of test issues with replacement, fixed seed,
percentile intervals (2.5th, 97.5th). All methods within a comparison share the
same resampled indices (paired bootstrap). Macro-F1 is recomputed on each
resample (sklearn, average="macro" over the labels present in gold or
predictions of that resample, zero_division=0).

Best assignee model = the one with higher top-1 on the full assignee test set
(ties broken by top-3); it is compared with the bot on the bot subset.

Output: results/metrics.json
Usage:  python scripts/evaluate.py
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score

from common import N_BOOT, RESULTS, SEED, write_json

METHODS = ("tfidf", "embedding")


def ci(boot):
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def read(path):
    by = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            by[r["method"]][int(r["number"])] = r
    return by


def type_metrics(rng):
    by = read(RESULTS / "per_issue_type.csv")
    nums = sorted(by["tfidf"])
    assert nums == sorted(by["embedding"])
    n = len(nums)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    gold = np.array([by["tfidf"][k]["gold"] for k in nums])
    out = {"n": n, "gold_distribution": {g: int((gold == g).sum()) for g in sorted(set(gold))}, "methods": {}}
    boots = {}
    for m in METHODS:
        pred = np.array([by[m][k]["pred"] for k in nums])
        corr = (pred == gold).astype(float)
        f1 = lambda ii: f1_score(gold[ii], pred[ii], average="macro", zero_division=0)
        b_acc = corr[idx].mean(axis=1)
        b_f1 = np.array([f1(ii) for ii in idx])
        boots[m] = (corr, b_acc, b_f1)
        out["methods"][m] = {"accuracy": float(corr.mean()), "accuracy_ci95": ci(b_acc),
                             "correct": int(corr.sum()),
                             "macro_f1": float(f1(np.arange(n))), "macro_f1_ci95": ci(b_f1),
                             "pred_distribution": {p: int((pred == p).sum()) for p in sorted(set(pred))}}
    e, t = boots["embedding"], boots["tfidf"]
    out["diff_embedding_minus_tfidf"] = {
        "accuracy": out["methods"]["embedding"]["accuracy"] - out["methods"]["tfidf"]["accuracy"],
        "accuracy_ci95": ci(e[1] - t[1]),
        "macro_f1": out["methods"]["embedding"]["macro_f1"] - out["methods"]["tfidf"]["macro_f1"],
        "macro_f1_ci95": ci(e[2] - t[2])}
    return out


def assignee_metrics(rng):
    by = read(RESULTS / "per_issue_assignee.csv")
    nums = sorted(by["tfidf"])
    assert nums == sorted(by["embedding"])
    n = len(nums)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    base = by["tfidf"]
    out = {"n": n,
           "gold_not_in_label_space": int(sum(base[k]["gold_in_label_space"] == "0" for k in nums)),
           "author_in_final_assignees": int(sum(base[k]["author"] in base[k]["gold"].split("|") for k in nums)),
           "methods": {}}
    arr = {}
    for m in METHODS:
        a1 = np.array([int(by[m][k]["top1_correct"]) for k in nums], float)
        a3 = np.array([int(by[m][k]["top3_correct"]) for k in nums], float)
        arr[m] = (a1, a3)
        out["methods"][m] = {"top1": float(a1.mean()), "top1_ci95": ci(a1[idx].mean(axis=1)),
                             "top1_correct": int(a1.sum()),
                             "top3": float(a3.mean()), "top3_ci95": ci(a3[idx].mean(axis=1)),
                             "top3_correct": int(a3.sum())}
    d1 = arr["embedding"][0] - arr["tfidf"][0]
    d3 = arr["embedding"][1] - arr["tfidf"][1]
    out["diff_embedding_minus_tfidf"] = {"top1": float(d1.mean()), "top1_ci95": ci(d1[idx].mean(axis=1)),
                                         "top3": float(d3.mean()), "top3_ci95": ci(d3[idx].mean(axis=1))}
    # Bot subset: test issues where vs-code-engineering[bot] made an assignment.
    best = max(METHODS, key=lambda m: (out["methods"][m]["top1"], out["methods"][m]["top3"]))
    sub = [i for i, k in enumerate(nums) if base[k]["bot_correct"] != ""]
    ns = len(sub)
    bot = np.array([int(base[nums[i]]["bot_correct"]) for i in sub], float)
    idx_s = rng.integers(0, ns, size=(N_BOOT, ns))
    bs = {"n": ns, "n_without_bot_assignment": n - ns, "best_model": best,
          "bot_top1": float(bot.mean()), "bot_top1_ci95": ci(bot[idx_s].mean(axis=1)), "bot_correct": int(bot.sum()),
          "methods": {}}
    for m in METHODS:
        s1 = arr[m][0][sub]
        bs["methods"][m] = {"top1": float(s1.mean()), "top1_ci95": ci(s1[idx_s].mean(axis=1)),
                            "top1_correct": int(s1.sum()), "top3": float(arr[m][1][sub].mean())}
    d = arr[best][0][sub] - bot
    bs["diff_best_minus_bot_top1"] = {"point": float(d.mean()), "ci95": ci(d[idx_s].mean(axis=1)),
                                      "model_only": int(((d > 0)).sum()), "bot_only": int((d < 0).sum()),
                                      "both": int(((arr[best][0][sub] == 1) & (bot == 1)).sum()),
                                      "neither": int(((arr[best][0][sub] == 0) & (bot == 0)).sum())}
    out["bot_subset"] = bs
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    res = {"n_boot": N_BOOT, "seed": SEED, "type": type_metrics(rng), "assignee": assignee_metrics(rng)}
    write_json(RESULTS / "metrics.json", res)
    import json
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
