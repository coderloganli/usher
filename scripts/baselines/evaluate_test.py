"""One-shot test evaluation of the frozen baselines (plan section 5).

Frozen methods and settings, read from the committed development records:
- Type: majority class; logistic regression selected by tune macro-F1; logistic regression
  selected by tune accuracy (outputs/baselines/type_tuning.json).
- Module: majority class (top-3 most frequent training modules); one-vs-rest logistic
  regression (outputs/baselines/module_tuning.json).
- Duplicate: BM25 and embedding retrieval. RRF was dropped on the development queries
  (outputs/baselines/duplicate_dev.json).
Final models are trained on the train set only, as in tuning.

Every metric gets a 95% percentile bootstrap interval (10,000 resamples of test issues,
fixed seed); all methods of a judgment share the same resamples, so differences are paired.
If a test set has issues flagged as exact-text duplicates of train, every metric is also
reported without them.

Output: outputs/baselines/<judgment>_test_predictions.json and outputs/baselines/test_metrics.json
Usage:  .venv/bin/python scripts/baselines/evaluate_test.py
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from classify import fit_quiet, logreg
from common import N_BOOT, OUT, SEED, issue_text, load_issues, load_sets, write_json
from dedup import BM25, KS, Corpus, embed, retrieve
from stats import boot_values, mean_boot_values, paired_difference, resample_indices, summary


def read(name: str) -> dict:
    with open(OUT / name, encoding="utf-8") as f:
        return json.load(f)


def subsets(test: list[dict]) -> dict[str, np.ndarray]:
    """Positions of the full test set and, if any issue is flagged, of the unflagged part."""
    out = {"all": np.arange(len(test))}
    keep = [i for i, x in enumerate(test) if not x.get("exact_dup_of_train")]
    if len(keep) < len(test):
        out["without_exact_dups"] = np.array(keep)
    return out


# ---------------------------------------------------------------- type

def evaluate_type(issues: dict) -> tuple[dict, list[dict]]:
    tuning = read("type_tuning.json")
    sets = load_sets("type")
    tr, te = sets["train"], sets["test"]
    ytr = [x["truth"] for x in tr]
    truth = np.array([x["truth"] for x in te])

    vec = TfidfVectorizer()
    Xtr = vec.fit_transform([issue_text(issues[x["number"]]) for x in tr])
    Xte = vec.transform([issue_text(issues[x["number"]]) for x in te])
    preds = {"majority": np.array([tuning["majority"]["label"]] * len(te))}
    converged = {}
    for name in ("logreg_macro_f1", "logreg_accuracy"):
        c = tuning["chosen"][name]
        m, converged[name] = fit_quiet(logreg(c["C"], c["class_weight"]), Xtr, ytr)
        preds[name] = m.predict(Xte)

    res = {"n": len(te), "settings": tuning["chosen"], "converged": converged, "subsets": {}}
    for sub, pos in subsets(te).items():
        t = truth[pos]
        idx = resample_indices(len(pos), N_BOOT, SEED)
        point, boots, block = {}, {}, {"n": len(pos), "methods": {}}
        for m, p in preds.items():
            p = p[pos]
            correct = (p == t).astype(float)
            f1 = lambda ii, p=p: f1_score(t[ii], p[ii], average="macro", zero_division=0)
            point[m, "accuracy"], boots[m, "accuracy"] = correct.mean(), mean_boot_values(correct, idx)
            point[m, "macro_f1"], boots[m, "macro_f1"] = f1(np.arange(len(pos))), boot_values(f1, idx)
            labels = sorted(set(t) | set(p))
            pr, rc, f, sup = precision_recall_fscore_support(t, p, labels=labels, zero_division=0)
            block["methods"][m] = {
                "accuracy": summary(point[m, "accuracy"], boots[m, "accuracy"]),
                "macro_f1": summary(point[m, "macro_f1"], boots[m, "macro_f1"]),
                "per_type": {l: {"test_count": int(s), "predicted": int((p == l).sum()), "precision": float(a),
                                 "recall": float(b), "f1": float(c)} for l, a, b, c, s in zip(labels, pr, rc, f, sup)},
            }
        pairs = [("logreg_macro_f1", "majority"), ("logreg_accuracy", "majority"),
                 ("logreg_macro_f1", "logreg_accuracy")]
        block["differences"] = {f"{a}_minus_{b}_{k}": paired_difference(point[a, k], boots[a, k], point[b, k], boots[b, k])
                                for a, b in pairs for k in ("macro_f1", "accuracy")}
        res["subsets"][sub] = block

    rows = [{"number": x["number"], "truth": x["truth"], **{m: str(preds[m][i]) for m in preds}}
            for i, x in enumerate(te)]
    return res, rows


# ---------------------------------------------------------------- module

def evaluate_module(issues: dict) -> tuple[dict, list[dict]]:
    tuning = read("module_tuning.json")
    sets = load_sets("module")
    tr, te = sets["train"], sets["test"]
    ytr, truth = [x["truth"] for x in tr], [x["truth"] for x in te]

    mlb = MultiLabelBinarizer()
    Ytr = mlb.fit_transform(ytr)
    classes = np.array(mlb.classes_)
    vec = TfidfVectorizer()
    Xtr = vec.fit_transform([issue_text(issues[x["number"]]) for x in tr])
    Xte = vec.transform([issue_text(issues[x["number"]]) for x in te])
    c = tuning["chosen"]
    m, converged = fit_quiet(OneVsRestClassifier(logreg(c["C"], c["class_weight"]), n_jobs=-1), Xtr, Ytr)
    order = np.argsort(-m.predict_proba(Xte), axis=1, kind="stable")[:, :3]
    ranked = {"majority": [tuning["majority"]["top3_labels"]] * len(te),
              "logreg": [list(classes[o]) for o in order]}

    hit = {(meth, k): np.array([float(bool(set(r[:k]) & set(t))) for t, r in zip(truth, ranked[meth])])
           for meth in ranked for k in (1, 3)}
    groups = subsets(te)
    multi = np.array([len(t) > 1 for t in truth])
    groups["single_module"] = np.where(~multi)[0]
    groups["multi_module"] = np.where(multi)[0]

    res = {"n": len(te), "settings": c, "converged": converged,
           "test_issues_with_module_unseen_in_train":
               [x["number"] for x in te if not set(x["truth"]) & set(classes)], "subsets": {}}
    for sub, pos in groups.items():
        idx = resample_indices(len(pos), N_BOOT, SEED)
        block = {"n": len(pos), "methods": {}, "differences": {}}
        for meth in ranked:
            block["methods"][meth] = {f"top{k}": summary(hit[meth, k][pos].mean(), mean_boot_values(hit[meth, k][pos], idx))
                                      for k in (1, 3)}
        for k in (1, 3):
            a, b = hit["logreg", k][pos], hit["majority", k][pos]
            block["differences"][f"logreg_minus_majority_top{k}"] = paired_difference(
                a.mean(), mean_boot_values(a, idx), b.mean(), mean_boot_values(b, idx))
        res["subsets"][sub] = block

    rows = [{"number": x["number"], "truth": x["truth"], "logreg_top3": ranked["logreg"][i],
             "logreg_top1_hit": int(hit["logreg", 1][i]), "logreg_top3_hit": int(hit["logreg", 3][i]),
             "majority_top1_hit": int(hit["majority", 1][i]), "majority_top3_hit": int(hit["majority", 3][i])}
            for i, x in enumerate(te)]
    return res, rows


# ---------------------------------------------------------------- duplicate

def evaluate_duplicate(issues: dict) -> tuple[dict, list[dict]]:
    dev = read("duplicate_dev.json")
    assert dev["metrics"]["keep_rrf"] is False
    methods = ("bm25", "embedding")
    te = load_sets("duplicate")["test"]
    corpus = Corpus(issues)
    rows = retrieve(te, corpus, BM25(corpus.texts), embed(corpus), methods)

    res = {"n": len(te), "methods_frozen": list(methods), "subsets": {}}
    for sub, pos in subsets(te).items():
        idx = resample_indices(len(pos), N_BOOT, SEED)
        block = {"n": len(pos), "methods": {}, "differences": {}}
        hits = {}
        for m in methods:
            ranks = np.array([rows[i][f"{m}_rank"] for i in pos])
            block["methods"][m] = {}
            for k in KS:
                h = (ranks <= k).astype(float)
                hits[m, k] = (h.mean(), mean_boot_values(h, idx))
                block["methods"][m][f"recall@{k}"] = summary(*hits[m, k])
        for k in KS:
            block["differences"][f"embedding_minus_bm25_recall@{k}"] = paired_difference(
                *hits["embedding", k], *hits["bm25", k])
        res["subsets"][sub] = block
    return res, rows


def main() -> None:
    issues = load_issues()
    metrics = {"n_boot": N_BOOT, "seed": SEED,
               "ground_truth_note": "reviewed by an AI model (Claude Opus 5.5), not human-verified"}
    for name, fn in (("type", evaluate_type), ("module", evaluate_module), ("duplicate", evaluate_duplicate)):
        res, rows = fn(issues)
        metrics[name] = res
        write_json(OUT / f"{name}_test_predictions.json", rows)
        print(name, "done", flush=True)
    write_json(OUT / "test_metrics.json", metrics)


if __name__ == "__main__":
    main()
