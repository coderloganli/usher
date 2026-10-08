"""Type and module baselines: majority class and TF-IDF + logistic regression.

Plan sections 4.1, 4.2 and 5. This script only uses the train and tune sets:
for every candidate setting it fits on train and scores on tune, then records the
chosen setting. The test set is handled by a later step after the settings are frozen.

- TF-IDF: scikit-learn TfidfVectorizer with library defaults, fitted on train text only.
- Type: multinomial logistic regression, two baselines from the same grid: one selected by
  tune macro-F1 (ties: accuracy), one selected by tune accuracy (ties: macro-F1).
- Module: one-vs-rest logistic regression (one yes/no classifier per module seen in train),
  modules ranked by probability; selection by tune top-1 hit rate (ties: top-3).
- Grid: C in {0.01, 0.1, 1, 10, 100} x class_weight in {None, "balanced"}.
- max_iter is raised from the default 100 to 5,000 so that every fit converges; it is
  a solver limit, not a model setting, and any non-convergence is recorded.

Output: outputs/baselines/<judgment>_tuning.json
Usage:  .venv/bin/python scripts/baselines/classify.py
"""
from __future__ import annotations

import time
import warnings
from collections import Counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from common import OUT, SEED, issue_text, load_issues, load_sets, write_json

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
CW_GRID = [None, "balanced"]
MAX_ITER = 5000


def logreg(C: float, cw) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight=cw, max_iter=MAX_ITER, random_state=SEED)


def fit_quiet(model, X, y) -> tuple[object, bool]:
    """Fit and report whether every logistic regression stopped before MAX_ITER.
    Iteration counts are checked directly because one-vs-rest fits run in worker processes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
    fitted = getattr(model, "estimators_", [model])
    return model, all(int(np.max(e.n_iter_)) < MAX_ITER for e in fitted)


# ---------------------------------------------------------------- type

def type_scores(truth: list[str], pred: list[str]) -> dict:
    t, p = np.array(truth), np.array(pred)
    return {"macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
            "accuracy": float((t == p).mean())}


def tune_type(issues: dict) -> dict:
    sets = load_sets("type")
    tr, tu = sets["train"], sets["tune"]
    ytr, ytu = [x["truth"] for x in tr], [x["truth"] for x in tu]

    majority = Counter(ytr).most_common(1)[0][0]
    vec = TfidfVectorizer()
    Xtr = vec.fit_transform([issue_text(issues[x["number"]]) for x in tr])
    Xtu = vec.transform([issue_text(issues[x["number"]]) for x in tu])

    grid = []
    for C in C_GRID:
        for cw in CW_GRID:
            t0 = time.time()
            m, ok = fit_quiet(logreg(C, cw), Xtr, ytr)
            row = {"C": C, "class_weight": cw, "converged": ok, "seconds": round(time.time() - t0, 1),
                   **type_scores(ytu, list(m.predict(Xtu)))}
            grid.append(row)
            print("type", row, flush=True)
    # Two type baselines (decided 2026-10-07 after the tune run showed that the two metrics
    # pick different settings): one selected by macro-F1, one selected by accuracy.
    best_f1 = max(grid, key=lambda g: (g["macro_f1"], g["accuracy"]))
    best_acc = max(grid, key=lambda g: (g["accuracy"], g["macro_f1"]))
    return {
        "n": {"train": len(tr), "tune": len(tu)},
        "train_label_counts": dict(Counter(ytr).most_common()),
        "tune_label_counts": dict(Counter(ytu).most_common()),
        "tfidf_vocabulary_size": len(vec.vocabulary_),
        "majority": {"label": majority, "tune": type_scores(ytu, [majority] * len(tu))},
        "logreg_grid": grid,
        "chosen": {
            "logreg_macro_f1": {"rule": "highest tune macro_f1, ties broken by tune accuracy",
                                "C": best_f1["C"], "class_weight": best_f1["class_weight"]},
            "logreg_accuracy": {"rule": "highest tune accuracy, ties broken by tune macro_f1",
                                "C": best_acc["C"], "class_weight": best_acc["class_weight"]},
        },
    }


# ---------------------------------------------------------------- module

def hit_rates(truth: list[list[str]], ranked: list[list[str]]) -> dict:
    top1 = np.mean([r[0] in t for t, r in zip(truth, ranked)])
    top3 = np.mean([bool(set(r[:3]) & set(t)) for t, r in zip(truth, ranked)])
    return {"top1": float(top1), "top3": float(top3)}


def tune_module(issues: dict) -> dict:
    sets = load_sets("module")
    tr, tu = sets["train"], sets["tune"]
    ytr, ytu = [x["truth"] for x in tr], [x["truth"] for x in tu]

    freq = Counter(m for t in ytr for m in t)
    majority = [m for m, _ in freq.most_common(3)]
    mlb = MultiLabelBinarizer()
    Ytr = mlb.fit_transform(ytr)
    classes = np.array(mlb.classes_)
    vec = TfidfVectorizer()
    Xtr = vec.fit_transform([issue_text(issues[x["number"]]) for x in tr])
    Xtu = vec.transform([issue_text(issues[x["number"]]) for x in tu])

    grid = []
    for C in C_GRID:
        for cw in CW_GRID:
            t0 = time.time()
            m, ok = fit_quiet(OneVsRestClassifier(logreg(C, cw), n_jobs=-1), Xtr, Ytr)
            order = np.argsort(-m.predict_proba(Xtu), axis=1, kind="stable")[:, :3]
            row = {"C": C, "class_weight": cw, "converged": ok, "seconds": round(time.time() - t0, 1),
                   **hit_rates(ytu, [list(classes[o]) for o in order])}
            grid.append(row)
            print("module", row, flush=True)
    best = max(grid, key=lambda g: (g["top1"], g["top3"]))
    unseen = sorted({m for t in ytu for m in t} - set(classes))
    return {
        "n": {"train": len(tr), "tune": len(tu)},
        "modules_in_train": len(classes),
        "tune_modules_unseen_in_train": unseen,
        "tune_issues_with_only_unseen_modules": sum(1 for t in ytu if not set(t) & set(classes)),
        "tfidf_vocabulary_size": len(vec.vocabulary_),
        "majority": {"top3_labels": majority, "tune": hit_rates(ytu, [majority] * len(tu))},
        "logreg_grid": grid,
        "selection_rule": "highest tune top1, ties broken by tune top3",
        "chosen": {"C": best["C"], "class_weight": best["class_weight"]},
    }


def main() -> None:
    issues = load_issues()
    write_json(OUT / "type_tuning.json", tune_type(issues))
    write_json(OUT / "module_tuning.json", tune_module(issues))


if __name__ == "__main__":
    main()
