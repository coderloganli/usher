"""Train TF-IDF + LR and bge-small embedding + LR for the type and assignee tasks.

Hyperparameters are chosen on the tune split (2026-06) only; the final models
are then fit on the train split (2026-01..05) and applied once to the test set.

- Type: multinomial LR; grid over C and class_weight; select by tune macro-F1.
- Assignee: multinomial LR over assignees seen in training; an issue with k
  final assignees becomes k training rows with weight 1/k; grid over C;
  select by tune top-1 accuracy (top-1 correct if it is in the final assignee set).
- Embeddings: reused from dedup_experiment/data/embeddings.npy (BAAI/bge-small-en-v1.5,
  pinned revision, same issue_text(), L2-normalised).

Output: results/per_issue_type.csv, results/per_issue_assignee.csv, results/tuning.json
Usage:  python scripts/train_predict.py
"""
from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from common import (DEDUP_DATA, MODEL_ID, MODEL_REVISION, RESULTS, SEED, issue_text,
                    load_text_corpus, write_json)

C_GRID = [0.1, 1.0, 10.0, 100.0]
CW_GRID = [None, "balanced"]


def load_features():
    corpus = load_text_corpus()
    emb_meta = json.load(open(DEDUP_DATA / "embeddings_meta.json"))
    assert emb_meta["model_id"] == MODEL_ID and emb_meta["revision"] == MODEL_REVISION
    assert emb_meta["n"] == len(corpus)
    assert emb_meta["numbers_first_last"] == [int(corpus[0]["number"]), int(corpus[-1]["number"])]
    emb = np.load(DEDUP_DATA / "embeddings.npy", mmap_mode="r")
    row = {int(r["number"]): i for i, r in enumerate(corpus)}
    text = {int(r["number"]): issue_text(r["title"], r["body"]) for r in corpus}
    return text, emb, row


def tfidf():
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9,
                           sublinear_tf=True, max_features=200_000)


def lr(C, cw=None):
    return LogisticRegression(C=C, class_weight=cw, max_iter=3000, random_state=SEED)


def featurize(method, train_nums, other_nums_list, text, emb, row):
    if method == "tfidf":
        vec = tfidf()
        Xtr = vec.fit_transform([text[n] for n in train_nums])
        return Xtr, [vec.transform([text[n] for n in ns]) for ns in other_nums_list]
    get = lambda ns: np.asarray(emb[[row[n] for n in ns]], dtype=np.float32)
    return get(train_nums), [get(ns) for ns in other_nums_list]


def run_type(sets, text, emb, row, tuning, per_issue):
    tr, tu, te = sets["train"], sets["tune"], sets["test"]
    ytr = [x["label"] for x in tr]
    ytu = [x["label"] for x in tu]
    for method in ("tfidf", "embedding"):
        Xtr, (Xtu, Xte) = featurize(method, [x["number"] for x in tr],
                                    [[x["number"] for x in tu], [x["number"] for x in te]], text, emb, row)
        grid = []
        for C in C_GRID:
            for cw in CW_GRID:
                p = lr(C, cw).fit(Xtr, ytr).predict(Xtu)
                grid.append({"C": C, "class_weight": cw, "tune_macro_f1": f1_score(ytu, p, average="macro"),
                             "tune_accuracy": float(np.mean(np.array(p) == np.array(ytu)))})
        best = max(grid, key=lambda g: (g["tune_macro_f1"], g["tune_accuracy"]))
        tuning[f"type/{method}"] = {"grid": grid, "best": best}
        pred = lr(best["C"], best["class_weight"]).fit(Xtr, ytr).predict(Xte)
        for x, p in zip(te, pred):
            per_issue.append({"number": x["number"], "method": method, "gold": x["label"], "pred": p,
                              "correct": int(p == x["label"])})
        print("type", method, best)


def expand(items, keep=None):
    nums, ys, ws = [], [], []
    for x in items:
        a = [u for u in x["assignees"] if keep is None or u in keep]
        for u in a:
            nums.append(x["number"]); ys.append(u); ws.append(1.0 / len(a))
    return nums, ys, np.array(ws)


def topk(model, X, k):
    proba = model.predict_proba(X)
    order = np.argsort(-proba, axis=1, kind="stable")[:, :k]
    return [[model.classes_[j] for j in r] for r in order]


def run_assignee(sets, text, emb, row, tuning, per_issue, bot):
    tr, tu, te = sets["train"], sets["tune"], sets["test"]
    nums_tr, ytr, wtr = expand(tr)
    space = set(ytr)
    for method in ("tfidf", "embedding"):
        Xtr, (Xtu, Xte) = featurize(method, nums_tr, [[x["number"] for x in tu], [x["number"] for x in te]],
                                    text, emb, row)
        grid = []
        for C in C_GRID:
            m = lr(C).fit(Xtr, ytr, sample_weight=wtr)
            t = topk(m, Xtu, 3)
            grid.append({"C": C,
                         "tune_top1": float(np.mean([p[0] in x["assignees"] for p, x in zip(t, tu)])),
                         "tune_top3": float(np.mean([bool(set(p) & set(x["assignees"])) for p, x in zip(t, tu)]))})
            print("  assignee", method, grid[-1])
        best = max(grid, key=lambda g: (g["tune_top1"], g["tune_top3"]))
        tuning[f"assignee/{method}"] = {"grid": grid, "best": best, "n_classes": len(space),
                                        "n_train_rows": len(ytr)}
        m = lr(best["C"]).fit(Xtr, ytr, sample_weight=wtr)
        for x, p in zip(te, topk(m, Xte, 3)):
            b = bot[str(x["number"])]["bot_assignee"]
            per_issue.append({
                "number": x["number"], "method": method, "gold": "|".join(x["assignees"]),
                "author": x["author"], "gold_in_label_space": int(bool(set(x["assignees"]) & space)),
                "top1": p[0], "top3": "|".join(p), "top1_correct": int(p[0] in x["assignees"]),
                "top3_correct": int(bool(set(p) & set(x["assignees"]))),
                "bot_assignee": b or "", "bot_correct": "" if b is None else int(b in x["assignees"])})
        print("assignee", method, best)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    sets = json.load(open(RESULTS / "sets.json"))["sets"]
    bot = json.load(open(RESULTS / "bot_assign.json"))
    text, emb, row = load_features()
    tuning, per_type, per_asg = {}, [], []
    run_type(sets["type"], text, emb, row, tuning, per_type)
    run_assignee(sets["assignee"], text, emb, row, tuning, per_asg, bot)
    write_csv(RESULTS / "per_issue_type.csv", per_type)
    write_csv(RESULTS / "per_issue_assignee.csv", per_asg)
    write_json(RESULTS / "tuning.json", tuning)


if __name__ == "__main__":
    main()
