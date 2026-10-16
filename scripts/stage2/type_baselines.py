"""Type judgment, stage 2: the evaluation subset and the baselines on it (step 1 of the type plan).

Plan approved by the user on 2026-10-15 (see EXPERIMENTS.md, entry 21 for the background):

- Evaluation subset: type issues opened by non-team reporters. "Team member" is approximated by
  "account that appears as an assignee on any snapshot issue" (as in analysis/triage_signals/).
  This only selects which issues are scored; no method uses it as an input.
- Baselines, all scored on the subset:
    always bug       the most common train type;
    reporter choice  the type the reporter chose (bug or feature template, or the Type field of
                     VS Code's issue reporter), `bug` when there is none;
    LR (macro-F1)    stage-1 logistic regression, C = 1, class_weight balanced;
    LR (accuracy)    stage-1 logistic regression, C = 1, no class weighting.
  Both logistic regressions are the stage-1 models unchanged: TF-IDF defaults and model fitted on
  the whole train set (team and non-team issues), title + raw body as input.
- Macro-F1 is scikit-learn's macro average over the labels that occur in truth or prediction,
  as in stage 1.

This script scores the tune subset only. The test subset is scored once, after the type LLM
method is frozen.

Output: outputs/stage2/type_baselines_tune.json
Usage:  .venv/bin/python scripts/stage2/type_baselines.py
"""
from __future__ import annotations

import re
import sys
import warnings
from collections import Counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from common2 import OUT2
from common import N_BOOT, SEED, issue_text, load_issues, load_sets, write_json
from stats import mean_boot_values, paired_difference, resample_indices, summary

TYPES = ("bug", "feature-request", "*question", "under-discussion", "debt", "polish", "upstream",
         "engineering", "upstream-issue-linked", "electron")
LR_SETTINGS = {"lr_macro_f1": {"C": 1.0, "class_weight": "balanced"},
               "lr_accuracy": {"C": 1.0, "class_weight": None}}
MAX_ITER = 5000


def team_accounts(issues: dict[int, dict]) -> set[str]:
    team = set()
    for r in issues.values():
        team.update(r["assignees"] or [])
    return team


def non_team(rows: list[dict], issues: dict[int, dict], team: set[str]) -> list[dict]:
    return [x for x in rows if issues[x["number"]]["author"] not in team]


def reporter_choice(body: str) -> str | None:
    """Same rule as analysis/triage_signals/signals.py (user_choice)."""
    m = re.search(r"^\s*Type:\s*<b>([^<]+)</b>", body, re.M)
    if m:
        return {"Bug": "bug", "Feature Request": "feature-request", "Performance Issue": "bug"}.get(m.group(1).strip())
    if "bug_report_template" in body or "Copilot Chat Extension Version" in body:
        return "bug"
    if "feature_request_template" in body:
        return "feature-request"
    return None


def fit_type_models(issues: dict[int, dict], train: list[dict]):
    """The two stage-1 type logistic regressions, fitted on the whole train set."""
    vec = TfidfVectorizer()
    X = vec.fit_transform([issue_text(issues[x["number"]]) for x in train])
    y = [x["truth"] for x in train]
    models = {}
    for name, s in LR_SETTINGS.items():
        m = LogisticRegression(C=s["C"], class_weight=s["class_weight"], max_iter=MAX_ITER, random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(X, y)
        assert int(np.max(m.n_iter_)) < MAX_ITER, f"{name} did not converge"
        models[name] = m
    return vec, models


def macro_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(truth, pred, average="macro", zero_division=0))


def score(preds: dict[str, list[str]], truth: list[str]) -> dict:
    """Accuracy and macro-F1 with 95% bootstrap intervals, plus paired differences on the
    comparisons named in the adoption rule and against the reporter's choice."""
    t = np.array(truth)
    idx = resample_indices(len(t), N_BOOT, SEED)
    acc, f1 = {}, {}
    for name, p in preds.items():
        p = np.array(p)
        hit = (p == t).astype(float)
        acc[name] = (hit.mean(), mean_boot_values(hit, idx))
        f1[name] = (macro_f1(t, p), np.array([macro_f1(t[r], p[r]) for r in idx]))
    out = {"methods": {n: {"accuracy": summary(*acc[n]), "macro_f1": summary(*f1[n])} for n in preds}}
    pairs = [("lr_accuracy", "reporter_choice"), ("lr_macro_f1", "reporter_choice"),
             ("reporter_choice", "always_bug")]
    out["differences"] = {f"{a} minus {b}": {"accuracy": paired_difference(*acc[a], *acc[b]),
                                             "macro_f1": paired_difference(*f1[a], *f1[b])} for a, b in pairs}
    return out


def per_type(preds: dict[str, list[str]], truth: list[str]) -> dict:
    """Per-type support, and recall and precision of each method."""
    out = {}
    for lab in TYPES:
        row = {"support": sum(t == lab for t in truth)}
        for name, p in preds.items():
            tp = sum(a == lab and b == lab for a, b in zip(truth, p))
            npred = sum(b == lab for b in p)
            row[name] = {"recall": tp / row["support"] if row["support"] else None,
                         "precision": tp / npred if npred else None, "predicted": npred}
        out[lab] = row
    return out


def main() -> None:
    issues = load_issues()
    sets = load_sets("type")
    team = team_accounts(issues)
    subset = {sp: non_team(sets[sp], issues, team) for sp in ("train", "tune", "test")}
    tune = subset["tune"]
    truth = [x["truth"] for x in tune]

    vec, models = fit_type_models(issues, sets["train"])
    X = vec.transform([issue_text(issues[x["number"]]) for x in tune])
    majority = Counter(x["truth"] for x in sets["train"]).most_common(1)[0][0]
    choice = [reporter_choice(issues[x["number"]]["body"] or "") for x in tune]
    preds = {"always_bug": [majority] * len(tune),
             "reporter_choice": [c or majority for c in choice],
             **{n: list(m.predict(X)) for n, m in models.items()}}

    with_choice = [i for i, c in enumerate(choice) if c]
    out = {
        "team_proxy_accounts": len(team),
        "subset_sizes": {sp: {"all": len(sets[sp]), "non_team": len(subset[sp])} for sp in subset},
        "tune_label_counts": dict(Counter(truth).most_common()),
        "tune_reporter_choice": {
            "with_choice": len(with_choice),
            "choice_matches_truth": sum(choice[i] == truth[i] for i in with_choice),
            "choices": dict(Counter(c or "(none)" for c in choice).most_common())},
        "majority_train_type": majority,
        "tune": score(preds, truth),
        "tune_per_type": per_type(preds, truth),
    }
    write_json(OUT2 / "type_baselines_tune.json", out)
    for n, m in out["tune"]["methods"].items():
        a, f = m["accuracy"], m["macro_f1"]
        print(f"{n:16s} acc {a['point']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]  "
              f"macro-F1 {f['point']:.3f} [{f['ci95'][0]:.3f}, {f['ci95'][1]:.3f}]")
    print(out["subset_sizes"], out["tune_label_counts"], out["tune_reporter_choice"], sep="\n")


if __name__ == "__main__":
    sys.exit(main())
