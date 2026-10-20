"""Type judgment: the LLM makes the final decision with reference material (stage 2).

Plan approved by the user on 2026-10-15 (EXPERIMENTS.md, entry 28); same structure as the module
judgment (scripts/stage2/module_llm.py).

Model: gpt-4o-mini-2024-07-18, Chat Completions (official `openai` package), temperature 0,
fixed seed, structured output {"reason": one sentence, "type": one of the 10 type labels}.

What the model sees for each issue:
- the issue's cleaned title + body (scripts/stage2/dedup_clean.clean_text), cut to 1,500 tokens;
- the reporter's own choice (bug or feature template, or the Type field of VS Code's issue
  reporter; "none" when absent). Cleaning removes the template markers, so it is given explicitly;
- reference 1: the probabilities of all 10 types from LR (accuracy), the stage-1 logistic
  regression (TF-IDF defaults, C = 1, no class weighting, fitted on the whole train set);
- reference 2: the 10 most similar train-period type issues created before the issue
  (bge-small cosine on cleaned text, stage-2 cache), with their titles, types and similarity;
- the definitions of the 10 types from the vscode wiki page "Issues-Triaging"
  (scripts/stage2/type_definitions.json). No tools.

Evaluation subset: type issues opened by non-team reporters (scripts/stage2/type_baselines.py).
Neighbours come from all train-period type issues, like the logistic regression's training data.

Protocol, fixed in advance:
  trial  30 random non-team train-period type issues: format and cost check only; the prompt is
         not changed because of it (the logistic regression is in-sample there).
  tune   all 890 non-team tune-period issues, once. Adopt only if (1) LLM accuracy minus LR
         (accuracy) accuracy has a 95% paired bootstrap interval (10,000 resamples, fixed seed)
         entirely above 0, and (2) LLM macro-F1 minus LR (macro-F1) macro-F1 has an interval that
         is not entirely below 0. Also reported: comparisons with the reporter's choice and the
         similar-issue vote, issues fixed and broken relative to LR (accuracy), per-type scores.
  test   run once after the method is frozen: all non-team test-period issues, scored the same way.

Cache: data/stage2/type_llm/<split>.jsonl (gitignored; reasons may paraphrase issue text).
Outputs: outputs/stage2/type_llm_{trial,tune,test}.json
Usage: .venv/bin/python scripts/stage2/type_llm.py {trial|tune|test}
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common2 import CACHE2, OUT2
from common import N_BOOT, ROOT, SEED, issue_text, load_issues, load_sets, write_json
from dedup_clean import clean_text
from module_llm import cost, knn_references, load_key
from stats import mean_boot_values, paired_difference, resample_indices, summary
from type_baselines import TYPES, fit_type_models, macro_f1, non_team, per_type, reporter_choice, team_accounts

LLM = "gpt-4o-mini-2024-07-18"
PRICE_PER_M = {"input": 0.15, "cached": 0.075, "output": 0.60}  # US dollars, checked 2026-10-10
ISSUE_TOKENS = 1500
N_TRIAL = 30
CACHE = CACHE2 / "type_llm"
DEFS = ROOT / "scripts" / "stage2" / "type_definitions.json"

SYSTEM = ("You triage GitHub issues for the microsoft/vscode repository. Choose the type label that the "
          "VS Code team would put on the issue. Answer with exactly one of the 10 type labels.")


# ---------------------------------------------------------------- references

def lr_references(issues: dict, vec, m, queries: list[dict]) -> dict[int, list[tuple[str, float]]]:
    """All type probabilities of LR (accuracy), most probable first."""
    P = m.predict_proba(vec.transform([issue_text(issues[x["number"]]) for x in queries]))
    return {x["number"]: [(str(m.classes_[i]), float(p[i])) for i in np.argsort(-p, kind="stable")]
            for x, p in zip(queries, P)}


def knn_vote(neigh: list[tuple[int, float]], truth_of: dict[int, str]) -> str | None:
    score = defaultdict(float)
    for n, s in neigh:
        score[truth_of[n]] += s
    return min(score, key=lambda t: (-score[t], t)) if score else None


def definitions_text() -> str:
    labels = json.load(open(DEFS, encoding="utf-8"))["labels"]
    lines = []
    for t in TYPES:
        d = labels[t]
        lines.append(f"- {t}: {d['type']}" + (f" (closing reason: {d['closing_reason']})" if "closing_reason" in d else ""))
    return "\n".join(lines)


def prompt(issue: dict, choice: str | None, lr: list, neigh: list, truth_of: dict, issues: dict,
           defs: str, enc) -> str:
    text = enc.decode(enc.encode(clean_text(issue))[:ISSUE_TOKENS])
    ref1 = "\n".join(f"- {t} ({p:.2f})" for t, p in lr)
    ref2 = "\n".join(f"- ({s:.2f}) \"{(issues[n]['title'] or '').strip()[:120]}\" -> {truth_of[n]}"
                     for n, s in neigh)
    return (f"Issue:\n<<<\n{text}\n>>>\n\n"
            f"The reporter's own choice when opening the issue: {choice or 'none'}. Reporters only choose "
            "between bug and feature-request, and the team often changes it.\n\n"
            "Reference 1. A word-based classifier trained on past labelled issues gives these probabilities. "
            f"It is often right but rarely predicts types other than bug and feature-request:\n{ref1}\n\n"
            "Reference 2. The most similar past issues (similarity in brackets) and the type labels the team "
            f"gave them:\n{ref2}\n\n"
            f"Type definitions from the VS Code triage guide:\n{defs}\n\n"
            "Use the references, the definitions and your own reading of the issue. First give a one-sentence "
            "reason, then the single best type label.")


# ---------------------------------------------------------------- run

def run(split: str, queries: list[dict], choice: dict, refs_lr: dict, refs_nn: dict, truth_of: dict,
        issues: dict) -> list[dict]:
    import tiktoken
    from openai import OpenAI
    load_key()
    client = OpenAI(max_retries=6)
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    defs = definitions_text()
    schema = {"type": "object", "additionalProperties": False, "required": ["reason", "type"],
              "properties": {"reason": {"type": "string"}, "type": {"type": "string", "enum": list(TYPES)}}}
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{split}.jsonl"
    done = {r["number"]: r for r in map(json.loads, open(path, encoding="utf-8"))} if path.exists() else {}
    todo = [q for q in queries if q["number"] not in done]

    def call(q: dict) -> dict:
        n = q["number"]
        r = client.chat.completions.create(
            model=LLM, temperature=0, seed=SEED, max_completion_tokens=150,
            response_format={"type": "json_schema", "json_schema": {"name": "type", "strict": True, "schema": schema}},
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt(issues[n], choice[n], refs_lr[n], refs_nn[n], truth_of,
                                                         issues, defs, enc)}])
        d = getattr(r.usage, "prompt_tokens_details", None)
        msg = r.choices[0].message
        ans = json.loads(msg.content) if msg.content else {"reason": "", "type": None}
        return {"number": n, "type": ans["type"], "reason": ans["reason"], "finish_reason": r.choices[0].finish_reason,
                "refusal": msg.refusal, "model": r.model, "system_fingerprint": r.system_fingerprint,
                "prompt_tokens": r.usage.prompt_tokens, "cached_tokens": (getattr(d, "cached_tokens", 0) or 0) if d else 0,
                "completion_tokens": r.usage.completion_tokens}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool, open(path, "a", encoding="utf-8") as f:
        for i, row in enumerate(pool.map(call, todo), 1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"{split}: {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    done = {r["number"]: r for r in map(json.loads, open(path, encoding="utf-8"))}
    return [done[q["number"]] for q in queries]


def scores(truth: list[str], preds: dict[str, list[str]]) -> dict:
    t = np.array(truth)
    idx = resample_indices(len(t), N_BOOT, SEED)
    acc, f1 = {}, {}
    for name, p in preds.items():
        p = np.array(p, dtype=object).astype(str)
        hit = (p == t).astype(float)
        acc[name] = (hit.mean(), mean_boot_values(hit, idx))
        f1[name] = (macro_f1(t, p), np.array([macro_f1(t[r], p[r]) for r in idx]))
    out = {"methods": {n: {"accuracy": summary(*acc[n]), "macro_f1": summary(*f1[n])} for n in preds},
           "differences": {}}
    for other in ("lr_accuracy", "lr_macro_f1", "reporter_choice", "knn_vote"):
        out["differences"][f"llm minus {other}"] = {"accuracy": paired_difference(*acc["llm"], *acc[other]),
                                                    "macro_f1": paired_difference(*f1["llm"], *f1[other])}
    a = out["differences"]["llm minus lr_accuracy"]["accuracy"]["ci95"]
    f = out["differences"]["llm minus lr_macro_f1"]["macro_f1"]["ci95"]
    out["rule"] = ("adopt only if LLM accuracy minus LR (accuracy) accuracy has a 95% interval entirely above 0, "
                   "and LLM macro-F1 minus LR (macro-F1) macro-F1 has an interval not entirely below 0")
    out["accuracy_condition"] = bool(a[0] > 0)
    out["macro_f1_condition"] = bool(f[1] >= 0)
    out["adopt"] = out["accuracy_condition"] and out["macro_f1_condition"]
    llm, lra = np.array(preds["llm"], dtype=object), np.array(preds["lr_accuracy"])
    out["fixed_vs_lr_accuracy"] = int(((llm == t) & (lra != t)).sum())
    out["broken_vs_lr_accuracy"] = int(((llm != t) & (lra == t)).sum())
    return out


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else ""
    if split not in ("trial", "tune", "test"):
        raise SystemExit("usage: type_llm.py {trial|tune|test}")
    issues = load_issues()
    sets = load_sets("type")
    team = team_accounts(issues)
    train = sets["train"]
    truth_of = {x["number"]: x["truth"] for x in train}
    if split == "trial":
        queries = random.Random(SEED).sample(non_team(train, issues, team), N_TRIAL)
    else:
        queries = non_team(sets[split], issues, team)
    choice = {q["number"]: reporter_choice(issues[q["number"]]["body"] or "") for q in queries}
    vec, models = fit_type_models(issues, train)
    refs_lr = lr_references(issues, vec, models["lr_accuracy"], queries)
    refs_nn = knn_references(issues, train, queries)
    rows = run(split, queries, choice, refs_lr, refs_nn, truth_of, issues)
    # module_llm.cost counts missing answers in the "module" field
    out = {"split": split, "n": len(queries), "model": LLM, "price_per_m_usd": PRICE_PER_M,
           "cost": cost([{**r, "module": r["type"]} for r in rows])}
    majority = Counter(truth_of.values()).most_common(1)[0][0]
    preds = {"llm": [r["type"] for r in rows],
             "lr_accuracy": [refs_lr[q["number"]][0][0] for q in queries],
             "reporter_choice": [choice[q["number"]] or majority for q in queries],
             "knn_vote": [knn_vote(refs_nn[q["number"]], truth_of) for q in queries]}
    if split == "trial":
        out["note"] = "format and cost check only; the logistic regression is in-sample on train issues, so no scores"
        out["valid_answers"] = sum(r["type"] in TYPES for r in rows)
        out["answers"] = dict(Counter(preds["llm"]).most_common())
    else:
        preds["lr_macro_f1"] = list(models["lr_macro_f1"].predict(
            vec.transform([issue_text(issues[q["number"]]) for q in queries])))
        truth = [q["truth"] for q in queries]
        out["scores"] = scores(truth, preds)
        out["per_type"] = per_type(preds, truth)
    out["per_issue"] = [{"number": q["number"], "truth": q["truth"], "reporter_choice_raw": choice[q["number"]],
                         **{k: v[i] for k, v in preds.items()}} for i, q in enumerate(queries)]
    write_json(OUT2 / f"type_llm_{split}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_issue", "per_type")}, indent=1))


if __name__ == "__main__":
    main()
