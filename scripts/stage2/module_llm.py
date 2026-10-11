"""Module judgment: the LLM makes the final decision with reference material (stage 2).

Plan agreed with the user on 2026-10-10, after the agent pilot (analysis/agent_module_pilot/),
the code-evidence analysis (analysis/module_repo_evidence/) and the similar-issue vote analysis
(analysis/module_knn/: vote top-1 59.1% vs LR 57.1% on tune, but only one of the two right on
256 issues, so a perfect chooser would reach 71.1%).

Model: gpt-4o-mini-2024-07-18, Chat Completions (official `openai` package), temperature 0,
fixed seed, structured output {"reason": one sentence, "module": one of the 291 module labels}.

What the model sees for each issue:
- the issue's cleaned title + body (scripts/stage2/dedup_clean.clean_text), cut to 1,500 tokens;
- reference 1: the stage-1 module LR's top 10 with probabilities (TF-IDF defaults, one-vs-rest,
  C = 10, balanced, fitted on train);
- reference 2: the 10 most similar train-period module issues created before the issue
  (bge-small cosine on cleaned text, stage-2 cache) with their titles, labels and similarity;
- GitHub descriptions of every candidate module from the two references
  (scripts/stage2/module_label_descriptions.json, fetched 2026-10-10), and the names of all
  291 module labels. No code-search tools.

Protocol, fixed in advance:
  trial  30 random train-period module issues: format and cost check only; the prompt is not
         changed because of it (LR is in-sample there, so its scores mean nothing).
  tune   all 986 tune-period issues, once. Adopt only if LLM top-1 minus LR top-1 has a 95%
         paired bootstrap interval (10,000 resamples, fixed seed) entirely above 0. Also reported:
         the comparison with the similar-issue vote, issues fixed and broken relative to LR, and
         groups by how many train issues carry the true module (1-19, 20-99, 100+).
  test   run once after the method was frozen (user approval 2026-10-11): all 687 test-period
         issues, same references (LR and similar issues from the train period only, as in the
         stage-1 final model); scored the same way as tune. The adoption rule is applied on tune only.

Cache: data/stage2/module_llm/<split>.jsonl (gitignored; reasons may paraphrase issue text).
Outputs: outputs/stage2/module_llm_{trial,tune,test}.json
Usage: .venv/bin/python scripts/stage2/module_llm.py {trial|tune|test}
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common2 import CACHE2, OUT2
from common import N_BOOT, ROOT, SEED, issue_text, load_issues, load_sets, write_json
from dedup_clean import clean_text
from stats import mean_boot_values, paired_difference, resample_indices, summary

LLM = "gpt-4o-mini-2024-07-18"
PRICE_PER_M = {"input": 0.15, "cached": 0.075, "output": 0.60}  # US dollars, checked 2026-10-10
ISSUE_TOKENS = 1500
K_LR, K_NN = 10, 10
N_TRIAL = 30
CACHE = CACHE2 / "module_llm"
EMB = CACHE2 / "embeddings_clean.npy"
DESC = ROOT / "scripts" / "stage2" / "module_label_descriptions.json"

SYSTEM = ("You triage GitHub issues for the microsoft/vscode repository. Choose the feature-area (module) label "
          "that the VS Code team would put on the issue. Answer with exactly one label from the list of module labels.")


def load_key() -> None:
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()


# ---------------------------------------------------------------- references

def lr_references(issues: dict, train: list[dict], queries: list[dict]) -> dict[int, list[tuple[str, float]]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform([x["truth"] for x in train])
    classes = np.array(mlb.classes_)
    vec = TfidfVectorizer()
    X = vec.fit_transform([issue_text(issues[x["number"]]) for x in train])
    lr = OneVsRestClassifier(LogisticRegression(C=10.0, class_weight="balanced", max_iter=5000,
                                                random_state=SEED), n_jobs=-1).fit(X, Y)
    P = lr.predict_proba(vec.transform([issue_text(issues[x["number"]]) for x in queries]))
    return {x["number"]: [(str(classes[i]), float(p[i])) for i in np.argsort(-p, kind="stable")[:K_LR]]
            for x, p in zip(queries, P)}


def knn_references(issues: dict, train: list[dict], queries: list[dict]) -> dict[int, list[tuple[int, float]]]:
    """The K_NN most similar train issues created strictly before each query."""
    order = [r["number"] for r in sorted(issues.values(), key=lambda r: (r["created_at"], r["number"]))]
    pos = {n: i for i, n in enumerate(order)}
    emb = np.load(EMB)
    assert emb.shape[0] == len(order)
    Etr = emb[[pos[x["number"]] for x in train]]
    created = np.array([x["created_at"] for x in train])
    out = {}
    for q in queries:
        s = Etr @ emb[pos[q["number"]]]
        s[created >= q["created_at"]] = -np.inf
        nb = np.argsort(-s, kind="stable")[:K_NN]
        out[q["number"]] = [(train[j]["number"], float(s[j])) for j in nb]
    return out


def knn_vote(neigh: list[tuple[int, float]], truth_of: dict[int, list[str]]) -> list[str]:
    score = defaultdict(float)
    for n, s in neigh:
        for m in truth_of[n]:
            score[m] += s
    return [m for m, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


def prompt(issue: dict, lr: list, neigh: list, truth_of: dict, issues: dict, desc: dict, enc) -> str:
    text = enc.decode(enc.encode(clean_text(issue))[:ISSUE_TOKENS])
    ref1 = "\n".join(f"- {m} ({p:.2f})" for m, p in lr)
    ref2 = "\n".join(f"- ({s:.2f}) \"{(issues[n]['title'] or '').strip()[:120]}\" -> {', '.join(truth_of[n])}"
                     for n, s in neigh)
    cands = sorted({m for m, _ in lr} | {m for n, _ in neigh for m in truth_of[n]})
    descs = "\n".join(f"- {m}: {desc[m]}" for m in cands if desc.get(m))
    return (f"Issue:\n<<<\n{text}\n>>>\n\n"
            "Reference 1. A word-based classifier trained on past labelled issues suggests these modules "
            f"(probability in brackets). It is often right but not always:\n{ref1}\n\n"
            "Reference 2. The most similar past issues (similarity in brackets) and the module labels the team "
            f"gave them:\n{ref2}\n\n"
            f"Descriptions of the candidate modules:\n{descs}\n\n"
            f"All module labels: {', '.join(desc)}\n\n"
            "Use both references and your own reading of the issue. First give a one-sentence reason, then the "
            "single best module label.")


# ---------------------------------------------------------------- run

def run(split: str, queries: list[dict], refs_lr: dict, refs_nn: dict, truth_of: dict, issues: dict) -> list[dict]:
    import tiktoken
    from openai import OpenAI
    load_key()
    client = OpenAI(max_retries=6)
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    desc = json.load(open(DESC, encoding="utf-8"))["labels"]
    schema = {"type": "object", "additionalProperties": False, "required": ["reason", "module"],
              "properties": {"reason": {"type": "string"}, "module": {"type": "string", "enum": list(desc)}}}
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{split}.jsonl"
    done = {r["number"]: r for r in map(json.loads, open(path, encoding="utf-8"))} if path.exists() else {}
    todo = [q for q in queries if q["number"] not in done]

    def call(q: dict) -> dict:
        n = q["number"]
        r = client.chat.completions.create(
            model=LLM, temperature=0, seed=SEED, max_completion_tokens=150,
            response_format={"type": "json_schema", "json_schema": {"name": "module", "strict": True, "schema": schema}},
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt(issues[n], refs_lr[n], refs_nn[n], truth_of, issues, desc, enc)}])
        d = getattr(r.usage, "prompt_tokens_details", None)
        msg = r.choices[0].message
        ans = json.loads(msg.content) if msg.content else {"reason": "", "module": None}
        return {"number": n, "module": ans["module"], "reason": ans["reason"], "finish_reason": r.choices[0].finish_reason,
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


def cost(rows: list[dict]) -> dict:
    pt = sum(r["prompt_tokens"] for r in rows)
    ct = sum(r["cached_tokens"] for r in rows)
    out = sum(r["completion_tokens"] for r in rows)
    usd = ((pt - ct) * PRICE_PER_M["input"] + ct * PRICE_PER_M["cached"] + out * PRICE_PER_M["output"]) / 1e6
    return {"calls": len(rows), "prompt_tokens": pt, "cached_tokens": ct, "completion_tokens": out,
            "cost_usd": round(usd, 4), "cost_per_issue_usd": round(usd / len(rows), 6),
            "finish_reason_not_stop": sum(r["finish_reason"] != "stop" for r in rows),
            "refusals": sum(bool(r["refusal"]) for r in rows), "no_answer": sum(r["module"] is None for r in rows)}


def scores(queries, rows, refs_lr, refs_nn, truth_of, freq) -> dict:
    hit = lambda t, m: float(m in t)
    llm = np.array([hit(q["truth"], r["module"]) for q, r in zip(queries, rows)])
    lr = np.array([hit(q["truth"], refs_lr[q["number"]][0][0]) for q in queries])
    nn = np.array([hit(q["truth"], (knn_vote(refs_nn[q["number"]], truth_of) or [None])[0]) for q in queries])
    groups = np.array(["0" if f == 0 else "1-19" if f < 20 else "20-99" if f < 100 else "100+"
                       for f in (max(freq[m] for m in q["truth"]) for q in queries)])
    outside = sum(r["module"] not in {m for m, _ in refs_lr[q["number"]]} | set(knn_vote(refs_nn[q["number"]], truth_of))
                  for q, r in zip(queries, rows))

    def block(mask):
        idx = resample_indices(int(mask.sum()), N_BOOT, SEED)
        b = {k: mean_boot_values(v[mask], idx) for k, v in (("llm", llm), ("lr", lr), ("knn", nn))}
        p = {k: float(v[mask].mean()) for k, v in (("llm", llm), ("lr", lr), ("knn", nn))}
        return {"n": int(mask.sum()),
                "llm_top1": summary(p["llm"], b["llm"]), "lr_top1": summary(p["lr"], b["lr"]),
                "knn_vote_top1": summary(p["knn"], b["knn"]),
                "llm_minus_lr": paired_difference(p["llm"], b["llm"], p["lr"], b["lr"]),
                "llm_minus_knn_vote": paired_difference(p["llm"], b["llm"], p["knn"], b["knn"]),
                "fixed_vs_lr": int(((llm == 1) & (lr == 0) & mask).sum()),
                "broken_vs_lr": int(((llm == 0) & (lr == 1) & mask).sum())}

    allm = np.ones(len(queries), bool)
    out = {"all": block(allm), "answers_outside_both_references": int(outside),
           "by_train_frequency_of_true_module": {g: block(groups == g) for g in ("0", "1-19", "20-99", "100+")
                                                 if (groups == g).any()}}
    d = out["all"]["llm_minus_lr"]
    out["rule"] = "adopt only if LLM top-1 minus LR top-1 has a 95% paired bootstrap interval entirely above 0"
    out["adopt"] = bool(d["ci95"][0] > 0)
    return out


def main() -> None:
    split = sys.argv[1]
    issues = load_issues()
    sets = load_sets("module")
    train = sets["train"]
    truth_of = {x["number"]: x["truth"] for x in train}
    freq = Counter(m for x in train for m in x["truth"])
    if split == "trial":
        queries = random.Random(SEED).sample(train, N_TRIAL)
    elif split in ("tune", "test"):
        queries = sets[split]
    else:
        raise SystemExit("usage: module_llm.py {trial|tune|test}")
    refs_lr = lr_references(issues, train, queries)
    refs_nn = knn_references(issues, train, queries)
    rows = run(split, queries, refs_lr, refs_nn, truth_of, issues)
    out = {"split": split, "n": len(queries), "model": LLM, "price_per_m_usd": PRICE_PER_M, "cost": cost(rows)}
    if split != "trial":
        out["scores"] = scores(queries, rows, refs_lr, refs_nn, truth_of, freq)
    else:
        out["note"] = "format and cost check only; LR is in-sample on train issues, so no scores are reported"
        out["valid_answers"] = sum(r["module"] is not None for r in rows)
    out["per_issue"] = [{"number": q["number"], "truth": q["truth"], "lr_top1": refs_lr[q["number"]][0][0],
                         "knn_top1": (knn_vote(refs_nn[q["number"]], truth_of) or [None])[0], "llm": r["module"]}
                        for q, r in zip(queries, rows)]
    write_json(OUT2 / f"module_llm_{split}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "per_issue"}, indent=1))


if __name__ == "__main__":
    main()
