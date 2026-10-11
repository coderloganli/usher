"""Feasibility pilot: can an LLM agent that searches the vscode code base judge an issue's module?

Agreed with the user on 2026-10-10. A pilot, not an evaluation: 20 tune-period module issues,
budget US$0.10 in total (the script stops new issues once the running cost reaches US$0.09).

Sample: the stage-1 module baseline (TF-IDF + one-vs-rest logistic regression, C = 10, balanced,
fitted on train) is refitted locally; from the tune set, 15 random issues whose LR top-1 is wrong
and 5 whose LR top-1 is right (seed fixed).

Two methods, same model (gpt-4o-mini-2024-07-18, Chat Completions, temperature 0, fixed seed):
  plain  the issue (title + raw body, cut to 1,500 tokens), the LR top 10 with probabilities and
         GitHub label descriptions, and the names of all 291 module labels; answer one label.
  agent  the same input plus three read-only tools over the vscode working tree checked out at
         the last main commit before 2026-04-01 (earlier than every tune issue): search_code
         (case-insensitive fixed string, ripgrep), read_file, list_dir. The .git directory is
         off limits (it holds later history). At most 8 tool calls; once the issue's cumulative
         input passes 25,000 tokens, or after 8 calls, the model must answer.

  agent2 added after the first run (user agreed 2026-10-10): the agent tools plus modules_for_path
         (the "code path -> module" table from build_path_map.py), an explicit step order, and no
         `answer` tool in the first 2 turns, so the model must search first. Same 20 issues, so
         this is a feasibility check, not an unbiased comparison.

Label descriptions: current GitHub label descriptions (205 of 291 module labels have one),
fetched 2026-10-10; they describe labels, not answers.

Outputs
  analysis/agent_module_pilot/sample.json              issue numbers, truth, LR top 10 (no text)
  analysis/agent_module_pilot/label_descriptions.json  module label -> GitHub description
  analysis/agent_module_pilot/summary.json             hits, tokens, cost, projection
  data/agent_pilot/{plain,agent}.jsonl                 cached responses and tool traces
                                                       (gitignored: they quote issue text)
Usage: .venv/bin/python analysis/agent_module_pilot/pilot.py {sample|plain|agent|agent2|summary}
"""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
from common import SEED, issue_text, load_issues, load_sets, write_json  # noqa: E402

CACHE = ROOT / "data" / "agent_pilot"
REPO = ROOT / "data" / "vscode_repo"
REPO_COMMIT = "9ed1e079300da92c7051094a05b4dbe136f9c23e"  # 2026-03-31T23:53:45Z
LLM = "gpt-4o-mini-2024-07-18"
PRICE_PER_M = {"input": 0.15, "cached": 0.075, "output": 0.60}  # US dollars, checked 2026-10-10
STOP_AT_USD = 0.09
ISSUE_TOKENS = 1500
MAX_TOOL_CALLS = 8
MAX_ISSUE_INPUT = 25_000
N_WRONG, N_RIGHT, TOP_K = 15, 5, 10

SYSTEM = ("You triage GitHub issues for the microsoft/vscode repository. Your job is to choose the feature-area "
          "(module) label that the VS Code team would put on the issue. Answer with exactly one label from the "
          "list of module labels.")
MIN_SEARCH_TURNS = 2
AGENT2_EXTRA = ("You can search the VS Code source code (as of the end of March 2026) with the tools. Work in this "
                "order: (1) pick the most specific identifiers in the issue: setting ids, command ids, exact UI text "
                "or error messages, feature names; (2) find them with search_code; (3) call modules_for_path on the "
                "most relevant file or directory, which tells you how past issues fixed there were labelled; "
                "(4) call `answer`. You have at most 8 tool calls.")
AGENT_EXTRA = ("You can search the VS Code source code (as of the end of March 2026) with the tools. Use them to "
               "find where the feature, setting, command or error message mentioned in the issue lives, and use the "
               "file locations to decide the module. You have at most 8 tool calls; then call `answer`.")


# ---------------------------------------------------------------- sample

def make_sample() -> None:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    issues = load_issues()
    sets = load_sets("module")
    tr, tu = sets["train"], sets["tune"]
    mlb = MultiLabelBinarizer()
    Ytr = mlb.fit_transform([x["truth"] for x in tr])
    classes = np.array(mlb.classes_)
    vec = TfidfVectorizer()
    Xtr = vec.fit_transform([issue_text(issues[x["number"]]) for x in tr])
    Xtu = vec.transform([issue_text(issues[x["number"]]) for x in tu])
    lr = OneVsRestClassifier(LogisticRegression(C=10.0, class_weight="balanced", max_iter=5000,
                                                random_state=SEED), n_jobs=-1).fit(Xtr, Ytr)
    P = lr.predict_proba(Xtu)
    rows = []
    for x, p in zip(tu, P):
        order = np.argsort(-p, kind="stable")[:TOP_K]
        top = [{"label": str(classes[i]), "prob": round(float(p[i]), 4)} for i in order]
        rows.append({"number": x["number"], "truth": x["truth"], "lr_top": top,
                     "lr_top1_hit": top[0]["label"] in x["truth"],
                     "lr_top10_hit": bool({t["label"] for t in top} & set(x["truth"]))})
    rng = random.Random(SEED)
    wrong = [r for r in rows if not r["lr_top1_hit"]]
    right = [r for r in rows if r["lr_top1_hit"]]
    sample = rng.sample(wrong, N_WRONG) + rng.sample(right, N_RIGHT)
    write_json(HERE / "sample.json", {
        "rule": f"tune set; {N_WRONG} random LR-top1-wrong + {N_RIGHT} random LR-top1-right; seed {SEED}",
        "tune_n": len(rows), "tune_lr_top1": sum(r["lr_top1_hit"] for r in rows) / len(rows),
        "tune_lr_top10": sum(r["lr_top10_hit"] for r in rows) / len(rows),
        "repo_commit": REPO_COMMIT, "issues": sample})

    labels = json.load(open(ROOT / "manifest" / "module_labels.json"))["module_labels"]
    gh = {r[0]: r[1] for r in csv.reader(open(CACHE / "gh_labels.tsv", encoding="utf-8"), delimiter="\t")
          if len(r) >= 2}
    write_json(HERE / "label_descriptions.json", {
        "source": "GitHub label descriptions of microsoft/vscode, fetched 2026-10-10",
        "labels": {l: gh.get(l, "").strip() for l in labels}})
    print(f"tune LR top1 {sum(r['lr_top1_hit'] for r in rows)}/{len(rows)}; sample {len(sample)}")


# ---------------------------------------------------------------- prompts and API

def load_key() -> None:
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()


def user_prompt(row: dict, issue: dict, desc: dict, enc) -> str:
    text = enc.decode(enc.encode(issue_text(issue))[:ISSUE_TOKENS])
    cands = "\n".join(f"- {c['label']} (p={c['prob']:.2f}){': ' + desc[c['label']] if desc.get(c['label']) else ''}"
                      for c in row["lr_top"])
    return (f"Issue:\n<<<\n{text}\n>>>\n\n"
            f"A word-based classifier suggests these modules (probability in brackets; it is often wrong):\n{cands}\n\n"
            f"All module labels: {', '.join(desc)}\n\n"
            "Choose the single best module label.")


def cost(usage: dict) -> float:
    cached = usage.get("cached", 0)
    return ((usage["prompt"] - cached) * PRICE_PER_M["input"] + cached * PRICE_PER_M["cached"]
            + usage["completion"] * PRICE_PER_M["output"]) / 1e6


def add_usage(total: dict, r) -> None:
    total["prompt"] += r.usage.prompt_tokens
    total["completion"] += r.usage.completion_tokens
    d = getattr(r.usage, "prompt_tokens_details", None)
    total["cached"] += (getattr(d, "cached_tokens", 0) or 0) if d else 0


def spent() -> float:
    total = 0.0
    for name in ("plain", "agent", "agent2"):
        p = CACHE / f"{name}.jsonl"
        if p.exists():
            total += sum(json.loads(l)["cost_usd"] for l in open(p, encoding="utf-8"))
    return total


def done(name: str) -> set[int]:
    p = CACHE / f"{name}.jsonl"
    return {json.loads(l)["number"] for l in open(p, encoding="utf-8")} if p.exists() else set()


# ---------------------------------------------------------------- tools (read-only, working tree only)

def safe_path(rel: str) -> Path | None:
    p = (REPO / rel.lstrip("/")).resolve()
    if not p.is_relative_to(REPO.resolve()) or ".git" in p.relative_to(REPO.resolve()).parts:
        return None
    return p


def search_code(text: str, path: str = "") -> str:
    base = safe_path(path or ".")
    if base is None or not base.exists():
        return "error: path not found"
    out = subprocess.run(["rg", "-n", "-i", "-F", "--max-count", "3", "--max-columns", "200",
                          "--glob", "!.git", "--", text, str(base)],
                         capture_output=True, text=True, timeout=60).stdout.splitlines()
    root = str(REPO.resolve()) + "/"
    lines = [l.replace(root, "") for l in out]
    head = f"{len(lines)} matching lines" + (" (first 30 shown)" if len(lines) > 30 else "")
    return head + "\n" + "\n".join(lines[:30])


def read_file(path: str, start_line: int = 1, num_lines: int = 60) -> str:
    p = safe_path(path)
    if p is None or not p.is_file():
        return "error: file not found"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, int(start_line))
    n = max(1, min(60, int(num_lines)))
    return "\n".join(f"{i}: {l[:200]}" for i, l in enumerate(lines[start - 1:start - 1 + n], start)) or "(empty)"


def list_dir(path: str = "") -> str:
    p = safe_path(path or ".")
    if p is None or not p.is_dir():
        return "error: directory not found"
    names = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir() if e.name != ".git")
    return "\n".join(names[:100]) + (f"\n... {len(names) - 100} more" if len(names) > 100 else "")


_PATH_MODULES: dict | None = None


def modules_for_path(path: str) -> str:
    """Modules of train-period issues whose fixes (before 2026-04-01) changed this path; falls back
    to the nearest parent directory that has data (build_path_map.py)."""
    global _PATH_MODULES
    if _PATH_MODULES is None:
        _PATH_MODULES = json.load(open(CACHE / "path_modules.json", encoding="utf-8"))
    parts = path.strip("/").split("/")
    for k in range(len(parts), 0, -1):
        key = "/".join(parts[:k])
        if key in _PATH_MODULES:
            top = list(_PATH_MODULES[key].items())[:8]
            return (f"Fixes of past issues that changed {key}{'' if k == len(parts) else ' (nearest parent with data)'} "
                    "were labelled: " + ", ".join(f"{m} ({c})" for m, c in top))
    return "no past fixes recorded for this path or its parents"


TOOL_FUNCS = {"search_code": search_code, "read_file": read_file, "list_dir": list_dir,
              "modules_for_path": modules_for_path}


def tool_specs(labels: list[str]) -> list[dict]:
    def fn(name, desc, props, required):
        return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
            "type": "object", "properties": props, "required": required, "additionalProperties": False}}}
    return [
        fn("search_code", "Case-insensitive fixed-string search over the VS Code source tree. Returns up to 30 "
           "matching lines as path:line:text.",
           {"text": {"type": "string", "description": "Exact text to look for, e.g. a setting id or error message."},
            "path": {"type": "string", "description": "Optional directory or file to search in, relative to the repo root."}},
           ["text"]),
        fn("read_file", "Read up to 60 lines of a file in the VS Code source tree.",
           {"path": {"type": "string"}, "start_line": {"type": "integer"}, "num_lines": {"type": "integer"}},
           ["path"]),
        fn("list_dir", "List a directory of the VS Code source tree (relative to the repo root; empty = root).",
           {"path": {"type": "string"}}, []),
        fn("modules_for_path", "Look up which module labels past issues had when their fixes changed this file or "
           "directory (counts of issues). Use it on the most relevant file you found.",
           {"path": {"type": "string", "description": "File or directory path relative to the repo root."}}, ["path"]),
        fn("answer", "Give the final module label.",
           {"module": {"type": "string", "enum": labels},
            "evidence": {"type": "string", "description": "One or two sentences: what you found and why it decides the module."}},
           ["module", "evidence"]),
    ]


# ---------------------------------------------------------------- runs

def setup():
    import tiktoken
    from openai import OpenAI
    load_key()
    sample = json.load(open(HERE / "sample.json"))["issues"]
    desc = json.load(open(HERE / "label_descriptions.json"))["labels"]
    return OpenAI(max_retries=6), tiktoken.encoding_for_model("gpt-4o-mini"), sample, desc, load_issues()


def run_plain() -> None:
    client, enc, sample, desc, issues = setup()
    labels = list(desc)
    schema = {"type": "object", "additionalProperties": False, "required": ["module"],
              "properties": {"module": {"type": "string", "enum": labels}}}
    CACHE.mkdir(parents=True, exist_ok=True)
    have = done("plain")
    with open(CACHE / "plain.jsonl", "a", encoding="utf-8") as f:
        for row in sample:
            if row["number"] in have:
                continue
            if spent() + 0.005 > STOP_AT_USD:
                print("budget stop"); break
            r = client.chat.completions.create(
                model=LLM, temperature=0, seed=SEED, max_completion_tokens=50,
                response_format={"type": "json_schema", "json_schema": {"name": "module", "strict": True, "schema": schema}},
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user_prompt(row, issues[row["number"]], desc, enc)}])
            usage = {"prompt": 0, "completion": 0, "cached": 0}
            add_usage(usage, r)
            module = json.loads(r.choices[0].message.content)["module"]
            rec = {"number": row["number"], "module": module, "hit": module in row["truth"],
                   "usage": usage, "cost_usd": cost(usage), "system_fingerprint": r.system_fingerprint}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(row["number"], module, row["truth"], rec["hit"], f"${spent():.4f}", flush=True)


def run_agent(v2: bool = False) -> None:
    """v1: tools search_code, read_file, list_dir; may answer at once.
    v2 (after the v1 pilot): adds modules_for_path, gives a step order, and the first
    MIN_SEARCH_TURNS turns cannot answer."""
    client, enc, sample, desc, issues = setup()
    labels = list(desc)
    name = "agent2" if v2 else "agent"
    tools = [t for t in tool_specs(labels) if v2 or t["function"]["name"] != "modules_for_path"]
    no_answer = [t for t in tools if t["function"]["name"] != "answer"]
    system = SYSTEM + " " + (AGENT2_EXTRA if v2 else AGENT_EXTRA)
    have = done(name)
    with open(CACHE / f"{name}.jsonl", "a", encoding="utf-8") as f:
        for row in sample:
            if row["number"] in have:
                continue
            if spent() + 0.006 > STOP_AT_USD:
                print("budget stop"); break
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": user_prompt(row, issues[row["number"]], desc, enc)}]
            usage = {"prompt": 0, "completion": 0, "cached": 0}
            trace, calls, final, forced = [], 0, None, False
            while final is None:
                must = calls >= MAX_TOOL_CALLS or usage["prompt"] >= MAX_ISSUE_INPUT
                forced = forced or must
                early = v2 and calls < MIN_SEARCH_TURNS and not must
                r = client.chat.completions.create(
                    model=LLM, temperature=0, seed=SEED, max_completion_tokens=300,
                    tools=no_answer if early else tools,
                    parallel_tool_calls=False, messages=msgs,
                    tool_choice={"type": "function", "function": {"name": "answer"}} if must else "required")
                add_usage(usage, r)
                m = r.choices[0].message
                msgs.append({"role": "assistant", "content": m.content, "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in (m.tool_calls or [])]})
                for c in m.tool_calls or []:
                    args = json.loads(c.function.arguments or "{}")
                    if c.function.name == "answer":
                        final = args
                        break
                    calls += 1
                    try:
                        out = TOOL_FUNCS[c.function.name](**args)
                    except Exception as e:  # bad arguments from the model
                        out = f"error: {e}"
                    trace.append({"tool": c.function.name, "args": args, "output_chars": len(out), "output_head": out[:300]})
                    msgs.append({"role": "tool", "tool_call_id": c.id, "content": out[:6000]})
            rec = {"number": row["number"], "module": final["module"], "evidence": final.get("evidence", ""),
                   "hit": final["module"] in row["truth"], "tool_calls": calls, "forced_answer": forced,
                   "usage": usage, "cost_usd": cost(usage), "trace": trace}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(row["number"], final["module"], row["truth"], rec["hit"], f"calls={calls}",
                  f"in={usage['prompt']}", f"${spent():.4f}", flush=True)


def summary() -> None:
    meta = json.load(open(HERE / "sample.json"))
    sample = {r["number"]: r for r in meta["issues"]}
    out = {"n_sample": len(sample), "repo_commit": REPO_COMMIT, "model": LLM, "price_per_m_usd": PRICE_PER_M,
           "tune_lr_top1": meta["tune_lr_top1"], "tune_lr_top10": meta["tune_lr_top10"], "methods": {}}
    for name in ("plain", "agent", "agent2"):
        p = CACHE / f"{name}.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(l) for l in open(p, encoding="utf-8")]
        grp = lambda ok: [r for r in recs if sample[r["number"]]["lr_top1_hit"] == ok]
        tok = sum(r["usage"]["prompt"] for r in recs)
        c = sum(r["cost_usd"] for r in recs)
        out["methods"][name] = {
            "issues_run": len(recs), "hits": sum(r["hit"] for r in recs),
            "hits_on_lr_wrong": f"{sum(r['hit'] for r in grp(False))}/{len(grp(False))}",
            "hits_on_lr_right": f"{sum(r['hit'] for r in grp(True))}/{len(grp(True))}",
            "answer_outside_lr_top10": sum(r["module"] not in {c_["label"] for c_ in sample[r["number"]]["lr_top"]}
                                           for r in recs),
            "input_tokens_per_issue": round(tok / len(recs)),
            "cached_tokens_per_issue": round(sum(r["usage"]["cached"] for r in recs) / len(recs)),
            "output_tokens_per_issue": round(sum(r["usage"]["completion"] for r in recs) / len(recs)),
            "cost_usd": round(c, 5), "cost_per_issue_usd": round(c / len(recs), 6),
            "projected_cost_full_tune_986_usd": round(c / len(recs) * 986, 2),
            "per_issue": [{"number": r["number"], "truth": sample[r["number"]]["truth"],
                           "lr_top1": sample[r["number"]]["lr_top"][0]["label"], "answer": r["module"], "hit": r["hit"],
                           **({"tool_calls": r["tool_calls"], "forced_answer": r["forced_answer"]} if name != "plain" else {})}
                          for r in recs]}
        if name != "plain":
            out["methods"][name]["tool_calls_per_issue"] = round(sum(r["tool_calls"] for r in recs) / len(recs), 2)
            out["methods"][name]["forced_answers"] = sum(r["forced_answer"] for r in recs)
    out["total_cost_usd"] = round(spent(), 5)
    write_json(HERE / "summary.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "methods"}, indent=1))
    for k, v in out["methods"].items():
        print(k, {a: b for a, b in v.items() if a != "per_issue"})


if __name__ == "__main__":
    {"sample": make_sample, "plain": run_plain, "agent": run_agent, "agent2": lambda: run_agent(v2=True),
     "summary": summary}[sys.argv[1]]()
