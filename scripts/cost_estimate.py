"""Estimate per-issue LLM cost (gpt-4o-mini) for the three judgments.

Counts real tokens with the gpt-4o-mini tokenizer (tiktoken, o200k_base) on
non-bot issues created 2026-07-01..2026-09-30. Prompt templates are not
designed yet, so each prompt is modelled as fixed instructions plus the real
text it would contain (assumptions in ASSUMPTIONS below).

Inputs (from the PoC, read-only):
  ../poc/dedup_experiment/data/issues_text.jsonl.gz
  ../poc/feasibility/data/d1_issues.jsonl.gz
  ../poc/feasibility/data/classifier.json
Output: results/cost_estimate.json

Usage: .venv/bin/python scripts/cost_estimate.py
"""
from __future__ import annotations

import gzip
import json
import random
import statistics
import sys
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
POC = ROOT.parent / "poc"
sys.path.insert(0, str(ROOT / "scripts" / "from_poc" / "dedup"))
from common import is_bot_login, issue_text  # noqa: E402

# gpt-4o-mini prices per 1M tokens, from https://developers.openai.com/api/docs/pricing
# (checked 2026-10-04). Batch API is 50% of standard.
PRICE_IN = 0.15
PRICE_OUT = 0.60

ASSUMPTIONS = {
    "instruction_tokens": 300,
    "type": {"output_tokens": 50},
    "assignee": {"similar_issues": 5, "similar_issue_chars": 500, "output_tokens": 50,
                 "ownership_table": "full classifier.json"},
    "dedup": {"candidates": 20, "candidate_chars": 1000, "output_tokens": 200},
}
TEST_START, TEST_END = "2026-07-01", "2026-10-01"
SEED = 20261004


def main() -> None:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    ntok = lambda s: len(enc.encode(s, disallowed_special=()))

    authors = {}
    with gzip.open(POC / "feasibility/data/d1_issues.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            authors[int(r["number"])] = r.get("author")

    texts = []
    with gzip.open(POC / "dedup_experiment/data/issues_text.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if TEST_START <= r["created_at"] < TEST_END and not is_bot_login(authors.get(int(r["number"]))):
                texts.append(issue_text(r["title"], r["body"] or ""))

    issue_tok = [ntok(t) for t in texts]
    short_tok = {c: [ntok(t[:c]) for t in texts] for c in (500, 1000)}
    owner_tok = ntok((POC / "feasibility/data/classifier.json").read_text())

    rng = random.Random(SEED)

    def sample_mean(xs, k):  # mean token count of k random other issues
        return statistics.mean(xs[i] for i in rng.sample(range(len(xs)), k))

    ins = ASSUMPTIONS["instruction_tokens"]
    a, d = ASSUMPTIONS["assignee"], ASSUMPTIONS["dedup"]
    per_issue = {"type": [], "assignee": [], "dedup": []}
    for t in issue_tok:
        per_issue["type"].append((ins + t, ASSUMPTIONS["type"]["output_tokens"]))
        per_issue["assignee"].append((ins + t + owner_tok + a["similar_issues"] * sample_mean(short_tok[500], a["similar_issues"]),
                                      a["output_tokens"]))
        per_issue["dedup"].append((ins + t + d["candidates"] * sample_mean(short_tok[1000], d["candidates"]),
                                   d["output_tokens"]))

    out = {"model": "gpt-4o-mini", "tokenizer": enc.name, "price_per_1M": {"input": PRICE_IN, "output": PRICE_OUT},
           "assumptions": ASSUMPTIONS, "n_issues": len(texts),
           "issue_tokens": {"mean": statistics.mean(issue_tok), "median": statistics.median(issue_tok),
                            "p90": sorted(issue_tok)[int(0.9 * len(issue_tok))]},
           "ownership_table_tokens": owner_tok, "per_issue": {}}
    for k, rows in per_issue.items():
        tin = statistics.mean(r[0] for r in rows)
        tout = statistics.mean(r[1] for r in rows)
        usd = (tin * PRICE_IN + tout * PRICE_OUT) / 1e6
        out["per_issue"][k] = {"input_tokens": round(tin), "output_tokens": round(tout),
                               "usd_standard": usd, "usd_batch": usd / 2}

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/cost_estimate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
