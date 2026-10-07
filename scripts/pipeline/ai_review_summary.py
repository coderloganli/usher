"""A9 (revised 2026-10-07): summarize the AI review of the type and duplicate ground truth.

The verdicts themselves are in review/ai_review_type.csv and review/ai_review_duplicate.csv
(columns: number, url, truth, verdict, reason). They were written by an AI model
(Claude Opus 5.5) from the public issue content, one per sampled issue in
review/review_sample.json, and are NOT verified by a person. This script only counts them.

accuracy = 对 / (对 + 错); 无法判断 is excluded from the denominator and reported as a share
of all reviewed items. The 95% interval is the Wilson score interval.

Output: review/ai_review_summary.json
Usage : .venv/bin/python scripts/pipeline/ai_review_summary.py
"""
from __future__ import annotations

import csv
import json
import math

from common import REVIEW, write_json

JUDGMENTS = ["type", "duplicate"]
VERDICTS = ["对", "错", "无法判断"]
STANDARD = 0.85  # spec A4
Z = 1.959963984540054  # two-sided 95%


def wilson(k: int, n: int, z: float = Z) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def main() -> None:
    sample = json.load(open(REVIEW / "review_sample.json"))["sample"]
    out = {"reviewer": "AI model (Claude Opus 5.5), not verified by a person",
           "accuracy_definition": "对 / (对 + 错); 95% Wilson score interval",
           "a4_standard": STANDARD, "judgments": {}}
    for j in JUDGMENTS:
        with open(REVIEW / f"ai_review_{j}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        expected = {it["number"] for it in sample[j]}
        got = [int(r["number"]) for r in rows]
        assert sorted(got) == sorted(expected), f"{j}: CSV does not match review_sample.json"
        bad = [r["verdict"] for r in rows if r["verdict"] not in VERDICTS]
        assert not bad, f"{j}: unknown verdicts {bad}"
        c = {v: sum(r["verdict"] == v for r in rows) for v in VERDICTS}
        k, n = c["对"], c["对"] + c["错"]
        lo, hi = wilson(k, n)
        out["judgments"][j] = {
            "reviewed": len(rows), "counts": c,
            "accuracy": k / n if n else None, "ci95_low": lo, "ci95_high": hi,
            "cannot_judge_share": c["无法判断"] / len(rows),
            "meets_a4_point_estimate": (k / n >= STANDARD) if n else None,
            "ci95_low_meets_a4": (lo >= STANDARD) if n else None,
            "wrong": [int(r["number"]) for r in rows if r["verdict"] == "错"],
            "cannot_judge": [int(r["number"]) for r in rows if r["verdict"] == "无法判断"],
        }
    write_json(REVIEW / "ai_review_summary.json", out)
    for j, v in out["judgments"].items():
        print(f"{j}: {v['counts']} accuracy={v['accuracy']:.3f} CI=[{v['ci95_low']:.3f}, {v['ci95_high']:.3f}]")


if __name__ == "__main__":
    main()
