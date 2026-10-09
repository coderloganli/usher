"""Summarise the audit verdicts by group.

The verdicts in verdicts.csv were written by an AI model (Claude Opus 5.5) from the public
title and body of each pair, in a shuffled order with the group and dates hidden (issue
numbers stayed visible, and they rise over time). They are not human-verified.

Share of "same" = same / (same + different), with a 95% Wilson score interval; "cannot_tell"
is left out of the share and reported separately.

Output: analysis/duplicate_label_audit/summary.json
Usage:  python3 analysis/duplicate_label_audit/summary.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
Z = 1.959964


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return [centre - half, centre + half]


def main() -> None:
    with open(HERE / "verdicts.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {"reviewer": "AI model (Claude Opus 5.5), not human-verified", "groups": {}}
    for g in ("old", "recent"):
        c = Counter(r["verdict"] for r in rows if r["group"] == g)
        n = c["same"] + c["different"]
        out["groups"][g] = {"pairs": sum(c.values()), "counts": dict(c),
                            "share_same": c["same"] / n if n else None, "share_same_ci95": wilson(c["same"], n)}
    with open(HERE / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
