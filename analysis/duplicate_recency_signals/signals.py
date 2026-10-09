"""Time and popularity signals in duplicate detection, and a metadata-reranking prototype.

All on development queries, embedding retrieval on cleaned text
(outputs/stage2/duplicate_clean_dev.json, data/stage2/embeddings_clean.npy), sibling rule.
1. Who creates duplicate links (data/duplicate_events.json).
2. Queries grouped by retrieval outcome: how much older the original is, how many links
   already point at it, how long until the query was linked.
3. Top-100 candidates of the train-period queries: share that are correct (original or a
   sibling linked before the query) by candidate age and by links already pointing at it.
4. Prototype reranker: logistic regression on cosine, log(1 + age days), log(1 + links
   before the query), 1/rank; fitted on train-period queries, scored on tune-period queries,
   overall and by how much older the original is. Not adopted as a method (2026-10-09).

Output: analysis/duplicate_recency_signals/signals.json
Usage:  .venv/bin/python analysis/duplicate_recency_signals/signals.py
"""
from __future__ import annotations

import bisect
import collections
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "stage2"))
from common2 import CACHE2, OUT2, duplicate_links, found_at  # noqa: E402
from common import DATA, N_BOOT, SEED, load_issues, load_sets  # noqa: E402
from dedup import Corpus  # noqa: E402
from stats import mean_boot_values, paired_difference, resample_indices, summary  # noqa: E402

K = 100
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
days = lambda a, b: (ts(a) - ts(b)) / 86400


def makers(links: dict) -> dict:
    events = json.load(open(DATA / "duplicate_events.json"))
    found = []
    for n, (o, t) in links.items():
        e = next(e for e in events[str(n)] if e["created_at"] == t and e["canonical"] == o)
        found.append(e)
    return {"links": len(found), "actor_type": dict(collections.Counter(e["actor_type"] for e in found)),
            "event": dict(collections.Counter(e["event"] for e in found)),
            "distinct_actors": len({e["actor"] for e in found}),
            "top_actor_links": collections.Counter(e["actor"] for e in found).most_common(1)[0][1]}


def outcome_groups(issues: dict, links: dict) -> dict:
    hub = collections.Counter(o for o, _ in links.values())
    rows = json.load(open(OUT2 / "duplicate_clean_dev.json"))["queries"]
    groups = {"found_top10": [], "rank_11_100": [], "rank_over_100": []}
    for r in rows:
        f = found_at(10, r["query"], r["original"], r["embedding_rank"], r["embedding_top10"], links,
                     issues[r["query"]]["created_at"])
        groups["found_top10" if f else "rank_11_100" if r["embedding_rank"] <= 100 else "rank_over_100"].append(r)
    out = {}
    for g, rs in groups.items():
        age = [days(issues[r["query"]]["created_at"], issues[r["original"]]["created_at"]) for r in rs]
        lag = [days(links[r["query"]][1], issues[r["query"]]["created_at"]) for r in rs if r["query"] in links]
        hubs = [hub[r["original"]] for r in rs]
        out[g] = {"queries": len(rs), "original_age_days_median": statistics.median(int(a) for a in age),
                  "original_within_7_days": sum(int(a) <= 7 for a in age) / len(rs),
                  "original_with_5_plus_links": sum(h >= 5 for h in hubs) / len(rs),
                  "days_until_linked_median": statistics.median(lag)}
    return out


def candidates(issues, links, corpus, emb, inbound, q):
    qi, p, qt = corpus.pos[q["number"]], corpus.pool_size(q["number"]), issues[q["number"]]["created_at"]
    s = emb[:p] @ emb[qi]
    top = np.argpartition(-s, K)[:K]
    top = top[np.argsort(-s[top], kind="stable")]
    rows = []
    for rank, i in enumerate(top, 1):
        c = corpus.numbers[i]
        sib = c in links and links[c][0] == q["truth"] and links[c][1] < qt
        rows.append({"rank": rank, "cos": float(s[i]), "age": days(qt, issues[c]["created_at"]),
                     "hub": bisect.bisect_left(inbound.get(c, []), qt), "pos": int(c == q["truth"] or sib)})
    return rows


def main() -> None:
    issues = load_issues()
    links = duplicate_links(issues)
    corpus = Corpus(issues)
    emb = np.load(CACHE2 / "embeddings_clean.npy")
    inbound = collections.defaultdict(list)
    for o, t in links.values():
        inbound[o].append(t)
    for o in inbound:
        inbound[o].sort()
    sets = load_sets("duplicate")
    data = {sp: [candidates(issues, links, corpus, emb, inbound, q) for q in sets[sp]] for sp in ("train", "tune")}

    pool = [r for rs in data["train"] for r in rs]
    total = sum(r["pos"] for r in pool)
    buckets = {}
    for name, key, edges in (("candidate_age_days", "age", [0, 7, 30, 180, math.inf]),
                             ("links_before_query", "hub", [0, 1, 2, 5, math.inf])):
        buckets[name] = []
        for a, b in zip(edges, edges[1:]):
            g = [r for r in pool if a <= r[key] < b]
            pos = sum(r["pos"] for r in g)
            buckets[name].append({"from": a, "to": None if b == math.inf else b, "candidates": len(g),
                                  "correct": pos, "correct_share_within_group": pos / len(g),
                                  "share_of_all_correct": pos / total})

    X = lambda r: [r["cos"], math.log1p(r["age"]), math.log1p(r["hub"]), 1.0 / r["rank"]]
    model = LogisticRegression(max_iter=1000).fit([X(r) for rs in data["train"] for r in rs],
                                                  [r["pos"] for rs in data["train"] for r in rs])
    hit10 = lambda rs, sc: float(any(rs[i]["pos"] for i in sorted(range(len(rs)), key=lambda i: -sc[i])[:10]))
    base = np.array([hit10(rs, [r["cos"] for r in rs]) for rs in data["tune"]])
    rer = np.array([hit10(rs, list(model.decision_function([X(r) for r in rs]))) for rs in data["tune"]])
    idx = resample_indices(len(base), N_BOOT, SEED)
    ages = np.array([days(issues[q["number"]]["created_at"], issues[q["truth"]]["created_at"]) for q in sets["tune"]])
    by_age = {}
    for name, m in (("original_up_to_7_days_older", ages <= 7), ("7_to_30_days", (ages > 7) & (ages <= 30)),
                    ("over_30_days", ages > 30)):
        by_age[name] = {"queries": int(m.sum()), "embedding": float(base[m].mean()), "reranked": float(rer[m].mean()),
                        "lost": int(((base == 1) & (rer == 0) & m).sum()), "gained": int(((base == 0) & (rer == 1) & m).sum())}
    out = {
        "link_makers": makers(links),
        "outcome_groups": outcome_groups(issues, links),
        "train_candidates": {"queries": len(data["train"]), "candidates": len(pool), "correct": total, **buckets},
        "rerank_prototype": {
            "coefficients": dict(zip(["cos", "log1p_age", "log1p_links", "inverse_rank"], model.coef_[0].tolist())),
            "tune_queries": len(base),
            "embedding_recall@10": summary(base.mean(), mean_boot_values(base, idx)),
            "reranked_recall@10": summary(rer.mean(), mean_boot_values(rer, idx)),
            "difference": paired_difference(rer.mean(), mean_boot_values(rer, idx), base.mean(), mean_boot_values(base, idx)),
            "ceiling_correct_in_top100": float(np.mean([any(r["pos"] for r in rs) for rs in data["tune"]])),
            "by_original_age": by_age},
    }
    with open(HERE / "signals.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
