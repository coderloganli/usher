"""Run BM25 and embedding retrieval for every test query.

Replay rule: the candidate pool of a query is every corpus issue created
strictly before it (the query itself is therefore excluded). BM25 statistics
(document frequencies, average length, N) are computed over that pool only.

BM25: lowercase `\\w+` tokens, k1 = 1.5, b = 0.75,
      idf = ln(1 + (N - df + 0.5) / (df + 0.5)), summed over query tokens
      (repeated query tokens count repeatedly).
Embedding: cosine similarity of L2-normalised embeddings (data/embeddings.npy).

The rank of the canonical is pessimistic on ties: rank = number of pool
issues whose score is >= the canonical's score.

Output: results/per_query.csv (one row per query and method)
Usage:  python scripts/retrieve.py
"""
from __future__ import annotations

import bisect
import csv
import json
import re
from collections import Counter

import numpy as np
from scipy import sparse

from common import DATA, RESULTS, issue_text, load_corpus

K1, B = 1.5, 0.75
TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def build_tf(texts: list[str]):
    vocab: dict[str, int] = {}
    indptr, indices, data = [0], [], []
    for t in texts:
        c = Counter(tokenize(t))
        for w, n in c.items():
            indices.append(vocab.setdefault(w, len(vocab)))
            data.append(n)
        indptr.append(len(indices))
    X = sparse.csr_matrix((np.array(data, np.float32), np.array(indices), np.array(indptr)),
                          shape=(len(texts), len(vocab)))
    return X.tocsc(), vocab, np.asarray(X.sum(axis=1)).ravel()


def bm25_scores(Xc, dl_cum, query_tokens, vocab, p) -> np.ndarray:
    """BM25 scores of rows [0, p) for the given query tokens."""
    scores = np.zeros(p, np.float64)
    avgdl = dl_cum[p] / p
    for w, qtf in Counter(query_tokens).items():
        j = vocab.get(w)
        if j is None:
            continue
        lo, hi = Xc.indptr[j], Xc.indptr[j + 1]
        rows = Xc.indices[lo:hi]
        keep = rows < p
        rows, tf = rows[keep], Xc.data[lo:hi][keep].astype(np.float64)
        df = len(rows)
        if df == 0:
            continue
        idf = np.log(1.0 + (p - df + 0.5) / (df + 0.5))
        dl = dl_cum[rows + 1] - dl_cum[rows]
        scores[rows] += qtf * idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
    return scores


def rank_and_top(scores: np.ndarray, target_idx: int, numbers: list[int], k: int = 10):
    rank = int((scores >= scores[target_idx]).sum())
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return rank, [numbers[i] for i in top]


def main() -> None:
    corpus = load_corpus()
    numbers = [r["number"] for r in corpus]
    created = [r["created_at"] for r in corpus]
    pos = {n: i for i, n in enumerate(numbers)}
    texts = [issue_text(r["title"], r["body"]) for r in corpus]
    tests = json.load(open(RESULTS / "testset.json"))["queries"]

    Xc, vocab, dl = build_tf(texts)
    dl_cum = np.concatenate([[0.0], np.cumsum(dl)])
    emb = np.load(DATA / "embeddings.npy")
    assert emb.shape[0] == len(corpus)

    out = []
    for q in tests:
        qi, ci = pos[q["query"]], pos[q["canonical"]]
        p = bisect.bisect_left(created, created[qi])  # pool = rows [0, p), created strictly before
        assert ci < p <= qi
        for method in ("bm25", "embedding"):
            if method == "bm25":
                s = bm25_scores(Xc, dl_cum, tokenize(texts[qi]), vocab, p)
            else:
                s = emb[:p] @ emb[qi]
            rank, top = rank_and_top(s, ci, numbers, 10)
            out.append({"query": q["query"], "canonical": q["canonical"], "method": method,
                        "pool_size": p, "rank": rank,
                        "hit@1": int(rank <= 1), "hit@5": int(rank <= 5), "hit@10": int(rank <= 10),
                        "canonical_score": round(float(s[ci]), 6),
                        "top10": " ".join(map(str, top))})
    with open(RESULTS / "per_query.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{len(tests)} queries x 2 methods -> results/per_query.csv")


if __name__ == "__main__":
    main()
