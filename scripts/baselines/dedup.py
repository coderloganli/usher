"""Duplicate baselines: BM25, embedding retrieval and their reciprocal rank fusion (RRF).

Plan sections 4.3 and 5. Neither retriever learns anything, so the train + tune duplicate
queries (2025-01..2026-06) are the development data. This script scores them only;
the test queries are handled by a later step after the method list is frozen.

Replay rule: the candidate pool of a query is every pulled issue (2023-01..2026-09)
created strictly before it; BM25 statistics (N, document frequencies, average length)
come from that pool only.

BM25 (Robertson & Zaragoza 2009): lowercase `\\w+` tokens, k1 = 1.2, b = 0.75.
  score(q, d) = sum over query tokens t of idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
  - repeated query tokens count repeatedly (the survey: qtf saturation "was unimportant,
    leading to a formula which was linear in qtf");
  - idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)), the non-negative form, so a token in more
    than half of the pool adds a small positive score instead of a negative one.
Embedding: BAAI/bge-small-en-v1.5 at a pinned revision, no instruction prefix, inputs
  truncated at the model's 512-token limit, L2-normalised vectors, cosine similarity.
RRF (Cormack, Clarke & Buettcher 2009): fused(d) = 1 / (60 + rank_bm25(d)) + 1 / (60 + rank_emb(d)),
  where each rank is the competition rank (1 + number of pool issues with a higher score).
Rank of the true original: pessimistic on ties, i.e. the number of pool issues whose score
  is greater than or equal to the original's score.

Output: outputs/baselines/duplicate_dev.json (per-query ranks and dev metrics)
Cache:  data/baselines/embeddings.npy, data/baselines/embeddings_meta.json
Usage:  .venv/bin/python scripts/baselines/dedup.py
"""
from __future__ import annotations

import bisect
import json
import re
import time
from collections import Counter

import numpy as np
from scipy import sparse

from common import CACHE, N_BOOT, OUT, SEED, issue_text, load_issues, load_sets, write_json
from stats import mean_boot_values, paired_difference, resample_indices, summary

MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
K1, B = 1.2, 0.75
RRF_K = 60
KS = (1, 5, 10)
METHODS = ("bm25", "embedding", "rrf")
TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class Corpus:
    """All pulled issues sorted by (created_at, number); a query's pool is a prefix of it."""

    def __init__(self, issues: dict[int, dict]):
        rows = sorted(issues.values(), key=lambda r: (r["created_at"], r["number"]))
        self.numbers = [r["number"] for r in rows]
        self.created = [r["created_at"] for r in rows]
        self.pos = {n: i for i, n in enumerate(self.numbers)}
        self.texts = [issue_text(r) for r in rows]

    def pool_size(self, number: int) -> int:
        """Number of corpus issues created strictly before this issue."""
        return bisect.bisect_left(self.created, self.created[self.pos[number]])


class BM25:
    def __init__(self, texts: list[str]):
        vocab: dict[str, int] = {}
        indptr, indices, data = [0], [], []
        for t in texts:
            for w, n in Counter(tokenize(t)).items():
                indices.append(vocab.setdefault(w, len(vocab)))
                data.append(n)
            indptr.append(len(indices))
        X = sparse.csr_matrix((np.array(data, np.float64), np.array(indices), np.array(indptr)),
                              shape=(len(texts), len(vocab)))
        self.vocab = vocab
        self.Xc = X.tocsc()
        self.Xc.sort_indices()  # rows ascending within each column, so a pool prefix is a slice
        self.dl_cum = np.concatenate([[0.0], np.cumsum(np.asarray(X.sum(axis=1)).ravel())])

    def scores(self, query: str, p: int) -> np.ndarray:
        """Scores of corpus rows [0, p) for this query."""
        s = np.zeros(p)
        avgdl = self.dl_cum[p] / p
        for w, qtf in Counter(tokenize(query)).items():
            j = self.vocab.get(w)
            if j is None:
                continue
            lo, hi = self.Xc.indptr[j], self.Xc.indptr[j + 1]
            hi = lo + int(np.searchsorted(self.Xc.indices[lo:hi], p))  # keep rows < p
            if hi == lo:
                continue
            rows, tf = self.Xc.indices[lo:hi], self.Xc.data[lo:hi]
            idf = np.log(1.0 + (p - (hi - lo) + 0.5) / ((hi - lo) + 0.5))
            dl = self.dl_cum[rows + 1] - self.dl_cum[rows]
            s[rows] += qtf * idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
        return s


def embed(corpus: Corpus) -> np.ndarray:
    """Embeddings of every corpus issue, cached under data/baselines/."""
    path, meta_path = CACHE / "embeddings.npy", CACHE / "embeddings_meta.json"
    if path.exists() and meta_path.exists():
        meta = json.load(open(meta_path))
        if (meta["model_id"], meta["revision"], meta["numbers_first_last"], meta["n"]) == \
                (MODEL_ID, MODEL_REVISION, [corpus.numbers[0], corpus.numbers[-1]], len(corpus.numbers)):
            return np.load(path)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device=device)
    t0 = time.time()
    emb = model.encode(corpus.texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    write_json(meta_path, {"model_id": MODEL_ID, "revision": MODEL_REVISION, "device": device,
                           "max_seq_length": model.max_seq_length, "n": len(corpus.numbers),
                           "dim": int(emb.shape[1]),
                           "numbers_first_last": [corpus.numbers[0], corpus.numbers[-1]],
                           "seconds": round(time.time() - t0, 1)})
    return emb


def competition_rank(s: np.ndarray) -> np.ndarray:
    """1 + number of entries with a strictly higher score."""
    order = np.sort(s)[::-1]
    return np.searchsorted(-order, -s, side="left") + 1


def pessimistic_rank(s: np.ndarray, target: int) -> int:
    return int((s >= s[target]).sum())


def top10(s: np.ndarray, numbers: list[int]) -> list[int]:
    top = np.argpartition(-s, 10)[:10]
    return [numbers[i] for i in top[np.argsort(-s[top], kind="stable")]]


def retrieve(queries: list[dict], corpus: Corpus, bm25: BM25, emb: np.ndarray) -> list[dict]:
    out = []
    for k, q in enumerate(queries):
        qi, ci = corpus.pos[q["number"]], corpus.pos[q["truth"]]
        p = corpus.pool_size(q["number"])
        assert ci < p <= qi, q
        s_bm = bm25.scores(corpus.texts[qi], p)
        s_em = emb[:p] @ emb[qi]
        s_rrf = 1.0 / (RRF_K + competition_rank(s_bm)) + 1.0 / (RRF_K + competition_rank(s_em))
        row = {"query": q["number"], "original": q["truth"], "pool_size": p}
        for m, s in (("bm25", s_bm), ("embedding", s_em), ("rrf", s_rrf)):
            row[f"{m}_rank"] = pessimistic_rank(s, ci)
            row[f"{m}_top10"] = top10(s, corpus.numbers)
        out.append(row)
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(queries)} queries", flush=True)
    return out


def metrics(rows: list[dict], seed: int) -> dict:
    """recall@k with bootstrap CIs; paired differences embedding - bm25 and rrf - better single."""
    idx = resample_indices(len(rows), N_BOOT, seed)
    hits, res = {}, {"n_queries": len(rows), "methods": {}}
    for m in METHODS:
        ranks = np.array([r[f"{m}_rank"] for r in rows])
        res["methods"][m] = {}
        for k in KS:
            h = (ranks <= k).astype(float)
            hits[m, k] = (h.mean(), mean_boot_values(h, idx))
            res["methods"][m][f"recall@{k}"] = summary(*hits[m, k])
    better = max(("bm25", "embedding"), key=lambda m: hits[m, 10][0])
    res["better_single_by_recall@10"] = better
    res["differences"] = {
        f"embedding_minus_bm25_recall@{k}": paired_difference(*hits["embedding", k], *hits["bm25", k]) for k in KS}
    res["differences"].update({
        f"rrf_minus_{better}_recall@{k}": paired_difference(*hits["rrf", k], *hits[better, k]) for k in KS})
    return res


def main() -> None:
    issues = load_issues()
    corpus = Corpus(issues)
    sets = load_sets("duplicate")
    dev = sorted(sets["train"] + sets["tune"], key=lambda x: (x["created_at"], x["number"]))
    t0 = time.time()
    bm25 = BM25(corpus.texts)
    print(f"BM25 index: {len(bm25.vocab)} tokens, {time.time() - t0:.0f}s", flush=True)
    emb = embed(corpus)
    rows = retrieve(dev, corpus, bm25, emb)
    res = metrics(rows, SEED)
    better = res["better_single_by_recall@10"]
    d = res["differences"][f"rrf_minus_{better}_recall@10"]
    res["decision_rule"] = ("keep RRF only if its recall@10 minus the better single method's "
                            "recall@10 has a 95% paired bootstrap interval entirely above 0")
    res["keep_rrf"] = bool(d["ci95"][0] > 0)
    write_json(OUT / "duplicate_dev.json", {"model_revision": MODEL_REVISION, "k1": K1, "b": B,
                                            "rrf_k": RRF_K, "metrics": res, "queries": rows})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
