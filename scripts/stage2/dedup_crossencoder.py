"""Rerank the top 50 of embedding retrieval with a local cross-encoder (no LLM).

Candidates: embedding retrieval on cleaned text (scripts/stage2/dedup_clean.py), top 50 per
query. A cross-encoder reads the query and one candidate together and gives the pair a score;
the 50 candidates are re-sorted by that score (ties keep the embedding order).

Models (chosen 2026-10-09), pinned revisions:
- cross-encoder/quora-distilroberta-base: trained on Quora duplicate questions, 512-token
  pair limit; each side is cut to 254 tokens of its own tokenizer.
- Alibaba-NLP/gte-reranker-modernbert-base: relevance reranker, 8,192-token limit; each side is
  cut to 1,024 tokens.

Protocol fixed in advance:
1. Model choice on the train-period development queries (2025-01..2026-03): higher recall@10
   (sibling rule) wins; a tie goes to the smaller model.
2. Validation on the tune-period queries (2026-04..06), chosen model vs embedding retrieval:
   adopted only if recall@10 difference has a 95% paired bootstrap interval entirely above 0,
   and the over-30-days group's interval is not entirely below 0. Runtime is reported.
Test queries are not used.

Output: outputs/stage2/duplicate_crossencoder_dev.json
Cache:  data/stage2/crossencoder_<model>_<split>.npy (scores, queries x 50)
Usage:  .venv/bin/python scripts/stage2/dedup_crossencoder.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from common2 import AGE_GROUPS, CACHE2, OUT2, age_group, duplicate_links, found_at
from common import N_BOOT, SEED, load_issues, load_sets, write_json
from dedup import Corpus
from dedup_clean import clean_text, embed_clean
from stats import mean_boot_values, paired_difference, resample_indices, summary

K = 50
MODELS = {
    "quora_distilroberta": {"id": "cross-encoder/quora-distilroberta-base",
                            "revision": "f62e7a4b20b97195c2868e53ec59126df5eac743",
                            "side_tokens": 254, "max_length": 512, "params_m": 82},
    "gte_modernbert": {"id": "Alibaba-NLP/gte-reranker-modernbert-base",
                       "revision": "f7481e6055501a30fb19d090657df9ec1f79ab2c",
                       "side_tokens": 1024, "max_length": 2056, "params_m": 149},
}


def top_candidates(queries, corpus, emb):
    out = []
    for q in queries:
        qi, p = corpus.pos[q["number"]], corpus.pool_size(q["number"])
        s = emb[:p] @ emb[qi]
        top = np.argsort(-s, kind="stable")[:K]
        out.append([corpus.numbers[i] for i in top])
    return out


def score(name: str, split: str, queries, cands, text) -> tuple[np.ndarray, float]:
    """Cross-encoder scores (queries x K) of one model on one split, cached under data/stage2/."""
    cfg = MODELS[name]
    path = CACHE2 / f"crossencoder_{name}_{split}.npy"
    meta = CACHE2 / f"crossencoder_{name}_{split}_meta.json"
    key = [cfg["revision"], cfg["side_tokens"], [q["number"] for q in queries], cands]
    if path.exists() and meta.exists() and json.load(open(meta))["key"] == key:
        return np.load(path), json.load(open(meta))["seconds"]
    import torch
    from sentence_transformers import CrossEncoder
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CrossEncoder(cfg["id"], revision=cfg["revision"], max_length=cfg["max_length"], device=device,
                         trust_remote_code=False)
    tok = model.tokenizer
    cut = {}

    def side(n: int) -> str:
        if n not in cut:
            ids = tok.encode(text(n), add_special_tokens=False)[: cfg["side_tokens"]]
            cut[n] = tok.decode(ids)
        return cut[n]

    t0 = time.time()
    scores = np.zeros((len(queries), K), dtype=np.float32)
    for i, (q, cs) in enumerate(zip(queries, cands)):
        scores[i] = model.predict([(side(q["number"]), side(c)) for c in cs], batch_size=25, show_progress_bar=False)
        if (i + 1) % 100 == 0:
            print(f"{name}: {i + 1}/{len(queries)} queries, {time.time() - t0:.0f}s", flush=True)
    secs = round(time.time() - t0, 1)
    CACHE2.mkdir(parents=True, exist_ok=True)
    np.save(path, scores)
    write_json(meta, {"key": key, "device": device, "seconds": secs})
    return scores, secs


def hits(queries, ranked, issues, links) -> np.ndarray:
    """recall@10 per query (sibling rule) for a list of ranked candidate lists."""
    out = []
    for q, r in zip(queries, ranked):
        rank = r.index(q["truth"]) + 1 if q["truth"] in r else 10 ** 6
        out.append(float(found_at(10, q["number"], q["truth"], rank, r, links, issues[q["number"]]["created_at"])))
    return np.array(out)


def main() -> None:
    issues = load_issues()
    links = duplicate_links(issues)
    sets = load_sets("duplicate")
    corpus = Corpus(issues)
    corpus.texts = [clean_text(issues[n]) for n in corpus.numbers]
    emb = embed_clean(corpus)
    text = lambda n: corpus.texts[corpus.pos[n]]

    res, per_query = {"k": K, "models": {}}, {}
    split_q = {sp: sorted(sets[sp], key=lambda x: (x["created_at"], x["number"])) for sp in ("train", "tune")}
    ranked = {}
    for sp, qs in split_q.items():
        cands = top_candidates(qs, corpus, emb)
        ranked[sp, "embedding"] = cands
        for name in MODELS:
            sc, secs = score(name, sp, qs, cands, text)
            ranked[sp, name] = [[c for _, c in sorted(zip(-s, cs), key=lambda t: t[0])] for s, cs in zip(sc, cands)]
            res["models"].setdefault(name, dict(MODELS[name]))[f"seconds_{sp}"] = secs

    # 1. model choice on train-period queries
    tq = split_q["train"]
    idx = resample_indices(len(tq), N_BOOT, SEED)
    choice = {}
    for m in ("embedding", *MODELS):
        h = hits(tq, ranked["train", m], issues, links)
        choice[m] = summary(h.mean(), mean_boot_values(h, idx))
    best = max(MODELS, key=lambda m: (choice[m]["point"], -MODELS[m]["params_m"]))
    res["choice_on_train_queries"] = {"n": len(tq), "recall@10": choice, "chosen": best}

    # 2. validation on tune-period queries
    vq = split_q["tune"]
    groups = np.array([age_group(issues, q["number"], q["truth"]) for q in vq])
    h_e = hits(vq, ranked["tune", "embedding"], issues, links)
    h_c = hits(vq, ranked["tune", best], issues, links)
    val = {}
    for sub in ("all", *AGE_GROUPS):
        pos = np.arange(len(vq)) if sub == "all" else np.where(groups == sub)[0]
        i2 = resample_indices(len(pos), N_BOOT, SEED)
        be, bc = mean_boot_values(h_e[pos], i2), mean_boot_values(h_c[pos], i2)
        val[sub] = {"n": len(pos), "embedding": summary(h_e[pos].mean(), be), best: summary(h_c[pos].mean(), bc),
                    "difference": paired_difference(h_c[pos].mean(), bc, h_e[pos].mean(), be)}
    res["validation_on_tune_queries"] = val
    res["adopt"] = bool(val["all"]["difference"]["ci95"][0] > 0 and not val["over_30_days"]["difference"]["ci95"][1] < 0)

    for sp, qs in split_q.items():
        for j, q in enumerate(qs):
            per_query[q["number"]] = {"split": sp, "original": q["truth"],
                                      **{f"{m}_top10": ranked[sp, m][j][:10] for m in ("embedding", *MODELS)}}
    write_json(OUT2 / "duplicate_crossencoder_dev.json", {"metrics": res, "queries": per_query})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
