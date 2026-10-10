"""LLM query rewriting for duplicate retrieval (stage 2; plan approved 2026-10-10).

Only the query is rewritten, and only for the retrieval step: the rewrite (alone, or put in
front of the query's cleaned text) is embedded with the same model as the cleaned-text
retrieval and the top 50 earlier issues are kept. Candidates keep their cleaned text.

Rewriting styles (prompts below):
  summary   a one-line title and two or three sentences (feature, symptom, exact errors)
  keywords  10 to 20 identifying keywords and phrases (as in CUPID, Zhang et al. 2023)
  hyde      a hypothetical earlier issue reporting the same problem (as in HyDE, Gao et al. 2022)
Each style is used two ways: "alone" and "plus_original" (rewrite + blank line + cleaned text),
so there are 6 configurations.

Model: gpt-4o-mini-2024-07-18 through the Chat Completions API (official `openai` package),
temperature 0, fixed seed (best effort only, per OpenAI's documentation), at most 300 output
tokens; the query's cleaned text is cut to 4,000 tokens. Every response is cached in
data/stage2/rewrites/<style>.jsonl (gitignored: rewrites paraphrase issue text), so reruns
make no API calls. The API key is read from the git-ignored .env file.

Subcommands
  rewrite --trial   rewrite 100 random train-period queries with every style (cost check)
  rewrite           rewrite all 1,205 development queries with every style
  retrieval         score the 6 configurations and the no-rewrite retrieval on the
                    train-period queries: recall@10 and recall@50 (sibling rule)
  rerank            rerank each configuration's top 50 with the adopted cross-encoder
                    (scripts/stage2/dedup_crossencoder.py, gte-reranker-modernbert-base, reading
                    the query's cleaned text, not the rewrite); choose the configuration with the
                    highest final recall@10 on train-period queries; validate it on tune-period
                    queries against the current best pipeline (no rewrite). Rule fixed in advance:
                    adopt only if the recall@10 difference has a 95% paired bootstrap interval
                    entirely above 0 and the over-30-days group's interval is not entirely below 0.
                    Pair scores already computed are reused (data/stage2/ce_pairs_gte.json).
Usage: .venv/bin/python scripts/stage2/dedup_rewrite.py <subcommand> [--trial]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common2 import CACHE2, OUT2, duplicate_links, found_at
from common import ROOT, SEED, load_issues, load_sets, write_json
from dedup import MODEL_ID, MODEL_REVISION, Corpus
from dedup_clean import clean_text, embed_clean

LLM = "gpt-4o-mini-2024-07-18"
PRICE_PER_M = {"input": 0.15, "output": 0.60}  # US dollars, standard tier, checked 2026-10-08
MAX_INPUT_TOKENS = 4000
MAX_OUTPUT_TOKENS = 300
K = 50
REWRITES = CACHE2 / "rewrites"
SYSTEM = "You help find duplicate GitHub issues in the microsoft/vscode repository."
PROMPTS = {
    "summary": ("Rewrite the issue below as a search query for finding an earlier issue that reports the same "
                "problem. Write one line with a short title, then two or three sentences saying which feature is "
                "involved, what goes wrong, and any error message (copy error messages exactly). Write in English. "
                "Output only the rewritten text."),
    "keywords": ("List 10 to 20 keywords and short phrases that best identify the problem in the issue below, for "
                 "finding an earlier issue that reports the same problem: feature and setting names, symptoms, and "
                 "error messages copied exactly. Write in English, one per line. Output only the list."),
    "hyde": ("Suppose the problem in the issue below was reported earlier by another user. Write the title and body "
             "of that earlier issue as it might have been written, in English, at most 150 words. Output only the "
             "title on the first line and the body after it."),
}
USES = ("alone", "plus_original")


def load_key() -> None:
    """Put OPENAI_API_KEY from the git-ignored .env into the environment (value never printed)."""
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()


def cached(style: str) -> dict[int, dict]:
    path = REWRITES / f"{style}.jsonl"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["number"]: r for r in map(json.loads, f)}


def rewrite(numbers: list[int], issues: dict) -> dict:
    import tiktoken
    from openai import OpenAI
    load_key()
    client = OpenAI(max_retries=6)
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    REWRITES.mkdir(parents=True, exist_ok=True)
    totals = {}
    for style, prompt in PROMPTS.items():
        done = cached(style)
        todo = [n for n in numbers if n not in done]

        def call(n: int) -> dict:
            text = enc.decode(enc.encode(clean_text(issues[n]))[:MAX_INPUT_TOKENS])
            r = client.chat.completions.create(
                model=LLM, temperature=0, seed=SEED, max_completion_tokens=MAX_OUTPUT_TOKENS,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"{prompt}\n\nIssue:\n{text}"}])
            return {"number": n, "text": r.choices[0].message.content, "model": r.model,
                    "system_fingerprint": r.system_fingerprint, "finish_reason": r.choices[0].finish_reason,
                    "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens}

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as pool, open(REWRITES / f"{style}.jsonl", "a", encoding="utf-8") as f:
            for i, row in enumerate(pool.map(call, todo), 1):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if i % 100 == 0:
                    print(f"{style}: {i}/{len(todo)}", flush=True)
        rows = [r for n, r in cached(style).items() if n in set(numbers)]
        pt, ct = sum(r["prompt_tokens"] for r in rows), sum(r["completion_tokens"] for r in rows)
        totals[style] = {"queries": len(rows), "new_calls": len(todo), "seconds": round(time.time() - t0, 1),
                         "prompt_tokens": pt, "completion_tokens": ct,
                         "cost_usd": round(pt / 1e6 * PRICE_PER_M["input"] + ct / 1e6 * PRICE_PER_M["output"], 4),
                         "finish_reason_length": sum(r["finish_reason"] == "length" for r in rows)}
        print(style, totals[style], flush=True)
    return totals


def query_embeddings(texts: list[str]) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device=device)
    return model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def retrieval_scores(queries, issues, links, corpus, emb, qvec) -> dict:
    """recall@10 and recall@50 (sibling rule) of embedding retrieval with the given query vectors."""
    h10, h50, tops = [], [], []
    for q, v in zip(queries, qvec):
        p = corpus.pool_size(q["number"])
        s = emb[:p] @ v
        top = [corpus.numbers[i] for i in np.argsort(-s, kind="stable")[:K]]
        rank = top.index(q["truth"]) + 1 if q["truth"] in top else 10 ** 6
        qt = issues[q["number"]]["created_at"]
        h10.append(float(found_at(10, q["number"], q["truth"], rank, top, links, qt)))
        h50.append(float(found_at(K, q["number"], q["truth"], rank, top, links, qt)))
        tops.append(top)
    return {"recall@10": float(np.mean(h10)), "recall@50": float(np.mean(h50)), "top50": tops,
            "hits10": h10, "hits50": h50}


def config_texts(queries, corpus, style: str, use: str) -> list[str]:
    rw = cached(style)
    clean = lambda n: corpus.texts[corpus.pos[n]]
    return [rw[q["number"]]["text"] if use == "alone" else rw[q["number"]]["text"] + "\n\n" + clean(q["number"])
            for q in queries]


def top50(queries, corpus, emb, qvec) -> list[list[int]]:
    out = []
    for q, v in zip(queries, qvec):
        s = emb[:corpus.pool_size(q["number"])] @ v
        out.append([corpus.numbers[i] for i in np.argsort(-s, kind="stable")[:K]])
    return out


class PairScores:
    """Cross-encoder scores per (query, candidate) pair, seeded from dedup_crossencoder.py caches."""

    def __init__(self, corpus):
        from dedup_crossencoder import MODELS
        self.cfg = MODELS["gte_modernbert"]
        self.corpus = corpus
        self.path = CACHE2 / "ce_pairs_gte.json"
        self.s = json.load(open(self.path)) if self.path.exists() else {}
        for split in ("train", "tune"):
            meta = CACHE2 / f"crossencoder_gte_modernbert_{split}_meta.json"
            if meta.exists():
                key = json.load(open(meta))["key"]
                sc = np.load(CACHE2 / f"crossencoder_gte_modernbert_{split}.npy")
                for qn, cs, row in zip(key[2], key[3], sc):
                    for c, v in zip(cs, row):
                        self.s.setdefault(f"{qn}-{c}", float(v))
        self.model = None
        self.cut = {}

    def side(self, n: int) -> str:
        if n not in self.cut:
            tok = self.model.tokenizer
            ids = tok.encode(self.corpus.texts[self.corpus.pos[n]], add_special_tokens=False)[: self.cfg["side_tokens"]]
            self.cut[n] = tok.decode(ids)
        return self.cut[n]

    def get(self, q: int, cands: list[int]) -> list[float]:
        missing = [c for c in cands if f"{q}-{c}" not in self.s]
        if missing:
            if self.model is None:
                import torch
                from sentence_transformers import CrossEncoder
                torch.manual_seed(SEED)
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                self.model = CrossEncoder(self.cfg["id"], revision=self.cfg["revision"],
                                          max_length=self.cfg["max_length"], device=device)
            vals = self.model.predict([(self.side(q), self.side(c)) for c in missing], batch_size=25,
                                      show_progress_bar=False)
            for c, v in zip(missing, vals):
                self.s[f"{q}-{c}"] = float(v)
        return [self.s[f"{q}-{c}"] for c in cands]

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.s, f)


def final_hits(queries, tops, pairs, issues, links) -> np.ndarray:
    out = []
    for q, top in zip(queries, tops):
        sc = pairs.get(q["number"], top)
        ranked = [c for _, c in sorted(zip([-x for x in sc], top), key=lambda t: t[0])]
        rank = ranked.index(q["truth"]) + 1 if q["truth"] in ranked else 10 ** 6
        out.append(float(found_at(10, q["number"], q["truth"], rank, ranked, links, issues[q["number"]]["created_at"])))
    return np.array(out)


def rerank(issues, train, tune) -> None:
    from common import N_BOOT
    from common2 import AGE_GROUPS, age_group
    from stats import mean_boot_values, paired_difference, resample_indices, summary
    links = duplicate_links(issues)
    corpus = Corpus(issues)
    corpus.texts = [clean_text(issues[n]) for n in corpus.numbers]
    emb = embed_clean(corpus)
    pairs = PairScores(corpus)
    base_vec = lambda qs: [emb[corpus.pos[q["number"]]] for q in qs]
    res = {"choice_on_train_queries": {"n": len(train), "recall@10": {}}}
    t0 = time.time()
    hits = {"no_rewrite": final_hits(train, top50(train, corpus, emb, base_vec(train)), pairs, issues, links)}
    for style in PROMPTS:
        for use in USES:
            name = f"{style}_{use}"
            qv = query_embeddings(config_texts(train, corpus, style, use))
            hits[name] = final_hits(train, top50(train, corpus, emb, qv), pairs, issues, links)
            pairs.save()
            print(name, round(float(hits[name].mean()), 4), f"{time.time() - t0:.0f}s", flush=True)
    idx = resample_indices(len(train), N_BOOT, SEED)
    for name, h in hits.items():
        res["choice_on_train_queries"]["recall@10"][name] = summary(h.mean(), mean_boot_values(h, idx))
    best = max((n for n in hits if n != "no_rewrite"), key=lambda n: hits[n].mean())
    res["choice_on_train_queries"]["chosen"] = best
    style, use = {f"{st}_{u}": (st, u) for st in PROMPTS for u in USES}[best]

    h_base = final_hits(tune, top50(tune, corpus, emb, base_vec(tune)), pairs, issues, links)
    h_new = final_hits(tune, top50(tune, corpus, emb, query_embeddings(config_texts(tune, corpus, style, use))),
                       pairs, issues, links)
    pairs.save()
    groups = np.array([age_group(issues, q["number"], q["truth"]) for q in tune])
    val = {}
    for sub in ("all", *AGE_GROUPS):
        pos = np.arange(len(tune)) if sub == "all" else np.where(groups == sub)[0]
        i2 = resample_indices(len(pos), N_BOOT, SEED)
        bb, bn = mean_boot_values(h_base[pos], i2), mean_boot_values(h_new[pos], i2)
        val[sub] = {"n": len(pos), "current_best": summary(h_base[pos].mean(), bb), best: summary(h_new[pos].mean(), bn),
                    "difference": paired_difference(h_new[pos].mean(), bn, h_base[pos].mean(), bb)}
    res["validation_on_tune_queries"] = val
    res["adopt"] = bool(val["all"]["difference"]["ci95"][0] > 0 and not val["over_30_days"]["difference"]["ci95"][1] < 0)
    res["seconds"] = round(time.time() - t0, 1)
    write_json(OUT2 / "duplicate_rewrite_rerank.json", res)
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["rewrite", "retrieval", "rerank"])
    ap.add_argument("--trial", action="store_true")
    a = ap.parse_args()
    issues = load_issues()
    sets = load_sets("duplicate")
    train = sorted(sets["train"], key=lambda x: (x["created_at"], x["number"]))
    tune = sorted(sets["tune"], key=lambda x: (x["created_at"], x["number"]))

    if a.command == "rewrite":
        if a.trial:
            nums = sorted(random.Random(SEED).sample([q["number"] for q in train], 100))
        else:
            nums = [q["number"] for q in train + tune]
        totals = rewrite(nums, issues)
        write_json(OUT2 / ("duplicate_rewrite_trial_cost.json" if a.trial else "duplicate_rewrite_cost.json"),
                   {"model": LLM, "price_per_million_usd": PRICE_PER_M, "styles": totals})
        return

    if a.command == "rerank":
        rerank(issues, train, tune)
        return

    links = duplicate_links(issues)
    corpus = Corpus(issues)
    corpus.texts = [clean_text(issues[n]) for n in corpus.numbers]
    emb = embed_clean(corpus)
    base = [emb[corpus.pos[q["number"]]] for q in train]
    res = {"queries": len(train), "split": "train", "configs": {}}
    r = retrieval_scores(train, issues, links, corpus, emb, base)
    res["configs"]["no_rewrite"] = {k: r[k] for k in ("recall@10", "recall@50")}
    for style in PROMPTS:
        rw = cached(style)
        for use in USES:
            texts = [rw[q["number"]]["text"] if use == "alone"
                     else rw[q["number"]]["text"] + "\n\n" + corpus.texts[corpus.pos[q["number"]]] for q in train]
            r = retrieval_scores(train, issues, links, corpus, emb, query_embeddings(texts))
            res["configs"][f"{style}_{use}"] = {k: r[k] for k in ("recall@10", "recall@50")}
            print(style, use, res["configs"][f"{style}_{use}"], flush=True)
    write_json(OUT2 / "duplicate_rewrite_retrieval_train.json", res)


if __name__ == "__main__":
    main()
