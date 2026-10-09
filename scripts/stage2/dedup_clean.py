"""Duplicate retrieval on cleaned text, compared with raw text on the development queries.

Cleaning (decided 2026-10-08) removes content that templates or VS Code add automatically:
1. HTML comments;
2. folded <details> blocks whose summary is System Info, A/B Experiments, Process Info,
   Workspace Info, Request IDs or Extensions (N); Logs, Diagnostics and stack traces are kept;
3. whole lines starting with "Type: <b>", "VS Code version:", "OS version:",
   "Remote OS version:", "Extension version:", "Modes:", "Does this issue occur when all
   extensions are disabled?", "Extensions: none";
4. <img> tags and lines that are only a github.com/user-attachments link.
The cleaned title + body replaces the raw text for both queries and candidates, for BM25 and
for embedding retrieval (same model, revision and settings as stage 1).

Scoring uses the sibling rule (common2.found_at). Decision rule fixed in advance: cleaning is
adopted for a method only if cleaned minus raw recall@10 has a 95% paired bootstrap interval
entirely above 0. Test queries are not used.

Output: outputs/stage2/duplicate_clean_dev.json
Cache:  data/stage2/embeddings_clean.npy (+ _meta.json)
Usage:  .venv/bin/python scripts/stage2/dedup_clean.py
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from common2 import CACHE2, OUT2, duplicate_links, found_at
from common import N_BOOT, OUT, SEED, load_issues, load_sets, write_json
from dedup import BM25, MODEL_ID, MODEL_REVISION, Corpus, retrieve
from stats import mean_boot_values, paired_difference, resample_indices, summary

KS = (1, 5, 10)
DROP_SUMMARY = re.compile(r"^(System Info|A/B Experiments|Process Info|Workspace Info|Request IDs|Extensions \(\d+\))$")
DETAILS = re.compile(r"<details>\s*<summary>(.*?)</summary>.*?</details>", re.S)
LINES = re.compile(r"^\s*(Type: <b>.*?</b>|(VS Code|OS|Remote OS|Extension) version:.*|Modes:.*|"
                   r"Does this issue occur when all extensions are disabled\?.*|Extensions: none.*|"
                   r"https://github\.com/user-attachments/\S+)\s*$", re.M)


def clean_text(r: dict) -> str:
    b = re.sub(r"<!--.*?-->", "", r["body"] or "", flags=re.S)
    b = DETAILS.sub(lambda m: "" if DROP_SUMMARY.match(m.group(1).strip()) else m.group(0), b)
    b = re.sub(r"<img[^>]*>", "", b)
    b = LINES.sub("", b)
    b = re.sub(r"\n\s*\n+", "\n\n", b).strip()
    return (r["title"] or "") + "\n\n" + b


def embed_clean(corpus: Corpus) -> np.ndarray:
    path, meta_path = CACHE2 / "embeddings_clean.npy", CACHE2 / "embeddings_clean_meta.json"
    key = [MODEL_ID, MODEL_REVISION, corpus.numbers[0], corpus.numbers[-1], len(corpus.numbers)]
    if path.exists() and meta_path.exists() and json.load(open(meta_path))["key"] == key:
        return np.load(path)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device=device)
    t0 = time.time()
    emb = model.encode(corpus.texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    CACHE2.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    write_json(meta_path, {"key": key, "device": device, "seconds": round(time.time() - t0, 1)})
    return emb


def main() -> None:
    issues = load_issues()
    links = duplicate_links(issues)
    sets = load_sets("duplicate")
    dev = sorted(sets["train"] + sets["tune"], key=lambda x: (x["created_at"], x["number"]))

    corpus = Corpus(issues)
    corpus.texts = [clean_text(issues[n]) for n in corpus.numbers]
    clean_rows = retrieve(dev, corpus, BM25(corpus.texts), embed_clean(corpus), ("bm25", "embedding"))
    raw_rows = {r["query"]: r for r in json.load(open(OUT / "duplicate_dev.json"))["queries"]}

    idx = resample_indices(len(dev), N_BOOT, SEED)
    res = {"n_queries": len(dev), "rule": "sibling rule; adopt cleaning if cleaned - raw recall@10 interval > 0",
           "methods": {}}
    for m in ("bm25", "embedding"):
        block = {}
        hits = {}
        for name, rows in (("raw", [raw_rows[q["number"]] for q in dev]), ("clean", clean_rows)):
            for k in KS:
                h = np.array([float(found_at(k, r["query"], r["original"], r[f"{m}_rank"], r[f"{m}_top10"],
                                             links, issues[r["query"]]["created_at"])) for r in rows])
                hits[name, k] = (h.mean(), mean_boot_values(h, idx))
                block.setdefault(name, {})[f"recall@{k}"] = summary(*hits[name, k])
        block["clean_minus_raw"] = {f"recall@{k}": paired_difference(*hits["clean", k], *hits["raw", k]) for k in KS}
        block["adopt_cleaning"] = bool(block["clean_minus_raw"]["recall@10"]["ci95"][0] > 0)
        res["methods"][m] = block
    write_json(OUT2 / "duplicate_clean_dev.json", {"metrics": res, "queries": clean_rows})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
