"""Embed every corpus issue (title + truncated body) with a sentence-transformers model.

Model: BAAI/bge-small-en-v1.5 at a pinned revision, no query instruction
(issue-to-issue similarity is symmetric; the model card says no instruction
is needed in that case). Embeddings are L2-normalised so dot product = cosine.

Output: data/embeddings.npy (float32, rows in load_corpus() order),
        data/embeddings_meta.json
Usage:  python scripts/embed.py
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import DATA, MODEL_ID, MODEL_REVISION, SEED, issue_text, load_corpus, write_json


def main() -> None:
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    corpus = load_corpus()
    texts = [issue_text(r["title"], r["body"]) for r in corpus]
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device=device)
    t0 = time.time()
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    secs = time.time() - t0
    np.save(DATA / "embeddings.npy", emb)
    write_json(DATA / "embeddings_meta.json", {
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "device": device,
        "max_seq_length": model.max_seq_length, "n": len(texts), "dim": int(emb.shape[1]),
        "numbers_first_last": [corpus[0]["number"], corpus[-1]["number"]], "seconds": round(secs, 1)})
    print(f"embedded {len(texts)} issues in {secs:.0f}s on {device}")


if __name__ == "__main__":
    main()
