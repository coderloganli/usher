# Pipeline details

Exact settings of the current pipeline for each judgment, as needed to reproduce it. The overview and results are in the [README](../README.md); why each setting was chosen is in [EXPERIMENTS.md](../EXPERIMENTS.md).

## Type: TF-IDF + logistic regression

- **Input text**: title plus the raw body (HTML comments kept, so issue-template markers are visible).
- **Features**: TF-IDF with scikit-learn defaults, fitted on train text only.
- **Model**: multinomial logistic regression trained on the train set, `max_iter` 5,000.
- **Two settings** are kept, because accuracy and macro-F1 pick different ones on the tune set (grid C in {0.01, 0.1, 1, 10, 100} x `class_weight` in {none, balanced}):

| Name | Selected by | Setting |
|---|---|---|
| LR (accuracy) | tune accuracy | C = 1, no class weighting |
| LR (macro-F1) | tune macro-F1 | C = 1, `class_weight="balanced"` |

Code: `scripts/baselines/classify.py` (tuning, writes `outputs/baselines/type_tuning.json`) and `scripts/baselines/evaluate_test.py` (trains the frozen settings on train, scores test).

## Module: LLM final judgment with LR and similar issues as reference

Code: `scripts/stage2/module_llm.py`.

**Label set.** The 291 keys of vscode's `.github/classifier.json` that occur in the snapshot. A guess is a hit if it is any of the issue's module labels.

**Prepared once, from train-period module issues only** (2025-01 to 2026-03, 8,057 issues):

- the logistic regression model below, covering the 272 modules seen in train;
- `BAAI/bge-small-en-v1.5` vectors of their cleaned text (shared with the duplicate pipeline, cached in `data/stage2/embeddings_clean.npy`);
- the 291 label names and the GitHub descriptions of 205 of them (`scripts/stage2/module_label_descriptions.json`, fetched 2026-10-10).

**Steps for one issue.**

1. **Reference 1, logistic regression.** Score the title + raw body; keep the 10 most probable modules with their probabilities.
2. **Clean the text** with the duplicate pipeline's `clean_text` (see below).
3. **Reference 2, similar issues.** The 10 train issues created before this issue with the highest cosine similarity to its cleaned text, with their titles, module labels and similarity.
4. **Prompt.** Cleaned issue text cut to 1,500 tokens, the two references, the descriptions of every candidate module in them, and the names of all 291 labels (about 3,100 input tokens per issue).
5. **Final judgment.** `gpt-4o-mini-2024-07-18` (Chat Completions, temperature 0, fixed seed, no tools) returns structured output: a one-sentence reason, then one label constrained to the 291.

Cost is about US$0.0004 per issue. Responses are cached in `data/stage2/module_llm/`; with the cache a rerun makes no API calls and reproduces the outputs exactly.

**Logistic regression used in step 1.** Same text and TF-IDF as type. One yes/no logistic regression per module seen in train (272 modules), C = 10, `class_weight="balanced"` (chosen on tune by top-1 hit rate); modules are ranked by probability. On its own this is the module baseline.

## Duplicate: cleaned text, embedding retrieval, cross-encoder rerank

Code: `scripts/stage2/dedup_clean.py` (cleaning), `scripts/stage2/dedup_final.py` (frozen pipeline).

**Candidates.** Every snapshot issue (from 2023-01) created strictly before the query.

**Steps for one query.**

1. **Clean the text** (`clean_text`), applied to queries and candidates alike:
   - remove HTML comments;
   - remove folded `<details>` blocks whose summary is System Info, A/B Experiments, Process Info, Workspace Info, Request IDs or Extensions (N) (Logs, Diagnostics and stack traces are kept);
   - remove whole lines starting with `Type: <b>`, `VS Code version:`, `OS version:`, `Remote OS version:`, `Extension version:`, `Modes:`, `Does this issue occur when all extensions are disabled?`, `Extensions: none`;
   - remove `<img>` tags and lines that are only a `github.com/user-attachments` link.

   Input is title + cleaned body.
2. **Embed and retrieve the top 50.** `BAAI/bge-small-en-v1.5` at revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, no instruction prefix, input truncated at 512 tokens, L2-normalised vectors, cosine similarity; keep the 50 highest.
3. **Rerank.** `Alibaba-NLP/gte-reranker-modernbert-base` at revision `f7481e6055501a30fb19d090657df9ec1f79ab2c` scores each (query, candidate) pair; each side is cut to 1,024 tokens of the model's tokenizer (`max_length` 2,056). The 50 candidates are re-sorted by this score (ties keep the embedding order).
4. **Output the top 10.**

**Runtime** on an Apple MPS device: embedding all 107,422 cleaned issues takes about 254 s; the cross-encoder takes about 2.2 s per query.

**Duplicate link times.** The sibling rule needs the creation time of every duplicate link. `scripts/stage2/pull_duplicate_events.py` takes it from GitHub's `MarkedAsDuplicateEvent`, or from a `ClosedEvent` with state reason DUPLICATE; 2,289 of 2,304 links are covered.

## Development runs

These scripts produced the development results that chose the pipelines above; they are not needed to rerun the final pipelines.

| Command | Writes |
|---|---|
| `.venv/bin/python scripts/stage2/dedup_clean.py` | `outputs/stage2/duplicate_clean_dev.json`; embedding cache `data/stage2/embeddings_clean.npy` |
| `.venv/bin/python scripts/stage2/dedup_crossencoder.py` | `outputs/stage2/duplicate_crossencoder_dev.json`; caches `data/stage2/crossencoder_<model>_<split>.npy` |
| `.venv/bin/python scripts/stage2/module_llm.py trial` / `tune` | `outputs/stage2/module_llm_trial.json`, `outputs/stage2/module_llm_tune.json` |

Other experiments, including ones not adopted, are listed in EXPERIMENTS.md with their scripts.
