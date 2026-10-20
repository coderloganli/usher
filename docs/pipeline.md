# Pipeline details

Exact settings of the current pipeline for each judgment, as needed to reproduce it. The overview and results are in the [README](../README.md); why each setting was chosen is in [EXPERIMENTS.md](../EXPERIMENTS.md).

## Type: LLM final judgment with the reporter's choice, logistic regression and similar issues

Code: `scripts/stage2/type_llm.py`; subset and logistic regressions in `scripts/stage2/type_baselines.py`.

**Label set.** The 10 type labels: `bug`, `feature-request`, `*question`, `under-discussion`, `debt`, `polish`, `upstream`, `engineering`, `upstream-issue-linked`, `electron`.

**Evaluation subset.** Type issues whose author is not a team-proxy account (an account that is an assignee on any snapshot issue; 107 accounts). Train 6,421, tune 890, test 556 issues. The subset only selects which issues are scored; the references below use all train-period type issues.

**Prepared once, from train-period type issues** (2025-01 to 2026-03, 13,554 issues):

- the logistic regression below;
- `BAAI/bge-small-en-v1.5` vectors of their cleaned text (shared with the duplicate pipeline);
- the type definitions from the vscode wiki page "Issues-Triaging", wiki commit `49c140c` (2026-08-27), copied verbatim into `scripts/stage2/type_definitions.json`: the "Categorizing Issues" row of each type, plus the "Closing Issues" row for `*question`, `upstream` and `upstream-issue-linked`.

**Steps for one issue.**

1. **Reporter's choice.** From the raw body: the Type field of VS Code's built-in issue reporter (`Type: <b>Bug</b>`, `Feature Request` or `Performance Issue`, the last counted as `bug`), else the bug or feature-request template marker; otherwise none.
2. **Reference 1, logistic regression.** Probabilities of all 10 types from the title + raw body.
3. **Clean the text** with the duplicate pipeline's `clean_text` (see below).
4. **Reference 2, similar issues.** The 10 train-period type issues created before this issue with the highest cosine similarity to its cleaned text, with their titles, types and similarity.
5. **Prompt.** Cleaned issue text cut to 1,500 tokens, the reporter's choice (with a note that reporters only choose between `bug` and `feature-request` and the team often changes it), the two references (with a note that the classifier rarely predicts other types), and the 10 definitions (about 1,100 input tokens per issue).
6. **Final judgment.** `gpt-4o-mini-2024-07-18` (Chat Completions, temperature 0, fixed seed, no tools) returns structured output: a one-sentence reason, then one label constrained to the 10.

Cost is about US$0.0002 per issue. Responses are cached in `data/stage2/type_llm/`; with the cache a rerun makes no API calls and reproduces the outputs exactly.

**Logistic regression used in step 2.** Title plus raw body (HTML comments kept, so issue-template markers are visible); TF-IDF with scikit-learn defaults fitted on train text only; multinomial logistic regression, C = 1, no class weighting, `max_iter` 5,000, trained on all train-period type issues. The stage-1 grid was C in {0.01, 0.1, 1, 10, 100} x `class_weight` in {none, balanced}; this setting has the highest tune accuracy. The setting with the highest tune macro-F1 (C = 1, `class_weight="balanced"`) is kept as a second baseline.

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
| `.venv/bin/python scripts/stage2/type_baselines.py` | `outputs/stage2/type_baselines_tune.json` |
| `.venv/bin/python scripts/stage2/type_llm.py trial` / `tune` | `outputs/stage2/type_llm_trial.json`, `outputs/stage2/type_llm_tune.json` |

Other experiments, including ones not adopted, are listed in EXPERIMENTS.md with their scripts.
