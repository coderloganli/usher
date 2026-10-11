# Usher: current best pipeline

Usher is an issue-triage agent for the `microsoft/vscode` GitHub repository. It is evaluated by replaying history: every issue is judged using only information that existed when it was created. Usher makes three judgments for each new issue:

- **Type**: which of the 10 vscode type labels (for example `bug`, `feature-request`) the issue gets.
- **Module**: which code module labels (keys of vscode's `.github/classifier.json`) the issue gets.
- **Duplicate**: which earlier issue it duplicates, given as a ranked list of candidates.

This file describes the best pipeline that exists today for each judgment and how to rerun it. The record of how it was reached is in [EXPERIMENTS.md](EXPERIMENTS.md).

## Status (2026-10-11)

- Stage 1 (data snapshot, evaluation framework and non-LLM baselines) is complete. It was not published (decided 2026-10-08).
- Stage 2 (LLM versions of the three judgments) is in progress. Duplicate detection is finished: its final pipeline is non-LLM (text cleaning and a local cross-encoder); LLM query rewriting was tried and not adopted. Module is finished: gpt-4o-mini makes the final judgment with the stage-1 LR candidates and similar past issues as reference material. Type has no stage-2 work yet; it is next.
- All work is on the git branch `stage1-data` (this worktree). Nothing has been pushed.

## Data and evaluation in brief

- Snapshot: 107,422 `microsoft/vscode` issues created 2023-01-01 to 2026-09-30, pulled 2026-10-05 (`manifest/counts.json`, `docs/data_pipeline.html`).
- Split by issue creation time (UTC): train 2025-01 to 2026-03, tune 2026-04 to 2026-06, test 2026-07 to 2026-09. The tune period was widened from one month to three on 2026-10-07.
- Issues per judgment after filtering (train / tune / test): type 13,554 / 2,165 / 1,848; module 8,057 / 986 / 687; duplicate queries 864 / 341 / 268 (`manifest/counts.json`).
- Ground truth is the maintainers' final labels and `duplicateOf` links. A 30-issue test sample per judgment was reviewed by an AI model (Claude Opus 5.5), not human-verified: 28 of 28 decidable type labels, 30 of 30 module labels and 27 of 28 decidable duplicate links were judged correct (`review/ai_review_summary.json`).
- The test set is used once per frozen pipeline: for the stage-1 baselines (2026-10-08), the final stage-2 duplicate pipeline (2026-10-10) and the stage-2 module pipeline (2026-10-11). No method was changed after seeing test results. Stage-2 duplicate methods are developed on the 1,205 train-period and tune-period duplicate queries ("development queries"); module methods on the 986 tune issues.
- Every score has a 95% percentile bootstrap interval (10,000 resamples, fixed seed). Comparisons use paired bootstrap; adoption rules are fixed before a run. The evaluation rules in force are listed in [EXPERIMENTS.md](EXPERIMENTS.md#evaluation-rules-in-force).

## Environment and data

1. Python 3.12 environment: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt` (exact pinned versions in `requirements.txt`).
2. Data lives in the gitignored `data/` directory. `./run_data.sh` pulls the snapshot from GitHub through the authenticated `gh` CLI (resumable) and builds the sets; `./run_data.sh --offline` rebuilds everything from already cached raw data. Main outputs: `data/issues.jsonl.gz` (all issues), `data/sets.json` (per-judgment sets with ground truth), `manifest/` (issue numbers, checksums, per-split issue lists, counts).

## Type: TF-IDF + logistic regression (stage-1 baseline)

**Method.** Input text is the title plus the raw body (HTML comments kept, so issue-template markers are visible). TF-IDF with scikit-learn defaults, fitted on train text only; multinomial logistic regression trained on the train set (`max_iter` 5,000). Two settings are kept because the two type metrics choose different settings on the tune set:

| Baseline | Selected by | Setting | Tune macro-F1 | Tune accuracy |
|---|---|---|---|---|
| LR (macro-F1) | tune macro-F1 | C = 1, `class_weight="balanced"` | 0.352 | 64.3% |
| LR (accuracy) | tune accuracy | C = 1, no class weighting | 0.248 | 80.2% |

**Run.** `./run_baselines.sh` (step 1 is `scripts/baselines/classify.py`, which scores the grid C in {0.01, 0.1, 1, 10, 100} x class_weight in {none, balanced} on tune and writes `outputs/baselines/type_tuning.json`; step 3 is `scripts/baselines/evaluate_test.py`, which trains the frozen settings on train and scores test).

**Scores on the test set (2026-07 to 2026-09, 1,848 issues), stage 1, used once:**

| Method | Macro-F1 [95% CI] | Accuracy [95% CI] |
|---|---|---|
| Majority class (`bug`) | 0.080 [0.079, 0.101] | 66.2% [64.1, 68.3] |
| LR (macro-F1) | **0.210** [0.192, 0.252] | 63.5% [61.3, 65.6] |
| LR (accuracy) | 0.203 [0.175, 0.279] | **79.4%** [77.5, 81.2] |

The two LR baselines' macro-F1 values are not significantly different on test (difference +0.008, interval -0.058 to +0.039). Stage-2 type methods are compared on macro-F1 with LR (macro-F1) and on accuracy with LR (accuracy). Source: `outputs/baselines/test_metrics.json`.

## Module: LLM final judgment with LR and similar issues as reference (stage 2, adopted 2026-10-10)

**Task.** Give the one module label (from the 291 keys of vscode's `.github/classifier.json` that occur in the snapshot) that the team put on the issue; a guess is a hit if it is any of the issue's module labels. Primary metric: top-1 hit rate.

**Prepared once, from train-period module issues only (2025-01 to 2026-03, 8,057 issues):**

- the stage-1 LR model (below), covering the 272 modules seen in train;
- `BAAI/bge-small-en-v1.5` vectors of their cleaned text (the duplicate pipeline's cache, `data/stage2/embeddings_clean.npy`);
- the 291 module label names and the GitHub descriptions of 205 of them (`scripts/stage2/module_label_descriptions.json`, fetched 2026-10-10).

**Pipeline for one issue, step by step** (`scripts/stage2/module_llm.py`).

1. **Reference 1, LR.** The stage-1 LR scores the title + raw body; keep the 10 most probable modules with their probabilities.
2. **Clean the text** with the duplicate pipeline's `clean_text` (templates, system-info blocks and images removed).
3. **Reference 2, similar issues.** The 10 train issues created before this issue with the highest cosine to its cleaned text; keep their titles, module labels and similarity.
4. **Prompt.** Cleaned issue text cut to 1,500 tokens, the two references, the descriptions of every candidate module in them, and the names of all 291 labels (about 3,100 input tokens per issue).
5. **Final judgment.** `gpt-4o-mini-2024-07-18` (Chat Completions, temperature 0, fixed seed, no tools) returns structured output: a one-sentence reason, then one label constrained to the 291.

Output: one label and a reason; no top-3 list and no confidence yet. Cost about US$0.0004 per issue.

**Run.** `./run_stage2_module.sh` runs the frozen pipeline on the test split (`module_llm.py test`). `module_llm.py trial` and `module_llm.py tune` are the development runs. Responses are cached in `data/stage2/module_llm/`; with the cache a rerun makes no API calls and reproduces the outputs exactly (checked for tune).

**Scores on the test set** (687 issues created 2026-07 to 2026-09, run once on 2026-10-11; `outputs/stage2/module_llm_test.json`). Top-1 hit [95% CI]:

| Method | All (687) | True module has 1-19 train issues (107) | 20-99 (203) | 100 or more (376) |
|---|---|---|---|---|
| Stage-1 LR | 48.2% [44.4, 52.0] | 18.7% | 42.9% | 59.6% |
| Similar-issue vote (10 neighbours) | 48.8% | 24.3% | 45.3% | 57.7% |
| **LLM final judgment** | **60.8%** [57.2, 64.5] | 35.5% | 59.6% | 68.9% |
| LLM minus LR | +12.7 [+9.3, +16.2] | +16.8 [+8.4, +26.2] | +16.7 [+10.8, +23.2] | +9.3 [+4.8, +13.8] |

Relative to LR the LLM fixed 122 issues and broke 35. One test issue has a module never seen in train; every method misses it.

**Scores on the tune set** (986 issues, 2026-04 to 2026-06; `outputs/stage2/module_llm_tune.json`), where the method was adopted by the pre-set rule (LLM minus LR interval entirely above 0): LLM 69.0%, LR 57.1%, similar-issue vote 59.1%; LLM minus LR +11.9 [+8.8, +14.9]; 176 fixed, 59 broken.

**Known limits.** References use the train period only (as in the stage-1 final model); a deployed system would use all labelled history. For modules with 100 or more training issues the LLM overrules a correct LR relatively often (tune: 92 fixed, 51 broken). Code-search tools were tried and not adopted (EXPERIMENTS.md entries 22 and 23).

### Stage-1 baseline: one-vs-rest TF-IDF + logistic regression

**Method.** Same text and TF-IDF as type. One yes/no logistic regression per module seen in train (272 modules); modules are ranked by probability. Setting chosen on tune by top-1 hit rate: C = 10, `class_weight="balanced"` (tune top-1 57.1%, top-3 76.8%).

**Run.** `./run_baselines.sh` (same steps as type; tuning record `outputs/baselines/module_tuning.json`).

**Scores on the test set (687 issues), stage 1, used once:**

| Method | Top-1 hit [95% CI] | Top-3 hit [95% CI] |
|---|---|---|
| Majority (`ux`; top-3 `ux`, `chat-terminal`, `git`) | 15.3% [12.7, 17.9] | 17.6% [14.8, 20.5] |
| One-vs-rest LR | **48.2%** [44.4, 52.0] | 68.0% [64.6, 71.3] |

On the 615 single-module test issues LR top-1 is 46.8%; on the 72 multi-module test issues it is 59.7%. One test issue (#330214) has a module never seen in train and counts as a miss. Source: `outputs/baselines/test_metrics.json`.

## Duplicate: cleaned text, embedding retrieval top 50, cross-encoder rerank (stage 2, adopted 2026-10-10)

**Task.** For each query issue, rank every snapshot issue (from 2023-01) created strictly before it; the answer is the maintainers' `duplicateOf` original. The primary metric is recall@10 under the sibling rule: a query counts as found if its original, or a "sibling" whose duplicate link to the same original was created before the query, is in the first 10 candidates.

**Pipeline, step by step.**

1. **Clean the text** (`clean_text` in `scripts/stage2/dedup_clean.py`). Applied to queries and candidates alike: remove HTML comments; remove folded `<details>` blocks whose summary is System Info, A/B Experiments, Process Info, Workspace Info, Request IDs or Extensions (N) (Logs, Diagnostics and stack traces are kept); remove whole lines starting with `Type: <b>`, `VS Code version:`, `OS version:`, `Remote OS version:`, `Extension version:`, `Modes:`, `Does this issue occur when all extensions are disabled?`, `Extensions: none`; remove `<img>` tags and lines that are only a `github.com/user-attachments` link. Input = title + cleaned body.
2. **Embed and retrieve the top 50.** `BAAI/bge-small-en-v1.5` at revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, no instruction prefix, input truncated at 512 tokens, L2-normalised vectors, cosine similarity over the pool of issues created before the query; keep the 50 highest.
3. **Rerank with a cross-encoder.** `Alibaba-NLP/gte-reranker-modernbert-base` at revision `f7481e6055501a30fb19d090657df9ec1f79ab2c` scores each (query, candidate) pair; each side is cut to 1,024 tokens of the model's tokenizer (`max_length` 2,056). The 50 candidates are re-sorted by this score (ties keep the embedding order).
4. **Output the top 10** of the reranked list.

**Run.** `./run_stage2_duplicate.sh` runs the frozen pipeline on the test split (it calls `scripts/stage2/pull_duplicate_events.py`, then `scripts/stage2/dedup_final.py --split test`, which also scores the stage-1 baseline on the same queries). The development experiments that chose this pipeline are separate scripts:

| Order | Command | Reads | Writes |
|---|---|---|---|
| 1 | `.venv/bin/python scripts/stage2/pull_duplicate_events.py` (needs network and `gh`; responses are cached, so a rerun resumes) | `data/issues.jsonl.gz` | `data/duplicate_events.json`, `data/duplicate_events_meta.json`, raw cache `data/raw/duplicate_events/` |
| 2 | `.venv/bin/python scripts/stage2/dedup_clean.py` | issues, sets, duplicate events | `outputs/stage2/duplicate_clean_dev.json`; cache `data/stage2/embeddings_clean.npy` (+ `_meta.json`) |
| 3 | `.venv/bin/python scripts/stage2/dedup_crossencoder.py` | the same, plus the cleaned-text embedding cache from step 2 | `outputs/stage2/duplicate_crossencoder_dev.json`; caches `data/stage2/crossencoder_<model>_<split>.npy` |

Step 1 provides the creation time of every duplicate link, which the sibling rule needs. Step 3 reuses the cleaned-text embeddings that step 2 caches (from the code, it calls the same `embed_clean` function, so it would recompute them if the cache were missing; this is an inference, not tested). Step 3 scores both candidate cross-encoders on train-period and tune-period queries only. Runtime on an Apple MPS device: embedding the 107,422 cleaned issues took 253.9 s; the chosen cross-encoder took 1,908.0 s for 864 train-period queries and 746.9 s for 341 tune-period queries, about 2.2 s per query.

**Scores on the test set** (268 queries created 2026-07 to 2026-09, run once on 2026-10-10; `outputs/stage2/duplicate_final_test.json`). Recall@10 under the sibling rule [95% CI]; the stage-1 baseline (embedding retrieval on raw text) is rescored with the same rule on the same queries, with no change to the method:

| Original created before the query by | Queries | Stage-1 baseline | Final pipeline | Difference [95% CI] |
|---|---|---|---|---|
| All | 268 | 45.9% [39.9, 51.9] | **61.2%** [55.2, 66.8] | +15.3 [+9.7, +20.9] |
| At most 7 days | 114 | 43.0% | 55.3% | +12.3 [+5.3, +20.2] |
| 8 to 30 days | 44 | 50.0% | 63.6% | +13.6 [+0.0, +27.3] |
| More than 30 days | 110 | 47.3% | 66.4% | +19.1 [+9.1, +28.2] |

Recall@1 rises from 22.4% to 29.5% (+7.1 [+1.5, +12.3]) and recall@5 from 40.7% to 53.7% (+13.1 [+7.1, +19.0]). Counting the original only (no siblings), recall@10 is 38.8% for the stage-1 baseline, the same as the stage-1 report, and 51.5% for the final pipeline. The cross-encoder took 740.7 s for the 268 queries.

**Scores on development data** (sibling rule).

Tune-period queries (2026-04 to 2026-06, 341 queries), recall@10 [95% CI], from `outputs/stage2/duplicate_crossencoder_dev.json`:

| Original created before the query by | Queries | Embedding top 10 (cleaned) | + cross-encoder rerank | Difference [95% CI] |
|---|---|---|---|---|
| All | 341 | 48.1% [42.8, 53.4] | **56.0%** [50.7, 61.3] | +7.9 [+3.8, +12.0] |
| At most 7 days | 208 | 48.1% | 55.3% | +7.2 [+2.4, +12.5] |
| 8 to 30 days | 54 | 40.7% | 48.1% | +7.4 [-3.7, +18.5] |
| More than 30 days | 79 | 53.2% | 63.3% | +10.1 [+1.3, +19.0] |

On the 864 train-period queries (used to choose the model), recall@10 was 55.2% for this cross-encoder, 51.4% for embedding retrieval alone, and 34.0% for `cross-encoder/quora-distilroberta-base`.

Development and test numbers come from different queries and are not directly comparable; compare methods within one table.

**Known limits.** The labels miss many real duplicates: an AI-reviewed (Claude Opus 5.5, not human-verified) audit found that most high-ranked "wrong" candidates for recent originals report the same problem, so all recall figures are likely underestimates (see EXPERIMENTS.md). Only links with a known creation time (2,289 of 2,304) can make a sibling count.

## Reproducing stage 1

- `./run_data.sh --offline` rebuilds the stage-1 data from the cached raw pulls.
- `./run_baselines.sh` reruns every stage-1 baseline result and writes the report `docs/baselines.html`; it reproduces all outputs byte for byte.
- Stage-1 records: `docs/data_pipeline.html` (data pipeline, Chinese) and `docs/baselines.html` (baseline results, English).
