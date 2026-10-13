# Usher

Usher triages new issues for the [`microsoft/vscode`](https://github.com/microsoft/vscode) GitHub repository. For each new issue it makes three judgments:

- **Type**: which of vscode's 10 type labels the issue gets (for example `bug`, `feature-request`).
- **Module**: which code-area label the issue gets (one of 291 keys in vscode's `.github/classifier.json`).
- **Duplicate**: which earlier issue it duplicates, as a ranked list of 10 candidates.

It is evaluated by replaying history: each issue is judged using only information that existed when it was created, and the answers are the labels and duplicate links that vscode maintainers later set.

## Results

Test set: issues created 2026-07 to 2026-09, scored once for each frozen pipeline. Brackets are 95% bootstrap confidence intervals.

| Judgment | Metric | Baseline | Current pipeline | Gain |
|---|---|---|---|---|
| Type | accuracy | 66.2% (always `bug`) | 79.4% [77.5, 81.2] | +13.2 |
| Type | macro-F1 | 0.080 (always `bug`) | 0.210 [0.192, 0.252] | +0.130 |
| Module | top-1 hit rate | 48.2% (logistic regression) | **60.8%** [57.2, 64.5] | +12.7 [+9.3, +16.2] |
| Duplicate | recall@10 | 45.9% (embedding retrieval) | **61.2%** [55.2, 66.8] | +15.3 [+9.7, +20.9] |

Test sizes: type 1,848 issues, module 687 issues, duplicate 268 queries. Type has no LLM version yet; its current pipeline is the logistic regression, with two settings (accuracy and macro-F1 rows come from different settings, see [docs/pipeline.md](docs/pipeline.md#type-tf-idf--logistic-regression)).

## How each judgment works

### Type

TF-IDF features of title and body, multinomial logistic regression. About two thirds of issues are `bug`, and four of the ten types have at most 5 issues each in the test set, so macro-F1 is low and unstable.

### Module

1. A logistic regression on TF-IDF features proposes its 10 most probable modules.
2. Embedding search finds the 10 most similar earlier issues and their module labels.
3. `gpt-4o-mini` reads the issue, both references and the label descriptions, and returns one module label with a one-sentence reason. It uses no tools.

The LLM gains most on modules with few past issues (+16.8 points for modules with fewer than 20 training issues). It outputs a single label; there is no top-3 list or confidence score yet.

### Duplicate

1. **Clean the text**: remove issue-template comments, system-information blocks, version lines and images, which otherwise fill the embedding model's 512-token window.
2. **Retrieve**: embed with `BAAI/bge-small-en-v1.5` and take the 50 most similar earlier issues.
3. **Rerank**: score each (issue, candidate) pair with the cross-encoder `Alibaba-NLP/gte-reranker-modernbert-base` and keep the top 10.

No LLM is used: rewriting the query with an LLM was tried and made results worse for old originals. The reranking takes about 2.2 s per query on an Apple MPS device.

The labelled answers are incomplete. An AI-assisted audit (not human-verified) found that many top-ranked "wrong" candidates report the same problem, so recall is likely underestimated.

Exact settings for all three pipelines (model revisions, cleaning rules, prompt contents) are in [docs/pipeline.md](docs/pipeline.md).

## Evaluation

- **Data**: 107,422 `microsoft/vscode` issues created 2023-01-01 to 2026-09-30, pulled on 2026-10-05.
- **Split by creation time**: train 2025-01 to 2026-03, tune 2026-04 to 2026-06, test 2026-07 to 2026-09. All method choices are made on train and tune; each frozen pipeline is run on test once.
- **Ground truth**: maintainers' final labels and `duplicateOf` links. A 30-issue test sample per judgment was checked by an AI model (not human-verified) and found almost entirely correct (`review/ai_review_summary.json`).
- **Duplicate sibling rule**: a query also counts as found if the top 10 contains another issue that had already been linked as a duplicate of the same original before the query was created. Duplicate results are also reported by the age of the original (at most 7 days, 8 to 30 days, over 30 days); no adopted method may be significantly worse on originals older than 30 days.
- **Statistics**: 95% percentile bootstrap intervals (10,000 resamples, fixed seed); method comparisons use paired bootstrap, with the adoption rule fixed before each run.

The full list of rules is in [EXPERIMENTS.md](EXPERIMENTS.md#evaluation-rules-in-force).

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/), an authenticated [`gh`](https://cli.github.com/) CLI for pulling data, and an OpenAI API key for the module pipeline.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
echo "OPENAI_API_KEY=..." > .env    # git-ignored; only needed for the module pipeline

./run_data.sh                 # pull the issue snapshot from GitHub and build the data sets (resumable)
./run_baselines.sh            # baselines and the type pipeline; writes docs/baselines.html
./run_stage2_duplicate.sh     # duplicate pipeline on the test set
./run_stage2_module.sh        # module pipeline on the test set (calls the OpenAI API)
```

Data is written to the git-ignored `data/` directory. `./run_data.sh --offline` rebuilds the sets from already pulled raw data. `manifest/` holds the issue numbers, content hashes and per-split issue lists of the snapshot, so a new pull can be checked against it.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/pipeline/` | Data pull, filtering and splitting |
| `scripts/baselines/` | Baselines and the type pipeline |
| `scripts/stage2/` | Module and duplicate pipelines and their development experiments |
| `scripts/from_poc/` | Scripts from an earlier proof of concept, partly reused |
| `outputs/` | Results of every method run (JSON) |
| `analysis/` | Data analyses, each with its script and outputs |
| `manifest/` | Snapshot description and checksums |
| `review/` | AI review of ground-truth samples |
| `docs/` | Pipeline details, data pipeline record, baseline report |

## History

[EXPERIMENTS.md](EXPERIMENTS.md) records every experiment and decision in order, including the methods that were tried and not adopted.
