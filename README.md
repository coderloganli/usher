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
| Type | accuracy | 85.8% (logistic regression) | **89.7%** [87.1, 92.3] | +4.0 [+1.8, +6.3] |
| Module | top-1 hit rate | 48.2% (logistic regression) | **60.8%** [57.2, 64.5] | +12.7 [+9.3, +16.2] |
| Duplicate | recall@10 | 45.9% (embedding retrieval) | **61.2%** [55.2, 66.8] | +15.3 [+9.7, +20.9] |

Test sizes: type 556 issues, module 687 issues, duplicate 268 queries. Type is scored only on issues opened by people outside the vscode team, since team members often label their own issues when opening them (see [Evaluation](#evaluation)); on these issues, simply taking the reporter's own choice of bug or feature template (`bug` when there is none) scores 82.4%. Type macro-F1 is not in the table: with eight of the ten types having 15 or fewer test issues it is too unstable to compare methods (LLM 0.242 [0.213, 0.332], no significant difference from either logistic regression; details in [EXPERIMENTS.md](EXPERIMENTS.md)).

## How each judgment works

### Type

1. Read the reporter's own choice (bug or feature template) from the issue body, if there is one.
2. A logistic regression on TF-IDF features gives the probability of each of the 10 types.
3. Embedding search finds the 10 most similar earlier issues and their types.
4. `gpt-4o-mini` reads the issue, the reporter's choice, both references and the type definitions from vscode's triage guide, and returns one type with a one-sentence reason. It uses no tools.

The gain comes from the two large types: more feature requests are recognised (`feature-request` recall 77% → 89%) and fewer issues are wrongly called bugs (`bug` precision 84% → 89%). The rare types (`*question`, `polish`, `upstream` and others) are still almost never found.

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
- **Ground truth**: maintainers' final labels and `duplicateOf` links. Type labels exist only for issues the team accepted for triage, so type results say nothing about issues that were closed or never labelled. A 30-issue test sample per judgment was checked by an AI model (not human-verified) and found almost entirely correct (`review/ai_review_summary.json`).
- **Type subset**: type is scored on issues whose author is not a team member, approximated as an account that is an assignee on some snapshot issue (107 accounts).
- **Duplicate sibling rule**: a query also counts as found if the top 10 contains another issue that had already been linked as a duplicate of the same original before the query was created. Duplicate results are also reported by the age of the original (at most 7 days, 8 to 30 days, over 30 days); no adopted method may be significantly worse on originals older than 30 days.
- **Statistics**: 95% percentile bootstrap intervals (10,000 resamples, fixed seed); method comparisons use paired bootstrap, with the adoption rule fixed before each run.

The full list of rules is in [EXPERIMENTS.md](EXPERIMENTS.md#evaluation-rules-in-force).

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/), an authenticated [`gh`](https://cli.github.com/) CLI for pulling data, and an OpenAI API key for the module and type pipelines.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
echo "OPENAI_API_KEY=..." > .env    # git-ignored; only needed for the module and type pipelines

./run_data.sh                 # pull the issue snapshot from GitHub and build the data sets (resumable)
./run_baselines.sh            # stage-1 baselines; writes docs/baselines.html
./run_stage2_duplicate.sh     # duplicate pipeline on the test set
./run_stage2_module.sh        # module pipeline on the test set (calls the OpenAI API)
./run_stage2_type.sh          # type pipeline on the test set (calls the OpenAI API)
```

Data is written to the git-ignored `data/` directory. `./run_data.sh --offline` rebuilds the sets from already pulled raw data. `manifest/` holds the issue numbers, content hashes and per-split issue lists of the snapshot, so a new pull can be checked against it.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/pipeline/` | Data pull, filtering and splitting |
| `scripts/baselines/` | Stage-1 baselines |
| `scripts/stage2/` | Type, module and duplicate pipelines and their development experiments |
| `scripts/from_poc/` | Scripts from an earlier proof of concept, partly reused |
| `outputs/` | Results of every method run (JSON) |
| `analysis/` | Data analyses, each with its script and outputs |
| `manifest/` | Snapshot description and checksums |
| `review/` | AI review of ground-truth samples |
| `docs/` | Pipeline details, data pipeline record, baseline report |

## History

[EXPERIMENTS.md](EXPERIMENTS.md) records every experiment and decision in order, including the methods that were tried and not adopted.
