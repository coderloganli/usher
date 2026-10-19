# Usher: experiment record

This is the chronological record of the experiments and decisions that led to the pipelines in [README.md](README.md) (overview and results) and [docs/pipeline.md](docs/pipeline.md) (exact settings). Each entry gives the date, the question, what was done (script and commit), the data used, the result numbers, the decision and the reason.

Conventions used throughout:

- "Train", "tune" and "test" refer to issues created 2025-01 to 2026-03, 2026-04 to 2026-06 and 2026-07 to 2026-09 (after the re-split of 2026-10-07; see entry 4).
- "Development queries" are the 1,205 duplicate queries from the train and tune periods (864 + 341). Stage-2 work uses only these. The test set is used once per frozen pipeline: entries 7 (stage-1 baselines), 20 (duplicate) and 26 (module).
- Intervals are 95% bootstrap intervals with 10,000 resamples and a fixed seed; differences are paired. "Points" are percentage points.
- Ground truth is the maintainers' final labels and `duplicateOf` links. Where a check of it is mentioned, it was reviewed by an AI model (Claude Opus 5.5), not human-verified.
- Dates are when the decision was made according to the lead session's decision notes; commit dates are given with each commit. Where they differ, both are stated.

## Evaluation rules in force

| Rule | Decided |
|---|---|
| Replay: a duplicate query is matched only against snapshot issues (from 2023-01) created strictly before it, including bot-authored and unlabeled issues; bot-authored issues are not queries. BM25 statistics come from that pool only. | 2026-10-07 (Part B plan) |
| Develop only on train/tune (for duplicates: the 1,205 development queries); freeze the method and settings; then use the test set once. No change after seeing test results. | 2026-10-07 (Part B plan) |
| Every score has a 95% percentile bootstrap interval with 10,000 resamples and a fixed seed (20261007 in `outputs/baselines/test_metrics.json`); differences use paired bootstrap on the same resamples. The resample count was raised from the spec's 1,000. | 2026-10-07 (Part B plan; recorded as a spec revision dated 2026-10-08) |
| Primary metrics: type macro-F1 (against the LR selected by macro-F1) and accuracy (against the LR selected by accuracy); module top-1 hit rate; duplicate recall@10 (recall@1 and recall@5 also reported). | 2026-10-07 |
| Ties in duplicate ranking are pessimistic: the true original is ranked last among candidates with an equal score. | 2026-10-07 |
| Adoption of a new method or variant only by a decision rule fixed before the run, based on a paired bootstrap interval of the difference lying entirely above 0. | 2026-10-07 (first used for RRF in entry 6) |
| Sibling rule: a duplicate query counts as found at k if the original, or a sibling whose duplicate link to the same original was created before the query, is among the first k candidates. | 2026-10-08 (timed version; see entries 10 and 11) |
| Age-group reporting: every duplicate method reports recall@10 overall and for originals created at most 7 days, 8 to 30 days and more than 30 days before the query (development queries 640 / 216 / 349; test queries 114 / 44 / 110). No new method may be significantly worse (paired bootstrap) on the more-than-30-days group. LLM methods are judged by age group too. | 2026-10-09 |
| Model choice for a component with candidate models: choose on the 864 train-period development queries, then validate the chosen one against the current method on the 341 tune-period queries with a pre-set rule. | 2026-10-09 (cross-encoder protocol, entry 18) |
| Ground truth is always described as reviewed by an AI model (Claude Opus 5.5), not human-verified. | 2026-10-07 |
| Every protocol that chooses among candidates states its tie-break in advance (default: the cheaper option, then the simpler one). Added after an unplanned tie in entry 19. | 2026-10-10 |

The age-group counts above were checked against `data/sets.json` and `data/issues.jsonl.gz` (train 432 / 162 / 270, tune 208 / 54 / 79, test 114 / 44 / 110).

## Stage 1: data and baselines

### 1. Data snapshot and filtering (2026-10-04 to 2026-10-06)

- **Question.** Can a clean, reproducible dataset for the judgments be built from public `microsoft/vscode` issues?
- **What was done.** GraphQL bulk pull of every issue created 2023-01-01 to 2026-09-30, filters, time split, snapshot manifest (`scripts/pipeline/`, `run_data.sh`; commits `ee23cb6`, `215c0cb`, `5b8e74d`, `f1293cd`, `0778f3f`, `eb6be7d`). Bots are detected by login and by GitHub account type; empty-body issues are dropped (both added 2026-10-06).
- **Data.** 107,422 issues pulled on 2026-10-05.
- **Result.** Filtered sets per judgment; counts in `manifest/counts.json`.
- **Decision.** Adopted as the stage-1 dataset.

### 2. AI review of the ground truth (2026-10-06 to 2026-10-07)

- **Question.** Is the ground truth correct often enough (spec standard: at least 85%)?
- **What was done.** 30 random test issues per judgment were judged from public content by an AI model (Claude Opus 5.5), not human-verified (`scripts/pipeline/ai_review_summary.py`; commits `f443647`, `5c5ac77`). The review was originally planned as a human review and was changed to an AI review on 2026-10-07.
- **Result** (`review/ai_review_summary.json`). Type: 28 right, 0 wrong, 2 cannot judge (accuracy 100%, interval 87.9% to 100%). Module: 30 right of 30 (interval 88.6% to 100%). Duplicate: 27 right, 1 wrong, 2 cannot judge (accuracy 96.4%, interval 82.3% to 99.4%; the lower bound is below 85%).
- **Decision.** All three judgments meet the 85% standard on the point estimate; the duplicate interval caveat is reported.

### 3. Assignee judgment replaced by module (2026-10-07)

- **Question.** Can the assignee ground truth be checked from public information?
- **What was done.** For every assignee test issue, a check of whether any final assignee acted on it (`scripts/pipeline/assignee_check.py`, commit `f443647`); module judgment added (`scripts/pipeline/module_labels.py`, `build_sets.py`, commit `5c5ac77`).
- **Result.** In 3,166 of the 6,569 assignee test issues (48.2%), no final assignee acted on the issue at all (`manifest/assignee_check.json`; the spec revision of 2026-10-07 gives the same 48.2%).
- **Decision.** Assignee discontinued; replaced by module (291 module labels from `classifier.json` keys that occur on at least one pulled issue). Reason: who should own an issue depends on internal team allocation and cannot be checked publicly.

### 4. Re-split: tune widened to three months (2026-10-07)

- **Question.** Is a one-month tune set (2026-06) large enough to choose settings?
- **What was done.** Train shortened to 2025-01 to 2026-03, tune widened to 2026-04 to 2026-06, test unchanged (commit `a8828d4`, 2026-10-07). Train starts 2025-01 by the user's choice (newer data), not by tuning.
- **Result.** Counts (train / tune / test): type 13,554 / 2,165 / 1,848; module 8,057 / 986 / 687; duplicate 864 / 341 / 268. Test sets, the AI review sample and raw checksums were unchanged. Before the change the June tune set had only 621 type and 255 module issues (spec revision note).
- **Decision.** Adopted.

### 5. Type and module baselines tuned on the tune set (2026-10-07)

- **Question.** Which settings of the TF-IDF + logistic regression baselines should be frozen?
- **What was done.** `scripts/baselines/classify.py` (commit `7ba4c44`): TF-IDF with scikit-learn defaults; grid C in {0.01, 0.1, 1, 10, 100} x class_weight in {none, balanced}; fit on train, score on tune. Outputs `outputs/baselines/type_tuning.json`, `module_tuning.json`.
- **Data.** Train and tune only.
- **Result.** Type: majority (`bug`) tune accuracy 63.8%, macro-F1 0.087. Best by macro-F1: C = 1, balanced (macro-F1 0.352, accuracy 64.3%). Best by accuracy: C = 1, no weighting (macro-F1 0.248, accuracy 80.2%). Module: majority top-1 12.3%, top-3 22.9%; chosen C = 10, balanced (top-1 57.1%, top-3 76.8%).
- **Decision.** Adopted. Because the two type metrics chose different settings, two type LR baselines are kept (a revision of the plan made the same day).

### 6. Duplicate retrievers on development queries, raw text (2026-10-07)

- **Question.** BM25 or embedding retrieval, and is their RRF fusion better than the better one?
- **What was done.** `scripts/baselines/dedup.py` (commit `3568728`): BM25 (lowercase word tokens, k1 = 1.2, b = 0.75); `BAAI/bge-small-en-v1.5` (pinned revision, no instruction, 512-token truncation, cosine); RRF with k = 60. Rule fixed in advance: keep RRF only if RRF minus the better single method on recall@10 has an interval entirely above 0. Output `outputs/baselines/duplicate_dev.json`.
- **Data.** 1,205 development queries; title + raw body; scored against the original only (the sibling rule did not exist yet).
- **Result.** Recall@10: BM25 25.2%, embedding 35.1%, RRF 36.2%. Embedding minus BM25 +9.9 points [+7.4, +12.5]. RRF minus embedding +1.1 points [-1.1, +3.2].
- **Decision.** RRF not adopted (fails the pre-set rule). BM25 and embedding frozen.

### 7. One-shot test evaluation of the frozen baselines (2026-10-08)

- **Question.** How good are the frozen baselines on the test set? These scores are the targets for stage 2.
- **What was done.** `scripts/baselines/evaluate_test.py` (commit `4fa7b4c`); final models trained on train only. Output `outputs/baselines/test_metrics.json`.
- **Data.** Test set (2026-07 to 2026-09), used once.
- **Result.** Type (1,848 issues): majority accuracy 66.2%, macro-F1 0.080; LR (macro-F1) macro-F1 0.210 [0.192, 0.252], accuracy 63.5%; LR (accuracy) accuracy 79.4% [77.5, 81.2], macro-F1 0.203. The two LRs' macro-F1 difference is +0.008 [-0.058, +0.039], not significant; the tune-set gap disappeared. Module (687 issues): LR top-1 48.2% [44.4, 52.0], top-3 68.0%; majority top-1 15.3%, top-3 17.6%. Duplicate (268 queries, original only, raw text): embedding recall@10 38.8% [33.2, 44.4], BM25 19.0%.
- **Decision.** Stage-2 targets set: type macro-F1 0.210 and accuracy 79.4%; module top-1 48.2%; duplicate recall@10 38.8%.

### 8. Stage-1 report and publication decision (2026-10-08)

- **What was done.** `scripts/baselines/report.py` writes `docs/baselines.html` from the outputs (commit `e144b0f`); `./run_baselines.sh` reproduces every output and the report byte for byte.
- **Decision.** Stage 1 complete. It is not published, because baselines alone are not worth publishing. If published later: a snapshot of the current files only, no git history, no HTML files, centred on a README.

## Stage 2: duplicate detection (no LLM used yet)

### 9. Which judgment first (2026-10-08)

- **Question.** Where should stage 2 start?
- **Decision.** Duplicate detection, raising recall@10 first, because retrieval recall caps any later reranking. Stage-2 work stays on branch `stage1-data`.

### 10. Diagnosis of 40 development misses (2026-10-08; committed 2026-10-09)

- **Question.** Why does retrieval miss the original?
- **What was done.** `analysis/duplicate_miss_diagnosis/sample.py` drew 40 of the development queries that both stage-1 retrievers missed in the top 10 (raw text); an AI model (Claude Opus 5.5) read and categorised them, not human-verified; `summary.py` counts them (commit `086fb8f`, 2026-10-09).
- **Data.** 706 development queries missed by both retrievers (raw text, original only); 40 sampled.
- **Result** (`analysis/duplicate_miss_diagnosis/summary.json`). Categories of the 40: different symptoms with the same root cause 10; near miss (original ranked 11 to 100) 9; same symptom in very different wording 6; original is a maintainer's technical or tracking issue 5; a sibling duplicate retrieved instead 4; cross-language 2; little text 2; pasted HTML 1; spam 1. Embedding rank of the 706 missed originals: 142 at 11 to 50, 130 at 51 to 200, 434 beyond 200. Of the 706 misses, 211 have a sibling (judged from final-state `duplicateOf` links, link time not checked) in the embedding or BM25 top 10 (178 in the embedding top 10). The reading also found that templates and system-information tables use up the 512-token embedding window.
- **Decision.** Analysis only. It motivated the sibling rule (entry 11) and text cleaning (entry 13).

### 11. Count sibling duplicates as found; time the links (2026-10-08)

- **Question.** Should a candidate that is another duplicate of the same original count as a correct answer, and how to avoid using links that did not yet exist?
- **What was done.** First decision (2026-10-08): count siblings as found, using final-state links. Then `scripts/stage2/pull_duplicate_events.py` (commit `b948b01`, 2026-10-08) pulled the timeline events that create duplicate links (GitHub GraphQL `MarkedAsDuplicateEvent`, `UnmarkedAsDuplicateEvent`, and `ClosedEvent` with state reason `DUPLICATE`; fields verified by schema introspection on 2026-10-08). The pull was re-run on 2026-10-09 (2026-10-09T15:36Z to 15:37Z, `data/duplicate_events_meta.json`) to add who created each event (commit `70fac06`, entry 14); apart from the new actor fields the 2,667 events were identical to the first pull.
- **Result.** 2,289 of the 2,304 snapshot links get a creation time. Development recall@10 of the stage-1 retrievers (raw text): original only, embedding 35.1% and BM25 25.2%; with final-state siblings 51.5% and 38.0% (decision notes; not found in a result file); with timed siblings (the adopted rule) 44.1% and 32.3% (`outputs/stage2/duplicate_clean_dev.json`, "raw"). The decision notes give a median of 2.5 days from issue creation to link; this figure is not in a result file.
- **Decision.** Adopted: the sibling rule with link times (a sibling counts only if its link to the same original was created before the query). Baselines are re-scored with it on development data; the stage-1 test scores have not been re-scored.

### 12. Following duplicate-graph links as a retrieval method (2026-10-08)

- **Question.** Should retrieval follow `duplicateOf` links from retrieved candidates to their originals?
- **Decision.** Not adopted, not run. Under the sibling rule a retrieved sibling already counts as found, so following its link adds nothing to the metric.

### 13. Text cleaning (2026-10-08)

- **Question.** Does removing automatically added template and system content raise recall?
- **What was done.** `scripts/stage2/dedup_clean.py` (commit `4b3eccc`): cleaning rules as listed in `docs/pipeline.md`; same BM25 and embedding settings as stage 1. Rule fixed in advance: adopt cleaning for a method only if cleaned minus raw recall@10 has an interval entirely above 0. Output `outputs/stage2/duplicate_clean_dev.json`.
- **Data.** 1,205 development queries, sibling rule.
- **Result.** Recall@10, raw to cleaned: embedding 44.1% to 50.5% (+6.4 [+4.4, +8.5]); BM25 32.3% to 43.5% (+11.2 [+9.0, +13.5]). Recall@1 and recall@5 also improve for both, with intervals above 0.
- **Decision.** Adopted for both. New comparison point: embedding retrieval on cleaned text, development recall@10 50.5%. Agreed order: non-LLM cleaning first, then LLM query rewriting on cleaned text, with the two gains reported separately (LLM rewriting has not been done).

### 14. Who creates duplicate links, and how many duplicates have links (2026-10-09)

- **Question.** Are duplicate links created by bots or by people, and how complete are they?
- **What was done.** Part 1 of `analysis/duplicate_recency_signals/signals.py` (commit `70fac06`, the link-maker record; script committed in `086fb8f`). The label count below was checked from `data/issues.jsonl.gz` for this record.
- **Result** (`analysis/duplicate_recency_signals/signals.json`). All 2,289 timed links were created by `User` accounts through a `closed` event (closing the issue as a duplicate); none by a bot or by a separate "marked as duplicate" event. 212 distinct accounts created them; the most active account created 331. An earlier report of 99 accounts was wrong (it was taken from one subgroup). Among issues created 2025-01 to 2026-09, 3,471 carry the `*duplicate` label but no link, while 2,168 have a link (2,052 of these without the label), so only about 38% of known duplicates have a usable link.
- **Decision.** Analysis only. The ground truth is human decisions and is incomplete.

### 15. Recency and popularity signals; metadata-reranking prototype (2026-10-09)

- **Question.** Do time and popularity signals predict the original, and does reranking by them help?
- **What was done.** Parts 2 to 4 of `analysis/duplicate_recency_signals/signals.py` (commit `086fb8f`): statistics on embedding retrieval on cleaned text, and a prototype logistic regression on cosine, log(1 + candidate age in days), log(1 + links already pointing at the candidate) and 1/rank, fitted on train-period queries and scored on tune-period queries (top 100).
- **Data.** Development queries, sibling rule.
- **Result.** In the top 100 of the 864 train-period queries, 75.8% of correct candidates (1,189 of 1,569) were created at most 7 days before the query. On the 341 tune-period queries, recall@10 rose from 48.1% to 62.8% (+14.7 [+10.3, +19.1]). By original age (the prototype's own grouping: 205 / 57 / 79 queries): at most 7 days 47.3% to 69.3% (45 gained, 0 lost); 7 to 30 days 43.9% to 57.9%; more than 30 days 53.2% to 49.4% (4 gained, 7 lost).
- **Decision.** Not adopted as a method; kept only as an "imitate maintainers" contrast (possibly a two-column card in stage 4). Reason: all the gain is on recent originals and old originals get worse; the model learns maintainers' limited memory from biased labels (the user raised this concern, citing the idea of learning from biased demonstrations; a literature check found the closest established framings to be positive-unlabeled learning with selection bias and unbiased learning-to-rank with propensity weighting). Approved consequences: the age-group reporting rule (see rules above), and an audit of high-ranked "wrong" candidates (entry 16). The final decision on metadata reranking was deferred until LLM methods are evaluated with age groups.

### 16. Audit of high-ranked "wrong" candidates (2026-10-09)

- **Question.** How often is a high-ranked candidate that the labels call wrong actually a duplicate, for recent and for old candidates?
- **What was done.** `analysis/duplicate_label_audit/sample.py` drew, from embedding retrieval on cleaned text, top-3 candidates that are neither the original nor a timed sibling: 30 created at most 7 days before the query ("recent") and 30 more than 30 days before ("old"). An AI model (Claude Opus 5.5) judged each pair from public title and body, partially blind (shuffled order, group and dates hidden, issue numbers visible); not human-verified. `summary.py` counts them (commit `750da7c`).
- **Result** (`analysis/duplicate_label_audit/summary.json`). Recent: 25 of 30 report the same problem (83%, Wilson interval 66% to 93%). Old: 6 of 22 decidable (27%, interval 13% to 48%), 16 different, 8 cannot tell.
- **Decision.** Analysis only. The labels miss many duplicates in both groups, so recall is underestimated. The idea that maintainers miss old duplicates more is not supported by this audit.

### 17. RRF fusion on cleaned text (2026-10-09)

- **Question.** With cleaned text, does RRF of BM25 and embedding beat embedding alone?
- **What was done.** `scripts/stage2/dedup_fusion.py` (commit `af7777a`): RRF k = 60 on cleaned text. Rule fixed in advance: adopt only if RRF minus embedding recall@10 has an interval entirely above 0 and the more-than-30-days interval is not entirely below 0. Output `outputs/stage2/duplicate_fusion_dev.json`.
- **Data.** 1,205 development queries, sibling rule.
- **Result.** Recall@10: RRF 52.8% vs embedding 50.5%, +2.3 [+0.2, +4.5]. At most 7 days (640 queries): +4.5 [+1.7, +7.2]. 8 to 30 days (216): +0.5 [-5.1, +6.0]. More than 30 days (349): recall@10 -0.6 [-4.6, +3.4]; recall@5 -3.7 [-8.0, +0.3]. The output file records `adopt_rrf: true`, meaning the pre-set rule was passed.
- **Decision.** Not adopted, by the user's decision on 2026-10-09 despite passing the rule. Reasons: the gain is small and confined to recent originals; it needs an extra BM25 index to maintain; and its rank-only scores are useless as confidence for stage 3. Main line stays embedding retrieval on cleaned text (development recall@10 50.5%).

### 18. Local cross-encoder rerank of the embedding top 50 (2026-10-09 to 2026-10-10)

- **Question.** Does a local cross-encoder reranking the embedding top 50 raise recall@10? (The user moved this before LLM reranking.)
- **What was done.** `scripts/stage2/dedup_crossencoder.py` (commit `542393d`, 2026-10-10). Candidates: embedding retrieval on cleaned text, top 50. Models chosen by the user: `cross-encoder/quora-distilroberta-base` (82M parameters, trained on Quora duplicate questions, 254 tokens per side) and `Alibaba-NLP/gte-reranker-modernbert-base` (149M parameters, 1,024 tokens per side), both at pinned revisions. Protocol fixed in advance: choose the model on train-period queries by recall@10 (tie to the smaller model); validate on tune-period queries, adopting only if the recall@10 difference interval is entirely above 0 and the more-than-30-days interval is not entirely below 0. Output `outputs/stage2/duplicate_crossencoder_dev.json`.
- **Data.** 864 train-period queries for the choice; 341 tune-period queries for validation; sibling rule.
- **Result.** Choice (train period), recall@10: gte-reranker-modernbert 55.2%, embedding alone 51.4%, quora-distilroberta 34.0%. Validation (tune period), recall@10: embedding 48.1% to reranked 56.0%, +7.9 [+3.8, +12.0]; at most 7 days (208) +7.2 [+2.4, +12.5]; 8 to 30 days (54) +7.4 [-3.7, +18.5]; more than 30 days (79) 53.2% to 63.3%, +10.1 [+1.3, +19.0]. Runtime on an Apple MPS device: about 2.2 s per query (1,908.0 s for 864 queries, 746.9 s for 341).
- **Decision.** Adopted. It passes both conditions, and unlike entries 15 and 17 its largest gain is on old originals. New comparison point for LLM methods: embedding top 50 + cross-encoder rerank, tune-period recall@10 56.0%. Its test score has not been measured.

### 19. LLM query rewriting before retrieval (2026-10-10)

- **Question.** Does rewriting the query with an LLM, used only for the retrieval step, raise recall@10 of the current best pipeline (embedding top 50 + cross-encoder)?
- **What was done.** `scripts/stage2/dedup_rewrite.py`. Model `gpt-4o-mini-2024-07-18` (Chat Completions, official `openai` package 3.28.0), temperature 0, fixed seed, at most 300 output tokens; query text = cleaned text cut to 4,000 tokens. Three styles: summary (a title and two or three sentences, errors copied exactly), keywords (10 to 20 identifying keywords and phrases, as in CUPID), hyde (a hypothetical earlier issue, at most 150 words, as in HyDE). Two uses each: the rewrite alone, or the rewrite followed by the cleaned text; 6 configurations. The rewrite is embedded with the same model to get the top 50; the cross-encoder then reads the query's cleaned text, not the rewrite. Protocol fixed in advance: choose the configuration with the highest final recall@10 on train-period queries; validate on tune-period queries against the current best; adopt only if the recall@10 difference interval is entirely above 0 and the more-than-30-days interval is not entirely below 0. Outputs `outputs/stage2/duplicate_rewrite_trial_cost.json`, `duplicate_rewrite_cost.json`, `duplicate_rewrite_retrieval_train.json`, `duplicate_rewrite_rerank.json`; the rewrites themselves stay in the git-ignored `data/stage2/rewrites/` because they paraphrase issue text.
- **Data.** 100-query trial, then all 1,205 development queries; sibling rule.
- **Cost.** 3,615 calls in total, US$0.43 at standard prices (trial US$0.036); 2 outputs reached the length limit. The 100 trial calls ran on 41 to 45 distinct `system_fingerprint` values per style, so repeating the calls would not reproduce the same text; only the cached responses make the results reproducible.
- **Result.** Retrieval only (864 train-period queries), recall@50: no rewrite 68.4%; best configuration (keywords + original) 70.4%; summary alone and keywords alone are below no rewrite. After cross-encoder reranking, recall@10: no rewrite 55.2%; summary alone 53.9%, summary + original 55.2%, keywords alone 53.4%, keywords + original 56.0%, hyde alone 55.9%, hyde + original 56.0%. Keywords + original and hyde + original tied; no tie-break had been set, and the code took the first (keywords), which is also the cheaper (US$0.13 vs US$0.17 for all queries). Validation (341 tune-period queries), recall@10: current best 56.0%, keywords + original 55.4%, difference -0.6 [-3.5, +2.3]; at most 7 days 0.0 [-3.8, +3.8]; 8 to 30 days +5.6 [-1.9, +13.0]; more than 30 days 63.3% to 57.0%, -6.3 [-12.7, -1.3].
- **Decision.** Not adopted (fails the rule; significantly worse on originals over 30 days old). Inference: rewriting turns the query into a generic description of the symptom and loses specific details such as exact error text and setting names, which matter more for older originals. Difference from CUPID: CUPID fed keywords to a word-matching method (REP); here they were only embedded; keywords with BM25 were not tried because BM25 is not in the main line. LLM reranking of the cross-encoder's list was set aside by the user (small remaining headroom; the user's view is that the LLM is more useful for classification).

### 20. One-time test of the final duplicate pipeline (2026-10-10)

- **Question.** How does the frozen stage-2 duplicate pipeline (cleaned text, embedding top 50, cross-encoder rerank) do on the test set, compared with the stage-1 baseline?
- **What was done.** `scripts/stage2/dedup_final.py`, run through `./run_stage2_duplicate.sh` once on the test split; first checked on tune-period queries, where it reproduced the earlier numbers exactly (final pipeline 56.0%, stage-1 baseline 41.6%, the latter matching the stage-1 development run restricted to those queries). The stage-1 baseline (embedding retrieval on raw text) was rescored with the sibling rule on the same queries; its method was not changed. Outputs `outputs/stage2/duplicate_final_tune.json`, `duplicate_final_test.json`.
- **Data.** 268 test queries (2026-07 to 2026-09), never used by any stage-2 experiment before.
- **Result.** Recall@10, sibling rule: stage-1 baseline 45.9% [39.9, 51.9], final pipeline 61.2% [55.2, 66.8], difference +15.3 [+9.7, +20.9]. By original age: at most 7 days (114) 43.0% to 55.3%, +12.3 [+5.3, +20.2]; 8 to 30 days (44) 50.0% to 63.6%, +13.6 [+0.0, +27.3]; more than 30 days (110) 47.3% to 66.4%, +19.1 [+9.1, +28.2]. Recall@1 +7.1 [+1.5, +12.3]; recall@5 +13.1 [+7.1, +19.0]. Original only: 38.8% (same as the stage-1 report) to 51.5%.
- **Decision.** Duplicate detection closed with this pipeline as the stage-2 result. As on development data, the largest gain is on originals over 30 days old.

## Stage 2: triage flow and module judgment

### 21. What the type label means in vscode's triage, and who applies it (2026-10-10)

- **Question.** Before building an LLM type classifier (the planned next step): is predicting the type useful, given that reporters choose a bug or feature template? Who applies type labels, when, and what do they change?
- **What was done.** `analysis/triage_signals/signals.py` (committed with this entry). Parts A and B use the snapshot (train and tune only); parts C and D query GitHub for samples of tune issues; part E reuses stage-1 data on vscode's own assignment bot; part F reads vscode's current `.github/classifier.json`. "Team member" is approximated by "account that appears as an assignee anywhere in the snapshot" (107 accounts), not a membership list. Also read on 2026-10-10: vscode's issue templates (`labels: ''`, so templates add no label), and the vscode wiki pages "Issues-Triaging" and "Automated-Issue-Triaging" (from the wiki's git repository).
- **Result** (`analysis/triage_signals/summary.json`).
  - Reporter's choice (A): of 15,719 train and tune type issues, 8,408 (53.5%) were opened by team members. Of the 7,311 opened by others, 5,914 carry the reporter's own choice (bug or feature template, or the Type field of VS Code's issue reporter); it matches the final type for 4,652 (78.7%). Most common mismatches: chose bug, final `*question` 337; chose bug, final `feature-request` 324; chose bug, final `under-discussion` 145.
  - Coverage (B): of non-bot, non-empty issues created 2025-01 to 2026-06, outside issues got a type label in 7,524 of 43,696 cases (17%), team issues in 8,858 of 13,961 (63%). Outside issues without one most often carry `info-needed` (13,119), no label (12,663), `triage-needed` (7,425), `*duplicate` (2,642) or `spam` (1,977). The type ground truth therefore covers only issues the team accepted.
  - Who labels (C, 60 + 60 tune issues): on team issues, 36 type labels were applied by the author at creation (median 0 hours) and 24 by someone else; on outside issues, 59 by another person and 1 by a bot, median 42.5 hours after creation, 24 of 60 within 24 hours.
  - What follows (D, outside tune issues): of 80 `feature-request` issues, 43 went to the `Backlog Candidates` milestone and were closed as not planned; of 80 `bug` issues, 30 were closed as completed in an iteration milestone and 28 are open without a milestone; 15 of 20 `*question` issues were closed as not planned without a milestone.
  - Wiki: the documented flow is: owner assigned (by the inbox tracker or a bot) → owner applies a type and a feature-area label → milestone. "Each issue must have a type label." Feature requests go through community voting (`Backlog Candidates`, 20 up-votes within 60 days, else closed by a bot); `*question` and `upstream` are closing reasons. Triage "usually takes around one business day".
  - Owner assignment (wiki, E, F): every half hour a bot runs two ML models on new issues: one predicts the feature area and assigns its owner from `classifier.json`, the other predicts the owner directly; below a confidence threshold (default 0.75) the issue is left for the inbox tracker. On the 6,569 stage-1 assignee test issues, the bot made the first assignment on 4,245 (64.6%), and its assignee was still among the final assignees on 3,371 of those (79.4%). The current `classifier.json` has 311 labels and 34 owners, a median of 6 areas per owner (maximum 29; 8 owners have one area), so the bot's module cannot be recovered from its assignee. `.github/similarity.yml` shows an experimental duplicate-detection bot that comments possible duplicates on outside issues.
- **Decision.** The user agreed that type prediction has a real use (routing to the bug, feature-request or closing flow, and correcting the reporter's choice), with two points to settle when type work starts: evaluating only issues opened by non-team reporters with the reporter's choice as an extra baseline, and stating that the ground truth covers accepted issues only. The module judgment was taken first; the user proposed an agent-centred design (an agent that can search the code base makes the final judgment, with fixed models as support).

### 22. Pilot: an LLM agent that searches the code base for the module (2026-10-10)

- **Question.** Can a gpt-4o-mini agent that searches the vscode code base judge an issue's module better than without tools? Pilot budget set by the user: US$0.10.
- **What was done.** `analysis/agent_module_pilot/pilot.py` (committed with this entry). 20 tune issues: 15 random issues whose stage-1 LR top-1 is wrong and 5 whose LR top-1 is right (`sample.json`). Code base: vscode working tree at the last main commit before 2026-04-01 (`9ed1e07`), earlier than every tune issue; the `.git` directory was off limits. `plain`: issue text, LR top 10 with probabilities and GitHub label descriptions, and all 291 label names. `agent`: the same plus read-only tools `search_code` (ripgrep), `read_file` and `list_dir`, at most 8 calls. After the first run (user agreed): `build_path_map.py` built a "code path → module" table from train-period issues whose closing pull requests or commits were dated before 2026-04-01 (2,004 of 8,057 issues usable; 3,120 paths; 184 module labels), and `agent2` added a `modules_for_path` tool, an explicit step order and no `answer` tool in the first 2 turns.
- **Result** (`analysis/agent_module_pilot/summary.json`). Hits out of 20: plain 8, agent 9, agent2 7 (on the 15 LR-wrong issues: 4, 4, 3). Cost per issue: US$0.00036, 0.00092, 0.00126. In `agent`, 11 of 20 issues got no tool call at all. Reading the traces (AI-reviewed, not human-verified): the agent searched generic words, invented paths, did only the minimum number of calls, and once found the decisive file (`src/vs/platform/browserView/…`) but answered `config` instead of `browser-integration`; `agent2` fixed that case through the table but lost others when its lookups of non-existent paths fell back to the repository root (a design flaw of the tool). Total pilot cost US$0.051.
- **Decision.** Not pursued in this form. The user decided not to test a stronger model ("no more branches") and to follow the plan: check the code evidence without an LLM first (entry 23).

### 23. How much module signal the code base carries, without an LLM (2026-10-10)

- **Question.** Taking the weak agent out of the loop: does code-location evidence, found by a fixed program, predict the module, and does it complement LR?
- **What was done.** `analysis/module_repo_evidence/evidence.py` (committed with this entry), all 986 tune issues, settings fixed before the run: dotted names, back-ticked and quoted text from the cleaned issue (at most 15 per issue); fixed-string ripgrep over `src/`, `extensions/` and `cli/` of the same snapshot, tests and localisation excluded, identifiers found in more than 20 files dropped; files mapped to modules with the table from entry 22 (up to two parent levels, never above three path parts); one fixed combination rule (evidence top-1 when available, else LR top-1).
- **Result** (`analysis/module_repo_evidence/summary.json`). Evidence exists for 340 of 986 issues. On those, top-1: evidence 36.5% vs LR 56.2%, difference -19.7 [-25.9, -13.5]; top-3 48.2% vs 76.8%. Only the evidence is right on 30, only LR on 97. Combination rule on all 986: 50.3% vs LR 57.1%, -6.8 [-9.0, -4.6]. Even a perfect chooser between the two top-1s would reach only 60.1%. Main reasons seen in errors: module labels are finer than code folders (several `terminal-*` modules share files); frequently fixed modules dominate the table; many identifiers are not specific.
- **Decision.** Code-search tools are not part of the module pipeline. Interpretation agreed in discussion: LR stores the team's labelling conventions implicitly (its strongest words include an issue number of a tracking issue and a frequent reporter's user name), which the code base does not contain.

### 24. Similar-issue vote as reference material (2026-10-10)

- **Question.** Step 1 of the agreed plan "the LLM makes the final judgment with reference material": how good, and how complementary to LR, is a vote of similar labelled issues?
- **What was done.** `analysis/module_knn/knn.py` (committed with this entry): for each tune issue, the 10 most similar train issues by bge-small cosine on cleaned text (stage-2 cache); each neighbour adds its cosine to each of its labels. Settings fixed before the run.
- **Result** (`analysis/module_knn/summary.json`, 986 tune issues). Top-1: vote 59.1% vs LR 57.1%, +2.0 [-1.2, +5.3]; top-3 77.1% vs 76.8%. Only the vote right on 138, only LR on 118; a perfect chooser between the two top-1s would reach 71.1%. By how many train issues carry the true module: 1 to 19 (112 issues) vote 25.0%, LR 25.9%; 20 to 99 (236) 47.0%, 41.5%; 100 or more (638) 69.6%, 68.3%.
- **Decision.** Similar issues go into the LLM's reference material together with LR. The user asked that the agent make the final judgment and that LR serve as reference (an established pattern: SuperICL, Xu et al. 2023, gives a small model's predictions and confidence to an LLM; Infer-Retrieve-Rank, 2024, lets an LLM rerank retrieved candidate labels; checked from abstracts and search summaries, not full texts). Code tools were left out of the first version, because entries 22 and 23 showed little value and adding one ingredient at a time keeps each gain measurable.

### 25. LLM final module judgment with reference material, tune set (2026-10-10)

- **Question.** Does gpt-4o-mini, given LR's candidates and similar issues, judge the module better than LR?
- **What was done.** `scripts/stage2/module_llm.py` (committed with this entry). Per issue the model sees: cleaned issue text cut to 1,500 tokens; LR top 10 with probabilities; the 10 most similar train issues (created before the issue) with titles, labels and similarity; GitHub descriptions of all candidate modules (`scripts/stage2/module_label_descriptions.json`, 205 of 291 labels have one); all 291 label names. Structured output: a one-sentence reason, then one label from the 291. Temperature 0, fixed seed, no tools. Protocol fixed in advance: a 30-issue train-period trial for format and cost only (prompt not changed afterwards); one run on all 986 tune issues; adopt only if LLM top-1 minus LR top-1 has an interval entirely above 0.
- **Result** (`outputs/stage2/module_llm_trial.json`, `module_llm_tune.json`). Trial: 30 of 30 valid answers, US$0.012. Tune, top-1: LLM 69.0%, LR 57.1%, vote 59.1%; LLM minus LR +11.9 [+8.8, +14.9]; LLM minus vote +9.8 [+6.8, +12.9]; relative to LR, 176 fixed and 59 broken; 2 answers lie outside both references. By train frequency of the true module: 1 to 19 (112) 44.6% vs LR 25.9%, +18.8 [+9.8, +26.8], 24 fixed / 3 broken; 20 to 99 (236) 64.8% vs 41.5%, +23.3 [+17.4, +29.7], 60 / 5; 100 or more (638) 74.8% vs 68.3%, +6.4 [+2.8, +10.2], 92 / 51. Cost US$0.407 (about 3,100 input tokens and US$0.0004 per issue).
- **Decision.** Adopted (passes the rule). The gain is largest for modules with few training issues, where LR's implicit knowledge is thin; for frequent modules the LLM overrules a correct LR more often (51 broken), worth watching for stage-3 confidence. The LR C = 10 was itself chosen on tune, which favours LR in this comparison.

### 26. One-time test of the module pipeline (2026-10-11)

- **Question.** How does the frozen pipeline of entry 25 do on the test set?
- **What was done.** `./run_stage2_module.sh` (`scripts/stage2/module_llm.py test`), run once after the user approved freezing. References come from the train period only, as in the stage-1 final model. Afterwards the tune output was recomputed from the cached responses and is byte-identical, so the script change for the test split did not alter it.
- **Data.** 687 test issues (2026-07 to 2026-09).
- **Result** (`outputs/stage2/module_llm_test.json`). Top-1: LLM 60.8% [57.2, 64.5]; LR 48.2% (same as the stage-1 report); vote 48.8%. LLM minus LR +12.7 [+9.3, +16.2]; minus vote +12.1 [+8.6, +15.7]; 122 fixed, 35 broken; 3 answers outside both references. By train frequency of the true module: 0 (1 issue) both miss; 1 to 19 (107) 35.5% vs 18.7%, +16.8 [+8.4, +26.2]; 20 to 99 (203) 59.6% vs 42.9%, +16.7 [+10.8, +23.2]; 100 or more (376) 68.9% vs 59.6%, +9.3 [+4.8, +13.8]. Cost US$0.291.
- **Decision.** Module judgment closed for stage 2 with this pipeline. The gain matches the tune set (+11.9 there); absolute scores are lower on test for every method, as the test period is further from the training data. Total LLM spending so far about US$1.19 of the US$10 budget.

## Project records

### 27. Public repository and README restructure (2026-10-12)

- **Question.** Should the work be published, and in what form?
- **What was done.** The user chose to publish the repository publicly with its full git history, replacing the stage-1 plan of entry 8 (snapshot only, no HTML files). Repository: https://github.com/coderloganli/usher, branches `main` and `stage1-data` (default). Before pushing, the full history was checked for API keys and tokens (none found). The README was then rewritten for readers new to the project: purpose, test results, how each pipeline works, evaluation rules, how to reproduce, repository layout. Exact settings moved to the new `docs/pipeline.md`; development scores and the reasons behind choices stay in this file only.
- **Decision.** From now on the README shows only the current best pipeline per judgment and its test result, `docs/pipeline.md` holds the exact settings, and this file holds the chronological record. All three are updated after each adopted change.

## Stage 2: type judgment

### 28. Evaluation subset and baselines for the type judgment (2026-10-15)

- **Question.** How do the existing type baselines score on the evaluation subset agreed for stage 2?
- **What was done.** Plan approved by the user on 2026-10-15: the LLM makes the final type judgment (no tools) with the reporter's choice, the probabilities of LR (accuracy), the 10 most similar earlier train-period type issues with their types, and the 10 type definitions from the vscode wiki page "Issues-Triaging" as reference material; trial on 30 train-period issues (format and cost only), one run on the tune subset, then one run on the test subset after freezing. Adoption rule fixed in advance: LLM accuracy minus LR (accuracy) accuracy has a 95% paired bootstrap interval entirely above 0, and LLM macro-F1 is not significantly worse than LR (macro-F1) macro-F1 (the interval of the difference is not entirely below 0). Comparisons with the reporter's choice and per-type scores are reported either way. Step 1 is `scripts/stage2/type_baselines.py`: the subset is the type issues whose author is not a team-proxy account (entry 21; 107 accounts); the two stage-1 logistic regressions are used unchanged (fitted on the whole train set) and scored on the subset only.
- **Data.** Non-team subset: train 6,421 of 13,554, tune 890 of 2,165, test 556 of 1,848 issues. Only the tune subset is scored here.
- **Result** (`outputs/stage2/type_baselines_tune.json`). Accuracy / macro-F1 [95% CI]: always `bug` 57.4% [54.0, 60.7] / 0.081 [0.078, 0.094]; reporter's choice (`bug` when none) 77.8% [74.9, 80.6] / 0.176 [0.170, 0.203]; LR (macro-F1) 66.7% [63.6, 69.8] / 0.321 [0.245, 0.391]; LR (accuracy) 83.3% [80.8, 85.7] / 0.227 [0.205, 0.275]. LR (accuracy) minus reporter's choice: accuracy +5.5 [+3.7, +7.4]. 643 of the 890 issues carry a reporter's choice, which matches the final type for 535. Tune truth: `bug` 511, `feature-request` 306, `*question` 31, `under-discussion` 12, `upstream` 11, `polish` 7, `upstream-issue-linked` 6, `debt` 5, `engineering` 1, `electron` 0. LR (accuracy) predicts a type other than `bug` or `feature-request` for only 6 issues (all `*question`); LR (macro-F1) predicts `*question` 130 times for 31 true ones.
- **Decision.** These numbers reproduce the figures computed in conversation on 2026-10-11 and are the comparison points for the type LLM on the tune subset.

### 29. Type LLM: prompt and trial run (2026-10-16)

- **Question.** Does the type pipeline of entry 28 run correctly, and what does it cost?
- **What was done.** `scripts/stage2/type_llm.py trial`: 30 random non-team train-period type issues (seed fixed). Type definitions copied verbatim from the vscode wiki page "Issues-Triaging" (wiki commit `49c140c`, 2026-08-27) into `scripts/stage2/type_definitions.json`: the "Categorizing Issues" row of each of the 10 types, plus the "Closing Issues" row for `*question`, `upstream` and `upstream-issue-linked`. The prompt adds two notes of its own: reporters only choose between `bug` and `feature-request` and the team often changes it; the word-based classifier rarely predicts other types. Neighbours come from all train-period type issues. Fixed in advance: the trial is a format and cost check only, and the prompt is not changed because of it.
- **Data.** 30 train-period issues; the logistic regression is in-sample there, so no scores are computed.
- **Result** (`outputs/stage2/type_llm_trial.json`). 30 valid answers (22 `bug`, 8 `feature-request`), no refusals or truncations; about 950 input tokens per issue; US$0.005 in total, US$0.000165 per issue.
- **Decision.** Format accepted by the user. Expected cost: about US$0.15 for the 890 tune issues and US$0.09 for the 556 test issues. Total LLM spending so far about US$1.20.

## Open items

- Only the stage-1 duplicate baseline has been re-scored under the sibling rule on the test set (entry 20); stage-1 BM25 has not.
- LLM query rewriting for duplicates was tried and not adopted (entry 19); LLM reranking was set aside. The planned model is `gpt-4o-mini-2024-07-18` within a total budget of 10 US dollars (`plan.html`). OpenAI's official documentation, checked on 2026-10-10, lists it as available at 0.15 / 0.075 / 0.60 US dollars per million input / cached input / output tokens, knowledge cutoff 2023-10-01. The API key is kept in the git-ignored `.env` file. Total LLM spending so far about US$1.20.
- The module LLM outputs only a top-1 label with a one-sentence reason; there is no top-3 list and no confidence yet (stage 3).
- Type has no stage-2 work yet; it is next. Decided by the user on 2026-10-11 (see entry 21): evaluate only issues opened by non-team reporters (assignee-based approximation), with the reporter's own choice as an extra baseline, and state that the ground truth covers accepted issues only. Plan and adoption rule approved on 2026-10-15 (entry 28); baselines on the tune subset done. After type: whether predicted modules help duplicate ranking.
- The vscode code base (snapshot `9ed1e07` in the git-ignored `data/vscode_repo/`) and the "code path → module" table are kept for possible later use; they are not part of any adopted pipeline.
