#!/usr/bin/env bash
# Rerun every Part B baseline result (plan: usher/stage1_partB_plan.html).
# Needs the Part A data in data/ (./run_data.sh --offline rebuilds it from the cached raw data).
#   1. type and module: score every candidate setting on the tune set, record the chosen ones
#   2. duplicate: score BM25, embedding and RRF on the development queries, decide on RRF
#   3. one-shot test evaluation of the frozen methods, with bootstrap intervals
#   4. the results report, docs/baselines.html
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
S=scripts/baselines

$PY $S/classify.py
$PY $S/dedup.py
$PY $S/evaluate_test.py
$PY $S/report.py
