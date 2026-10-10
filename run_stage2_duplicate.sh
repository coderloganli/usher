#!/usr/bin/env bash
# Run the frozen stage-2 duplicate pipeline (see README.md) on the test split.
# Needs the Part A data in data/ (./run_data.sh) and network access for the link-time pull
# (responses are cached, so a rerun makes no new requests).
#   1. creation time of every duplicate link (needed by the sibling rule)
#   2. cleaned text -> embedding top 50 -> cross-encoder rerank -> top 10, scored together with
#      the stage-1 baseline on the same test queries
# The development experiments that led to this pipeline are separate scripts (EXPERIMENTS.md).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
S=scripts/stage2

$PY $S/pull_duplicate_events.py
$PY $S/dedup_final.py --split test
