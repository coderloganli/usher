#!/usr/bin/env bash
# Run the frozen stage-2 type pipeline (see README.md) on the non-team test subset.
# Needs the Part A data in data/ (./run_data.sh), the cleaned-text embedding cache
# data/stage2/embeddings_clean.npy (written by scripts/stage2/dedup_clean.py), and the OpenAI API
# key in the git-ignored .env. LLM responses are cached in data/stage2/type_llm/, so a rerun
# with the cache makes no API calls.
#   reporter's choice + LR probabilities + 10 most similar train issues + wiki type definitions
#   -> gpt-4o-mini picks one type label
# The development steps that led to this pipeline are listed in EXPERIMENTS.md (entries 28-31).
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python scripts/stage2/type_llm.py test
