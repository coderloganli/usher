#!/usr/bin/env bash
# Run the frozen stage-2 module pipeline (see README.md) on the test split.
# Needs the Part A data in data/ (./run_data.sh), the cleaned-text embedding cache
# data/stage2/embeddings_clean.npy (written by scripts/stage2/dedup_clean.py), and the OpenAI API
# key in the git-ignored .env. LLM responses are cached in data/stage2/module_llm/, so a rerun
# with the cache makes no API calls.
#   LR top 10 + 10 most similar train issues -> gpt-4o-mini picks one module label
# The development steps that led to this pipeline are listed in EXPERIMENTS.md (entries 21-26).
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python scripts/stage2/module_llm.py test
