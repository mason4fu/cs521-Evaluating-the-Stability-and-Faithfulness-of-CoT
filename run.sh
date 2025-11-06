#!/usr/bin/env bash
set -euo pipefail

# Activate venv if present (optional)
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
# optional model name argument
MODEL_NAME=${MODEL_NAME:-toy}
export MODEL_NAME
python src/collect_cot.py
python src/metrics.py
python src/visualize.py

echo "Done. Check outputs/ for runs.jsonl, summary.csv, and figures/"