#!/usr/bin/env bash
set -euo pipefail

# Activate venv if present (optional)
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
# optional model name argument
MODEL_NAME=${MODEL_NAME:-toy}
export MODEL_NAME

# Check and install requirements
if [ -f "requirements.txt" ]; then
  echo "Checking and installing requirements..."
  pip install -q -r requirements.txt
  echo "Requirements satisfied."
else
  echo "Warning: requirements.txt not found, skipping dependency check."
fi

echo ""

python src/collect_cot.py
python src/metrics.py
python src/visualize.py

echo "Done. Check outputs/ for runs.jsonl, summary.csv, and figures/"