#!/usr/bin/env bash
# Download results from VM using rsync
set -euo pipefail

VM_USER="exouser"
VM_HOST="149.165.151.46"
VM_PATH="~/cot-stability/outputs"
LOCAL_PATH="./outputs"

echo "⬇️  Downloading results from VM..."
rsync -avz "${VM_USER}@${VM_HOST}:${VM_PATH}/" "${LOCAL_PATH}/"

echo "✅ Download complete!"

