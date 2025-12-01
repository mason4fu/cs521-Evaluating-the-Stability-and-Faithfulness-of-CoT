#!/usr/bin/env bash
# Sync local code to VM using rsync (best practice)
set -euo pipefail

VM_USER="exouser"
VM_HOST="149.165.151.46"
VM_PATH="~/cot-stability"
LOCAL_PATH="."

echo "🔄 Syncing code to VM..."
rsync -avz \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.DS_Store' \
  --exclude='outputs/' \
  --exclude='.env' \
  --exclude='*.pyc' \
  "${LOCAL_PATH}/" "${VM_USER}@${VM_HOST}:${VM_PATH}/"

echo "✅ Sync complete!"
echo "📝 Note: .env file is excluded for security. Copy it manually if needed."

