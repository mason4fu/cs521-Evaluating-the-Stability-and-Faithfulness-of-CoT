#!/usr/bin/env bash
# Run experiment on VM (use with tmux for long-running jobs)
set -euo pipefail

VM_USER="exouser"
VM_HOST="149.165.151.46"
VM_PATH="~/cot-stability"

echo "🚀 Running experiment on VM..."
echo "💡 Tip: Use 'tmux new -s experiment' on VM to keep job running if connection drops"
echo ""

ssh "${VM_USER}@${VM_HOST}" << 'ENDSSH'
cd ~/cot-stability

# Activate venv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the experiment
sh run.sh
ENDSSH

echo "✅ Experiment complete!"

