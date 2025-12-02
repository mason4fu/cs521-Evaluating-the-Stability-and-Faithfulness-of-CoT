#!/bin/bash
# Script to run long experiment in tmux session
# This allows you to disconnect and reconnect later

VM_USER=${VM_USER:-exouser}
VM_HOST=${VM_HOST:-149.165.151.46}
SESSION_NAME=${SESSION_NAME:-cot-experiment}

echo "🚀 Starting long experiment in tmux session: $SESSION_NAME"
echo ""
echo "Commands to remember:"
echo "  Detach: Ctrl+B, then D"
echo "  Reattach: ssh $VM_USER@$VM_HOST && tmux attach -t $SESSION_NAME"
echo ""

# Load .env file if it exists (for HF_TOKEN and other secrets)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Get experiment configuration from environment or defaults
NUM_QUESTIONS=${NUM_QUESTIONS:-100}
NUM_SAMPLES=${NUM_SAMPLES:-3}

# Create tmux session and run experiment
ssh $VM_USER@$VM_HOST << ENDSSH
cd ~/cot-stability
source .venv/bin/activate
export HF_HOME=/media/volume/cot-llm-storage/hf-cache
export TMPDIR=/media/volume/cot-llm-storage/tmp
export TMP=\$TMPDIR
export TEMP=\$TMPDIR
export HF_TOKEN=\${HF_TOKEN:-}

# Experiment configuration (passed from local environment)
export NUM_QUESTIONS=$NUM_QUESTIONS
export NUM_SAMPLES=$NUM_SAMPLES

# Check if session exists
if tmux has-session -t cot-experiment 2>/dev/null; then
    echo "⚠️  Session 'cot-experiment' already exists"
    echo "Attach to it: tmux attach -t cot-experiment"
    echo "Or kill it first: tmux kill-session -t cot-experiment"
    exit 1
fi

# Create new session and run experiment
tmux new-session -d -s cot-experiment -c ~/cot-stability
tmux send-keys -t cot-experiment "source .venv/bin/activate" C-m
tmux send-keys -t cot-experiment "export HF_HOME=/media/volume/cot-llm-storage/hf-cache" C-m
tmux send-keys -t cot-experiment "export TMPDIR=/media/volume/cot-llm-storage/tmp" C-m
tmux send-keys -t cot-experiment "export HF_TOKEN=\${HF_TOKEN:-}" C-m
tmux send-keys -t cot-experiment "export NUM_QUESTIONS=$NUM_QUESTIONS" C-m
tmux send-keys -t cot-experiment "export NUM_SAMPLES=$NUM_SAMPLES" C-m
tmux send-keys -t cot-experiment "python run_all.py" C-m

echo "✅ Experiment started in tmux session 'cot-experiment'"
echo ""
echo "To attach and see progress:"
echo "  ssh $VM_USER@$VM_HOST"
echo "  tmux attach -t cot-experiment"
echo ""
echo "To detach (keep running):"
echo "  Press: Ctrl+B, then D"
ENDSSH

echo ""
echo "✅ Done! Your experiment is running in tmux session 'cot-experiment'"
echo ""
echo "Next steps:"
echo "  1. Attach to see progress: ssh $VM_USER@$VM_HOST && tmux attach -t cot-experiment"
echo "  2. Detach anytime: Ctrl+B, then D"
echo "  3. Close your terminal - experiment keeps running!"
echo "  4. Reconnect later: ssh $VM_USER@$VM_HOST && tmux attach -t cot-experiment"

