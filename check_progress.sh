#!/bin/bash
# Quick script to check experiment progress without attaching

VM_USER=${VM_USER:-exouser}
VM_HOST=${VM_HOST:-149.165.151.46}

echo "📊 Checking experiment progress..."
echo ""

ssh $VM_USER@$VM_HOST << 'ENDSSH'
echo "=== tmux Session Status ==="
if tmux has-session -t cot-experiment 2>/dev/null; then
    echo "✅ tmux session 'cot-experiment' is running"
else
    echo "❌ No active tmux session found"
fi
echo ""

echo "=== Process Status ==="
if ps aux | grep -E 'python.*run_all' | grep -v grep > /dev/null; then
    echo "✅ Experiment process is running:"
    ps aux | grep -E 'python.*run_all' | grep -v grep | head -1
else
    echo "❌ No experiment process found"
fi
echo ""

echo "=== GPU Status ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

echo "=== Output Files ==="
if [ -d ~/cot-stability/outputs ]; then
    echo "Files in outputs/:"
    ls -lh ~/cot-stability/outputs/*.csv ~/cot-stability/outputs/*.jsonl 2>/dev/null | \
        awk '{print $5, $9}' | column -t
    echo ""
    
    if [ -f ~/cot-stability/outputs/cot_samples.jsonl ]; then
        SAMPLE_COUNT=$(wc -l < ~/cot-stability/outputs/cot_samples.jsonl)
        echo "📈 CoT samples generated: $SAMPLE_COUNT"
    fi
else
    echo "❌ Outputs directory not found"
fi
ENDSSH

echo ""
echo "To see full output: ssh $VM_USER@$VM_HOST && tmux attach -t cot-experiment"
