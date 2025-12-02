#!/usr/bin/env bash
# Check experiment progress on VM
set -euo pipefail

VM_USER=${VM_USER:-exouser}
VM_HOST=${VM_HOST:-149.165.151.46}

echo "📊 Checking Experiment Progress..."
echo "=================================="
echo ""

ssh ${VM_USER}@${VM_HOST} << 'ENDSSH'
cd ~/cot-stability

# Check if tmux session is running
if tmux has-session -t cot-experiment 2>/dev/null; then
    echo "✅ Experiment is RUNNING in tmux session 'cot-experiment'"
    echo ""
    echo "Recent output (last 25 lines):"
    echo "--------------------------------"
    tmux capture-pane -t cot-experiment -p | tail -25
    echo ""
else
    echo "❌ tmux session 'cot-experiment' NOT FOUND"
    echo "   (Experiment may have completed or crashed)"
    echo ""
fi

# Check output files
echo "📁 Output Files Status:"
echo "--------------------------------"
if [ -f outputs/cot_samples.jsonl ]; then
    COT_COUNT=$(wc -l < outputs/cot_samples.jsonl)
    FILE_SIZE=$(du -h outputs/cot_samples.jsonl | cut -f1)
    echo "✅ cot_samples.jsonl: $COT_COUNT samples ($FILE_SIZE)"
else
    echo "⏳ cot_samples.jsonl: Not created yet"
fi

if [ -f outputs/early_answering_results.csv ]; then
    TRUNC_COUNT=$(wc -l < outputs/early_answering_results.csv)
    FILE_SIZE=$(du -h outputs/early_answering_results.csv | cut -f1)
    echo "✅ early_answering_results.csv: $TRUNC_COUNT rows ($FILE_SIZE)"
else
    echo "⏳ early_answering_results.csv: Not created yet"
fi

if [ -f outputs/adding_mistakes_results.csv ]; then
    MISTAKE_COUNT=$(wc -l < outputs/adding_mistakes_results.csv)
    FILE_SIZE=$(du -h outputs/adding_mistakes_results.csv | cut -f1)
    echo "✅ adding_mistakes_results.csv: $MISTAKE_COUNT rows ($FILE_SIZE)"
else
    echo "⏳ adding_mistakes_results.csv: Not created yet"
fi

if [ -f outputs/paraphrasing_results.csv ]; then
    PARA_COUNT=$(wc -l < outputs/paraphrasing_results.csv)
    FILE_SIZE=$(du -h outputs/paraphrasing_results.csv | cut -f1)
    echo "✅ paraphrasing_results.csv: $PARA_COUNT rows ($FILE_SIZE)"
else
    echo "⏳ paraphrasing_results.csv: Not created yet"
fi

if [ -f outputs/filler_tokens_results.csv ]; then
    FILLER_COUNT=$(wc -l < outputs/filler_tokens_results.csv)
    FILE_SIZE=$(du -h outputs/filler_tokens_results.csv | cut -f1)
    echo "✅ filler_tokens_results.csv: $FILLER_COUNT rows ($FILE_SIZE)"
else
    echo "⏳ filler_tokens_results.csv: Not created yet"
fi

echo ""
echo "📈 Experiment Progress:"
echo "--------------------------------"

# Experiment 1: CoT Generation
if [ -f outputs/cot_samples.jsonl ]; then
    COT_COUNT=$(wc -l < outputs/cot_samples.jsonl)
    UNIQUE_QUESTIONS=$(grep -o '"question_id":"[^"]*"' outputs/cot_samples.jsonl 2>/dev/null | sort -u | wc -l || echo "0")
    # Check if tmux session is running to determine if experiment is in progress
    if tmux has-session -t cot-experiment 2>/dev/null; then
        RECENT=$(tmux capture-pane -t cot-experiment -p 2>/dev/null | tail -5 | grep -i "experiment 1\|cot generation\|generating" || true)
        if [ -n "$RECENT" ]; then
            echo "   🔄 Experiment 1 (CoT Generation): IN PROGRESS"
            echo "      Samples so far: $COT_COUNT"
        else
            echo "   ✅ Experiment 1 (CoT Generation): COMPLETE"
            echo "      Samples: $COT_COUNT"
        fi
    else
        echo "   ✅ Experiment 1 (CoT Generation): COMPLETE"
        echo "      Samples: $COT_COUNT"
    fi
    else
    if tmux has-session -t cot-experiment 2>/dev/null; then
        echo "   🔄 Experiment 1 (CoT Generation): IN PROGRESS (file not created yet)"
else
    echo "   ⏳ Experiment 1 (CoT Generation): Not started"
    fi
fi

# Experiment 2: Truncation
if [ -f outputs/early_answering_results.csv ]; then
    TRUNC_COUNT=$(wc -l < outputs/early_answering_results.csv)
    # Check if experiment 2 is mentioned in recent output
    if tmux has-session -t cot-experiment 2>/dev/null; then
        RECENT=$(tmux capture-pane -t cot-experiment -p 2>/dev/null | tail -5 | grep -i "experiment 2\|truncation" || true)
        if [ -z "$RECENT" ]; then
            echo "   ✅ Experiment 2 (Truncation): COMPLETE"
            echo "      Results: $((TRUNC_COUNT - 1)) data rows"
        else
            echo "   🔄 Experiment 2 (Truncation): IN PROGRESS"
            echo "      Results so far: $((TRUNC_COUNT - 1)) rows"
        fi
    else
        echo "   ✅ Experiment 2 (Truncation): COMPLETE"
        echo "      Results: $((TRUNC_COUNT - 1)) data rows"
    fi
else
    echo "   ⏳ Experiment 2 (Truncation): Not started"
fi

# Experiment 3: Mistakes
if [ -f outputs/adding_mistakes_results.csv ]; then
    MISTAKE_COUNT=$(wc -l < outputs/adding_mistakes_results.csv)
    if tmux has-session -t cot-experiment 2>/dev/null; then
        RECENT=$(tmux capture-pane -t cot-experiment -p 2>/dev/null | tail -5 | grep -i "experiment 3\|mistake" || true)
        if [ -z "$RECENT" ]; then
            echo "   ✅ Experiment 3 (Mistakes): COMPLETE"
            echo "      Results: $((MISTAKE_COUNT - 1)) data rows"
        else
            echo "   🔄 Experiment 3 (Mistakes): IN PROGRESS"
            echo "      Results so far: $((MISTAKE_COUNT - 1)) rows"
        fi
    else
        echo "   ✅ Experiment 3 (Mistakes): COMPLETE"
        echo "      Results: $((MISTAKE_COUNT - 1)) data rows"
    fi
else
    if tmux has-session -t cot-experiment 2>/dev/null; then
        RECENT=$(tmux capture-pane -t cot-experiment -p 2>/dev/null | tail -5 | grep -i "experiment 3\|mistake" || true)
        if [ -n "$RECENT" ]; then
            echo "   🔄 Experiment 3 (Mistakes): IN PROGRESS (file not created yet)"
        else
            echo "   ⏳ Experiment 3 (Mistakes): Waiting for Experiment 2"
        fi
    else
        echo "   ⏳ Experiment 3 (Mistakes): Waiting for Experiment 2"
    fi
fi

# Experiment 4: Paraphrasing
if [ -f outputs/paraphrasing_results.csv ]; then
    PARA_COUNT=$(wc -l < outputs/paraphrasing_results.csv)
    echo "   ✅ Experiment 4 (Paraphrasing): COMPLETE"
    echo "      Results: $((PARA_COUNT - 1)) data rows"
else
    echo "   ⏳ Experiment 4 (Paraphrasing): Waiting for previous experiments"
fi

# Experiment 5: Filler Tokens
if [ -f outputs/filler_tokens_results.csv ]; then
    FILLER_COUNT=$(wc -l < outputs/filler_tokens_results.csv)
    echo "   ✅ Experiment 5 (Filler Tokens): COMPLETE"
    echo "      Results: $((FILLER_COUNT - 1)) data rows"
else
    echo "   ⏳ Experiment 5 (Filler Tokens): Waiting for previous experiments"
fi

echo ""
echo "💡 Commands:"
echo "   Monitor live: ssh exouser@149.165.151.46 && tmux attach -t cot-experiment"
echo "   Download results: bash sync_from_vm.sh"
ENDSSH

echo ""
echo "✅ Progress check complete!"
