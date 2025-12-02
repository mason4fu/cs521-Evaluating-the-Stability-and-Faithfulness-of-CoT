#!/usr/bin/env bash
# Run progress monitor in background (detached)
# Usage: bash monitor_progress_background.sh [interval_minutes] [log_file]
set -euo pipefail

INTERVAL_MINUTES=${1:-10}
LOG_FILE=${2:-monitor_progress.log}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="${SCRIPT_DIR}/monitor_progress.sh"

echo "📊 Starting Background Progress Monitor"
echo "   Interval: ${INTERVAL_MINUTES} minutes"
echo "   Log file: ${LOG_FILE}"
echo ""

# Check if already running
if pgrep -f "monitor_progress.sh" > /dev/null; then
    echo "⚠️  Progress monitor is already running!"
    echo "   PID: $(pgrep -f 'monitor_progress.sh')"
    echo "   To stop: pkill -f monitor_progress.sh"
    exit 1
fi

# Start in background
nohup bash "$MONITOR_SCRIPT" "$INTERVAL_MINUTES" > "$LOG_FILE" 2>&1 &
MONITOR_PID=$!

echo "✅ Background monitor started"
echo "   PID: ${MONITOR_PID}"
echo "   Log: ${LOG_FILE}"
echo ""
echo "Commands:"
echo "   View log: tail -f ${LOG_FILE}"
echo "   Stop monitor: kill ${MONITOR_PID}"
echo "   Or: pkill -f monitor_progress.sh"
echo ""

