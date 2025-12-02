#!/usr/bin/env bash
# Monitor experiment progress every N minutes
# Usage: bash monitor_progress.sh [interval_minutes]
set -euo pipefail

INTERVAL_MINUTES=${1:-10}
INTERVAL_SECONDS=$((INTERVAL_MINUTES * 60))

LOG_FILE="${LOG_FILE:-monitor_progress.log}"

echo "📊 Starting Progress Monitor"
echo "   Interval: ${INTERVAL_MINUTES} minutes"
echo "   Log file: ${LOG_FILE}"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

# Function to run check_progress.sh and log output
check_and_log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local output
    
    echo ""
    echo "[$timestamp] Progress Check:"
    echo "--------------------------------"
    
    # Capture output
    output=$(bash check_progress.sh 2>&1)
    
    # Display to terminal
    echo "$output"
    
    # Append to log
    echo "" >> "$LOG_FILE"
    echo "==========================================" >> "$LOG_FILE"
    echo "[$timestamp] Progress Check" >> "$LOG_FILE"
    echo "==========================================" >> "$LOG_FILE"
    echo "$output" >> "$LOG_FILE"
}

# Trap Ctrl+C for graceful exit
trap 'echo ""; echo "⏸️  Monitoring stopped"; exit 0' INT

# Initial check
check_and_log

# Loop every interval
while true; do
    echo ""
    echo "⏰ Waiting ${INTERVAL_MINUTES} minutes until next check..."
    sleep "$INTERVAL_SECONDS"
    check_and_log
done

