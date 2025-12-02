#!/usr/bin/env bash
# Script to check VM specifications
set -euo pipefail

echo "=========================================="
echo "VM System Specifications"
echo "=========================================="
echo ""

echo "--- CPU Information ---"
if command -v lscpu &> /dev/null; then
    lscpu | grep -E "Architecture|CPU\(s\)|Thread|Core|Model name|MHz"
else
    echo "CPU Cores: $(nproc)"
    echo "CPU Model: $(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d: -f2 | xargs)"
fi
echo ""

echo "--- Memory Information ---"
free -h
echo ""

echo "--- Disk Space ---"
df -h / | tail -1
echo ""

echo "--- Operating System ---"
if [ -f /etc/os-release ]; then
    cat /etc/os-release | grep -E "NAME|VERSION" | head -2
fi
uname -a
echo ""

echo "--- GPU Information (if available) ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "No NVIDIA GPU detected (nvidia-smi not found)"
fi
echo ""

echo "--- Python Environment ---"
if command -v python3 &> /dev/null; then
    echo "Python version: $(python3 --version)"
    if [ -d ".venv" ]; then
        echo "Virtual environment: .venv exists"
        if [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
            echo "Python in venv: $(python --version 2>&1)"
        fi
    fi
else
    echo "Python 3 not found"
fi
echo ""

echo "=========================================="

