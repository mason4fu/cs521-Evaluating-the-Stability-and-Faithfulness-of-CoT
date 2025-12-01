#!/usr/bin/env bash
# Authenticate with HuggingFace for gated model access

set -euo pipefail

echo "🔐 HuggingFace Authentication Required"
echo ""
echo "The Llama-4-Maverick-17B model is gated and requires authentication."
echo ""
echo "You need a HuggingFace token with access to meta-llama models."
echo ""
echo "Get your token from: https://huggingface.co/settings/tokens"
echo ""

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Enter your HuggingFace token (or set HF_TOKEN environment variable):"
    read -s HF_TOKEN
    echo ""
fi

if [ -z "$HF_TOKEN" ]; then
    echo "❌ No token provided"
    exit 1
fi

# Login using huggingface-cli
echo "🔐 Authenticating..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

echo ""
echo "✅ Authentication complete!"

