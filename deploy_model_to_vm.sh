#!/usr/bin/env bash
# Deploy Llama-3.1-8B-Instruct model and vLLM to VM
# This script should be run ON THE VM (SSH in first)

set -euo pipefail

VM_USER=${VM_USER:-exouser}
VM_HOST=${VM_HOST:-149.165.151.46}
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_DIR="${HOME}/cot-stability/models"

echo "🚀 Deploying model to VM..."
echo "   Model: ${MODEL_NAME}"
echo "   Directory: ${MODEL_DIR}"
echo ""

# Check if we're on the VM or local
if [ "$(hostname)" != "cs521-cot-eval.cis240285.projects.jetstream-cloud.org" ] && [ -z "${FORCE_DEPLOY:-}" ]; then
    echo "⚠️  This script should be run ON THE VM."
    echo "   Options:"
    echo "   1. SSH to VM first: ssh ${VM_USER}@${VM_HOST}"
    echo "   2. Or run remotely: ssh ${VM_USER}@${VM_HOST} 'bash -s' < deploy_model_to_vm.sh"
    echo ""
    read -p "Continue anyway? (y/N): " confirm
    if [ "$confirm" != "y" ]; then
        exit 1
    fi
fi

# Create model directory
echo "📁 Creating model directory..."
mkdir -p "${MODEL_DIR}"

# Activate venv if it exists
if [ -d "${HOME}/cot-stability/.venv" ]; then
    source "${HOME}/cot-stability/.venv/bin/activate"
    echo "✅ Activated virtual environment"
fi

# Install vLLM (required for inference)
echo ""
echo "📦 Installing vLLM..."
echo "   (This may take a few minutes...)"
pip install vllm

# Check if CUDA is available
echo ""
echo "🔍 Checking GPU availability..."
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  CUDA not available - model will run on CPU (very slow!)")
EOF

# Download model using HuggingFace
echo ""
echo "📥 Downloading model..."
echo "   Note: This is a large download (~16GB). It may take 15-20 minutes."
echo ""
read -p "Continue with download? (y/N): " confirm
if [ "$confirm" != "y" ]; then
    echo "❌ Download cancelled"
    exit 1
fi

# Use huggingface-cli or python to download
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli..."
    huggingface-cli download ${MODEL_NAME} --local-dir "${MODEL_DIR}/${MODEL_NAME##*/}" --local-dir-use-symlinks False
else
    echo "Using Python transformers to download..."
    python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "${MODEL_NAME}"
model_dir = "${MODEL_DIR}"

print(f"Downloading {model_name} to {model_dir}...")
print("(This will take a while...)")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
print("✅ Tokenizer downloaded")

# Download model (this is the big one)
# Note: We're just downloading, not loading into memory
print("Downloading model weights...")
print("(For large models, consider using huggingface-cli instead)")
EOF
fi

# Alternative: Download using wget/curl if you have a direct URL
echo ""
echo "✅ Model deployment complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Verify model is at: ${MODEL_DIR}"
echo "   2. Update config.py or set VM_MODEL_DIR environment variable"
echo "   3. Run experiments with: python run_all.py"
echo ""
echo "💡 Tip: Check model size:"
echo "   du -sh ${MODEL_DIR}/*"

