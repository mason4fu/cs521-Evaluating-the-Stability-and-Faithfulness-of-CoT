#!/usr/bin/env bash
# Setup script to run on VM - installs dependencies and sets up environment
set -euo pipefail

echo "🔧 Setting up VM environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📥 Installing requirements..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found!"
fi

echo "✅ VM setup complete!"
echo "💡 To activate the environment, run: source .venv/bin/activate"
echo ""
echo "📋 Next steps:"
echo "   1. Deploy model: bash deploy_model_to_vm.sh"
echo "   2. Or install vLLM manually: pip install vllm"
echo "   3. Download model: see DEPLOYMENT.md"

