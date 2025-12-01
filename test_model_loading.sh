#!/bin/bash
# Test model loading with proper environment

set -euo pipefail

cd ~/cot-stability
source .venv/bin/activate

# Set all paths to volume
export HF_HOME=/media/volume/cot-llm-storage/hf-cache
export TRANSFORMERS_CACHE=/media/volume/cot-llm-storage/hf-cache
export TMPDIR=/media/volume/cot-llm-storage/tmp
export TMP=/media/volume/cot-llm-storage/tmp
export TEMP=/media/volume/cot-llm-storage/tmp
export HF_TOKEN=${HF_TOKEN:-}

# Create directories
mkdir -p "$HF_HOME"
mkdir -p "$TMPDIR"
mkdir -p /media/volume/cot-llm-storage/models

echo "🚀 Testing model loading..."
echo "   Volume: /media/volume/cot-llm-storage (98GB free)"
echo "   GPU: Checking actual memory..."
nvidia-smi --query-gpu=name,memory.total --format=csv

python3 << 'PYEOF'
from vllm import LLM, SamplingParams
import sys

model_id = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'

print('\n🧪 Testing with GPU memory check...')
print(f'   Model: {model_id}')
print('')

try:
    # Try with conservative settings first
    print('⏳ Loading model (this will download ~35GB on first use)...')
    print('   Using 80% GPU memory utilization...')
    
    llm = LLM(
        model=model_id,
        gpu_memory_utilization=0.80,  # Conservative
        max_model_len=1024,  # Reduced context
        dtype='float16',
        enable_prefix_caching=True,
        download_dir='/media/volume/cot-llm-storage/models'
    )
    print('✅ Model loaded successfully!')
    
    # Test generation
    print('\n🧪 Testing generation...')
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
    outputs = llm.generate(['What is 2+2? Answer:'], sampling_params)
    result = outputs[0].outputs[0].text
    print(f'✅ Generation works! Output: {result[:100]}')
    print('\n🎉 Model is ready!')
    
except Exception as e:
    error_msg = str(e)
    print(f'\n❌ Error: {type(e).__name__}')
    print(f'   {error_msg[:500]}')
    sys.exit(1)
PYEOF

