"""Configuration for CoT Faithfulness Experiments"""
import os
from pathlib import Path
from typing import Optional

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_RUNTIME = os.getenv("MODEL_RUNTIME", "vllm")  # "vllm" or "llamacpp"
MODEL_QUANTIZATION = os.getenv("MODEL_QUANTIZATION", None)  # "Q4" or "Q4_K_M" for llama.cpp
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000")  # vLLM API server URL (None = load model directly)

# VM/SSH configuration
VM_USER = os.getenv("VM_USER", "exouser")
VM_HOST = os.getenv("VM_HOST", "149.165.151.46")
VM_HOME = os.getenv("VM_HOME", "~/cot-stability")
VM_MODEL_DIR = os.getenv("VM_MODEL_DIR", "${VM_HOME}/models")
VM_PYTHON_PATH = os.getenv("VM_PYTHON_PATH", "python3")

# Experiment configuration
GSM8K_SEED = 44
# Number of questions (None = all questions, set via NUM_QUESTIONS env var)
NUM_QUESTIONS = int(os.getenv("NUM_QUESTIONS")) if os.getenv("NUM_QUESTIONS") else None
NUM_SAMPLES_PER_QUESTION = int(os.getenv("NUM_SAMPLES", "20"))  # N samples per question (default 20)
TEMPERATURE = 0.8
NUCLEUS_P = 0.95
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))  # Reduced from 1024 for faster inference

# Prompt templates (matching paper Table 1)
# For instruct models, this will be wrapped in chat template
COT_PROMPT_TEMPLATE = """Question: {question}

Answer: Let's think step by step.

"""

FINAL_ANSWER_PROMPT = "\n\nThe answer is"

# Directories
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Test run configuration (for small tests)
TEST_MODE = os.getenv("TEST_MODE", "false").lower() in ("true", "1", "yes")
TEST_NUM_QUESTIONS = int(os.getenv("TEST_NUM_QUESTIONS", "2"))
TEST_NUM_SAMPLES = int(os.getenv("TEST_NUM_SAMPLES", "2"))

