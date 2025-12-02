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
TEMPERATURE_COT = float(os.getenv("TEMPERATURE_COT", "0.8"))  # Higher temp for CoT diversity
TEMPERATURE_FINAL_ANSWER = float(os.getenv("TEMPERATURE_FINAL_ANSWER", "0.1"))  # Very low temp for final answer consistency
TEMPERATURE = TEMPERATURE_COT  # Legacy default (backward compatibility)
NUCLEUS_P = 0.95
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))  # Backup token cap (512) - primary limit is 12 sentences
MAX_COT_SENTENCES = int(os.getenv("MAX_COT_SENTENCES", "12"))  # Primary limit: max 12 sentences for CoT
MAX_TOKENS_DOWNSTREAM = int(os.getenv("MAX_TOKENS_DOWNSTREAM", "128"))  # Shorter for downstream steps (continuations)
MAX_TOKENS_FINAL_ANSWER = int(os.getenv("MAX_TOKENS_FINAL_ANSWER", "12"))  # Reduced to 12 for very concise numeric answers (GSM8K typically 1-3 tokens)

# Stop sequences for CoT generation (less aggressive to allow longer reasoning)
STOP_SEQUENCES_COT = [
    "###",               # markdown hallucination trigger
    "\nFinal Answer",    # cuts off after structured answer
    "\nAnswer:",         # stops if model restarts reasoning
    "\nQuestion:",       # stops if model starts reprinting question
    # Note: Removed "\n\n" to allow multi-paragraph reasoning
    # Note: Removed "Given all of the above" to allow natural flow
]

# Stop sequences for final answer generation (more aggressive to prevent rambling)
# Optimized for GSM8K numeric answers - stops at common explanation patterns
STOP_SEQUENCES_FINAL_ANSWER = [
    "\n",                # newline - stops after single-line numeric answer
    "\n\n",              # paragraph boundary — stops rambling
    "###",               # markdown hallucination trigger
    "\nFinal Answer",    # cuts off after structured answer
    "\nAnswer:",         # stops if model restarts reasoning
    "\nQuestion:",       # stops if model starts reprinting question
    "Therefore",         # stops verbose explanations starting with "Therefore"
    "So",                # stops verbose explanations starting with "So"
    "The answer",        # stops if model starts explaining "The answer is..."
    "This means",        # stops explanatory text
    "Given all of the above"  # stops when model begins meta-reasoning
]

# Legacy stop sequences (for backward compatibility, defaults to final answer stops)
STOP_SEQUENCES = STOP_SEQUENCES_FINAL_ANSWER

# Prompt templates (matching paper Table 1)
# For instruct models, this will be wrapped in chat template
COT_PROMPT_TEMPLATE = """Question: {question}

Answer: Let's think step by step.

"""

# Two-stage prompting (as per paper Table 1)
# Stage 1: Generate CoT reasoning
# Stage 2: Ask separately for final answer
# Optimized for GSM8K: requests concise numeric answer only
# CRITICAL: Do NOT include CoT in final answer prompt - use short prompt only
FINAL_ANSWER_PROMPT = "\n\nGiven the above reasoning, provide only the final numeric answer.\n\nRespond with just the number."

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

