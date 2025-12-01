"""Utility functions for CoT faithfulness experiments"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import re
from typing import List, Dict, Optional

import config

# Directory aliases (from config)
DATA = config.DATA_DIR
OUT = config.OUTPUTS_DIR
FIG = config.FIGURES_DIR

def load_json(path):
    """Load JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data, indent=2):
    """Save data to JSON file"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def append_jsonl(path, obj):
    """Append object to JSONL file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def extract_number_from_text(text: str) -> str:
    """Extract the last numeric value from text"""
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    return matches[-1] if matches else "unknown"

def extract_gold_answer(gold_text: str) -> str:
    """Extract gold answer from GSM8K format (#### number)"""
    matches = re.findall(r"####\s*([-+]?\d*\.?\d+)", gold_text)
    return matches[-1] if matches else "unknown"

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    # Remove $, commas, whitespace
    normalized = answer.replace("$", "").replace(",", "").strip()
    try:
        # Try to convert to float then back to string to normalize format
        return str(int(float(normalized)))
    except:
        return normalized.lower()

def sentences_split(text: str) -> List[str]:
    """Split text into sentences using NLTK punkt tokenizer"""
    try:
        import nltk
        # Download punkt if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except ImportError:
        # Fallback to simple sentence splitting if NLTK not available
        # Split on period, exclamation, question mark followed by space or newline
        import re
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]