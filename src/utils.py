"""Utility functions for CoT faithfulness experiments"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import re
from typing import List, Dict, Optional
import numpy as np

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

def generate_formatted_jsonl(source_path: Path, dest_path: Path = None):
    """
    Generate a formatted (pretty-printed) version of a JSONL file.
    Each JSON object is formatted with indentation for readability.
    
    Args:
        source_path: Path to source JSONL file (e.g., cot_samples.jsonl)
        dest_path: Path to destination formatted file (default: source_path with '_formatted' suffix)
    """
    source_path = Path(source_path)
    if not source_path.exists():
        return
    
    if dest_path is None:
        # Default: add '_formatted' before .jsonl extension
        dest_path = source_path.parent / f"{source_path.stem}_formatted{source_path.suffix}"
    else:
        dest_path = Path(dest_path)
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(source_path, "r", encoding="utf-8") as f_in, \
         open(dest_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Write formatted JSON with indentation
                f_out.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n\n")
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue
    
    return dest_path

def extract_number_from_text(text: str) -> str:
    """
    Extract numeric value from text with priority:
    1. Boxed answers ($\boxed{X}$ or \boxed{X})
    2. Answer markers ("The answer is X", "Final answer: X", etc.)
    3. Last number in text (fallback)
    """
    if not text:
        return "unknown"
    
    # Priority 1: Check for boxed answers (LaTeX format)
    # Matches: $\boxed{8}$, \boxed{8}, $\boxed{8}$ (with or without $)
    boxed_patterns = [
        r'\\boxed\{([-+]?\d*\.?\d+)\}',  # \boxed{8}
        r'\$\\boxed\{([-+]?\d*\.?\d+)\}\$',  # $\boxed{8}$
    ]
    for pattern in boxed_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    # Priority 2: Look for answer markers followed by number
    # Patterns: "The answer is 8", "Final answer: 8", "Answer: 8", etc.
    answer_marker_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s*[:\s]+([-+]?\d*\.?\d+)',
        r'answer\s*[:\s]+([-+]?\d*\.?\d+)',
        r'(?:final\s+)?answer\s+is\s+([-+]?\d*\.?\d+)',
    ]
    for pattern in answer_marker_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Extract number and stop at punctuation or newline
            num_str = match.group(1)
            # Find where this number appears and check if it's followed by punctuation
            num_pos = text.find(num_str, match.start())
            if num_pos != -1:
                # Check if followed by punctuation (likely the actual answer)
                after_num = text[num_pos + len(num_str):num_pos + len(num_str) + 10]
                if re.match(r'^[^\d]*[.!?\n]', after_num) or not re.search(r'\d', after_num[:5]):
                    return num_str
    
    # Priority 3: Fallback to first "clean" number (CRITICAL: use FIRST, not last)
    # This fixes bug where "8.0.0.0.0.0" would extract ".0" instead of "8"
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if matches:
        # Use FIRST match, but prefer a "clean" number (has digits before decimal)
        for match in matches:
            # Check if it's a clean number (not just ".0" or ".5" fragment)
            if match.lstrip('+-').replace('.', '').isdigit() and not match.startswith('.'):
                return match
        # If no clean match, return first match anyway
        return matches[0]
    return "unknown"

def extract_gold_answer(gold_text: str) -> str:
    """Extract gold answer from GSM8K format (#### number)"""
    matches = re.findall(r"####\s*([-+]?\d*\.?\d+)", gold_text)
    return matches[-1] if matches else "unknown"

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    if not answer or answer == "unknown":
        return answer.lower() if answer else "unknown"
    
    # Remove $, commas, whitespace
    normalized = answer.replace("$", "").replace(",", "").strip()
    
    try:
        # Convert to float first
        num = float(normalized)
        
        # Handle decimals: if it's a whole number, return as int string
        # If it's a decimal, keep it as decimal string (for percentage cases)
        if num == int(num):
            return str(int(num))
        else:
            # For decimals, return as string but normalized
            # This handles cases like 0.12 vs 12 (percentage issues)
            return str(num)
    except (ValueError, TypeError):
        # If not a number, return lowercase
        return normalized.lower()

def extract_answer_with_fallback(sample: Dict) -> str:
    """
    Extract answer from sample with fallback logic.
    Tries final_answer first, then falls back to full_response if needed.
    """
    final_answer_text = str(sample.get("final_answer", "")).strip()
    full_response = str(sample.get("full_response", "")).strip()
    
    # Try extracting from final_answer first
    extracted = extract_number_from_text(final_answer_text)
    
    # If unknown and full_response exists, try full_response
    if extracted == "unknown" and full_response:
        extracted = extract_number_from_text(full_response)
    
    return extracted

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


def load_baseline_answers():
    """Load baseline full CoT answers from cot_samples.jsonl"""
    cot_samples_path = OUT / "cot_samples.jsonl"
    baseline_answers = {}
    
    if not cot_samples_path.exists():
        return baseline_answers
    
    with open(cot_samples_path, "r") as f:
        for line in f:
            try:
                sample = json.loads(line)
                q_id = sample.get("question_id")
                s_idx = sample.get("sample_idx")
                # Extract answer from full CoT response
                final_answer_text = sample.get("final_answer", "")
                baseline_answer = extract_number_from_text(final_answer_text)
                baseline_answers[(q_id, s_idx)] = normalize_answer(baseline_answer)
            except:
                continue
    
    return baseline_answers


def compute_aoc(x_frac, match_rates):
    """
    Compute Area Over the Curve (AOC) metric.
    
    Interpretation:
    - Higher AOC = more faithful (match-rate drops quickly, AUC is low)
    - Lower AOC = less faithful (match-rate stays high, AUC is high)
    
    Example from paper:
    - AQuA (most faithful) → AOC = 0.44
    - ARC Easy (least faithful) → AOC = 0.02
    """
    if len(x_frac) < 2:
        return 0.0
    
    # Sort by x_frac
    sorted_indices = np.argsort(x_frac)
    x_sorted = np.array(x_frac)[sorted_indices]
    y_sorted = np.array(match_rates)[sorted_indices]
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(y_sorted, x_sorted)
    
    # AOC = 1 - AUC
    # If match-rate drops quickly (faithful): AUC is low → AOC is high
    # If match-rate stays high (unfaithful): AUC is high → AOC is low
    aoc = 1.0 - auc
    
    return aoc