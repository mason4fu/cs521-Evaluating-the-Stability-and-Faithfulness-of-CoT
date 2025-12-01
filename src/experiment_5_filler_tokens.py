"""Experiment 5: Filler Tokens Test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
import numpy as np

import config
from src.utils import (
    load_json, extract_number_from_text, normalize_answer,
    extract_gold_answer, OUT
)
from src.model_runner import ModelRunner, LocalModelRunner

COT_SAMPLES_PATH = OUT / "cot_samples.jsonl"
FILLER_RESULTS_PATH = OUT / "filler_tokens_results.csv"


def load_cot_samples() -> List[Dict]:
    """Load previously generated CoT samples"""
    samples = []
    with open(COT_SAMPLES_PATH, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def get_filler_tokens(num_tokens: int) -> str:
    """Generate filler token sequence (ellipsis dots)"""
    # Use " ..." as filler (space + ellipsis, repeated)
    return " ..." * num_tokens


def run_filler_tokens_test(
    runner,
    question: str,
    gold_answer: str,
    filler_length: int
) -> Dict:
    """Run filler tokens test"""
    # Create prompt with filler tokens instead of CoT
    prompt = config.COT_PROMPT_TEMPLATE.format(question=question)
    
    if filler_length > 0:
        filler = get_filler_tokens(filler_length)
        prompt += filler
    
    prompt += config.FINAL_ANSWER_PROMPT
    
    # Generate answer
    try:
        response = runner.generate(
            prompt,
            temperature=config.TEMPERATURE,
            top_p=config.NUCLEUS_P,
            max_tokens=config.MAX_TOKENS
        )
        
        # Extract answer
        final_answer = extract_number_from_text(response)
        
        # Check correctness
        is_correct = normalize_answer(final_answer) == normalize_answer(gold_answer)
        
        return {
            "filler_length": filler_length,
            "filler_tokens": get_filler_tokens(filler_length),
            "response": response,
            "final_answer": final_answer,
            "gold_answer": gold_answer,
            "is_correct": is_correct
        }
        
    except Exception as e:
        print(f"  ⚠️ Error with filler_length={filler_length}: {e}")
        return {
            "filler_length": filler_length,
            "error": str(e),
            "is_correct": False
        }


def run_experiment_5(
    use_local: bool = False,
    limit_questions: int = None
):
    """Run Experiment 5: Filler Tokens Test"""
    
    print("\n" + "="*60)
    print("Experiment 5: Filler Tokens Test")
    print("="*60)
    
    # Load CoT samples to get question set
    print(f"\n📂 Loading CoT samples from {COT_SAMPLES_PATH}...")
    samples = load_cot_samples()
    
    if not samples:
        raise ValueError(f"No CoT samples found at {COT_SAMPLES_PATH}. Run experiment 1 first.")
    
    print(f"   Loaded {len(samples)} samples")
    
    # Get unique questions
    questions_dict = {}
    cot_lengths = []
    
    for sample in samples:
        q_id = sample["question_id"]
        if q_id not in questions_dict:
            questions_dict[q_id] = {
                "question": sample["question"],
                "gold_answer": sample.get("gold_answer", "")
            }
        
        # Track CoT lengths for determining filler token ranges
        cot_text = sample.get("cot_text", "")
        if cot_text:
            # Approximate token count (rough: 1 token ≈ 4 characters)
            approx_tokens = len(cot_text) // 4
            cot_lengths.append(approx_tokens)
    
    questions = list(questions_dict.values())
    
    if limit_questions:
        questions = questions[:limit_questions]
    
    print(f"   Testing {len(questions)} unique questions")
    
    # Determine filler token lengths to test (reduced for speed)
    # Test at key points: 0, 10, 20, 30, 50, 75, 100, and percentiles if needed
    if cot_lengths:
        max_tokens = max(cot_lengths)
        # Reduced granularity: test every 10 tokens up to 50, then key points
        filler_lengths = [0, 10, 20, 30, 50]
        if max_tokens > 50:
            filler_lengths.extend([75, 100])
        if max_tokens > 100:
            # Add key percentiles: 50th, 75th, 90th, max
            percentiles = [50, 75, 90, 100]
            for p in percentiles:
                token_count = int(np.percentile(cot_lengths, p))
                if token_count > 100 and token_count not in filler_lengths:
                    filler_lengths.append(token_count)
            filler_lengths.sort()
    else:
        # Default reduced range
        filler_lengths = [0, 10, 20, 30, 50, 75, 100]
    
    print(f"   Testing filler lengths: {filler_lengths[:10]}... (and {len(filler_lengths)-10} more)")
    
    # Check existing results to skip already-processed tests
    existing_results = set()
    if FILLER_RESULTS_PATH.exists():
        try:
            existing_df = pd.read_csv(FILLER_RESULTS_PATH)
            if len(existing_df) > 0:
                for _, row in existing_df.iterrows():
                    question_text = str(row.get("question", ""))
                    filler_len = row.get("filler_length", "")
                    if question_text and filler_len != "":
                        existing_results.add((question_text, filler_len))
                print(f"   Found {len(existing_results)} existing test results")
        except Exception as e:
            print(f"   Could not read existing results: {e}")
    
    # Initialize model runner
    if use_local:
        runner = LocalModelRunner()
    else:
        runner = ModelRunner()
    
    # Run filler tokens tests
    all_results = []
    
    # Batch processing: collect all prompts first, then process in batches
    batch_size = 8
    all_prompts = []
    prompt_metadata = []
    
    for q_data in questions:
        question = q_data["question"]
        gold_answer = q_data["gold_answer"]
        
        if not gold_answer:
            continue
        
        for filler_length in filler_lengths:
            # Skip if already processed
            key = (question, filler_length)
            if key in existing_results:
                continue
            
            # Create prompt
            prompt = config.COT_PROMPT_TEMPLATE.format(question=question)
            if filler_length > 0:
                filler = get_filler_tokens(filler_length)
                prompt += filler
            prompt += config.FINAL_ANSWER_PROMPT
            
            all_prompts.append(prompt)
            prompt_metadata.append({
                "question": question,
                "gold_answer": gold_answer,
                "filler_length": filler_length
            })
    
    if not all_prompts:
        total_tests = len(questions) * len(filler_lengths)
        print(f"\n✅ All {total_tests} filler token tests already completed. Skipping.")
        return
    
    total_tests = len(questions) * len(filler_lengths)
    skipped = total_tests - len(all_prompts)
    print(f"\n🔄 Running filler tokens tests ({len(all_prompts)} new, {skipped} already exist) with batching...\n")
    print("💡 Tip: Press Ctrl+C to stop gracefully after current question completes\n")
    
    # Import stop flag function
    try:
        from run_all import get_should_stop, set_current_question, clear_current_question
    except ImportError:
        def get_should_stop(): return False
        def set_current_question(q): pass
        def clear_current_question(): pass
    
    # Track which question we're processing (for stop flag)
    # Group prompts by question to track progress
    question_to_prompts = {}
    for i, metadata in enumerate(prompt_metadata):
        q = metadata["question"]
        if q not in question_to_prompts:
            question_to_prompts[q] = []
        question_to_prompts[q].append(i)
    
    # Process in batches
    current_question = None
    with tqdm(total=len(all_prompts), desc="Filler tokens tests") as pbar:
        for batch_start in range(0, len(all_prompts), batch_size):
            # Check stop flag before each batch
            if get_should_stop():
                # Find which question this batch belongs to
                if batch_start < len(prompt_metadata):
                    batch_q = prompt_metadata[batch_start]["question"]
                    print(f"\n⏸️  Stop requested. Finished up to question: {batch_q}")
                break
            
            batch_end = min(batch_start + batch_size, len(all_prompts))
            batch_prompts = all_prompts[batch_start:batch_end]
            batch_metadata = prompt_metadata[batch_start:batch_end]
            
            # Track current question
            if batch_metadata:
                batch_q = batch_metadata[0]["question"]
                if batch_q != current_question:
                    if current_question:
                        clear_current_question()
                    set_current_question(batch_q)
                    current_question = batch_q
            
            try:
                if hasattr(runner, 'batch_generate') and len(batch_prompts) > 1:
                    # Use batch processing
                    responses = runner.batch_generate(
                        batch_prompts,
                        temperature=config.TEMPERATURE,
                        top_p=config.NUCLEUS_P,
                        max_tokens=config.MAX_TOKENS
                    )
                else:
                    # Fallback to sequential
                    responses = []
                    for prompt in batch_prompts:
                        response = runner.generate(
                            prompt,
                            temperature=config.TEMPERATURE,
                            top_p=config.NUCLEUS_P,
                            max_tokens=config.MAX_TOKENS
                        )
                        responses.append(response)
                
                # Process responses
                for i, response in enumerate(responses):
                    metadata = batch_metadata[i]
                    try:
                        final_answer = extract_number_from_text(response)
                        is_correct = normalize_answer(final_answer) == normalize_answer(metadata["gold_answer"])
                        
                        result = {
                            "filler_length": metadata["filler_length"],
                            "filler_tokens": get_filler_tokens(metadata["filler_length"]),
                            "response": response,
                            "final_answer": final_answer,
                            "gold_answer": metadata["gold_answer"],
                            "is_correct": is_correct,
                            "question": metadata["question"]
                        }
                        all_results.append(result)
                    except Exception as e:
                        all_results.append({
                            "filler_length": metadata["filler_length"],
                            "question": metadata["question"],
                            "error": str(e),
                            "is_correct": False
                        })
                    
                    pbar.update(1)
                
                # Check stop flag after each batch
                if get_should_stop():
                    print(f"\n⏸️  Stop requested. Completed batch for question: {current_question}")
                    break
                    
            except Exception as e:
                # Fallback to individual processing
                for metadata in batch_metadata:
                    try:
                        prompt = config.COT_PROMPT_TEMPLATE.format(question=metadata["question"])
                        if metadata["filler_length"] > 0:
                            prompt += get_filler_tokens(metadata["filler_length"])
                        prompt += config.FINAL_ANSWER_PROMPT
                        
                        response = runner.generate(
                            prompt,
                            temperature=config.TEMPERATURE,
                            top_p=config.NUCLEUS_P,
                            max_tokens=config.MAX_TOKENS
                        )
                        final_answer = extract_number_from_text(response)
                        is_correct = normalize_answer(final_answer) == normalize_answer(metadata["gold_answer"])
                        
                        all_results.append({
                            "filler_length": metadata["filler_length"],
                            "filler_tokens": get_filler_tokens(metadata["filler_length"]),
                            "response": response,
                            "final_answer": final_answer,
                            "gold_answer": metadata["gold_answer"],
                            "is_correct": is_correct,
                            "question": metadata["question"]
                        })
                    except Exception as e2:
                        all_results.append({
                            "filler_length": metadata["filler_length"],
                            "question": metadata["question"],
                            "error": str(e2),
                            "is_correct": False
                        })
                    pbar.update(1)
    
    if current_question:
        clear_current_question()
    
    # Convert to DataFrame
    if not all_results:
        print("\n❌ No results generated!")
        return
    
    df = pd.DataFrame(all_results)
    
    # Compute accuracy by filler length
    accuracy_by_length = df.groupby("filler_length")["is_correct"].mean().reset_index()
    accuracy_by_length.columns = ["filler_length", "accuracy"]
    
    # Append results to existing file
    if FILLER_RESULTS_PATH.exists() and len(existing_results) > 0:
        df.to_csv(FILLER_RESULTS_PATH, mode='a', header=False, index=False)
        full_df = pd.read_csv(FILLER_RESULTS_PATH)
        print(f"\n✅ Experiment 5 complete!")
        print(f"   Added {len(df)} new filler token tests")
        print(f"   Total filler tests: {len(full_df)}")
    else:
        df.to_csv(FILLER_RESULTS_PATH, index=False)
        print(f"\n✅ Experiment 5 complete!")
        print(f"   Results saved to: {FILLER_RESULTS_PATH}")
        print(f"   Total filler tests: {len(df)}")
    
    # Print summary
    print("\n   Accuracy by filler length (sample):")
    for _, row in accuracy_by_length.head(10).iterrows():
        print(f"     {row['filler_length']} tokens: {row['accuracy']:.4f}")
    
    # Baseline: without CoT (filler_length = 0)
    baseline_acc = accuracy_by_length[accuracy_by_length["filler_length"] == 0]["accuracy"].values
    if len(baseline_acc) > 0:
        print(f"\n   Baseline accuracy (no CoT): {baseline_acc[0]:.4f}")
        print(f"   (Paper expects no improvement with filler tokens)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Use local model runner")
    parser.add_argument("--limit-questions", type=int, default=None, help="Limit number of questions to test")
    args = parser.parse_args()
    
    run_experiment_5(use_local=args.local, limit_questions=args.limit_questions)

