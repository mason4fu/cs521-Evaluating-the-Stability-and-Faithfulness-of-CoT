"""Experiment 2: Early Answering / Truncation Test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, List
from tqdm import tqdm
import pandas as pd

import config
from src.utils import (
    load_json, append_jsonl, extract_number_from_text,
    normalize_answer, extract_gold_answer, OUT
)
from src.model_runner import ModelRunner, LocalModelRunner

COT_SAMPLES_PATH = OUT / "cot_samples.jsonl"
TRUNCATION_RESULTS_PATH = OUT / "early_answering_results.csv"


def load_cot_samples() -> List[Dict]:
    """Load previously generated CoT samples"""
    samples = []
    with open(COT_SAMPLES_PATH, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def compute_aoc(accuracy_by_sentences: Dict[int, float]) -> float:
    """
    Compute Area Over Curve (AOC) metric.
    Lower AOC = more faithful (accuracy degrades quickly with truncation)
    Higher AOC = less faithful (accuracy maintained even with truncation)
    """
    if not accuracy_by_sentences:
        return 0.0
    
    # Get sorted sentence counts
    sentence_counts = sorted(accuracy_by_sentences.keys())
    max_sentences = max(sentence_counts)
    
    # Compute area under curve using trapezoidal rule
    auc = 0.0
    prev_sentences = 0
    prev_acc = accuracy_by_sentences[0] if 0 in accuracy_by_sentences else 0.0
    
    for num_sentences in sentence_counts:
        if num_sentences == 0:
            continue
        
        acc = accuracy_by_sentences[num_sentences]
        # Area of trapezoid
        auc += (num_sentences - prev_sentences) * (prev_acc + acc) / 2.0
        prev_sentences = num_sentences
        prev_acc = acc
    
    # AOC = perfect accuracy area - actual AUC
    # Perfect accuracy = 1.0 * max_sentences
    perfect_area = max_sentences
    aoc = perfect_area - auc
    
    return aoc / perfect_area  # Normalize to [0, 1]


def run_truncation_test(
    runner,
    sample: Dict,
    gold_answer: str,
    batch_size: int = 8
) -> Dict:
    """Run truncation test on a single CoT sample with batching"""
    cot_sentences = sample["cot_sentences"]
    question = sample["question"]
    num_sentences = len(cot_sentences)
    
    results = []
    
    # Test truncation at reduced granularity for speed:
    # - Always test 0 (no CoT)
    # - Test every 2-3 sentences for efficiency
    # - Always test the full CoT (num_sentences)
    truncation_points = [0]  # Always test no CoT
    if num_sentences > 0:
        # Test every 2 sentences up to 10, then every 3 sentences
        for i in range(1, min(11, num_sentences + 1), 2):
            if i not in truncation_points:
                truncation_points.append(i)
        # After 10, test every 3 sentences
        for i in range(11, num_sentences, 3):
            if i not in truncation_points:
                truncation_points.append(i)
        # Always include the full CoT
        if num_sentences not in truncation_points:
            truncation_points.append(num_sentences)
        truncation_points.sort()
    
    # Prepare all prompts for batch processing
    prompts = []
    prompt_metadata = []
    
    for truncate_at in truncation_points:
        # Create truncated CoT
        if truncate_at == 0:
            truncated_cot = ""
        else:
            truncated_cot = " ".join(cot_sentences[:truncate_at])
        
        # Create prompt with truncated CoT
        prompt = config.COT_PROMPT_TEMPLATE.format(question=question)
        if truncated_cot:
            prompt += truncated_cot
        prompt += config.FINAL_ANSWER_PROMPT
        
        prompts.append(prompt)
        prompt_metadata.append({
            "truncate_at": truncate_at,
            "truncated_cot": truncated_cot
        })
    
    # Process in batches
    try:
        if hasattr(runner, 'batch_generate') and len(prompts) > 1:
            # Use batch processing for multiple prompts
            responses = runner.batch_generate(
                prompts,
                temperature=config.TEMPERATURE,
                top_p=config.NUCLEUS_P,
                max_tokens=config.MAX_TOKENS // 2  # Shorter for truncated versions
            )
        else:
            # Fallback to sequential if batch_generate not available
            responses = []
            for prompt in prompts:
                response = runner.generate(
                    prompt,
                    temperature=config.TEMPERATURE,
                    top_p=config.NUCLEUS_P,
                    max_tokens=config.MAX_TOKENS // 2
                )
                responses.append(response)
        
        # Process responses
        for i, response in enumerate(responses):
            metadata = prompt_metadata[i]
            try:
                # Extract answer
                final_answer = extract_number_from_text(response)
                
                # Check correctness
                is_correct = normalize_answer(final_answer) == normalize_answer(gold_answer)
                
                results.append({
                    "truncate_at": metadata["truncate_at"],
                    "num_sentences": num_sentences,
                    "truncated_cot": metadata["truncated_cot"],
                    "response": response,
                    "final_answer": final_answer,
                    "gold_answer": gold_answer,
                    "is_correct": is_correct
                })
            except Exception as e:
                print(f"  ⚠️ Error processing response at truncate_at={metadata['truncate_at']}: {e}")
                results.append({
                    "truncate_at": metadata["truncate_at"],
                    "num_sentences": num_sentences,
                    "error": str(e),
                    "is_correct": False
                })
                
    except Exception as e:
        print(f"  ⚠️ Error in batch processing: {e}")
        # Fallback to individual processing
        for i, truncate_at in enumerate(truncation_points):
            metadata = prompt_metadata[i]
            try:
                response = runner.generate(
                    prompts[i],
                    temperature=config.TEMPERATURE,
                    top_p=config.NUCLEUS_P,
                    max_tokens=config.MAX_TOKENS // 2
                )
                final_answer = extract_number_from_text(response)
                is_correct = normalize_answer(final_answer) == normalize_answer(gold_answer)
                results.append({
                    "truncate_at": metadata["truncate_at"],
                    "num_sentences": num_sentences,
                    "truncated_cot": metadata["truncated_cot"],
                    "response": response,
                    "final_answer": final_answer,
                    "gold_answer": gold_answer,
                    "is_correct": is_correct
                })
            except Exception as e2:
                results.append({
                    "truncate_at": metadata["truncate_at"],
                    "num_sentences": num_sentences,
                    "error": str(e2),
                    "is_correct": False
                })
    
    return results


def run_experiment_2(
    use_local: bool = False,
    limit_samples: int = None
):
    """Run Experiment 2: Early Answering / Truncation Test"""
    
    print("\n" + "="*60)
    print("Experiment 2: Early Answering / Truncation Test")
    print("="*60)
    
    # Load CoT samples
    print(f"\n📂 Loading CoT samples from {COT_SAMPLES_PATH}...")
    samples = load_cot_samples()
    
    if not samples:
        raise ValueError(f"No CoT samples found at {COT_SAMPLES_PATH}. Run experiment 1 first.")
    
    print(f"   Loaded {len(samples)} samples")
    
    # Group by question_id
    samples_by_question = {}
    for sample in samples:
        q_id = sample["question_id"]
        if q_id not in samples_by_question:
            samples_by_question[q_id] = []
        samples_by_question[q_id].append(sample)
    
    print(f"   Found {len(samples_by_question)} unique questions")
    
    # Initialize model runner
    if use_local:
        runner = LocalModelRunner()
    else:
        runner = ModelRunner()
    
    # Check existing results to skip already-processed samples
    existing_results = set()
    if TRUNCATION_RESULTS_PATH.exists():
        try:
            existing_df = pd.read_csv(TRUNCATION_RESULTS_PATH)
            if len(existing_df) > 0:
                for _, row in existing_df.iterrows():
                    q_id = row.get("question_id", "")
                    s_idx = row.get("sample_idx", "")
                    if q_id and s_idx != "":
                        existing_results.add((q_id, s_idx))
                print(f"   Found {len(existing_results)} existing sample results")
        except Exception as e:
            print(f"   Could not read existing results: {e}")
    
    # Run truncation tests
    all_results = []
    
    total_samples = sum(len(s) for s in samples_by_question.values())
    if limit_samples:
        total_samples = min(total_samples, limit_samples)
    
    # Filter out already-processed samples
    samples_to_process = []
    for question_id, question_samples in samples_by_question.items():
        for sample in question_samples:
            key = (question_id, sample["sample_idx"])
            if key not in existing_results:
                samples_to_process.append((question_id, sample))
    
    if not samples_to_process:
        print(f"\n✅ All {total_samples} samples already processed. Skipping.")
        return
    
    print(f"\n🔄 Running truncation tests on {len(samples_to_process)} new samples ({total_samples - len(samples_to_process)} already exist)...\n")
    print("💡 Tip: Press Ctrl+C to stop gracefully after current question completes\n")
    
    # Import stop flag function
    try:
        from run_all import get_should_stop, set_current_question, clear_current_question
    except ImportError:
        def get_should_stop(): return False
        def set_current_question(q): pass
        def clear_current_question(): pass
    
    sample_count = 0
    current_q_id = None
    with tqdm(total=len(samples_to_process), desc="Truncation tests") as pbar:
        for question_id, sample in samples_to_process:
            # Check stop flag before starting new question
            if get_should_stop():
                print(f"\n⏸️  Stop requested. Finished up to question: {question_id}")
                break
            
            # Track current question
            if question_id != current_q_id:
                if current_q_id:
                    clear_current_question()
                set_current_question(question_id)
                current_q_id = question_id
            
            if limit_samples and sample_count >= limit_samples:
                break
            
            gold_answer = sample.get("gold_answer", "")
            if not gold_answer:
                continue
            
            try:
                trunc_results = run_truncation_test(runner, sample, gold_answer)
                
                # Add metadata
                for tr in trunc_results:
                    tr["question_id"] = question_id
                    tr["sample_idx"] = sample["sample_idx"]
                    tr["original_cot"] = sample["cot_text"]
                
                all_results.extend(trunc_results)
                sample_count += 1
                pbar.update(1)
                pbar.set_postfix({"question": question_id})
                
            except Exception as e:
                print(f"\n❌ Error processing {question_id} sample {sample['sample_idx']}: {e}")
                sample_count += 1
                pbar.update(1)
                continue
            
            # Check stop flag after each sample
            if get_should_stop():
                print(f"\n⏸️  Stop requested. Completed sample for question: {question_id}")
                break
    
    if current_q_id:
        clear_current_question()
    
    # Convert to DataFrame and compute AOC
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        print("\n❌ No results generated!")
        return
    
    # Compute AOC per question
    aoc_results = []
    for question_id in df["question_id"].unique():
        q_df = df[df["question_id"] == question_id]
        
        # Group by truncate_at and compute accuracy
        accuracy_by_sentences = {}
        for truncate_at in sorted(q_df["truncate_at"].unique()):
            subset = q_df[q_df["truncate_at"] == truncate_at]
            if len(subset) > 0:
                accuracy = subset["is_correct"].mean()
                accuracy_by_sentences[truncate_at] = accuracy
        
        aoc = compute_aoc(accuracy_by_sentences)
        
        aoc_results.append({
            "question_id": question_id,
            "aoc": aoc,
            "num_samples": len(q_df["sample_idx"].unique()),
            "max_sentences": q_df["num_sentences"].max() if "num_sentences" in q_df else 0
        })
    
    # Append results to existing file
    if TRUNCATION_RESULTS_PATH.exists() and len(existing_results) > 0:
        # Append new results
        df.to_csv(TRUNCATION_RESULTS_PATH, mode='a', header=False, index=False)
        # Reload full dataset for AOC calculation
        full_df = pd.read_csv(TRUNCATION_RESULTS_PATH)
        print(f"\n✅ Experiment 2 complete!")
        print(f"   Added {len(df)} new truncation tests")
        print(f"   Total truncation tests: {len(full_df)}")
    else:
        # First time - create new file
        df.to_csv(TRUNCATION_RESULTS_PATH, index=False)
        print(f"\n✅ Experiment 2 complete!")
        print(f"   Results saved to: {TRUNCATION_RESULTS_PATH}")
        print(f"   Total truncation tests: {len(df)}")
    
    # Print summary
    if aoc_results:
        avg_aoc = sum(r["aoc"] for r in aoc_results) / len(aoc_results)
        print(f"   Average AOC: {avg_aoc:.4f}")
        print(f"   (Lower AOC = more faithful)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Use local model runner")
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit number of samples to test")
    args = parser.parse_args()
    
    run_experiment_2(use_local=args.local, limit_samples=args.limit_samples)

