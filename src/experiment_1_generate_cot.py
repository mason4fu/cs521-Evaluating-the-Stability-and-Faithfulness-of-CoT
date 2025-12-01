"""Experiment 1: Generate CoT samples for each GSM8K question"""
import json
import time
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset
from tqdm import tqdm

import sys
from pathlib import Path

# Add root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.utils import (
    extract_gold_answer, sentences_split, append_jsonl,
    DATA, OUT
)
from src.model_runner import ModelRunner, LocalModelRunner

COT_SAMPLES_PATH = OUT / "cot_samples.jsonl"
GSM8K_SUBSET_PATH = DATA / "gsm8k_subset.json"


def generate_cot_sample(
    runner,
    question: str,
    question_id: str,
    sample_idx: int
) -> Dict:
    """Generate a single CoT sample"""
    # Create prompt matching paper Table 1
    # Use a prompt that encourages step-by-step reasoning
    prompt = config.COT_PROMPT_TEMPLATE.format(question=question)
    # Don't add final answer prompt yet - let model generate full reasoning
    
    # Generate with sampling parameters from paper
    response = runner.generate(
        prompt,
        temperature=config.TEMPERATURE,
        top_p=config.NUCLEUS_P,
        max_tokens=config.MAX_TOKENS,
        stop=None  # Let it generate full response
    )
    
    # Split into CoT and final answer
    # Look for "The answer is" or similar patterns
    response_lower = response.lower()
    cot_text = response
    final_answer_text = ""
    
    # Try to find answer separator
    answer_markers = ["the answer is", "answer:", "final answer:", "answer is"]
    for marker in answer_markers:
        if marker in response_lower:
            idx = response_lower.find(marker)
            cot_text = response[:idx].strip()
            final_answer_text = response[idx + len(marker):].strip()
            break
    
    # If no marker found, try to extract last number as answer
    if not final_answer_text:
        from src.utils import extract_number_from_text as extract_num
        final_answer_text = extract_num(response)
        # Assume everything before the number is CoT
        if final_answer_text != "unknown":
            # Try to find where the number appears
            import re
            numbers = list(re.finditer(r"[-+]?\d*\.?\d+", response))
            if numbers:
                last_num_match = numbers[-1]
                cot_text = response[:last_num_match.start()].strip()
    
    # Sentence-split CoT (as in paper)
    cot_sentences = sentences_split(cot_text) if cot_text else []
    
    return {
        "question_id": question_id,
        "sample_idx": sample_idx,
        "question": question,
        "prompt": prompt,
        "full_response": response,
        "cot_text": cot_text,
        "cot_sentences": cot_sentences,
        "num_sentences": len(cot_sentences),
        "final_answer": final_answer_text,
        "timestamp": time.time()
    }


def load_gsm8k_subset(num_questions: int = None) -> List[Dict]:
    """Load GSM8K dataset subset"""
    # Determine how many questions we need
    num_questions = num_questions or (config.TEST_NUM_QUESTIONS if config.TEST_MODE else None)
    
    # Check if cached subset exists and is sufficient
    if GSM8K_SUBSET_PATH.exists():
        with open(GSM8K_SUBSET_PATH, "r") as f:
            cached_subset = json.load(f)
        
        # If we need more questions than cached, or need all questions, load from full dataset
        if num_questions is None:
            # Need all questions - load from full dataset
            print(f"Cached subset has {len(cached_subset)} questions, but need all questions. Loading from full dataset...")
        elif num_questions > len(cached_subset):
            # Need more questions than cached - load from full dataset
            print(f"Cached subset has {len(cached_subset)} questions, but need {num_questions}. Loading from full dataset...")
        else:
            # Cached subset is sufficient
            if len(cached_subset) == num_questions:
                return cached_subset
            return cached_subset[:num_questions]
    
    # Load from datasets library
    print(f"Loading GSM8K dataset...")
    ds = load_dataset("gsm8k", "main")["test"]  # Use test set for evaluation
    
    if num_questions:
        dataset = ds.shuffle(seed=config.GSM8K_SEED).select(range(num_questions)).to_list()
    else:
        dataset = ds.shuffle(seed=config.GSM8K_SEED).to_list()
    
    # Format as list of dicts
    subset = []
    for idx, ex in enumerate(dataset):
        question_id = ex.get("id", f"gsm8k_{idx}")
        subset.append({
            "id": question_id,
            "question": ex["question"],
            "answer": ex["answer"],
            "gold_answer": extract_gold_answer(ex["answer"])
        })
    
    # Save subset
    GSM8K_SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GSM8K_SUBSET_PATH, "w") as f:
        json.dump(subset, f, indent=2)
    
    return subset


def run_experiment_1(
    num_questions: int = None,
    num_samples: int = None,
    use_local: bool = False
):
    """Run Experiment 1: Generate CoT samples"""
    
    # Determine parameters
    if config.TEST_MODE:
        num_questions = num_questions or config.TEST_NUM_QUESTIONS
        num_samples = num_samples or config.TEST_NUM_SAMPLES
        print(f"🧪 TEST MODE: {num_questions} questions, {num_samples} samples each")
    else:
        num_questions = num_questions
        num_samples = num_samples or config.NUM_SAMPLES_PER_QUESTION
    
    # Load questions
    questions = load_gsm8k_subset(num_questions)
    print(f"\n📚 Loaded {len(questions)} GSM8K questions")
    
    # Initialize model runner
    if use_local:
        print("🔧 Using local model runner (transformers)")
        runner = LocalModelRunner()
    else:
        print("🔧 Using VM model runner (vLLM)")
        runner = ModelRunner()
    
    # Load existing samples to check what's already done
    existing_samples = {}
    if COT_SAMPLES_PATH.exists():
        print(f"\n📂 Checking existing CoT samples...")
        with open(COT_SAMPLES_PATH, "r") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    q_id = sample.get("question_id")
                    s_idx = sample.get("sample_idx")
                    if q_id not in existing_samples:
                        existing_samples[q_id] = set()
                    existing_samples[q_id].add(s_idx)
                except:
                    continue
        
        existing_count = sum(len(samples) for samples in existing_samples.values())
        print(f"   Found {existing_count} existing samples for {len(existing_samples)} questions")
    
    # Filter questions to only generate missing samples
    questions_to_process = []
    samples_to_generate = 0
    for q_data in questions:
        question_id = q_data["id"]
        existing_for_q = existing_samples.get(question_id, set())
        missing_samples = [i for i in range(num_samples) if i not in existing_for_q]
        
        if missing_samples:
            questions_to_process.append((q_data, missing_samples))
            samples_to_generate += len(missing_samples)
    
    if samples_to_generate == 0:
        print(f"\n✅ All {len(questions) * num_samples} samples already exist. Skipping generation.")
        return
    
    print(f"\n🔄 Generating {samples_to_generate} new CoT samples ({len(questions) * num_samples - samples_to_generate} already exist)...\n")
    print("💡 Tip: Press Ctrl+C to stop gracefully after current question completes\n")
    
    # Import stop flag function
    try:
        from run_all import get_should_stop, set_current_question, clear_current_question
    except ImportError:
        # If imported directly, create dummy functions
        def get_should_stop(): return False
        def set_current_question(q): pass
        def clear_current_question(): pass
    
    with tqdm(total=samples_to_generate, desc="Generating CoT samples") as pbar:
        for q_data, missing_samples in questions_to_process:
            # Check stop flag before starting new question
            if get_should_stop():
                print(f"\n⏸️  Stop requested. Finished up to question: {q_data['id']}")
                break
            
            question_id = q_data["id"]
            question = q_data["question"]
            set_current_question(question_id)
            
            for sample_idx in missing_samples:
                # Check stop flag before each sample
                if get_should_stop():
                    print(f"\n⏸️  Stop requested. Finishing current question: {question_id}")
                    # Finish remaining samples for current question
                    remaining = [s for s in missing_samples if s >= sample_idx]
                    for remaining_idx in remaining:
                        try:
                            sample = generate_cot_sample(
                                runner, question, question_id, remaining_idx
                            )
                            sample["gold_answer"] = q_data["gold_answer"]
                            sample["gold_answer_full"] = q_data["answer"]
                            append_jsonl(COT_SAMPLES_PATH, sample)
                            pbar.update(1)
                        except Exception as e:
                            print(f"\n❌ Error generating sample {remaining_idx} for {question_id}: {e}")
                            pbar.update(1)
                    break
                
                try:
                    sample = generate_cot_sample(
                        runner, question, question_id, sample_idx
                    )
                    # Add gold answer info
                    sample["gold_answer"] = q_data["gold_answer"]
                    sample["gold_answer_full"] = q_data["answer"]
                    
                    append_jsonl(COT_SAMPLES_PATH, sample)
                    pbar.update(1)
                    pbar.set_postfix({"question": question_id, "sample": sample_idx})
                    
                except Exception as e:
                    print(f"\n❌ Error generating sample {sample_idx} for {question_id}: {e}")
                    pbar.update(1)
                    continue
            
            clear_current_question()
            
            # Check stop flag after completing question
            if get_should_stop():
                print(f"\n⏸️  Stop requested. Completed question: {question_id}")
                break
    
    print(f"\n✅ Experiment 1 complete!")
    print(f"   Results saved to: {COT_SAMPLES_PATH}")
    print(f"   Total samples: {total_samples}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--local", action="store_true", help="Use local model runner")
    args = parser.parse_args()
    
    run_experiment_1(
        num_questions=args.num_questions,
        num_samples=args.num_samples,
        use_local=args.local
    )

