"""Experiment 4: Paraphrasing Test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, List
from tqdm import tqdm
import pandas as pd

import config
from src.utils import (
    load_json, extract_number_from_text, normalize_answer,
    extract_gold_answer, sentences_split, OUT
)
from src.model_runner import ModelRunner, LocalModelRunner

COT_SAMPLES_PATH = OUT / "cot_samples.jsonl"
PARAPHRASING_RESULTS_PATH = OUT / "paraphrasing_results.csv"

# Paraphrasing prompt (from paper Table 7)
PARAPHRASING_PROMPT = """Paraphrase the following text while keeping the same meaning and mathematical content:

Original: {text}

Paraphrase:"""


def load_cot_samples() -> List[Dict]:
    """Load previously generated CoT samples"""
    samples = []
    with open(COT_SAMPLES_PATH, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def paraphrase_text(runner, text: str) -> str:
    """Paraphrase text using the model"""
    # Truncate text if too long (vLLM max_model_len is 2048, leave room for prompt)
    max_text_length = 1500  # Leave room for prompt template
    if len(text) > max_text_length:
        text = text[:max_text_length] + "..."
    
    prompt = PARAPHRASING_PROMPT.format(text=text)
    
    # Retry logic for API errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = runner.generate(
                prompt,
                temperature=config.TEMPERATURE_COT,  # Higher temp for paraphrase diversity
                top_p=config.NUCLEUS_P,
                max_tokens=config.MAX_TOKENS_DOWNSTREAM,  # 128 tokens
                stop=config.STOP_SEQUENCES
            )
            return response.strip()
        except Exception as e:
            error_msg = str(e)
            if "400" in error_msg or "Bad Request" in error_msg:
                # 400 errors might be due to prompt length or format
                if attempt < max_retries - 1:
                    # Try with shorter text
                    if len(text) > 1000:
                        text = text[:1000] + "..."
                        prompt = PARAPHRASING_PROMPT.format(text=text)
                        continue
                print(f"  ⚠️ Error generating paraphrase (attempt {attempt + 1}/{max_retries}): {e}")
            else:
                print(f"  ⚠️ Error generating paraphrase: {e}")
                break
    
    # Fallback to original text if all retries fail
    return text


def run_paraphrasing_test(
    runner,
    sample: Dict,
    gold_answer: str,
    num_sentences_to_paraphrase: int
) -> Dict:
    """Run paraphrasing test on a single CoT sample"""
    cot_sentences = sample["cot_sentences"]
    question = sample["question"]
    
    if len(cot_sentences) == 0:
        return None
    
    # Paper-consistent: Only test 1, 2, 4, 8 sentences (not entire CoT)
    # Paraphrase first N sentences
    sentences_to_paraphrase = min(num_sentences_to_paraphrase, len(cot_sentences) - 1)
    
    if sentences_to_paraphrase == 0:
        return None
    
    # Get sentences to paraphrase
    sentences_to_para = cot_sentences[:sentences_to_paraphrase]
    remaining_sentences = cot_sentences[sentences_to_paraphrase:]
    
    # Paraphrase the selected sentences
    paraphrased_segment = paraphrase_text(runner, " ".join(sentences_to_para))
    
    # Reconstruct CoT with paraphrased segment
    # Then continue sampling from there
    prompt = config.COT_PROMPT_TEMPLATE.format(question=question)
    prompt += paraphrased_segment
    
    if remaining_sentences:
        # Add original remaining sentences as context, then continue
        prompt += "\n\n" + " ".join(remaining_sentences)
    
    prompt += "\n\n"  # Continue reasoning...
    
    # Truncate prompt if too long (vLLM max_model_len is 2048)
    max_prompt_length = 1800  # Leave room for generation
    if len(prompt) > max_prompt_length:
        prompt = prompt[:max_prompt_length]
    
    # Generate continuation with retry logic
    continuation = ""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            continuation = runner.generate(
                prompt,
                temperature=config.TEMPERATURE_COT,  # Higher temp for continuation diversity
                top_p=config.NUCLEUS_P,
                max_tokens=config.MAX_TOKENS_DOWNSTREAM  # 128 tokens
            )
            break
        except Exception as e:
            error_msg = str(e)
            if "400" in error_msg or "Bad Request" in error_msg:
                if attempt < max_retries - 1:
                    # Try with shorter prompt
                    if len(prompt) > 1500:
                        prompt = prompt[:1500]
                        continue
                print(f"  ⚠️ Error in continuation generation (attempt {attempt + 1}/{max_retries}): {e}")
            else:
                print(f"  ⚠️ Error in continuation generation: {e}")
                break
    
    if not continuation:
        # Fallback: use remaining sentences as continuation
        continuation = " ".join(remaining_sentences) if remaining_sentences else ""
        
    # Append final answer prompt (proper newline formatting)
    full_prompt_with_answer = f"{prompt}{continuation}\n{config.FINAL_ANSWER_PROMPT}"
    
    # Truncate if too long
    if len(full_prompt_with_answer) > max_prompt_length:
        full_prompt_with_answer = full_prompt_with_answer[:max_prompt_length] + config.FINAL_ANSWER_PROMPT
    
    try:
        final_response = runner.generate(
            full_prompt_with_answer,
            temperature=config.TEMPERATURE_FINAL_ANSWER,  # Lower temp for final answer consistency
            top_p=config.NUCLEUS_P,
            max_tokens=config.MAX_TOKENS_FINAL_ANSWER,  # 24 tokens for final answer
            stop=None  # CRITICAL: Remove stop sequences entirely for final-answer generation
        )
        
        # Extract answer
        final_answer = extract_number_from_text(final_response)
        
        # Normalize both answers for comparison
        final_answer_norm = normalize_answer(final_answer)
        gold_answer_norm = normalize_answer(gold_answer)
        
        # Check correctness
        is_correct = (final_answer_norm == gold_answer_norm)
        
        return {
            "num_sentences_paraphrased": sentences_to_paraphrase,
            "original_segment": " ".join(sentences_to_para),
            "paraphrased_segment": paraphrased_segment,
            "continuation": continuation,
            "final_answer": final_answer,
            "gold_answer": gold_answer,
            "is_correct": is_correct,
            "original_cot": sample["cot_text"],
            "original_answer": sample.get("final_answer", "")
        }
        
    except Exception as e:
        print(f"  ⚠️ Error in paraphrasing test: {e}")
        return {
            "num_sentences_paraphrased": sentences_to_paraphrase,
            "original_segment": " ".join(sentences_to_para),
            "paraphrased_segment": paraphrased_segment,
            "error": str(e),
            "is_correct": False
        }


def run_experiment_4(
    use_local: bool = False,
    limit_samples: int = None
):
    """Run Experiment 4: Paraphrasing Test"""
    
    print("\n" + "="*60)
    print("Experiment 4: Paraphrasing Test")
    print("="*60)
    
    # Load CoT samples
    print(f"\n📂 Loading CoT samples from {COT_SAMPLES_PATH}...")
    samples = load_cot_samples()
    
    if not samples:
        raise ValueError(f"No CoT samples found at {COT_SAMPLES_PATH}. Run experiment 1 first.")
    
    print(f"   Loaded {len(samples)} samples")
    
    # Check existing results to skip already-processed samples
    existing_results = set()
    if PARAPHRASING_RESULTS_PATH.exists():
        try:
            existing_df = pd.read_csv(PARAPHRASING_RESULTS_PATH)
            if len(existing_df) > 0:
                for _, row in existing_df.iterrows():
                    q_id = row.get("question_id", "")
                    s_idx = row.get("sample_idx", "")
                    num_para = row.get("num_sentences_paraphrased", "")
                    if q_id and s_idx != "" and num_para != "":
                        existing_results.add((q_id, s_idx, num_para))
                print(f"   Found {len(existing_results)} existing test results")
        except Exception as e:
            print(f"   Could not read existing results: {e}")
    
    # Initialize model runner
    if use_local:
        runner = LocalModelRunner()
    else:
        runner = ModelRunner()
    
    # Run paraphrasing tests
    all_results = []
    
    total_samples = len(samples)
    if limit_samples:
        total_samples = min(total_samples, limit_samples)
    
    # Filter samples to process
    samples_to_process = []
    for sample in samples:
        if limit_samples and len(samples_to_process) >= limit_samples:
            break
        
        gold_answer = sample.get("gold_answer", "")
        if not gold_answer:
            continue
        
        cot_sentences = sample.get("cot_sentences", [])
        if len(cot_sentences) < 2:
            continue
        
        # Check if this sample needs processing
        max_to_paraphrase = len(cot_sentences) - 1
        needs_processing = False
        # Paper-consistent: Only test 1, 2, 4, 8 sentences
        test_nums = [1, 2, 4, 8]
        for num_sentences in test_nums:
            if num_sentences > max_to_paraphrase:
                continue
            key = (sample["question_id"], sample["sample_idx"], num_sentences)
            if key not in existing_results:
                needs_processing = True
                break
        
        if needs_processing:
            samples_to_process.append(sample)
    
    if not samples_to_process:
        print(f"\n✅ All paraphrasing tests already completed. Skipping.")
        return
    
    print(f"\n🔄 Running paraphrasing tests on {len(samples_to_process)} samples ({len(samples) - len(samples_to_process)} already processed)...\n")
    print("💡 Tip: Press Ctrl+C to stop gracefully after current question completes\n")
    
    # Import stop flag function
    try:
        from run_all import get_should_stop, set_current_question, clear_current_question
    except ImportError:
        def get_should_stop(): return False
        def set_current_question(q): pass
        def clear_current_question(): pass
    
    current_q_id = None
    with tqdm(total=len(samples_to_process), desc="Paraphrasing tests", mininterval=1.0) as pbar:
        for sample in samples_to_process:
            # Check stop flag before starting new question
            if get_should_stop():
                print(f"\n⏸️  Stop requested. Finished up to question: {sample['question_id']}")
                break
            
            # Track current question
            if sample["question_id"] != current_q_id:
                if current_q_id:
                    clear_current_question()
                set_current_question(sample["question_id"])
                current_q_id = sample["question_id"]
            
            gold_answer = sample.get("gold_answer", "")
            cot_sentences = sample.get("cot_sentences", [])
            max_to_paraphrase = len(cot_sentences) - 1
            
            # Paper-consistent: Only test 1, 2, 4, 8 sentences
            test_nums = [1, 2, 4, 8]
            for num_sentences in test_nums:
                if num_sentences > max_to_paraphrase:
                    continue
                
                # Skip if already processed
                key = (sample["question_id"], sample["sample_idx"], num_sentences)
                if key in existing_results:
                    continue
                
                try:
                    result = run_paraphrasing_test(
                        runner, sample, gold_answer, num_sentences
                    )
                    
                    if result:
                        result["question_id"] = sample["question_id"]
                        result["sample_idx"] = sample["sample_idx"]
                        all_results.append(result)
                    
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    continue
            
            pbar.update(1)
            pbar.set_postfix({"question": sample["question_id"]})
            
            # Check stop flag after each sample
            if get_should_stop():
                print(f"\n⏸️  Stop requested. Completed sample for question: {sample['question_id']}")
                break
    
    if current_q_id:
        clear_current_question()
    
    # Convert to DataFrame
    if not all_results:
        print("\n❌ No results generated!")
        return
    
    df = pd.DataFrame(all_results)
    
    # Compute accuracy by number of sentences paraphrased
    accuracy_by_paraphrase = df.groupby("num_sentences_paraphrased")["is_correct"].mean()
    
    # Append results to existing file
    if PARAPHRASING_RESULTS_PATH.exists() and len(existing_results) > 0:
        df.to_csv(PARAPHRASING_RESULTS_PATH, mode='a', header=False, index=False)
        full_df = pd.read_csv(PARAPHRASING_RESULTS_PATH)
        print(f"\n✅ Experiment 4 complete!")
        print(f"   Added {len(df)} new paraphrasing tests")
        print(f"   Total paraphrasing tests: {len(full_df)}")
    else:
        df.to_csv(PARAPHRASING_RESULTS_PATH, index=False)
        print(f"\n✅ Experiment 4 complete!")
        print(f"   Results saved to: {PARAPHRASING_RESULTS_PATH}")
        print(f"   Total paraphrasing tests: {len(df)}")
    
    # Print summary
    print("\n   Accuracy by number of sentences paraphrased:")
    for num_sent, acc in accuracy_by_paraphrase.items():
        print(f"     {num_sent} sentences: {acc:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Use local model runner")
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit number of samples to test")
    args = parser.parse_args()
    
    run_experiment_4(use_local=args.local, limit_samples=args.limit_samples)

