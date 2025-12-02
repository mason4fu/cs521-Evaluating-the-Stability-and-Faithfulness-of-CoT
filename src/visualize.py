"""Generate plots matching paper figures - Updated to use match-rate metric"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import config
from src.utils import OUT, FIG, normalize_answer, extract_number_from_text, compute_aoc


def load_baseline_answers():
    """Load baseline full CoT answers from cot_samples.jsonl"""
    cot_samples_path = OUT / "cot_samples.jsonl"
    baseline_answers = {}
    
    if not cot_samples_path.exists():
        print(f"⚠️  {cot_samples_path} not found. Cannot compute match-rate.")
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


def plot_truncation_curve():
    """Plot truncation curves (Figure 3 in paper) - Match-rate vs CoT fraction"""
    results_path = OUT / "early_answering_results.csv"
    cot_samples_path = OUT / "cot_samples.jsonl"
    
    if not results_path.exists():
        print(f"⚠️  {results_path} not found. Run experiment 2 first.")
        return
    
    df = pd.read_csv(results_path)
    
    # Load baseline answers
    baseline_answers = load_baseline_answers()
    
    # Compute match-rate: perturbed_answer == baseline_full_cot_answer
    df["baseline_answer"] = df.apply(
        lambda row: baseline_answers.get((row["question_id"], row["sample_idx"]), ""),
        axis=1
    )
    df["perturbed_answer_norm"] = df["final_answer"].apply(
        lambda x: normalize_answer(extract_number_from_text(str(x)))
    )
    df["matches_full"] = df["perturbed_answer_norm"] == df["baseline_answer"]
    
    # Filter invalid samples: exclude samples where num_sentences < 2
    df = df[df["num_sentences"] >= 2].copy()
    
    if len(df) == 0:
        print(f"⚠️  No valid samples (num_sentences >= 2) found. Cannot plot.")
        return
    
    # Compute truncation fraction: truncate_at / total_sentences
    df["frac"] = df["truncate_at"] / df["num_sentences"].replace(0, 1)  # Avoid division by zero
    df["frac"] = df["frac"].clip(0, 1)  # Ensure [0, 1]
    
    # Group by fraction and compute match-rate (explicitly sort)
    match_rate_by_frac = df.groupby("frac")["matches_full"].mean().reset_index()
    match_rate_by_frac.columns = ["cot_fraction", "match_rate"]
    match_rate_by_frac = match_rate_by_frac.sort_values(by="cot_fraction")
    
    # Compute AOC
    aoc = compute_aoc(match_rate_by_frac["cot_fraction"].values, 
                     match_rate_by_frac["match_rate"].values)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        match_rate_by_frac["cot_fraction"],
        match_rate_by_frac["match_rate"],
        marker="o",
        linewidth=2,
        markersize=6
    )
    plt.xlabel("CoT Fraction", fontsize=12)
    plt.ylabel("Matching Probability", fontsize=12)
    plt.title(f"Early Answering / Truncation Test (AOC = {aoc:.3f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    
    out_path = FIG / "truncation_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path} (AOC = {aoc:.3f})")


def plot_mistakes_curve():
    """Plot mistake impact curves (Figure 4 in paper) - Match-rate vs mistake location fraction"""
    results_path = OUT / "adding_mistakes_results.csv"
    
    if not results_path.exists():
        print(f"⚠️  {results_path} not found. Run experiment 3 first.")
        return
    
    df = pd.read_csv(results_path)
    
    # Load baseline answers
    baseline_answers = load_baseline_answers()
    
    # Compute match-rate: perturbed_answer == baseline_full_cot_answer
    df["baseline_answer"] = df.apply(
        lambda row: baseline_answers.get((row["question_id"], row["sample_idx"]), ""),
        axis=1
    )
    df["perturbed_answer_norm"] = df["final_answer"].apply(
        lambda x: normalize_answer(extract_number_from_text(str(x)))
    )
    df["matches_full"] = df["perturbed_answer_norm"] == df["baseline_answer"]
    
    # Load CoT samples to get total sentence count for each sample
    cot_samples_path = OUT / "cot_samples.jsonl"
    sample_sentence_counts = {}
    if cot_samples_path.exists():
        with open(cot_samples_path, "r") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    q_id = sample.get("question_id")
                    s_idx = sample.get("sample_idx")
                    num_sentences = sample.get("num_sentences", 1)
                    sample_sentence_counts[(q_id, s_idx)] = num_sentences
                except:
                    continue
    
    # Compute mistake location as fraction of CoT
    df["total_sentences"] = df.apply(
        lambda row: sample_sentence_counts.get((row["question_id"], row["sample_idx"]), 1),
        axis=1
    )
    
    # Filter invalid samples: exclude samples where total_sentences < 2
    df = df[df["total_sentences"] >= 2].copy()
    
    if len(df) == 0:
        print(f"⚠️  No valid samples (total_sentences >= 2) found. Cannot plot.")
        return
    
    # Fix mistake fraction: sentence_idx is 0-indexed, use proper denominator
    denom = df["total_sentences"].apply(lambda x: max(x - 1, 1))
    df["mistake_frac"] = df["sentence_idx"] / denom
    df["mistake_frac"] = df["mistake_frac"].clip(0, 1)
    
    # Group by mistake fraction and compute match-rate (explicitly sort)
    match_rate_by_frac = df.groupby("mistake_frac")["matches_full"].mean().reset_index()
    match_rate_by_frac.columns = ["mistake_location_frac", "match_rate"]
    match_rate_by_frac = match_rate_by_frac.sort_values(by="mistake_location_frac")
    
    # Compute AOC
    aoc = compute_aoc(match_rate_by_frac["mistake_location_frac"].values,
                     match_rate_by_frac["match_rate"].values)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        match_rate_by_frac["mistake_location_frac"],
        match_rate_by_frac["match_rate"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="red"
    )
    plt.xlabel("Mistake Location (Fraction of CoT)", fontsize=12)
    plt.ylabel("Matching Probability", fontsize=12)
    plt.title(f"Adding Mistakes Test (AOC = {aoc:.3f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    
    out_path = FIG / "mistakes_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path} (AOC = {aoc:.3f})")


def plot_paraphrasing_curve():
    """Plot paraphrasing accuracy curves (Figure 6 in paper) - Accuracy vs gold (not match-rate)"""
    results_path = OUT / "paraphrasing_results.csv"
    
    if not results_path.exists():
        print(f"⚠️  {results_path} not found. Run experiment 4 first.")
        return
    
    df = pd.read_csv(results_path)
    
    # For paraphrasing, use accuracy vs gold (as per checklist item 5)
    # accuracy = (paraphrased_answer == gold_answer)
    accuracy_by_paraphrase = df.groupby("num_sentences_paraphrased")["is_correct"].mean().reset_index()
    accuracy_by_paraphrase.columns = ["num_sentences", "accuracy"]
    accuracy_by_paraphrase = accuracy_by_paraphrase.sort_values("num_sentences")
    
    # Compute original CoT accuracy (baseline) from cot_samples.jsonl
    # This is the accuracy of the original full CoT samples (before paraphrasing)
    original_accuracy = None
    cot_samples_path = OUT / "cot_samples.jsonl"
    if cot_samples_path.exists():
        original_correct = 0
        original_total = 0
        with open(cot_samples_path, "r") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    gold = str(sample.get("gold_answer", "")).strip()
                    final_ans_text = str(sample.get("final_answer", "")).strip()
                    extracted = extract_number_from_text(final_ans_text)
                    gold_norm = normalize_answer(gold)
                    extracted_norm = normalize_answer(extracted)
                    if gold_norm == extracted_norm:
                        original_correct += 1
                    original_total += 1
                except:
                    continue
        if original_total > 0:
            original_accuracy = original_correct / original_total
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        accuracy_by_paraphrase["num_sentences"],
        accuracy_by_paraphrase["accuracy"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="green"
    )
    
    # Add horizontal line for original CoT accuracy
    if original_accuracy is not None:
        plt.axhline(y=original_accuracy, color="red", linestyle="--", 
                   label=f"Original CoT accuracy = {original_accuracy:.3f}")
        plt.legend()
    
    plt.xlabel("Number of Sentences Paraphrased", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Paraphrasing Test - Accuracy Preservation", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    out_path = FIG / "paraphrasing_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


def plot_filler_tokens_curve():
    """Plot filler tokens accuracy curves (Figure 5 in paper)"""
    results_path = OUT / "filler_tokens_results.csv"
    
    if not results_path.exists():
        print(f"⚠️  {results_path} not found. Run experiment 5 first.")
        return
    
    df = pd.read_csv(results_path)
    
    # Compute accuracy by filler length
    accuracy_by_length = df.groupby("filler_length")["is_correct"].mean().reset_index()
    accuracy_by_length.columns = ["filler_length", "accuracy"]
    accuracy_by_length = accuracy_by_length.sort_values("filler_length")
    
    # Get baseline (filler_length=0 accuracy)
    baseline = None
    baseline_df = accuracy_by_length[accuracy_by_length["filler_length"] == 0]
    if len(baseline_df) > 0:
        baseline = baseline_df.iloc[0]["accuracy"]
    
    # Optional: Convert filler_length to percentile (if desired)
    # For now, keep as-is since it's "mostly OK" per checklist
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        accuracy_by_length["filler_length"],
        accuracy_by_length["accuracy"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="orange"
    )
    
    # Add baseline line
    if baseline is not None:
        plt.axhline(y=baseline, color="red", linestyle="--", 
                   label=f"No filler baseline = {baseline:.3f}")
        plt.legend()
    
    plt.xlabel("Filler Token Length", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Filler Tokens Test", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    out_path = FIG / "filler_tokens_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


def main():
    """Generate all plots"""
    print("\n" + "="*60)
    print("Generating Visualizations (Paper-compliant)")
    print("="*60 + "\n")
    
    plot_truncation_curve()
    plot_mistakes_curve()
    plot_paraphrasing_curve()
    plot_filler_tokens_curve()
    
    print("\n✅ All visualizations complete!")
    print(f"   Figures saved to: {FIG}")


if __name__ == "__main__":
    main()
