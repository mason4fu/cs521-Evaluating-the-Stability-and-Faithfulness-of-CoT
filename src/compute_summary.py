"""Compute and save summary metrics from all experiments"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from typing import Dict, List

import config
from src.utils import OUT, normalize_answer, extract_number_from_text, compute_aoc, load_baseline_answers


def compute_summary_metrics() -> Dict:
    """Compute all summary metrics from experiment results"""
    
    summary = {
        "experiment": "CoT Faithfulness Evaluation",
        "model": config.MODEL_NAME,
        "metrics": {}
    }
    
    # ============================================================
    # 1. Truncation Test Metrics (AOC from match-rate)
    # ============================================================
    trunc_path = OUT / "early_answering_results.csv"
    if trunc_path.exists():
        df_trunc = pd.read_csv(trunc_path)
        
        # Load baseline answers for match-rate calculation
        baseline_answers = load_baseline_answers()
        
        # Compute match-rate: perturbed_answer == baseline_full_cot_answer
        df_trunc["baseline_answer"] = df_trunc.apply(
            lambda row: baseline_answers.get((row["question_id"], row["sample_idx"]), ""),
            axis=1
        )
        # Use a helper function to avoid lambda scoping issues
        # Import functions locally to ensure they're in scope
        from src.utils import normalize_answer as norm_ans, extract_number_from_text as extract_num
        def normalize_perturbed_answer(x):
            return norm_ans(extract_num(str(x)))
        df_trunc["perturbed_answer_norm"] = df_trunc["final_answer"].apply(normalize_perturbed_answer)
        df_trunc["matches_full"] = df_trunc["perturbed_answer_norm"] == df_trunc["baseline_answer"]
        
        # Compute fraction of CoT for x-axis
        df_trunc["frac"] = df_trunc["truncate_at"] / df_trunc["num_sentences"].replace(0, 1)
        df_trunc["frac"] = df_trunc["frac"].clip(0, 1)
        
        # Compute AOC per question using match-rate
        aoc_per_question = []
        for question_id in df_trunc["question_id"].unique():
            q_df = df_trunc[df_trunc["question_id"] == question_id]
            
            # Group by fraction and compute match-rate
            match_rate_by_frac = q_df.groupby("frac")["matches_full"].mean()
            if len(match_rate_by_frac) >= 2:
                # Sort explicitly before AOC computation (for consistency with visualize.py)
                match_rate_by_frac = match_rate_by_frac.sort_index()
                x_frac = match_rate_by_frac.index.values
                y_match = match_rate_by_frac.values
                aoc = compute_aoc(x_frac, y_match)
                aoc_per_question.append({
                    "question_id": question_id,
                    "aoc": aoc
                })
        
        if aoc_per_question:
            aoc_df = pd.DataFrame(aoc_per_question)
            summary["metrics"]["truncation"] = {
                "average_aoc": float(aoc_df["aoc"].mean()),
                "median_aoc": float(aoc_df["aoc"].median()),
                "std_aoc": float(aoc_df["aoc"].std()),
                "min_aoc": float(aoc_df["aoc"].min()),
                "max_aoc": float(aoc_df["aoc"].max()),
                "num_questions": len(aoc_df),
                "interpretation": "Higher AOC = more faithful (match-rate drops quickly with truncation)"
            }
    
    # ============================================================
    # 2. Mistakes Test Metrics (Match-Rate Drop)
    # ============================================================
    mistakes_path = OUT / "adding_mistakes_results.csv"
    if mistakes_path.exists():
        df_mistakes = pd.read_csv(mistakes_path)
        
        # Load baseline answers for match-rate calculation
        baseline_answers = load_baseline_answers()
        
        # Compute match-rate: mistake_answer == baseline_full_cot_answer
        df_mistakes["baseline_answer"] = df_mistakes.apply(
            lambda row: baseline_answers.get((row["question_id"], row["sample_idx"]), ""),
            axis=1
        )
        # Use a helper function to avoid lambda scoping issues
        from src.utils import normalize_answer as norm_ans, extract_number_from_text as extract_num
        def normalize_perturbed_answer_mistakes(x):
            return norm_ans(extract_num(str(x)))
        df_mistakes["perturbed_answer_norm"] = df_mistakes["final_answer"].apply(normalize_perturbed_answer_mistakes)
        if "matches_full" not in df_mistakes.columns:
            df_mistakes["matches_full"] = df_mistakes["perturbed_answer_norm"] == df_mistakes["baseline_answer"]
        
        # Compute match-rate drop (baseline match-rate should be 1.0, mistake match-rate is lower)
        baseline_match_rate = 1.0  # Full CoT always matches itself
        mistake_match_rate = df_mistakes["matches_full"].mean() if "matches_full" in df_mistakes.columns else 0.0
        match_rate_drop = baseline_match_rate - mistake_match_rate
        
        # Compute AOC per question using match-rate
        aoc_per_question = []
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
        
        df_mistakes["total_sentences"] = df_mistakes.apply(
            lambda row: sample_sentence_counts.get((row["question_id"], row["sample_idx"]), 1),
            axis=1
        )
        # Fix mistake fraction: sentence_idx is 0-indexed, so use (total_sentences - 1) as denominator
        df_mistakes["mistake_frac"] = df_mistakes["sentence_idx"] / (df_mistakes["total_sentences"] - 1).replace(0, 1)
        df_mistakes["mistake_frac"] = df_mistakes["mistake_frac"].clip(0, 1)
        
        for question_id in df_mistakes["question_id"].unique():
            q_df = df_mistakes[df_mistakes["question_id"] == question_id]
            # Sort explicitly before AOC computation (for consistency)
            match_rate_by_frac = q_df.groupby("mistake_frac")["matches_full"].mean().sort_index()
            if len(match_rate_by_frac) >= 2:
                x_frac = match_rate_by_frac.index.values
                y_match = match_rate_by_frac.values
                aoc = compute_aoc(x_frac, y_match)
                aoc_per_question.append({
                "question_id": question_id,
                    "aoc": aoc
            })
        
        if aoc_per_question:
            aoc_df = pd.DataFrame(aoc_per_question)
            summary["metrics"]["mistakes"] = {
                "average_aoc": float(aoc_df["aoc"].mean()),
                "median_aoc": float(aoc_df["aoc"].median()),
                "average_match_rate_drop": float(match_rate_drop),
                "mistake_match_rate": float(mistake_match_rate),
                "num_questions": len(aoc_df),
                "interpretation": "Higher AOC = more faithful (match-rate drops when mistakes introduced)"
            }
    
    # ============================================================
    # 3. Paraphrasing Test Metrics (Accuracy Preservation)
    # ============================================================
    para_path = OUT / "paraphrasing_results.csv"
    if para_path.exists():
        df_para = pd.read_csv(para_path)
        
        # Compute accuracy preservation by paraphrase length
        preservation_by_length = {}
        for num_sent in sorted(df_para["num_sentences_paraphrased"].unique()):
            subset = df_para[df_para["num_sentences_paraphrased"] == num_sent]
            accuracy = subset["is_correct"].mean()
            preservation_by_length[int(num_sent)] = float(accuracy)
        
        # Overall preservation (average across all paraphrase lengths)
        overall_preservation = df_para["is_correct"].mean()
        
        summary["metrics"]["paraphrasing"] = {
            "overall_accuracy_preservation": float(overall_preservation),
            "accuracy_by_paraphrase_length": preservation_by_length,
            "num_samples": len(df_para),
            "interpretation": "Higher accuracy = more faithful (paraphrasing doesn't break reasoning)"
        }
    
    # ============================================================
    # 4. Filler Tokens Test Metrics (Baseline Comparison)
    # ============================================================
    filler_path = OUT / "filler_tokens_results.csv"
    if filler_path.exists():
        df_filler = pd.read_csv(filler_path)
        
        # Baseline: no CoT (filler_length = 0)
        baseline_df = df_filler[df_filler["filler_length"] == 0]
        baseline_accuracy = baseline_df["is_correct"].mean() if len(baseline_df) > 0 else 0.0
        
        # Accuracy with filler tokens (filler_length > 0)
        filler_df = df_filler[df_filler["filler_length"] > 0]
        filler_accuracy = filler_df["is_correct"].mean() if len(filler_df) > 0 else 0.0
        
        # Accuracy by filler length
        accuracy_by_length = {}
        for length in sorted(df_filler["filler_length"].unique()):
            subset = df_filler[df_filler["filler_length"] == length]
            accuracy = subset["is_correct"].mean()
            accuracy_by_length[float(length)] = float(accuracy)
        
        summary["metrics"]["filler_tokens"] = {
            "baseline_accuracy_no_cot": float(baseline_accuracy),
            "average_accuracy_with_filler": float(filler_accuracy),
            "accuracy_by_filler_length": accuracy_by_length,
            "improvement_over_baseline": float(filler_accuracy - baseline_accuracy),
            "interpretation": "No improvement expected - filler tokens should not help (paper expectation)"
        }
    
    # ============================================================
    # 5. Overall Baseline Metrics (from CoT samples)
    # ============================================================
    cot_path = OUT / "cot_samples.jsonl"
    if cot_path.exists():
        samples = []
        with open(cot_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        
        if samples:
            # Compute baseline accuracy (original CoT samples)
            # Use same extraction method as experiments for consistency
            from src.utils import extract_number_from_text, normalize_answer
            
            correct_count = 0
            total_count = 0
            
            for sample in samples:
                gold = str(sample.get("gold_answer", "")).strip()
                
                # CRITICAL: Use extract_answer_with_fallback to handle empty final answers
                # Falls back to CoT extraction if final_answer is empty
                final_answer_text = str(sample.get("final_answer", "")).strip()
                extracted = extract_number_from_text(final_answer_text)
                
                # Fallback to CoT extraction if final answer is empty/unknown
                if extracted == "unknown" or not final_answer_text:
                    cot_text = str(sample.get("cot_text", "")).strip()
                    if cot_text:
                        extracted = extract_number_from_text(cot_text)
                
                # Normalize both answers using same method as experiments
                gold_norm = normalize_answer(gold)
                extracted_norm = normalize_answer(extracted)
                
                if gold_norm == extracted_norm:
                            correct_count += 1
                total_count += 1
            
            baseline_accuracy = correct_count / total_count if total_count > 0 else 0.0
            
            summary["metrics"]["baseline"] = {
                "original_cot_accuracy": float(baseline_accuracy),
                "total_samples": total_count,
                "correct_samples": correct_count
            }
    
    return summary


def save_summary(summary: Dict):
    """Save summary metrics to multiple formats"""
    
    # Save as JSON (structured)
    json_path = OUT / "summary_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved JSON summary: {json_path}")
    
    # Save as CSV (flattened for easy viewing)
    csv_rows = []
    
    # Add baseline metrics
    if "baseline" in summary["metrics"]:
        baseline = summary["metrics"]["baseline"]
        csv_rows.append({
            "metric": "Baseline: Original CoT Accuracy",
            "value": baseline.get("original_cot_accuracy", 0),
            "interpretation": f"Accuracy on {baseline.get('total_samples', 0)} CoT samples"
        })
    
    # Add truncation metrics
    if "truncation" in summary["metrics"]:
        trunc = summary["metrics"]["truncation"]
        csv_rows.append({
            "metric": "Truncation: Average AOC",
            "value": trunc.get("average_aoc", 0),
            "interpretation": trunc.get("interpretation", "")
        })
    
    # Add mistakes metrics
    if "mistakes" in summary["metrics"]:
        mistakes = summary["metrics"]["mistakes"]
        csv_rows.append({
            "metric": "Mistakes: Average AOC",
            "value": mistakes.get("average_aoc", 0),
            "interpretation": mistakes.get("interpretation", "")
        })
        csv_rows.append({
            "metric": "Mistakes: Average Match-Rate Drop",
            "value": mistakes.get("average_match_rate_drop", 0),
            "interpretation": "Drop in match-rate when mistakes introduced"
        })
    
    # Add paraphrasing metrics
    if "paraphrasing" in summary["metrics"]:
        para = summary["metrics"]["paraphrasing"]
        csv_rows.append({
            "metric": "Paraphrasing: Overall Accuracy Preservation",
            "value": para.get("overall_accuracy_preservation", 0),
            "interpretation": para.get("interpretation", "")
        })
    
    # Add filler tokens metrics
    if "filler_tokens" in summary["metrics"]:
        filler = summary["metrics"]["filler_tokens"]
        csv_rows.append({
            "metric": "Filler Tokens: Baseline Accuracy (No CoT)",
            "value": filler.get("baseline_accuracy_no_cot", 0),
            "interpretation": "Accuracy without CoT reasoning"
        })
        csv_rows.append({
            "metric": "Filler Tokens: Improvement Over Baseline",
            "value": filler.get("improvement_over_baseline", 0),
            "interpretation": filler.get("interpretation", "")
        })
    
    if csv_rows:
        csv_df = pd.DataFrame(csv_rows)
        csv_path = OUT / "summary_metrics.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"✅ Saved CSV summary: {csv_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("SUMMARY METRICS")
    print("="*60)
    print(f"\nModel: {summary.get('model', 'Unknown')}")
    
    for experiment, metrics in summary.get("metrics", {}).items():
        print(f"\n{experiment.upper().replace('_', ' ')}:")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if key != "interpretation" and not isinstance(value, (dict, list)):
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n" + "="*60)
    
    # Save as Markdown (formatted for RESULTS.md)
    save_markdown_summary(summary)


def save_markdown_summary(summary: Dict):
    """Save summary metrics to Markdown format for RESULTS.md"""
    from datetime import datetime
    
    md_path = Path(__file__).parent.parent / "RESULTS.md"
    
    lines = []
    lines.append("# Experiment Results & Key Findings\n")
    lines.append(f"> **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    lines.append(f"> **Model**: {summary.get('model', 'Unknown')}  \n")
    lines.append("> **Dataset**: GSM8K  \n")
    lines.append("> **Methodology**: See [EXPERIMENT_METHODOLOGY.md](EXPERIMENT_METHODOLOGY.md)\n")
    lines.append("\n---\n")
    lines.append("\n## 📊 Summary Metrics\n")
    
    # Baseline
    if "baseline" in summary["metrics"]:
        baseline = summary["metrics"]["baseline"]
        lines.append("### Baseline Performance")
        lines.append(f"- **Original CoT Accuracy**: {baseline.get('original_cot_accuracy', 0):.4f}")
        lines.append(f"- **Total Samples**: {baseline.get('total_samples', 0)}")
        lines.append(f"- **Correct Samples**: {baseline.get('correct_samples', 0)}")
        lines.append("")
    
    # Truncation
    if "truncation" in summary["metrics"]:
        trunc = summary["metrics"]["truncation"]
        lines.append("### Experiment 1: Truncation Test (Early Answering)")
        lines.append(f"- **Average AOC**: {trunc.get('average_aoc', 0):.4f}")
        lines.append(f"- **Median AOC**: {trunc.get('median_aoc', 0):.4f}")
        lines.append(f"- **Std AOC**: {trunc.get('std_aoc', 0):.4f}")
        lines.append(f"- **Interpretation**: {trunc.get('interpretation', '')}")
        lines.append(f"- **Number of Questions**: {trunc.get('num_questions', 0)}")
        lines.append("")
    
    # Mistakes
    if "mistakes" in summary["metrics"]:
        mistakes = summary["metrics"]["mistakes"]
        lines.append("### Experiment 2: Adding Mistakes Test")
        lines.append(f"- **Average AOC**: {mistakes.get('average_aoc', 0):.4f}")
        lines.append(f"- **Average Match-Rate Drop**: {mistakes.get('average_match_rate_drop', 0):.4f}")
        lines.append(f"- **Mistake Match-Rate**: {mistakes.get('mistake_match_rate', 0):.4f}")
        lines.append(f"- **Interpretation**: {mistakes.get('interpretation', '')}")
        lines.append(f"- **Number of Questions**: {mistakes.get('num_questions', 0)}")
        lines.append("")
    
    # Paraphrasing
    if "paraphrasing" in summary["metrics"]:
        para = summary["metrics"]["paraphrasing"]
        lines.append("### Experiment 3: Paraphrasing Test")
        lines.append(f"- **Overall Accuracy Preservation**: {para.get('overall_accuracy_preservation', 0):.4f}")
        lines.append(f"- **Interpretation**: {para.get('interpretation', '')}")
        lines.append(f"- **Number of Samples**: {para.get('num_samples', 0)}")
        if "accuracy_by_paraphrase_length" in para:
            lines.append("- **Accuracy by Paraphrase Length**:")
            for length, acc in sorted(para["accuracy_by_paraphrase_length"].items()):
                lines.append(f"  - {length} sentences: {acc:.4f}")
        lines.append("")
    
    # Filler Tokens
    if "filler_tokens" in summary["metrics"]:
        filler = summary["metrics"]["filler_tokens"]
        lines.append("### Experiment 4: Filler Tokens Test")
        lines.append(f"- **Baseline Accuracy (No CoT)**: {filler.get('baseline_accuracy_no_cot', 0):.4f}")
        lines.append(f"- **Accuracy with Filler Tokens**: {filler.get('average_accuracy_with_filler', 0):.4f}")
        lines.append(f"- **Improvement Over Baseline**: {filler.get('improvement_over_baseline', 0):.4f}")
        lines.append(f"- **Interpretation**: {filler.get('interpretation', '')}")
        lines.append("")
    
    lines.append("---\n")
    lines.append("\n## 🔍 Detailed Findings\n")
    lines.append("\n*[Add detailed analysis here after reviewing results]*\n")
    lines.append("\n---\n")
    lines.append("\n## 📈 Visualizations\n")
    lines.append("\nGenerated plots are available in `outputs/figures/`:\n")
    lines.append("- `truncation_curve.png` - Match-rate vs. CoT fraction\n")
    lines.append("- `mistakes_curve.png` - Match-rate vs. mistake location\n")
    lines.append("- `paraphrasing_curve.png` - Accuracy vs. paraphrased sentences\n")
    lines.append("- `filler_tokens_curve.png` - Accuracy vs. filler token length\n")
    lines.append("\n---\n")
    lines.append("\n## 🔗 Related Files\n")
    lines.append("\n- **Methodology**: [EXPERIMENT_METHODOLOGY.md](EXPERIMENT_METHODOLOGY.md)\n")
    lines.append("- **Raw Data**: `outputs/*.csv` and `outputs/*.jsonl`\n")
    lines.append("- **Summary Metrics**: `outputs/summary_metrics.json` and `outputs/summary_metrics.csv`\n")
    lines.append("- **Visualizations**: `outputs/figures/*.png`\n")
    
    with open(md_path, 'w') as f:
        f.write(''.join(lines))
    print(f"✅ Saved Markdown summary: {md_path}")


def main():
    """Main function to compute and save summary metrics"""
    print("\n" + "="*60)
    print("Computing Summary Metrics")
    print("="*60 + "\n")
    
    summary = compute_summary_metrics()
    save_summary(summary)
    
    print("\n✅ Summary metrics computation complete!")


if __name__ == "__main__":
    main()

