#!/usr/bin/env python3
"""Test script to run experiments 1-5 with 1 question each and record timing"""
import time
import json
import sys
from pathlib import Path
from datetime import datetime

# Add root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.experiment_1_generate_cot import run_experiment_1
from src.experiment_2_truncation import run_experiment_2
from src.experiment_3_mistakes import run_experiment_3
from src.experiment_4_paraphrasing import run_experiment_4
from src.experiment_5_filler_tokens import run_experiment_5
import config

# Backup and clean output files for clean run
OUTPUT_DIR = config.OUTPUTS_DIR
BACKUP_DIR = OUTPUT_DIR / "backup_test_run"

def backup_existing_files():
    """Backup existing output files"""
    BACKUP_DIR.mkdir(exist_ok=True)
    backup_files = [
        "cot_samples.jsonl",
        "early_answering_results.csv",
        "adding_mistakes_results.csv",
        "paraphrasing_results.csv",
        "filler_tokens_results.csv"
    ]
    
    for filename in backup_files:
        src = OUTPUT_DIR / filename
        if src.exists():
            import shutil
            dst = BACKUP_DIR / filename
            shutil.copy2(src, dst)
            print(f"   Backed up {filename} to {dst}")

def restore_existing_files():
    """Restore backed up files"""
    import shutil
    for filename in BACKUP_DIR.glob("*"):
        if filename.is_file():
            dst = OUTPUT_DIR / filename.name
            shutil.copy2(filename, dst)
            print(f"   Restored {filename.name}")

def clean_test_files():
    """Remove test output files"""
    test_files = [
        OUTPUT_DIR / "cot_samples.jsonl",
        OUTPUT_DIR / "early_answering_results.csv",
        OUTPUT_DIR / "adding_mistakes_results.csv",
        OUTPUT_DIR / "paraphrasing_results.csv",
        OUTPUT_DIR / "filler_tokens_results.csv"
    ]
    
    for f in test_files:
        if f.exists():
            f.unlink()
            print(f"   Removed {f.name}")

def main():
    print("="*80)
    print("Testing Experiments 1-5 with 1 Question Each")
    print("="*80)
    
    # Backup existing files
    print("\n📦 Backing up existing output files...")
    backup_existing_files()
    
    # Clean test files for clean run
    print("\n🧹 Cleaning test output files for clean run...")
    clean_test_files()
    
    # Record timing
    timing_results = {
        "test_run_date": datetime.now().isoformat(),
        "experiments": {}
    }
    
    use_local = "--local" in sys.argv
    
    # Experiment 1: Generate CoT samples
    print("\n" + "="*80)
    print("🧪 Experiment 1: Generate CoT Samples (1 question, 1 sample)")
    print("="*80)
    start_time = time.time()
    try:
        run_experiment_1(num_questions=1, num_samples=1, use_local=use_local)
        elapsed = time.time() - start_time
        timing_results["experiments"]["experiment_1"] = {
            "status": "success",
            "time_seconds": elapsed,
            "time_minutes": elapsed / 60.0
        }
        print(f"\n✅ Experiment 1 completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    except Exception as e:
        elapsed = time.time() - start_time
        timing_results["experiments"]["experiment_1"] = {
            "status": "error",
            "error": str(e),
            "time_seconds": elapsed,
            "time_minutes": elapsed / 60.0
        }
        print(f"\n❌ Experiment 1 failed after {elapsed:.2f} seconds: {e}")
    
    # Experiment 2: Truncation test
    print("\n" + "="*80)
    print("🧪 Experiment 2: Truncation Test (1 question)")
    print("="*80)
    start_time = time.time()
    try:
        run_experiment_2(use_local=use_local, limit_samples=1)
        elapsed = time.time() - start_time
        timing_results["experiments"]["experiment_2"] = {
            "status": "success",
            "time_seconds": elapsed,
            "time_minutes": elapsed / 60.0
        }
        print(f"\n✅ Experiment 2 completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    except Exception as e:
        elapsed = time.time() - start_time
        timing_results["experiments"]["experiment_2"] = {
            "status": "error",
            "error": str(e),
            "time_seconds": elapsed,
            "time_minutes": elapsed / 60.0
        }
        print(f"\n❌ Experiment 2 failed after {elapsed:.2f} seconds: {e}")
    
    # Experiment 3: Mistakes test
    print("\n" + "="*80)
    print("🧪 Experiment 3: Mistakes Test (1 question)")
    print("="*80)
    start_time = time.time()
    try:
        run_experiment_3(use_local=use_local, limit_samples=1)
        elapsed = time.time() - start_time
        timing_results["experiments"]["experiment_3"] = {
            "status": "success",
            "time_seconds": elapsed,
            "time_minutes": elapsed / 60.0
        }
        print(f"\n✅ Experiment 3 completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    except Exception as e:
        elapsed = time.time() - start_time
        timing_results["experiments"]["experiment_3"] = {
            "status": "error",
            "error": str(e),
            "time_seconds": elapsed,
            "time_minutes": elapsed / 60.0
        }
        print(f"\n❌ Experiment 3 failed after {elapsed:.2f} seconds: {e}")
    
    # Experiment 4: Paraphrasing test
    print("\n" + "="*80)
    print("🧪 Experiment 4: Paraphrasing Test (1 question)")
    print("="*80)
    start_time = time.time()
    try:
        run_experiment_4(use_local=use_local, limit_samples=1)
        elapsed = time.time() - start_time
        timing_results["experiments"]["experiment_4"] = {
            "status": "success",
            "time_seconds": elapsed,
            "time_minutes": elapsed / 60.0
        }
        print(f"\n✅ Experiment 4 completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    except Exception as e:
        elapsed = time.time() - start_time
        timing_results["experiments"]["experiment_4"] = {
            "status": "error",
            "error": str(e),
            "time_seconds": elapsed,
            "time_minutes": elapsed / 60.0
        }
        print(f"\n❌ Experiment 4 failed after {elapsed:.2f} seconds: {e}")
    
    # Experiment 5: Filler tokens test
    print("\n" + "="*80)
    print("🧪 Experiment 5: Filler Tokens Test (1 question)")
    print("="*80)
    start_time = time.time()
    try:
        run_experiment_5(use_local=use_local, limit_questions=1)
        elapsed = time.time() - start_time
        timing_results["experiments"]["experiment_5"] = {
            "status": "success",
            "time_seconds": elapsed,
            "time_minutes": elapsed / 60.0
        }
        print(f"\n✅ Experiment 5 completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    except Exception as e:
        elapsed = time.time() - start_time
        timing_results["experiments"]["experiment_5"] = {
            "status": "error",
            "error": str(e),
            "time_seconds": elapsed,
            "time_minutes": elapsed / 60.0
        }
        print(f"\n❌ Experiment 5 failed after {elapsed:.2f} seconds: {e}")
    
    # Save timing results
    timing_file = OUTPUT_DIR / "test_timing.json"
    with open(timing_file, "w") as f:
        json.dump(timing_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 Timing Summary")
    print("="*80)
    total_time = 0
    for exp_name, exp_data in timing_results["experiments"].items():
        status_icon = "✅" if exp_data["status"] == "success" else "❌"
        print(f"{status_icon} {exp_name.replace('_', ' ').title()}: {exp_data['time_seconds']:.2f}s ({exp_data['time_minutes']:.2f}m)")
        total_time += exp_data["time_seconds"]
    
    print(f"\n⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\n💾 Timing results saved to: {timing_file}")
    
    # Restore backed up files
    print("\n📦 Restoring backed up files...")
    restore_existing_files()
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()

