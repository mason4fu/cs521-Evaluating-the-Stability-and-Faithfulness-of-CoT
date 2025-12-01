#!/usr/bin/env python3
"""Main script to run all CoT faithfulness experiments"""
import argparse
import sys
import time
import signal
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import config
from src.experiment_1_generate_cot import run_experiment_1
from src.experiment_2_truncation import run_experiment_2
from src.experiment_3_mistakes import run_experiment_3
from src.experiment_4_paraphrasing import run_experiment_4
from src.experiment_5_filler_tokens import run_experiment_5
from src.utils import OUT

# Global flag for graceful shutdown
_should_stop = False
_current_question = None

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global _should_stop, _current_question
    print(f"\n\n⚠️  Interrupt received (signal {signum})")
    if _current_question:
        print(f"   Finishing current question: {_current_question}")
    print("   Will stop after current question completes...")
    print("   (Press Ctrl+C again to force immediate exit)\n")
    _should_stop = True

def get_should_stop():
    """Get the stop flag"""
    return _should_stop

def set_current_question(question_id):
    """Set the current question being processed"""
    global _current_question
    _current_question = question_id

def clear_current_question():
    """Clear the current question"""
    global _current_question
    _current_question = None


def main():
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Run CoT faithfulness experiments"
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Run specific experiment (1-5). If not specified, runs all."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local model runner (for testing)"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions (overrides test mode)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples per question (overrides test mode)"
    )
    parser.add_argument(
        "--skip-experiment-1",
        action="store_true",
        help="Skip experiment 1 (assumes CoT samples already exist)"
    )
    
    args = parser.parse_args()
    
    # Determine parameters
    if config.TEST_MODE:
        num_questions = args.num_questions or config.TEST_NUM_QUESTIONS
        num_samples = args.num_samples or config.TEST_NUM_SAMPLES
        print("🧪 TEST MODE ENABLED")
        print(f"   Questions: {num_questions}")
        print(f"   Samples per question: {num_samples}")
    else:
        num_questions = args.num_questions or config.NUM_QUESTIONS
        num_samples = args.num_samples or config.NUM_SAMPLES_PER_QUESTION
        print("🚀 FULL EXPERIMENT MODE")
        print(f"   Questions: {num_questions or 'all'}")
        print(f"   Samples per question: {num_samples}")
    
    print(f"   Model: {config.MODEL_NAME}")
    print(f"   Runtime: {config.MODEL_RUNTIME}")
    print(f"   Use local: {args.local}")
    print()
    
    # Start timing
    start_time = time.time()
    experiment_times = {}
    
    # Run experiments
    if args.experiment:
        experiments_to_run = [args.experiment]
    else:
        experiments_to_run = [1, 2, 3, 4, 5]
    
    # Experiment 1: Generate CoT samples
    if 1 in experiments_to_run and not args.skip_experiment_1:
        exp_start = time.time()
        print("\n" + "="*70)
        print("EXPERIMENT 1: Generate CoT Samples")
        print("="*70)
        try:
            run_experiment_1(
                num_questions=num_questions,
                num_samples=num_samples,
                use_local=args.local
            )
            experiment_times[1] = time.time() - exp_start
            print(f"   ⏱️  Time: {experiment_times[1]/60:.1f} minutes")
            
            # Check if stop was requested
            if get_should_stop():
                print("\n⏸️  Experiment stopped by user request")
                print("   Partial results saved. You can resume later.")
                return
        except KeyboardInterrupt:
            print("\n⏸️  Experiment interrupted by user")
            print("   Partial results saved. You can resume later.")
            return
        except Exception as e:
            print(f"\n❌ Experiment 1 failed: {e}")
            if args.experiment == 1:
                sys.exit(1)
            print("   Continuing to next experiment...\n")
    else:
        print("\n⏭️  Skipping Experiment 1 (CoT samples should already exist)")
    
    # Check stop flag between experiments
    if get_should_stop():
        print("\n⏸️  Stop requested between experiments")
        return
    
    # Experiment 2: Truncation
    if 2 in experiments_to_run:
        print("\n" + "="*70)
        print("EXPERIMENT 2: Early Answering / Truncation Test")
        print("="*70)
        exp_start = time.time()
        try:
            limit_samples = None
            if config.TEST_MODE:
                limit_samples = num_questions * num_samples
            run_experiment_2(
                use_local=args.local,
                limit_samples=limit_samples
            )
            experiment_times[2] = time.time() - exp_start
            print(f"   ⏱️  Time: {experiment_times[2]/60:.1f} minutes")
            
            if get_should_stop():
                print("\n⏸️  Experiment stopped by user request")
                return
        except KeyboardInterrupt:
            print("\n⏸️  Experiment interrupted by user")
            return
        except Exception as e:
            print(f"\n❌ Experiment 2 failed: {e}")
            if args.experiment == 2:
                sys.exit(1)
            print("   Continuing to next experiment...\n")
    
    # Check stop flag between experiments
    if get_should_stop():
        print("\n⏸️  Stop requested between experiments")
        return
    
    # Experiment 3: Adding Mistakes
    if 3 in experiments_to_run:
        print("\n" + "="*70)
        print("EXPERIMENT 3: Adding Mistakes Test")
        print("="*70)
        exp_start = time.time()
        try:
            limit_samples = None
            if config.TEST_MODE:
                limit_samples = num_questions * num_samples
            run_experiment_3(
                use_local=args.local,
                limit_samples=limit_samples
            )
            experiment_times[3] = time.time() - exp_start
            print(f"   ⏱️  Time: {experiment_times[3]/60:.1f} minutes")
            
            if get_should_stop():
                print("\n⏸️  Experiment stopped by user request")
                return
        except KeyboardInterrupt:
            print("\n⏸️  Experiment interrupted by user")
            return
        except Exception as e:
            print(f"\n❌ Experiment 3 failed: {e}")
            if args.experiment == 3:
                sys.exit(1)
            print("   Continuing to next experiment...\n")
    
    # Check stop flag between experiments
    if get_should_stop():
        print("\n⏸️  Stop requested between experiments")
        return
    
    # Experiment 4: Paraphrasing
    if 4 in experiments_to_run:
        print("\n" + "="*70)
        print("EXPERIMENT 4: Paraphrasing Test")
        print("="*70)
        exp_start = time.time()
        try:
            limit_samples = None
            if config.TEST_MODE:
                limit_samples = num_questions * num_samples
            run_experiment_4(
                use_local=args.local,
                limit_samples=limit_samples
            )
            experiment_times[4] = time.time() - exp_start
            print(f"   ⏱️  Time: {experiment_times[4]/60:.1f} minutes")
            
            if get_should_stop():
                print("\n⏸️  Experiment stopped by user request")
                return
        except KeyboardInterrupt:
            print("\n⏸️  Experiment interrupted by user")
            return
        except Exception as e:
            print(f"\n❌ Experiment 4 failed: {e}")
            if args.experiment == 4:
                sys.exit(1)
            print("   Continuing to next experiment...\n")
    
    # Check stop flag between experiments
    if get_should_stop():
        print("\n⏸️  Stop requested between experiments")
        return
    
    # Experiment 5: Filler Tokens
    if 5 in experiments_to_run:
        print("\n" + "="*70)
        print("EXPERIMENT 5: Filler Tokens Test")
        print("="*70)
        exp_start = time.time()
        try:
            limit_questions = num_questions if config.TEST_MODE else None
            run_experiment_5(
                use_local=args.local,
                limit_questions=limit_questions
            )
            experiment_times[5] = time.time() - exp_start
            print(f"   ⏱️  Time: {experiment_times[5]/60:.1f} minutes")
            
            if get_should_stop():
                print("\n⏸️  Experiment stopped by user request")
                return
        except KeyboardInterrupt:
            print("\n⏸️  Experiment interrupted by user")
            return
        except Exception as e:
            print(f"\n❌ Experiment 5 failed: {e}")
            if args.experiment == 5:
                sys.exit(1)
            print("   Continuing to next experiment...\n")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    
    # Print timing summary
    print("\n⏱️  Timing Summary:")
    for exp_num in sorted(experiment_times.keys()):
        exp_name = {
            1: "Generate CoT Samples",
            2: "Truncation Test",
            3: "Mistakes Test",
            4: "Paraphrasing Test",
            5: "Filler Tokens Test"
        }.get(exp_num, f"Experiment {exp_num}")
        minutes = experiment_times[exp_num] / 60
        hours = minutes / 60
        if hours >= 1:
            print(f"   Experiment {exp_num} ({exp_name}): {hours:.2f} hours ({minutes:.1f} minutes)")
        else:
            print(f"   Experiment {exp_num} ({exp_name}): {minutes:.1f} minutes")
    
    total_hours = total_time / 3600
    total_minutes = total_time / 60
    if total_hours >= 1:
        print(f"   Total time: {total_hours:.2f} hours ({total_minutes:.1f} minutes)")
    else:
        print(f"   Total time: {total_minutes:.1f} minutes")
    
    # Save timing log
    timing_log_path = OUT / "experiment_timing.json"
    timing_data = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": num_questions,
        "num_samples": num_samples,
        "experiment_times": {str(k): v for k, v in experiment_times.items()},
        "total_time_seconds": total_time,
        "total_time_hours": total_hours,
        "total_time_minutes": total_minutes
    }
    import json
    with open(timing_log_path, "w") as f:
        json.dump(timing_data, f, indent=2)
    print(f"\n   Timing log saved to: {timing_log_path}")
    
    print("\nResults are in the outputs/ directory:")
    print("  - cot_samples.jsonl")
    print("  - early_answering_results.csv")
    print("  - adding_mistakes_results.csv")
    print("  - paraphrasing_results.csv")
    print("  - filler_tokens_results.csv")
    print("\nRun visualization script to generate plots:")
    print("  python src/visualize.py")


if __name__ == "__main__":
    main()

