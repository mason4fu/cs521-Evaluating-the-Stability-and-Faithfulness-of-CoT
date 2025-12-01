# CoT Faithfulness Experiments

Reproducing "Measuring Faithfulness in Chain-of-Thought Reasoning" (Lanham et al., 2023) using Llama-3.1-8B-Instruct on GSM8K.

## Overview

This project implements the five faithfulness evaluation experiments from the paper:

1. **Generate CoT Samples** - Generate multiple Chain-of-Thought reasoning samples per question
2. **Early Answering / Truncation Test** - Measure how accuracy degrades when CoT is truncated
3. **Adding Mistakes Test** - Measure impact of introducing mistakes in CoT reasoning
4. **Paraphrasing Test** - Measure accuracy preservation when CoT is paraphrased
5. **Filler Tokens Test** - Measure whether filler tokens improve performance (they shouldn't)

## Model Setup

**Model:** meta-llama/Llama-3.1-8B-Instruct  
**Runtime:** vLLM (via API on VM)  
**Task:** GSM8K math word problems

### Configuration

All configuration is in `config.py`. Key settings:

- `MODEL_NAME`: Model identifier (default: "meta-llama/Llama-3.1-8B-Instruct")
- `MODEL_RUNTIME`: "vllm" (for VM inference)
- `TEMPERATURE`: 0.8 (as per paper)
- `NUCLEUS_P`: 0.95 (as per paper)
- `MAX_TOKENS`: Maximum tokens to generate (default: 512)

### Experiment Parameters

Control experiment scope via environment variables:

- `NUM_QUESTIONS`: Number of questions to process (default: 100, use `None` for all)
- `NUM_SAMPLES`: Number of CoT samples per question (default: 3)

### Test Mode

For quick testing, set environment variables:

```bash
export TEST_MODE=true
export TEST_NUM_QUESTIONS=2
export TEST_NUM_SAMPLES=2
```

This runs experiments on 2 questions with 2 samples each.

### Incremental Processing & Skip Logic

The experiments support incremental processing:
- **Skip existing data**: Automatically detects and skips already-processed questions/samples
- **Append mode**: New results are appended to existing files, preserving previous work
- **Expandable**: Run with more questions/samples without re-processing existing data

Example: If you ran 100 questions with 3 samples, then run 300 questions with 3 samples:
- Existing 100 questions × 3 samples = 300 samples are preserved
- Only new 200 questions × 3 samples = 600 samples are generated

## Quick Start

### Running a Test (Small Size)

```bash
# Set test mode
export TEST_MODE=true
export TEST_NUM_QUESTIONS=2
export TEST_NUM_SAMPLES=2

# Run all experiments
python run_all.py --local
```

Or use the shell script:

```bash
TEST_MODE=true TEST_NUM_QUESTIONS=2 TEST_NUM_SAMPLES=2 sh run.sh --local
```

### Running Individual Experiments

```bash
# Generate CoT samples only
python run_all.py --experiment 1 --local

# Run truncation test only
python run_all.py --experiment 2 --local --skip-experiment-1

# Run all experiments
python run_all.py --local
```

## Running on VM

### Initial Setup

1. **Setup SSH keys** (one-time):
   ```bash
   sh setup_ssh_keys.sh
   ```

2. **Sync code to VM**:
   ```bash
   sh sync_to_vm.sh
   ```

3. **Setup VM environment** (SSH to VM):
   ```bash
   ssh exouser@149.165.151.46
   cd ~/cot-stability
   sh setup_vm.sh
   ```

### Daily Workflow

1. **Sync code changes**:
   ```bash
   sh sync_to_vm.sh
   ```

2. **Run experiments on VM**:
   ```bash
   # Recommended: Use run_long_experiment.sh (handles tmux automatically)
   NUM_QUESTIONS=300 NUM_SAMPLES=3 bash run_long_experiment.sh
   
   # This will:
   # - Create a detached tmux session
   # - Run all experiments
   # - Allow you to disconnect and reconnect later
   
   # To check progress:
   ssh exouser@149.165.151.46
   tmux attach -t cot-experiment
   # Detach: Ctrl+B then D
   
   # Alternative: Manual tmux session
   ssh exouser@149.165.151.46
   tmux new -s experiment
   cd ~/cot-stability
   source .venv/bin/activate
   export NUM_QUESTIONS=300
   export NUM_SAMPLES=3
   python run_all.py
   # Detach: Ctrl+B then D
   # Reattach: tmux attach -t experiment
   ```

3. **Download results**:
   ```bash
   sh sync_from_vm.sh
   ```

## Experiment Structure

```
cot-stability/
├── config.py                    # Configuration
├── run_all.py                   # Main script to run all experiments
├── run_long_experiment.sh       # Script to run long experiments in tmux
├── sync_to_vm.sh                # Sync code to VM
├── sync_from_vm.sh              # Download results from VM
├── src/
│   ├── model_runner.py          # Model runner (vLLM API support)
│   ├── utils.py                 # Utility functions
│   ├── experiment_1_generate_cot.py    # Generate CoT samples
│   ├── experiment_2_truncation.py      # Truncation test
│   ├── experiment_3_mistakes.py        # Mistakes test
│   ├── experiment_4_paraphrasing.py    # Paraphrasing test
│   ├── experiment_5_filler_tokens.py   # Filler tokens test
│   └── visualize.py             # Generate plots
├── data/                        # GSM8K data
├── outputs/                     # Results
│   ├── cot_samples.jsonl
│   ├── early_answering_results.csv
│   ├── adding_mistakes_results.csv
│   ├── paraphrasing_results.csv
│   ├── filler_tokens_results.csv
│   ├── experiment_timing.json   # Timing logs
│   └── figures/                 # Generated plots
└── requirements.txt
```

## Output Files

- **cot_samples.jsonl**: Generated CoT samples with sentence splits
- **early_answering_results.csv**: Truncation test results with AOC metric
- **adding_mistakes_results.csv**: Mistake insertion test results
- **paraphrasing_results.csv**: Paraphrasing test results
- **filler_tokens_results.csv**: Filler tokens test results
- **experiment_timing.json**: Timing logs for each experiment run
- **figures/**: Plots matching paper figures

## Visualization

Generate plots after running experiments:

```bash
python src/visualize.py
```

This creates plots matching the paper's figures:
- `truncation_curve.png` (Fig. 3)
- `mistakes_hist.png` (Fig. 4)
- `paraphrasing_curve.png` (Fig. 6)
- `filler_tokens_curve.png` (Fig. 5)

## VM Configuration

Edit `config.py` or set environment variables:

```bash
export VM_USER=exouser
export VM_HOST=149.165.151.46
export VM_HOME=~/cot-stability
export VM_MODEL_DIR=${VM_HOME}/models
export MODEL_RUNTIME=vllm
```

## Requirements

See `requirements.txt`. Key dependencies:

- transformers, torch (for local model runner)
- vLLM (for VM inference)
- pandas, matplotlib (for analysis/plots)
- nltk (for sentence splitting)
- datasets (for GSM8K)

## Features

### Graceful Shutdown
- Press `Ctrl+C` to interrupt experiments
- Current question processing completes before stopping
- All results are saved before exit
- Resume later - skip logic automatically continues from where you left off

### Time Tracking
- Each experiment logs its duration
- Total runtime is tracked and saved to `outputs/experiment_timing.json`
- Useful for estimating completion times for large experiments

### Optimizations
- **Batch processing**: Multiple prompts sent in parallel to vLLM API
- **Reduced granularity**: Optimized test points for truncation and filler token experiments
- **Skip logic**: Automatically skips already-processed data
- **Incremental saves**: Results saved after each question/sample

## Notes

- The model runner uses vLLM API for efficient batch inference
- Test mode uses small numbers for quick validation
- Full experiments can take significant time (3 samples × 300+ questions)
- Results are saved incrementally as JSONL/CSV files
- Experiments can be safely interrupted and resumed
