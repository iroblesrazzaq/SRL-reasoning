# Google Colab Benchmarking Cells

Complete Colab cells to benchmark your trained SFT and SRL models on math benchmarks (AMC23, AIME24, AIME25).

## Setup Cell (Run First)

```python
# ============================================================================
# 0. Setup and Installation
# ============================================================================
# Runtime: GPU (A100 recommended for 7B models, L4/T4 for smaller models)
# Make sure to select GPU: Runtime -> Change runtime type -> GPU

REPO_URL = "https://github.com/iroblesrazzaq/SRL-reasoning.git"
BRANCH = "main"
WORKDIR = "/content/SRL-reasoning"

# Install dependencies
%pip install -U pip
import os

# Clone repo if not exists
if not os.path.exists(WORKDIR):
    !git clone --branch $BRANCH $REPO_URL $WORKDIR

%cd $WORKDIR
!git pull  # Get latest changes

# Install package
%pip install -e .

# Import required modules
from pathlib import Path
import time
import shutil
from benchmarks import load_benchmark_data, MathEvaluator, BenchmarkResult
from benchmarks import load_all_results, summarize_results

print("✓ Setup complete. Repo at", WORKDIR)
```

## Mount Google Drive (Optional but Recommended)

```python
# ============================================================================
# 1. Mount Google Drive (for loading trained models and saving results)
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

# Paths
DRIVE_MODELS_DIR = '/content/drive/MyDrive/srl_outputs'  # Where your trained models are
DRIVE_RESULTS_DIR = '/content/drive/MyDrive/srl_bench_results'  # Where to save results

print("✓ Google Drive mounted")
print(f"  Models: {DRIVE_MODELS_DIR}")
print(f"  Results: {DRIVE_RESULTS_DIR}")
```

## Configuration Cell

```python
# ============================================================================
# 2. Configuration - Set Your Model Paths
# ============================================================================

# Option 1: Load from Google Drive (if you saved trained models there)
SFT_MODEL_PATH = "/content/drive/MyDrive/srl_outputs/sft_model"  # Update this!
SRL_MODEL_PATH = "/content/drive/MyDrive/srl_outputs/grpo_model"  # Update this!

# Option 2: Use HuggingFace model IDs (for base model comparison)
BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"  # Or Qwen2.5-1.5B-Instruct for testing

# Model display names (for results)
SFT_MODEL_NAME = "SFT-7B"
SRL_MODEL_NAME = "SRL-7B"
BASE_MODEL_NAME = "Qwen2.5-7B-Instruct"

# Evaluation settings
GPU_MEMORY_UTILIZATION = 0.8  # Fraction of GPU memory to use (0.8 = 80%)
RESULTS_DIR = "benchmarks/results"
SEED = 42

# Benchmarks to run (from paper: AMC23, AIME24, AIME25)
BENCHMARKS = ["amc23", "aime24", "aime25"]
MODES = ["greedy", "avg32"]  # Greedy = single sample, avg32 = 32 samples with majority voting

print("✓ Configuration loaded")
print(f"  SFT Model: {SFT_MODEL_PATH}")
print(f"  SRL Model: {SRL_MODEL_PATH}")
print(f"  Base Model: {BASE_MODEL_PATH}")
```

## Benchmark Base Model

```python
# ============================================================================
# 3. Benchmark Base Model (Baseline)
# ============================================================================
import os
os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", str(GPU_MEMORY_UTILIZATION))

print("=" * 80)
print("BENCHMARKING BASE MODEL")
print("=" * 80)

# Initialize evaluator for base model
evaluator_base = MathEvaluator(
    BASE_MODEL_PATH,
    model_type="base",  # Base models don't use <think> tags
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION
)

# Run all benchmarks
for benchmark in BENCHMARKS:
    for mode in MODES:
        print(f"\n[{BASE_MODEL_NAME}] {benchmark} - {mode}")
        eval_start = time.time()
        
        data = load_benchmark_data(benchmark)
        score = evaluator_base.evaluate(data, mode=mode)
        eval_end = time.time() - eval_start
        
        benchmark_type = "Avg@32" if mode == "avg32" else "Greedy"
        result = BenchmarkResult(
            benchmark=benchmark,
            benchmark_type=benchmark_type,
            score=score,
            model_name=BASE_MODEL_NAME,
            model_path=BASE_MODEL_PATH,
            num_questions=len(data),
            eval_time_seconds=eval_end,
            seed=SEED,
        )
        
        path = result.save(RESULTS_DIR)
        print(f"  Score: {score:.2%} | Time: {eval_end/60:.1f} min | Saved: {path}")
        
        # Backup to Drive
        if DRIVE_RESULTS_DIR:
            dest = Path(DRIVE_RESULTS_DIR)
            dest.mkdir(parents=True, exist_ok=True)
            for f in Path(RESULTS_DIR).glob("*.json"):
                shutil.copy2(f, dest / f.name)

print("\n✓ Base model benchmarking complete!")
```

## Benchmark SFT Model

```python
# ============================================================================
# 4. Benchmark SFT Model
# ============================================================================
print("=" * 80)
print("BENCHMARKING SFT MODEL")
print("=" * 80)

# Check if model exists
if not Path(SFT_MODEL_PATH).exists():
    print(f"⚠️  WARNING: SFT model path '{SFT_MODEL_PATH}' does not exist!")
    print("   Please update SFT_MODEL_PATH in the configuration cell.")
    print("   Skipping SFT evaluation...")
else:
    # Initialize evaluator for SFT model (uses SRL prompt template)
    evaluator_sft = MathEvaluator(
        SFT_MODEL_PATH,
        model_type="srl",  # SFT models are trained with <think> tags
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION
    )
    
    # Run all benchmarks
    for benchmark in BENCHMARKS:
        for mode in MODES:
            print(f"\n[{SFT_MODEL_NAME}] {benchmark} - {mode}")
            eval_start = time.time()
            
            data = load_benchmark_data(benchmark)
            score = evaluator_sft.evaluate(data, mode=mode)
            eval_end = time.time() - eval_start
            
            benchmark_type = "Avg@32" if mode == "avg32" else "Greedy"
            result = BenchmarkResult(
                benchmark=benchmark,
                benchmark_type=benchmark_type,
                score=score,
                model_name=SFT_MODEL_NAME,
                model_path=SFT_MODEL_PATH,
                num_questions=len(data),
                eval_time_seconds=eval_end,
                seed=SEED,
            )
            
            path = result.save(RESULTS_DIR)
            print(f"  Score: {score:.2%} | Time: {eval_end/60:.1f} min | Saved: {path}")
            
            # Backup to Drive
            if DRIVE_RESULTS_DIR:
                dest = Path(DRIVE_RESULTS_DIR)
                dest.mkdir(parents=True, exist_ok=True)
                for f in Path(RESULTS_DIR).glob("*.json"):
                    shutil.copy2(f, dest / f.name)
    
    print("\n✓ SFT model benchmarking complete!")
```

## Benchmark SRL Model (GRPO)

```python
# ============================================================================
# 5. Benchmark SRL Model (GRPO-trained)
# ============================================================================
print("=" * 80)
print("BENCHMARKING SRL MODEL (GRPO)")
print("=" * 80)

# Check if model exists
if not Path(SRL_MODEL_PATH).exists():
    print(f"⚠️  WARNING: SRL model path '{SRL_MODEL_PATH}' does not exist!")
    print("   Please update SRL_MODEL_PATH in the configuration cell.")
    print("   Skipping SRL evaluation...")
else:
    # Initialize evaluator for SRL model
    evaluator_srl = MathEvaluator(
        SRL_MODEL_PATH,
        model_type="srl",  # SRL models use <think> tags
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION
    )
    
    # Run all benchmarks
    for benchmark in BENCHMARKS:
        for mode in MODES:
            print(f"\n[{SRL_MODEL_NAME}] {benchmark} - {mode}")
            eval_start = time.time()
            
            data = load_benchmark_data(benchmark)
            score = evaluator_srl.evaluate(data, mode=mode)
            eval_end = time.time() - eval_start
            
            benchmark_type = "Avg@32" if mode == "avg32" else "Greedy"
            result = BenchmarkResult(
                benchmark=benchmark,
                benchmark_type=benchmark_type,
                score=score,
                model_name=SRL_MODEL_NAME,
                model_path=SRL_MODEL_PATH,
                num_questions=len(data),
                eval_time_seconds=eval_end,
                seed=SEED,
            )
            
            path = result.save(RESULTS_DIR)
            print(f"  Score: {score:.2%} | Time: {eval_end/60:.1f} min | Saved: {path}")
            
            # Backup to Drive
            if DRIVE_RESULTS_DIR:
                dest = Path(DRIVE_RESULTS_DIR)
                dest.mkdir(parents=True, exist_ok=True)
                for f in Path(RESULTS_DIR).glob("*.json"):
                    shutil.copy2(f, dest / f.name)
    
    print("\n✓ SRL model benchmarking complete!")
```

## View Results Summary

```python
# ============================================================================
# 6. View Results Summary
# ============================================================================
print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

# Load all results
results = list(load_all_results(RESULTS_DIR))
print(f"\nLoaded {len(results)} result file(s)\n")

# Display all results
print("All Results:")
print("-" * 80)
for r in sorted(results, key=lambda x: (x.benchmark, x.benchmark_type, x.model_name)):
    print(f"{r.benchmark:8} | {r.benchmark_type:8} | {r.model_name:20} | {r.score:6.2%}")

# Best scores per benchmark
print("\n" + "=" * 80)
print("Best Scores (per benchmark and mode):")
print("=" * 80)
best = summarize_results(results)
for (bench, model), sc in sorted(best.items()):
    print(f"{bench:8} | {model:20} | {sc:6.2%}")

# Calculate averages (matching paper Table 1 format)
print("\n" + "=" * 80)
print("Average Scores (across all benchmarks):")
print("=" * 80)
model_scores = {}
for r in results:
    key = f"{r.model_name}_{r.benchmark_type}"
    if key not in model_scores:
        model_scores[key] = []
    model_scores[key].append(r.score)

for key, scores in sorted(model_scores.items()):
    avg = sum(scores) / len(scores)
    print(f"{key:30} | Average: {avg:6.2%} ({len(scores)} benchmarks)")
```

## Quick Single Benchmark (Testing)

```python
# ============================================================================
# 7. Quick Test - Single Benchmark (for testing)
# ============================================================================
# Use this cell to quickly test one benchmark before running the full suite

TEST_MODEL_PATH = "/content/drive/MyDrive/srl_outputs/sft_model"  # Update this!
TEST_MODEL_NAME = "SFT-7B-Test"
TEST_BENCHMARK = "amc23"  # Quick test on AMC23
TEST_MODE = "greedy"  # Faster than avg32

print(f"Quick test: {TEST_MODEL_NAME} on {TEST_BENCHMARK} ({TEST_MODE})")

if Path(TEST_MODEL_PATH).exists():
    evaluator = MathEvaluator(
        TEST_MODEL_PATH,
        model_type="srl",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION
    )
    
    data = load_benchmark_data(TEST_BENCHMARK)
    score = evaluator.evaluate(data, mode=TEST_MODE)
    
    print(f"\n✓ Test Result: {score:.2%}")
else:
    print(f"⚠️  Model not found: {TEST_MODEL_PATH}")
```

## Notes

1. **Model Paths**: Update `SFT_MODEL_PATH` and `SRL_MODEL_PATH` to point to your trained models in Google Drive
2. **GPU Memory**: Adjust `GPU_MEMORY_UTILIZATION` if you get OOM errors (try 0.7 or 0.6)
3. **Benchmarks**: The code runs AMC23, AIME24, AIME25 (matching paper Table 1)
4. **Modes**: 
   - `greedy`: Single deterministic sample (faster)
   - `avg32`: 32 samples with majority voting (slower but more accurate)
5. **Results**: All results are saved to `benchmarks/results/` and backed up to Google Drive
6. **Model Types**: 
   - `model_type="base"` for base models (no `<think>` tags)
   - `model_type="srl"` for SFT/SRL models (uses `<think>` tags)

## Expected Results Format

Results are saved as JSON files in `benchmarks/results/` with format:
- `{benchmark}_{benchmark_type}_{model_name}_{timestamp}.json`

Each result contains:
- `benchmark`: amc23, aime24, or aime25
- `benchmark_type`: Greedy or Avg@32
- `score`: Accuracy (0.0 to 1.0)
- `model_name`: Display name
- `num_questions`: Number of problems evaluated
- `eval_time_seconds`: Time taken

