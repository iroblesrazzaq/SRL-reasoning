# Google Colab - SFT Model Benchmarking

Quick cells to benchmark your trained SFT model on math benchmarks.

## Cell 1: Setup

```python
# ============================================================================
# Setup and Installation
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

## Cell 2: Mount Google Drive

```python
# ============================================================================
# Mount Google Drive (for loading trained models and saving results)
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

## Cell 3: Configuration

```python
# ============================================================================
# Configuration - Set Your SFT Model Path
# ============================================================================

# Path to your trained SFT model (update this!)
SFT_MODEL_PATH = "/content/drive/MyDrive/srl_outputs/sft_model"  # ⚠️ UPDATE THIS!

# Model display name (for results)
SFT_MODEL_NAME = "SFT-7B"  # Change if needed

# Evaluation settings
GPU_MEMORY_UTILIZATION = 0.8  # Fraction of GPU memory to use (0.8 = 80%)
RESULTS_DIR = "benchmarks/results"
SEED = 42

# Benchmarks to run (from paper: AMC23, AIME24, AIME25)
BENCHMARKS = ["amc23", "aime24", "aime25"]
MODES = ["greedy", "avg32"]  # Greedy = single sample, avg32 = 32 samples with majority voting

print("✓ Configuration loaded")
print(f"  SFT Model: {SFT_MODEL_PATH}")
print(f"  Model exists: {Path(SFT_MODEL_PATH).exists()}")
```

## Cell 4: Benchmark SFT Model

```python
# ============================================================================
# Benchmark SFT Model
# ============================================================================
import os
os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", str(GPU_MEMORY_UTILIZATION))

print("=" * 80)
print("BENCHMARKING SFT MODEL")
print("=" * 80)

# Check if model exists
if not Path(SFT_MODEL_PATH).exists():
    print(f"⚠️  ERROR: SFT model path '{SFT_MODEL_PATH}' does not exist!")
    print("   Please update SFT_MODEL_PATH in the configuration cell.")
else:
    # Initialize evaluator for SFT model (uses SRL prompt template)
    print(f"\nInitializing vLLM with model: {SFT_MODEL_PATH}")
    evaluator_sft = MathEvaluator(
        SFT_MODEL_PATH,
        model_type="srl",  # SFT models are trained with <think> tags
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION
    )
    
    print("✓ Model loaded successfully\n")
    
    # Run all benchmarks
    for benchmark in BENCHMARKS:
        for mode in MODES:
            print(f"\n[{SFT_MODEL_NAME}] {benchmark} - {mode}")
            print("-" * 80)
            eval_start = time.time()
            
            data = load_benchmark_data(benchmark)
            print(f"Loaded {len(data)} problems")
            
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
            print(f"  ✓ Score: {score:.2%} | Time: {eval_end/60:.1f} min | Saved: {path}")
            
            # Backup to Drive
            if DRIVE_RESULTS_DIR:
                dest = Path(DRIVE_RESULTS_DIR)
                dest.mkdir(parents=True, exist_ok=True)
                for f in Path(RESULTS_DIR).glob("*.json"):
                    shutil.copy2(f, dest / f.name)
                print(f"  ✓ Backed up to Drive: {dest}")
    
    print("\n" + "=" * 80)
    print("✓ SFT model benchmarking complete!")
    print("=" * 80)
```

## Cell 5: View Results

```python
# ============================================================================
# View Results Summary
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

# Calculate averages (matching paper Table 1 format)
print("\n" + "=" * 80)
print("Average Scores (across all benchmarks):")
print("=" * 80)
model_scores = {}
for r in results:
    if r.model_name == SFT_MODEL_NAME:  # Only show SFT results
        key = f"{r.model_name}_{r.benchmark_type}"
        if key not in model_scores:
            model_scores[key] = []
        model_scores[key].append(r.score)

for key, scores in sorted(model_scores.items()):
    avg = sum(scores) / len(scores)
    print(f"{key:30} | Average: {avg:6.2%} ({len(scores)} benchmarks)")

print("\n" + "=" * 80)
print("Results saved to:", RESULTS_DIR)
if DRIVE_RESULTS_DIR:
    print("Results backed up to:", DRIVE_RESULTS_DIR)
print("=" * 80)
```

## Quick Test Cell (Optional - for quick verification)

```python
# ============================================================================
# Quick Test - Single Benchmark (for testing before full run)
# ============================================================================
# Use this cell to quickly test one benchmark before running the full suite

TEST_BENCHMARK = "amc23"  # Quick test on AMC23
TEST_MODE = "greedy"  # Faster than avg32

print(f"Quick test: {SFT_MODEL_NAME} on {TEST_BENCHMARK} ({TEST_MODE})")
print("-" * 80)

if Path(SFT_MODEL_PATH).exists():
    evaluator = MathEvaluator(
        SFT_MODEL_PATH,
        model_type="srl",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION
    )
    
    data = load_benchmark_data(TEST_BENCHMARK)
    print(f"Loaded {len(data)} problems")
    
    score = evaluator.evaluate(data, mode=TEST_MODE)
    
    print(f"\n✓ Test Result: {score:.2%} ({score*100:.2f}%)")
else:
    print(f"⚠️  Model not found: {SFT_MODEL_PATH}")
```

## Notes

1. **Update SFT_MODEL_PATH**: Change line in Cell 3 to point to your trained SFT model in Google Drive
2. **GPU Memory**: If you get OOM errors, reduce `GPU_MEMORY_UTILIZATION` to 0.7 or 0.6
3. **Benchmarks**: Runs AMC23, AIME24, AIME25 (matching paper Table 1)
4. **Modes**: 
   - `greedy`: Single deterministic sample (faster, ~5-10 min per benchmark)
   - `avg32`: 32 samples with majority voting (slower, ~30-60 min per benchmark)
5. **Results**: Saved to `benchmarks/results/` and backed up to Google Drive automatically

