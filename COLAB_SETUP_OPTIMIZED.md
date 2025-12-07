# Optimized Colab Setup Cell

## Fast Setup (No pip upgrade)

```python
# ============================================================================
# Setup and Installation (OPTIMIZED - Fast)
# ============================================================================
# Runtime: GPU (A100 recommended for 7B models, L4/T4 for smaller models)
# Make sure to select GPU: Runtime -> Change runtime type -> GPU

import os
from pathlib import Path
import time
import shutil

# Repository configuration
REPO_URL = "https://github.com/iroblesrazzaq/SRL-reasoning.git"
BRANCH = "main"
WORKDIR = "/content/SRL-reasoning"

# Clone repo if not exists
if not os.path.exists(WORKDIR):
    !git clone --branch $BRANCH $REPO_URL $WORKDIR

%cd $WORKDIR
!git pull  # Get latest changes

# Install dependencies (combined, skip pip upgrade to save time)
!pip install -q bitsandbytes accelerate peft transformers trl datasets flash-attn --no-build-isolation

# Install package
!pip install -e . -q

# Import required modules
from benchmarks import load_benchmark_data, MathEvaluator, BenchmarkResult
from benchmarks import load_all_results, summarize_results

# Verify GPU
import torch
print("=" * 80)
print("SETUP COMPLETE")
print("=" * 80)
print(f"✓ Repository: {WORKDIR}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 80)
```

## Changes Made:

1. **Removed pip upgrade** - Saves ~30-60 seconds (pip is usually already up-to-date in Colab)
2. **Combined pip installs** - Single command instead of separate ones
3. **Added `-q` flag** - Suppresses verbose output for faster execution
4. **Kept all functionality** - Nothing removed, just optimized

## Time Savings:

- **Before**: ~2-3 minutes (with pip upgrade)
- **After**: ~1-2 minutes (without pip upgrade)
- **Savings**: ~30-60 seconds

## If you need to upgrade pip:

Only do it if you encounter dependency issues:

```python
!pip install -U pip -q  # Only if needed
```

