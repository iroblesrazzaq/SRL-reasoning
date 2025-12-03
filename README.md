# SRL-Reasoning

A Python implementation of the paper "Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning".

## Installation

```bash
pip install -e .
```

Or using `uv` (recommended for faster dependency management):

```bash
uv sync
```

## Dependencies

- torch
- transformers
- accelerate
- vllm
- datasets (for data processing)
- pandas
- tqdm (for progress bars)

## Project Structure

- `src/shared/` - Common prompts, formatting, generation, data building, and split helpers
- `src/srl/` - SRL-specific pieces (rewards, GRPO utilities)
- `src/sft/` - SFT-specific dataset + collator
- `scripts/` - CLI entrypoints (train SRL/SFT, build data, create splits)
- `benchmarks/` - Benchmark evaluation code
- `data/` - Processed data artifacts (JSONL, etc.)

Tip for notebooks/Colab: run `pip install -e .` first so `import src...` works without manual `sys.path` tweaks.

## Usage

### Using the SRL Library

```python
from src.shared import parse_model_output
from src.srl import compute_srl_reward, dynamic_sampling_filter, compute_advantages

# Parse model output with <think> tags
thought, action = parse_model_output("<think>reasoning here</think>final answer")

# Compute reward based on similarity to expert target
reward = compute_srl_reward(model_completion, expert_target)

# GRPO utilities for training
mask = dynamic_sampling_filter(rewards_tensor)
advantages = compute_advantages(rewards_tensor)
```

### Data Processing

Process expert trajectories into step-wise SRL training examples:

```bash
# Using the CLI (after pip install -e .)
srl-build-data

# Or using uv
uv run srl-build-data

# Or run the module directly
python -m src.shared.build_srl_data
```

You can also use the data building functions programmatically:

```python
from src.shared import load_teacher_dataset, normalize_dataset, build_srl_dataset, save_jsonl

# Load and process your own dataset
ds = load_teacher_dataset("your-dataset/name", split="train")
trajectories = normalize_dataset(ds)
srl_examples = build_srl_dataset(trajectories)
save_jsonl(srl_examples, "data/output.jsonl")
```

The data builder:
- Loads teacher trajectories from Hugging Face datasets
- Splits reasoning into discrete steps
- Creates SRL training examples where each example has:
  - `traj_id`: ID of the original trajectory
  - `step_idx`: Zero-based index of the step (0, 1, 2, ...)
  - `problem`: The problem statement
  - `previous_steps`: List of previous reasoning steps (state)
  - `step_title`: Title/header of the step (used for context injection)
  - `step_body`: The target step content to predict (action)

Output is saved to `data/srl_steps.jsonl`.
