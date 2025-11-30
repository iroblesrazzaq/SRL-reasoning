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

- `srl_lib/` - Core SRL library with reward computation, GRPO utilities, and data processing
  - `srl_lib/data/` - Data building utilities for creating SRL training examples
- `benchmarks/` - Benchmark evaluation code
- `tests/` - Test suite
- `data/` - Processed data files

## Usage

### Using the SRL Library

```python
from srl_lib import parse_model_output, compute_srl_reward
from srl_lib import dynamic_sampling_filter, compute_advantages

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
python -m srl_lib.data.builder
```

You can also use the data building functions programmatically:

```python
from srl_lib.data import load_teacher_dataset, normalize_dataset, build_srl_dataset, save_jsonl

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
  - `teacher_step`: The target step to predict (action)

Output is saved to `data/srl_steps.jsonl`.

## Running Tests

```bash
python -m pytest tests/
```
