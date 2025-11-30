# SRL-Reasoning

A Python implementation of the paper "Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning".

## Installation

```bash
pip install -e .
```

## Dependencies

- torch
- transformers
- accelerate

## Usage

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

## Running Tests

```bash
python -m pytest tests/
```

