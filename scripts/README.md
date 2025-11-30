# Data Processing Pipeline

This directory contains scripts for processing expert reasoning trajectories into step-wise SRL (Supervised Reinforcement Learning) training examples.

## Overview

The data processing pipeline transforms teacher trajectories (expert solutions to problems) into a format suitable for SRL training, where the model learns to predict the next reasoning step given the problem and previous steps.

## Script: `build_srl_data.py`

### Purpose

Converts expert reasoning trajectories from Hugging Face datasets into step-wise SRL training examples. Each example represents a state-action pair where:
- **State**: Problem statement + all previous reasoning steps
- **Action**: The next reasoning step to predict

### How It Works

#### 1. **Load Dataset** (`load_teacher_dataset`)
- Loads a Hugging Face dataset (default: `simplescaling/s1K-1.1`)
- Each row contains:
  - `id`: Unique identifier
  - `question`: Problem statement
  - `cot`: Chain-of-thought reasoning (either a list or string)
  - `final`: Final answer (optional)

#### 2. **Normalize Trajectories** (`normalize_trajectory`)
Converts raw dataset rows into a standardized format:

**Input**: Raw dataset row
```python
{
  "id": "1",
  "question": "Solve 2+2",
  "cot": "Step 1: Add 2 and 2\nStep 2: Result is 4"
}
```

**Output**: Normalized trajectory
```python
{
  "id": "1",
  "problem": "Solve 2+2",
  "steps": ["Step 1: Add 2 and 2", "Step 2: Result is 4"]
}
```

**Processing steps**:
- Extracts problem from `question` or `prompt` field
- Handles `cot` in two formats:
  - **List format**: Already split into steps → clean and filter
  - **String format**: Split using heuristic step boundary detection
    - Detects patterns like "Step 1:", "1.", "First,", etc.
    - Groups lines between boundaries into steps
- Filters out invalid trajectories:
  - Missing problem statement
  - Empty or missing `cot` field
  - No valid steps after processing
  - Steps shorter than 3 characters

#### 3. **Build SRL Examples** (`build_srl_examples`)
Expands each trajectory into multiple training examples:

For a trajectory with N steps `[S1, S2, ..., SN]`, creates N examples:

| step_idx | previous_steps | teacher_step |
|----------|----------------|--------------|
| 0        | `[]`           | S1           |
| 1        | `[S1]`         | S2           |
| 2        | `[S1, S2]`     | S3           |
| ...      | ...            | ...          |
| N-1      | `[S1, ..., S_{N-1}]` | SN      |

**Example**:
```python
# Input trajectory
{
  "id": "1",
  "problem": "Solve 2+2",
  "steps": ["Add 2 and 2", "Result is 4"]
}

# Output examples
[
  {
    "traj_id": "1",
    "step_idx": 0,
    "problem": "Solve 2+2",
    "previous_steps": [],
    "teacher_step": "Add 2 and 2"
  },
  {
    "traj_id": "1",
    "step_idx": 1,
    "problem": "Solve 2+2",
    "previous_steps": ["Add 2 and 2"],
    "teacher_step": "Result is 4"
  }
]
```

#### 4. **Save to JSONL** (`save_jsonl`)
Writes all examples to `data/srl_steps.jsonl`, one JSON object per line.

### Output Format

Each line in `data/srl_steps.jsonl` is a JSON object:

```json
{
  "traj_id": "1",
  "step_idx": 0,
  "problem": "Problem statement...",
  "previous_steps": [],
  "teacher_step": "First reasoning step..."
}
```

**Fields**:
- `traj_id`: ID of the original trajectory (all examples from the same problem share this)
- `step_idx`: Zero-based index of the step (0, 1, 2, ...)
- `problem`: The problem statement
- `previous_steps`: List of previous reasoning steps (the "state")
- `teacher_step`: The target step to predict (the "action")

### Usage

```bash
# Using uv (recommended)
uv run python scripts/build_srl_data.py

# Or with standard Python
python scripts/build_srl_data.py
```

### Customization

To use a different dataset, modify the `load_teacher_dataset()` call in the `__main__` block:

```python
ds = load_teacher_dataset("your-dataset/name", split="train")
```

To adjust field names for your dataset, modify `normalize_trajectory()`:
- Change `row.get("question")` to match your problem field
- Change `row.get("cot")` to match your reasoning field name

### Step Boundary Detection

The heuristic splitter (`_is_step_boundary`) detects step boundaries using:
- **Pattern**: `"Step N:"` or `"Step N."` (e.g., "Step 1:", "Step 2.")
- **Pattern**: `"N."` where N is a number (e.g., "1.", "2.", "10.")
- **Pattern**: Ordinal words (e.g., "First,", "Second,", "Third,")

### Filtering Statistics

The script automatically filters out invalid trajectories. You'll see output like:
```
Normalized 664 trajectories from 1000 raw examples
```

This means 336 trajectories were filtered out due to:
- Missing required fields
- Empty reasoning
- Unable to split into valid steps

### Progress Tracking

The script uses `tqdm` to show progress bars for:
- Normalizing trajectories
- Building SRL examples
- Saving to JSONL

