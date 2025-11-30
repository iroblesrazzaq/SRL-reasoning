"""Data splitting utilities for SRL training.

Splits data by problem (traj_id) to avoid leakage between train/val/test.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def split_by_trajectory(
    data_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split SRL data by trajectory ID to avoid leakage.
    
    Groups all examples by traj_id, then splits trajectories (not individual steps)
    into train/val/test. This ensures that all steps from the same problem
    stay in the same split.
    
    Args:
        data_path: Path to JSONL file with SRL examples
        train_ratio: Fraction of trajectories for training
        val_ratio: Fraction of trajectories for validation
        test_ratio: Fraction of trajectories for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_examples, val_examples, test_examples)
        
    Example:
        >>> train, val, test = split_by_trajectory("data/srl_steps.jsonl")
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    import random
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Load all examples
    examples = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    # Group by trajectory ID
    traj_groups = defaultdict(list)
    for ex in examples:
        traj_id = ex["traj_id"]
        traj_groups[traj_id].append(ex)
    
    # Get unique trajectory IDs
    traj_ids = list(traj_groups.keys())
    
    # Shuffle with seed
    rng = random.Random(seed)
    rng.shuffle(traj_ids)
    
    # Calculate split indices
    n_trajs = len(traj_ids)
    train_end = int(n_trajs * train_ratio)
    val_end = train_end + int(n_trajs * val_ratio)
    
    # Split trajectory IDs
    train_traj_ids = set(traj_ids[:train_end])
    val_traj_ids = set(traj_ids[train_end:val_end])
    test_traj_ids = set(traj_ids[val_end:])
    
    # Assign examples to splits based on their trajectory ID
    train_examples = []
    val_examples = []
    test_examples = []
    
    for ex in examples:
        traj_id = ex["traj_id"]
        if traj_id in train_traj_ids:
            train_examples.append(ex)
        elif traj_id in val_traj_ids:
            val_examples.append(ex)
        elif traj_id in test_traj_ids:
            test_examples.append(ex)
        else:
            # Shouldn't happen, but handle gracefully
            train_examples.append(ex)
    
    return train_examples, val_examples, test_examples


def save_splits(
    train_examples: List[Dict],
    val_examples: List[Dict],
    test_examples: List[Dict],
    output_dir: str = "data",
) -> None:
    """
    Save train/val/test splits to separate JSONL files.
    
    Args:
        train_examples: Training examples
        val_examples: Validation examples
        test_examples: Test examples
        output_dir: Directory to save the split files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save each split
    for split_name, examples in [
        ("train", train_examples),
        ("val", val_examples),
        ("test", test_examples),
    ]:
        output_file = output_path / f"{split_name}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Saved {len(examples)} examples to {output_file}")

