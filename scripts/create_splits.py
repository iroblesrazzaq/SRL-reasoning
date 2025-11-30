#!/usr/bin/env python3
"""
Create train/val/test splits from SRL data.

Splits data by trajectory ID to avoid leakage between splits.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from srl_lib.data.splits import split_by_trajectory, save_splits


def main():
    parser = argparse.ArgumentParser(description="Split SRL data by trajectory")
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/srl_steps.jsonl",
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save split files",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of trajectories for training",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of trajectories for validation",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of trajectories for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    print(f"Splitting data from: {args.data_path}")
    print(f"Ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    
    train_examples, val_examples, test_examples = split_by_trajectory(
        args.data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    
    print(f"\nSplit results:")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Val:   {len(val_examples)} examples")
    print(f"  Test:  {len(test_examples)} examples")
    
    save_splits(train_examples, val_examples, test_examples, args.output_dir)
    
    print(f"\nSplits saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

