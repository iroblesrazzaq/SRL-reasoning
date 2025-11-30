#!/usr/bin/env python3
"""CLI runner for math benchmark evaluation."""

import argparse
import sys

from .data_loader import load_benchmark_data
from .evaluator import MathEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reasoning models on math benchmarks."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model or HuggingFace model ID.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["amc23", "aime24", "aime25"],
        required=True,
        help="Benchmark to evaluate on.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["greedy", "avg32"],
        default="greedy",
        help="Evaluation mode: 'greedy' (single sample) or 'avg32' (32 samples).",
    )
    
    args = parser.parse_args()
    
    # Load benchmark data
    print(f"Loading {args.benchmark}...")
    try:
        data = load_benchmark_data(args.benchmark)
        print(f"Loaded {len(data)} problems.")
    except Exception as e:
        print(f"Error loading benchmark: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize evaluator
    print(f"Initializing vLLM with {args.model_path}...")
    try:
        evaluator = MathEvaluator(args.model_path)
    except Exception as e:
        print(f"Error initializing model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run evaluation
    print(f"Evaluating with mode={args.mode}...")
    try:
        score = evaluator.evaluate(data, mode=args.mode)
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print result
    print()
    print(f"[RESULT] Benchmark: {args.benchmark} | Mode: {args.mode} | Score: {score:.2%}")


if __name__ == "__main__":
    main()

