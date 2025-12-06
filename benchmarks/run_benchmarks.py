#!/usr/bin/env python3
"""CLI runner for math benchmark evaluation."""

import argparse
import sys
import time
from pathlib import Path

from .data_loader import load_benchmark_data
from .evaluator import MathEvaluator
from .results import BenchmarkResult


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
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional display name for the model (defaults to basename of model_path).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="benchmarks/results",
        help="Directory to store serialized benchmark results.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Optional base model ID for repairing corrupted config/tokenizer files.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["srl", "base"],
        default="srl",
        help="Model type: 'srl' (trained with think tags) or 'base' (standard model).",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="Fraction of GPU memory vLLM should reserve (default: 0.8).",
    )
    
    args = parser.parse_args()
    model_display = args.model_name or Path(args.model_path).name
    
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
        evaluator = MathEvaluator(
            args.model_path,
            model_type=args.model_type,
            gpu_memory_utilization=args.gpu_memory_utilization,
            base_model=args.base_model,
        )
    except Exception as e:
        print(f"Error initializing model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run evaluation
    print(f"Evaluating with mode={args.mode}...")
    start = time.time()
    try:
        score = evaluator.evaluate(data, mode=args.mode)
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.time() - start
    
    # Print result
    print()
    print(f"[RESULT] Benchmark: {args.benchmark} | Mode: {args.mode} | Score: {score:.2%}")
    benchmark_type = "Avg@32" if args.mode == "avg32" else "Greedy"
    result = BenchmarkResult(
        benchmark=args.benchmark,
        benchmark_type=benchmark_type,
        score=score,
        model_name=model_display,
        model_path=args.model_path,
        num_questions=len(data),
        eval_time_seconds=elapsed,
    )
    saved_path = result.save(args.results_dir)
    print(f"Saved result to {saved_path}")


if __name__ == "__main__":
    main()
