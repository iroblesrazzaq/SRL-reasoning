"""Benchmarking suite for SRL reasoning models."""

from typing import TYPE_CHECKING

from .data_loader import load_benchmark_data
from .results import BenchmarkResult, load_all_results, summarize_results

if TYPE_CHECKING:  # pragma: no cover
    # For type checkers / IDEs; actual import is lazy to avoid vllm dependency
    from .evaluator import MathEvaluator, extract_answer, is_correct


def __getattr__(name):
    """Lazy-import heavy evaluator bits to avoid vllm import unless needed."""
    if name in {"MathEvaluator", "extract_answer", "is_correct"}:
        try:
            from .evaluator import MathEvaluator, extract_answer, is_correct
        except ModuleNotFoundError as e:  # Likely missing vllm
            raise ImportError(
                f"{name} requires the vllm dependency. Install vllm or import from benchmarks.data_loader/results instead."
            ) from e
        return {"MathEvaluator": MathEvaluator, "extract_answer": extract_answer, "is_correct": is_correct}[name]
    raise AttributeError(f"module 'benchmarks' has no attribute '{name}'")


__all__ = [
    "load_benchmark_data",
    "BenchmarkResult",
    "load_all_results",
    "summarize_results",
    # Lazy-loaded members:
    "MathEvaluator",
    "extract_answer",
    "is_correct",
]
