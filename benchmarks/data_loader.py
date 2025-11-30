"""Data loader for math benchmarks using Hugging Face datasets."""

from typing import List, Dict
from datasets import load_dataset


# Dataset mappings: benchmark_name -> (dataset_id, split)
BENCHMARK_CONFIGS = {
    "amc23": ("AI-MO/aimo-validation-amc", "train"),
    "aime24": ("AI-MO/aimo-validation-aime", "train"),
    "aime25": ("math-ai/aime25", "train"),
}


def load_benchmark_data(benchmark_name: str) -> List[Dict[str, str]]:
    """
    Load benchmark data from Hugging Face datasets.
    
    Args:
        benchmark_name: One of 'amc23', 'aime24', or 'aime25'.
        
    Returns:
        A list of dictionaries with 'problem' and 'solution' keys.
        
    Raises:
        ValueError: If benchmark_name is not recognized.
    """
    if benchmark_name not in BENCHMARK_CONFIGS:
        valid_names = list(BENCHMARK_CONFIGS.keys())
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. "
            f"Valid options are: {valid_names}"
        )
    
    dataset_id, split = BENCHMARK_CONFIGS[benchmark_name]
    dataset = load_dataset(dataset_id, split=split)
    
    # Normalize the data to a consistent format
    data = []
    for item in dataset:
        problem = _extract_problem(item)
        solution = _extract_solution(item)
        data.append({"problem": problem, "solution": solution})
    
    return data


def _extract_problem(item: Dict) -> str:
    """Extract the problem text from a dataset item."""
    # Try common field names for problem text
    for field in ["problem", "question", "input", "prompt"]:
        if field in item:
            return str(item[field])
    
    # Fallback: return the first string field
    for value in item.values():
        if isinstance(value, str) and len(value) > 10:
            return value
    
    raise KeyError(f"Could not find problem field in item: {list(item.keys())}")


def _extract_solution(item: Dict) -> str:
    """Extract the ground truth solution/answer from a dataset item."""
    # Try common field names for the answer
    for field in ["answer", "solution", "output", "target", "expected"]:
        if field in item:
            return str(item[field])
    
    # Fallback: return the last string field that looks like an answer
    for key in reversed(list(item.keys())):
        value = item[key]
        if isinstance(value, (str, int, float)):
            return str(value)
    
    raise KeyError(f"Could not find solution field in item: {list(item.keys())}")

