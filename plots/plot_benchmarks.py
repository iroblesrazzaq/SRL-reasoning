"""
Benchmark Results Plotting Script

Plots bar graphs of benchmark performance for different models.
Designed to run in Google Colab with Drive mounted.
"""

import json
import os
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


# Constants
DRIVE_RESULTS_FOLDER = "/Users/ismaelrobles-razzaq/Library/CloudStorage/GoogleDrive-ismael.rr.54321@gmail.com/My Drive/srl_bench_results"
VALID_BENCHMARKS = ["amc23", "aime25"]
VALID_EVAL_TYPES = ["greedy", "avg32"]

# Mapping for display names
BENCHMARK_DISPLAY = {
    "amc23": "AMC23",
    "aime25": "AIME25",
}

EVAL_TYPE_DISPLAY = {
    "greedy": "Greedy",
    "avg32": "Avg@32",
}


def mount_drive():
    """Mount Google Drive in Colab environment."""
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("Drive mounted successfully.")
    except ImportError:
        print("Not running in Colab - assuming Drive is already accessible or using local files.")
    except Exception as e:
        print(f"Error mounting drive: {e}")


def load_results(results_folder: str = DRIVE_RESULTS_FOLDER) -> list[dict]:
    """
    Load all JSON result files from the specified folder.
    
    Args:
        results_folder: Path to folder containing JSON result files.
        
    Returns:
        List of result dictionaries.
    """
    results = []
    folder = Path(results_folder)
    
    if not folder.exists():
        raise FileNotFoundError(f"Results folder not found: {results_folder}")
    
    for json_file in folder.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                results.append(data)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {json_file.name}: {e}")
        except Exception as e:
            print(f"Warning: Error reading {json_file.name}: {e}")
    
    print(f"Loaded {len(results)} result files from {results_folder}")
    return results


def get_model_short_name(model_path: str) -> str:
    """
    Extract short model name from model path.
    
    Args:
        model_path: Full model path (e.g., 'Qwen/Qwen3-4B-Instruct-2507')
        
    Returns:
        Short name (e.g., 'Qwen3-4B-Instruct-2507')
    """
    return model_path.split("/")[-1] if "/" in model_path else model_path


def normalize_eval_type(benchmark_type: str) -> str:
    """
    Normalize benchmark_type field to standard eval type.
    
    Args:
        benchmark_type: Raw benchmark type from JSON (e.g., 'Avg32', 'Greedy')
        
    Returns:
        Normalized type ('greedy' or 'avg32')
    """
    bt_lower = benchmark_type.lower()
    if "avg" in bt_lower or "32" in bt_lower:
        return "avg32"
    elif "greedy" in bt_lower:
        return "greedy"
    return bt_lower


def filter_results(
    results: list[dict],
    benchmarks: list[str] | str = "all",
    eval_types: list[str] | str = "all",
    models: list[str] | str = "all",
) -> list[dict]:
    """
    Filter results by benchmark, eval type, and model.
    
    Args:
        results: List of result dictionaries.
        benchmarks: List of benchmarks or 'all'.
        eval_types: List of eval types ('greedy', 'avg32') or 'all'.
        models: List of model paths/names or 'all'.
        
    Returns:
        Filtered list of results.
    """
    filtered = []
    
    # Normalize inputs
    if benchmarks == "all":
        benchmarks = VALID_BENCHMARKS
    if eval_types == "all":
        eval_types = VALID_EVAL_TYPES
    if isinstance(benchmarks, str):
        benchmarks = [benchmarks]
    if isinstance(eval_types, str):
        eval_types = [eval_types]
    
    # Normalize model list for comparison
    if models != "all":
        if isinstance(models, str):
            models = [models]
        # Create both full paths and short names for matching
        model_short_names = {get_model_short_name(m).lower() for m in models}
        model_full_paths = {m.lower() for m in models}
    
    for r in results:
        benchmark = r.get("benchmark", "").lower()
        eval_type = normalize_eval_type(r.get("benchmark_type", ""))
        model_path = r.get("model_path", "")
        model_short = get_model_short_name(model_path)
        
        # Check benchmark
        if benchmark not in [b.lower() for b in benchmarks]:
            continue
        
        # Check eval type
        if eval_type not in [e.lower() for e in eval_types]:
            continue
        
        # Check model
        if models != "all":
            if model_path.lower() not in model_full_paths and model_short.lower() not in model_short_names:
                continue
        
        filtered.append(r)
    
    return filtered


def get_unique_values(results: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """
    Get unique benchmarks, eval types, and models from results.
    
    Returns:
        Tuple of (benchmarks, eval_types, model_short_names)
    """
    benchmarks = set()
    eval_types = set()
    models = set()
    
    for r in results:
        benchmarks.add(r.get("benchmark", "").lower())
        eval_types.add(normalize_eval_type(r.get("benchmark_type", "")))
        models.add(get_model_short_name(r.get("model_path", "")))
    
    # Sort for consistent ordering
    benchmarks = sorted([b for b in benchmarks if b in VALID_BENCHMARKS])
    eval_types = sorted([e for e in eval_types if e in VALID_EVAL_TYPES], 
                        key=lambda x: VALID_EVAL_TYPES.index(x))
    models = sorted(models)
    
    return benchmarks, eval_types, models


def plot_benchmark_results(
    benchmarks: list[str] | str = "all",
    eval_types: list[str] | str = "all",
    models: list[str] | str = "all",
    save_pdf: bool = True,
    save_to_drive: bool = False,
    drive_save_path: str | None = None,
    output_filename: str = "benchmark_results.pdf",
    results_folder: str = DRIVE_RESULTS_FOLDER,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Plot benchmark results as a grouped bar chart.
    
    Models are on the x-axis, grouped by benchmark/eval_type combination.
    Greedy results use solid bars, Avg@32 uses hatched bars.
    
    Args:
        benchmarks: List of benchmarks ('amc23', 'aime25') or 'all'.
        eval_types: List of eval types ('greedy', 'avg32') or 'all'.
        models: List of model paths/names or 'all'.
        save_pdf: Whether to save the plot as PDF.
        save_to_drive: If True, save PDF to Google Drive instead of local.
        drive_save_path: Path within Drive to save PDF (if save_to_drive=True).
                        Defaults to results folder.
        output_filename: Name of the output PDF file.
        results_folder: Path to folder containing JSON result files.
        title: Custom title for the plot. Auto-generated if None.
        figsize: Custom figure size (width, height). Auto-calculated if None.
        
    Returns:
        The matplotlib Figure object.
    """
    # Load and filter results
    all_results = load_results(results_folder)
    filtered = filter_results(all_results, benchmarks, eval_types, models)
    
    if not filtered:
        print("No results match the specified filters.")
        return None
    
    # Get unique values from filtered results
    unique_benchmarks, unique_eval_types, unique_models = get_unique_values(filtered)
    
    print(f"Plotting: {len(unique_models)} models, {len(unique_benchmarks)} benchmarks, {len(unique_eval_types)} eval types")
    
    # Build data structure: {model: {(benchmark, eval_type): score}}
    data = {}
    for r in filtered:
        model = get_model_short_name(r.get("model_path", ""))
        benchmark = r.get("benchmark", "").lower()
        eval_type = normalize_eval_type(r.get("benchmark_type", ""))
        score = r.get("score", 0)
        
        if model not in data:
            data[model] = {}
        data[model][(benchmark, eval_type)] = score
    
    # Create benchmark/eval_type combinations for grouping
    groups = []
    for benchmark in unique_benchmarks:
        for eval_type in unique_eval_types:
            groups.append((benchmark, eval_type))
    
    # Setup figure
    n_models = len(unique_models)
    n_groups = len(groups)
    
    if figsize is None:
        # Auto-calculate figure size
        width = max(10, n_models * n_groups * 0.4 + 2)
        height = 6
        figsize = (width, height)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Bar positioning
    x = np.arange(n_models)
    total_width = 0.8  # Total width for all bars in a group
    bar_width = total_width / n_groups if n_groups > 0 else 0.8
    
    # Colors for different benchmarks
    colors = {
        "amc23": "#2ecc71",   # Green
        "aime25": "#3498db",  # Blue
    }
    
    # Hatch patterns for eval types
    hatches = {
        "greedy": "",      # Solid
        "avg32": "///",    # Hatched
    }
    
    # Plot bars for each group
    bars_for_legend = {}
    for i, (benchmark, eval_type) in enumerate(groups):
        offset = (i - n_groups / 2 + 0.5) * bar_width
        scores = []
        
        for model in unique_models:
            score = data.get(model, {}).get((benchmark, eval_type), None)
            scores.append(score if score is not None else np.nan)
        
        # Convert to numpy array for plotting
        scores_arr = np.array(scores, dtype=float)
        
        # Create bars (skip NaN values automatically via matplotlib)
        label = f"{BENCHMARK_DISPLAY.get(benchmark, benchmark)} {EVAL_TYPE_DISPLAY.get(eval_type, eval_type)}"
        bar = ax.bar(
            x + offset,
            scores_arr,
            bar_width * 0.9,  # Slight gap between bars
            label=label,
            color=colors.get(benchmark, "#95a5a6"),
            hatch=hatches.get(eval_type, ""),
            edgecolor="black",
            linewidth=0.5,
        )
        bars_for_legend[label] = bar
    
    # Customize plot
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_models, rotation=45, ha="right", fontsize=10)
    
    # Title
    if title is None:
        title = "Benchmark Performance by Model"
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Legend
    ax.legend(loc="upper right", fontsize=10)
    
    # Grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or display
    if save_pdf:
        if save_to_drive:
            save_path = drive_save_path if drive_save_path else results_folder
            output_path = os.path.join(save_path, output_filename)
        else:
            output_path = output_filename
        
        plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Saved plot to: {output_path}")
    
    plt.show()
    return fig


def list_available_results(results_folder: str = DRIVE_RESULTS_FOLDER) -> None:
    """
    Print summary of available results in the folder.
    
    Args:
        results_folder: Path to folder containing JSON result files.
    """
    results = load_results(results_folder)
    benchmarks, eval_types, models = get_unique_values(results)
    
    print("\n=== Available Results Summary ===")
    print(f"\nBenchmarks: {benchmarks}")
    print(f"Eval Types: {eval_types}")
    print(f"Models: {models}")
    
    print("\n=== Results Matrix ===")
    print(f"{'Model':<40} | ", end="")
    for b in benchmarks:
        for e in eval_types:
            label = f"{b}_{e}"
            print(f"{label:<15}", end="")
    print()
    print("-" * (40 + 3 + len(benchmarks) * len(eval_types) * 15))
    
    # Build lookup
    lookup = {}
    for r in results:
        model = get_model_short_name(r.get("model_path", ""))
        benchmark = r.get("benchmark", "").lower()
        eval_type = normalize_eval_type(r.get("benchmark_type", ""))
        score = r.get("score", 0)
        lookup[(model, benchmark, eval_type)] = score
    
    for model in models:
        print(f"{model:<40} | ", end="")
        for b in benchmarks:
            for e in eval_types:
                score = lookup.get((model, b, e), None)
                if score is not None:
                    print(f"{score:<15.3f}", end="")
                else:
                    print(f"{'--':<15}", end="")
        print()
