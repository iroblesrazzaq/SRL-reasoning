"""Data processing utilities for SRL training."""

from .build_srl_data import (
    load_teacher_dataset,
    normalize_dataset,
    build_srl_dataset,
    save_jsonl,
    build_srl_examples,
)
from .splits import split_by_trajectory, save_splits
from .dataset import StepDataset, DataCollator

__all__ = [
    "load_teacher_dataset",
    "normalize_dataset",
    "build_srl_dataset",
    "build_srl_examples",
    "save_jsonl",
    "split_by_trajectory",
    "save_splits",
    "StepDataset",
    "DataCollator",
]

from .build_srl_data import (
    load_teacher_dataset,
    normalize_trajectory,
    normalize_dataset,
    build_srl_examples,
    build_srl_dataset,
    save_jsonl,
    main as build_data,
)

__all__ = [
    "load_teacher_dataset",
    "normalize_trajectory",
    "normalize_dataset",
    "build_srl_examples",
    "build_srl_dataset",
    "save_jsonl",
    "build_data",
]

