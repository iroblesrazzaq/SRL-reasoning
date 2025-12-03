"""SRL-Reasoning: shared, SRL, and SFT utilities."""

from .shared import (
    parse_model_output,
    STOP_TOKENS,
    build_srl_prompt,
    build_srl_prompt_with_target,
    format_srl_prompt,
    extract_next_step_from_output,
    generate_student_step,
    generate_student_step_batch,
    compute_token_logprobs,
    split_by_trajectory,
    save_splits,
    load_teacher_dataset,
    normalize_trajectory,
    normalize_dataset,
    build_srl_examples,
    build_srl_dataset,
    save_jsonl,
)
from .srl import compute_srl_reward, dynamic_sampling_filter, compute_advantages
from .sft import StepDataset, DataCollator

__all__ = [
    # Shared utilities
    "parse_model_output",
    "STOP_TOKENS",
    "build_srl_prompt",
    "build_srl_prompt_with_target",
    "format_srl_prompt",
    "extract_next_step_from_output",
    "generate_student_step",
    "generate_student_step_batch",
    "compute_token_logprobs",
    "split_by_trajectory",
    "save_splits",
    "load_teacher_dataset",
    "normalize_trajectory",
    "normalize_dataset",
    "build_srl_examples",
    "build_srl_dataset",
    "save_jsonl",
    # SRL-specific
    "compute_srl_reward",
    "dynamic_sampling_filter",
    "compute_advantages",
    # SFT-specific
    "StepDataset",
    "DataCollator",
]
