"""Shared utilities used by both SRL and SFT."""

from .formatting import parse_model_output
from .prompts import (
    STOP_TOKENS,
    build_srl_prompt,
    build_srl_prompt_with_target,
    format_srl_prompt,
    extract_next_step_from_output,
)
from .generation import generate_student_step, generate_student_step_batch, compute_token_logprobs
from .splits import split_by_trajectory, save_splits
from .build_srl_data import (
    load_teacher_dataset,
    normalize_trajectory,
    normalize_dataset,
    build_srl_examples,
    build_srl_dataset,
    save_jsonl,
)

__all__ = [
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
]
