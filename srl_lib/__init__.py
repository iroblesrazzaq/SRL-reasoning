"""SRL-Reasoning: Supervised Reinforcement Learning for Step-wise Reasoning."""

from .formatting import parse_model_output
from .rewards import compute_srl_reward
from .grpo_utils import dynamic_sampling_filter, compute_advantages
from .prompts import (
    # XML-style prompts (spec-compliant)
    build_prompt,
    build_prompt_with_target,
    extract_next_step_from_output,
    STOP_TOKENS,
    # Legacy prompts (backward compatibility)
    format_srl_prompt,
    format_base_prompt,
    format_srl_eval_prompt,
    format_chat_messages,
)
from .generation import (
    generate_student_step,
    generate_student_step_batch,
    compute_token_logprobs,
)
from .data.dataset import StepDataset, DataCollator
from .data.splits import split_by_trajectory, save_splits

__all__ = [
    "parse_model_output",
    "compute_srl_reward",
    "dynamic_sampling_filter",
    "compute_advantages",
    # XML-style prompts
    "build_prompt",
    "build_prompt_with_target",
    "extract_next_step_from_output",
    "STOP_TOKENS",
    # Generation utilities
    "generate_student_step",
    "generate_student_step_batch",
    "compute_token_logprobs",
    # SFT training utilities
    "StepDataset",
    "DataCollator",
    "split_by_trajectory",
    "save_splits",
    # Legacy prompts
    "format_srl_prompt",
    "format_base_prompt",
    "format_srl_eval_prompt",
    "format_chat_messages",
]

