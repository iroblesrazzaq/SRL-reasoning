"""Prompt helpers shared across SRL and SFT flows."""

from typing import List, Optional

# Stop tokens for generation
STOP_TOKENS = ["</think>"]


def build_srl_prompt(
    problem: str,
    previous_steps: List[str],
    step_title: Optional[str] = None,
    include_closing_tag: bool = True,
) -> str:
    """
    Build a structured prompt for the next-step task.
    
    Args:
        problem: The problem statement
        previous_steps: Steps generated so far (may be empty)
        step_title: Optional title of the upcoming step
        include_closing_tag: If False, leaves the reasoning block open (useful for appending targets)
    """
    parts = [
        "<system>You are a helpful assistant. Continue the solution one step at a time.</system>",
        "<problem>",
        problem.strip(),
        "</problem>",
        "<reasoning_so_far>",
    ]
    for step in previous_steps:
        parts.append(step.strip())
    if include_closing_tag:
        parts.append("</reasoning_so_far>")
    if step_title:
        parts.append(f"<next_step_title>{step_title.strip()}</next_step_title>")
    parts.append("<next_step>")
    return "\n".join(parts)


def build_srl_prompt_with_target(
    problem: str,
    previous_steps: List[str],
    target_step: str,
    step_title: Optional[str] = None,
) -> str:
    """
    Build the full prompt including the target step (for SFT teacher forcing).
    """
    prompt = build_srl_prompt(
        problem=problem,
        previous_steps=previous_steps,
        step_title=step_title,
        include_closing_tag=True,
    )
    return prompt + "\n" + target_step.strip() + "\n</next_step>"


def format_srl_prompt(
    problem: str,
    previous_steps: List[str],
    step_title: Optional[str] = None,
    include_closing_tag: bool = True,
) -> str:
    """
    Backward-compatible alias for SRL prompt construction.
    """
    return build_srl_prompt(
        problem=problem,
        previous_steps=previous_steps,
        step_title=step_title,
        include_closing_tag=include_closing_tag,
    )


def extract_next_step_from_output(text: str) -> str:
    """
    Extract the generated step from model output after the </think> block.
    """
    end_tag = "</think>"
    end_idx = text.find(end_tag)
    if end_idx == -1:
        return text.strip()
    return text[end_idx + len(end_tag):].strip()
