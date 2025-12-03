"""Prompt helpers shared across SRL and SFT flows."""

from typing import List, Optional

# Stop tokens for generation
STOP_TOKENS = ["</think>"]

PROMPT_PREAMBLE = (
    "You are a helpful assistant for solving mathematical problems. "
    "A user will provide a math problem, which may include a partial solution. "
    "Your task is to continue the solution by providing the very next logical step. "
    "A user will ask you to solve a task. You should first draft your thinking process "
    "(inner monologue). Then, generate the solution. "
    "Your response format must follow the template below:\n"
    "<think> Your thoughts or/and draft, like working through an exercise on scratch paper. "
    "Be as casual and as long as you want until you are confident to generate a correct solution. </think>\n"
    "Provide only the single, next step to continue the solution. Do not solve the entire problem."
)


def build_srl_prompt(
    problem: str,
    previous_steps: List[str],
    step_title: Optional[str] = None,
    include_closing_tag: bool = True,
) -> str:
    """
    Build the SRL prompt matching the paper wording, including problem and prior steps.
    
    Args:
        problem: The problem statement.
        previous_steps: Steps generated so far (may be empty).
        step_title: Optional title of the upcoming step.
        include_closing_tag: Retained for backward compatibility (no effect).
    """
    parts: List[str] = [
        PROMPT_PREAMBLE,
        "",
        "Problem:",
        problem.strip(),
        "",

    ]
    if previous_steps:
        for i, step in enumerate(previous_steps, 1):
            parts.append(step.strip())
    if step_title:
        parts.append(step_title.strip())
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
    target = "<think></think>\n" + target_step.strip()
    return prompt + "\n" + target


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
