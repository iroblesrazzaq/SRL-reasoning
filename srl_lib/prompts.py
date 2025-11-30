"""Prompt formatting utilities for SRL training and evaluation."""

from typing import List, Optional


# ============================================================================
# XML-Style Prompt Template (for SFT training and structured generation)
# ============================================================================

def build_prompt(
    problem: str,
    previous_steps: List[str],
    include_closing_tag: bool = False,
) -> str:
    """
    Build an XML-style prompt for step-wise reasoning (per spec).
    
    Creates a structured prompt with:
    - <problem>: The problem statement
    - <reasoning_so_far>: Previous reasoning steps (if any)
    - <instructions>: Instructions for the model
    - <next_step>: Marker where generation should start
    
    This is the spec-compliant prompt format that makes it easy to:
    - Mask prompt tokens in SFT training (everything before <next_step>)
    - Extract generated steps (everything after <next_step>)
    - Set clear stop conditions (stop on </next_step>)
    
    Args:
        problem: The problem statement
        previous_steps: List of previous reasoning steps (can be empty for first step)
        include_closing_tag: If True, include </next_step> in the prompt (for training targets)
        
    Returns:
        The formatted prompt string
        
    Example:
        >>> prompt = build_prompt(
        ...     problem="Solve 2+2",
        ...     previous_steps=["First, I need to add 2 and 2"]
        ... )
        >>> # Returns XML-structured prompt ending with <next_step>
    """
    lines = []
    
    # Problem section
    lines.append("<problem>")
    lines.append(problem.strip())
    lines.append("</problem>")
    lines.append("")  # blank line
    
    # Reasoning so far section
    lines.append("<reasoning_so_far>")
    if previous_steps:
        for i, step in enumerate(previous_steps):
            lines.append(f'<step index="{i}">')
            lines.append(step.strip())
            lines.append("</step>")
    else:
        # Empty reasoning section for first step
        lines.append("(No previous steps)")
    lines.append("</reasoning_so_far>")
    lines.append("")  # blank line
    
    # Instructions section
    lines.append("<instructions>")
    lines.append(
        "You are solving the problem step by step.\n"
        "Only output the next reasoning step.\n"
        "Do not restate the problem.\n"
        "Do not output the final answer.\n"
        "Limit yourself to at most 3-5 sentences."
    )
    lines.append("</instructions>")
    lines.append("")  # blank line
    
    # Generation anchor
    lines.append("<next_step>")
    if include_closing_tag:
        lines.append("")  # Will be filled with teacher_step + closing tag
        lines.append("</next_step>")
    
    return "\n".join(lines)


def build_prompt_with_target(
    problem: str,
    previous_steps: List[str],
    teacher_step: str,
) -> str:
    """
    Build a complete prompt with the target step (for SFT training).
    
    This is the same as build_prompt() but includes the teacher_step and closing tag.
    Used for creating training examples where the model learns to predict the step.
    
    Args:
        problem: The problem statement
        previous_steps: List of previous reasoning steps
        teacher_step: The target step the model should predict
        
    Returns:
        Complete prompt with target step and closing tag
    """
    prompt = build_prompt(problem, previous_steps, include_closing_tag=False)
    return prompt + "\n" + teacher_step.strip() + "\n</next_step>"


def extract_next_step_from_output(text: str) -> str:
    """
    Extract the generated step from model output.
    
    Looks for content after <next_step> tag and before </next_step> or other stop tokens.
    
    Args:
        text: The model's generated text (may include the full prompt + generation)
        
    Returns:
        The extracted step text, or empty string if not found
    """
    # Find where generation starts
    start_tag = "<next_step>"
    start_idx = text.find(start_tag)
    
    if start_idx == -1:
        return ""
    
    # Get everything after <next_step>
    start_idx += len(start_tag)
    remaining = text[start_idx:].strip()
    
    # Stop at closing tag or other stop tokens
    stop_tokens = ["</next_step>", "</reasoning_so_far>", "<problem>"]
    for stop_token in stop_tokens:
        if stop_token in remaining:
            remaining = remaining.split(stop_token)[0]
            break
    
    return remaining.strip()


# Stop tokens for generation
STOP_TOKENS = ["</next_step>", "</reasoning_so_far>", "<problem>"]


# ============================================================================
# Legacy Prompt Formats (for backward compatibility)
# ============================================================================


# System prompt for SRL training/inference with think tags
SRL_SYSTEM_PROMPT = """You are a helpful assistant for solving mathematical problems. A user will provide a math problem, which may include a partial solution. Your task is to continue the solution by providing the very next logical step. You should first draft your thinking process (inner monologue). Then, generate the solution. Your response format must follow the template below: <think> Your thoughts... </think> Provide only the single, next step to continue the solution."""

# System prompt for base model evaluation (no think tags expected)
BASE_SYSTEM_PROMPT = """You are a helpful assistant for solving mathematical problems. Solve the problem step by step and provide your final answer in \\boxed{}."""


def format_srl_prompt(
    problem: str,
    previous_steps: List[str],
    step_title: Optional[str] = None,
) -> str:
    """
    Format a prompt for SRL training or inference.
    
    This creates the input context for the model to generate the next reasoning step.
    When step_title is provided (Context Injection), the model receives a hint about
    what the next step should be, which empirically boosts performance per the paper.
    
    Args:
        problem: The original math problem.
        previous_steps: List of previous reasoning steps (may be empty for first step).
        step_title: Optional title/header of the current step to predict (context injection).
        
    Returns:
        Formatted prompt string ending with <think> tag for model to continue.
    """
    parts = [SRL_SYSTEM_PROMPT, "", f"Question: {problem}"]
    
    # Add previous steps as context if any
    if previous_steps:
        parts.append("")
        parts.append("Previous steps:")
        for i, step in enumerate(previous_steps, 1):
            parts.append(f"{i}. {step}")
    
    # Add step title hint if provided (Context Injection)
    if step_title:
        parts.append("")
        parts.append(f"Next step: {step_title}")
    
    # End with think tag for model to continue
    parts.append("<think>")
    
    return "\n".join(parts)


def format_base_prompt(problem: str) -> str:
    """
    Format a standard prompt for base model evaluation.
    
    This is used when evaluating the base Qwen model (not SRL-trained).
    The prompt doesn't include think tags since the base model wasn't
    trained with that format.
    
    Args:
        problem: The math problem to solve.
        
    Returns:
        Formatted prompt string for standard generation.
    """
    return f"{BASE_SYSTEM_PROMPT}\n\nQuestion: {problem}\n\nSolution:"


def format_srl_eval_prompt(problem: str) -> str:
    """
    Format a prompt for SRL model evaluation.
    
    This is used when evaluating an SRL-trained model on benchmarks.
    The model is expected to generate a full solution with think/action format.
    
    Args:
        problem: The math problem to solve.
        
    Returns:
        Formatted prompt string ending with <think> tag.
    """
    prompt = f"""{SRL_SYSTEM_PROMPT}

Question: {problem}
<think>"""
    return prompt


def format_chat_messages(
    problem: str,
    previous_steps: List[str] = None,
    step_title: Optional[str] = None,
    model_type: str = "srl",
) -> List[dict]:
    """
    Format prompt as chat messages for models that expect chat format.
    
    This is useful for models like Qwen2.5-Instruct that work best with
    the chat template format.
    
    Args:
        problem: The math problem to solve.
        previous_steps: List of previous reasoning steps (may be None or empty).
        step_title: Optional step title for context injection.
        model_type: Either "srl" or "base" to determine prompt style.
        
    Returns:
        List of message dicts with "role" and "content" keys.
    """
    if model_type == "base":
        system_content = BASE_SYSTEM_PROMPT
        user_content = f"Question: {problem}"
    else:
        system_content = SRL_SYSTEM_PROMPT
        
        # Build user content
        user_parts = [f"Question: {problem}"]
        
        if previous_steps:
            user_parts.append("")
            user_parts.append("Previous steps:")
            for i, step in enumerate(previous_steps, 1):
                user_parts.append(f"{i}. {step}")
        
        if step_title:
            user_parts.append("")
            user_parts.append(f"Next step: {step_title}")
        
        user_content = "\n".join(user_parts)
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    
    # For SRL models, add the start of assistant response with think tag
    if model_type == "srl":
        messages.append({"role": "assistant", "content": "<think>"})
    
    return messages

