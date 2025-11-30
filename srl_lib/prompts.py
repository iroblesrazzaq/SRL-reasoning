"""Prompt formatting utilities for SRL training and evaluation."""

from typing import List, Optional


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

