"""Prompt formatting utilities for SRL training and evaluation."""

from typing import List, Optional


SRL_SYSTEM_PROMPT = """You are a helpful assistant for solving mathematical problems. 
A user will provide a math problem, which may include a partial solution. 
Your task is to continue the solution by providing the very next logical step.

A user will ask you to solve a task. 
You should first draft your thinking process (inner monologue). 
Then, generate the solution.

Your response format must follow the template below:

<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. 
Be as casual and as long as you want until you are confident to generate a correct solution.
</think>
Provide only the single, next step to continue the solution. 
Do not solve the entire problem."""

# Stop tokens for generation
STOP_TOKENS = ["</think>"]


def build_prompt(
    problem: str,
    previous_steps: List[str],
    step_title: Optional[str] = None,
) -> str:
    """
    Build a prompt for step-wise reasoning using the SRL system prompt.
    
    Format: [system prompt, problem, steps 1...k-1, title of kth step]
    
    Args:
        problem: The problem statement
        previous_steps: List of previous reasoning steps (can be empty for first step)
        step_title: Optional title of the next step to generate
        
    Returns:
        The formatted prompt string
    """
    parts = [SRL_SYSTEM_PROMPT, "", f"Question: {problem}"]
    
    if previous_steps:
        parts.append("")
        for i, step in enumerate(previous_steps, 1):
            parts.append(f"{i}. {step}")
    
    if step_title:
        parts.append("")
        parts.append(f"Next step: {step_title}")
    
    return "\n".join(parts)


def build_prompt_with_target(
    problem: str,
    previous_steps: List[str],
    target_step: str,
    step_title: Optional[str] = None,
) -> str:
    """
    Build a complete prompt with the target step (for SFT training).
    
    Args:
        problem: The problem statement
        previous_steps: List of previous reasoning steps
        target_step: The target step the model should predict
        step_title: Optional title of the step
        
    Returns:
        Complete prompt with target step
    """
    prompt = build_prompt(problem, previous_steps, step_title)
    return prompt + "\n" + target_step.strip()


def extract_next_step_from_output(text: str) -> str:
    """
    Extract the generated step from model output.
    
    Looks for content after </think> tag.
    
    Args:
        text: The model's generated text
        
    Returns:
        The extracted step text, or empty string if not found
    """
    end_tag = "</think>"
    end_idx = text.find(end_tag)
    
    if end_idx == -1:
        return text.strip()
    
    # Get everything after </think>
    return text[end_idx + len(end_tag):].strip()

