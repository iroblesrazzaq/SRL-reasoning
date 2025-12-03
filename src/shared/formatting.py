"""Formatting utilities for parsing model outputs."""

from typing import Tuple, Optional


def parse_model_output(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse model output to extract thought and action components.
    
    The model output format is strictly: <think> [internal monologue] </think> [action step].
    
    Args:
        text: The raw model output string.
        
    Returns:
        A tuple (thought, action) where:
        - thought: Content inside <think>...</think> tags, stripped of whitespace
        - action: Everything after </think>, stripped of whitespace
        
        Returns (None, None) if </think> is missing.
    """
    if text is None:
        return (None, None)
    
    # Check if </think> tag is present - this is required
    close_tag = "</think>"
    close_idx = text.find(close_tag)
    
    if close_idx == -1:
        # Missing </think> tag - invalid format
        return (None, None)
    
    # Extract the action (everything after </think>)
    action = text[close_idx + len(close_tag):].strip()
    
    # Extract the thought (content between <think> and </think>)
    open_tag = "<think>"
    open_idx = text.find(open_tag)
    
    if open_idx == -1:
        # <think> is missing but </think> is present
        # Extract everything before </think> as thought
        thought = text[:close_idx].strip()
    else:
        # Both tags present - extract content between them
        thought = text[open_idx + len(open_tag):close_idx].strip()
    
    return (thought, action)
