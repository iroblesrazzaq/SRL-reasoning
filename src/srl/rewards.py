"""Reward computation for SRL training."""

import difflib
from .formatting import parse_model_output


def compute_srl_reward(model_completion: str, expert_target: str) -> float:
    """
    Compute the SRL reward based on similarity between model action and expert target.
    
    Uses the formula: R = (2 * M) / T
    where:
        M = sum of lengths of all matching blocks
        T = total length of both strings (len(pred_action) + len(expert_target))
    
    Args:
        model_completion: The full model output including <think>...</think> tags.
        expert_target: The expected action from the expert trajectory.
        
    Returns:
        A reward value:
        - -1.0 if the model output has format errors (missing </think>)
        - 0.0 to 1.0 based on similarity between predicted action and expert target
    """
    # Parse the model output to extract the action
    thought, pred_action = parse_model_output(model_completion)
    
    # Penalty for format errors
    if pred_action is None:
        return -1.0
    
    # Calculate total length T
    total_length = len(pred_action) + len(expert_target)
    
    # Handle division by zero (both strings are empty)
    if total_length == 0:
        return 0.0  # Avoid rewarding empty actions; treat as no signal
    
    # Use SequenceMatcher to find matching blocks
    matcher = difflib.SequenceMatcher(None, pred_action, expert_target)
    matching_blocks = matcher.get_matching_blocks()
    
    # Sum of lengths of all matching blocks (M)
    # Note: get_matching_blocks() returns a final dummy block with size 0
    matching_length = sum(match.size for match in matching_blocks)
    
    # Compute reward: R = (2 * M) / T
    reward = (2.0 * matching_length) / total_length
    
    return reward
