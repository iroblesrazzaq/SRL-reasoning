"""GRPO (Group Relative Policy Optimization) utilities."""

import torch
from torch import Tensor


def dynamic_sampling_filter(rewards: Tensor, epsilon: float = 1e-4) -> Tensor:
    """
    Filter out prompts where rewards have zero variance across rollouts.
    
    Prompts where the model is consistently failing or succeeding provide
    zero gradient signal and should be removed from training.
    
    Args:
        rewards: Tensor of shape (Batch_Size, Num_Rollouts) containing rewards.
        epsilon: Threshold for standard deviation. Default is 1e-4.
        
    Returns:
        A boolean mask of shape (Batch_Size,). 
        True if std > epsilon (keep this sample), False otherwise (discard).
    """
    # Calculate standard deviation across the rollout dimension (dim=1)
    std = rewards.std(dim=1)
    
    # Create boolean mask: True if std > epsilon
    mask = std > epsilon
    
    return mask


def compute_advantages(rewards: Tensor) -> Tensor:
    """
    Compute normalized advantages for GRPO training.
    
    Normalizes rewards per group using the formula:
    A_i = (r_i - mean(r)) / (std(r) + epsilon)
    
    Args:
        rewards: Tensor of shape (Batch, Rollouts) containing rewards.
        
    Returns:
        Tensor of advantages with the same shape (Batch, Rollouts).
    """
    epsilon = 1e-8
    
    # Calculate mean and std across the rollout dimension (dim=1), keeping dims for broadcasting
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    
    # Normalize: A_i = (r_i - mean) / (std + epsilon)
    advantages = (rewards - mean) / (std + epsilon)
    
    return advantages
