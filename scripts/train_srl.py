#!/usr/bin/env python3
"""
SRL Training Script using TRL's GRPOTrainer.

This script implements Step-wise Reinforcement Learning (SRL) using
Group Relative Policy Optimization (GRPO) from the TRL library.

Usage:
    python scripts/train_srl.py --data_path data/srl_steps.jsonl --output_dir outputs/srl_model
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from srl_lib.rewards import compute_srl_reward
from srl_lib.grpo_utils import dynamic_sampling_filter
from srl_lib.prompts import format_srl_prompt


def load_srl_dataset(data_path: str) -> Dataset:
    """
    Load the SRL step-wise training dataset from JSONL.
    
    Args:
        data_path: Path to the JSONL file with SRL examples.
        
    Returns:
        Hugging Face Dataset with columns: prompt, expert_target
    """
    examples = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    # Convert to format expected by GRPOTrainer
    processed = []
    for ex in examples:
        # Format the prompt using step title context injection
        prompt = format_srl_prompt(
            problem=ex["problem"],
            previous_steps=ex.get("previous_steps", []),
            step_title=ex.get("step_title"),
        )
        
        # The expert target is the step body (or full teacher step for backward compat)
        expert_target = ex.get("step_body") or ex.get("teacher_step", "")
        
        processed.append({
            "prompt": prompt,
            "expert_target": expert_target,
        })
    
    return Dataset.from_list(processed)


def create_reward_function(tokenizer):
    """
    Create a reward function wrapper for GRPO training.
    
    The reward function computes sequence similarity between
    the model's completion and the expert target.
    
    Args:
        tokenizer: The tokenizer (used for decoding if needed).
        
    Returns:
        A callable that takes completions and returns rewards.
    """
    def reward_fn(completions: List[str], prompts: List[str], expert_targets: List[str]) -> List[float]:
        """
        Compute rewards for a batch of completions.
        
        Args:
            completions: List of model-generated completions.
            prompts: List of input prompts (not used, but required by TRL).
            expert_targets: List of expert target strings.
            
        Returns:
            List of reward values.
        """
        rewards = []
        for completion, target in zip(completions, expert_targets):
            reward = compute_srl_reward(completion, target)
            rewards.append(reward)
        return rewards
    
    return reward_fn


class SRLGRPOTrainer(GRPOTrainer):
    """
    Custom GRPO Trainer with dynamic sampling filter.
    
    This extends TRL's GRPOTrainer to filter out samples
    where reward variance is near zero (too hard or too easy),
    as they provide weak learning signals.
    """
    
    def __init__(self, *args, filter_epsilon: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_epsilon = filter_epsilon
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override to apply dynamic sampling filter before computing loss.
        
        This filters out prompts where the model is consistently failing
        or succeeding across all rollouts (zero gradient signal).
        """
        # Get the rewards tensor from inputs if available
        if "rewards" in inputs and inputs["rewards"] is not None:
            rewards = inputs["rewards"]
            
            # Apply dynamic sampling filter
            if rewards.dim() == 2 and rewards.size(1) > 1:
                mask = dynamic_sampling_filter(rewards, epsilon=self.filter_epsilon)
                
                # Only proceed if we have samples to keep
                if mask.any():
                    # Filter inputs based on mask
                    for key in inputs:
                        if torch.is_tensor(inputs[key]) and inputs[key].size(0) == mask.size(0):
                            inputs[key] = inputs[key][mask]
                    
                    # Log filtering statistics
                    kept = mask.sum().item()
                    total = mask.size(0)
                    if hasattr(self, "accelerator"):
                        self.accelerator.log({
                            "dynamic_sampling/kept_ratio": kept / total,
                            "dynamic_sampling/filtered_out": total - kept,
                        })
        
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


def main():
    parser = argparse.ArgumentParser(description="Train SRL model using GRPO")
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/srl_steps.jsonl",
        help="Path to SRL training data (JSONL format)",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/srl_model",
        help="Directory to save trained model",
    )
    
    # Training arguments
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Number of completions to generate per prompt (k in paper)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    
    # Dynamic sampling arguments
    parser.add_argument(
        "--filter_epsilon",
        type=float,
        default=1e-4,
        help="Epsilon threshold for dynamic sampling filter",
    )
    
    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    print(f"Loading model: {args.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
    
    print(f"Loading dataset from: {args.data_path}")
    dataset = load_srl_dataset(args.data_path)
    print(f"Loaded {len(dataset)} training examples")
    
    # Create reward function
    reward_fn = create_reward_function(tokenizer)
    
    # Configure GRPO training
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        seed=args.seed,
        # GRPO-specific settings
        num_generations=args.num_generations,  # k=8 rollouts per prompt
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,  # For diverse rollouts
        # Generation settings
        max_length=args.max_length,
    )
    
    print("Initializing GRPO Trainer...")
    
    # Initialize trainer with dynamic sampling
    trainer = SRLGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        filter_epsilon=args.filter_epsilon,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training complete!")


if __name__ == "__main__":
    main()

