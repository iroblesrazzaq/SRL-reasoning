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
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType, get_peft_model

from src.srl.rewards import compute_srl_reward
from src.srl.grpo_utils import dynamic_sampling_filter
from src.shared.prompts import format_srl_prompt


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
        
        # The expert target is the step body: full teacher step without the step label
        target = ex["step_body"]
        
        processed.append({
            "prompt": prompt,
            "expert_target": target,
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
        A callable compatible with TRL v0.25.1's reward function API.
    """
    def reward_fn(completions: List[str], expert_target: List[str], **kwargs) -> List[float]:
        """
        Compute rewards for a batch of completions.
        
        TRL v0.25.1 calls reward functions with keyword arguments:
        - prompts: List of input prompts
        - completions: List of model-generated completions
        - completions_ids: Tokenized completion IDs
        - expert_target: List of expert targets (from dataset column)
        - trainer_state: Current trainer state
        
        Args:
            completions: List of model-generated completions.
            expert_target: List of expert target strings (from dataset).
            **kwargs: Additional TRL-provided args (prompts, completions_ids, 
                      trainer_state, etc.) - ignored but required for compatibility.
            
        Returns:
            List of reward values.
        """
        rewards = []
        for completion, target in zip(completions, expert_target):
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
    
    The filtering is applied at the advantage computation stage,
    where we have access to per-prompt reward distributions.
    """
    
    def __init__(self, *args, filter_epsilon: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_epsilon = filter_epsilon
        self._last_filter_stats = {"kept": 0, "total": 0}
    
    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Override advantage computation to apply dynamic sampling filter.
        
        This zeros out advantages for prompts where rewards have zero variance
        (model is consistently succeeding or failing), effectively removing
        their contribution to the gradient.
        
        Args:
            rewards: Tensor of shape (batch_size, num_generations) containing rewards.
            
        Returns:
            Tensor of advantages with same shape.
        """
        # Get the mask for samples with meaningful variance
        mask = dynamic_sampling_filter(rewards, epsilon=self.filter_epsilon)
        
        # Track statistics for logging
        kept = mask.sum().item()
        total = mask.size(0)
        self._last_filter_stats = {"kept": kept, "total": total}
        
        # Compute base advantages using parent method
        # GRPO normalizes rewards per-group: A_i = (r_i - mean(r)) / std(r)
        advantages = super()._compute_advantages(rewards)
        
        # Zero out advantages for filtered samples (no gradient contribution)
        # This is numerically stable and doesn't change batch sizes
        if mask.dim() == 1 and advantages.dim() == 2:
            # Expand mask to match advantages shape (batch, num_generations)
            mask_expanded = mask.unsqueeze(1).expand_as(advantages)
            advantages = advantages * mask_expanded.float()
        
        return advantages
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training step to log dynamic sampling statistics.
        """
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Log filtering statistics after each step
        if self._last_filter_stats["total"] > 0:
            kept_ratio = self._last_filter_stats["kept"] / self._last_filter_stats["total"]
            filtered_out = self._last_filter_stats["total"] - self._last_filter_stats["kept"]
            
            self.log({
                "dynamic_sampling/kept_ratio": kept_ratio,
                "dynamic_sampling/filtered_out": filtered_out,
            })
        
        return loss


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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bfloat16 precision (default: True, use --no-bf16 to disable)",
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
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        choices=["steps", "epoch", "no"],
        help="Checkpoint save strategy",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,  # Safe default: only keep last 3 checkpoints
        help="Maximum number of checkpoints to keep (older ones deleted)",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_8bit",
        help="Optimizer to use (e.g., adamw_8bit, adamw_torch)",
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
    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    # Enable input gradients (required for training)
    model.enable_input_require_grads()
    
    # Set to training mode
    model.train()
    
    # Verify setup
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model setup:")
    print(f"  Trainable params: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total params: {total_params / 1e6:.2f}M")
    
    if torch.cuda.is_available():
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    print(f"Loading dataset from: {args.data_path}")
    dataset = load_srl_dataset(args.data_path)
    print(f"Loaded {len(dataset)} training examples")
    
    # Create reward function
    reward_fn = create_reward_function(tokenizer)
    
    # Configure GRPO training
    # Note: GRPOConfig may not accept max_length or max_new_tokens
    # These are typically handled by the trainer or vLLM internally
    # Build config with only supported parameters
    grpo_config_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "optim": args.optim,
        "bf16": args.bf16,
        "seed": args.seed,
        "num_generations": args.num_generations,  # k rollouts per prompt
        "report_to": "none",  # Disable wandb/tensorboard by default for non-interactive runs
    }
    
    # Try to add max_length if supported
    import inspect
    sig = inspect.signature(GRPOConfig.__init__)
    if 'max_length' in sig.parameters:
        grpo_config_dict["max_length"] = args.max_length
    if 'max_new_tokens' in sig.parameters:
        grpo_config_dict["max_new_tokens"] = args.max_new_tokens
    
    grpo_config = GRPOConfig(**grpo_config_dict)
    
    print("Initializing GRPO Trainer...")
    print(f"  num_generations: {args.num_generations}")
    if 'max_length' in grpo_config_dict:
        print(f"  max_length: {args.max_length}")
    if 'max_new_tokens' in grpo_config_dict:
        print(f"  max_new_tokens: {args.max_new_tokens}")
    if 'max_length' not in grpo_config_dict and 'max_new_tokens' not in grpo_config_dict:
        print(f"  Note: Generation length controlled by vLLM defaults")
    
    # Initialize trainer with dynamic sampling
    # Note: GRPOTrainer may not accept tokenizer directly
    # It may extract it from the model or config
    trainer_kwargs = {
        "model": model,
        "args": grpo_config,
        "train_dataset": dataset,
        "reward_funcs": reward_fn,
        "filter_epsilon": args.filter_epsilon,  # For SRLGRPOTrainer
    }
    
    # Try to add tokenizer if supported by GRPOTrainer
    import inspect
    sig = inspect.signature(GRPOTrainer.__init__)
    if 'tokenizer' in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    
    trainer = SRLGRPOTrainer(**trainer_kwargs)
    
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
