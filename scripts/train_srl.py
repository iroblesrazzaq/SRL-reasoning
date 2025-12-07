#!/usr/bin/env python3
"""
SRL Training Script using TRL's GRPOTrainer.

This script implements Step-wise Reinforcement Learning (SRL) using
Group Relative Policy Optimization (GRPO) from the TRL library.

Matches paper settings (2510.25992v1):
- Default: 3 epochs training (~123 steps, 1/10th of paper's 30 epochs)
- Epoch-based checkpoint saving
- Best model selection based on eval_reward
- Batch size 512 (via gradient accumulation)

Usage:
    python scripts/train_srl.py --data_path data/srl_steps.jsonl --output_dir outputs/srl_model
"""

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

# Try to import GRPO - may require TRL from git
try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    # Try alternative import paths
    try:
        from trl.trainer.grpo_trainer import GRPOTrainer
        from trl.trainer.grpo_config import GRPOConfig
    except ImportError:
        try:
            from trl.trainer import GRPOTrainer
            from trl.trainer.grpo_config import GRPOConfig
        except ImportError:
            raise ImportError(
                "GRPOConfig and GRPOTrainer not found in TRL. "
                "GRPO requires TRL from git. Install with:\n"
                "  pip install git+https://github.com/huggingface/trl.git\n"
                "Or in Colab:\n"
                "  !pip install git+https://github.com/huggingface/trl.git"
            )

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
        # If filter_epsilon is None or very large, disable filtering
        if self.filter_epsilon is None or self.filter_epsilon >= 1.0:
            # Disable filtering - keep all samples
            mask = torch.ones(rewards.size(0), dtype=torch.bool, device=rewards.device)
        else:
            mask = dynamic_sampling_filter(rewards, epsilon=self.filter_epsilon)
        
        # Track statistics for logging
        kept = mask.sum().item()
        total = mask.size(0)
        self._last_filter_stats = {"kept": kept, "total": total}
        
        # Log warning if too many samples are filtered
        if kept == 0 and total > 0:
            print(f"⚠️  WARNING: All {total} samples filtered out (zero reward variance). "
                  f"Consider disabling filter (set --filter_epsilon to None or >= 1.0) "
                  f"or check if model is generating proper format.")
        
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
    parser = argparse.ArgumentParser(
        description="Train SRL model using GRPO (matches paper 2510.25992v1 settings)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/srl_steps.jsonl",
        help="Path to SRL training data (JSONL format)",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=None,
        help="Path to validation data (JSONL format). If provided, enables evaluation and best model selection.",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-4B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/srl_model",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="Attention backend (e.g., flash_attention_2, sdpa, eager). Defaults to 'sdpa' (PyTorch built-in).",
    )
    
    # Training arguments (matching paper defaults for 7B, adjusted for 4B)
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs (paper: 30, reduced to 3 for faster training ~123 steps)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device (paper: 8 for 7B/A100 80GB, adjusted for 4B)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=64,
        help="Gradient accumulation steps (paper: 64 for 7B, total batch size = 8 * 64 = 512)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-7,
        help="Learning rate (paper: 5e-7)",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Number of completions to generate per prompt (k in paper, default: 8)",
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
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping (paper: 1.0)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Warmup ratio (paper: 0.0, no warmup)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="KL divergence coefficient (paper: 0.0 for SRL)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for generation rollouts (paper: 1.0)",
    )
    
    # Dynamic sampling arguments
    parser.add_argument(
        "--filter_epsilon",
        type=float,
        default=1e-4,
        help="Epsilon threshold for dynamic sampling filter. Set to None or >= 1.0 to disable.",
    )
    
    # Checkpoint and evaluation arguments (matching paper)
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["steps", "epoch", "no"],
        help="Checkpoint save strategy (paper: 'epoch')",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (only used if save_strategy='steps')",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep (paper: 2, only last 2 kept)",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy (paper: 'epoch', requires --val_data_path)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps (only used if evaluation_strategy='steps')",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        default=True,
        help="Load best model at end based on eval_reward (paper: True)",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="eval_reward",
        help="Metric for best model selection (paper: 'eval_reward')",
    )
    parser.add_argument(
        "--greater_is_better",
        action="store_true",
        default=True,
        help="Whether greater metric value is better (paper: True for reward)",
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
        help="Use bfloat16 precision (paper: True, default: True)",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every N steps (paper: 1)",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_8bit",
        help="Optimizer to use (e.g., adamw_8bit, adamw_torch)",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Reporting backend (e.g., 'wandb', 'tensorboard', 'none')",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    print("=" * 80)
    print("SRL TRAINING WITH GRPO")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.per_device_train_batch_size} * {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Num generations (k): {args.num_generations}")
    print("=" * 80)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",  # Left padding for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    
    # Apply LoRA
    print("Applying LoRA...")
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
    
    # Load datasets
    print(f"\nLoading training dataset: {args.data_path}")
    train_dataset = load_srl_dataset(args.data_path)
    print(f"  Loaded {len(train_dataset)} training examples")
    
    val_dataset = None
    if args.val_data_path and Path(args.val_data_path).exists():
        print(f"Loading validation dataset: {args.val_data_path}")
        val_dataset = load_srl_dataset(args.val_data_path)
        print(f"  Loaded {len(val_dataset)} validation examples")
    elif args.val_data_path:
        print(f"⚠️  Warning: Validation data path '{args.val_data_path}' not found. Disabling evaluation.")
        args.evaluation_strategy = "no"
        args.load_best_model_at_end = False
    else:
        print("  No validation dataset provided. Disabling evaluation.")
        args.evaluation_strategy = "no"
        args.load_best_model_at_end = False
    
    # Create reward function
    reward_fn = create_reward_function(tokenizer)
    
    # Configure GRPO training
    # Build config dict with paper-matching defaults
    grpo_config_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "optim": args.optim,
        "bf16": args.bf16,
        "seed": args.seed,
        "num_generations": args.num_generations,
        "report_to": args.report_to,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "beta": args.beta,
        "temperature": args.temperature,
    }
    
    # Add save_steps if using step-based saving
    if args.save_strategy == "steps":
        grpo_config_dict["save_steps"] = args.save_steps
    
    # Add evaluation settings if validation dataset provided
    if val_dataset is not None:
        grpo_config_dict["evaluation_strategy"] = args.evaluation_strategy
        if args.evaluation_strategy == "steps":
            grpo_config_dict["eval_steps"] = args.eval_steps
        
        if args.load_best_model_at_end:
            grpo_config_dict["load_best_model_at_end"] = True
            grpo_config_dict["metric_for_best_model"] = args.metric_for_best_model
            grpo_config_dict["greater_is_better"] = args.greater_is_better
    
    # Try to add max_length/max_new_tokens if supported by GRPOConfig
    sig = inspect.signature(GRPOConfig.__init__)
    if 'max_length' in sig.parameters:
        grpo_config_dict["max_length"] = args.max_length
    if 'max_new_tokens' in sig.parameters:
        grpo_config_dict["max_new_tokens"] = args.max_new_tokens
    
    # Create GRPOConfig
    grpo_config = GRPOConfig(**grpo_config_dict)
    
    print("\n" + "=" * 80)
    print("GRPO CONFIGURATION")
    print("=" * 80)
    print(f"  num_generations: {args.num_generations}")
    print(f"  save_strategy: {args.save_strategy}")
    if args.save_strategy == "steps":
        print(f"  save_steps: {args.save_steps}")
    print(f"  save_total_limit: {args.save_total_limit}")
    if val_dataset:
        print(f"  evaluation_strategy: {args.evaluation_strategy}")
        if args.load_best_model_at_end:
            print(f"  load_best_model_at_end: True")
            print(f"  metric_for_best_model: {args.metric_for_best_model}")
    if 'max_length' in grpo_config_dict:
        print(f"  max_length: {args.max_length}")
    if 'max_new_tokens' in grpo_config_dict:
        print(f"  max_new_tokens: {args.max_new_tokens}")
    print("=" * 80)
    
    # Initialize trainer
    filter_epsilon = args.filter_epsilon if args.filter_epsilon is not None else None
    
    trainer_kwargs = {
        "model": model,
        "args": grpo_config,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "reward_funcs": reward_fn,
        "filter_epsilon": filter_epsilon,
    }
    
    # Try to add tokenizer if supported by GRPOTrainer
    sig = inspect.signature(GRPOTrainer.__init__)
    if 'tokenizer' in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    
    trainer = SRLGRPOTrainer(**trainer_kwargs)
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    print(f"Saving to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    if args.load_best_model_at_end and val_dataset:
        print(f"Best model (based on {args.metric_for_best_model}) has been loaded.")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
