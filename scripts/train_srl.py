#!/usr/bin/env python3
"""
SRL Training Script using TRL's GRPOTrainer with vLLM optimization.

This script implements Step-wise Reinforcement Learning (SRL) using
Group Relative Policy Optimization (GRPO) from the TRL library.

Uses vLLM for 6-10x faster generation (the 90%+ bottleneck in GRPO training).
Falls back to standard transformers if vLLM is unavailable.

Matches paper settings (2510.25992v1):
- Default: 3 epochs training (~123 steps, 1/10th of paper's 30 epochs)
- Epoch-based checkpoint saving
- Best model selection based on eval_reward
- Batch size 128 (via gradient accumulation)

Usage:
    python scripts/train_srl.py --data_path data/srl_steps.jsonl --output_dir outputs/srl_model
"""

import argparse
import gc
import inspect
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import torch

# Enable expandable segments to reduce CUDA memory fragmentation
# This helps when you have enough total memory but allocation fails due to fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datasets import Dataset

# Use standard transformers + PEFT (vLLM handles generation speedup)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

# Check vLLM availability
try:
    import vllm
    VLLM_AVAILABLE = True
    print("✓ vLLM available - using optimized generation")
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM not available - falling back to HuggingFace generation")

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
            print(f"WARNING: All {total} samples filtered out (zero reward variance). "
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
        # Clear cache before step to maximize available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Clear CUDA cache after each step to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Log filtering statistics after each step
        if self._last_filter_stats["total"] > 0:
            kept_ratio = self._last_filter_stats["kept"] / self._last_filter_stats["total"]
            filtered_out = self._last_filter_stats["total"] - self._last_filter_stats["kept"]
            
            self.log({
                "dynamic_sampling/kept_ratio": kept_ratio,
                "dynamic_sampling/filtered_out": filtered_out,
            })
        
        return loss


def load_model(model_name: str, bf16: bool, attn_implementation: str, use_8bit: bool):
    """
    Load model using standard transformers + PEFT.
    
    vLLM will handle fast generation during GRPO training.
    
    Args:
        model_name: HuggingFace model name or path
        bf16: Whether to use bfloat16
        attn_implementation: Attention implementation to use
        use_8bit: Whether to use 8-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import BitsAndBytesConfig
    
    print(f"Loading model with transformers + PEFT: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if bf16 else torch.float32,
        "trust_remote_code": True,
        "device_map": "auto",
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": True,
    }
    
    if use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model_kwargs["quantization_config"] = quantization_config
        print("Using 8-bit quantization (may conflict with LoRA)")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
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
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train SRL model using GRPO with vLLM optimization",
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
        default="Qwen/Qwen3-4B-Instruct-2507",
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
        help="Attention backend for non-Unsloth mode (e.g., flash_attention_2, sdpa, eager)",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=False,
        help="Use 4-bit quantization (optional, not needed for A100 80GB)",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        default=False,
        help="Use 8-bit quantization (fallback mode only, may conflict with LoRA)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length for Unsloth model (prompt + completion)",
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
        default=4,
        help="Batch size per device (increased with vLLM efficiency)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Gradient accumulation steps (effective batch size = 4 * 32 = 128)",
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
        default=4,
        help="Number of completions to generate per prompt (k in paper, default: 4, paper: 8)",
    )
    
    # vLLM arguments
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        default=True,
        help="Use vLLM for fast generation (6-10x speedup). Disable with --no-use_vllm",
    )
    parser.add_argument(
        "--no-use_vllm",
        action="store_false",
        dest="use_vllm",
        help="Disable vLLM and use HuggingFace generation",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.5,
        help="GPU memory fraction for vLLM (0.5 = 50%%, leaves room for training model)",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=512,
        help="Maximum prompt length (paper: 2048, reduced for memory)",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=256,
        help="Maximum completion length (paper: 512, reduced for memory)",
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
    
    # Determine if vLLM will be used
    use_vllm = args.use_vllm and VLLM_AVAILABLE
    if args.use_vllm and not VLLM_AVAILABLE:
        print("⚠️  vLLM requested but not available, falling back to HF generation")
    
    print("=" * 80)
    print("SRL TRAINING WITH GRPO" + (" (vLLM)" if use_vllm else " (HuggingFace)"))
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.per_device_train_batch_size} * {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Num generations (k): {args.num_generations}")
    print(f"use_vllm: {use_vllm}")
    if use_vllm:
        print(f"vllm_gpu_memory_utilization: {args.vllm_gpu_memory_utilization}")
    print("=" * 80)
    
    # Load model and tokenizer
    model, tokenizer = load_model(
        model_name=args.model_name,
        bf16=args.bf16,
        attn_implementation=args.attn_implementation,
        use_8bit=args.use_8bit,
    )
    
    # Verify setup
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n Model setup:")
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
        print(f"Warning: Validation data path '{args.val_data_path}' not found. Disabling evaluation.")
        args.evaluation_strategy = "no"
        args.load_best_model_at_end = False
    else:
        print("  No validation dataset provided. Disabling evaluation.")
        args.evaluation_strategy = "no"
        args.load_best_model_at_end = False
    
    # Create reward function
    reward_fn = create_reward_function(tokenizer)
    
    # Configure GRPO training
    # Check which parameter names are supported
    sig = inspect.signature(GRPOConfig.__init__)
    supported_params = set(sig.parameters.keys())
    
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
        # vLLM settings for fast generation
        "use_vllm": use_vllm,
    }
    
    # Add vLLM memory utilization if using vLLM
    if use_vllm and "vllm_gpu_memory_utilization" in supported_params:
        grpo_config_dict["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    
    # Add token length limits if supported
    if "max_prompt_length" in supported_params:
        grpo_config_dict["max_prompt_length"] = args.max_prompt_length
    if "max_completion_length" in supported_params:
        grpo_config_dict["max_completion_length"] = args.max_completion_length
    
    # Add save_steps if using step-based saving
    if args.save_strategy == "steps":
        grpo_config_dict["save_steps"] = args.save_steps
    
    # Add evaluation settings if validation dataset provided
    if val_dataset is not None:
        # Handle eval_strategy vs evaluation_strategy naming
        if "eval_strategy" in supported_params:
            grpo_config_dict["eval_strategy"] = args.evaluation_strategy
        else:
            grpo_config_dict["evaluation_strategy"] = args.evaluation_strategy
        
        if args.evaluation_strategy == "steps":
            grpo_config_dict["eval_steps"] = args.eval_steps
        
        if args.load_best_model_at_end:
            grpo_config_dict["load_best_model_at_end"] = True
            grpo_config_dict["metric_for_best_model"] = args.metric_for_best_model
            grpo_config_dict["greater_is_better"] = args.greater_is_better
    
    # Filter to only supported parameters
    grpo_config_dict = {k: v for k, v in grpo_config_dict.items() if k in supported_params}
    
    # Create GRPOConfig
    grpo_config = GRPOConfig(**grpo_config_dict)
    
    print("\n" + "=" * 80)
    print("GRPO CONFIGURATION")
    print("=" * 80)
    print(f"  num_generations: {args.num_generations}")
    print(f"  max_prompt_length: {args.max_prompt_length}")
    print(f"  max_completion_length: {args.max_completion_length}")
    print(f"  use_vllm: {use_vllm}")
    if use_vllm:
        print(f"  vllm_gpu_memory_utilization: {args.vllm_gpu_memory_utilization}")
    print(f"  save_strategy: {args.save_strategy}")
    if args.save_strategy == "steps":
        print(f"  save_steps: {args.save_steps}")
    print(f"  save_total_limit: {args.save_total_limit}")
    if val_dataset:
        print(f"  evaluation_strategy: {args.evaluation_strategy}")
        if args.load_best_model_at_end:
            print(f"  load_best_model_at_end: True")
            print(f"  metric_for_best_model: {args.metric_for_best_model}")
    print("=" * 80)
    
    # Clear memory before trainer initialization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
    trainer_sig = inspect.signature(GRPOTrainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    
    trainer = SRLGRPOTrainer(**trainer_kwargs)
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"GPU Memory before training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    if use_vllm:
        print("vLLM will handle fast generation (6-10x speedup)")
    
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    print(f"Saving to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    if args.load_best_model_at_end and val_dataset:
        print(f"Best model (based on {args.metric_for_best_model}) has been loaded.")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()