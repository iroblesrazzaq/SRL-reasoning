#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) Training Script for Next-Step Prediction.

This script trains a model to predict the next reasoning step given
a problem and previous steps, using teacher forcing.
"""

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model

from src.sft.dataset import StepDataset, DataCollator


def main():
    parser = argparse.ArgumentParser(description="Train SFT model for next-step prediction")
    
    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/train.jsonl",
        help="Path to training data (JSONL)",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/val.jsonl",
        help="Path to validation data (JSONL)",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="Attention backend (e.g., flash_attention_2, sdpa, eager). Defaults to 'sdpa' (PyTorch built-in) to avoid flash-attn dependency.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/sft_model",
        help="Directory to save trained model",
    )
    
    # Training arguments
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Eval batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_8bit",
        choices=["adamw_torch", "adamw_8bit", "adamw_bnb_8bit"],
        help="Optimizer to use. Use adamw_8bit or adamw_bnb_8bit for 8-bit optimizer (requires bitsandbytes)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 precision (default if neither fp16 nor bf16 specified)",
    )
    
    # Logging/saving
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use_xml_prompts",
        action="store_true",
        default=True,
        help="Use XML-style prompts (build_prompt)",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    print(f"Loading model: {args.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    # Don't use device_map="auto" with Trainer - let Trainer handle device placement
    # This ensures proper GPU usage during training
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    
    # Move model to GPU if available (Trainer will handle this, but doing it explicitly ensures it's on GPU)
    if torch.cuda.is_available():
        model = model.to("cuda")
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
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Verify setup
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model setup:")
    print(f"  Trainable params: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total params: {total_params / 1e6:.2f}M")
    
    if torch.cuda.is_available():
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    print(f"Loading training dataset from: {args.train_data}")
    train_dataset = StepDataset(
        args.train_data,
        tokenizer,
        max_length=args.max_length,
        use_xml_prompts=args.use_xml_prompts,
    )
    
    val_dataset = None
    if args.val_data and Path(args.val_data).exists():
        print(f"Loading validation dataset from: {args.val_data}")
        val_dataset = StepDataset(
            args.val_data,
            tokenizer,
            max_length=args.max_length,
            use_xml_prompts=args.use_xml_prompts,
        )
    
    # Data collator
    collator = DataCollator(tokenizer)
    
    # Training arguments
    # Build args dict to support both eval_strategy and evaluation_strategy
    # (different transformers versions use different parameter names)
    import inspect
    
    training_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": args.optim,  # Support 8-bit optimizer
        "report_to": "none",  # Disable wandb/tensorboard by default
    }
    
    # Handle fp16/bf16: only one can be True
    # If fp16 is specified, use it; otherwise default to bf16
    # Note: FP16 can have issues with gradient clipping, so we disable it
    if args.fp16:
        training_args_dict["fp16"] = True
        training_args_dict["bf16"] = False
        # Disable gradient clipping for FP16 to avoid "Attempting to unscale FP16 gradients" error
        training_args_dict["max_grad_norm"] = None
    else:
        # Default to bf16 if neither is explicitly set, or use bf16 if specified
        training_args_dict["fp16"] = False
        training_args_dict["bf16"] = args.bf16 if args.bf16 else True  # Default to True if not specified
        # bf16 works fine with gradient clipping, so use default (1.0)
        if "max_grad_norm" not in training_args_dict:
            training_args_dict["max_grad_norm"] = 1.0
    
    # Add eval-related args if validation dataset exists
    if val_dataset:
        training_args_dict["eval_steps"] = args.eval_steps
        # Check which parameter name is supported by inspecting TrainingArguments signature
        sig = inspect.signature(TrainingArguments.__init__)
        if "eval_strategy" in sig.parameters:
            training_args_dict["eval_strategy"] = "steps"
        elif "evaluation_strategy" in sig.parameters:
            training_args_dict["evaluation_strategy"] = "steps"
        else:
            # Fallback: try eval_strategy first (more common in recent versions)
            training_args_dict["eval_strategy"] = "steps"
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Custom compute_metrics to handle NaN gracefully
    def compute_metrics(eval_pred):
        """Compute metrics for evaluation, handling NaN cases."""
        predictions, labels = eval_pred
        # If loss is NaN, return a default value
        # The actual loss is computed by the Trainer internally
        return {}
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics if val_dataset else None,
    )
    
    print("Starting SFT training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("SFT training complete!")


if __name__ == "__main__":
    main()
