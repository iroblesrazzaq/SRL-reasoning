#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) Training Script for Next-Step Prediction.

This script trains a model to predict the next reasoning step given
a problem and previous steps, using teacher forcing.
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from srl_lib.data.dataset import StepDataset, DataCollator


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
        default=True,
        help="Use BF16 precision",
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
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        trust_remote_code=True,
        device_map="auto",
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
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
        "fp16": args.fp16,
        "bf16": args.bf16,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": "none",  # Disable wandb/tensorboard by default
    }
    
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
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
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

