# Colab Test Cells for SFT and GRPO

## Setup Cell (Run First)

```python
# Install dependencies
!pip install -q bitsandbytes accelerate peft transformers trl datasets flash-attn --no-build-isolation

# Clone repo (replace with your repo URL)
!git clone https://github.com/yourusername/SRL-reasoning.git
%cd SRL-reasoning

# Install package
!pip install -e .

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

## Data Preparation Cell

```python
# Create train/val/test splits from existing data
# Skip this if you already have splits
import os
from pathlib import Path

# Check if splits exist
if not Path("data/train.jsonl").exists():
    print("Creating train/val/test splits...")
    !python scripts/create_splits.py \
        --data_path data/srl_steps.jsonl \
        --output_dir data \
        --train_ratio 0.8 \
        --val_ratio 0.1 \
        --test_ratio 0.1 \
        --seed 42
else:
    print("Splits already exist, skipping...")
```

## Test SFT Training (Small Test Run)

```python
# Quick SFT test with minimal epochs and small batch
!python scripts/train_sft.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --output_dir outputs/sft_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --max_length 1024 \
    --optim adamw_8bit \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 2 \
    --seed 42
```

## Test GRPO Training (Small Test Run)

```python
# Quick GRPO test with minimal settings
!python scripts/train_srl.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --data_path data/train.jsonl \
    --output_dir outputs/srl_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_generations 4 \
    --max_new_tokens 256 \
    --max_length 1024 \
    --learning_rate 1e-6 \
    --optim adamw_8bit \
    --bf16 \
    --logging_steps 5 \
    --save_steps 50 \
    --save_total_limit 2 \
    --filter_epsilon 1e-4 \
    --seed 42
```

## Full SFT Training (Production)

```python
# Full SFT training with recommended settings
!python scripts/train_sft.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --output_dir outputs/sft_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --max_length 2048 \
    --optim adamw_8bit \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --save_total_limit 3 \
    --seed 42
```

## Full GRPO Training (Production)

```python
# Full GRPO training with recommended settings
!python scripts/train_srl.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --data_path data/train.jsonl \
    --output_dir outputs/srl_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_generations 8 \
    --max_new_tokens 512 \
    --max_length 2048 \
    --learning_rate 1e-6 \
    --optim adamw_8bit \
    --bf16 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --filter_epsilon 1e-4 \
    --seed 42
```

## Notes

- **Both scripts use LoRA** (PEFT) by default, which reduces memory usage significantly
- **8-bit optimizer** (`adamw_8bit`) is default for both scripts
- **Gradient checkpointing** is enabled by default for SFT
- **Memory optimizations**: Use `--max_length 1024` or `512` if you run out of memory
- **Batch size**: Keep `per_device_train_batch_size=1` and increase `gradient_accumulation_steps` instead
- **Test runs**: Use `num_train_epochs=1` and smaller `save_steps` for quick verification
- **Production runs**: Use full settings above for actual training

## Troubleshooting

If you get OOM errors:
1. Reduce `--max_length` to 512 or 1024
2. Reduce `--num_generations` for GRPO (try 4 instead of 8)
3. Increase `--gradient_accumulation_steps` to maintain effective batch size
4. Use a smaller model: `Qwen/Qwen2.5-1.5B-Instruct`

