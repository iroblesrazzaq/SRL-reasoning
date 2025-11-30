"""Dataset classes for SFT training."""

import json
import torch
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ..prompts import build_prompt, build_prompt_with_target


class StepDataset(Dataset):
    """
    Dataset for SFT training of next-step prediction.
    
    Each example contains:
    - problem: The problem statement
    - previous_steps: List of previous reasoning steps
    - teacher_step: The target step to predict
    
    The dataset builds prompts using XML-style format and creates
    labels that mask the prompt tokens (only compute loss on teacher_step).
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        use_xml_prompts: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to JSONL file with SRL examples
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            use_xml_prompts: If True, use XML-style prompts (build_prompt),
                           else use legacy format_srl_prompt
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_xml_prompts = use_xml_prompts
        
        # Load examples
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training example.
        
        Returns:
            Dict with keys:
            - input_ids: Tokenized full sequence (prompt + teacher_step)
            - labels: Same as input_ids, but with -100 for prompt tokens
        """
        ex = self.data[idx]
        problem = ex["problem"]
        previous_steps = ex.get("previous_steps", [])
        
        # Get teacher step (prefer step_body if available, fallback to teacher_step)
        teacher_step = ex.get("step_body") or ex.get("teacher_step", "")
        
        # Build prompt with target (for SFT training)
        if self.use_xml_prompts:
            full_text = build_prompt_with_target(problem, previous_steps, teacher_step)
            # Extract just the prompt part (without target)
            prompt = build_prompt(problem, previous_steps, include_closing_tag=False)
        else:
            # Legacy format (for backward compatibility)
            from ..prompts import format_srl_prompt
            prompt = format_srl_prompt(
                problem=problem,
                previous_steps=previous_steps,
                step_title=ex.get("step_title"),
            )
            full_text = prompt + teacher_step
        
        # Tokenize full sequence
        enc_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc_full.input_ids[0]  # [L]
        
        # Tokenize prompt to find boundary
        enc_prompt = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = enc_prompt.input_ids.shape[-1]
        
        # Create labels: -100 for prompt tokens, actual IDs for teacher_step tokens
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Ignore loss on prompt
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class DataCollator:
    """
    Data collator for batching SFT training examples.
    
    Pads sequences to the same length within a batch.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize the collator.
        
        Args:
            tokenizer: Tokenizer for padding
        """
        self.tokenizer = tokenizer
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Args:
            batch: List of example dicts with input_ids and labels
            
        Returns:
            Batched dict with padded input_ids and labels
        """
        input_ids = [ex["input_ids"] for ex in batch]
        labels = [ex["labels"] for ex in batch]
        
        # Pad sequences
        padded = self.tokenizer.pad(
            {"input_ids": input_ids, "labels": labels},
            return_tensors="pt",
            padding=True,
        )
        
        return padded

