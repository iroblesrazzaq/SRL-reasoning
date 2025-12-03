"""Dataset classes for SFT training."""

import json
import torch
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ..shared.prompts import build_srl_prompt, build_srl_prompt_with_target


class StepDataset(Dataset):
    """
    Dataset for SFT training of next-step prediction.
    
    Each example contains:
    - problem: The problem statement
    - previous_steps: List of previous reasoning steps
    - step_body: The target step to predict
    
    The dataset builds prompts using XML-style format and creates
    labels that mask the prompt tokens (only compute loss on step_body).
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
            - input_ids: Tokenized full sequence (prompt + step_body)
            - labels: Same as input_ids, but with -100 for prompt tokens
        """
        ex = self.data[idx]
        problem = ex["problem"]
        previous_steps = ex.get("previous_steps", [])
        
        # Get the target step body
        step_body = ex["step_body"]
        
        # Build prompt with target (for SFT training)
        if self.use_xml_prompts:
            full_text = build_srl_prompt_with_target(problem, previous_steps, step_body)
            # Extract just the prompt part (without target)
            prompt = build_srl_prompt(problem, previous_steps, include_closing_tag=False)
        else:
            # Legacy format
            from ..shared.prompts import format_srl_prompt
            prompt = format_srl_prompt(
                problem=problem,
                previous_steps=previous_steps,
                step_title=ex.get("step_title"),
            )
            full_text = prompt + step_body
        
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
        
        # Create labels: -100 for prompt tokens, actual IDs for step_body tokens
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
        # Extract input_ids and labels as lists of tensors
        input_ids_list = [ex["input_ids"] for ex in batch]
        labels_list = [ex["labels"] for ex in batch]
        
        # Find max length in batch
        max_len = max(ids.shape[0] for ids in input_ids_list)
        
        # Pad sequences manually
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        padded_input_ids = []
        padded_labels = []
        
        for input_ids, labels in zip(input_ids_list, labels_list):
            # Convert to list if tensor
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            
            # Pad to max_len
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [pad_token_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)  # -100 for padding in labels
        
        # Convert to tensors
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
