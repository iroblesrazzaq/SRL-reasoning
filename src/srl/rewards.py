"""Reward computation for SRL training."""

import difflib
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..shared.formatting import parse_model_output


def compute_string_similarity(pred_action: str, expert_target: str) -> float:
    """
    Compute string similarity between predicted action and expert target.
    
    Uses the formula: R = (2 * M) / T
    where:
        M = sum of lengths of all matching blocks
        T = total length of both strings
    
    Args:
        pred_action: The predicted action string.
        expert_target: The expected action from the expert trajectory.
        
    Returns:
        Similarity score between 0.0 and 1.0.
    """
    total_length = len(pred_action) + len(expert_target)
    
    if total_length == 0:
        return 0.0
    
    matcher = difflib.SequenceMatcher(None, pred_action, expert_target)
    matching_blocks = matcher.get_matching_blocks()
    matching_length = sum(match.size for match in matching_blocks)
    
    return (2.0 * matching_length) / total_length


def compute_srl_reward(model_completion: str, expert_target: str) -> float:
    """
    Compute the SRL reward based on similarity between model action and expert target.
    
    Uses the formula: R = (2 * M) / T
    where:
        M = sum of lengths of all matching blocks
        T = total length of both strings (len(pred_action) + len(expert_target))
    
    Args:
        model_completion: The full model output including <think>...</think> tags.
        expert_target: The expected action from the expert trajectory.
        
    Returns:
        A reward value:
        - -1.0 if the model output has format errors (missing </think>)
        - 0.0 to 1.0 based on similarity between predicted action and expert target
    """
    # Parse the model output to extract the action
    thought, pred_action = parse_model_output(model_completion)
    
    # Penalty for format errors
    if pred_action is None:
        return -1.0
    
    return compute_string_similarity(pred_action, expert_target)


class EmbeddingRewardModel:
    """
    Manages an embedding model for computing cosine similarity rewards.
    
    Loads the model once and reuses it for all reward computations.
    Supports batch processing for efficiency.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        """
        Initialize the embedding reward model.
        
        Args:
            model_name: HuggingFace model name for embeddings.
            device: Device to load model on. Defaults to CUDA if available.
            use_fp16: Whether to use half precision (fp16) for memory efficiency.
        """
        from transformers import AutoModel, AutoTokenizer
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        
        print(f"Loading embedding model: {model_name}")
        print(f"  Device: {self.device}, FP16: {self.use_fp16}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        self.model.to(self.device)
        if self.use_fp16:
            self.model.half()
        self.model.eval()
        
        # Get embedding dimension from model config
        self.embed_dim = getattr(self.model.config, "hidden_size", 1024)
        print(f"  Embedding dimension: {self.embed_dim}")
        print(f"  ✓ Embedding model loaded")
    
    def _encode_batch(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        Encode a batch of texts into embeddings.
        
        Args:
            texts: List of text strings to encode.
            max_length: Maximum token length for encoding.
            
        Returns:
            Tensor of shape (batch_size, embed_dim) containing embeddings.
        """
        if not texts:
            return torch.empty(0, self.embed_dim, device=self.device)
        
        # Tokenize all texts in batch
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Handle different model output formats
            if hasattr(outputs, "last_hidden_state"):
                # Standard transformer output - use mean pooling
                attention_mask = inputs["attention_mask"]
                hidden_states = outputs.last_hidden_state
                
                # Mean pooling with attention mask
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                # Use pooler output if available
                embeddings = outputs.pooler_output
            else:
                # Fallback: use [CLS] token or first token
                embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def compute_cosine_batch(
        self,
        texts_a: List[str],
        texts_b: List[str],
    ) -> List[float]:
        """
        Compute cosine similarity for paired texts in batch.
        
        Args:
            texts_a: First list of texts (e.g., predicted actions).
            texts_b: Second list of texts (e.g., expert targets).
            
        Returns:
            List of cosine similarity scores (0.0 to 1.0) for each pair.
        """
        if len(texts_a) != len(texts_b):
            raise ValueError(f"Mismatched batch sizes: {len(texts_a)} vs {len(texts_b)}")
        
        if not texts_a:
            return []
        
        # Encode both batches
        emb_a = self._encode_batch(texts_a)
        emb_b = self._encode_batch(texts_b)
        
        # Compute cosine similarity for each pair
        # Since embeddings are normalized, cosine_sim = dot product
        similarities = F.cosine_similarity(emb_a, emb_b, dim=1)
        
        # Convert to list and ensure values are in [0, 1] range
        # (cosine similarity is in [-1, 1], but for text it's typically positive)
        similarities = similarities.clamp(min=0.0, max=1.0)
        
        return similarities.tolist()


def compute_combined_reward_batch(
    completions: List[str],
    expert_targets: List[str],
    embedding_model: Optional[EmbeddingRewardModel] = None,
    string_weight: float = 1.0,
    cosine_weight: float = 0.0,
) -> Tuple[List[float], dict]:
    """
    Compute combined rewards for a batch of completions.
    
    Combines string similarity and cosine similarity with configurable weights.
    
    Args:
        completions: List of model-generated completions.
        expert_targets: List of expert target strings.
        embedding_model: EmbeddingRewardModel instance (required if cosine_weight > 0).
        string_weight: Weight for string similarity reward (default: 1.0).
        cosine_weight: Weight for cosine similarity reward (default: 0.0).
        
    Returns:
        Tuple of (rewards_list, stats_dict) where:
        - rewards_list: List of combined reward values
        - stats_dict: Dictionary with statistics (format_errors, avg_string_sim, avg_cosine_sim)
    """
    if cosine_weight > 0 and embedding_model is None:
        raise ValueError("embedding_model required when cosine_weight > 0")
    
    # Normalize weights to sum to 1.0
    total_weight = string_weight + cosine_weight
    if total_weight <= 0:
        raise ValueError("string_weight + cosine_weight must be > 0")
    
    string_weight = string_weight / total_weight
    cosine_weight = cosine_weight / total_weight
    
    batch_size = len(completions)
    rewards = []
    
    # Track statistics
    format_errors = 0
    string_sims = []
    cosine_sims = []
    
    # Parse all completions and extract actions
    parsed_actions = []
    valid_indices = []
    valid_actions = []
    valid_targets = []
    
    for i, (completion, target) in enumerate(zip(completions, expert_targets)):
        thought, pred_action = parse_model_output(completion)
        parsed_actions.append(pred_action)
        
        if pred_action is not None:
            valid_indices.append(i)
            valid_actions.append(pred_action)
            valid_targets.append(target)
        else:
            format_errors += 1
    
    # Compute string similarities for valid actions
    string_sim_results = {}
    if string_weight > 0:
        for idx, (action, target) in zip(valid_indices, zip(valid_actions, valid_targets)):
            sim = compute_string_similarity(action, target)
            string_sim_results[idx] = sim
            string_sims.append(sim)
    
    # Compute cosine similarities in batch for valid actions
    cosine_sim_results = {}
    if cosine_weight > 0 and valid_actions:
        cosine_scores = embedding_model.compute_cosine_batch(valid_actions, valid_targets)
        for idx, score in zip(valid_indices, cosine_scores):
            cosine_sim_results[idx] = score
            cosine_sims.append(score)
    
    # Combine rewards
    for i in range(batch_size):
        if parsed_actions[i] is None:
            # Format error - apply penalty
            rewards.append(-1.0)
        else:
            reward = 0.0
            if string_weight > 0:
                reward += string_weight * string_sim_results.get(i, 0.0)
            if cosine_weight > 0:
                reward += cosine_weight * cosine_sim_results.get(i, 0.0)
            rewards.append(reward)
    
    # Compute statistics
    stats = {
        "format_errors": format_errors,
        "avg_string_sim": sum(string_sims) / len(string_sims) if string_sims else 0.0,
        "avg_cosine_sim": sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0,
    }
    
    return rewards, stats
