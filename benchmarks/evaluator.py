"""Core vLLM evaluation engine for math benchmarks."""

import gc
import re
import time
from typing import List, Dict, Optional
from tqdm import tqdm

import torch
import vllm
from vllm.sampling_params import SamplingParams


# Prompt template for SRL-trained models (with think tags)
SRL_PROMPT_TEMPLATE = """You are a helpful assistant for solving mathematical problems. A user will provide a math problem, which may include a partial solution. Your task is to continue the solution by providing the very next logical step. You should first draft your thinking process (inner monologue). Then, generate the solution. Your response format must follow the template below: <think> Your thoughts... </think> Provide only the single, next step to continue the solution.

Question: {problem}
<think>"""

# Prompt template for base models (no think tags expected)
# Qwen evaluation style: append the CoT hint after the question text.
BASE_PROMPT_TEMPLATE = """Problem: {problem}

Please reason step by step, and put your final answer within \\boxed{}."""

# Backward compatibility alias
PROMPT_TEMPLATE = SRL_PROMPT_TEMPLATE


def extract_answer(text: str, strip_think: bool = True) -> Optional[str]:
    """
    Extract the final answer from model output.
    
    For SRL models, first strips content before </think> to only search
    in the action portion. This prevents mistaking intermediate numbers
    in the thought process as the final answer.
    
    Then looks for \\boxed{...} patterns (handling nested braces).
    If not found, falls back to extracting the last numerical value.
    
    Args:
        text: The model's generated text.
        strip_think: If True, only search for answers after </think> tag.
        
    Returns:
        The extracted answer string, or None if no answer found.
    """
    if not text:
        return None
    
    search_text = text
    
    # Strip content before </think> to only look at the action portion
    if strip_think:
        close_tag = "</think>"
        close_idx = text.find(close_tag)
        if close_idx != -1:
            # Only search in the action portion (after </think>)
            search_text = text[close_idx + len(close_tag):]
    
    # Look for \boxed{...} patterns in the action portion
    boxed_answers = _extract_boxed(search_text)
    if boxed_answers:
        # Return the last boxed answer (most likely the final answer)
        return boxed_answers[-1]
    
    # Fallback: find the last numerical value in action portion
    return _extract_last_number(search_text)


def _extract_boxed(text: str) -> List[str]:
    """
    Extract all \\boxed{...} contents, handling nested braces.
    
    Args:
        text: Text to search for boxed expressions.
        
    Returns:
        List of contents inside \\boxed{}.
    """
    results = []
    
    # Find all occurrences of \boxed{
    pattern = r'\\boxed\{'
    for match in re.finditer(pattern, text):
        start = match.end()
        content = _extract_balanced_braces(text, start)
        if content is not None:
            results.append(content)
    
    return results


def _extract_balanced_braces(text: str, start: int) -> Optional[str]:
    """
    Extract content within balanced braces starting at position start.
    
    Args:
        text: The full text.
        start: Position right after the opening brace.
        
    Returns:
        The content within the braces, or None if unbalanced.
    """
    depth = 1
    i = start
    
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    
    if depth == 0:
        return text[start:i-1]
    return None


def _extract_last_number(text: str) -> Optional[str]:
    """
    Extract the last numerical value from text.
    
    Handles integers, decimals, negative numbers, and fractions.
    
    Args:
        text: Text to search for numbers.
        
    Returns:
        The last number found as a string, or None.
    """
    # Pattern for numbers: integers, decimals, negatives, fractions
    pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?'
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the last number, removing commas
        return matches[-1].replace(',', '')
    return None


def is_correct(pred: Optional[str], truth: str) -> bool:
    """
    Check if predicted answer matches ground truth.
    
    Normalizes both values before comparison:
    - Strips whitespace
    - Handles float vs int (e.g., 1.0 == 1)
    - Removes commas from numbers
    
    Args:
        pred: The predicted answer (may be None).
        truth: The ground truth answer.
        
    Returns:
        True if answers match, False otherwise.
    """
    if pred is None:
        return False
    
    # Normalize both strings
    pred_norm = _normalize_answer(pred)
    truth_norm = _normalize_answer(truth)
    
    # Direct string comparison
    if pred_norm == truth_norm:
        return True
    
    # Try numerical comparison
    try:
        pred_num = float(pred_norm)
        truth_num = float(truth_norm)
        
        # Check if they're equal (handles 1.0 == 1)
        if pred_num == truth_num:
            return True
        
        # Check with small tolerance for floating point
        if abs(pred_num - truth_num) < 1e-9:
            return True
    except (ValueError, TypeError):
        pass
    
    return False


def _normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.
    
    Args:
        answer: The answer string.
        
    Returns:
        Normalized answer string.
    """
    # Strip whitespace
    answer = answer.strip()
    
    # Remove commas from numbers
    answer = answer.replace(',', '')
    
    # Remove leading/trailing whitespace from LaTeX
    answer = answer.strip('$').strip()
    
    # Try to simplify floats like "1.0" to "1"
    try:
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        pass
    
    return answer


class MathEvaluator:
    """
    Evaluator for math reasoning using vLLM.
    
    Supports greedy decoding and avg@32 (majority voting with 32 samples).
    Handles both base models and SRL-trained models with appropriate prompts.
    """
    
    def __init__(self, model_path: str, model_type: str = "srl", gpu_memory_utilization: float = 0.8, base_model: Optional[str] = None):
        """
        Initialize the evaluator with a vLLM model.
        
        Args:
            model_path: Path to the model or HuggingFace model ID.
            model_type: Either "srl" (trained with think tags) or "base" (standard model).
                       Using the wrong prompt for the model type will lower scores.
            gpu_memory_utilization: Fraction of GPU memory vLLM should reserve (default 0.8).
                       Setting below 0.9 leaves headroom for other processes.
            base_model: Optional base model ID to use for repairing corrupted config/tokenizer files.
                       If provided and loading fails, will attempt to repair the merged model.
        """
        if model_type not in ("srl", "base"):
            raise ValueError(f"model_type must be 'srl' or 'base', got '{model_type}'")
        
        # Proactively clear GPU memory before vLLM initialization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model_path = model_path
        self.model_type = model_type
        
        # Try to initialize vLLM, with repair fallback if needed
        try:
            self.llm = vllm.LLM(
                model=model_path,
                dtype="bfloat16",
                trust_remote_code=True,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        except (AttributeError, TypeError, ValueError) as e:
            # Check if error is related to config/tokenizer loading
            error_str = str(e).lower()
            if ("model_type" in error_str or "config" in error_str or "tokenizer" in error_str) and base_model:
                import warnings
                from pathlib import Path
                from transformers import AutoConfig, AutoTokenizer
                
                warnings.warn(
                    f"Failed to load model due to config/tokenizer issue. "
                    f"Attempting to repair using base model '{base_model}'...",
                    UserWarning
                )
                
                model_dir = Path(model_path)
                if model_dir.exists() and model_dir.is_dir():
                    # Repair config and tokenizer
                    print(f"Repairing config/tokenizer files in {model_dir}...")
                    base_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
                    base_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
                    base_config.save_pretrained(model_dir)
                    base_tokenizer.save_pretrained(model_dir)
                    print("✓ Repair complete. Retrying vLLM initialization...")
                    
                    # Retry initialization
                    self.llm = vllm.LLM(
                        model=model_path,
                        dtype="bfloat16",
                        trust_remote_code=True,
                        gpu_memory_utilization=gpu_memory_utilization,
                    )
                else:
                    raise
            else:
                raise
    
    def _get_prompt_template(self) -> str:
        """Get the appropriate prompt template based on model type."""
        if self.model_type == "base":
            return BASE_PROMPT_TEMPLATE
        else:
            return SRL_PROMPT_TEMPLATE
    
    def evaluate(
        self,
        data: List[Dict[str, str]],
        mode: str = 'greedy'
    ) -> float:
        """
        Evaluate the model on a dataset.
        
        Args:
            data: List of dicts with 'problem' and 'solution' keys.
            mode: Either 'greedy' or 'avg32'.
            
        Returns:
            Mean score across all problems (0.0 to 1.0).
        """
        # Build prompts using appropriate template
        prompt_template = self._get_prompt_template()
        # Use simple replacement to avoid brace conflicts with templates like \boxed{}
        prompts = [
            prompt_template.replace("{problem}", item["problem"])
            for item in data
        ]
        
        # Configure sampling parameters
        if mode == 'greedy':
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=4096,
            )
        elif mode == 'avg32':
            sampling_params = SamplingParams(
                temperature=1.0,
                top_p=0.95,
                max_tokens=4096,
                n=32,
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'greedy' or 'avg32'.")
        
        # Generate outputs
        print(f"Generating outputs for {len(prompts)} problems (model_type={self.model_type})...")
        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        elapsed = time.time() - start_time
        print(f"Generation completed in {elapsed:.2f}s")
        
        # Determine whether to strip think tags when extracting answers
        # For SRL models, we want to only look at the action portion
        # For base models, search the entire output
        strip_think = (self.model_type == "srl")
        
        # Score outputs
        scores = []
        # Wrap outputs with tqdm for a progress bar
        for i, output in enumerate(tqdm(outputs, desc="Scoring")):
            truth = data[i]['solution']
            
            if mode == 'greedy':
                # Single output - score is 0 or 1
                generated_text = output.outputs[0].text
                pred = extract_answer(generated_text, strip_think=strip_think)
                score = 1.0 if is_correct(pred, truth) else 0.0
            else:
                # Multiple outputs - calculate pass rate
                correct_count = 0
                for sample in output.outputs:
                    pred = extract_answer(sample.text, strip_think=strip_think)
                    if is_correct(pred, truth):
                        correct_count += 1
                score = correct_count / len(output.outputs)
            
            scores.append(score)
        
        # Return mean score
        return sum(scores) / len(scores) if scores else 0.0