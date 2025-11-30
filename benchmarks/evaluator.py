"""Core vLLM evaluation engine for math benchmarks."""

import re
import time
from typing import List, Dict, Optional

import vllm
from vllm import SamplingParams


# Prompt template for reasoning
PROMPT_TEMPLATE = """You are a helpful assistant for solving mathematical problems. A user will provide a math problem, which may include a partial solution. Your task is to continue the solution by providing the very next logical step. You should first draft your thinking process (inner monologue). Then, generate the solution. Your response format must follow the template below: <think> Your thoughts... </think> Provide only the single, next step to continue the solution.

Question: {problem}
<think>"""


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from model output.
    
    First looks for \\boxed{...} patterns (handling nested braces).
    If not found, falls back to extracting the last numerical value.
    
    Args:
        text: The model's generated text.
        
    Returns:
        The extracted answer string, or None if no answer found.
    """
    if not text:
        return None
    
    # Look for \boxed{...} patterns
    boxed_answers = _extract_boxed(text)
    if boxed_answers:
        # Return the last boxed answer (most likely the final answer)
        return boxed_answers[-1]
    
    # Fallback: find the last numerical value
    return _extract_last_number(text)


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
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the evaluator with a vLLM model.
        
        Args:
            model_path: Path to the model or HuggingFace model ID.
        """
        self.model_path = model_path
        self.llm = vllm.LLM(
            model=model_path,
            dtype="bfloat16",
            trust_remote_code=True,
        )
    
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
        # Build prompts
        prompts = [
            PROMPT_TEMPLATE.format(problem=item['problem'])
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
        print(f"Generating outputs for {len(prompts)} problems...")
        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        elapsed = time.time() - start_time
        print(f"Generation completed in {elapsed:.2f}s")
        
        # Score outputs
        scores = []
        for i, output in enumerate(outputs):
            truth = data[i]['solution']
            
            if mode == 'greedy':
                # Single output - score is 0 or 1
                generated_text = output.outputs[0].text
                pred = extract_answer(generated_text)
                score = 1.0 if is_correct(pred, truth) else 0.0
            else:
                # Multiple outputs - calculate pass rate
                correct_count = 0
                for sample in output.outputs:
                    pred = extract_answer(sample.text)
                    if is_correct(pred, truth):
                        correct_count += 1
                score = correct_count / len(output.outputs)
            
            scores.append(score)
        
        # Return mean score
        return sum(scores) / len(scores) if scores else 0.0

