"""Generation utilities for SRL training and inference.

Provides generation wrapper for RL rollouts and standalone inference
with proper stop token handling.
"""

from typing import List, Dict, Optional, Tuple
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from .prompts import build_srl_prompt, STOP_TOKENS, extract_next_step_from_output


def generate_student_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    problem: str,
    previous_steps: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop_tokens: Optional[List[str]] = None,
) -> Tuple[str, Dict]:
    """
    Generate a single reasoning step from the student model.
    
    This is used for RL rollouts and evaluation. The model generates
    the next step given the problem and previous reasoning steps.
    
    Args:
        model: The language model (transformers PreTrainedModel)
        tokenizer: The tokenizer (transformers PreTrainedTokenizer)
        problem: The problem statement
        previous_steps: List of previous reasoning steps (can be empty)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling parameter
        stop_tokens: Custom stop tokens (defaults to STOP_TOKENS from prompts)
        
    Returns:
        A tuple of:
        - generated_step: The extracted step text (without XML tags)
        - metadata: Dict with 'input_ids', 'output_ids', 'full_output' for logging/debugging
        
    Example:
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> 
        >>> step, meta = generate_student_step(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     problem="Solve 2+2",
        ...     previous_steps=[],
        ...     max_new_tokens=64
        ... )
        >>> print(step)  # "First, I need to add 2 and 2"
    """
    if stop_tokens is None:
        stop_tokens = STOP_TOKENS
    
    # Build the prompt matching the paper wording
    prompt = build_srl_prompt(problem, previous_steps, include_closing_tag=False)
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=None,  # We'll handle stopping manually
        )[0]
    
    # Decode the full output (prompt + generation)
    full_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    
    # Extract just the generated step (after </think>)
    generated_step = extract_next_step_from_output(full_output)
    
    # Also apply manual stop token filtering as backup
    for stop_token in stop_tokens:
        if stop_token in generated_step:
            generated_step = generated_step.split(stop_token)[0].strip()
            break
    
    metadata = {
        "input_ids": input_ids,
        "output_ids": output_ids,
        "full_output": full_output,
        "prompt": prompt,
    }
    
    return generated_step, metadata


def compute_token_logprobs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    output_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probabilities of generated tokens for RL training.
    
    Given input_ids (prompt) and output_ids (prompt + generation),
    computes the log probability of each generated token.
    
    Args:
        model: The language model
        input_ids: Token IDs for the prompt [1, L_in]
        output_ids: Token IDs for prompt + generation [1, L_in + L_gen]
        
    Returns:
        Token log probabilities for generated tokens [1, L_gen]
        
    Example:
        >>> prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        >>> full_ids = model.generate(prompt_ids, max_new_tokens=64)[0]
        >>> logprobs = compute_token_logprobs(model, prompt_ids, full_ids.unsqueeze(0))
        >>> # logprobs shape: [1, 64] - one logprob per generated token
    """
    with torch.no_grad():
        # Get logits for all positions except the last (we predict next token)
        logits = model(output_ids[:, :-1]).logits  # [1, L-1, V]
    
    # Identify where generation starts
    gen_start = input_ids.shape[-1]
    
    # Extract logits for generated tokens
    # We need logits at positions gen_start-1 to gen_start+L_gen-1
    # (because we predict token at position i using logits at position i-1)
    gen_logits = logits[:, gen_start-1:]  # [1, L_gen, V]
    
    # Extract the actual generated token IDs
    gen_token_ids = output_ids[:, gen_start:]  # [1, L_gen]
    
    # Compute log probabilities
    log_probs = torch.log_softmax(gen_logits, dim=-1)  # [1, L_gen, V]
    
    # Gather logprobs for the actual tokens that were generated
    token_log_probs = log_probs.gather(
        -1,
        gen_token_ids.unsqueeze(-1)  # [1, L_gen, 1]
    ).squeeze(-1)  # [1, L_gen]
    
    return token_log_probs


def generate_student_step_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    problems: List[str],
    previous_steps_list: List[List[str]],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop_tokens: Optional[List[str]] = None,
) -> List[Tuple[str, Dict]]:
    """
    Generate steps for a batch of problems.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        problems: List of problem statements
        previous_steps_list: List of lists of previous steps (one per problem)
        max_new_tokens: Maximum tokens to generate per step
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        stop_tokens: Custom stop tokens
        
    Returns:
        List of (generated_step, metadata) tuples, one per problem
    """
    if len(problems) != len(previous_steps_list):
        raise ValueError("problems and previous_steps_list must have the same length")
    
    if stop_tokens is None:
        stop_tokens = STOP_TOKENS
    
    # Build prompts for all problems
    prompts = [
        build_srl_prompt(problem, prev_steps, include_closing_tag=False)
        for problem, prev_steps in zip(problems, previous_steps_list)
    ]
    
    # Tokenize batch
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    
    # Generate batch
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=None,
        )
    
    # Extract steps from each output
    results = []
    for i, output_id_seq in enumerate(output_ids):
        full_output = tokenizer.decode(output_id_seq, skip_special_tokens=False)
        generated_step = extract_next_step_from_output(full_output)
        
        # Apply stop token filtering
        for stop_token in stop_tokens:
            if stop_token in generated_step:
                generated_step = generated_step.split(stop_token)[0].strip()
                break
        
        metadata = {
            "input_ids": inputs["input_ids"][i],
            "output_ids": output_id_seq,
            "full_output": full_output,
            "prompt": prompts[i],
        }
        results.append((generated_step, metadata))
    
    return results
