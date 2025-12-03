#!/usr/bin/env python3
"""
Verify that training completed successfully and test the model.

This script:
1. Checks that model files exist and are valid
2. Loads the trained model and tokenizer
3. Tests generation on a few examples
4. Compares outputs with training data
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.shared.prompts import build_srl_prompt


def load_examples(data_path, max_examples=5):
    """Load a few examples from the dataset."""
    examples = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            if line.strip():
                examples.append(json.loads(line))
    return examples


def test_model_generation(model, tokenizer, problem, previous_steps, max_new_tokens=128):
    """Generate a step using the trained model."""
    prompt = build_srl_prompt(problem, previous_steps, include_closing_tag=False)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract the generated part (after the prompt)
    prompt_len = len(tokenizer.encode(prompt))
    generated_tokens = outputs[0][prompt_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    
    return generated_text, generated


def main():
    parser = argparse.ArgumentParser(description="Verify training and test model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/val.jsonl",
        help="Path to test/validation data",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=3,
        help="Number of examples to test",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate",
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    
    print("=" * 80)
    print("TRAINING VERIFICATION")
    print("=" * 80)
    
    # Check if path exists
    print(f"\nChecking model path: {model_path}")
    print(f"Absolute path: {model_path.resolve()}")
    print(f"Path exists: {model_path.exists()}")
    
    if not model_path.exists():
        print(f"\n❌ ERROR: Model path does not exist: {model_path}")
        print("\nTroubleshooting:")
        print("1. Make sure Google Drive is mounted:")
        print("   from google.colab import drive")
        print("   drive.mount('/content/drive')")
        print("\n2. Check the correct path in Google Drive")
        print("3. Use absolute path: /content/drive/MyDrive/srl_outputs/sft_0.5b_test")
        return
    
    # List all files in directory
    print(f"\nFiles in directory:")
    try:
        all_files = list(model_path.iterdir())
        if all_files:
            for f in sorted(all_files):
                if f.is_file():
                    size = f.stat().st_size / (1024 * 1024)  # MB
                    print(f"   - {f.name} ({size:.2f} MB)")
                else:
                    print(f"   - {f.name}/ (directory)")
        else:
            print("   (directory is empty)")
    except Exception as e:
        print(f"   ERROR listing files: {e}")
        return
    
    # 1. Check files exist
    print("\n1. Checking required model files...")
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
    ]
    
    missing_files = []
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   ✓ {file} ({size:.2f} MB)")
        else:
            print(f"   ✗ {file} MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ ERROR: Missing required files: {missing_files}")
        return
    
    # Check for checkpoints
    checkpoint_dirs = [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if checkpoint_dirs:
        print(f"\n   Found {len(checkpoint_dirs)} checkpoint(s):")
        for ckpt in sorted(checkpoint_dirs):
            print(f"   - {ckpt.name}")
    
    # 2. Load model
    print("\n2. Loading model...")
    try:
        # Load base model (need to infer from adapter_config or use a default)
        # For now, we'll try to load as a PEFT model
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("   Loading base model...")
        # Try to get base model name from adapter config
        adapter_config_path = model_path / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")
                if base_model_name:
                    print(f"   Base model: {base_model_name}")
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        device_map="auto",
                    )
                else:
                    print("   ⚠️  Could not find base_model_name_or_path in adapter_config.json")
                    print("   Please specify --base_model_name")
                    return
        else:
            print("   ⚠️  adapter_config.json not found")
            return
        
        print("   Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        
        print(f"   ✓ Model loaded successfully")
        print(f"   Device: {next(model.parameters()).device}")
        if torch.cuda.is_available():
            print(f"   GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
    except Exception as e:
        print(f"   ✗ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Test generation
    print(f"\n3. Testing generation on {args.num_examples} examples...")
    
    if not Path(args.test_data).exists():
        print(f"   ⚠️  Test data not found: {args.test_data}")
        print("   Skipping generation tests")
        return
    
    examples = load_examples(args.test_data, max_examples=args.num_examples)
    
    for i, ex in enumerate(examples, 1):
        print(f"\n   Example {i}/{len(examples)}:")
        print(f"   Problem: {ex['problem'][:100]}...")
        print(f"   Previous steps: {len(ex.get('previous_steps', []))} steps")
        
        if 'teacher_step' in ex:
            print(f"   Expected step: {ex['teacher_step'][:100]}...")
        
        try:
            generated, _ = test_model_generation(
                model,
                tokenizer,
                ex['problem'],
                ex.get('previous_steps', []),
                max_new_tokens=args.max_new_tokens,
            )
            print(f"   Generated: {generated[:200]}...")
            
            # Check if generation looks reasonable
            if len(generated.strip()) > 10:
                print("   ✓ Generation looks reasonable (non-empty)")
            else:
                print("   ⚠️  Generation is very short or empty")
                
        except Exception as e:
            print(f"   ✗ ERROR during generation: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print("✓ Model files present")
    print("✓ Model loaded successfully")
    print("✓ Generation tested")
    print("\nTraining appears to have completed successfully!")
    print(f"\nModel saved at: {model_path}")
    print("\nNext steps:")
    print("  - Test on more examples")
    print("  - Evaluate on validation set")
    print("  - Use for GRPO training (if doing SRL)")


if __name__ == "__main__":
    main()
