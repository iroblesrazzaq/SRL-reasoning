#!/usr/bin/env python3
"""
Inspect and optionally repair tokenizer/config files for a merged model.

Typical usage on Colab (after mounting Drive):
    python scripts/check_tokenizer_files.py \\
        --model_dir /content/drive/MyDrive/srl_outputs/merged_sft_0.5b_test \\
        --base_model Qwen/Qwen2.5-0.5B-Instruct \\
        --repair

The script checks:
- config.json: presence of a model_type field
- tokenizer_config.json: warns if model_type is present or if tokenizer_class is missing
- tokenizer.json / vocab files: existence

If --repair is provided, the script re-saves the base model's config and tokenizer
into the merged directory to ensure transformers/vLLM can load them.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import AutoConfig, AutoTokenizer


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON content if the file exists."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_files(model_dir: Path) -> Dict[str, Any]:
    """Check key tokenizer/config files and report potential issues."""
    report = {"model_dir": str(model_dir)}

    config_path = model_dir / "config.json"
    tok_config_path = model_dir / "tokenizer_config.json"
    tokenizer_json = model_dir / "tokenizer.json"
    vocab_json = model_dir / "vocab.json"
    merges_txt = model_dir / "merges.txt"

    config = read_json(config_path)
    tok_config = read_json(tok_config_path)

    report["config.json"] = {
        "exists": config is not None,
        "has_model_type": bool(config and config.get("model_type")),
    }

    report["tokenizer_config.json"] = {
        "exists": tok_config is not None,
        "has_model_type_field": bool(tok_config and "model_type" in tok_config),
        "tokenizer_class": tok_config.get("tokenizer_class") if tok_config else None,
        "model_max_length": tok_config.get("model_max_length") if tok_config else None,
    }

    report["tokenizer_files"] = {
        "tokenizer.json": tokenizer_json.exists(),
        "vocab.json": vocab_json.exists(),
        "merges.txt": merges_txt.exists(),
    }

    return report


def print_report(report: Dict[str, Any]) -> None:
    """Pretty-print the inspection report."""
    print(f"Model dir: {report['model_dir']}")
    cfg = report["config.json"]
    print(f"- config.json: {'found' if cfg['exists'] else 'MISSING'} "
          f"(model_type present: {cfg['has_model_type']})")

    tok = report["tokenizer_config.json"]
    print(f"- tokenizer_config.json: {'found' if tok['exists'] else 'MISSING'} "
          f"(contains model_type: {tok['has_model_type_field']}, "
          f"tokenizer_class: {tok['tokenizer_class']})")

    files = report["tokenizer_files"]
    print(f"- tokenizer.json: {'found' if files['tokenizer.json'] else 'MISSING'}")
    print(f"- vocab.json: {'found' if files['vocab.json'] else 'MISSING'}")
    print(f"- merges.txt: {'found' if files['merges.txt'] else 'MISSING'}")


def repair(model_dir: Path, base_model: str) -> None:
    """
    Re-save the base model's config and tokenizer into the merged directory.

    This fixes missing model_type in config.json and removes any stray model_type
    fields in tokenizer_config.json by overwriting with a clean save.
    """
    print(f"Loading base config/tokenizer from '{base_model}'...")
    base_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"Writing clean config/tokenizer files into '{model_dir}'...")
    base_config.save_pretrained(model_dir)
    base_tokenizer.save_pretrained(model_dir)
    print("✓ Repair complete. Re-run the checker to verify.")


def main():
    parser = argparse.ArgumentParser(description="Check tokenizer/config files for a merged model.")
    parser.add_argument("--model_dir", required=True, help="Path to merged model directory")
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model ID or path to copy config/tokenizer from",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Overwrite config/tokenizer files with the base model versions",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    report = check_files(model_dir)
    print_report(report)

    if args.repair:
        repair(model_dir, args.base_model)
        print("\nRe-checking after repair...\n")
        report = check_files(model_dir)
        print_report(report)


if __name__ == "__main__":
    main()
