"""Data splitting utilities shared across SRL and SFT."""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


def _load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def split_by_trajectory(
    data_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split SRL examples by trajectory id to avoid leakage across splits.
    """
    examples = _load_jsonl(data_path)
    if not examples:
        return [], [], []

    by_traj = {}
    for ex in examples:
        traj_id = ex.get("traj_id")
        by_traj.setdefault(traj_id, []).append(ex)

    traj_ids = list(by_traj.keys())
    random.Random(seed).shuffle(traj_ids)

    total = len(traj_ids)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    train_ids = set(traj_ids[:train_count])
    val_ids = set(traj_ids[train_count:train_count + val_count])
    test_ids = set(traj_ids[train_count + val_count:])

    train = [ex for tid in train_ids for ex in by_traj.get(tid, [])]
    val = [ex for tid in val_ids for ex in by_traj.get(tid, [])]
    test = [ex for tid in test_ids for ex in by_traj.get(tid, [])]

    return train, val, test


def save_splits(
    train_examples: List[Dict],
    val_examples: List[Dict],
    test_examples: List[Dict],
    output_dir: str,
) -> None:
    """
    Save split datasets to JSONL files in the provided directory.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, records in (
        ("train", train_examples),
        ("val", val_examples),
        ("test", test_examples),
    ):
        path = Path(output_dir) / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
