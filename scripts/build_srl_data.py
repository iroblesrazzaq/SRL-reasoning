import json
import re
from pathlib import Path
from typing import List, Dict, Iterable, Optional

from datasets import load_dataset, Dataset
from tqdm import tqdm


# ---------- 1. Load teacher trajectories ----------

def load_teacher_dataset(
    source: str = "simplescaling/s1K-1.1",
    split: str = "train",
) -> Dataset:
    """
    Load a dataset of teacher trajectories from Hugging Face.

    For s1K-1.1, each row roughly looks like:
      {
        "id": "...",
        "question": "...",
        "cot": ["step1 ...", "step2 ...", ...] OR string,
        "final": "..."
      }

    Args:
        source: Hugging Face dataset identifier
        split: Dataset split to load (e.g., "train", "validation")

    Returns:
        Hugging Face Dataset object
    """
    print(f"Loading dataset: {source} (split: {split})...")
    ds = load_dataset(source, split=split)
    print(f"Loaded {len(ds)} examples")
    return ds


# ---------- 2. Normalize each row into {id, problem, steps[]} ----------

def _is_step_boundary(line: str) -> bool:
    """
    Detect if a line looks like the start of a new reasoning step.
    
    Patterns detected:
    - "Step 1:", "Step 2:", etc.
    - "1.", "2.", "3.", etc. (any number followed by period)
    - "First,", "Second,", "Third,", etc.
    - Lines starting with common step indicators
    """
    line_lower = line.lower().strip()
    
    # Pattern: "Step N:" or "Step N."
    if re.match(r'^step\s+\d+[.:]', line_lower):
        return True
    
    # Pattern: "N." where N is a number (e.g., "1.", "2.", "10.")
    if re.match(r'^\d+\.\s', line):
        return True
    
    # Pattern: "First,", "Second,", "Third,", etc.
    ordinal_patterns = ['first', 'second', 'third', 'fourth', 'fifth', 
                       'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
    if any(line_lower.startswith(ord + ',') or line_lower.startswith(ord + '.') 
           for ord in ordinal_patterns):
        return True
    
    return False


def normalize_trajectory(row: Dict, idx: int) -> Optional[Dict]:
    """
    Convert a raw dataset row into a normalized trajectory dict.

    This function extracts the problem statement and splits the chain-of-thought
    reasoning into discrete steps. The output format is:
        {
          "id": str,
          "problem": str,
          "steps": [str, str, ...]
        }

    The steps will later be used to create SRL training examples where each step
    becomes a target action given the previous steps as state.

    Args:
        row: Raw dataset row (dict)
        idx: Fallback index if row doesn't have an "id" field

    Returns:
        Normalized trajectory dict, or None if the row is invalid/missing required fields

    Note:
        You can customize field names (e.g., "question" -> "prompt") depending on
        your dataset schema.
    """
    # ---- ID ----
    ex_id = str(row.get("id", idx))

    # ---- Problem text ----
    # s1K-style datasets usually use "question" or "prompt"
    # Adjust these field names based on your dataset
    problem = row.get("question") or row.get("prompt")
    if problem is None:
        # Skip rows without a problem statement
        return None

    # ---- Reasoning steps ----
    # Many datasets store CoT as either:
    #   - a list of step strings (already structured)
    #   - a single long string (needs heuristic splitting)
    cot = row.get("cot") or row.get("reasoning") or row.get("solution")

    steps: List[str] = []

    if isinstance(cot, list):
        # Already step-wise: just clean and filter
        steps = [s.strip() for s in cot if s and s.strip()]
    elif isinstance(cot, str):
        text = cot.strip()
        if not text:
            return None

        # Heuristic splitter: break on step boundaries
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        current = []
        for ln in lines:
            # If this line looks like a step boundary and we have accumulated content,
            # save the previous step and start a new one
            if _is_step_boundary(ln) and current:
                steps.append(" ".join(current).strip())
                current = [ln]
            else:
                current.append(ln)
        
        # Don't forget the last step
        if current:
            steps.append(" ".join(current).strip())
    else:
        return None

    # Filter out very short / junk steps
    steps = [s for s in steps if len(s) > 3]

    if not steps:
        return None

    return {
        "id": ex_id,
        "problem": problem.strip(),
        "steps": steps,
    }


def normalize_dataset(ds: Dataset) -> List[Dict]:
    """
    Normalize all rows in a dataset into trajectory dicts.

    Args:
        ds: Hugging Face Dataset

    Returns:
        List of normalized trajectory dicts
    """
    norm = []
    for i, row in enumerate(tqdm(ds, desc="Normalizing trajectories", unit="example")):
        ex = normalize_trajectory(row, i)
        if ex is not None:
            norm.append(ex)
    return norm


# ---------- 3. Build SRL step-wise examples ----------

def build_srl_examples(traj: Dict) -> Iterable[Dict]:
    """
    Transform a teacher trajectory into step-wise SRL training examples.

    This is the core SRL transformation: for each step k in the trajectory,
    we create an example where:
      - State (input): problem + previous steps [S1, ..., S_k]
      - Action (target): the k-th step S_{k+1}

    This corresponds to the SRL idea where the student model sees the problem
    and all previous reasoning steps, then must predict the next step. The
    teacher's step serves as the target label for supervised learning.

    Given one normalized trajectory:
        { "id": ..., "problem": ..., "steps": [S1, S2, ..., SN] }

    This produces N examples (one for each step):

      k = 0: previous_steps = [],        teacher_step = S1
      k = 1: previous_steps = [S1],      teacher_step = S2
      ...
      k = N-1: previous_steps = [S1..S_{N-1}], teacher_step = SN

    Args:
        traj: Normalized trajectory dict with "id", "problem", and "steps"

    Yields:
        Dict with keys: traj_id, step_idx, problem, previous_steps, teacher_step
    """
    problem = traj["problem"]
    steps = traj["steps"]
    tid = traj["id"]

    for k in range(len(steps)):
        prev_steps = steps[:k]          # teacher steps 0..k-1 (state)
        teacher_step = steps[k]         # kth step (target action)

        yield {
            "traj_id": tid,
            "step_idx": k,
            "problem": problem,
            "previous_steps": prev_steps,
            "teacher_step": teacher_step,
        }


def build_srl_dataset(norm_trajs: List[Dict]) -> List[Dict]:
    """
    Build SRL training examples from all normalized trajectories.

    Args:
        norm_trajs: List of normalized trajectory dicts

    Returns:
        List of SRL step-wise training examples
    """
    out = []
    for traj in tqdm(norm_trajs, desc="Building SRL examples", unit="trajectory"):
        out.extend(list(build_srl_examples(traj)))
    return out


# ---------- 4. Save to jsonl ----------

def save_jsonl(records: List[Dict], path: str) -> None:
    """
    Save records to a JSONL file (one JSON object per line).

    Args:
        records: List of dicts to save
        path: Output file path (directory will be created if needed)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in tqdm(records, desc="Saving to JSONL", unit="example"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------- 5. Main entrypoint ----------

if __name__ == "__main__":
    """
    CLI entrypoint: Load dataset, normalize trajectories, build SRL examples, and save.

    Usage:
        python build_srl_data.py

    To use a different dataset, modify the arguments to load_teacher_dataset().
    """
    # 1) Load teacher trajectories (s1K-1.1 by default)
    # You can change the dataset name and split here, or make them CLI arguments
    ds = load_teacher_dataset("simplescaling/s1K-1.1", split="train")
    print()  # blank line for readability

    # 2) Normalize (extract problem and split reasoning into steps)
    norm_trajs = normalize_dataset(ds)
    print(f"\n✓ Normalized {len(norm_trajs)} trajectories from {len(ds)} raw examples")

    # 3) Expand into SRL step-wise examples
    # Each trajectory with N steps becomes N training examples
    srl_examples = build_srl_dataset(norm_trajs)
    print(f"\n✓ Built {len(srl_examples)} SRL step examples")

    # 4) Save to JSONL file
    output_path = "data/srl_steps.jsonl"
    save_jsonl(srl_examples, output_path)
    print(f"\n✓ Saved to {output_path}")
    
    # Print some statistics
    if srl_examples:
        avg_steps = len(srl_examples) / len(norm_trajs) if norm_trajs else 0
        print(f"\n📊 Statistics:")
        print(f"   Average steps per trajectory: {avg_steps:.2f}")
        print(f"   Total trajectories: {len(norm_trajs)}")
        print(f"   Total SRL examples: {len(srl_examples)}")
