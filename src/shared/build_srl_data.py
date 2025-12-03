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

# Regex pattern to match structured steps in deepseek_attempt field
# Format: "N. **Step Name**:" where N is a number
STEP_HEADER_PATTERN = re.compile(r'(\d+)\.\s*\*\*([^*]+)\*\*:?\s*', re.MULTILINE)

# Validation pattern to check if text has the expected step format
# Requires at least 2 numbered steps with **bold titles**
VALID_FORMAT_PATTERN = re.compile(r'^\d+\.\s*\*\*[^*]+\*\*', re.MULTILINE)


def _has_valid_step_format(text: str) -> bool:
    """
    Check if text follows the expected structured step format.
    
    Valid format has numbered steps with bold titles like:
        1. **Step Name**:
           - Content here
        
        2. **Another Step**:
           - More content
    
    Args:
        text: The deepseek_attempt text to validate
        
    Returns:
        True if text has at least 2 properly formatted steps
    """
    if not text:
        return False
    matches = VALID_FORMAT_PATTERN.findall(text)
    return len(matches) >= 2


def _parse_structured_steps(text: str) -> List[Dict[str, str]]:
    """
    Parse structured steps from deepseek_attempt text.
    
    Extracts step number, title, and body from formatted text like:
        1. **Prime Factorization**:
           - The prime factors are...
        
        2. **Coprime Pairs**:
           - For a x b = 20!...
    
    Args:
        text: The deepseek_attempt text to parse
        
    Returns:
        List of dicts with keys: step_num, step_title, step_body
    """
    matches = list(STEP_HEADER_PATTERN.finditer(text))
    if not matches:
        return []
    
    steps = []
    for i, match in enumerate(matches):
        step_num = match.group(1)
        step_title = match.group(2).strip()
        
        # Extract body: from end of this header to start of next header (or end of text)
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        step_body = text[body_start:body_end].strip()
        
        steps.append({
            "step_num": step_num,
            "step_title": step_title,
            "step_body": step_body,
        })
    
    return steps


def normalize_trajectory(row: Dict, idx: int) -> Optional[Dict]:
    """
    Convert a raw dataset row into a normalized trajectory dict.

    This function extracts the problem statement and parses the structured
    reasoning steps from the deepseek_attempt field. The output format is:
        {
          "id": str,
          "problem": str,
          "steps": [{"step_num": str, "step_title": str, "step_body": str}, ...]
        }

    Only examples with properly formatted steps (N. **Step Name**:) are kept.
    Examples without this format are discarded.

    Args:
        row: Raw dataset row (dict)
        idx: Fallback index if row doesn't have an "id" field

    Returns:
        Normalized trajectory dict, or None if the row is invalid/missing required fields
    """
    # ---- ID ----
    ex_id = str(row.get("id", idx))

    # ---- Problem text ----
    problem = row.get("question") or row.get("prompt")
    if problem is None:
        return None

    # ---- Reasoning steps from deepseek_attempt ----
    # Use deepseek_attempt which has structured steps with numbered **bold titles**
    # This field contains cleaner, more structured reasoning than deepseek_thinking_trajectory
    attempt = row.get("deepseek_attempt")
    
    if not attempt or not isinstance(attempt, str):
        return None
    
    # Validate that the attempt follows the expected format
    if not _has_valid_step_format(attempt):
        return None
    
    # Parse the structured steps
    steps = _parse_structured_steps(attempt)
    
    # Filter out empty steps
    steps = [s for s in steps if s["step_body"].strip()]
    
    if len(steps) < 2:
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
    step_body serves as the target label for supervised learning.

    Given one normalized trajectory:
        { "id": ..., "problem": ..., "steps": [{step_num, step_title, step_body}, ...] }

    This produces N examples (one for each step):

      k = 0: previous_steps = [],        step_body = S1
      k = 1: previous_steps = [S1],      step_body = S2
      ...
      k = N-1: previous_steps = [S1..S_{N-1}], step_body = SN

    Args:
        traj: Normalized trajectory dict with "id", "problem", and "steps"

    Yields:
        Dict with keys: traj_id, step_idx, problem, previous_steps, step_title, step_body
    """
    problem = traj["problem"]
    steps = traj["steps"]
    tid = traj["id"]

    for k in range(len(steps)):
        # Previous steps as formatted strings for context
        prev_steps_formatted = []
        for ps in steps[:k]:
            formatted = f"{ps['step_num']}. **{ps['step_title']}**: {ps['step_body']}"
            prev_steps_formatted.append(formatted)
        
        current_step = steps[k]

        yield {
            "traj_id": tid,
            "step_idx": k,
            "problem": problem,
            "previous_steps": prev_steps_formatted,
            "step_title": current_step["step_title"],
            "step_body": current_step["step_body"],
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

def main():
    """
    CLI entrypoint: Load dataset, normalize trajectories, build SRL examples, and save.

    Usage:
        python -m src.shared.build_srl_data
        or
        srl-build-data

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


if __name__ == "__main__":
    main()
