"""Utilities for saving and loading benchmark results."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def _utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_slug(text: str) -> str:
    """Convert arbitrary text into a filesystem-safe slug."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown"


def _get_git_commit() -> Optional[str]:
    """Best-effort retrieval of the current git commit hash."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


@dataclass
class BenchmarkResult:
    benchmark: str
    benchmark_type: str  # e.g., "greedy" or "avg32"
    score: float
    model_name: str
    model_path: str
    num_questions: int
    eval_time_seconds: Optional[float] = None
    seed: Optional[int] = None
    run_id: str = field(default_factory=_utc_timestamp)
    created_at_utc: str = field(default_factory=_utc_timestamp)
    commit_hash: Optional[str] = field(default_factory=_get_git_commit)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    def filename(self) -> str:
        """Generate a consistent filename for this result."""
        parts = [
            self.created_at_utc.replace(":", "").replace("-", "").replace("Z", "Z"),
            _safe_slug(self.benchmark),
            _safe_slug(self.benchmark_type),
            _safe_slug(self.model_name),
        ]
        return "-".join(parts) + ".json"

    def save(self, output_dir: Path | str = "benchmarks/results") -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / self.filename()
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "BenchmarkResult":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def load_all_results(directory: Path | str = "benchmarks/results") -> Iterable[BenchmarkResult]:
    directory = Path(directory)
    if not directory.exists():
        return []
    for path in sorted(directory.glob("*.json")):
        try:
            yield BenchmarkResult.load(path)
        except Exception:
            continue


def summarize_results(results: Iterable[BenchmarkResult]) -> Dict[Tuple[str, str], float]:
    """
    Compute the best score per (benchmark, model_name) pair.
    Useful for quick plotting/analysis.
    """
    best: Dict[Tuple[str, str], float] = {}
    for res in results:
        key = (res.benchmark, res.model_name)
        best[key] = max(best.get(key, float("-inf")), res.score)
    return best
