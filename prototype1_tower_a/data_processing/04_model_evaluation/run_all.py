"""
Tower A - Phase 4: Run All Model Evaluations

Runs evaluate.py for every combination of split × regime × model.
Each job runs as a subprocess so that environment variables set for
single-threaded BLAS take effect before sklearn imports.

Usage:
  cd data_processing/
  python 04_model_evaluation/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

def main() -> None:
    here = Path(__file__).resolve().parent
    eval_py = here / "evaluate.py"

    if not eval_py.exists():
        raise FileNotFoundError(f"Missing: {eval_py}")

    jobs = []
    for split in ["random", "etld1"]:
        for model in ["lr", "rf", "hgb"]:
            jobs.append((split, "AE", model))
        jobs.append((split, "ngrams", "sgd"))

    print(f"Running {len(jobs)} evaluation jobs...\n")

    for split, regime, model in jobs:
        cmd = [
            sys.executable, str(eval_py),
            "--split", split,
            "--regime", regime,
            "--model", model,
            "--seed", "42",
            "--fpr_cap", "0.02",
            "--recall_target", "0.95",
            "--latency_sample", "10000",
        ]
        print(f"Running: {split}/{regime}/{model}")
        # Run from data_processing/ so that project_root() in io_utils resolves correctly.
        subprocess.run(cmd, check=True, cwd=str(here.parent))

    print(f"\nAll evaluations complete.")
    print(f"Summary: {here / 'outputs' / 'summary_results.csv'}")

if __name__ == "__main__":
    main()