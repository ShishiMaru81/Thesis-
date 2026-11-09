#!/usr/bin/env python3
"""Stage 6D — evaluate remaining arms in the 16-cell matrix (skip already done)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT_ROOT / "reports" / "s6" / "predictions"
ENV_PYTHON = PROJECT_ROOT.parent / "finetune_env" / "Scripts" / "python.exe"

ZEROSHOT_ARMS = [
    "mistral_nemo_zeroshot",
    "phi35_mini_zeroshot",
    "phi4_mini_zeroshot",
]

FINETUNE_ARMS = [
    "llama31_8b_seed1337",
    "llama31_8b_seed2024",
    "mistral_nemo_seed42",
    "mistral_nemo_seed1337",
    "mistral_nemo_seed2024",
    "phi35_mini_seed42",
    "phi35_mini_seed1337",
    "phi35_mini_seed2024",
    "phi4_mini_seed42",
    "phi4_mini_seed1337",
    "phi4_mini_seed2024",
]

ALL_ARMS = ZEROSHOT_ARMS + FINETUNE_ARMS


def main() -> None:
    assert (PROJECT_ROOT / "data/augmented/train_augmented.jsonl").exists()
    assert (PROJECT_ROOT / "data/hardened/test_hardened.jsonl").exists()

    pending = [a for a in ALL_ARMS if not (PRED_DIR / f"{a}.jsonl").exists()]
    done = [a for a in ALL_ARMS if (PRED_DIR / f"{a}.jsonl").exists()]

    print("=" * 80)
    print("STAGE 6D — EVALUATE REMAINING ARMS")
    print("=" * 80)
    print(f"Matrix arms in this run: {len(ALL_ARMS)}")
    print(f"Already on disk: {len(done)}  {done}")
    print(f"To evaluate: {len(pending)}")
    print(f"Zero-shot pending: {sum(1 for a in pending if a.endswith('_zeroshot'))}")
    print(f"Fine-tuned pending: {sum(1 for a in pending if not a.endswith('_zeroshot'))}")
    print()

    if not pending:
        print("All arms already evaluated.")
        return

    total = len(pending)
    for run_count, arm in enumerate(pending, start=1):
        print("=" * 80)
        print(f"ARM {run_count}/{total} — {arm}")
        print("=" * 80)

        start_time = time.time()
        cmd = [
            str(ENV_PYTHON),
            str(PROJECT_ROOT / "scripts" / "s6_eval.py"),
            "--arms", arm,
            "--batch_size", "16",
            "--append",
            "--skip_manifest_verify",
        ]
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)

        elapsed = (time.time() - start_time) / 60
        print(f"Completed in {elapsed:.1f} min\n")

        if result.returncode != 0:
            print(f"FAILED: {arm}")
            print("Stopping. Check logs and fix before restarting.")
            sys.exit(1)

    print("=" * 80)
    print("ALL REMAINING ARMS COMPLETE")
    print("=" * 80)
    print("Next: python scripts/s6d_analyze_clarify_comparison.py")


if __name__ == "__main__":
    main()
