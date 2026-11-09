#!/usr/bin/env python3
"""Stage 6E — evaluate clarify-retrained Mistral, Phi-3.5, Phi-4."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PYTHON = PROJECT_ROOT.parent / "finetune_env" / "Scripts" / "python.exe"
S6E_ROOT = PROJECT_ROOT / "models" / "s6e"

ARMS = [
    "mistral_nemo_seed42_clarify",
    "phi35_mini_seed42_clarify",
    "phi4_mini_seed42_clarify",
]


def main() -> int:
    print("=" * 80)
    print("STAGE 6E — EVALUATE CLARIFY-TRAINED MODELS")
    print("=" * 80)
    print("Test set: test_hardened.jsonl (362 scorable rows)")
    print(f"Models: {len(ARMS)} retrained with clarify data\n")

    for idx, arm in enumerate(ARMS, 1):
        adapter = S6E_ROOT / arm / "final_adapter" / "adapter_model.safetensors"
        if not adapter.exists():
            raise SystemExit(f"missing adapter for {arm}: {adapter}")

        print("=" * 80)
        print(f"ARM {idx}/{len(ARMS)} — {arm}")
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
        print(f"Command: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)

        elapsed = (time.time() - start_time) / 60
        print(f"Completed in {elapsed:.1f} min\n")

        if result.returncode != 0:
            print(f"FAILED: {arm}")
            return 1

    print("=" * 80)
    print("ALL 3 CLARIFY-TRAINED MODELS EVALUATED")
    print("=" * 80)
    print("Next: python scripts/s6e_analyze_clarify_impact.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
