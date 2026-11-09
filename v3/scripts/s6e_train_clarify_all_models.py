#!/usr/bin/env python3
"""Stage 6E — retrain Mistral, Phi-3.5, Phi-4 with clarify-augmented corpus."""
from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PYTHON = PROJECT_ROOT.parent / "finetune_env" / "Scripts" / "python.exe"
EXPECTED_SHA = "53ce9f24ed21dc57f2c8f0452ede0f6a0b9cb99b1fba22dad70255cbd5cfa71c"
TRAIN_RENDERED = PROJECT_ROOT / "data" / "rendered" / "train_clarify.jsonl"
TRAIN_AUGMENTED = PROJECT_ROOT / "data" / "augmented" / "train_augmented_clarify.jsonl"
OUT_ROOT = PROJECT_ROOT / "models" / "s6e"
LOG_DIR = PROJECT_ROOT / "logs"
S5_ROOT = PROJECT_ROOT / "models" / "s5"

MODELS = [
    ("mistral_nemo", 42),
    ("phi35_mini", 42),
    ("phi4_mini", 42),
]
SEEDS = [42, 1337, 2024]
MODEL_KEYS = ["llama31_8b", "mistral_nemo", "phi35_mini", "phi4_mini"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry_gate() -> None:
    print("=" * 80)
    print("ENTRY GATE — STAGE 6E")
    print("=" * 80)

    if not TRAIN_AUGMENTED.exists():
        raise SystemExit(f"FAIL: missing {TRAIN_AUGMENTED}")
    sha = sha256_file(TRAIN_AUGMENTED)
    if sha != EXPECTED_SHA:
        raise SystemExit(f"FAIL: train_augmented_clarify SHA {sha} != {EXPECTED_SHA}")
    print(f"OK  train_augmented_clarify.jsonl ({sum(1 for _ in TRAIN_AUGMENTED.open(encoding='utf-8'))} rows)")
    print(f"OK  SHA-256 {sha}")

    if not TRAIN_RENDERED.exists():
        raise SystemExit(f"FAIL: missing rendered train {TRAIN_RENDERED}")
    print(f"OK  train_clarify.jsonl (rendered, {sum(1 for _ in TRAIN_RENDERED.open(encoding='utf-8'))} rows)")

    missing = []
    for model in MODEL_KEYS:
        for seed in SEEDS:
            adapter = S5_ROOT / f"{model}_seed{seed}" / "final_adapter" / "adapter_model.safetensors"
            if not adapter.exists():
                missing.append(str(adapter.relative_to(PROJECT_ROOT)))
    if missing:
        raise SystemExit(f"FAIL: {len(missing)} Stage 5 adapters missing:\n  " + "\n  ".join(missing[:5]))
    print(f"OK  12 Stage 5 adapters on disk")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"OK  logs/ and models/s6e/ ready\n")


def main() -> int:
    entry_gate()

    print("=" * 80)
    print("STAGE 6E — CLARIFY TRAINING FOR 3 MODELS")
    print("=" * 80)
    print("Training data: train_clarify.jsonl (3,852 rows, rendered)")
    print("Hyperparameters: Stage 5 defaults (3 epochs, r=16, alpha=32)")
    print(f"Output: {OUT_ROOT.relative_to(PROJECT_ROOT)}/\n")

    total = len(MODELS)
    for idx, (model, seed) in enumerate(MODELS, 1):
        tag = f"{model}_seed{seed}_clarify"
        print("=" * 80)
        print(f"MODEL {idx}/{total} — {model} seed {seed} (CLARIFY)")
        print("=" * 80)

        start_time = time.time()
        cmd = [
            str(ENV_PYTHON),
            str(PROJECT_ROOT / "scripts" / "s5_train.py"),
            "--model", model,
            "--seed", str(seed),
            "--train_file", str(TRAIN_RENDERED),
            "--run_suffix", "clarify",
            "--out_root", str(OUT_ROOT),
        ]
        print(f"Command: {' '.join(cmd)}\n")
        log_path = LOG_DIR / f"train_{tag}.log"
        with log_path.open("w", encoding="utf-8") as log_f:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log_f, stderr=subprocess.STDOUT)

        elapsed = (time.time() - start_time) / 60
        print(f"Completed in {elapsed:.1f} min  (log: {log_path.relative_to(PROJECT_ROOT)})")

        if result.returncode != 0:
            print(f"FAILED: {tag}")
            print("Stopping. Check log and fix before restarting.")
            return 1
        print()

    print("=" * 80)
    print("ALL 3 MODELS RETRAINED WITH CLARIFY DATA")
    print("=" * 80)
    print("Next: python scripts/s6e_eval_clarify_models.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
