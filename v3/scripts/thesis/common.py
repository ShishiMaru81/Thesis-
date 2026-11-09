"""Shared helpers for scripts/thesis/*. Read-only access to the repo."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "thesis" / "generated"
sys.path.insert(0, str(ROOT / "scripts"))

TEST_PATH = ROOT / "data" / "hardened" / "test_hardened.jsonl"
TEST_SHA_PATH = ROOT / "data" / "hardened" / "test_hardened.sha256"
PRED_DIR = ROOT / "reports" / "s6" / "predictions"

# (repo key, macro stem, display label)
MODELS = [("llama31_8b", "Llama", "Llama-3.1-8B"),
          ("mistral_nemo", "Mistral", "Mistral-NeMo-12B"),
          ("phi35_mini", "PhiThree", "Phi-3.5-mini"),
          ("phi4_mini", "PhiFour", "Phi-4-mini")]
SEEDS = [42, 1337, 2024]

# Training/validation files: never opened by thesis scripts.
FORBIDDEN = {"train_augmented.jsonl", "train_augmented_clarify.jsonl", "train.jsonl",
             "train_clarify.jsonl", "train_seed.jsonl", "val.jsonl", "s6b_clarify_train.jsonl",
             "train_augmented_clarify_bn.jsonl", "train_augmented_clarify_en.jsonl"}
# Stage 8 inferential outputs were blocked until Stage 8 was released (2026-09-24).
FORBIDDEN_STATS: set[str] = set()


def guard(path: Path) -> Path:
    if path.name in FORBIDDEN or path.name in FORBIDDEN_STATS:
        raise SystemExit(f"ABORT: thesis scripts must not read {path}")
    return path


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with guard(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_test_sha() -> str:
    want = TEST_SHA_PATH.read_text(encoding="utf-8").split()[0]
    got = sha256_file(TEST_PATH)
    if got != want:
        raise SystemExit(f"ABORT: {rel(TEST_PATH)} SHA-256 {got} != locked {want}")
    return got


def read_text_any(path: Path) -> str:
    """Read a log whatever its encoding. Several logs are UTF-16LE console
    captures (PowerShell redirects); the rest are UTF-8."""
    b = guard(path).read_bytes()
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16")
    if b[:3] == b"\xef\xbb\xbf":
        return b[3:].decode("utf-8", errors="replace")
    return b.decode("utf-8", errors="replace")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in guard(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_tokenizer(model_key: str):
    """Same loader as the pre-registered Stage 8 code (s8_stats.load_tokenizer),
    but strictly offline: local Hugging Face cache only, never the network."""
    from transformers import AutoTokenizer
    import s6_eval
    return AutoTokenizer.from_pretrained(s6_eval.MODEL_REGISTRY[model_key], local_files_only=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=False) + "\n",
                    encoding="utf-8", newline="\n")
