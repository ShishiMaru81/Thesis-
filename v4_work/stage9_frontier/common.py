"""Shared paths, loaders and request construction for Stage 9. Read-only on v3 and stage8b."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
THESIS = V4.parent
V3 = THESIS / "v3"
S8B = V4 / "results" / "stage8b"
OUT = V4 / "results" / "stage9_frontier"
RAW = OUT / "raw"

TEST_PATH = V3 / "data" / "hardened" / "test_hardened.jsonl"
TEST_SHA_PREFIX = "9314d181"
PROMPT_P = S8B / "prompt_P.txt"
FEWSHOT = S8B / "fewshot_examples.json"
# Frozen in stage8b/FROZEN.txt; hashed over LF line endings (the Drive zip stores CRLF).
PROMPT_P_SHA = "cfb493045a9f52e51857ca9f8b4f1cea7a1c0c23831e55cb1d3dbd28d541da0a"
FEWSHOT_SHA = "329b484a4c9517bd37c2d36a4febe04d35635f781444756fcd6634add679c993"

ARMS = ["ZS", "ZS+P"]
ARM_SLUG = {"ZS": "ZS", "ZS+P": "ZS_P"}

# Included verbatim under "Limitations" in REPORT.md (Step 4).
REPORT_LIMITATIONS = [
    "The thinking level was set to low and the API reported no thinking tokens, "
    "so it is unknown whether hidden reasoning occurred.",
]

sys.path.insert(0, str(V3 / "scripts"))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def canonical(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def active_models(cfg: dict) -> tuple[list[dict], list[dict]]:
    active, skipped = [], []
    for m in cfg["models"]:
        if not m.get("model_id") or m["model_id"].startswith("<FILL"):
            skipped.append({**m, "reason": "model_id not filled in config.yaml"})
        else:
            active.append(m)
    return active, skipped


def load_test() -> list[dict]:
    got = sha256_file(TEST_PATH)
    if not got.startswith(TEST_SHA_PREFIX):
        raise SystemExit(f"STOP: test_hardened.jsonl sha256 {got[:16]} does not start with {TEST_SHA_PREFIX}")
    with TEST_PATH.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def select_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """The four locked subsets, in test-file order."""
    sub = {
        "clarify_68": [r for r in rows if r["qtype"] == "clarify" and r["gold_status"] != "pending_human"],
        "numeric_146": [r for r in rows if r["qtype"] == "numeric_runtime"],
        "abstain_seed_34": [r for r in rows if r["qtype"] == "abstain" and r["category"] == "seed"],
        "ood_hard_40": [r for r in rows if r["qtype"] == "abstain" and r["category"] == "out_of_domain"],
    }
    sub["numeric_single_116"] = [r for r in sub["numeric_146"] if "scenario" in r]
    expect = {"clarify_68": 68, "numeric_146": 146, "abstain_seed_34": 34, "ood_hard_40": 40, "numeric_single_116": 116}
    for k, n in expect.items():
        if len(sub[k]) != n:
            raise SystemExit(f"STOP: subset {k} has {len(sub[k])} rows, expected {n}")
    cats = {}
    for r in sub["clarify_68"]:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    if cats != {"missing_parameter": 40, "misspelled_crop": 16, "unit_ambiguity": 12}:
        raise SystemExit(f"STOP: clarify category mix {cats}")
    return sub


def run_rows(sub: dict[str, list[dict]]) -> list[dict]:
    """Rows sent to the API: clarify 68 + numeric 146 + seed abstain 34 + hard OOD 40 = 288."""
    return sub["clarify_68"] + sub["numeric_146"] + sub["abstain_seed_34"] + sub["ood_hard_40"]


def _lf(p: Path) -> str:
    return p.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def load_prompt_assets() -> tuple[str, list[dict]]:
    p_text = _lf(PROMPT_P)
    fs_text = _lf(FEWSHOT)
    if sha256_bytes(p_text.encode("utf-8")) != PROMPT_P_SHA:
        raise SystemExit("STOP: prompt_P.txt does not match the Stage 8B frozen hash")
    if sha256_bytes(fs_text.encode("utf-8")) != FEWSHOT_SHA:
        raise SystemExit("STOP: fewshot_examples.json does not match the Stage 8B frozen hash")
    return p_text, json.loads(fs_text)["examples"]


def build_messages(arm: str, question: str, p_text: str, examples: list[dict]) -> dict:
    """Provider-neutral request content. ZS: one user turn, no system. ZS+P: Stage 8B layout."""
    if arm == "ZS":
        return {"system": None, "turns": [{"role": "user", "text": question}]}
    turns = []
    for ex in examples:
        turns.append({"role": "user", "text": ex["question"]})
        turns.append({"role": "assistant", "text": ex["answer"]})
    turns.append({"role": "user", "text": question})
    return {"system": p_text, "turns": turns}
