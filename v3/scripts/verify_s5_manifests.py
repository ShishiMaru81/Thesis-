#!/usr/bin/env python3
"""
verify_s5_manifests.py — Stage 6 entry gate.

Confirms the twelve Stage 5 manifests are complete, mutually consistent, and
describe twelve genuinely distinct training runs. Checks, in order:

  1. all 12 cells present (4 models x 3 seeds), no extras
  2. identical top-level key sets across all 12
  3. lora_target_modules / lora_modules_attached / trainable_params populated,
     and every declared target actually owns adapted modules
  4. masking_gate == PASS everywhere
  5. adapter_sha256 present, well-formed, and unique  (a duplicate would mean
     two cells wrote the same weights - a silently short-circuited run)
  6. the adapter file each manifest points at exists on disk
  7. trainable_params consistent within a model across seeds
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

REP = Path(__file__).resolve().parents[1] / "reports" / "s5"
MODELS = ["llama31_8b", "mistral_nemo", "phi35_mini", "phi4_mini"]
SEEDS = [42, 1337, 2024]

REQUIRED = ["lora_target_modules", "lora_modules_attached", "trainable_params",
            "masking_gate", "adapter_sha256", "adapter_dir", "global_step"]

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL  {msg}")


paths = sorted(p for p in glob.glob(str(REP / "*_seed*.json"))
               if "smoke" not in os.path.basename(p))
manifests = {os.path.basename(p)[:-5]: json.load(open(p, encoding="utf-8")) for p in paths}

print("=" * 78)
print("STAGE 5 MANIFEST VERIFICATION")
print("=" * 78)

# 1 -------------------------------------------------------------- completeness
expected = {f"{m}_seed{s}" for m in MODELS for s in SEEDS}
print(f"\n[1] cells present: {len(manifests)}/12")
for miss in sorted(expected - set(manifests)):
    fail(f"missing manifest: {miss}")
for extra in sorted(set(manifests) - expected):
    fail(f"unexpected manifest: {extra}")

# 2 -------------------------------------------------------------- schema parity
print("\n[2] schema parity")
keysets = {name: frozenset(m.keys()) for name, m in manifests.items()}
if len(set(keysets.values())) == 1:
    print(f"  OK    all {len(manifests)} share an identical {len(next(iter(keysets.values())))}-key schema")
else:
    ref_name, ref_keys = next(iter(keysets.items()))
    for name, ks in keysets.items():
        if ks != ref_keys:
            fail(f"{name}: missing={sorted(ref_keys - ks)} extra={sorted(ks - ref_keys)}")

# 3 -------------------------------------------------------------- lora fields
print("\n[3] LoRA placement fields")
for name, m in sorted(manifests.items()):
    for k in REQUIRED:
        if m.get(k) in (None, "", []):
            fail(f"{name}: '{k}' missing or empty")
    tgts, att = m.get("lora_target_modules"), m.get("lora_modules_attached")
    if isinstance(tgts, list) and isinstance(att, dict):
        unattached = [t for t in tgts if att.get(t, 0) == 0]
        if unattached:
            fail(f"{name}: declared targets with zero adapted modules: {unattached}")
    elif tgts is not None:
        fail(f"{name}: lora_target_modules should be a list, got {type(tgts).__name__}")
if not failures:
    print("  OK    all populated; every declared target owns adapted modules")

# 4 -------------------------------------------------------------- masking gate
print("\n[4] masking gate")
bad = {n: m.get("masking_gate") for n, m in manifests.items() if m.get("masking_gate") != "PASS"}
if bad:
    for n, v in bad.items():
        fail(f"{n}: masking_gate = {v!r}")
else:
    print(f"  OK    {len(manifests)}/12 PASS")

# 5 -------------------------------------------------------------- sha uniqueness
print("\n[5] adapter SHA-256 uniqueness")
shas: dict[str, list[str]] = {}
for name, m in manifests.items():
    s = m.get("adapter_sha256")
    if not s or not re.fullmatch(r"[0-9a-f]{64}", str(s)):
        fail(f"{name}: malformed adapter_sha256 {s!r}")
    else:
        shas.setdefault(s, []).append(name)
dupes = {s: c for s, c in shas.items() if len(c) > 1}
if dupes:
    for s, cells in dupes.items():
        fail(f"duplicate adapter_sha256 {s[:12]}... shared by {cells}")
else:
    print(f"  OK    {len(shas)} unique SHAs across {len(manifests)} cells")

# 6 -------------------------------------------------------------- adapters exist
print("\n[6] adapter files on disk")
missing_files = 0
for name, m in sorted(manifests.items()):
    f = Path(m["adapter_dir"]) / "adapter_model.safetensors"
    if not f.exists():
        fail(f"{name}: adapter file not found at {f}")
        missing_files += 1
if not missing_files:
    print(f"  OK    all {len(manifests)} adapter files present")

# 7 -------------------------------------------------------------- within-model consistency
print("\n[7] trainable_params consistent across seeds")
for model in MODELS:
    vals = {n: m.get("trainable_params") for n, m in manifests.items()
            if m.get("model_key") == model}
    uniq = set(vals.values())
    if len(uniq) > 1:
        fail(f"{model}: trainable_params differs across seeds: {vals}")
    elif uniq:
        print(f"  OK    {model:<14} {next(iter(uniq)):>12,} (x{len(vals)} seeds)")

# ---------------------------------------------------------------- summary
print("\n" + "=" * 78)
if failures:
    print(f"VERIFICATION FAILED - {len(failures)} problem(s)")
    for f_ in failures:
        print(f"  - {f_}")
    raise SystemExit(1)

print(f"{'cell':<24}{'trainable':>12}{'tgt':>5}{'gate':>7}  sha256[:12]")
print("-" * 78)
for name in sorted(manifests):
    m = manifests[name]
    print(f"{name:<24}{m['trainable_params']:>12,}{len(m['lora_target_modules']):>5}"
          f"{m['masking_gate']:>7}  {m['adapter_sha256'][:12]}")

print("\n\u2705 All 12 manifests ready for Stage 6")
