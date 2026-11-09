# Environment and path repair (post-move)

**Date:** 2026-09-25  
**Repo root (new):** `D:\Thesis code files\Thesis\v3`  
**Previous root:** `D:\Research Lab\Thesis\v3`  
**Scope:** minimal fix of `finetune_env` base interpreter + five hard-coded path sites. No package reinstall/upgrade. Old reports untouched.

---

## 1. `finetune_env` base interpreter

### Backup
- Copied `finetune_env\pyvenv.cfg` → `finetune_env\pyvenv.cfg.bak` (unchanged original).

### Change (Python **3.12.3 → 3.12.7**)

| Field | Before (bak) | After |
|-------|--------------|-------|
| `home` | `C:\Program Files\Python312` (missing on this PC) | `C:\Users\paula\AppData\Local\Programs\Python\Python312` |
| `version` | `3.12.3` | `3.12.7` |
| `executable` | `C:\Program Files\Python312\python.exe` | `C:\Users\paula\AppData\Local\Programs\Python\Python312\python.exe` |
| `command` | left as historical provenance (`… -m venv D:\Research Lab\Thesis\finetune_env`) | unchanged |

No packages were reinstalled or upgraded.

### Verification

```
finetune_env\Scripts\python.exe --version
→ Python 3.12.7
```

**`pip freeze` vs `reports/stats/env_freeze.txt` (123 packages):**
- version mismatches: **0**
- missing: **0**
- extra: **0**  
(Earlier expectation of a lone `pip` delta did not appear: `pip freeze` matches the recorded freeze exactly.)

**Import / versions:**

| Package | Version |
|---------|---------|
| torch | 2.11.0+cu128 |
| transformers | 5.5.0 |
| peft | 0.20.0 |
| bitsandbytes | 0.50.0 |
| trl | 0.24.0 |
| unsloth (pkg metadata) | 2026.8.12 |

**CUDA / GPU (this machine):**
- `torch.cuda.is_available()` → **False**
- `torch.cuda.device_count()` → **0**
- GPU name → **None**
- VRAM → **None**
- `torch.version.cuda` → **12.8** (build tag; no device present)
- `import unsloth` → **fails** with `NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.`  
  Package is installed; import requires a torch accelerator. Training/eval that needs Unsloth needs a GPU on this PC (or a different machine).

---

## 2. Hard-coded `D:\Research Lab` path fixes (5 files only)

Diffs also saved at `reproduce/s6/path_fixes.diff`.

### `scripts/verify_s5_manifests.py`
```diff
-REP = Path(r"D:\Research Lab\Thesis\v3\reports\s5")
+REP = Path(__file__).resolve().parents[1] / "reports" / "s5"
```

### `scripts/s6d_eval_remaining_15.py`
```diff
-ENV_PYTHON = Path(r"D:\Research Lab\Thesis\finetune_env\Scripts\python.exe")
+ENV_PYTHON = PROJECT_ROOT.parent / "finetune_env" / "Scripts" / "python.exe"
```

### `scripts/s6e_train_clarify_all_models.py`
```diff
-ENV_PYTHON = Path(r"D:\Research Lab\Thesis\finetune_env\Scripts\python.exe")
+ENV_PYTHON = PROJECT_ROOT.parent / "finetune_env" / "Scripts" / "python.exe"
```

### `scripts/s6e_eval_clarify_models.py`
```diff
-ENV_PYTHON = Path(r"D:\Research Lab\Thesis\finetune_env\Scripts\python.exe")
+ENV_PYTHON = PROJECT_ROOT.parent / "finetune_env" / "Scripts" / "python.exe"
```

### `launch_stage5.ps1`
```diff
-$Python = "D:\Research Lab\Thesis\finetune_env\Scripts\python.exe"
+$Python = Join-Path (Split-Path $ProjectRoot -Parent) "finetune_env\Scripts\python.exe"
```

No other scripts edited. No reports edited.

---

## 3. `scripts/verify_s5_manifests.py` run

Command: `finetune_env\Scripts\python.exe scripts\verify_s5_manifests.py`  
Full transcript: `reproduce/s6/verify_s5_manifests_output.txt`  
Exit code: **1** (24 reported problems)

**What worked after the path fix:**
- Manifests resolved under the new `reports/s5` location.
- Masking gate PASS; 16 unique adapter SHAs; trainable_params consistent within models.

**Failures (expected / not introduced by this repair):**
1. **Clarify manifests treated as unexpected** — script expects exactly 12 cells (`4 models × 3 seeds`); four `*_seed42_clarify` manifests also live in `reports/s5` (schema extras: `batch_config`, `run_suffix`, `train_file`).
2. **Adapter existence check still uses absolute `adapter_dir` from old manifests** — every cell points at `D:\Research Lab\Thesis\v3\models\…`, which does not exist on this PC. Adapters are present under the new root (hashes already verified in the move audit). Reports were intentionally not rewritten.

---

## 4. CPU reproducibility — re-score `llama31_8b_seed42`

**Method:** load `reports/s6/predictions/llama31_8b_seed42.jsonl`, call current `s6_eval.score_arm` (including BERTScore), write outputs only under `reproduce/s6/`. Did not modify `reports/s6/*`.

**Artifacts:**
- `reproduce/s6/llama31_8b_seed42_results_master_rescore.csv`
- `reproduce/s6/llama31_8b_seed42_summary_rescore.json`
- `reproduce/s6/llama31_8b_seed42_rescore_compare.json`
- `reproduce/s6/rescore_run.log`

**vs `reports/s6/results_master.csv` (arm = `llama31_8b_seed42`):**
- recorded rows: 3078  
- rescored rows: 3078  
- keys only in recorded / only in rescored: **0 / 0**
- value differences: **83**, all metric = **`bertscore_f1`**
- max \|Δ\| ≈ **4.77×10⁻⁷** (float noise)
- No non-BERTScore metric differed.
- Aggregate `summarise()` vs `reports/s6/summary_by_arm.csv` for this arm: **0** metric differences (within 1e-6).

---

## 5. Open follow-ups (not done here)

- Remap or relativize `adapter_dir` in Stage 5 manifests (or teach `verify_s5_manifests.py` to resolve old prefixes) so step [6] passes without editing weights.
- Decide whether clarify manifests belong in the Stage 5 verifier’s expected set.
- GPU required for Unsloth-dependent training/eval on this machine.

---

## 6. Stop

Environment and path repair logged. Awaiting next instruction.
