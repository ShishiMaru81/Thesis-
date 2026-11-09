# Stage 8B noise floor

**Date:** 2026-09-26  
**Arm:** `llama31_8b_seed42`  
**Purpose:** Flip rates under re-decode, for interpreting Stage 8B prompted-arm deltas. Not a pass criterion by itself.

## Sources

| Source | Script | Inference | Row set | Batch composition |
|---|---|---|---|---|
| 50-row HW check | `v4_work/scripts/stage8b_hw_check.py` | `s6_eval.generate_arm` | 50 stratified | **Different** from Stage 6 (subset) |
| Full-arm replication | `v4_work/scripts/stage8b_hw_replication.py` | `s6_eval.generate_arm` | all 422, Stage 6 order | **Same** as Stage 6 |

## Flip rates

### Classification (answer / clarify / abstain / empty)

| Source | Compared | Same | Flips | Flip rate |
|---|---:|---:|---:|---:|
| 50-row check | 50 | 50 | 0 | **0%** |
| Full replication | 422 | 422 | 0 | **0%** |

### exact_0.1 (single-crop numeric only)

| Source | Compared | Same | Flips | Flip rate |
|---|---:|---:|---:|---:|
| 50-row check | 12 | 11 | 1 | **8.3%** (1/12) |
| Full replication | 116 | 116 | 0 | **0%** |

The single 50-row exact_0.1 flip was row `9eab0c0c-5d00-4d49-8e46-236c6c891e3b` (numeric_runtime/en): saved `False` → check `True`. Classification was unchanged (`answer`→`answer`). That row is byte-identical under the full-arm replication.

## Byte surface (noise context; not a PASS gate)

| Source | Byte-identical |
|---|---|
| 50-row check | 36/50 (72%) — expected under different padding/batch neighbours |
| Full replication | 422/422 (100%) |

## Takeaway

When batch composition matches Stage 6, this hardware reproduces the saved arm with **zero** classification and **zero** exact_0.1 flips. The 50-row exact_0.1 flip (1/12) is attributable to subset batch composition under 4-bit greedy decode, not to a broken Stage 6 path. Use the full-replication floor (0% class, 0% exact_0.1) when reading prompted-arm deltas; treat subset re-runs as upper-bound surface noise only.
