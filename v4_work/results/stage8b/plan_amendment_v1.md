# Stage 8B plan amendment v1

**Written:** 2026-09-26 00:43:00 +0600  
**Status:** written **before** any Part 2 run. Frozen Step 0 files (`prompt_P.txt`, `fewshot_examples.json`, `plan.json`) are **not** edited.  
**Keeps:** `results/stage8b/hw_check.md` (the 50-row failure) unchanged as a record.

## Reason the 50-row HW check was flawed

The 50-row check called `s6_eval.generate_arm` (same Stage 6 function) but on a **stratified 50-row subset**. Under that subset, batch grouping by prompt length and right-padding for 4-bit greedy decode differ from the full 422-row run. Padding and batch neighbours can change the decoded surface text even when the weights and decoding flags are identical. Byte-identity against the original full-arm generations was therefore not a fair pass criterion for a subset re-run.

Observed on that check (kept in `hw_check.md`): classification 50/50; byte-identical only 36/50 (72%); exact_0.1 11/12. That failure is explained by batch composition, not by a demonstrated broken Stage 6 path.

## Code-path note (50-row check)

| Item | Value |
|---|---|
| Script | `v4_work/scripts/stage8b_hw_check.py` (separate wrapper) |
| Inference function | `s6_eval.generate_arm` (Stage 6), not a reimplementation |
| Difference from Stage 6 | row set = 50 stratified rows, not all 422 in Stage 6 order |

## New hardware / replication rule (replaces subset byte-identity)

1. Re-run **`llama31_8b_seed42` on all 422 rows** using the **same Stage 6 inference code path**:
   - Script: `v3/scripts/s6_eval.py`
   - Function: `generate_arm`
   - Same row order as `load_test_rows()` / Stage 6
   - Same settings as the saved arm meta: `batch_size=16`, `max_new_tokens=512`, `max_seq_len=2048` (`INFER_MAX_SEQ_LEN`), greedy
2. Write outputs under `results/stage8b/hw_replication/` (do **not** overwrite `v3/reports/s6/predictions/`).
3. **PASS** if both hold vs the saved Stage 6 arm:
   - classification agreement ≥ **99%** of 422 rows
   - each headline count within **±1 row**:
     - `exact_0.1` on the 116 single-crop numeric rows
     - `invented_value` on the 68 scorable clarify rows
     - abstain correct on the 34 seed abstain rows
4. Byte-identical count is reported for the noise floor but is **not** a pass criterion.
5. If PASS → start Part 2 as previously approved. If FAIL → stop and report.

## Noise floor

From the 50-row check and the 422-row replication, record flip rates (classification, exact_0.1) in `results/stage8b/noise_floor.md`.

## Unchanged

- Step 0 freeze SHAs and P7 Holm family of 8
- Part 1 O1/O2/O3 outputs
- Part 2 design (8 prompted arms, 422 rows, `INFER_MAX_SEQ_LEN=3072`, etc.) once HW replication PASSes
