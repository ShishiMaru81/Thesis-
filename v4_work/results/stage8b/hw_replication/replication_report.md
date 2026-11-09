# Stage 8B full-arm replication (`llama31_8b_seed42`, n=422)

- Code path: `v3/scripts/s6_eval.py` → **`generate_arm`**
- Settings: batch_size=16, max_new_tokens=512, max_seq_len=2048, greedy
- Row order: `load_test_rows()` (Stage 6 order)
- 50-row check: separate script `stage8b_hw_check.py` calling the same `generate_arm` on a subset

## Agreement vs saved Stage 6 arm

| Metric | Value |
|---|---:|
| Byte-identical | 422/422 (100.0%) |
| Classification agree | 422/422 (100.000%) |
| exact_0.1 agree (single-crop compared) | 116/116 |

## Headline counts (saved → replication)

| Headline | Saved | Replication | Diff | Within ±1 |
|---|---:|---:|---:|:---:|
| exact_0.1 on 116 | 77 | 77 | +0 | yes |
| invented_value on 68 | 68 | 68 | +0 | yes |
| abstain on 34 seed | 34 | 34 | +0 | yes |

## Pass: **YES**

