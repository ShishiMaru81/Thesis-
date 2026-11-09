# Stage 8B Part 2 gate

**Status: PASS — Part 2 authorized**

Date: 2026-09-26
Amendment: `results/stage8b/plan_amendment_v1.md`
Replication: `results/stage8b/hw_replication/replication_report.json`

## Gate results (full-arm replication)

| Criterion | Result |
|---|---|
| classification agreement | 422/422 (100%) ≥ 99% |
| exact_0.1 on 116 | 77 → 77 (±0) |
| invented_value on 68 | 68 → 68 (±0) |
| abstain on 34 seed | 34 → 34 (±0) |
| **PASS** | **YES** |

Code path: `v3/scripts/s6_eval.py` → `generate_arm`
Noise floor: `results/stage8b/noise_floor.md`

Part 2 may proceed (8 prompted arms).
