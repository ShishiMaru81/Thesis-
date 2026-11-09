# Stage 8B Part 1 — O1 / O2 / O3 (counts only; no tests)

## O1 — adversarial out-of-domain abstention (n=40)

| Model | ZS abstain % | FT seed42 abstain % | FT−ZS pp | McNemar b (ZS only) | McNemar c (FT only) |
|---|---:|---:|---:|---:|---:|
| llama31_8b | 2.5 | 80.0 | 77.5 | 0 | 31 |
| mistral_nemo | 2.5 | 80.0 | 77.5 | 0 | 31 |
| phi35_mini | 0.0 | 45.0 | 45.0 | 0 | 18 |
| phi4_mini | 0.0 | 60.0 | 60.0 | 0 | 24 |

## O2 — easy (34 seed) vs hard (40 adversarial), FT seed42

| Model | Easy abstain % | Hard abstain % | easy a / easy not | hard a / hard not |
|---|---:|---:|---:|---:|
| llama31_8b | 100.0 | 80.0 | 34 / 0 | 32 / 8 |
| mistral_nemo | 100.0 | 80.0 | 34 / 0 | 32 / 8 |
| phi35_mini | 100.0 | 45.0 | 34 / 0 | 18 / 22 |
| phi4_mini | 100.0 | 60.0 | 34 / 0 | 24 / 16 |

## O3 — conflicting_values class counts (n=20)

| Arm | answer | clarify | abstain | empty |
|---|---:|---:|---:|---:|
| llama31_8b_zeroshot | 20 | 0 | 0 | 0 |
| llama31_8b_seed42 | 20 | 0 | 0 | 0 |
| llama31_8b_seed42_clarify | 20 | 0 | 0 | 0 |
| mistral_nemo_zeroshot | 20 | 0 | 0 | 0 |
| mistral_nemo_seed42 | 20 | 0 | 0 | 0 |
| mistral_nemo_seed42_clarify | 20 | 0 | 0 | 0 |
| phi35_mini_zeroshot | 20 | 0 | 0 | 0 |
| phi35_mini_seed42 | 20 | 0 | 0 | 0 |
| phi35_mini_seed42_clarify | 17 | 3 | 0 | 0 |
| phi4_mini_zeroshot | 19 | 1 | 0 | 0 |
| phi4_mini_seed42 | 20 | 0 | 0 | 0 |
| phi4_mini_seed42_clarify | 18 | 0 | 2 | 0 |
