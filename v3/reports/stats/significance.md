# Stage 8 — Statistical significance (Gate 8 report)

*Generated 2026-09-24 by `scripts/s8_stats.py --part report`; pre-registered plan `reports/stats/test_plan.json` v3 (changelog in Appendix B).*

## Scope and methods

Scope: 362 of 422 test rows scored. The 60 rows needing human-authored golds (40 out_of_domain, 20 conflicting_values) are excluded from this thesis by design. The human evaluation sample additionally contains 18 authored Bengali translations of English test items, created to form bn/en pairs; results are reported with and without them.

- **Test set:** 362 of 422 rows scored. Excluded 60 `pending_human`: {'conflicting_values': 20, 'out_of_domain': 40} (qtype: {'clarify': 20, 'abstain': 40}). SHA-256 of `test_hardened.jsonl` verified against `test_hardened.sha256`.
- **Arms:** 20 scored arms in `results_master.csv` (4 zero-shot, 12 fine-tuned = 4 models × seeds 42/1337/2024, 4 clarify-trained at seed 42) plus 1 detector prediction set.
- **Seed policy:** primary = seed 42 (matches human eval and all clarify-trained arms). Every FT-involving automatic test that has a like-for-like FT arm at seeds 1337 and 2024 (A1, A2, A6, A7) is repeated on those seeds. A claim is 'robust' only if significant after Holm on all three seeds. Clarify-trained arms exist at seed 42 only; A3 and A4 have no seed-robustness check.
- **Multiple comparisons:** Holm-Bonferroni across PRIMARY tests within each holm_family. SECONDARY and SENSITIVITY report raw p only. DESCRIPTIVE report bootstrap 95% CIs, no p. PRIMARY tests: **31 automatic (seed 42) + 29 human = 60**. Every A-family PRIMARY test with a like-for-like FT arm is repeated at seeds 1337 and 2024 (Table 4).
- **Exact-match denominators:** Two exact-match denominators are in use and are named wherever either figure appears: Stage 8 correct_0p1 = all 146 numeric rows (multi-crop rows counted correct only if every gold value is within 0.1 mm; rows with no extracted depth counted wrong); Stage 6 headline exact_0.1 = the 116 single-crop rows. No test was changed; both are tabulated per arm. (Table 8.)
- **Human sample:** 200 questions × 4 arms = 800 judgments; 100 judgments rated by all 5 annotators, 700 by one. PRIMARY = the 182 questions that are locked test rows; SENSITIVITY = all 200 (adds 18 authored Bengali translations: 15 clarify, 3 numeric). Per-judgment `mean_rating` = mean of its raters; `farmer_yes` = majority vote.
- **Mixed-model method actually used (B4):** Python `statsmodels` fallback — MixedLM (rating ~ arm, groups = annotator, variance component for row_id) and BinomialBayesMixedGLM (variational Bayes, crossed annotator + row_id). `Rscript` is not on PATH and R was not installed. MixedLM treats the 1–5 rating as interval; the Bayesian model yields credible intervals, not p-values. B4 is supporting, not primary.
- **Agreement (100 shared judgments × 5 raters):** would_give_farmer Fleiss κ = 0.647 [0.542, 0.740]; ordinal Krippendorff α = 0.721 [0.635, 0.787]; nominal Fleiss κ on 1–5 = 0.392. **Decision recorded:** the primary human measures are `would_give_farmer` and ordinal α. Nominal κ treats a 4-vs-5 disagreement as a full disagreement and so understates agreement on an ordinal scale; it is reported for completeness (Table 7). All three audit values reproduced within 0.0005; package and independent implementations agree to 4 dp.
- **Environment:** Python 3.12.3, numpy 2.4.4, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.0, krippendorff (installed), pandas 3.0.5; R not available. Full freeze: `reports/stats/env_freeze.txt`. Seed 42 throughout; 10,000 bootstrap resamples.
- **Verdict wording:** non-significant results are written as *no significant difference detected* together with the 95% CI of the difference, so the reader can see how large a difference is ruled out. *Equivalent* and a bare *no difference* are never used.

## Table 1 — Hypothesis summary

| ID | Claim | Tests | Verdict | Key numbers |
|---|---|---|---|---|
| H1 | Fine-tuning beats zero-shot | A1, B2-i, B3-i | **Supported** | A1: all 4 models p_holm < .001 at every seed (robust on 3 seeds); Δcorrect_0p1 (/146) +29 to +52 pp. Human: rating r_rb = 0.76 [0.65, 0.85], farmer_yes +31.9 [+24.7, +39.0] pp (n = 182), both p_holm < .001. |
| H2 | Bengali vs English | A6, B5 | **Partially supported** | Factual chrF++ (automatic, paired on fact_id, n = 57): English > Bengali in 3 of 4 models at seed 42 (Mistral-NeMo-12B, Phi-3.5-mini, Phi-4-mini); robust on all seeds in 2 of 4 (Phi-3.5-mini, Phi-4-mini); Llama not significant. Numeric (unpaired Fisher, 73 vs 73): no significant difference detected in any model or seed; CI half-widths are ±15–16 pp, too wide to rule out gaps of about 15 points. Human (B5, 100 pairs): none of 8 tests significant after Holm; point estimates lean English in all three fine-tuned arms. Exploratory: Mistral zero-shot answered only 30% of Bengali questions in Bengali (§H2 note). |
| H3 | Human agreement | B6 | **Supported on the pre-specified primary measures** | farmer κ = 0.647, ordinal α = 0.721; nominal κ = 0.392 reported and explained (Table 7). |
| T | Fine-tuning converts invented values into confident errors (thesis title) | A7 | **Supported in all four models on all seeds** | Registered (a): FT − ZS = Llama-3.1-8B +53 pp, Mistral-NeMo-12B +16 pp, Phi-3.5-mini +68 pp, Phi-4-mini +24 pp; all p_holm ≤ .001. Size of change depends on the base model: zero-shot baselines Llama 46%, Mistral 84%, Phi-3.5 31%, Phi-4 75%; every FT arm ≥ 98.5%. Strict version (b), which matches the thesis definition, is stronger (§A7). |
| C1 | Clarify training teaches clarification | A3a, B3-ii | **Supported** | A3a: clarify_exact 0% → 29–38% in all 4 models, p_holm < .001. Human B3-ii clarify (n = 25): farmer_yes +56.0 [+36.0, +76.0] pp, p_holm < .001. |
| C2 | Clarify training reduces invented values | A3b | **Supported** | A3b: invented_value 100% → 50–66% in all 4 models, p_holm < .001 (Δ −34 to −50 pp). |
| C3 | Clarify training does not harm numeric accuracy | A4 | **No loss detected** | correct_0p1 (/146), clarify-FT − FT: Llama-3.1-8B +1.4 pp [-5.5, +8.2]; Mistral-NeMo-12B +2.1 pp [-5.5, +9.6]; Phi-3.5-mini -2.7 pp [-8.2, +2.7]; Phi-4-mini +0.7 pp [-6.2, +7.5]; no test significant. Phi-3.5 (flagged) is the only negative point estimate; a loss larger than 8.2 pp is ruled out at 95%. |
| D | Detector adds safety | A5 | **Descriptive only; oracle-gated upper bound** | Safe handling 36.8% → 83.8%, risk difference +47.1 [+35.3, +58.8] pp; b = 32, c = 0 by construction. Oracle-gated diagnostic. The detector reads gold labels, runs only on known-incomplete queries, and its rules were written with the test categories in view. This is an upper bound on what surface rules can catch, not a performance estimate. |
| M | Llama FT vs Mistral FT | A2, B2-iii, B3-iii, B4 | **No significant difference detected** | A2 McNemar OR 0.88 [0.46, 1.64], accuracy Δ -2.1 [-11.0, +6.8] pp, non-significant at all 3 seeds; human rating r_rb -0.12 [-0.34, 0.10], farmer_yes -2.7 [-9.3, +3.8] pp; annotator-adjusted B4 unchanged in direction and significance. Friedman shows the four FT models do differ (W = 0.09); the tier contrast {Llama, Mistral} > {Phi} is significant and robust. |

## Table 2 — Automatic results (Part A), seed 42, grouped by family

Paired by `row_id` unless stated. Effects: McNemar odds ratio = b/c (b = rows where the treatment arm scores 1 and the baseline 0); Wilcoxon rank-biserial r; Fisher OR = odds(en)/odds(bn). Differences are treatment minus baseline: FT − ZS (A1, A7), Mistral − Llama (A2), clarify-FT − FT (A3, A4), hybrid − model-only (A5), en − bn (A6). SENSITIVITY rows carry raw p only; their verdict strings are raw-p based.

### A1 — Fine-tuning helps (zero-shot vs FT, numeric rows, correct_0p1 over 146)

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A1_llama31_8b | 146 | 76/0 | McNemar exact on correct_0p1 | 0.000 | <0.001 | <0.001 | inf [20.11, inf] | +52.1 [+43.8, +60.3] pp | supported: Llama FT > Llama ZS, significant after Holm; CI [+43.8, +60.3] pp |
| A1_mistral_nemo | 146 | 71/1 | McNemar exact on correct_0p1 | 1.000 | <0.001 | <0.001 | 71.00 [12.34, 2843.35] | +47.9 [+39.7, +56.2] pp | supported: Mistral FT > Mistral ZS, significant after Holm; CI [+39.7, +56.2] pp |
| A1_phi35_mini | 146 | 43/1 | McNemar exact on correct_0p1 | 1.000 | <0.001 | <0.001 | 43.00 [7.32, 1737.41] | +28.8 [+21.2, +36.3] pp | supported: Phi-3.5 FT > Phi-3.5 ZS, significant after Holm; CI [+21.2, +36.3] pp |
| A1_phi4_mini | 146 | 49/0 | McNemar exact on correct_0p1 | 0.000 | <0.001 | <0.001 | inf [12.79, inf] | +33.6 [+26.0, +41.1] pp | supported: Phi-4 FT > Phi-4 ZS, significant after Holm; CI [+26.0, +41.1] pp |

### A2 — Model comparison after fine-tuning (numeric rows, correct_0p1 over 146)

> Pre-registered null expectation for Llama FT vs Mistral FT. No pairwise post-hoc tests were planned or run.

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A2_friedman | 146 | — | Friedman on correct_0p1 | 38.418 | <0.001 | <0.001 | 0.09 [0.05, 0.15] | — | significant (arms > each other); CI Kendall's W [0.05, 0.15] |
| A2_llama_vs_mistral | 146 | 21/24 | McNemar exact on correct_0p1 | 21.000 | 0.766 | 0.766 | 0.88 [0.46, 1.64] | -2.1 [-11.0, +6.8] pp | no significant difference detected; consistent with the pre-registered null expectation; 95% CI of the difference [-11.0, +6.8] pp |
| A2_tier | 146 | — | Wilcoxon signed-rank | 277.000 | <0.001 | <0.001 | 0.73 [0.55, 0.89] | +19.9 [+13.4, +26.4] pp | supported: tier {Llama,Mistral} > tier {Phi-3.5,Phi-4}, significant after Holm; CI rank-biserial r [0.55, 0.89]; mean diff [+13.4, +26.4] pp |

### A3a — Clarify training teaches clarification (clarify rows, n = 68)

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A3a_llama31_8b | 68 | 23/0 | McNemar exact on clarify_exact | 0.000 | <0.001 | <0.001 | inf [5.75, inf] | +33.8 [+22.1, +45.6] pp | supported: Llama clarify-FT > Llama FT, significant after Holm; CI [+22.1, +45.6] pp |
| A3a_mistral_nemo | 68 | 26/0 | McNemar exact on clarify_exact | 0.000 | <0.001 | <0.001 | inf [6.56, inf] | +38.2 [+26.5, +50.0] pp | supported: Mistral clarify-FT > Mistral FT, significant after Holm; CI [+26.5, +50.0] pp |
| A3a_phi35_mini | 68 | 20/0 | McNemar exact on clarify_exact | 0.000 | <0.001 | <0.001 | inf [4.94, inf] | +29.4 [+19.1, +39.7] pp | supported: Phi-3.5 clarify-FT > Phi-3.5 FT, significant after Holm; CI [+19.1, +39.7] pp |
| A3a_phi4_mini | 68 | 26/0 | McNemar exact on clarify_exact | 0.000 | <0.001 | <0.001 | inf [6.56, inf] | +38.2 [+26.5, +50.0] pp | supported: Phi-4 clarify-FT > Phi-4 FT, significant after Holm; CI [+26.5, +50.0] pp |

### A3b — Clarify training reduces invented values (clarify rows, n = 68)

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A3b_llama31_8b | 68 | 0/25 | McNemar exact on invented_value | 0.000 | <0.001 | <0.001 | 0.00 [0.00, 0.16] | -36.8 [-48.5, -25.0] pp | supported: Llama clarify-FT < Llama FT, significant after Holm; CI [-48.5, -25.0] pp |
| A3b_mistral_nemo | 68 | 0/34 | McNemar exact on invented_value | 0.000 | <0.001 | <0.001 | 0.00 [0.00, 0.11] | -50.0 [-61.8, -38.2] pp | supported: Mistral clarify-FT < Mistral FT, significant after Holm; CI [-61.8, -38.2] pp |
| A3b_phi35_mini | 68 | 0/23 | McNemar exact on invented_value | 0.000 | <0.001 | <0.001 | 0.00 [0.00, 0.17] | -33.8 [-45.6, -22.1] pp | supported: Phi-3.5 clarify-FT < Phi-3.5 FT, significant after Holm; CI [-45.6, -22.1] pp |
| A3b_phi4_mini | 68 | 0/28 | McNemar exact on invented_value | 0.000 | <0.001 | <0.001 | 0.00 [0.00, 0.14] | -41.2 [-52.9, -29.4] pp | supported: Phi-4 clarify-FT < Phi-4 FT, significant after Holm; CI [-52.9, -29.4] pp |

### A4 — Clarify training does not harm numeric accuracy (numeric rows, correct_0p1 over 146)

> Hypothesis is *no harm*; the CI of the accuracy difference is the result. Phi-3.5 was flagged in advance (Stage 6E slot-F1 drop). Clarify-trained arms exist at seed 42 only; A3 and A4 have no seed-robustness check.

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A4_llama31_8b | 146 | 14/12 | McNemar exact on correct_0p1 | 12.000 | 0.845 | 1.000 | 1.17 [0.50, 2.76] | +1.4 [-5.5, +8.2] pp | no significant difference detected; consistent with the pre-registered null expectation; 95% CI of the difference [-5.5, +8.2] pp |
| A4_mistral_nemo | 146 | 17/14 | McNemar exact on correct_0p1 | 14.000 | 0.720 | 1.000 | 1.21 [0.56, 2.66] | +2.1 [-5.5, +9.6] pp | no significant difference detected; consistent with the pre-registered null expectation; 95% CI of the difference [-5.5, +9.6] pp |
| A4_phi35_mini | 146 | 7/11 | McNemar exact on correct_0p1 | 7.000 | 0.481 | 1.000 | 0.64 [0.21, 1.80] | -2.7 [-8.2, +2.7] pp | no significant difference detected; consistent with the pre-registered null expectation; 95% CI of the difference [-8.2, +2.7] pp |
| A4_phi4_mini | 146 | 13/12 | McNemar exact on correct_0p1 | 12.000 | 1.000 | 1.000 | 1.08 [0.46, 2.60] | +0.7 [-6.2, +7.5] pp | no significant difference detected; consistent with the pre-registered null expectation; 95% CI of the difference [-6.2, +7.5] pp |

### A5 — Detector (DESCRIPTIVE, plan v3)

> Oracle-gated diagnostic. The detector reads gold labels, runs only on known-incomplete queries, and its rules were written with the test categories in view. This is an upper bound on what surface rules can catch, not a performance estimate. Rules and the test categories that informed each are in Appendix A.

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A5_detector | 68 | 32/0 | DESCRIPTIVE: risk difference, paired bootstrap (no test; plan v3) | — | — | — | 0.47 [0.35, 0.59] | +47.1 [+35.3, +58.8] pp | DESCRIPTIVE (oracle-gated diagnostic): safe handling 36.8% -> 83.8%, risk difference [+35.3, +58.8] pp; b=32, c=0 by construction; no test |

### A6 — Bengali vs English, automatic (H2)

> Pairing key is `fact_id` (the locked test set has no `pair_id`): 57 factual pairs, 17 numeric pairs, 0 clarify pairs. Numeric PRIMARY is therefore the unpaired Fisher test on 73 vs 73 rows; the 17-pair McNemar is SECONDARY and underpowered.

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A6_num_unpaired_llama31_8b | 146 | — | Fisher exact on correct_0p1 | 1.057 | 1.000 | 1.000 | 1.06 [0.55, 2.02] | +1.4 [-15.1, +17.8] pp | no significant difference detected; 95% CI of the difference [-15.1, +17.8] pp |
| A6_num_paired_llama31_8b | 17 | 1/1 | McNemar exact on correct_0p1 | 1.000 | 1.000 | — | 1.00 [0.01, 78.50] | +0.0 [-17.6, +17.6] pp | [SECONDARY, raw p only] no significant difference detected; 95% CI of the difference [-17.6, +17.6] pp |
| A6_fac_llama31_8b | 57 | — | Wilcoxon signed-rank | 590.000 | 0.132 | 0.526 | 0.23 [-0.07, 0.52] | +6.27 [+1.06, +11.96] chrF++ pts | no significant difference detected; 95% CI of the difference rank-biserial r [-0.07, 0.52]; mean diff [1.06, 11.96] metric points |
| A6_num_unpaired_mistral_nemo | 146 | — | Fisher exact on correct_0p1 | 1.000 | 1.000 | 1.000 | 1.00 [0.52, 1.91] | +0.0 [-16.4, +16.4] pp | no significant difference detected; 95% CI of the difference [-16.4, +16.4] pp |
| A6_num_paired_mistral_nemo | 17 | 0/1 | McNemar exact on correct_0p1 | 0.000 | 1.000 | — | 0.00 [0.00, 39.00] | -5.9 [-17.6, +0.0] pp | [SECONDARY, raw p only] no significant difference detected; 95% CI of the difference [-17.6, +0.0] pp |
| A6_fac_mistral_nemo | 57 | — | Wilcoxon signed-rank | 354.000 | <0.001 | 0.002 | 0.56 [0.29, 0.78] | +11.11 [+5.07, +17.62] chrF++ pts | supported: en > bn, significant after Holm; CI rank-biserial r [0.29, 0.78]; mean diff [5.07, 17.62] metric points |
| A6_num_unpaired_phi35_mini | 146 | — | Fisher exact on correct_0p1 | 1.934 | 0.104 | 0.520 | 1.93 [0.94, 3.98] | +13.7 [-1.4, +28.8] pp | no significant difference detected; 95% CI of the difference [-1.4, +28.8] pp |
| A6_num_paired_phi35_mini | 17 | 3/1 | McNemar exact on correct_0p1 | 1.000 | 0.625 | — | 3.00 [0.24, 157.49] | +11.8 [-11.8, +35.3] pp | [SECONDARY, raw p only] no significant difference detected; 95% CI of the difference [-11.8, +35.3] pp |
| A6_fac_phi35_mini | 57 | — | Wilcoxon signed-rank | 354.000 | <0.001 | 0.001 | 0.57 [0.31, 0.80] | +7.36 [+2.31, +12.78] chrF++ pts | supported: en > bn, significant after Holm; CI rank-biserial r [0.31, 0.80]; mean diff [2.31, 12.78] metric points |
| A6_num_unpaired_phi4_mini | 146 | — | Fisher exact on correct_0p1 | 1.203 | 0.726 | 1.000 | 1.20 [0.60, 2.39] | +4.1 [-11.0, +19.2] pp | no significant difference detected; 95% CI of the difference [-11.0, +19.2] pp |
| A6_num_paired_phi4_mini | 17 | 2/2 | McNemar exact on correct_0p1 | 2.000 | 1.000 | — | 1.00 [0.07, 13.80] | +0.0 [-23.5, +23.5] pp | [SECONDARY, raw p only] no significant difference detected; 95% CI of the difference [-23.5, +23.5] pp |
| A6_fac_phi4_mini | 57 | — | Wilcoxon signed-rank | 322.000 | <0.001 | <0.001 | 0.61 [0.37, 0.81] | +12.84 [+6.60, +19.83] chrF++ pts | supported: en > bn, significant after Holm; CI rank-biserial r [0.37, 0.81]; mean diff [6.60, 19.83] metric points |

**Llama factual.** The registered Wilcoxon signed-rank is not significant (p_raw = 0.132, p_holm = 0.526; rank-biserial r = 0.23 [-0.07, 0.52]), while the paired-bootstrap 95% CI of the *mean* chrF++ difference, +6.27 [+1.06, +11.96] chrF++ pts, excludes zero. The paired differences are skewed (a few English rows gain a lot; most pairs are close to zero), which pulls the mean but not the ranks. The registered test's verdict stands: no significant difference detected for Llama.

**Numeric.** No significant difference detected in any model at any seed. The CIs of the en − bn accuracy difference have half-widths of ±15–16 pp (e.g. Llama +1.4 [-15.1, +17.8] pp); they are too wide to rule out gaps of about 15 points in either direction. **Pattern check:** numeric shows no consistent direction (as expected from Stage 6); factual favours English significantly in 3 of 4 models, not in all four as Stage 6 suggested.

**Clarify (descriptive; 0 valid pairs).** bn vs en rates per arm with bootstrap CIs:

| arm | metric | bn (n = 34) | en (n = 34) |
|---|---|---|---|
| llama31_8b_seed42 | clarify_exact | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| llama31_8b_seed42 | invented_value | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| mistral_nemo_seed42 | clarify_exact | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| mistral_nemo_seed42 | invented_value | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| phi35_mini_seed42 | clarify_exact | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| phi35_mini_seed42 | invented_value | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| phi4_mini_seed42 | clarify_exact | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| phi4_mini_seed42 | invented_value | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| llama31_8b_seed42_clarify | clarify_exact | 0.29 [0.15, 0.44] | 0.38 [0.24, 0.56] |
| llama31_8b_seed42_clarify | invented_value | 0.65 [0.47, 0.79] | 0.62 [0.44, 0.76] |
| mistral_nemo_seed42_clarify | clarify_exact | 0.35 [0.21, 0.50] | 0.41 [0.24, 0.59] |
| mistral_nemo_seed42_clarify | invented_value | 0.47 [0.29, 0.65] | 0.53 [0.35, 0.71] |
| phi35_mini_seed42_clarify | clarify_exact | 0.18 [0.06, 0.32] | 0.41 [0.26, 0.59] |
| phi35_mini_seed42_clarify | invented_value | 0.82 [0.68, 0.94] | 0.50 [0.32, 0.68] |
| phi4_mini_seed42_clarify | clarify_exact | 0.29 [0.15, 0.44] | 0.47 [0.29, 0.65] |
| phi4_mini_seed42_clarify | invented_value | 0.68 [0.50, 0.82] | 0.50 [0.32, 0.68] |

### A7 — Confident error (clarify rows, n = 68): zero-shot vs FT

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A7_llama31_8b | 68 | 36/0 | McNemar exact on confident_error | 0.000 | <0.001 | <0.001 | inf [9.27, inf] | +52.9 [+41.2, +64.7] pp | supported: Llama FT > Llama ZS, significant after Holm; CI [+41.2, +64.7] pp |
| A7_mistral_nemo | 68 | 11/0 | McNemar exact on confident_error | 0.000 | <0.001 | <0.001 | inf [2.51, inf] | +16.2 [+7.4, +25.0] pp | supported: Mistral FT > Mistral ZS, significant after Holm; CI [+7.4, +25.0] pp |
| A7_phi35_mini | 68 | 46/0 | McNemar exact on confident_error | 0.000 | <0.001 | <0.001 | inf [11.98, inf] | +67.6 [+55.9, +77.9] pp | supported: Phi-3.5 FT > Phi-3.5 ZS, significant after Holm; CI [+55.9, +77.9] pp |
| A7_phi4_mini | 68 | 17/1 | McNemar exact on confident_error | 1.000 | <0.001 | <0.001 | 17.00 [2.66, 710.46] | +23.5 [+13.2, +35.3] pp | supported: Phi-4 FT > Phi-4 ZS, significant after Holm; CI [+13.2, +35.3] pp |

### A7 — detail

**Definition (registered, primary (a)).** confident_error = 1 iff (a) s6_eval.classify_response(generation) == 'answer' AND (b) not truncated: generated tokens (re-tokenised with the model's own tokenizer; no finish_reason stored) < 511 AND (c) net_extracted == 1. Definition motivated by Stage 6 observations (zero-shot hitting the 512-token cap); confirmatory of an observed pattern, not a blind prediction. decode->re-encode is not perfectly idempotent; truncated generations re-tokenise to 509-513 tokens; the threshold ≥ 511 was kept as registered. Rows at 505–510 tokens treated as not truncated: {'llama31_8b': 0, 'mistral_nemo': 0, 'phi35_mini': 6, 'phi4_mini': 5}; any truncated row counted as not-truncated can only raise zero-shot confident_error, i.e. shrink the FT-minus-ZS difference tested in A7 (conservative for the pre-registered direction).

**Registered primary (a) — reported first.** All four McNemar tests are significant after Holm at all three seeds (Table 4). The size of the change depends on the base model: Mistral and Phi-4 already produce confident errors on most clarify rows zero-shot, so fine-tuning adds 16 and 24 pp; Llama and Phi-3.5 mostly ran into the token cap or hedged zero-shot, so fine-tuning adds 53 and 68 pp. Mistral's zero-shot baseline is partly English-language output: it answered only 30% of Bengali questions in Bengali (Table 9), and its English answers carry the hedges the marker list detects.

**Sensitivity (b) — the definition that matches the thesis.** (b) additionally requires `explicit_assumption = 0`: a confident error is a complete numeric answer that neither hits the token cap nor states that a value was assumed. This is the thesis definition of a confident error, and it is *stronger* than (a) in every model because hedged zero-shot answers drop out of the zero-shot count while no fine-tuned answer hedges:

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A7_sens_llama31_8b | 68 | 56/0 | McNemar exact on confident_error_strict | 0.000 | <0.001 | — | inf [14.69, inf] | +82.4 [+72.1, +91.2] pp | [SENSITIVITY, raw p only] supported: Llama FT > Llama ZS, significant after Holm; CI [+72.1, +91.2] pp |
| A7_sens_mistral_nemo | 68 | 33/0 | McNemar exact on confident_error_strict | 0.000 | <0.001 | — | inf [8.46, inf] | +48.5 [+36.8, +60.3] pp | [SENSITIVITY, raw p only] supported: Mistral FT > Mistral ZS, significant after Holm; CI [+36.8, +60.3] pp |
| A7_sens_phi35_mini | 68 | 52/0 | McNemar exact on confident_error_strict | 0.000 | <0.001 | — | inf [13.60, inf] | +76.5 [+66.2, +86.8] pp | [SENSITIVITY, raw p only] supported: Phi-3.5 FT > Phi-3.5 ZS, significant after Holm; CI [+66.2, +86.8] pp |
| A7_sens_phi4_mini | 68 | 26/1 | McNemar exact on confident_error_strict | 1.000 | <0.001 | — | 26.00 [4.27, 1065.94] | +36.8 [+25.0, +48.5] pp | [SENSITIVITY, raw p only] supported: Phi-4 FT > Phi-4 ZS, significant after Holm; CI [+25.0, +48.5] pp |

**Descriptive per model (clarify rows, n = 68; bootstrap 95% CIs).** conf_err (a) = registered; (b) = strict.

| model | variant | truncation rate | median tokens | invented_value | explicit_assumption among invented | conf_err (a) | conf_err (b) strict |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | zeroshot | 0.54 [0.43, 0.66] | 512 | 1.00 [1.00, 1.00] | 24/68 = 0.35 [0.24, 0.47] | 0.46 [0.34, 0.57] | 0.16 [0.07, 0.25] |
| Llama-3.1-8B | ft_seed42 | 0.00 [0.00, 0.00] | 96 | 1.00 [1.00, 1.00] | 0/68 = 0.00 [0.00, 0.00] | 0.99 [0.96, 1.00] | 0.99 [0.96, 1.00] |
| Mistral-NeMo-12B | zeroshot | 0.13 [0.06, 0.22] | 384 | 0.97 [0.93, 1.00] | 28/66 = 0.42 [0.30, 0.55] | 0.84 [0.75, 0.93] | 0.51 [0.40, 0.63] |
| Mistral-NeMo-12B | ft_seed42 | 0.00 [0.00, 0.00] | 68 | 1.00 [1.00, 1.00] | 0/68 = 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Phi-3.5-mini | zeroshot | 0.66 [0.54, 0.76] | 512 | 0.99 [0.96, 1.00] | 14/67 = 0.21 [0.12, 0.31] | 0.31 [0.21, 0.43] | 0.22 [0.13, 0.32] |
| Phi-3.5-mini | ft_seed42 | 0.00 [0.00, 0.00] | 70 | 1.00 [1.00, 1.00] | 0/68 = 0.00 [0.00, 0.00] | 0.99 [0.96, 1.00] | 0.99 [0.96, 1.00] |
| Phi-4-mini | zeroshot | 0.16 [0.07, 0.25] | 326 | 0.99 [0.96, 1.00] | 10/67 = 0.15 [0.07, 0.24] | 0.75 [0.65, 0.85] | 0.62 [0.50, 0.74] |
| Phi-4-mini | ft_seed42 | 0.00 [0.00, 0.00] | 58 | 1.00 [1.00, 1.00] | 0/68 = 0.00 [0.00, 0.00] | 0.99 [0.96, 1.00] | 0.99 [0.96, 1.00] |

**Explicit-assumption markers.** English: ['assum', 'suppos', "let's say", 'let us say', 'if we take', 'taking a typical', 'typical value', 'not provided', 'not given', 'not mentioned']; Bengali: ['ধরে', 'ধরা যাক', 'অনুমান', 'দেওয়া নেই', 'উল্লেখ নেই']; excluded: ['about', 'approximately', 'প্রায়']. 20 example generations (5 per model, all zero-shot — no fine-tuned output carries a marker) are in `a7_explicit_assumption_examples.json`; all 20 hit `assum`, one also `not provided`.

**Bengali hedge check (exploratory, added before this report; no change to the registered definition).** The registered Bengali list missed phrasings used by zero-shot Bengali answers: ['জানতে হবে', 'জানা প্রয়োজন', 'জানাতে হবে', 'জানান', 'প্রদান করেননি', 'প্রদান করা হয়নি', 'উল্লেখ করেননি', 'উল্লেখ করা হয়নি'] … Searching the 34 Bengali clarify outputs per arm:

| arm | outputs in Bengali script /34 | registered BN hits | unlisted hedge hits | phrases | strict (b) rate, bn rows, registered | strict (b) rate, bn rows, extended (exploratory) |
|---|---|---|---|---|---|---|
| llama31_8b_zeroshot | 34 | 1 | 7 | {'যদি': 5, 'জানতে হবে': 3} | 0.00 | 0.00 |
| llama31_8b_seed42 | 34 | 0 | 0 | — | 0.97 | 0.97 |
| mistral_nemo_zeroshot | 1 | 0 | 0 | — | 0.38 | 0.38 |
| mistral_nemo_seed42 | 34 | 0 | 0 | — | 1.00 | 1.00 |
| phi35_mini_zeroshot | 34 | 0 | 0 | — | 0.18 | 0.18 |
| phi35_mini_seed42 | 34 | 0 | 0 | — | 0.97 | 0.97 |
| phi4_mini_zeroshot | 34 | 0 | 12 | {'যদি': 8, 'সাধারণত': 6, 'জানান': 1, 'প্রদান করেননি': 1} | 0.62 | 0.47 |
| phi4_mini_seed42 | 34 | 0 | 0 | — | 0.97 | 0.97 |

Every fine-tuned Bengali clarify output (0/34 in each of the four models) contains no hedge under either list; 40 hand-read FT outputs (10 per model) are all the bare templated line `নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি = … মিমি। <area> … লিটার।` with an invented area (e.g. "৫.০ একর", "১ বর্গকিলোমিটার"). The under-detection therefore biases only zero-shot rows, in the conservative direction; the adjusted strict rate changes only for Phi-4 zero-shot Bengali rows (0.62 → 0.47) and leaves every FT rate unchanged.

## Table 3 — Human results (Part B), PRIMARY sample (182 test-set questions), grouped by family

Per-judgment `mean_rating` (1–5) and `farmer_yes` (majority vote). Wilcoxon signed-rank and McNemar exact, paired by `row_id`. Clarify n = 25 in the primary sample and 40 in the sensitivity sample.

### B1 — Omnibus (Holm family of 2)

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| B1_friedman_test | 182 | — | Friedman on mean_rating | 117.142 | <0.001 | <0.001 | 0.21 [0.16, 0.28] | — | significant (arms > each other); CI Kendall's W [0.16, 0.28] |
| B1_cochranq_test | 182 | — | Cochran's Q on farmer_yes | 113.735 | <0.001 | <0.001 | — | — | significant (arms > each other); CI  |

### B2 — Contrasts on mean_rating (9 cells, Holm within family)

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| B2_i_overall_test | 182 | — | Wilcoxon signed-rank | 1408.500 | <0.001 | <0.001 | 0.76 [0.65, 0.85] | +1.17 [+0.95, +1.39] rating pts | supported: Llama FT > Llama ZS, significant after Holm; CI rank-biserial r [0.65, 0.85]; mean diff [0.95, 1.39] rating points (1-5) |
| B2_i_numeric_runtime_test | 67 | — | Wilcoxon signed-rank | 168.500 | <0.001 | <0.001 | 0.85 [0.72, 0.94] | +1.56 [+1.21, +1.90] rating pts | supported: Llama FT > Llama ZS, significant after Holm; CI rank-biserial r [0.72, 0.94]; mean diff [1.21, 1.90] rating points (1-5) |
| B2_i_factual_test | 70 | — | Wilcoxon signed-rank | 261.000 | <0.001 | <0.001 | 0.71 [0.52, 0.87] | +0.96 [+0.64, +1.30] rating pts | supported: Llama FT > Llama ZS, significant after Holm; CI rank-biserial r [0.52, 0.87]; mean diff [0.64, 1.30] rating points (1-5) |
| B2_ii_overall_test | 182 | — | Wilcoxon signed-rank | 2175.000 | 0.026 | 0.128 | 0.25 [0.04, 0.45] | +0.29 [+0.08, +0.50] rating pts | no significant difference detected; 95% CI of the difference rank-biserial r [0.04, 0.45]; mean diff [0.08, 0.50] rating points (1-5) |
| B2_ii_clarify_test | 25 | — | Wilcoxon signed-rank | 3.500 | <0.001 | 0.002 | 0.96 [0.81, 1.00] | +2.06 [+1.34, +2.74] rating pts | supported: Llama FT+clarify > Llama FT, significant after Holm; CI rank-biserial r [0.81, 1.00]; mean diff [1.34, 2.74] rating points (1-5) |
| B2_ii_numeric_runtime_test | 67 | — | Wilcoxon signed-rank | 423.000 | 0.543 | 1.000 | 0.11 [-0.24, 0.45] | +0.07 [-0.26, +0.39] rating pts | no significant difference detected; 95% CI of the difference rank-biserial r [-0.24, 0.45]; mean diff [-0.26, 0.39] rating points (1-5) |
| B2_iii_overall_test | 182 | — | Wilcoxon signed-rank | 2362.500 | 0.296 | 0.888 | -0.12 [-0.34, 0.10] | -0.08 [-0.27, +0.10] rating pts | no significant difference detected; consistent with the pre-registered null expectation; 95% CI of the difference rank-biserial r [-0.34, 0.10]; mean diff [-0.27, 0.10] rating points (1-5) |
| B2_iii_numeric_runtime_test | 67 | — | Wilcoxon signed-rank | 450.500 | 0.990 | 1.000 | 0.00 [-0.35, 0.35] | +0.05 [-0.27, +0.37] rating pts | no significant difference detected; consistent with the pre-registered null expectation; 95% CI of the difference rank-biserial r [-0.35, 0.35]; mean diff [-0.27, 0.37] rating points (1-5) |
| B2_iii_factual_test | 70 | — | Wilcoxon signed-rank | 503.000 | 0.088 | 0.351 | -0.27 [-0.56, 0.03] | -0.32 [-0.66, +0.04] rating pts | no significant difference detected; consistent with the pre-registered null expectation; 95% CI of the difference rank-biserial r [-0.56, 0.03]; mean diff [-0.66, 0.04] rating points (1-5) |

### B3 — Contrasts on farmer_yes (9 cells, Holm within family)

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| B3_i_overall_test | 182 | 61/3 | McNemar exact on farmer_yes | 3.000 | <0.001 | <0.001 | 20.33 [6.64, 101.32] | +31.9 [+24.7, +39.0] pp | supported: Llama FT > Llama ZS, significant after Holm; CI [+24.7, +39.0] pp |
| B3_i_numeric_runtime_test | 67 | 28/1 | McNemar exact on farmer_yes | 1.000 | <0.001 | <0.001 | 28.00 [4.63, 1144.94] | +40.3 [+28.4, +52.2] pp | supported: Llama FT > Llama ZS, significant after Holm; CI [+28.4, +52.2] pp |
| B3_i_factual_test | 70 | 22/2 | McNemar exact on farmer_yes | 2.000 | <0.001 | <0.001 | 11.00 [2.70, 96.50] | +28.6 [+17.1, +40.0] pp | supported: Llama FT > Llama ZS, significant after Holm; CI [+17.1, +40.0] pp |
| B3_ii_overall_test | 182 | 35/15 | McNemar exact on farmer_yes | 15.000 | 0.007 | 0.033 | 2.33 [1.24, 4.60] | +11.0 [+3.8, +18.1] pp | supported: Llama FT+clarify > Llama FT, significant after Holm; CI [+3.8, +18.1] pp |
| B3_ii_clarify_test | 25 | 14/0 | McNemar exact on farmer_yes | 0.000 | <0.001 | <0.001 | inf [3.32, inf] | +56.0 [+36.0, +76.0] pp | supported: Llama FT+clarify > Llama FT, significant after Holm; CI [+36.0, +76.0] pp |
| B3_ii_numeric_runtime_test | 67 | 12/5 | McNemar exact on farmer_yes | 5.000 | 0.143 | 0.574 | 2.40 [0.79, 8.70] | +10.4 [-1.5, +22.4] pp | no significant difference detected; 95% CI of the difference [-1.5, +22.4] pp |
| B3_iii_overall_test | 182 | 17/22 | McNemar exact on farmer_yes | 17.000 | 0.522 | 1.000 | 0.77 [0.39, 1.52] | -2.7 [-9.3, +3.8] pp | no significant difference detected; consistent with the pre-registered null expectation; 95% CI of the difference [-9.3, +3.8] pp |
| B3_iii_numeric_runtime_test | 67 | 9/8 | McNemar exact on farmer_yes | 8.000 | 1.000 | 1.000 | 1.12 [0.39, 3.35] | +1.5 [-10.4, +13.4] pp | no significant difference detected; consistent with the pre-registered null expectation; 95% CI of the difference [-10.4, +13.4] pp |
| B3_iii_factual_test | 70 | 8/14 | McNemar exact on farmer_yes | 8.000 | 0.286 | 0.859 | 0.57 [0.21, 1.46] | -8.6 [-21.4, +4.3] pp | no significant difference detected; consistent with the pre-registered null expectation; 95% CI of the difference [-21.4, +4.3] pp |

### B5 — Bengali vs English, human (all 200 questions → 100 pair_id pairs; 18 pairs contain an authored translation; 8 cells, Holm)

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| B5_rating_llama31_8b_zeroshot_all100 | 100 | — | Wilcoxon signed-rank | 953.000 | 0.699 | 0.906 | 0.05 [-0.23, 0.32] | +0.01 [-0.18, +0.20] rating pts | no significant difference detected; 95% CI of the difference rank-biserial r [-0.23, 0.32]; mean diff [-0.18, 0.20] rating points (1-5) |
| B5_farmer_llama31_8b_zeroshot_all100 | 100 | 2/5 | McNemar exact on farmer_yes | 2.000 | 0.453 | 0.906 | 0.40 [0.04, 2.44] | -3.0 [-8.0, +2.0] pp | no significant difference detected; 95% CI of the difference [-8.0, +2.0] pp |
| B5_rating_llama31_8b_seed42_all100 | 100 | — | Wilcoxon signed-rank | 562.500 | 0.022 | 0.179 | 0.34 [0.05, 0.61] | +0.27 [+0.03, +0.50] rating pts | no significant difference detected; 95% CI of the difference rank-biserial r [0.05, 0.61]; mean diff [0.03, 0.50] rating points (1-5) |
| B5_farmer_llama31_8b_seed42_all100 | 100 | 13/7 | McNemar exact on farmer_yes | 7.000 | 0.263 | 0.790 | 1.86 [0.69, 5.50] | +6.0 [-3.0, +15.0] pp | no significant difference detected; 95% CI of the difference [-3.0, +15.0] pp |
| B5_rating_llama31_8b_seed42_clarify_all100 | 100 | — | Wilcoxon signed-rank | 630.000 | 0.117 | 0.577 | 0.24 [-0.06, 0.52] | +0.26 [-0.04, +0.56] rating pts | no significant difference detected; 95% CI of the difference rank-biserial r [-0.06, 0.52]; mean diff [-0.04, 0.56] rating points (1-5) |
| B5_farmer_llama31_8b_seed42_clarify_all100 | 100 | 15/5 | McNemar exact on farmer_yes | 5.000 | 0.041 | 0.262 | 3.00 [1.04, 10.55] | +10.0 [+2.0, +18.0] pp | no significant difference detected; 95% CI of the difference [+2.0, +18.0] pp |
| B5_rating_mistral_nemo_seed42_all100 | 100 | — | Wilcoxon signed-rank | 405.000 | 0.037 | 0.262 | 0.34 [0.02, 0.61] | +0.28 [+0.06, +0.52] rating pts | no significant difference detected; 95% CI of the difference rank-biserial r [0.02, 0.61]; mean diff [0.06, 0.52] rating points (1-5) |
| B5_farmer_mistral_nemo_seed42_all100 | 100 | 14/6 | McNemar exact on farmer_yes | 6.000 | 0.115 | 0.577 | 2.33 [0.84, 7.41] | +8.0 [-1.0, +17.0] pp | no significant difference detected; 95% CI of the difference [-1.0, +17.0] pp |

### B8 — Human vs automatic on numeric (PRIMARY: Spearman only; Jonckheere–Terpstra dropped in plan v3)

| test_id | n | b/c | test | stat | p_raw | p_holm | effect [95% CI] | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| B8_spearman | 180 | — | Spearman rho | -0.741 | <0.001 | <0.001 | -0.74 [-0.82, -0.65] | — | supported: abs_err < rating, significant after Holm; CI rho [-0.82, -0.65] |

### B4 — Annotator-adjusted models (SECONDARY, supporting)

Reference arm `llama31_8b_seed42`. MixedLM coefficients are adjusted rating differences (Wald 95% CI, raw p). Bayesian rows are posterior log-odds of `farmer_yes` with 95% credible intervals (mean ± 1.96 sd), no p-values. Full summaries in `mixed_model.txt`.

| test_id | n ratings | comparison | adjusted effect [95% CI / CrI] | raw p | unadjusted difference (rating pts; farmer_yes proportion for Bayesian rows) | verdict |
|---|---|---|---|---|---|---|
| B4_mixedlm_test_llama31_8b_zeroshot | 1096 | Llama ZS vs Llama FT (adjusted) | -1.02 [-1.23, -0.82] | <0.001 | -1.039 | [SECONDARY, raw p only] significant (Llama ZS < Llama FT); CI adjusted diff [-1.23, -0.82] rating points |
| B4_mixedlm_test_llama31_8b_seed42_clarify | 1096 | Llama FT+clarify vs Llama FT (adjusted) | 0.37 [0.17, 0.57] | <0.001 | 0.388 | [SECONDARY, raw p only] significant (Llama FT+clarify > Llama FT); CI adjusted diff [0.17, 0.57] rating points |
| B4_mixedlm_test_mistral_nemo_seed42 | 1096 | Mistral FT vs Llama FT (adjusted) | -0.07 [-0.27, 0.13] | 0.499 | -0.088 | [SECONDARY, raw p only] no significant difference detected; 95% CI of the difference adjusted diff [-0.27, 0.13] rating points |
| B4_bayesglm_test_llama31_8b_zeroshot | 1096 | Llama ZS vs Llama FT (adjusted) | -3.24 [-3.71, -2.77] | — | -0.288 | [SECONDARY, Bayesian] adjusted log-odds -3.24, 95% credible interval [-3.71, -2.77] — excludes 0 |
| B4_bayesglm_test_llama31_8b_seed42_clarify | 1096 | Llama FT+clarify vs Llama FT (adjusted) | 0.83 [0.49, 1.17] | — | 0.130 | [SECONDARY, Bayesian] adjusted log-odds +0.83, 95% credible interval [+0.49, +1.17] — excludes 0 |
| B4_bayesglm_test_mistral_nemo_seed42 | 1096 | Mistral FT vs Llama FT (adjusted) | -0.31 [-0.67, 0.05] | — | -0.024 | [SECONDARY, Bayesian] adjusted log-odds -0.31, 95% credible interval [-0.67, +0.05] — includes 0 |
| B4_mixedlm_all_llama31_8b_zeroshot | 1200 | Llama ZS vs Llama FT (adjusted) | -0.96 [-1.15, -0.76] | <0.001 | -0.980 | [SECONDARY, raw p only] significant (Llama ZS < Llama FT); CI adjusted diff [-1.15, -0.76] rating points |
| B4_mixedlm_all_llama31_8b_seed42_clarify | 1200 | Llama FT+clarify vs Llama FT (adjusted) | 0.34 [0.15, 0.53] | <0.001 | 0.351 | [SECONDARY, raw p only] significant (Llama FT+clarify > Llama FT); CI adjusted diff [0.15, 0.53] rating points |
| B4_mixedlm_all_mistral_nemo_seed42 | 1200 | Mistral FT vs Llama FT (adjusted) | -0.07 [-0.27, 0.13] | 0.479 | -0.079 | [SECONDARY, raw p only] no significant difference detected; 95% CI of the difference adjusted diff [-0.27, 0.13] rating points |
| B4_bayesglm_all_llama31_8b_zeroshot | 1200 | Llama ZS vs Llama FT (adjusted) | -3.24 [-3.71, -2.77] | — | -0.274 | [SECONDARY, Bayesian] adjusted log-odds -3.24, 95% credible interval [-3.71, -2.77] — excludes 0 |
| B4_bayesglm_all_llama31_8b_seed42_clarify | 1200 | Llama FT+clarify vs Llama FT (adjusted) | 0.86 [0.53, 1.19] | — | 0.115 | [SECONDARY, Bayesian] adjusted log-odds +0.86, 95% credible interval [+0.53, +1.19] — excludes 0 |
| B4_bayesglm_all_mistral_nemo_seed42 | 1200 | Mistral FT vs Llama FT (adjusted) | -0.31 [-0.67, 0.04] | — | -0.026 | [SECONDARY, Bayesian] adjusted log-odds -0.31, 95% credible interval [-0.67, +0.04] — includes 0 |

**Llama FT vs Mistral FT after adjustment:** coefficient -0.07 [-0.27, +0.13], p = 0.50; Bayesian log-odds -0.31 with a credible interval that includes 0. Neither direction nor significance changes relative to B2-iii / B3-iii. **One difference from B2:** the clarify-vs-FT *overall* rating contrast is significant in the adjusted model (+0.37 [+0.17, +0.57], raw p < .001) but not after Holm in B2 (p_raw = 0.026, p_holm = 0.128); same direction, and B4 is uncorrected for multiplicity, so B2's verdict stands.

### B8 — detail

Spearman ρ(abs_err, mean_rating) = -0.74 [-0.82, -0.65], p < .001, n = 180 FT-family numeric outputs (3 fine-tuned arms × 67 test-set numeric questions = 201; 21 multi-crop outputs have no scalar `abs_err` and are excluded; 0 excluded for non-extraction).

| abs(final depth − gold) bin (mm) | n | mean rating [95% CI] | farmer_yes [95% CI] |
|---|---|---|---|
| exact | 59 | 4.69 [4.50, 4.85] | 97% [92, 100] |
| <=0.1 | 78 | 3.39 [3.19, 3.58] | 38% [28, 50] |
| <=1.0 | 40 | 2.61 [2.29, 2.92] | 12% [2, 22] |
| >1.0 | 3 | 1.40 [1.00, 2.00] | 0% [0, 0] |

**Registered headline:** human acceptance tracks the ±0.1 mm tier, not ±1 mm — the ≤1.0 mm bin is accepted 12% of the time. **Additional finding:** with the reference visible, annotators accepted 97% of exact matches (< 0.005 mm, n = 59) but only 38% of answers within 0.005–0.1 mm (n = 78). Human acceptance tracks reference agreement, and the ±0.1 mm tolerance is more lenient than that standard. Table 8 therefore adds an exact (< 0.005 mm) column per arm.

## Table 4 — Seed robustness (A1, A2, A6, A7; significant after Holm on all three seeds?)

> Clarify-trained arms exist at seed 42 only; A3 and A4 have no seed-robustness check. (Limitation.)

| test_id | expected | seed 42 | seed 1337 | seed 2024 | robust |
|---|---|---|---|---|---|
| A1_llama31_8b | FT > zero-shot | p_holm=<0.001 eff=inf SIG | p_holm=<0.001 eff=inf SIG | p_holm=<0.001 eff=inf SIG | **yes** |
| A1_mistral_nemo | FT > zero-shot | p_holm=<0.001 eff=71.00 SIG | p_holm=<0.001 eff=79.00 SIG | p_holm=<0.001 eff=74.00 SIG | **yes** |
| A1_phi35_mini | FT > zero-shot | p_holm=<0.001 eff=43.00 SIG | p_holm=<0.001 eff=23.50 SIG | p_holm=<0.001 eff=22.00 SIG | **yes** |
| A1_phi4_mini | FT > zero-shot | p_holm=<0.001 eff=inf SIG | p_holm=<0.001 eff=inf SIG | p_holm=<0.001 eff=inf SIG | **yes** |
| A2_friedman | any difference | p_holm=<0.001 eff=0.09 SIG | p_holm=<0.001 eff=0.09 SIG | p_holm=<0.001 eff=0.11 SIG | **yes** |
| A2_llama_vs_mistral | not significant (null expected) | p_holm=0.766 eff=0.88 ns | p_holm=1.000 eff=1.04 ns | p_holm=0.451 eff=0.76 ns | no |
| A2_tier | large tier > small tier | p_holm=<0.001 eff=0.73 SIG | p_holm=<0.001 eff=0.72 SIG | p_holm=<0.001 eff=0.80 SIG | **yes** |
| A6_num_unpaired_llama31_8b | no consistent direction | p_holm=1.000 eff=1.06 ns | p_holm=1.000 eff=0.95 ns | p_holm=1.000 eff=1.06 ns | no |
| A6_fac_llama31_8b | English > Bengali | p_holm=0.526 eff=0.23 ns | p_holm=0.266 eff=0.30 ns | p_holm=1.000 eff=0.14 ns | no |
| A6_num_unpaired_mistral_nemo | no consistent direction | p_holm=1.000 eff=1.00 ns | p_holm=1.000 eff=1.12 ns | p_holm=1.000 eff=1.18 ns | no |
| A6_fac_mistral_nemo | English > Bengali | p_holm=0.002 eff=0.56 SIG | p_holm=0.058 eff=0.40 ns | p_holm=0.285 eff=0.30 ns | no |
| A6_num_unpaired_phi35_mini | no consistent direction | p_holm=0.520 eff=1.93 ns | p_holm=1.000 eff=1.21 ns | p_holm=0.520 eff=1.93 ns | no |
| A6_fac_phi35_mini | English > Bengali | p_holm=0.001 eff=0.57 SIG | p_holm=0.006 eff=0.51 SIG | p_holm=<0.001 eff=0.68 SIG | **yes** |
| A6_num_unpaired_phi4_mini | no consistent direction | p_holm=1.000 eff=1.20 ns | p_holm=1.000 eff=1.06 ns | p_holm=1.000 eff=0.94 ns | no |
| A6_fac_phi4_mini | English > Bengali | p_holm=<0.001 eff=0.61 SIG | p_holm=<0.001 eff=0.62 SIG | p_holm=<0.001 eff=0.60 SIG | **yes** |
| A7_llama31_8b | FT > zero-shot | p_holm=<0.001 eff=inf SIG | p_holm=<0.001 eff=inf SIG | p_holm=<0.001 eff=inf SIG | **yes** |
| A7_mistral_nemo | FT > zero-shot | p_holm=<0.001 eff=inf SIG | p_holm=<0.001 eff=inf SIG | p_holm=<0.001 eff=inf SIG | **yes** |
| A7_phi35_mini | FT > zero-shot | p_holm=<0.001 eff=inf SIG | p_holm=<0.001 eff=inf SIG | p_holm=<0.001 eff=inf SIG | **yes** |
| A7_phi4_mini | FT > zero-shot | p_holm=<0.001 eff=17.00 SIG | p_holm=<0.001 eff=17.00 SIG | p_holm=<0.001 eff=17.00 SIG | **yes** |

Reading: H1 (A1), the tier contrast (A2_tier), the Friedman omnibus and the thesis-title claim (A7) are robust. Llama FT vs Mistral FT is non-significant at all three seeds (consistent with the pre-registered null). H2-factual is robust for Phi-3.5 and Phi-4 only; Mistral's English advantage holds at seed 42 but not at 1337 or 2024, and Llama never shows it. A6 numeric is non-significant at every seed.

## Table 5 — Human sample: primary (182 test rows) vs sensitivity (all 200 / excluding annotator A1)

| cell | n (primary) | primary raw p | primary p_holm | all 200 (raw p) | excl. A1 (raw p) | raw-p pattern change? |
|---|---|---|---|---|---|---|
| B1_friedman | 182 | <0.001 (sig) | <0.001 (sig) | n=200, raw p <0.001 (sig) | — | no |
| B1_cochranq | 182 | <0.001 (sig) | <0.001 (sig) | n=200, raw p <0.001 (sig) | — | no |
| B2_i_overall | 182 | <0.001 (sig) | <0.001 (sig) | n=200, raw p <0.001 (sig) | n=61, raw p <0.001 (sig) | no |
| B3_i_overall | 182 | <0.001 (sig) | <0.001 (sig) | n=200, raw p <0.001 (sig) | n=61, raw p <0.001 (sig) | no |
| B2_i_numeric_runtime | 67 | <0.001 (sig) | <0.001 (sig) | n=70, raw p <0.001 (sig) | n=20, raw p <0.001 (sig) | no |
| B3_i_numeric_runtime | 67 | <0.001 (sig) | <0.001 (sig) | n=70, raw p <0.001 (sig) | n=20, raw p 0.004 (sig) | no |
| B2_i_factual | 70 | <0.001 (sig) | <0.001 (sig) | n=70, raw p <0.001 (sig) | n=23, raw p 0.004 (sig) | no |
| B3_i_factual | 70 | <0.001 (sig) | <0.001 (sig) | n=70, raw p <0.001 (sig) | n=23, raw p 0.006 (sig) | no |
| B2_ii_overall | 182 | 0.026 (sig) | 0.128 (ns) | n=200, raw p 0.019 (sig) | n=61, raw p 0.023 (sig) | no |
| B3_ii_overall | 182 | 0.007 (sig) | 0.033 (sig) | n=200, raw p 0.004 (sig) | n=61, raw p 0.013 (sig) | no |
| B2_ii_clarify | 25 | <0.001 (sig) | 0.002 (sig) | n=40, raw p <0.001 (sig) | n=12, raw p 0.016 (sig) | no |
| B3_ii_clarify | 25 | <0.001 (sig) | <0.001 (sig) | n=40, raw p <0.001 (sig) | n=12, raw p 0.031 (sig) | no |
| B2_ii_numeric_runtime | 67 | 0.543 (ns) | 1.000 (ns) | n=70, raw p 0.721 (ns) | n=20, raw p 0.672 (ns) | no |
| B3_ii_numeric_runtime | 67 | 0.143 (ns) | 0.574 (ns) | n=70, raw p 0.238 (ns) | n=20, raw p 0.219 (ns) | no |
| B2_iii_overall | 182 | 0.296 (ns) | 0.888 (ns) | n=200, raw p 0.321 (ns) | n=61, raw p 0.790 (ns) | no |
| B3_iii_overall | 182 | 0.522 (ns) | 1.000 (ns) | n=200, raw p 0.430 (ns) | n=61, raw p 1.000 (ns) | no |
| B2_iii_numeric_runtime | 67 | 0.990 (ns) | 1.000 (ns) | n=70, raw p 0.968 (ns) | n=20, raw p 0.449 (ns) | no |
| B3_iii_numeric_runtime | 67 | 1.000 (ns) | 1.000 (ns) | n=70, raw p 1.000 (ns) | n=20, raw p 0.688 (ns) | no |
| B2_iii_factual | 70 | 0.088 (ns) | 0.351 (ns) | n=70, raw p 0.088 (ns) | n=23, raw p 0.962 (ns) | no |
| B3_iii_factual | 70 | 0.286 (ns) | 0.859 (ns) | n=70, raw p 0.286 (ns) | n=23, raw p 0.754 (ns) | no |

No verdict changes: every contrast keeps its direction and its raw-p significance pattern (sensitivity runs are compared on raw p, since Holm is applied only in the primary analysis) when the 18 authored translations are added or when A1 (the most divergent annotator; leave-one-out α rises to 0.781 without A1) is removed. B5 sensitivity on the 82 native pairs: same direction (English ≥ Bengali in FT arms); raw p < .05 for Llama FT and Mistral FT ratings; nothing survives Holm in the primary analysis.

Per-arm human summary, primary sample (n = 182 per arm; bootstrap 95% CI):

| arm | mean rating | farmer_yes | acceptable (mean ≥ 4) |
|---|---|---|---|
| Llama ZS | 1.96 [1.82, 2.11] | 0.10 [0.06, 0.15] | 0.08 [0.04, 0.12] |
| Llama FT | 3.13 [2.91, 3.35] | 0.42 [0.35, 0.49] | 0.41 [0.34, 0.48] |
| Llama FT+clarify | 3.42 [3.19, 3.63] | 0.53 [0.46, 0.60] | 0.50 [0.42, 0.57] |
| Mistral FT | 3.05 [2.83, 3.27] | 0.40 [0.32, 0.47] | 0.38 [0.31, 0.45] |

## Table 6 — Dice validation (B7; 280 factual (row_id, arm) judgments, seed 42 FT arms; closes deferred M5)

| metric | point-biserial r vs farmer_yes | ROC-AUC predicting farmer_yes | Spearman ρ vs mean_rating |
|---|---|---|---|
| dice_word | 0.588 [0.503, 0.665] | 0.843 [0.788, 0.892] | 0.543 [0.443, 0.629] |
| rouge_l | 0.575 [0.491, 0.652] | 0.831 [0.775, 0.881] | 0.524 [0.422, 0.613] |
| chrf_pp | 0.549 [0.463, 0.627] | 0.824 [0.767, 0.876] | 0.509 [0.404, 0.602] |
| bertscore_f1 | 0.520 [0.428, 0.601] | 0.809 [0.750, 0.863] | 0.465 [0.355, 0.564] |
| dice_char3 | 0.514 [0.420, 0.598] | 0.793 [0.726, 0.853] | 0.453 [0.339, 0.555] |
| dice_char5 | 0.493 [0.402, 0.577] | 0.788 [0.724, 0.846] | 0.455 [0.344, 0.555] |

`dice_word` has the best point estimate on all three criteria; its CI overlaps ROUGE-L and chrF++, so it is best but not distinguishably better. Character-level Dice tracks human judgment worst. Join asserted at 280/280 rows.

## Table 7 — Agreement (B6; 100 shared judgments × 5 annotators; bootstrap 95% CI over items)

| coefficient | estimate [95% CI] | package / statsmodels | Stage 7 audit |
|---|---|---|---|
| fleiss_kappa_nominal_1to5 | 0.392 [0.321, 0.455] | 0.3923 (agree) | 0.392 (Δ 0.0003) |
| kripp_alpha_ordinal | 0.721 [0.635, 0.787] | 0.7207 (agree) | 0.721 (Δ 0.0003) |
| kripp_alpha_interval | 0.746 [0.670, 0.806] | 0.7457 (agree) | — |
| fleiss_kappa_bad_mid_good | 0.561 [0.482, 0.633] | 0.5607 (agree) | — |
| fleiss_kappa_farmer | 0.647 [0.542, 0.740] | 0.6474 (agree) | 0.647 (Δ 0.0004) |
| kripp_alpha_farmer_nominal | 0.648 [0.543, 0.741] | 0.6481 (agree) | — |

Leave-one-annotator-out ordinal α: without A1 = 0.781, without A2 = 0.685, without A3 = 0.729, without A4 = 0.716, without A5 = 0.687.

Primary human measures: would_give_farmer (kappa 0.647) and ordinal Krippendorff alpha (0.721). Nominal kappa on the 1-5 scale (0.392) is reported for completeness; it treats a 4-vs-5 disagreement as a full disagreement and so understates agreement on an ordinal scale.

## Table 8 — Exact-match denominators per arm

Two exact-match denominators are in use and are named wherever either figure appears: Stage 8 correct_0p1 = all 146 numeric rows (multi-crop rows counted correct only if every gold value is within 0.1 mm; rows with no extracted depth counted wrong); Stage 6 headline exact_0.1 = the 116 single-crop rows. No test was changed; both are tabulated per arm. The exact (< 0.005 mm) column is the standard human acceptance tracked in B8.

| arm | correct_0p1 (/146) | exact_0.1 single-crop (/116) | exact < 0.005 mm single-crop (/116) | single rows with extracted depth (/116) | multi-crop all values ≤ 0.1 mm (/30) |
|---|---|---|---|---|---|
| llama31_8b_seed1337 | 81 (55.5%) | 81 (69.8%) | 29 (25.0%) | 116 | 0 |
| llama31_8b_seed2024 | 83 (56.8%) | 83 (71.6%) | 26 (22.4%) | 116 | 0 |
| llama31_8b_seed42 | 77 (52.7%) | 77 (66.4%) | 26 (22.4%) | 116 | 0 |
| llama31_8b_seed42_clarify | 79 (54.1%) | 79 (68.1%) | 24 (20.7%) | 116 | 0 |
| llama31_8b_zeroshot | 1 (0.7%) | 1 (0.9%) | 0 (0.0%) | 99 | 0 |
| mistral_nemo_seed1337 | 82 (56.2%) | 82 (70.7%) | 28 (24.1%) | 116 | 0 |
| mistral_nemo_seed2024 | 77 (52.7%) | 77 (66.4%) | 29 (25.0%) | 116 | 0 |
| mistral_nemo_seed42 | 74 (50.7%) | 74 (63.8%) | 26 (22.4%) | 116 | 0 |
| mistral_nemo_seed42_clarify | 77 (52.7%) | 77 (66.4%) | 29 (25.0%) | 113 | 0 |
| mistral_nemo_zeroshot | 4 (2.7%) | 4 (3.4%) | 2 (1.7%) | 113 | 0 |
| phi35_mini_seed1337 | 47 (32.2%) | 47 (40.5%) | 21 (18.1%) | 112 | 0 |
| phi35_mini_seed2024 | 44 (30.1%) | 44 (37.9%) | 21 (18.1%) | 112 | 0 |
| phi35_mini_seed42 | 44 (30.1%) | 44 (37.9%) | 20 (17.2%) | 108 | 0 |
| phi35_mini_seed42_clarify | 40 (27.4%) | 40 (34.5%) | 21 (18.1%) | 101 | 0 |
| phi35_mini_zeroshot | 2 (1.4%) | 2 (1.7%) | 0 (0.0%) | 78 | 0 |
| phi4_mini_seed1337 | 51 (34.9%) | 51 (44.0%) | 23 (19.8%) | 108 | 0 |
| phi4_mini_seed2024 | 49 (33.6%) | 49 (42.2%) | 23 (19.8%) | 104 | 0 |
| phi4_mini_seed42 | 49 (33.6%) | 49 (42.2%) | 21 (18.1%) | 108 | 0 |
| phi4_mini_seed42_clarify | 50 (34.2%) | 50 (43.1%) | 25 (21.6%) | 87 | 0 |
| phi4_mini_zeroshot | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 99 | 0 |

No arm gets all values of a multi-crop row within 0.1 mm (0/30 everywhere), so `correct_0p1` over 146 equals `exact_0.1` over 116 in count for every arm and differs only in denominator.

## Table 9 — Response language (exploratory; added after Part A, plan v3)

Exploratory descriptive added after Part A (plan v3), motivated by the A7 marker finding that zero-shot answers to Bengali questions were written in English. Response language is classified by script: bn if >= 50% of alphabetic characters are in the Bengali block (U+0980-U+09FF), else en. No test, no Holm.

| arm | all Bengali questions | numeric | factual | clarify | abstain |
|---|---|---|---|---|---|
| llama31_8b_zeroshot | 100.0% [100, 100] (n=181) | 100.0% [100, 100] (n=73) | 100.0% [100, 100] (n=57) | 100.0% [100, 100] (n=34) | 100.0% [100, 100] (n=17) |
| mistral_nemo_zeroshot | 30.4% [24, 37] (n=181) | 2.7% [0, 7] (n=73) | 68.4% [56, 81] (n=57) | 0.0% [0, 0] (n=34) | 82.4% [65, 100] (n=17) |
| phi35_mini_zeroshot | 98.3% [96, 100] (n=181) | 98.6% [96, 100] (n=73) | 100.0% [100, 100] (n=57) | 97.1% [91, 100] (n=34) | 94.1% [82, 100] (n=17) |
| phi4_mini_zeroshot | 91.7% [87, 96] (n=181) | 86.3% [78, 93] (n=73) | 91.2% [82, 98] (n=57) | 100.0% [100, 100] (n=34) | 100.0% [100, 100] (n=17) |
| llama31_8b_seed42 | 100.0% [100, 100] (n=181) | 100.0% [100, 100] (n=73) | 100.0% [100, 100] (n=57) | 100.0% [100, 100] (n=34) | 100.0% [100, 100] (n=17) |
| mistral_nemo_seed42 | 100.0% [100, 100] (n=181) | 100.0% [100, 100] (n=73) | 100.0% [100, 100] (n=57) | 100.0% [100, 100] (n=34) | 100.0% [100, 100] (n=17) |
| phi35_mini_seed42 | 98.9% [97, 100] (n=181) | 97.3% [93, 100] (n=73) | 100.0% [100, 100] (n=57) | 100.0% [100, 100] (n=34) | 100.0% [100, 100] (n=17) |
| phi4_mini_seed42 | 100.0% [100, 100] (n=181) | 100.0% [100, 100] (n=73) | 100.0% [100, 100] (n=57) | 100.0% [100, 100] (n=34) | 100.0% [100, 100] (n=17) |
| llama31_8b_seed42_clarify | 100.0% [100, 100] (n=181) | 100.0% [100, 100] (n=73) | 100.0% [100, 100] (n=57) | 100.0% [100, 100] (n=34) | 100.0% [100, 100] (n=17) |
| mistral_nemo_seed42_clarify | 100.0% [100, 100] (n=181) | 100.0% [100, 100] (n=73) | 100.0% [100, 100] (n=57) | 100.0% [100, 100] (n=34) | 100.0% [100, 100] (n=17) |
| phi35_mini_seed42_clarify | 100.0% [100, 100] (n=181) | 100.0% [100, 100] (n=73) | 100.0% [100, 100] (n=57) | 100.0% [100, 100] (n=34) | 100.0% [100, 100] (n=17) |
| phi4_mini_seed42_clarify | 100.0% [100, 100] (n=181) | 100.0% [100, 100] (n=73) | 100.0% [100, 100] (n=57) | 100.0% [100, 100] (n=34) | 100.0% [100, 100] (n=17) |

**H2 note (exploratory).** Mistral-NeMo zero-shot answered only 30.4% [23.8, 37.0] of Bengali questions in Bengali script; every fine-tuned and clarify-trained arm answers ≥ 98.9% in Bengali. Fine-tuning fixes response language completely. This also qualifies the A7 Mistral zero-shot baseline: its 84% confident-error rate is measured largely on English-language answers to Bengali questions.

## Table 10 — Bootstrap 95% CIs for Stage 6 Table A headlines (DESCRIPTIVE; 10,000 row resamples)

Denominators: em_0p1 here is `correct_0p1` over 146; em_1p0 over the 116 single-crop rows; MAE over single-crop rows with an extracted depth; net_extract_rate over 146; abstain accuracy over 34; clarify_exact and invented_value_rate over 68; dice_word / chrF++ / BERTScore over 114 factual rows. `*_FT_3seeds` rows are mean ± std of the three seed estimates.

| arm | em_0p1 | em_1p0 | mae | net_extract_rate | abstain_accuracy | clarify_exact | invented_value_rate | dice_word | chrf_pp | bertscore_f1 |
|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b_seed1337 | 0.555 [0.473, 0.637] | 0.940 [0.888, 0.983] | 0.187 [0.109, 0.287] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.304 [0.255, 0.355] | 29.548 [25.061, 34.339] | 0.758 [0.742, 0.776] |
| llama31_8b_seed2024 | 0.568 [0.486, 0.651] | 0.991 [0.974, 1.000] | 0.097 [0.066, 0.141] | 0.986 [0.966, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.298 [0.249, 0.350] | 29.789 [25.281, 34.646] | 0.753 [0.735, 0.771] |
| llama31_8b_seed42 | 0.527 [0.445, 0.610] | 0.983 [0.957, 1.000] | 0.146 [0.081, 0.236] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.299 [0.254, 0.347] | 29.198 [24.995, 33.709] | 0.756 [0.740, 0.772] |
| llama31_8b_seed42_clarify | 0.541 [0.459, 0.623] | 0.983 [0.957, 1.000] | 0.109 [0.079, 0.145] | 0.986 [0.966, 1.000] | 1.000 [1.000, 1.000] | 0.338 [0.221, 0.456] | 0.632 [0.515, 0.750] | 0.301 [0.253, 0.353] | 28.914 [24.451, 33.792] | 0.753 [0.735, 0.772] |
| llama31_8b_zeroshot | 0.007 [0.000, 0.021] | 0.138 [0.078, 0.207] | 55.109 [17.799, 107.616] | 0.856 [0.795, 0.911] | 0.147 [0.029, 0.265] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.063 [0.054, 0.074] | 11.947 [10.858, 13.093] | 0.612 [0.605, 0.620] |
| mistral_nemo_seed1337 | 0.562 [0.479, 0.637] | 0.983 [0.957, 1.000] | 0.107 [0.070, 0.154] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.315 [0.269, 0.363] | 29.289 [25.409, 33.524] | 0.762 [0.747, 0.779] |
| mistral_nemo_seed2024 | 0.527 [0.445, 0.610] | 1.000 [1.000, 1.000] | 0.088 [0.067, 0.111] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.322 [0.273, 0.375] | 30.284 [25.731, 35.182] | 0.764 [0.747, 0.782] |
| mistral_nemo_seed42 | 0.507 [0.425, 0.589] | 0.983 [0.957, 1.000] | 0.142 [0.089, 0.211] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.312 [0.264, 0.360] | 31.190 [26.839, 35.791] | 0.758 [0.741, 0.775] |
| mistral_nemo_seed42_clarify | 0.527 [0.445, 0.610] | 0.957 [0.914, 0.991] | 0.121 [0.070, 0.199] | 0.966 [0.932, 0.993] | 1.000 [1.000, 1.000] | 0.382 [0.265, 0.500] | 0.500 [0.382, 0.618] | 0.315 [0.268, 0.363] | 30.714 [26.539, 35.229] | 0.763 [0.747, 0.779] |
| mistral_nemo_zeroshot | 0.027 [0.007, 0.055] | 0.147 [0.086, 0.216] | 208.594 [11.578, 526.924] | 0.979 [0.952, 1.000] | 0.147 [0.029, 0.265] | 0.015 [0.000, 0.044] | 0.971 [0.926, 1.000] | 0.043 [0.035, 0.051] | 8.147 [6.956, 9.341] | 0.581 [0.573, 0.589] |
| phi35_mini_seed1337 | 0.322 [0.247, 0.397] | 0.828 [0.759, 0.897] | 0.664 [0.356, 1.112] | 0.973 [0.945, 0.993] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.218 [0.177, 0.262] | 21.572 [17.842, 25.707] | 0.716 [0.697, 0.735] |
| phi35_mini_seed2024 | 0.301 [0.226, 0.377] | 0.810 [0.741, 0.879] | 0.486 [0.333, 0.664] | 0.966 [0.932, 0.993] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.218 [0.180, 0.258] | 20.757 [17.402, 24.431] | 0.720 [0.704, 0.737] |
| phi35_mini_seed42 | 0.301 [0.226, 0.377] | 0.784 [0.707, 0.853] | 0.557 [0.375, 0.766] | 0.932 [0.890, 0.966] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.239 [0.198, 0.281] | 22.372 [18.723, 26.245] | 0.728 [0.712, 0.745] |
| phi35_mini_seed42_clarify | 0.274 [0.205, 0.349] | 0.733 [0.655, 0.810] | 0.431 [0.307, 0.570] | 0.856 [0.801, 0.911] | 1.000 [1.000, 1.000] | 0.294 [0.191, 0.397] | 0.662 [0.544, 0.779] | 0.225 [0.184, 0.268] | 21.389 [17.711, 25.416] | 0.711 [0.692, 0.731] |
| phi35_mini_zeroshot | 0.014 [0.000, 0.034] | 0.086 [0.043, 0.138] | 870.546 [41.328, 2152.978] | 0.685 [0.610, 0.760] | 0.206 [0.088, 0.353] | 0.029 [0.000, 0.074] | 0.985 [0.956, 1.000] | 0.038 [0.031, 0.045] | 8.800 [7.846, 9.794] | 0.586 [0.580, 0.592] |
| phi4_mini_seed1337 | 0.349 [0.274, 0.425] | 0.819 [0.750, 0.888] | 0.364 [0.250, 0.501] | 0.938 [0.897, 0.973] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.249 [0.206, 0.296] | 23.867 [20.059, 28.034] | 0.732 [0.714, 0.751] |
| phi4_mini_seed2024 | 0.336 [0.260, 0.411] | 0.819 [0.750, 0.888] | 0.331 [0.221, 0.460] | 0.911 [0.863, 0.952] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.234 [0.191, 0.281] | 23.133 [19.174, 27.423] | 0.727 [0.710, 0.745] |
| phi4_mini_seed42 | 0.336 [0.260, 0.411] | 0.845 [0.776, 0.905] | 0.388 [0.248, 0.549] | 0.945 [0.904, 0.979] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.259 [0.217, 0.305] | 24.797 [20.987, 28.961] | 0.735 [0.719, 0.753] |
| phi4_mini_seed42_clarify | 0.342 [0.267, 0.418] | 0.698 [0.612, 0.784] | 0.273 [0.177, 0.384] | 0.767 [0.699, 0.836] | 1.000 [1.000, 1.000] | 0.382 [0.265, 0.500] | 0.588 [0.471, 0.706] | 0.234 [0.190, 0.281] | 23.860 [19.773, 28.278] | 0.725 [0.707, 0.743] |
| phi4_mini_zeroshot | 0.000 [0.000, 0.000] | 0.181 [0.112, 0.250] | 1743.863 [6.407, 4483.523] | 0.856 [0.795, 0.911] | 0.088 [0.000, 0.206] | 0.029 [0.000, 0.074] | 0.985 [0.956, 1.000] | 0.043 [0.035, 0.051] | 8.641 [7.579, 9.747] | 0.584 [0.576, 0.592] |
| llama31_8b_FT_3seeds | 0.550 ± 0.021 | 0.971 ± 0.028 | 0.144 ± 0.045 | 0.995 ± 0.008 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.300 ± 0.004 | 29.512 ± 0.297 | 0.756 ± 0.003 |
| mistral_nemo_FT_3seeds | 0.532 ± 0.028 | 0.989 ± 0.010 | 0.112 ± 0.028 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.316 ± 0.005 | 30.254 ± 0.951 | 0.761 ± 0.003 |
| phi35_mini_FT_3seeds | 0.308 ± 0.012 | 0.807 ± 0.022 | 0.569 ± 0.090 | 0.957 ± 0.022 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.225 ± 0.012 | 21.567 ± 0.808 | 0.722 ± 0.006 |
| phi4_mini_FT_3seeds | 0.340 ± 0.008 | 0.828 ± 0.015 | 0.361 ± 0.028 | 0.932 ± 0.018 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.247 ± 0.013 | 23.932 ± 0.834 | 0.732 ± 0.004 |

### A1b — abs_err, single-crop rows where both arms extracted a depth (DESCRIPTIVE)

| model | n both extracted (/116) | dropped | median abs_err ZS (mm) | median abs_err FT (mm) | paired median Δ FT − ZS [95% CI] |
|---|---|---|---|---|---|
| Llama-3.1-8B | 99 | 17 | 4.330 | 0.060 | -4.270 [-5.310, -3.250] |
| Mistral-NeMo-12B | 113 | 3 | 3.950 | 0.050 | -3.900 [-4.400, -2.905] |
| Phi-3.5-mini | 76 | 40 | 4.350 | 0.175 | -4.175 [-7.005, -3.435] |
| Phi-4-mini | 94 | 22 | 2.940 | 0.120 | -2.820 [-3.770, -2.290] |

## What changed — Stage 6/7 claims not supported once tested, or qualified

1. **H2 factual is not universal.** Stage 6 reported English > Bengali chrF++ in every model. Tested pairwise on 57 fact_id pairs with Holm, it holds for 3 of 4 models at seed 42 and for only 2 of 4 (Phi-3.5, Phi-4) on all three seeds. Llama's gap (+6.3 chrF++ mean) is not significant by the registered rank test; Mistral's is seed-42-only.
2. **±0.1 mm is not the human standard.** The registered B8 headline (acceptance tracks ±0.1 mm, not ±1 mm) holds, but annotators accepted 97% of exact matches and only 38% of answers in the 0.005–0.1 mm band. `exact_0.1` over-counts human-acceptable answers by roughly that band; Table 8 now shows the exact column.
3. **The detector result is not a performance estimate.** The Stage 6C figure (safe handling 36.8% → 83.8%) is reproduced, but A5 was demoted from a PRIMARY test to a descriptive, oracle-gated upper bound: Oracle-gated diagnostic. The detector reads gold labels, runs only on known-incomplete queries, and its rules were written with the test categories in view. This is an upper bound on what surface rules can catch, not a performance estimate.
4. **Clarify training's overall human benefit is measure-dependent.** On clarify rows it is unambiguous (B2/B3-ii clarify, p_holm ≤ .002). Overall, the farmer_yes gain (+11 pp, p_holm = .033) survives Holm but the mean-rating gain (r_rb = 0.25, p_holm = .128) does not; the annotator-adjusted model finds it significant uncorrected. Report as: significant on farmer_yes, not on rating after correction.
5. **Mistral's zero-shot 'clarify' behaviour was partly a language artefact.** 70% of its Bengali-question answers were in English (exploratory Table 9), which the Stage 6 metrics did not distinguish from Bengali answers.
6. **Confident-error marker coverage (Bengali).** The registered Bengali hedge list under-detects zero-shot Bengali hedging (Llama 7/34, Phi-4 12/34 rows use unlisted phrasings). This biases A7 conservatively and leaves all fine-tuned rows (0/34 hedges in every model) unaffected; the registered definition was not changed.
7. **Nothing else reversed.** No PRIMARY test, automatic or human, contradicted its pre-registered direction; no null-expected test came out significant; no human verdict changed between the 182-question primary sample and the 200-question sensitivity sample or when annotator A1 was excluded.

## Limitations recorded in this stage

- Clarify-trained arms exist at seed 42 only; A3 and A4 have no seed-robustness check.
- The locked test set carries no `pair_id`; bn/en pairing for A6 uses `fact_id` and yields 0 clarify pairs and 17 numeric pairs, so A6-numeric is unpaired and A6-clarify is descriptive.
- Truncation is inferred by re-tokenising generations (no `finish_reason` stored); decode→re-encode is not perfectly idempotent (see A7 detail). The Mistral-NeMo tokenizer regex warning from `transformers` does not affect the flag because the same default tokenizer was used at generation time.
- B4 uses interval-scale MixedLM and a variational Bayesian GLM instead of an ordinal mixed model (`ordinal::clmm`), because R is not available.
- The human sample's 18 authored Bengali translations are not locked test rows; they are excluded from the PRIMARY human analyses and their model answers were generated after Stage 6 with the same adapters, prompt and decoding (provenance verified in `precheck.json`).
- The detector (A5) reads gold labels; see Appendix A.
- Registered Bengali explicit-assumption markers under-detect Bengali hedging (zero-shot only; conservative).

## Appendix A — Detector rules (Stage 6C `s6c_postfix_clarify_detector.py`) and the test categories that informed each

> Oracle-gated diagnostic. The detector reads gold labels, runs only on known-incomplete queries, and its rules were written with the test categories in view. This is an upper bound on what surface rules can catch, not a performance estimate.

Scorable clarify rows by category: {'misspelled_crop': 16, 'missing_parameter': 40, 'unit_ambiguity': 12}; by subtype: {'ambiguous': 16, 'missing_et0': 10, 'missing_rainfall': 10, 'missing_area': 10, 'missing_stage': 10, 'unknown_local_unit': 12}. The 20 `conflicting_values` clarify rows are `pending_human` and unscored.

| # | rule (as implemented) | reads | test categories / subtypes that informed it |
|---|---|---|---|
| 1 | Gate: run only on rows whose `qtype == "clarify"`; all other rows get `detector_flag = False`. | **gold qtype** | All clarify categories. This gate alone rules out false positives on numeric/factual/abstain rows and makes c = 0 by construction. |
| 2 | `invented_answer` = `classify_response(generation) == "answer"` AND the text matches a numeric-with-unit pattern (number followed by mm / liter / litre / l, or the Bengali unit words মিমি / লিটার). | generation | Observed Stage 6B outputs: worked-answer template always emits `… মিমি … লিটার`. |
| 3a | Keyword gap: a slot is 'missing' if none of its keywords appears in the question — `et0_mm` {et0, eto, evapotranspiration, ইটিও, বাষ্পীভবন}, `rainfall_mm` {rain…, বৃষ্টি…}, `area` {acre, decimal, bigha, hectare, m², sqm, square, একর, ডেসিমাল, বিঘা, হেক্টর, বর্গমিটার, জমির পরিমাণ}, `stage` {initial, early, seedling, mid, …, প্রাথমিক, মধ্য, শেষ, পর্যায়}. | question | `missing_parameter` subtypes `missing_rainfall`, `missing_area`, `missing_stage` (Stage 6B training slots); **`missing_et0` was added in Stage 6C after inspecting the test rows** — the training data never deleted ET0. |
| 3b | Flag if `invented_answer` AND (any slot missing by 3a **OR** `category ∈ {missing_parameter, misspelled_crop, unit_ambiguity}`). | **gold category** | Exactly the three scorable clarify categories in the test set. The OR-branch flags every invented answer on a clarify row regardless of the question text. |
| 4 | Correction text: `Please provide: <slot name>` where the slot comes from `SUBTYPE_TO_SLOT[subtype]` (`ambiguous` → 'which crop', `unknown_local_unit` → 'field size and unit', …). | **gold subtype** | `misspelled_crop/ambiguous`, `unit_ambiguity/unknown_local_unit`, `missing_parameter/missing_*`. |

Consequence for A5: with rule 1 and rule 3b, every clarify row on which the clarify-trained Llama still produced a numeric answer is flagged; the only unflagged unsafe rows are those where `classify_response` did not return `answer` or no unit pattern matched. Safe handling 36.8% → 83.8% (risk difference +47.1 [+35.3, +58.8] pp) is therefore the ceiling of this rule set on this test set, not a generalisation estimate.

## Appendix B — Test-plan changelog

- **v1** (2026-09-23): initial pre-registration (176 tests)
- **v2** (2026-09-23): 
  - A3a/A3b/A4: removed seed-1337/2024 robustness cells (FT seed s vs clarify seed 42 confounds seed with treatment). Plan-level note added: 'Clarify-trained arms exist at seed 42 only; A3 and A4 have no seed-robustness check.' Carried into significance.md as a limitation.
  - A5: effect size changed from odds ratio to risk difference in safe handling (hybrid minus model-only) with paired-bootstrap 95% CI; McNemar exact on b retained. In-sample caveat attached to A5 and to every detector mention in significance.md; detector rules and the test categories that informed each go in an appendix.
  - B8: Jonckheere-Terpstra dropped; Spearman rho is the only B8 test.
  - B1 kept as a Holm family of 2.
  - Operationalisation recorded (no change to the tests): correct_0p1 on the 146 numeric rows = em_0p1 for single-crop rows where a depth was extracted; multi_all_correct (all gold values within 0.1 mm) for the 30 multi-crop rows; 0 when no depth was extracted. Both use the 0.1 mm tolerance from s6_eval.score_row.
- **v3** (2026-09-23): 
  - A5: demoted from PRIMARY to DESCRIPTIVE and removed from Holm (validity finding, not results-driven): the detector runs only on rows whose GOLD qtype is clarify and one rule reads the GOLD category, so it cannot produce a false positive or a c-cell by construction; McNemar is not a meaningful test. Risk difference + paired-bootstrap CI reported descriptively with the label: 'Oracle-gated diagnostic. The detector reads gold labels, runs only on known-incomplete queries, and its rules were written with the test categories in view. This is an upper bound on what surface rules can catch, not a performance estimate.' PRIMARY count at seed 42 falls from 32 to 31.
  - Exact-match denominators made explicit (no test changed): Two exact-match denominators are in use and are named wherever either figure appears: Stage 8 correct_0p1 = all 146 numeric rows (multi-crop rows counted correct only if every gold value is within 0.1 mm; rows with no extracted depth counted wrong); Stage 6 headline exact_0.1 = the 116 single-crop rows. No test was changed; both are tabulated per arm.
  - New DESCRIPTIVE family LANG (response language), exploratory: Exploratory descriptive added after Part A (plan v3), motivated by the A7 marker finding that zero-shot answers to Bengali questions were written in English. Response language is classified by script: bn if >= 50% of alphabetic characters are in the Bengali block (U+0980-U+09FF), else en. No test, no Holm.
  - Report-only: A7 reports registered primary (a) first, then sensitivity (b) as the definition matching the thesis; per-model zero-shot baselines reported prominently. A6 factual Llama: registered Wilcoxon verdict stands; bootstrap CI of the mean difference noted with skew. A6 numeric: CIs too wide to rule out ~15-point gaps. diff_unit column added to results CSVs so rate differences (pp) and metric-point differences are not confused.

## Appendix C — Files

- `reports/stats/test_plan.json`
- `reports/stats/entry_gate.json`
- `reports/stats/precheck.json`
- `reports/stats/results_automatic.csv`
- `reports/stats/results_human.csv`
- `reports/stats/bootstrap_cis.csv`
- `reports/stats/seed_robustness_A.csv`
- `reports/stats/a1b_descriptive.csv`
- `reports/stats/a6_clarify_descriptive.csv`
- `reports/stats/a7_descriptive.csv`
- `reports/stats/a7_explicit_assumption_examples.json`
- `reports/stats/a7_truncation_caveat.json`
- `reports/stats/bn_hedge_check.json`
- `reports/stats/exact_match_denominators.csv`
- `reports/stats/response_language.csv`
- `reports/stats/agreement.json`
- `reports/stats/mixed_model.txt`
- `reports/stats/dice_validation.csv`
- `reports/stats/b8_error_bins.csv`
- `reports/stats/human_arm_summary.csv`
- `reports/stats/env_freeze.txt`
- `reports/stats/significance.md`
- `logs/s8_stats.log`