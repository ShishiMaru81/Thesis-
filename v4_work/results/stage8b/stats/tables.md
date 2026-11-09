# Stage 8B — statistics (Phase 3)

Columns: model | n | rate A | rate B | diff pp [95% CI] | p | p_holm | verdict | outcome meaning

Noise floor: full-arm replication gave 0% classification-flip and 0% exact_0.1-flip (`results/stage8b/noise_floor.md`); the 8.3% figure is a 50-row-subset upper-bound caveat, reported but not used as the pass threshold.


## O1

| model | n | rate A | rate B | diff pp [95% CI] | p | p_holm | verdict | outcome meaning |
|---|---:|---:|---:|---|---:|---:|---|---|
| Llama FT seed42 vs Llama ZS | 40 | 2.5 | 80.0 | +77.5 [+65.0, +90.0] pp | <0.001 | <0.001 | significant (Llama FT seed42 > Llama ZS); CI [+65.0, +90.0] pp | Abstention transfers from the easy seed probe to hard adversarial OOD, or at least does not collapse. |
| Mistral FT seed42 vs Mistral ZS | 40 | 2.5 | 80.0 | +77.5 [+65.0, +90.0] pp | <0.001 | <0.001 | significant (Mistral FT seed42 > Mistral ZS); CI [+65.0, +90.0] pp | Abstention transfers from the easy seed probe to hard adversarial OOD, or at least does not collapse. |
| Phi-3.5 FT seed42 vs Phi-3.5 ZS | 40 | 0.0 | 45.0 | +45.0 [+30.0, +60.0] pp | <0.001 | <0.001 | significant (Phi-3.5 FT seed42 > Phi-3.5 ZS); CI [+30.0, +60.0] pp | Abstention transfers from the easy seed probe to hard adversarial OOD, or at least does not collapse. |
| Phi-4 FT seed42 vs Phi-4 ZS | 40 | 0.0 | 60.0 | +60.0 [+45.0, +75.0] pp | <0.001 | <0.001 | significant (Phi-4 FT seed42 > Phi-4 ZS); CI [+45.0, +75.0] pp | Abstention transfers from the easy seed probe to hard adversarial OOD, or at least does not collapse. |

## O2

| model | n | rate A | rate B | diff pp [95% CI] | p | p_holm | verdict | outcome meaning |
|---|---:|---:|---:|---|---:|---:|---|---|
| Llama easy (34 seed) vs Llama hard (40 adversarial) | 74 | 80.0 | 100.0 | +20.0 [+7.5, +32.5] pp | 0.006 | 0.013 | supported: Llama easy (34 seed) > Llama hard (40 adversarial), significant after Holm; CI [+7.5, +32.5] pp | The published 100% abstain figure is an easy-probe result (viva Q10). Hard OOD is the number to quote beside it. |
| Mistral easy (34 seed) vs Mistral hard (40 adversarial) | 74 | 80.0 | 100.0 | +20.0 [+7.5, +32.5] pp | 0.006 | 0.013 | supported: Mistral easy (34 seed) > Mistral hard (40 adversarial), significant after Holm; CI [+7.5, +32.5] pp | The published 100% abstain figure is an easy-probe result (viva Q10). Hard OOD is the number to quote beside it. |
| Phi-3.5 easy (34 seed) vs Phi-3.5 hard (40 adversarial) | 74 | 45.0 | 100.0 | +55.0 [+40.0, +70.0] pp | <0.001 | <0.001 | supported: Phi-3.5 easy (34 seed) > Phi-3.5 hard (40 adversarial), significant after Holm; CI [+40.0, +70.0] pp | The published 100% abstain figure is an easy-probe result (viva Q10). Hard OOD is the number to quote beside it. |
| Phi-4 easy (34 seed) vs Phi-4 hard (40 adversarial) | 74 | 60.0 | 100.0 | +40.0 [+25.0, +55.0] pp | <0.001 | <0.001 | supported: Phi-4 easy (34 seed) > Phi-4 hard (40 adversarial), significant after Holm; CI [+25.0, +55.0] pp | The published 100% abstain figure is an easy-probe result (viva Q10). Hard OOD is the number to quote beside it. |

## P1

| model | n | rate A | rate B | diff pp [95% CI] | p | p_holm | verdict | outcome meaning |
|---|---:|---:|---:|---|---:|---:|---|---|
| Llama zeroshot+P vs Llama zeroshot | 68 | 100.0 | 60.3 | -39.7 [-51.5, -27.9] pp | <0.001 | <0.001 | supported: Llama zeroshot+P < Llama zeroshot, significant after Holm; CI [-51.5, -27.9] pp | An instruction is enough to cut invention on the base model. |
| Mistral zeroshot+P vs Mistral zeroshot | 68 | 97.1 | 92.6 | -4.4 [-11.8, +2.9] pp | 0.453 | 0.750 | no significant difference detected; 95% CI of the difference [-11.8, +2.9] pp | The base model does not follow the ask instruction. Weak prompted baseline. |
| Phi-3.5 zeroshot+P vs Phi-3.5 zeroshot | 68 | 98.5 | 94.1 | -4.4 [-11.8, +1.5] pp | 0.375 | 0.750 | no significant difference detected; 95% CI of the difference [-11.8, +1.5] pp | The base model does not follow the ask instruction. Weak prompted baseline. |
| Phi-4 zeroshot+P vs Phi-4 zeroshot | 68 | 98.5 | 79.4 | -19.1 [-29.4, -8.8] pp | <0.001 | 0.003 | supported: Phi-4 zeroshot+P < Phi-4 zeroshot, significant after Holm; CI [-29.4, -8.8] pp | An instruction is enough to cut invention on the base model. |

## P2

| model | n | rate A | rate B | diff pp [95% CI] | p | p_holm | verdict | outcome meaning |
|---|---:|---:|---:|---|---:|---:|---|---|
| Llama FT seed42 vs Llama zeroshot+P | 116 | 3.4 | 66.4 | +62.9 [+54.3, +71.6] pp | <0.001 | <0.001 | supported: Llama FT seed42 > Llama zeroshot+P, significant after Holm; CI [+54.3, +71.6] pp | Numeric gains from fine-tuning are not just 'the model was never told the rules'. |
| Mistral FT seed42 vs Mistral zeroshot+P | 116 | 18.1 | 63.8 | +45.7 [+35.3, +56.0] pp | <0.001 | <0.001 | supported: Mistral FT seed42 > Mistral zeroshot+P, significant after Holm; CI [+35.3, +56.0] pp | Numeric gains from fine-tuning are not just 'the model was never told the rules'. |
| Phi-3.5 FT seed42 vs Phi-3.5 zeroshot+P | 116 | 4.3 | 37.9 | +33.6 [+24.1, +43.1] pp | <0.001 | <0.001 | supported: Phi-3.5 FT seed42 > Phi-3.5 zeroshot+P, significant after Holm; CI [+24.1, +43.1] pp | Numeric gains from fine-tuning are not just 'the model was never told the rules'. |
| Phi-4 FT seed42 vs Phi-4 zeroshot+P | 116 | 0.9 | 42.2 | +41.4 [+32.8, +50.0] pp | <0.001 | <0.001 | supported: Phi-4 FT seed42 > Phi-4 zeroshot+P, significant after Holm; CI [+32.8, +50.0] pp | Numeric gains from fine-tuning are not just 'the model was never told the rules'. |

## P3

| model | n | rate A | rate B | diff pp [95% CI] | p | p_holm | verdict | outcome meaning |
|---|---:|---:|---:|---|---:|---:|---|---|
| Llama zeroshot+P vs Llama FT seed42 | 68 | 100.0 | 60.3 | -39.7 [-51.5, -27.9] pp | <0.001 | <0.001 | supported: Llama zeroshot+P < Llama FT seed42, significant after Holm; CI [-51.5, -27.9] pp | An instructed base is safer on missing inputs than a fine-tuned model with no instruction. Supports the title. |
| Mistral zeroshot+P vs Mistral FT seed42 | 68 | 100.0 | 92.6 | -7.4 [-14.7, -1.5] pp | 0.062 | 0.125 | no significant difference detected; 95% CI of the difference [-14.7, -1.5] pp | Standard FT is not worse than an instructed base on invention. Weakens the 'FT creates the failure' reading; keep the form-of-error (A7/P6) argument. |
| Phi-3.5 zeroshot+P vs Phi-3.5 FT seed42 | 68 | 100.0 | 94.1 | -5.9 [-11.8, -1.5] pp | 0.125 | 0.125 | no significant difference detected; 95% CI of the difference [-11.8, -1.5] pp | Standard FT is not worse than an instructed base on invention. Weakens the 'FT creates the failure' reading; keep the form-of-error (A7/P6) argument. |
| Phi-4 zeroshot+P vs Phi-4 FT seed42 | 68 | 100.0 | 79.4 | -20.6 [-30.9, -11.8] pp | <0.001 | <0.001 | supported: Phi-4 zeroshot+P < Phi-4 FT seed42, significant after Holm; CI [-30.9, -11.8] pp | An instructed base is safer on missing inputs than a fine-tuned model with no instruction. Supports the title. |

## P4a

| model | n | rate A | rate B | diff pp [95% CI] | p | p_holm | verdict | outcome meaning |
|---|---:|---:|---:|---|---:|---:|---|---|
| Llama FT+P vs Llama FT | 68 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] pp | nan | nan | not testable, identical outcomes (0 discordant pairs) | Fine-tuning overrides an explicit instruction to ask. Strongest support for the title. |
| Mistral FT+P vs Mistral FT | 68 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] pp | nan | nan | not testable, identical outcomes (0 discordant pairs) | Fine-tuning overrides an explicit instruction to ask. Strongest support for the title. |
| Phi-3.5 FT+P vs Phi-3.5 FT | 68 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] pp | nan | nan | not testable, identical outcomes (0 discordant pairs) | Fine-tuning overrides an explicit instruction to ask. Strongest support for the title. |
| Phi-4 FT+P vs Phi-4 FT | 68 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] pp | nan | nan | not testable, identical outcomes (0 discordant pairs) | Fine-tuning overrides an explicit instruction to ask. Strongest support for the title. |

## P4b

| model | n | rate A | rate B | diff pp [95% CI] | p | p_holm | verdict | outcome meaning |
|---|---:|---:|---:|---|---:|---:|---|---|
| Llama FT+P vs Llama FT | 68 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] pp | nan | nan | not testable, identical outcomes (0 discordant pairs) | Fine-tuning overrides an explicit instruction to ask. Strongest support for the title. |
| Mistral FT+P vs Mistral FT | 68 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] pp | nan | nan | not testable, identical outcomes (0 discordant pairs) | Fine-tuning overrides an explicit instruction to ask. Strongest support for the title. |
| Phi-3.5 FT+P vs Phi-3.5 FT | 68 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] pp | nan | nan | not testable, identical outcomes (0 discordant pairs) | Fine-tuning overrides an explicit instruction to ask. Strongest support for the title. |
| Phi-4 FT+P vs Phi-4 FT | 68 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] pp | nan | nan | not testable, identical outcomes (0 discordant pairs) | Fine-tuning overrides an explicit instruction to ask. Strongest support for the title. |

## P4c

| model | n | rate A | rate B | diff pp [95% CI] | p | p_holm | verdict | outcome meaning |
|---|---:|---:|---:|---|---:|---:|---|---|
| Llama FT+P vs Llama FT | 116 | 66.4 | 54.3 | -12.1 [-22.4, -1.7] pp | 0.034 | 0.034 | SIGNIFICANT (FT+P < FT) — instruction harms numeric accuracy; CI [-22.4, -1.7] pp | The instruction harms numeric accuracy. Qualify any P4a/P4b gains. |
| Mistral FT+P vs Mistral FT | 116 | 63.8 | 47.4 | -16.4 [-26.7, -6.9] pp | 0.003 | 0.006 | SIGNIFICANT (FT+P < FT) — instruction harms numeric accuracy; CI [-26.7, -6.9] pp | The instruction harms numeric accuracy. Qualify any P4a/P4b gains. |
| Phi-3.5 FT+P vs Phi-3.5 FT | 116 | 37.9 | 17.2 | -20.7 [-31.0, -10.3] pp | <0.001 | 0.002 | SIGNIFICANT (FT+P < FT) — instruction harms numeric accuracy; CI [-31.0, -10.3] pp | The instruction harms numeric accuracy. Qualify any P4a/P4b gains. |
| Phi-4 FT+P vs Phi-4 FT | 116 | 42.2 | 25.9 | -16.4 [-25.9, -6.9] pp | 0.002 | 0.006 | SIGNIFICANT (FT+P < FT) — instruction harms numeric accuracy; CI [-25.9, -6.9] pp | The instruction harms numeric accuracy. Qualify any P4a/P4b gains. |

## P5

| model | n | rate A | rate B | diff pp [95% CI] | p | p_holm | verdict | outcome meaning |
|---|---:|---:|---:|---|---:|---:|---|---|
| Llama clarify-trained vs Llama zeroshot+P | 68 | 19.1 | 33.8 | +14.7 [+2.9, +26.5] pp | 0.041 | 0.041 | supported: Llama clarify-trained > Llama zeroshot+P, significant after Holm; CI [+2.9, +26.5] pp | Three hundred targeted examples teach more than the prompt. Clarify training is not redundant. |
| Mistral clarify-trained vs Mistral zeroshot+P | 68 | 5.9 | 38.2 | +32.4 [+20.6, +44.1] pp | <0.001 | <0.001 | supported: Mistral clarify-trained > Mistral zeroshot+P, significant after Holm; CI [+20.6, +44.1] pp | Three hundred targeted examples teach more than the prompt. Clarify training is not redundant. |
| Phi-3.5 clarify-trained vs Phi-3.5 zeroshot+P | 68 | 5.9 | 29.4 | +23.5 [+13.2, +33.8] pp | <0.001 | <0.001 | supported: Phi-3.5 clarify-trained > Phi-3.5 zeroshot+P, significant after Holm; CI [+13.2, +33.8] pp | Three hundred targeted examples teach more than the prompt. Clarify training is not redundant. |
| Phi-4 clarify-trained vs Phi-4 zeroshot+P | 68 | 11.8 | 38.2 | +26.5 [+14.7, +38.2] pp | <0.001 | <0.001 | supported: Phi-4 clarify-trained > Phi-4 zeroshot+P, significant after Holm; CI [+14.7, +38.2] pp | Three hundred targeted examples teach more than the prompt. Clarify training is not redundant. |

## P7

| model | n | rate A | rate B | diff pp [95% CI] | p | p_holm | verdict | outcome meaning |
|---|---:|---:|---:|---|---:|---:|---|---|
| llama31_8b_zeroshot_P vs llama31_8b_zeroshot | 146 | 0.7 | 22.6 | +21.9 [+15.1, +28.8] pp | <0.001 | <0.001 | SIGNIFICANT difference (llama31_8b_zeroshot_P > llama31_8b_zeroshot) — CONTRADICTS the pre-registered null expectation; CI [+15.1, +28.8] pp | A prompted arm asks on complete inputs. Read its gains in P1/P3/P4/P5 against this over-asking rate; do not treat clarify gains as unqualified. |
| llama31_8b_seed42_P vs llama31_8b_seed42 | 146 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] pp | nan | nan | not testable, identical outcomes (0 discordant pairs) | Prompted arms do not ask on complete numeric questions. P1/P3/P4/P5 gains are not an artefact of blanket asking. |
| mistral_nemo_zeroshot_P vs mistral_nemo_zeroshot | 146 | 4.1 | 6.8 | +2.7 [-2.1, +7.5] pp | 0.388 | 0.388 | no significant difference detected; consistent with the pre-registered null expectation; 95% CI of the difference [-2.1, +7.5] pp | Prompted arms do not ask on complete numeric questions. P1/P3/P4/P5 gains are not an artefact of blanket asking. |
| mistral_nemo_seed42_P vs mistral_nemo_seed42 | 146 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] pp | nan | nan | not testable, identical outcomes (0 discordant pairs) | Prompted arms do not ask on complete numeric questions. P1/P3/P4/P5 gains are not an artefact of blanket asking. |
| phi35_mini_zeroshot_P vs phi35_mini_zeroshot | 146 | 4.1 | 0.7 | -3.4 [-7.5, +0.0] pp | 0.125 | 0.250 | no significant difference detected; consistent with the pre-registered null expectation; 95% CI of the difference [-7.5, +0.0] pp | Prompted arms do not ask on complete numeric questions. P1/P3/P4/P5 gains are not an artefact of blanket asking. |
| phi35_mini_seed42_P vs phi35_mini_seed42 | 146 | 0.0 | 12.3 | +12.3 [+7.5, +17.8] pp | <0.001 | <0.001 | SIGNIFICANT difference (phi35_mini_seed42_P > phi35_mini_seed42) — CONTRADICTS the pre-registered null expectation; CI [+7.5, +17.8] pp | A prompted arm asks on complete inputs. Read its gains in P1/P3/P4/P5 against this over-asking rate; do not treat clarify gains as unqualified. |
| phi4_mini_zeroshot_P vs phi4_mini_zeroshot | 146 | 2.7 | 16.4 | +13.7 [+7.5, +20.5] pp | <0.001 | <0.001 | SIGNIFICANT difference (phi4_mini_zeroshot_P > phi4_mini_zeroshot) — CONTRADICTS the pre-registered null expectation; CI [+7.5, +20.5] pp | A prompted arm asks on complete inputs. Read its gains in P1/P3/P4/P5 against this over-asking rate; do not treat clarify gains as unqualified. |
| phi4_mini_seed42_P vs phi4_mini_seed42 | 146 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] pp | nan | nan | not testable, identical outcomes (0 discordant pairs) | Prompted arms do not ask on complete numeric questions. P1/P3/P4/P5 gains are not an artefact of blanket asking. |

## O3 — conflicting_values class counts (descriptive, n=20)

| model | arm | answer | clarify | abstain | empty |
|---|---|---:|---:|---:|---:|
| llama31_8b | llama31_8b_zeroshot | 20 | 0 | 0 | 0 |
| llama31_8b | llama31_8b_seed42 | 20 | 0 | 0 | 0 |
| llama31_8b | llama31_8b_seed42_clarify | 20 | 0 | 0 | 0 |
| mistral_nemo | mistral_nemo_zeroshot | 20 | 0 | 0 | 0 |
| mistral_nemo | mistral_nemo_seed42 | 20 | 0 | 0 | 0 |
| mistral_nemo | mistral_nemo_seed42_clarify | 20 | 0 | 0 | 0 |
| phi35_mini | phi35_mini_zeroshot | 20 | 0 | 0 | 0 |
| phi35_mini | phi35_mini_seed42 | 20 | 0 | 0 | 0 |
| phi35_mini | phi35_mini_seed42_clarify | 17 | 3 | 0 | 0 |
| phi4_mini | phi4_mini_zeroshot | 19 | 1 | 0 | 0 |
| phi4_mini | phi4_mini_seed42 | 20 | 0 | 0 | 0 |
| phi4_mini | phi4_mini_seed42_clarify | 18 | 0 | 2 | 0 |

## P6 — A7 form-of-error rates on clarify rows (descriptive)

| model | variant | truncation % | invented % | confident_error(a) % | confident_error(b, strict) % | explicit_assumption % |
|---|---|---:|---:|---:|---:|---:|
| llama31_8b | zeroshot | 54.4 | 100.0 | 45.6 | 16.2 | 35.3 |
| llama31_8b | zeroshot_P | 19.1 | 60.3 | 23.5 | 23.5 | 1.5 |
| llama31_8b | seed42 | 0.0 | 100.0 | 98.5 | 98.5 | 0.0 |
| llama31_8b | seed42_P | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| llama31_8b | seed42_clarify | 0.0 | 63.2 | 47.1 | 47.1 | 0.0 |
| mistral_nemo | zeroshot | 13.2 | 97.1 | 83.8 | 51.5 | 44.1 |
| mistral_nemo | zeroshot_P | 0.0 | 92.7 | 92.7 | 92.7 | 0.0 |
| mistral_nemo | seed42 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| mistral_nemo | seed42_P | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| mistral_nemo | seed42_clarify | 0.0 | 50.0 | 30.9 | 30.9 | 0.0 |
| phi35_mini | zeroshot | 66.2 | 98.5 | 30.9 | 22.1 | 22.1 |
| phi35_mini | zeroshot_P | 27.9 | 94.1 | 66.2 | 58.8 | 11.8 |
| phi35_mini | seed42 | 0.0 | 100.0 | 98.5 | 98.5 | 0.0 |
| phi35_mini | seed42_P | 0.0 | 100.0 | 55.9 | 55.9 | 0.0 |
| phi35_mini | seed42_clarify | 0.0 | 66.2 | 51.5 | 51.5 | 0.0 |
| phi4_mini | zeroshot | 16.2 | 98.5 | 75.0 | 61.8 | 14.7 |
| phi4_mini | zeroshot_P | 2.9 | 79.4 | 22.1 | 22.1 | 0.0 |
| phi4_mini | seed42 | 0.0 | 100.0 | 98.5 | 98.5 | 0.0 |
| phi4_mini | seed42_P | 0.0 | 100.0 | 91.2 | 91.2 | 0.0 |
| phi4_mini | seed42_clarify | 0.0 | 58.8 | 36.8 | 36.8 | 0.0 |

## P_DESC — factual chrF++ and abstain accuracy, prompted arms (descriptive)

| model | arm | metric | n | value | 95% CI |
|---|---|---|---:|---:|---|
| llama31_8b | llama31_8b_zeroshot_P | chrf_pp | 114 | 15.174 | [13.53, 17.01] |
| llama31_8b | llama31_8b_zeroshot_P | abstain_correct | 34 | 0.5 | [0.32, 0.68] |
| llama31_8b | llama31_8b_seed42_P | chrf_pp | 114 | 13.331 | [9.74, 17.33] |
| llama31_8b | llama31_8b_seed42_P | abstain_correct | 34 | 1.0 | [1.00, 1.00] |
| mistral_nemo | mistral_nemo_zeroshot_P | chrf_pp | 114 | 14.158 | [12.17, 16.29] |
| mistral_nemo | mistral_nemo_zeroshot_P | abstain_correct | 34 | 0.6471 | [0.47, 0.79] |
| mistral_nemo | mistral_nemo_seed42_P | chrf_pp | 114 | 21.924 | [18.93, 25.27] |
| mistral_nemo | mistral_nemo_seed42_P | abstain_correct | 34 | 1.0 | [1.00, 1.00] |
| phi35_mini | phi35_mini_zeroshot_P | chrf_pp | 114 | 11.27 | [9.99, 12.59] |
| phi35_mini | phi35_mini_zeroshot_P | abstain_correct | 34 | 0.7059 | [0.56, 0.85] |
| phi35_mini | phi35_mini_seed42_P | chrf_pp | 114 | 6.687 | [5.38, 8.05] |
| phi35_mini | phi35_mini_seed42_P | abstain_correct | 34 | 1.0 | [1.00, 1.00] |
| phi4_mini | phi4_mini_zeroshot_P | chrf_pp | 114 | 13.462 | [11.97, 15.02] |
| phi4_mini | phi4_mini_zeroshot_P | abstain_correct | 34 | 0.5882 | [0.41, 0.74] |
| phi4_mini | phi4_mini_seed42_P | chrf_pp | 114 | 16.882 | [14.61, 19.45] |
| phi4_mini | phi4_mini_seed42_P | abstain_correct | 34 | 1.0 | [1.00, 1.00] |
