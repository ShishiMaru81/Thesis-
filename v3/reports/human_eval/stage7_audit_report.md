# Stage 7 — Human Evaluation Audit

Five annotators · 1,200 judgments · 800 unique model outputs (200 test questions × 4 arms) · 100 rated by all five.

**Verdict: the data is clean, complete and usable. Both flagged cells are corrected. One agreement decision needs making before Stage 8.**

---

## 1. Integrity checks

| Check | Result |
|---|---|
| Arm key: 800 unique judgment_ids | Pass |
| Assignment key matches the one generated for distribution | Pass (byte-identical content) |
| Arm key and assignment key agree on row_id, qtype, lang | Pass, 0 mismatches |
| Each arm has 200 judgments, same qtype mix (70/70/40/20) | Pass |
| Each of the 200 questions appears once per arm | Pass |
| Each sheet has exactly its 240 assigned judgment_ids | Pass, all five |
| Duplicate or missing rows | None |
| Question, gold or model text edited by an annotator | None (checked against the source workbook) |
| Ratings are whole numbers 1–5 | Pass (A1, A4, A5 stored as 1.0 etc.; cast to integer) |
| All 1,200 ratings present | Pass |
| All 1,200 yes/no present | Pass (A2 J105 blank on return; corrected to yes) |
| Rating and yes/no consistent | Pass (A4 J199 contradicted on return; corrected to no) |
| Straight-lining (longest identical run) | 4–7 across annotators; no sign of it |

The arm balance in the 100 overlap rows is uneven but acceptable: Llama FT + clarify 29, zero-shot 24, Llama FT 24, Mistral 23.

## 2. Cells returned for correction

Both are solo rows, so each had only one rating. Both were confirmed by their annotator and applied; no `audit_flag` values remain.

| Annotator | judgment_id | Problem | Status |
|---|---|---|---|
| A2 | J105 | `would_give_farmer` blank (rating 5, Llama FT, abstain row) | **Corrected to yes** |
| A4 | J199 | Rated 1 but marked **yes** (Mistral FT, clarify row). The answer invents `১ দ্রোণ = ১৩৪ বর্গমিটার`. | **Corrected to no** |

## 3. Agreement on the 100 shared rows

| Measure | Value | Threshold 0.6 |
|---|---|---|
| **would_give_farmer** — Fleiss' κ | **0.647** | Pass |
| **would_give_farmer** — Krippendorff's α | **0.648** | Pass |
| **rating 1–5** — Krippendorff's α, ordinal | **0.721** | Pass |
| rating 1–5 — Krippendorff's α, interval | 0.746 | Pass |
| rating collapsed to bad (1–2) / mid (3) / good (4–5) — Fleiss' κ | 0.561 | Below |
| rating 1–5 — Fleiss' κ, nominal | 0.392 | Below |

Pairwise, annotators agree within one point on 78–94% of shared rows.

**How to read the gap between 0.72 and 0.39.** Nominal Fleiss' κ counts a 1-versus-2 disagreement as badly as 1-versus-5. Ordinal α weights by distance. The large gap means most disagreement is between adjacent scores. The Stage 7 guide specified ordinal α as the appropriate statistic for the Likert scale before any data came in, so it can stand as the primary measure. But the pipeline spec also said "if κ < 0.6, calibrate and re-annotate", and the nominal figure is below that. This has to be disclosed and a decision recorded (section 8).

**Per annotator.** Removing A1 raises ordinal α from 0.721 to 0.781; A1 also has the lowest correlation with the others' mean (0.70; the others are 0.84–0.90). Removing any other annotator lowers α. A1 is the outlier, but not badly enough to exclude.

## 4. Where the rubric drifted

### 4a. The 1-versus-2 boundary, in both directions

The rubric: 1 = wrong and a farmer would not notice (invented values); 2 = wrong and obvious.

Zero-shot numeric answers ramble and never finish, which is the textbook 2:

| | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| Zero-shot numeric scored 1 | 0 | 4 | 3 | **19 of 21** | 10 of 16 |
| Zero-shot numeric scored 2 | **21 of 21** | 19 | 14 | 2 | 5 |

Fine-tuned Llama clarify answers invent a missing value every time, which is the textbook 1:

| | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| Llama FT on clarify rows scored 1 | 8 of 8 | 10 of 10 | **3 of 9** | 13 of 13 | 12 of 12 |

A4, and A5 about half the time, treat 1 as "bad". A3 gives 2s and 3s to invented values. A1 and A2 apply the rule as written.

**Consequence:** the human 1-versus-2 split cannot carry the "confidently wrong versus obviously wrong" distinction. Everything coarser is reliable.

### 4b. Small numeric differences

The rubric did not say how to score a result that is right to within a rounding difference. With gold visible, annotators marked these down as soon as a digit differed:

| Final depth vs gold (fine-tuned rows) | Mean human rating |
|---|---|
| Exact match | 4.33 – 5.00 |
| Within 0.1 mm | 2.68 – 3.97 |
| Within 1.0 mm | 1.47 – 2.76 |
| More than 1.0 mm off | 1.00 – 2.33 |

A1 scored several answers 1 with the note "Wrong input values" where the model's effective rainfall was 5.91 against the gold's 5.89.

This is informative rather than a defect. Human scores fall in step with distance from gold, which validates the tolerance tiers, and it shows annotators treat ±1 mm as too loose. It supports reporting the ±0.1 mm figure (69.3% for Llama FT) as the headline rather than ±1 mm (97.1%). It also means the human numeric score measures agreement with the reference more than agronomic adequacy: a 0.04 mm difference is irrelevant in a field. Say so in the methods.

## 5. Results by arm

Per-judgment score = mean over its raters (overlap rows averaged across five). "Farmer yes" uses a majority vote on overlap rows.

| Arm | Mean rating | % rated 4–5 | % rated 1–2 | % farmer yes |
|---|---|---|---|---|
| Llama zero-shot | 1.91 | 8.4 | 78.8 | 9.5 |
| Llama FT | 2.98 | 38.6 | 37.7 | 39.0 |
| **Llama FT + clarify** | **3.28** | **48.8** | **34.1** | **50.0** |
| Mistral FT | 2.91 | 35.4 | 42.0 | 36.0 |

### By question type — % farmer yes

| Arm | Numeric | Factual | Clarify | Abstain |
|---|---|---|---|---|
| Llama zero-shot | 1.4 | 12.9 | 0.0 | 45.0 |
| Llama FT | 41.4 | 41.4 | **0.0** | 100.0 |
| Llama FT + clarify | 50.0 | 40.0 | **42.5** | 100.0 |
| Mistral FT | 41.4 | 32.9 | **0.0** | 100.0 |

### By language — % farmer yes

| Arm | Bengali | English | Gap |
|---|---|---|---|
| Llama zero-shot | 11.0 | 8.0 | +3.0 |
| Llama FT | 36.0 | 42.0 | −6.0 |
| Llama FT + clarify | 45.0 | 55.0 | −10.0 |
| Mistral FT | 32.0 | 40.0 | −8.0 |

## 6. How this lines up with Stage 6

| Stage 6 automatic finding | Stage 7 human judgment | Agrees? |
|---|---|---|
| Standard FT invents values on 100% of clarify rows | Annotators would give **0%** of Llama FT and **0%** of Mistral FT clarify answers to a farmer — 240 of 240 judgments rejected | Yes, exactly |
| Clarify training: 33.8% correct clarification (Llama) | 42.5% of FT + clarify clarify answers judged fit for a farmer | Yes |
| Abstention 100% after FT | 100% farmer yes on abstain rows for all three fine-tuned arms | Yes |
| Llama and Mistral statistically tied on numeric | 41.4% vs 41.4% farmer yes on numeric | Yes |
| Bengali text worse than English, all models | Bengali 6–10 points lower on farmer yes for every fine-tuned arm (Llama FT −6, Mistral FT −8, FT + clarify −10) | Yes |
| Llama FT ±1 mm accuracy 97.1% | Only ~41% judged fit for a farmer | **No** — explained in 4b; humans grade near exact match |
| Numeric accuracy of the clarify-trained arm was never re-reported in Stage 6B | Numeric farmer yes is 50.0 vs 41.4 for plain Llama FT | Unknown — pull its ±0.1 mm figure from `results_master.csv` and test in Stage 8 |

The first row is the thesis finding confirmed by people. Across the three arms without clarify training, not one of the 240 judgments on missing-parameter questions was marked fit to give a farmer. The automatic invented-value rate of 100% and the human acceptance rate of 0% agree perfectly, and they were arrived at independently.

## 7. A confound Stage 8 must handle

In the solo rows, arms are not evenly spread across annotators, and annotators differ in severity (overlap means: A3 3.05, A4 2.65, A5 2.65). A4, the harshest, rated 39 zero-shot and 39 Llama FT rows but only 27 Mistral rows. Raw means therefore partly reflect who rated what. The Llama FT vs Mistral FT gap (2.98 vs 2.91) is small enough for this to matter. Stage 8 must model annotator as a random effect rather than compare raw means.

## 8. What is left in Stage 7

1. ~~Corrections~~ — done. J105 (A2) set to yes, J199 (A4) set to no, both confirmed by their annotators.
2. **Record the agreement decision.** Recommended: primary human measures are would_give_farmer (κ 0.647) and the rating under ordinal α (0.721); the 1-versus-2 distinction is reported as unreliable, with confident-versus-obvious error severity taken from Stage 6's automatic invented-value classifier instead. Disclose the nominal κ of 0.392 and why it differs. If your supervisor holds to "κ < 0.6 means re-annotate", the fallback is a targeted recalibration: re-brief on sections 4a and 4b, re-rate only the 100 shared rows (about 30 minutes each), and recompute.
3. **Signed consent forms** in `reports/human_eval/consent/` for all five annotators.
4. **Dice validation (Gate 6, Table B)** — now possible. Join `stage7_ratings_merged.csv` to Stage 6's per-row `results_master.csv` on row_id and arm, then compute the point-biserial correlation and ROC-AUC of dice_word against farmer yes on factual rows. This closes the M5 semantic-judge item deferred to Stage 7.

Separate from Stage 7 and still open: the 60 `pending_human` golds from Stage 4. Without them, Stage 6 abstention figures stay partial.

## 9. Stage 8 — statistics

Every question appears once per arm, so the design is **paired**. Use paired tests throughout.

**Human data**
- Ordinal mixed model (CLMM): rating ~ arm + (1 | annotator) + (1 | question). This handles the severity confound in section 7.
- Friedman test across the four arms on per-question mean ratings, then pairwise Wilcoxon signed-rank with Holm–Bonferroni.
- Cochran's Q across the four arms on farmer yes/no, then pairwise McNemar with Holm–Bonferroni.
- Pre-planned contrasts: zero-shot vs FT; FT vs FT + clarify; Llama FT vs Mistral FT. Report each by question type.
- Bengali vs English: pair on `pair_id`, which links each Bengali question to its English counterpart.

**Automatic data (Stage 6)**
- Zero-shot vs fine-tuned, per model: McNemar on ±0.1 mm exact match; Wilcoxon on absolute error.
- Across the four models: Friedman on per-item correctness, Nemenyi post-hoc.
- Llama vs Mistral at ±0.1 mm: McNemar. Expected null.
- Before vs after clarify training, per model: McNemar on the 68 clarify rows.
- Bootstrap 95% CIs on every headline figure; effect sizes as odds ratios (McNemar) and rank-biserial r (Wilcoxon).

**Output:** `reports/stats/significance.md`.

---

Files: `stage7_ratings_merged.csv` (1,200 rows, arm-labelled, with `audit_flag`) · this report.
