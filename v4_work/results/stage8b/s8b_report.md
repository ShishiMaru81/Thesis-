# Stage 8B report

Written 2026-09-26. All numbers below are read from files under
`results/stage8b/` (Part 3 sheet-building numbers are read from
`results/stage8b/part3/` and, where the underlying sheets themselves live
under `docs/`, from those files directly — cited per row). Nothing here is
hand-typed from memory.

## 1. What Stage 8B is for

Stage 8B is an improvement/completion round on top of the completed Stage 8
statistics: it scores the 60 test rows that Stage 8 excluded by design
(`gold_status = pending_human`), adds a prompted-baseline arm (system prompt
+ few-shot examples, no fine-tuning weights changed) to separate "the model
was never told the rules" from "fine-tuning overrides an instruction," and
builds the human-facing check sheets (digitisation, crop-coefficient
provenance, semantic-leakage review, external-rater agreement, and an
expert-reviewed rubric v2) that Stage 8's own disclosures flagged as open.
It does not re-run or change any Stage 8 Part A/B test, and it authors no
new gold text for the 60 excluded rows — those stay class-level-only by
design.

## 2. The frozen plan

`results/stage8b/FROZEN.txt`:
- `prompt_P.txt` sha256 `cfb493045a9f52e51857ca9f8b4f1cea7a1c0c23831e55cb1d3dbd28d541da0a`
- `fewshot_examples.json` sha256 `329b484a4c9517bd37c2d36a4febe04d35635f781444756fcd6634add679c993`
- `plan.json` sha256 `34ae9b65ef3c1c07dce107adcdf038b8746c14be5de0176bbd9faabf528396a8` (canonical: `sha256` field null, then hashed)

Re-verified in this session: `scripts/s8b_stats.py`'s entry gate recomputes
the canonical plan.json hash and the test_hardened.jsonl SHA before running
any test, and aborts on mismatch. Both matched; see `docs/LOG.md`.

**Amendment v1** (`results/stage8b/plan_amendment_v1.md`, sha256
`f495ddb447828a84359a3dbb48cea5ae4fee520046640ce2063dd798c683e744`): the
Step-0 hardware check sampled 50 rows in a different batch composition than
Stage 6 used, and under 4-bit greedy decoding, batch composition changes the
numerics — so byte-identity on that 50-row subset was not a fair pass/fail
criterion. The amendment replaced it with a full 422-row replication using
the unmodified Stage 6 code path, PASS = classification agreement ≥99% and
each headline count within ±1 row. The frozen Step-0 files themselves were
not touched.

## 3. Hardware check, replication, and noise floor

**v0 check (FAILED)** — `results/stage8b/hw_check.md`: 50 rows, stratified by
qtype×lang, same decoding as Stage 6. Byte-identical 36/50 (72.0%, required
≥90%) → **FAIL**. Classification matched 50/50; exact_0.1 matched 11/12
compared. This is the failure the amendment above explains and supersedes as
a pass criterion (not as evidence — the underlying flip was real, see below).

**Full-arm replication (PASSED)** — `results/stage8b/hw_replication/replication_report.md`
(and `.json`): all 422 rows, `v3/scripts/s6_eval.py::generate_arm`, batch 16,
max_new_tokens 512, max_seq_len 2048, greedy, Stage 6 row order.
- Byte-identical: 422/422 (100.0%)
- Classification agreement: 422/422 (100.0%)
- exact_0.1 agreement (116 single-crop): 116/116
- Headline counts, saved → replication: exact_0.1 on 116 = 77 → 77; invented_value on 68 = 68 → 68; abstain on 34 seed = 34 → 34 (all ±0)
- **PASS.**

**Noise floor** — `results/stage8b/noise_floor.md`: with batch composition
matching Stage 6 (the full replication), classification flip rate = 0% and
exact_0.1 flip rate = 0%. The 50-row subset (different batch composition)
showed one exact_0.1 flip (1/12, 8.3%) with zero classification flips — an
upper-bound artefact of subset batching under 4-bit greedy decoding, not
evidence against the Stage 6 code path. Part 2/3 statistics below use the
0%/0% full-replication floor as the operative noise floor and report the
8.3% figure only as a caveat on exact_0.1-type tests.

## 4. Part 1 — class-level scoring of the 60 excluded rows

Counts: `results/stage8b/part1/{o1_hard_ood_abstention,o2_easy_vs_hard,o3_conflicting_values}.json`,
`part1/part1_tables.md`. Tests: `results/stage8b/stats/part1_tests.csv`,
`stats/tables.md` (families O1, O2).

**O1 — adversarial (hard) out-of-domain abstention, n=40, McNemar exact, Holm within family O1:**

| Model | ZS abstain % | FT seed42 abstain % | diff pp [95% CI] | p_holm | Outcome meaning |
|---|---:|---:|---|---:|---|
| Llama | 2.5 | 80.0 | +77.5 [+65.0, +90.0] | <0.001 | Abstention transfers from the easy seed probe to hard adversarial OOD, or at least does not collapse. |
| Mistral | 2.5 | 80.0 | +77.5 [+65.0, +90.0] | <0.001 | same |
| Phi-3.5 | 0.0 | 45.0 | +45.0 [+30.0, +60.0] | <0.001 | same |
| Phi-4 | 0.0 | 60.0 | +60.0 [+45.0, +75.0] | <0.001 | same |

All four significant, all in the direction of FT abstaining *more* than
zero-shot on hard OOD — so `O1_FT_still_abstains` applies to all four (no
model showed a significant collapse of abstention on hard OOD).

**O2 — easy (34 seed) vs hard (40 adversarial), FT seed42, Fisher exact, Holm within family O2:**

| Model | Easy abstain % | Hard abstain % | diff pp [95% CI] | p_holm | Outcome meaning |
|---|---:|---:|---|---:|---|
| Llama | 100.0 | 80.0 | +20.0 [+7.5, +32.5] | 0.013 | `O2_easy_higher` — the published 100% abstain figure is an easy-probe result; hard OOD (80.0%) is the number to quote beside it. |
| Mistral | 100.0 | 80.0 | +20.0 [+7.5, +32.5] | 0.013 | same |
| Phi-3.5 | 100.0 | 45.0 | +55.0 [+40.0, +70.0] | <0.001 | same |
| Phi-4 | 100.0 | 60.0 | +40.0 [+25.0, +55.0] | <0.001 | same |

All four significant: the easy seed-abstain probe is easier than hard
adversarial OOD in every model.

**O3 — conflicting-value rows (n=20), descriptive only, no gold policy, no p-values** —
`part1/o3_conflicting_values.json`: every arm answers (does not clarify or
abstain) on all or nearly all 20 rows: 20/20 `answer` for 9 of 12 arms;
Phi-3.5 clarify-trained gives 17 answer / 3 clarify; Phi-4 zero-shot gives
19 answer / 1 clarify; Phi-4 clarify-trained gives 18 answer / 2 abstain.
Outcome meaning `O3` applies throughout: no policy gold exists for these
rows, so this is a behavioural description only, not an accuracy claim.

## 5. Part 2 — prompted baseline

Predictions: `results/stage8b/predictions/*_P.jsonl` (+ `_meta.json`), all 8
arms × 422 rows. Scoring: `results/stage8b/part2/scores/*.jsonl` (+
`arm_summary.md`). Tests: `results/stage8b/stats/part2_tests.csv`,
`p6_descriptive.csv`, `p_desc_factual_abstain.csv`, `tables.md` (families
P1-P7, P6, P_DESC).

**Per-arm summary** (`part2/arm_summary.md`):

| Arm | invented_value % (68) | clarify_exact % (68) | exact_0.1 % (116) | over-ask % (146) |
|---|---:|---:|---:|---:|
| llama31_8b_zeroshot_P | 60.3 | 19.1 | 3.4 | 22.6 |
| llama31_8b_seed42_P | 100.0 | 0.0 | 54.3 | 0.0 |
| mistral_nemo_zeroshot_P | 92.6 | 5.9 | 18.1 | 6.8 |
| mistral_nemo_seed42_P | 100.0 | 0.0 | 47.4 | 0.0 |
| phi35_mini_zeroshot_P | 94.1 | 5.9 | 4.3 | 0.7 |
| phi35_mini_seed42_P | 100.0 | 0.0 | 17.2 | 12.3 |
| phi4_mini_zeroshot_P | 79.4 | 11.8 | 0.9 | 16.4 |
| phi4_mini_seed42_P | 100.0 | 0.0 | 25.9 | 0.0 |

Factual chrF++ (114 rows) and abstain accuracy (34 seed rows), computed
fresh for this report (see §8, item 1) — `stats/p_desc_factual_abstain.csv`:

| Model | zeroshot_P chrF++ | zeroshot_P abstain % | seed42_P chrF++ | seed42_P abstain % |
|---|---:|---:|---:|---:|
| Llama | 15.2 | 50.0 | 13.3 | 100.0 |
| Mistral | 14.2 | 64.7 | 21.9 | 100.0 |
| Phi-3.5 | 11.3 | 70.6 | 6.7 | 100.0 |
| Phi-4 | 13.5 | 58.8 | 16.9 | 100.0 |

**P1-P7, per model, from `stats/tables.md`:**

| Test | Comparison basis | Significant in | Outcome |
|---|---|---|---|
| P1 (invented, prompt vs ZS, 68) | Llama, Phi-4 significant (P lower); Mistral, Phi-3.5 not | 2/4 | `P1_P_helps_base` for Llama/Phi-4; `P1_P_fails_on_base` for Mistral/Phi-3.5 |
| P2 (exact_0.1, FT vs prompt, 116) | all 4 significant, FT higher | 4/4 | `P2_FT_beats_P` for all four |
| P3 (invented, prompt vs FT, 68) | Llama, Phi-4 significant (P lower); Mistral, Phi-3.5 not | 2/4 | `P3_P_invents_less_than_FT` for Llama/Phi-4; `P3_FT_invents_less_or_equal` for Mistral/Phi-3.5 |
| P4a (clarify_exact, FT+P vs FT, 68) | not testable, all 4 (both arms 0% clarify_exact on this subset — 0 discordant pairs) | 0/4 | `P4_P_fails_on_FT` (paired with P4b below) |
| P4b (invented, FT+P vs FT, 68) | not testable, all 4 (both arms 100% invented — 0 discordant pairs) | 0/4 | `P4_P_fails_on_FT` |
| P4c (exact_0.1, FT+P vs FT, 116, null expected) | all 4 significant, FT+P **lower** | 4/4 | `P4c_loss` for all four — "SIGNIFICANT (FT+P < FT) — instruction harms numeric accuracy" |
| P5 (clarify_exact, clarify-trained vs prompt, 68) | all 4 significant, clarify-trained higher | 4/4 | `P5_CL_beats_P` for all four |
| P7 (over-ask on 146, prompted vs unprompted) | zeroshot_P arm: Llama, Phi-4 significant over-ask; Mistral, Phi-3.5 not. seed42_P arm: Phi-3.5 significant over-ask; Llama, Mistral, Phi-4 not testable (0 discordant pairs, both 0%) | 3/8 cells | `P7_overask` for llama_zs, phi4_zs, phi35_ft; `P7_no_overask` for the other 5 cells |

**P4/P4a/P4b/P4c together, the strongest single finding of Stage 8B**: adding
the same instruction prompt P to an already fine-tuned model does **not**
raise named clarification (P4a, 0% → 0%, not even testable — the FT+P arm
never once names the missing parameter on these 68 rows) and does not lower
invented values (P4b, 100% → 100%, not testable), while it **significantly
lowers** numeric accuracy on complete inputs (P4c, all four models, all
p_holm ≤ 0.034). This is `P4_P_fails_on_FT`: "fine-tuning overrides an
explicit instruction to ask" — the strongest support in the plan's own
outcome-meaning language for the thesis's title claim, and unlike P1-P3 it
costs something (P4c) rather than being free.

**P6 (descriptive, form-of-error rates on clarify rows)** —
`stats/p6_descriptive.csv`: fine-tuned arms (seed42, seed42_P) show 0%
truncation and 0% explicit-assumption markers in every model, same as Stage
8's A7 finding for the original fine-tuned arms — the prompted+fine-tuned
arms invent just as confidently and completely as the unprompted fine-tuned
arms; the prompt does not make the failures look more hedged.

**Noise floor applied to Part 2**: `stats/part2_tests.csv`'s
`exceeds_noise_floor` column compares each |diff| against the 0.0 pp
full-replication floor from §3; every non-zero difference above exceeds it
(the floor is 0% under matched-batch greedy decoding). For the four
exact_0.1-type tests (P2, P4c), the file also carries the 8.3 pp
subset-upper-bound caveat from noise_floor.md for context — every P2/P4c
difference reported above (24.1-71.6 pp) is far larger than even that
conservative bound.

## 6. Part 3 / 4 status

| Sheet | Built | Location | Waiting on |
|---|---|---|---|
| Digitisation check (2 copies) | yes | `results/stage8b/part3/digitisation_check_C1.xlsx`, `_C2.xlsx`, `digitisation_sample_manifest.json` | 2 people, ~60-90 min each, independently |
| Digitisation agreement script | yes (build only) | `scripts/digitisation_agreement.py` | both sheets returned to `part3/returned/` |
| Kc provenance | yes, fully computed | `results/stage8b/part3/kc_provenance.xlsx`, `kc_provenance_summary.md` | 1 person to review/verify the flagged rows; no further web lookup needed — FAO-56 Table 12 was fetched live (§8) |
| Semantic leakage review (2 copies) | yes | `results/stage8b/part3/leakage_review_R1.xlsx`, `_R2.xlsx`, `leakage_review_note.md` | 2 people, ~45-60 min each, independently |
| Leakage agreement script | yes (build only) | `scripts/leakage_agreement.py` | both sheets returned to `part3/returned/` |
| External-rater agreement script | yes (built + self-tested on synthetic data) | `scripts/s8b_external_agreement.py` | both external raters' filled sheets in `docs/stage7_materials/returned/` |

No Part 3/4 script was run on real human data because none exists yet
(`part3/returned/` and `docs/stage7_materials/returned/` do not exist). The
digitisation-check and Kc-provenance work surfaced two findings while
building the sheets, not from any returned data:
- The calendar image files on disk do not byte-match the SHA-256 recorded at
  digitisation time for 27/28 crops (visually spot-checked as re-saves, not
  wrong files — see §8).
- The 60-pair leakage sample contains no literal duplicate-question pair
  (all 8 `same_fact_id=True` rows differ in their numeric inputs).

## 7. Rubric v2

Status: **not yet reviewed by the expert.** Nothing under rubric v2 has been
used to re-score any real judgment; the numeric dry run below is explicitly
marked `dry_run=true` on every row and uses only the team's own (unapproved)
draft thresholds.

- `scripts/s8b_rubric_v2_numeric.py`: adapted from the team's draft. Reads
  `docs/rubric_v2/returned/Rubric_v2_Expert_Verification.xlsx` (does not
  exist yet) for per-crop tolerances and the minimum error; **refuses to run
  without `--dry-run`** while that file is absent.
- Dry run on the 280 numeric judgments (`docs/rubric_v2/dry_run/numeric_v2_dryrun.csv`):
  0/280 rows flagged for a human; score distribution 5:147, 1:99, 4:21, 2:7,
  3:6. By arm: `llama31_8b_zeroshot` is 70/70 score-1 (consistent with Stage
  6's near-zero zero-shot numeric accuracy); the three fine-tuned arms in
  this sample score roughly 47-51/70 fives.
- Re-rating sheets for the 520 non-numeric judgments, one per team member,
  built in `docs/rubric_v2/rerating_sheets/rubric_v2_rerating_A{1-5}.xlsx` —
  each is the same 65 shared items plus that member's own 91 solo items,
  reusing Stage 7's own overlap assignment rather than a new split.
- v2 versions of the external sheets, same 100 shared items, in
  `docs/stage7_materials/v2/Stage7_External_E{1,2}_v2.xlsx`.
- All rerating/external-v2 workbooks carry a "DO NOT DISTRIBUTE UNTIL THE
  EXPERT APPROVES" banner in their README sheet.
- **After approval**: re-run `s8b_rubric_v2_numeric.py` (no `--dry-run`) on
  the 280 numeric judgments, collect the team's 520 binary re-ratings and the
  externals' v2 ratings, then re-run Stage 8 Part B statistics on rubric v2
  scores. Keep the v1 results; report v2 as a revision requested by the
  supervisor after results were known, per `THESIS_MASTER_CONTEXT.md`
  §11b.

## 8. Open items, limitations, and things found while building this

1. **Phase 2 scoring gap, fixed for this report only**: `part2/scores/*.jsonl`
   has no `chrf_pp` for factual rows because `scripts/stage8b_part2.py`
   called `s6_eval.score_row` directly and never ran `score_arm()`'s batch
   chrF++ step. `scripts/s8b_stats.py` recomputes it fresh with the same
   `sacrebleu.CHRF(word_order=2)` + `bn_normalize.normalize` call for the
   `P_DESC_factual_abstain` descriptive test only; the score files themselves
   were left unchanged. Evidence: `scripts/s8b_stats.py::StageBData.factual_chrf`,
   `docs/LOG.md` 2026-09-26.
2. **Calendar image provenance mismatch**: source_image SHA-256 recorded at
   digitisation time matches disk for only 1/28 crops (Coconut). Visual spot
   check of Brinjal.jpg confirmed a correct, legible BARC calendar, so this
   is treated as re-saving/re-compression, not wrong files — but it was
   never verified for all 28 crops, only spot-checked for one. Evidence:
   `results/stage8b/part3/digitisation_sample_manifest.json` (`image_sha_note`).
3. **Kc citation is mislabeled, not necessarily wrong**: 26/29 crops in
   `kc_values_hybrid.csv` cite "Doorenbos & Pruitt 1977" but the fetched
   FAO-56 Table 12 (Allen et al. 1998) numbers match in most cases — the
   author/year attribution is wrong, not (always) the number. Confirmed
   mismatches: Onion kc_ini (0.15 vs FAO-56's 0.7 group default — the
   outlier named in the original prompt), Brinjal kc_mid/kc_end. Confirmed
   unverifiable: Mustard, Jute (identical to this CSV's own Cotton row),
   Guava/Mango/Litchi/Jackfruit/Jujube (no FAO-56 entry or analog at all).
   Evidence: `results/stage8b/part3/kc_provenance_summary.md`.
4. **Semantic leakage sample has zero literal duplicates**: worth reporting
   as a finding in itself — see §6. Evidence: `results/stage8b/part3/leakage_review_note.md`.
5. **Rubric v2 numeric rule does not check hard rule A5** (volume given in an
   unusable unit) — documented as an unimplemented limitation in
   `scripts/s8b_rubric_v2_numeric.py`'s docstring, not silently assumed
   passed. Hard rule A2 (invented/altered input) is checked only by a
   heuristic (comparing the model's restated ET0 against the question's own
   ET0) and flags for a human rather than scoring automatically — it will
   miss invented inputs that are not restated in the model's own formula
   line.
6. **Part 4 and rubric v2 real numbers cannot exist yet** — no returned
   sheets exist for external raters, digitisation checkers, leakage
   reviewers, or the rubric v2 expert. This is a human-task dependency, not
   something the assistant can close.
7. **Consent and supervisor email**: `docs/stage7_materials/Stage7_Consent_Form.docx/.pdf`
   exist but no signed copies are on file; `docs/supervisor/supervisor_email_draft.md`
   exists but per `docs/LOG.md` (2026-09-26) the email has not been sent.

## 9. Numbers the manuscript's `\STAGEEIGHTB` placeholders will need

| Number | Source file | Column/field |
|---|---|---|
| Hardware replication result | `results/stage8b/hw_replication/replication_report.json` | `pass`, byte/class/exact agreement fields |
| Noise floor (class, exact_0.1) | `results/stage8b/noise_floor.md` | "Flip rates" tables |
| O1 rate difference + CI per model | `results/stage8b/stats/part1_tests.csv` | rows `test_id` starting `O1_`, columns `diff`, `diff_ci_low/high`, `p_holm` |
| O2 odds ratio + CI per model | `results/stage8b/stats/part1_tests.csv` | rows `O2_*`, columns `effect`, `effect_ci_low/high`, `p_holm` |
| O3 class counts | `results/stage8b/part1/o3_conflicting_values.json` | `arms.<arm>.{answer,clarify,abstain,empty}` |
| Per-arm prompted-baseline rates | `results/stage8b/part2/arm_summary.md` or `part2/scores/*_summary.json` | `invented_value_rate`, `clarify_exact_rate`, `exact_0p1_rate`, `overask_rate` |
| Factual chrF++ / abstain accuracy, prompted arms | `results/stage8b/stats/p_desc_factual_abstain.csv` | `rate_or_mean`, `ci_low`, `ci_high` per `arm`/`metric` |
| P1-P5, P7 rate differences + CIs | `results/stage8b/stats/part2_tests.csv` | rows by `test_id`, columns `diff`, `diff_ci_low/high`, `p_raw`, `p_holm`, `verdict` |
| P4c wording ("no loss" / "harms accuracy") | `results/stage8b/stats/part2_tests.csv` | `verdict` column, rows `P4c_*` |
| P6 form-of-error rates | `results/stage8b/stats/p6_descriptive.csv` | `truncation_rate`, `invented_value_rate`, `confident_error_rate_a/b_strict`, `explicit_assumption_rate` |
| Kc provenance match/mismatch counts | `results/stage8b/part3/kc_provenance_summary.md` | bulleted counts |
| Digitisation sample composition | `results/stage8b/part3/digitisation_sample_manifest.json` | `crops_covered`, `fact_type_counts` |
| Rubric v2 numeric dry-run distribution | `docs/rubric_v2/dry_run/numeric_v2_dryrun.csv` | `score`, `flagged_for_human`, `arm` (dry-run only, not for the manuscript's real numbers) |

Real (non-dry-run) rubric v2 numbers, and all Part 3/4 human-sourced
numbers, do not exist yet and must not be placed in the manuscript until the
corresponding sheet is returned and its script has been run for real.
