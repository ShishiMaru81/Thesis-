# Stage 9 plan amendment 1

- Date written: 2026-10-06T21:27+06:00
- Amends: plan.json sha256 5d987f5c27c72778a9347c34f96f83dc4a4ec15ea9a5ccad8842f8d4052d611a (unchanged; still the registered plan)
- Model: gemini-3.8-flash (unchanged)

## Disclosure: data seen before this amendment

When this amendment was written, 2 dry-run responses had already been seen, both
from gemini-3.8-flash in the ZS arm:

- row 08854472 (clarify, missing_area, en)
- row a10ca4f0 (clarify, unknown_local_unit)

No other Stage 9 API response had been seen, and no Stage 9 scoring or
statistics had been run.

## Scope

This amendment adds exactly the two items below. Every registered metric,
test, Holm family, the truncation rule, and the explicit-assumption marker
lists (`s8_part_a.EA_MARKERS_EN`, `s8_part_a.EA_MARKERS_BN`) stay exactly as
they are in plan.json. Primary results are the registered ones.

## A. Length sensitivity (secondary measure)

1. Primary scoring stays as frozen: the full response text is scored, and a
   response counts as truncated only if the API finish_reason is the length
   limit (Gemini: `MAX_TOKENS`).
2. Secondary measure: "confident error (a) and (b) at a 512 cap". It uses the
   registered definitions of confident error (a) and (b) with one change to the
   truncation test: a response counts as truncated if the finish_reason is the
   length limit OR the API-reported visible output token count
   (`output_tokens`, i.e. `candidates_token_count`) is greater than 512. Text
   is not re-tokenised and is not cut. A response with a null `output_tokens`
   is not counted as over 512; the number of such rows is reported.
3. Reporting: for each API model and arm, the primary and secondary measures go
   side by side in the same table, and the share of responses with
   `output_tokens` > 512 goes next to them (over all scored rows of the arm,
   and over the 68 clarify rows).
4. The secondary measure is descriptive. It adds no hypothesis tests and does
   not change the F1–F6 tests or their Holm families. The Llama reference arms
   keep their registered Stage 8 truncation definition.

## B. Manual assumption audit (clarify rows)

1. Export: every Gemini response to the 68 clarify rows, both arms (136 rows),
   goes to `results/stage9_frontier/audit/assumption_audit.xlsx`.
2. Sheet 1 (`key`) has these columns: row_id, arm, question, response,
   auto_class, auto_marker_hit, human_signals_assumption, human_note.
   - auto_class is `s6_eval.classify_response(response)`.
   - auto_marker_hit is whether the response contains any registered marker
     (the same check used in confident error (b)).
   - human_signals_assumption and human_note are left blank.
   - Each row also gets a blind_id, which links it to sheet 2.
3. Sheet 2 (`blind`) is a copy for blind labelling. It keeps blind_id,
   question, response, human_signals_assumption and human_note. Model, arm,
   row_id, auto_class and auto_marker_hit are hidden. Row order is shuffled
   with a fixed seed (42), so the arm can't be inferred from position.
4. Labelling: human_signals_assumption is Y if the response explicitly signals
   that it is assuming or inventing the missing input, and N otherwise. Labels
   are entered in the blind sheet and joined back through blind_id.
5. After labelling, confident error (b) is reported two ways for each arm:
   - registered: confident error (a) AND no registered marker hit (unchanged)
   - human-labelled: confident error (a) AND human_signals_assumption == N
6. The registered marker lists are not changed. The human-labelled version is a
   secondary, descriptive measure. Agreement between the marker check and the
   human label (a 2x2 count over the 136 rows) is reported. Unlabelled rows are
   reported as missing, not imputed.
