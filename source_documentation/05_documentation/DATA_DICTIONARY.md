# Data dictionary

Keys below are the keys present in the files, not a wished-for schema.

## Crop JSON (`03_source_calendars/json`, 28 files)

Top-level keys seen in every file: `crop_name_bn`, `crop_name_en`, `source_document`, `source_image`, `source_image_sha256`, `extraction_timestamp`, `human_verified`, `irrigation_features`, `computable_link`, `_extraction_audit`.

`Coconut.json` also has `extraction_model`. Most other files also have `source_image_path` and `source_image_bytes`.

`human_verified` is false in all 28 files.

### `irrigation_features`

A dictionary of calendar fields. Keys inside each field object: ['calendar_system_in_source', 'calendar_system_normalised', 'condition_bn', 'condition_en', 'dedup_note', 'depth_cm_max', 'depth_cm_min', 'duplicate_of', 'exclude_from_qa_generation', 'irrigation_adjustment_details_bn', 'irrigation_adjustment_details_en', 'irrigation_adjustment_stated', 'irrigations_max', 'irrigations_min', 'missing_months_gregorian', 'modifier_type', 'months', 'needs_human_review', 'needs_reextraction', 'note', 'numeric_backfill_declined', 'numeric_provenance', 'other_day_candidates', 'parsed_by', 'parsed_day_values', 'parser_confidence', 'period_bn', 'period_en', 'provenance', 'radius_m', 'soil_preference_bn', 'soil_preference_en', 'source', 'source_quote_bn', 'stages', 'timing_bn', 'timing_en', 'value', 'value_bn', 'value_cm_max', 'value_cm_min', 'value_days_max', 'value_days_min', 'value_en', 'wheel_complete'].

`provenance` values and how many fields carry each one: {'explicit': 257, 'derived': 75, 'partial': 10, 'absent': 321, 'derived_from_shared_passage': 9}.

`explicit` was read off the calendar. `derived` was obtained from other stated fields by a stated rule. `absent` is not on the calendar and must not be imputed. Stage 0 skips `absent`, and also skips a field flagged `exclude_from_qa_generation`, `needs_human_review` or `numeric_backfill_declined`.

### `_extraction_audit`

Per-file summary copied to `04_source_verification/verification_by_crop.csv`: field counts, provenance counts, the list of fields still needing review, `human_verified`, `verification_note`, and `image_hash_verified` where that key exists.

## Seed row (`v3/data/seed/seed_corpus.jsonl`, 902 rows, not included in this zip)

Keys present on at least one row: ['answer', 'answer_kind', 'crop', 'crop_bn', 'fact_id', 'gold_trace', 'lang', 'origin', 'provenance', 'qtype', 'question', 'row_id', 'scenario', 'scenario_hash', 'source_id'].

Stage 0 writes one row per fact and language. `qtype` is `factual`, `numeric_runtime` or `abstain`. Numeric answers come from `irrigation_calc.py`. The scenario values are drawn with `random.Random(42)` and identified by `scenario_hash`.

## Splits

| File | Rows | Role |
|---|---|---|
| `data/splits/train_seed.jsonl` | 632 | Seed rows later augmented. Not in this zip. |
| `data/splits/val.jsonl` | 88 | Validation. Not in this zip. |
| `data/splits/test.jsonl` | 182 | Seed test. Included in folder 06. SHA-256 matches `test.sha256`: True. |

632 + 88 + 182 = 902, the seed count in `gate0_seed_corpus.json`.

## Augmented and hardened files (not in this zip)

Gate 2: `train_augmented.jsonl`, 3,552 rows = 632 seed + 2,610 synthetic + 310 noise. No paraphrase or back-translation rows.

Gate 2b: the same 3,552 rows rendered to `data/rendered/train.jsonl` as `prompt` / `completion`.

Stage 6B: `s6b_clarify_train.jsonl` (300) merged into `train_augmented_clarify.jsonl` (3,852).

Gate 4: `test_hardened.jsonl`, 422 rows = 182 seed + 240 adversarial. 60 of those have `gold_status` = `pending_human` (40 out-of-domain, 20 conflicting values) and an empty answer.
