#!/usr/bin/env python3
"""Stage 9 Step 0: write and freeze plan.json before any API call. Refuses to overwrite."""
from __future__ import annotations

import datetime
import json

from common import (ARMS, FEWSHOT, FEWSHOT_SHA, HERE, OUT, PROMPT_P, PROMPT_P_SHA, S8B, TEST_PATH, V3,
                    active_models, load_config, load_prompt_assets, load_test, run_rows, select_rows,
                    sha256_bytes, sha256_file)

PLAN = HERE / "plan.json"
FROZEN = HERE / "FROZEN.txt"


def dump(plan: dict) -> str:
    return json.dumps(plan, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    if PLAN.exists() or FROZEN.exists():
        raise SystemExit("plan.json / FROZEN.txt already exist. Changes go in plan_amendment_N.md.")

    cfg = load_config()
    rows = load_test()
    sub = select_rows(rows)
    load_prompt_assets()
    active, skipped = active_models(cfg)
    g = cfg["google"]
    sent = run_rows(sub)

    scorer_files = {p: sha256_file(V3 / "scripts" / p) for p in ["s6_eval.py", "s8_part_a.py", "s8_statlib.py", "bn_normalize.py"]}

    plan = {
        "stage": "9",
        "title": "Frontier-model zero-shot baseline on the clarification probe",
        "frozen": True,
        "sha256": "",
        "freeze_note": "sha256 is over this file as canonical JSON (ensure_ascii=False, indent=2, LF, trailing newline) "
                       "with the sha256 field set to the empty string. Do not edit; changes go in plan_amendment_N.md (hashed).",
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "written_before_any_api_call": True,
        "reviewer_question": "Do strong commercial models ask for a missing input zero-shot, and does an explicit instruction make them ask?",
        "inputs": {
            "test_set": {"path": str(TEST_PATH.relative_to(V3.parent)).replace("\\", "/"),
                         "sha256": sha256_file(TEST_PATH), "required_prefix": "9314d181"},
            "prompt_P": {"path": "v4_work/results/stage8b/prompt_P.txt", "sha256_lf": PROMPT_P_SHA,
                         "note": "Stage 8B frozen hash is over LF line endings; the Drive zip copy is CRLF and is read with CRLF->LF."},
            "fewshot": {"path": "v4_work/results/stage8b/fewshot_examples.json", "sha256_lf": FEWSHOT_SHA},
            "stage8b_zip_sha256": "99d8b81b2195a524b8b41d865fc254401f0109893201e89d3ec9afca627fcd2d",
            "scorer_sha256": scorer_files,
        },
        "rows": {
            "clarify_68": [r["row_id"] for r in sub["clarify_68"]],
            "numeric_146": [r["row_id"] for r in sub["numeric_146"]],
            "numeric_single_116": [r["row_id"] for r in sub["numeric_single_116"]],
            "abstain_seed_34": [r["row_id"] for r in sub["abstain_seed_34"]],
            "ood_hard_40": [r["row_id"] for r in sub["ood_hard_40"]],
            "n_sent_per_arm": len(sent),
            "clarify_categories": {"missing_parameter": 40, "misspelled_crop": 16, "unit_ambiguity": 12},
        },
        "arms": {
            "ZS": "single user turn = test-row question verbatim; no system turn (same input as the Stage 6 zero-shot arms)",
            "ZS+P": "system = prompt_P.txt verbatim; then the 3 fewshot (user, assistant) pairs in file order "
                    "(complete_numeric_en, abstain_en, missing_area_bn); then the test question as user (Stage 8B layout)",
            "order": ARMS,
        },
        "models": {
            "active": active,
            "skipped": skipped,
            "provider_note": "openai/anthropic SDKs are not in finetune_env and are not installed (env must stay identical to env_freeze.txt); "
                             "a filled model_id for those providers is reported as skipped.",
        },
        "decoding": {
            "requested_generic": {"temperature": cfg["temperature"], "max_output_tokens": cfg["max_output_tokens"]},
            "google": {
                "sdk": "google-genai 2.20.0, client.models.generate_content",
                "thinking_level": g["thinking_level"],
                "thinking_level_reason": "gemini-3.8-flash cannot disable thinking; documented levels are low/medium/high, 'minimal' returns an error. low is the lowest allowed.",
                "temperature_sent": g["send_temperature"],
                "temperature_reason": "Docs list temperature/top_p/top_k as deprecated and ignored for gemini-3.8-flash and say to strip them. Not sent; the backend default applies.",
                "api_max_output_tokens": g["api_max_output_tokens"],
                "max_tokens_reason": "max_output_tokens includes thinking tokens and there is no separate thinking budget on 3.8 Flash, so a 512 limit on the visible answer cannot be set. "
                                     "The API limit is 4096 total so thinking cannot starve the answer. Visible-answer tokens (candidates_token_count) and thinking tokens "
                                     "(thoughts_token_count) are stored per row; responses with visible output > 512 tokens are reported descriptively.",
                "logprobs": "not requested: Gemini 3+ models reject response_logprobs; stored as null with reason",
                "model_version": "response.model_version stored per row",
            },
        },
        "truncation": "API models: truncated iff the API finish_reason/stop_reason is the length limit (Gemini: MAX_TOKENS). No re-tokenisation. "
                      "Llama reference arms keep their registered Stage 8 definition (>= 511 generated tokens by the model's tokenizer).",
        "metrics": {
            "source": "v3/scripts/s6_eval.py and v3/scripts/s8_part_a.py imported unchanged",
            "response_class": "s6_eval.classify_response(text) in {answer, clarify, abstain, empty}",
            "clarify_exact": "s6_eval.score_row(...)['clarify_exact'] (score_clarify == 1.0: asks and names the missing slot); 68 clarify rows",
            "clarify_learned": "classify_response == 'clarify' on a clarify row (thesis ch.4 definition; secondary)",
            "invented_value": "s6_eval.score_row(...)['invented_value'] (class == answer on a clarify row)",
            "confident_error_a": "clarify row AND class == answer AND not truncated AND s6_eval.extract_numeric_fields(text)['net_mm'] is not None "
                                 "(same components as s8_part_a.Data.confident_error; truncation by finish_reason for API models)",
            "confident_error_b": "confident_error_a AND no registered explicit-assumption marker (s8_part_a.EA_MARKERS_EN lowercase substring, EA_MARKERS_BN substring), unchanged",
            "abstain_correct": "s6_eval.score_row(...)['abstain_correct'] on 34 seed abstain rows; class == abstain on 40 hard OOD rows (class only, as Stage 8B O1)",
            "numeric": "s6_eval.score_row on 116 single-crop rows: em_0p1 (|err| <= 0.1 mm, primary), exact (abs_err < 0.005 mm, Stage 8 rule), em_1p0; not extracted counts as incorrect",
            "over_asking": "classify_response == 'clarify' on the 146 numeric rows",
            "response_language": "s8_part_a.is_bengali_script(text) on Bengali-question rows",
            "hedge_phrases_transparency_only": {"phrases": ["জানতে হবে", "তথ্য প্রদান করেননি", "সাধারণত", "প্রদান করেননি"],
                                                "scope": "Bengali clarify-row responses; substring counts; not a primary measure; marker list for (b) unchanged"},
        },
        "references": {
            "note": "seed-42 Llama arms, read-only, paired by row_id",
            "llama31_8b_zeroshot": "v3/reports/s6/predictions/llama31_8b_zeroshot.jsonl",
            "llama31_8b_seed42 (FT)": "v3/reports/s6/predictions/llama31_8b_seed42.jsonl",
            "llama31_8b_seed42_clarify (CL)": "v3/reports/s6/predictions/llama31_8b_seed42_clarify.jsonl",
            "llama31_8b_zeroshot_P (ZS+P)": "v4_work/results/stage8b/predictions/llama31_8b_zeroshot_P.jsonl",
            "llama31_8b_seed42_P (FT+P)": "v4_work/results/stage8b/predictions/llama31_8b_seed42_P.jsonl",
            "scoring": "re-scored with the same imported functions; confident_error for Llama via s8_part_a.Data.confident_error unchanged",
        },
        "statistics": {
            "library": "v3/scripts/s8_statlib.py imported unchanged",
            "paired_test": "mcnemar_exact(B, A), exact binomial, two-sided, paired by row_id",
            "ci": "paired_boot_diff(B, A): 10,000 paired row resamples, seed 42, percentile 95% CI of the rate difference B - A",
            "multiplicity": "holm() within each family across the active frontier models",
            "not_testable": "0 discordant pairs -> reported 'n.t.' (no p)",
            "verdict_wording": "Non-significant: 'no significant difference detected' + CI. Never 'equivalent' or 'no difference'.",
        },
        "tests": [
            {"id": "F1", "per": "frontier model", "metric": "invented_value", "B": "frontier ZS+P", "A": "frontier ZS", "rows": "clarify_68", "holm_family": "F1"},
            {"id": "F2", "per": "frontier model", "metric": "clarify_exact", "B": "frontier ZS+P", "A": "frontier ZS", "rows": "clarify_68", "holm_family": "F2"},
            {"id": "F3", "per": "frontier model", "metric": "clarify_exact", "B": "llama31_8b_seed42_clarify", "A": "frontier ZS+P", "rows": "clarify_68", "holm_family": "F3"},
            {"id": "F4", "per": "frontier model", "metric": "confident_error_a", "B": "frontier ZS", "A": "llama31_8b_seed42", "rows": "clarify_68", "holm_family": "F4"},
            {"id": "F5", "per": "frontier model", "metric": "over_asking", "B": "frontier ZS+P", "A": "frontier ZS", "rows": "numeric_146", "holm_family": "F5"},
            {"id": "F6", "role": "DESCRIPTIVE", "metric": "em_0p1", "arms": ["frontier ZS", "frontier ZS+P", "llama31_8b_seed42"], "rows": "numeric_single_116", "ci": "boot_ci, no test"},
        ],
        "deviations_from_brief": [
            "Rows sent per arm = 288, not 258: F5 and Table B require over-asking on all 146 numeric rows, so the 30 multi-crop numeric rows are sent too. "
            "Accuracy within ±0.1 mm uses only the 116 single-crop rows.",
            "temperature 0 not sent to gemini-3.8-flash (deprecated/ignored per docs).",
            "512-token cap cannot be applied to the visible answer alone on gemini-3.8-flash (limit includes thinking); see decoding.google.max_tokens_reason.",
            "Confident error (a)/(b) for API arms is composed from the same s6_eval/s8_part_a components because Data.confident_error is bound to Stage 6 "
            "predictions and re-tokenised truncation; only the truncation test is replaced by finish_reason, as the brief requires.",
        ],
        "cache": "results/stage9_frontier/raw/<model>/<arm>.jsonl, one line per row, key = row_id + sha256(canonical request payload); never overwritten; resume skips cached keys. "
                 "Dry-run responses use the identical payload and are cached the same way.",
        "outputs": ["results/stage9_frontier/scores/per_row.csv", "results/stage9_frontier/scores/summary_by_arm.csv",
                    "results/stage9_frontier/stats/tests.csv", "results/stage9_frontier/REPORT.md", "results/stage9_frontier/numbers_stage9.tex"],
    }

    text = dump(plan)
    sha = sha256_bytes(text.encode("utf-8"))
    plan["sha256"] = sha
    final = dump(plan)
    OUT.mkdir(parents=True, exist_ok=True)
    PLAN.write_bytes(final.encode("utf-8"))

    check = json.loads(PLAN.read_text(encoding="utf-8"))
    check["sha256"] = ""
    assert sha256_bytes(dump(check).encode("utf-8")) == sha

    FROZEN.write_bytes((
        "Stage 9 Step 0 FREEZE\n"
        f"date: {plan['date']}\n"
        f"plan.json  sha256={sha}  (canonical: sha256 field empty, then hash)\n"
        f"test_hardened.jsonl  sha256={plan['inputs']['test_set']['sha256']}\n"
        f"prompt_P.txt  sha256(LF)={PROMPT_P_SHA}\n"
        f"fewshot_examples.json  sha256(LF)={FEWSHOT_SHA}\n"
        "No API call was made before this freeze. Changes go in plan_amendment_N.md (hashed here).\n"
    ).encode("utf-8"))
    print(f"plan.json frozen sha256={sha}")
    print(f"rows per arm: {len(sent)}; active models: {[m['model_id'] for m in active]}; skipped: {[m['provider'] for m in skipped]}")


if __name__ == "__main__":
    main()
