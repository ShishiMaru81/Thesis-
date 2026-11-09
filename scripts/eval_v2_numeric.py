#!/usr/bin/env python3
"""
eval_v2_numeric.py — Stage 7 held-out numeric evaluation.

Runs LIVE inference with models/v2_run/merged_model on every numeric_runtime
question in data/v2/test.jsonl (180 rows, not the ~72 the spec estimated from
v1's test set), parses each generation with irrigation_bridge, and scores
against the gold grounding recorded at corpus-generation time.

NOTE ON WHAT "the model's assistant turn" MEANS HERE: test.jsonl's
messages[1]["content"] is the GOLD tool call written during corpus generation,
not a model output. Evaluating gold against itself through the same calculator
that produced it would be tautological — every row would trivially score 100%,
telling us nothing about the trained model. This script instead prompts the
model with messages[0] (the question) and evaluates what IT generates.

Metrics:
  A. parse success rate
  B. per-slot exact-match accuracy + macro-F1 across the six slot types
     (precision_k = correct_k / n_parsed, recall_k = correct_k / n_total,
      F1_k = harmonic mean; macro-F1 = mean of the six F1_k)
  C. net_mm / liters MAE, RMSE (full set and non-zero-gold subset), MAPE
     (non-zero subset). Rows with no computable prediction (parse or calc
     failure) are EXCLUDED from continuous-error metrics and reported
     separately as a fraction, rather than imputed with an arbitrary value.
  D. per-language, per-tier, per-crop, per-stage, per-area-unit breakdowns.

Outputs: reports/v2_numeric_eval/v2_numeric_eval.csv
         reports/v2_numeric_eval/v2_numeric_eval_summary.txt
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import irrigation_calc as ic  # noqa: E402
import irrigation_bridge as ib  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_DIR = PROJECT_ROOT / "models" / "v2_run" / "merged_model"
TEST_PATH = PROJECT_ROOT / "data" / "v2" / "test.jsonl"
OUT_DIR = PROJECT_ROOT / "reports" / "v2_numeric_eval"
MAX_LEN = 1024
SLOT_KEYS = ("crop", "stage", "et0_mm", "rainfall_mm", "area_value", "area_unit")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def generate(model, tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def slot_eq(gold, pred, key: str) -> bool:
    if pred is None:
        return False
    if key in ("et0_mm", "rainfall_mm", "area_value"):
        try:
            return abs(float(gold) - float(pred)) < 1e-6
        except (TypeError, ValueError):
            return False
    return str(gold).strip().casefold() == str(pred).strip().casefold()


def mae_rmse(pairs: list[tuple[float, float]]) -> tuple[float | None, float | None]:
    if not pairs:
        return None, None
    errs = [abs(g - p) for g, p in pairs]
    sq = [(g - p) ** 2 for g, p in pairs]
    return sum(errs) / len(errs), math.sqrt(sum(sq) / len(sq))


def mape(pairs: list[tuple[float, float]]) -> float | None:
    vals = [abs(g - p) / g for g, p in pairs if g != 0]
    return (sum(vals) / len(vals) * 100) if vals else None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows = [r for r in load_jsonl(TEST_PATH) if r["question_type"] == "numeric_runtime"]
    print(f"Loaded {len(rows)} numeric_runtime test rows "
          f"(spec estimated ~72 from v1; v2's actual test allocation is 180)")

    print(f"\nLoading merged model from {MERGED_DIR} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MERGED_DIR), max_seq_length=MAX_LEN, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded.\n")

    csv_rows = []
    error_cases = []

    slot_correct = {k: 0 for k in SLOT_KEYS}
    n_parsed = 0
    n_calc_ok = 0

    mm_pairs_full: list[tuple[float, float]] = []
    liters_pairs_full: list[tuple[float, float]] = []
    mm_pairs_nz: list[tuple[float, float]] = []
    liters_pairs_nz: list[tuple[float, float]] = []

    by_lang = defaultdict(lambda: {"n": 0, "all_match": 0, "mm_pairs": []})
    by_tier = defaultdict(lambda: {"n": 0, "all_match": 0, "mm_pairs": []})
    by_crop = Counter()
    by_stage = Counter()
    by_area_unit = Counter()
    n_floored_gold = 0

    for i, row in enumerate(rows):
        question = row["messages"][0]["content"]
        gold_params = row["grounding"]["params"]
        gold_net_mm = row["grounding"]["expected_net_mm"]
        gold_liters = row["grounding"]["expected_liters"]
        floored = row["grounding"]["floored_at_zero"]
        tier = row["grounding"]["tier"]
        lang = row["lang"]

        by_crop[row["crop"]] += 1
        by_stage[gold_params["stage"]] += 1
        by_area_unit[gold_params["area_unit"]] += 1
        if floored:
            n_floored_gold += 1

        model_output = generate(model, tokenizer, question)
        result = ib.answer_numeric(model_output, ic.compute)

        parsed = bool(result["parsed"])
        parse_error = result["parse_error"]
        pred_params = result["params"]
        net_mm_pred = result["net_mm"]
        liters_pred = result["liters"]
        calc_error = result["calc_error"]

        row_slot_match = {}
        if parsed:
            n_parsed += 1
            for k in SLOT_KEYS:
                ok = slot_eq(gold_params[k], pred_params.get(k), k)
                row_slot_match[k] = ok
                if ok:
                    slot_correct[k] += 1
        all_slots_match = parsed and all(row_slot_match.get(k, False) for k in SLOT_KEYS)

        if parsed and calc_error is None:
            n_calc_ok += 1

        if error_type := (
            None if (parsed and calc_error is None) else
            f"parse_error:{parse_error}" if not parsed else
            f"calc_error:{calc_error}"
        ):
            error_cases.append({
                "id": row["id"], "lang": lang, "error_type": error_type,
                "model_output": model_output,
            })
        elif not all_slots_match:
            error_type = "slot_mismatch"
            error_cases.append({
                "id": row["id"], "lang": lang, "error_type": error_type,
                "gold": gold_params, "pred": pred_params,
            })
        else:
            error_type = "none"

        if net_mm_pred is not None:
            mm_pairs_full.append((gold_net_mm, net_mm_pred))
            liters_pairs_full.append((gold_liters, liters_pred))
            if gold_net_mm > 0:
                mm_pairs_nz.append((gold_net_mm, net_mm_pred))
                liters_pairs_nz.append((gold_liters, liters_pred))

        by_lang[lang]["n"] += 1
        by_lang[lang]["all_match"] += int(all_slots_match)
        if net_mm_pred is not None:
            by_lang[lang]["mm_pairs"].append((gold_net_mm, net_mm_pred))

        by_tier[tier]["n"] += 1
        by_tier[tier]["all_match"] += int(all_slots_match)
        if net_mm_pred is not None:
            by_tier[tier]["mm_pairs"].append((gold_net_mm, net_mm_pred))

        csv_rows.append({
            "id": row["id"], "crop": row["crop"], "stage": gold_params["stage"], "lang": lang,
            "area_value": gold_params["area_value"], "area_unit": gold_params["area_unit"],
            "et0_mm": gold_params["et0_mm"], "rainfall_mm": gold_params["rainfall_mm"],
            "parsed": parsed, "parse_error": parse_error or "",
            "slot_match": all_slots_match,
            "net_mm_gold": gold_net_mm, "net_mm_pred": net_mm_pred,
            "liters_gold": gold_liters, "liters_pred": liters_pred,
            "floored": floored, "tier": tier, "error_type": error_type,
        })

        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(rows)} done")

    elapsed = time.time() - t0
    print(f"\nInference + scoring done in {elapsed:.1f}s ({elapsed/len(rows):.2f}s/row)")

    # ---------------------------------------------------------------- CSV
    csv_path = OUT_DIR / "v2_numeric_eval.csv"
    fieldnames = ["id", "crop", "stage", "lang", "area_value", "area_unit", "et0_mm",
                  "rainfall_mm", "parsed", "parse_error", "slot_match", "net_mm_gold",
                  "net_mm_pred", "liters_gold", "liters_pred", "floored", "tier", "error_type"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Wrote {csv_path}")

    # ---------------------------------------------------------------- metrics
    n = len(rows)
    parse_rate = n_parsed / n * 100

    precision = {k: (slot_correct[k] / n_parsed if n_parsed else 0.0) for k in SLOT_KEYS}
    recall = {k: slot_correct[k] / n for k in SLOT_KEYS}
    f1 = {k: (2 * precision[k] * recall[k] / (precision[k] + recall[k])
              if (precision[k] + recall[k]) > 0 else 0.0) for k in SLOT_KEYS}
    macro_f1 = sum(f1.values()) / len(f1)

    mae_full, rmse_full = mae_rmse(mm_pairs_full)
    mae_full_l, rmse_full_l = mae_rmse(liters_pairs_full)
    mae_nz, rmse_nz = mae_rmse(mm_pairs_nz)
    mae_nz_l, rmse_nz_l = mae_rmse(liters_pairs_nz)
    mape_nz = mape(mm_pairs_nz)

    lang_table = {}
    for lg, d in by_lang.items():
        m, r = mae_rmse(d["mm_pairs"])
        lang_table[lg] = {"n": d["n"], "accuracy": d["all_match"] / d["n"] * 100 if d["n"] else 0,
                          "mae_mm": m, "rmse_mm": r, "n_with_pred": len(d["mm_pairs"])}

    tier_table = {}
    for t, d in sorted(by_tier.items()):
        m, r = mae_rmse(d["mm_pairs"])
        tier_table[t] = {"n": d["n"], "accuracy": d["all_match"] / d["n"] * 100 if d["n"] else 0,
                         "mae_mm": m, "rmse_mm": r, "n_with_pred": len(d["mm_pairs"])}

    n_excluded_full = n - len(mm_pairs_full)
    n_nz_gold = sum(1 for r in csv_rows if r["net_mm_gold"] > 0)
    n_excluded_nz = n_nz_gold - len(mm_pairs_nz)

    # ---------------------------------------------------------------- summary
    lines = []
    def p(s=""):
        lines.append(s)

    p("=" * 78)
    p("STAGE 7 — HELD-OUT NUMERIC EVALUATION SUMMARY")
    p("=" * 78)
    p(f"Model: {MERGED_DIR}")
    p(f"Test set: {TEST_PATH}  (numeric_runtime rows: {n})")
    p("NOTE: spec estimated ~72 rows from v1's test set; v2's actual allocation is 180.")
    p("NOTE: evaluated LIVE model generations (greedy decode), not the gold tool call "
      "stored in messages[1] — comparing gold to itself would be tautological.")
    p(f"Inference wall time: {elapsed:.1f}s ({elapsed/n:.2f}s/row)")
    p("")

    p("-- A. Parse success --")
    p(f"parse_success_rate: {parse_rate:.1f}%  ({n_parsed}/{n})  target >=95%: "
      f"{'PASS' if parse_rate >= 95 else 'FAIL'}")
    p("")

    p("-- B. Slot accuracy (recall = correct/total_rows; precision = correct/parsed_rows) --")
    p(f"{'slot':<14}{'accuracy(recall)':>18}{'precision':>12}{'F1':>10}")
    for k in SLOT_KEYS:
        p(f"{k:<14}{recall[k]*100:>17.1f}%{precision[k]*100:>11.1f}%{f1[k]:>10.3f}")
    p(f"\nmacro-F1 across 6 slots: {macro_f1:.3f}   target >=0.90: "
      f"{'PASS' if macro_f1 >= 0.90 else 'FAIL'}")
    p("")

    p("-- C. End-to-end numeric error --")
    p(f"rows with a computable prediction: {len(mm_pairs_full)}/{n} "
      f"({n_excluded_full} excluded: parse or calc failure)")
    p(f"MAE_mm  (full, n={len(mm_pairs_full)}): {mae_full}")
    p(f"RMSE_mm (full, n={len(mm_pairs_full)}): {rmse_full}")
    p(f"MAE_liters  (full): {mae_full_l}")
    p(f"RMSE_liters (full): {rmse_full_l}")
    p(f"\nfraction of gold rows floored to 0mm: {n_floored_gold}/{n} ({n_floored_gold/n*100:.1f}%)")
    p(f"non-zero-gold rows: {n_nz_gold}/{n}  ({n_excluded_nz} excluded from non-zero metrics "
      f"due to parse/calc failure)")
    p(f"MAE_mm  (non-zero subset, n={len(mm_pairs_nz)}): {mae_nz}")
    p(f"RMSE_mm (non-zero subset, n={len(mm_pairs_nz)}): {rmse_nz}")
    p(f"MAE_liters  (non-zero subset): {mae_nz_l}")
    p(f"RMSE_liters (non-zero subset): {rmse_nz_l}")
    p(f"MAPE (non-zero subset, net_mm): "
      f"{f'{mape_nz:.1f}%' if mape_nz is not None else 'n/a'}")
    p("")

    p("-- D. Per-language --")
    p(f"{'lang':<6}{'n':>5}{'accuracy(all-slots)':>22}{'MAE_mm':>10}{'n_w/pred':>10}")
    for lg in ("en", "bn"):
        d = lang_table.get(lg, {"n": 0, "accuracy": 0, "mae_mm": None, "n_with_pred": 0})
        mae_s = f"{d['mae_mm']:.3f}" if d["mae_mm"] is not None else "n/a"
        p(f"{lg:<6}{d['n']:>5}{d['accuracy']:>21.1f}%{mae_s:>10}{d['n_with_pred']:>10}")
    p("")

    p("-- D. Per-tier (effective-rainfall tier) --")
    p(f"{'tier':<8}{'range':<12}{'n':>5}{'accuracy(all-slots)':>22}{'MAE_mm':>10}{'n_w/pred':>10}")
    tier_ranges = {1: "<=5mm", 2: "5-15mm", 3: ">15mm"}
    for t in (1, 2, 3):
        d = tier_table.get(t, {"n": 0, "accuracy": 0, "mae_mm": None, "n_with_pred": 0})
        mae_s = f"{d['mae_mm']:.3f}" if d["mae_mm"] is not None else "n/a"
        p(f"{t:<8}{tier_ranges[t]:<12}{d['n']:>5}{d['accuracy']:>21.1f}%{mae_s:>10}{d['n_with_pred']:>10}")
    p("")

    p(f"-- D. Crop coverage: {len(by_crop)} distinct crops / 28 total in test set --")
    for crop, cnt in by_crop.most_common():
        p(f"  {crop:<16} {cnt}")
    p("")

    p(f"-- D. Stage coverage: {len(by_stage)} distinct stages (expect 3) --")
    for stage, cnt in sorted(by_stage.items()):
        p(f"  {stage:<14} {cnt}")
    p("")

    p(f"-- D. Area-unit distribution --")
    for unit, cnt in by_area_unit.most_common():
        p(f"  {unit:<10} {cnt}")
    p("")

    p(f"-- Error cases ({len(error_cases)}) --")
    for e in error_cases[:50]:
        p(f"  id={e['id']}  lang={e['lang']}  type={e['error_type']}")
    if len(error_cases) > 50:
        p(f"  ... and {len(error_cases)-50} more (see CSV for full list)")

    summary_path = OUT_DIR / "v2_numeric_eval_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {summary_path}")

    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
