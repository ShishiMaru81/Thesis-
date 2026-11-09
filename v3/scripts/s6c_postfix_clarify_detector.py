#!/usr/bin/env python3
"""Stage 6C — post-hoc clarification detector for Llama clarify-retrained arm."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from s6_eval import classify_response  # noqa: E402

REQUIRED_SLOTS = {
    "et0_mm": ["et0", "et 0", "eto", "evapotranspiration", "ইটিও", "বাষ্পীভবন"],
    "rainfall_mm": ["rain", "rainfall", "rained", "precipitation", "বৃষ্টি", "বৃষ্টিপাত"],
    "area": ["acre", "decimal", "bigha", "hectare", "m²", "sqm", "square",
             "একর", "ডেসিমাল", "বিঘা", "হেক্টর", "বর্গমিটার", "জমির পরিমাণ"],
    "stage": ["initial", "early", "seedling", "mid", "middle", "flowering", "final", "late",
              "late-season", "mid-season", "mature", "প্রাথমিক", "মধ্য", "শেষ", "পর্যায়"],
}

SLOT_NAMES = {
    "et0_mm": "ET0 (mm)",
    "rainfall_mm": "rainfall (mm)",
    "area": "field size",
    "stage": "growth stage",
    "ambiguous": "which crop",
    "unknown_local_unit": "field size and unit",
}

SUBTYPE_TO_SLOT = {
    "missing_et0": "et0_mm",
    "missing_rainfall": "rainfall_mm",
    "missing_area": "area",
    "missing_stage": "stage",
    "ambiguous": "ambiguous",
    "unknown_local_unit": "unknown_local_unit",
}


def detect_missing_slots(question: str) -> list[str]:
    q_lower = question.lower()
    missing = []
    for slot, keywords in REQUIRED_SLOTS.items():
        if not any(kw in q_lower for kw in keywords):
            missing.append(slot)
    return missing


def is_numeric_output(answer_text: str) -> bool:
    patterns = [
        r"\d+\.?\d*\s*(?:mm|liter|litre|l)(?:\b|$)",
        r"\d+(?:,\d+)*(?:\.\d+)?\s*(?:mm|liter|litre)(?:\b|$)",
        r"মিমি|লিটার",
    ]
    return any(re.search(p, answer_text, re.IGNORECASE) for p in patterns)


def apply_detector(row: dict) -> dict:
    gen = row.get("generation") or row.get("answer") or ""
    row["predicted_qtype"] = classify_response(gen)

    if row.get("qtype") != "clarify":
        row["missing_slots"] = []
        row["detector_flag"] = False
        row["detector_correction"] = None
        return row

    question = row.get("question", "")
    missing = detect_missing_slots(question)
    row["missing_slots"] = missing

    has_numeric = is_numeric_output(gen)
    invented_answer = row["predicted_qtype"] == "answer" and has_numeric

    # Keyword gaps (training-aligned slots) OR any invented numeric on clarify rows.
    subtype = row.get("subtype") or ""
    slot_key = SUBTYPE_TO_SLOT.get(subtype) or (missing[0] if missing else subtype or "parameter")

    if invented_answer and (missing or row.get("category") in {
        "missing_parameter", "misspelled_crop", "unit_ambiguity"
    }):
        row["detector_flag"] = True
        row["detector_correction"] = (
            f"Please provide: {SLOT_NAMES.get(slot_key, slot_key.replace('_', ' '))}"
        )
    else:
        row["detector_flag"] = False
        row["detector_correction"] = None

    return row


def main() -> None:
    input_file = PROJECT_ROOT / "reports/s6/predictions/llama31_8b_seed42_clarify.jsonl"
    output_file = PROJECT_ROOT / "reports/s6/predictions/llama31_8b_seed42_clarify_with_detector.jsonl"

    if not input_file.exists():
        raise SystemExit(f"missing input: {input_file}")

    print(f"Loading {input_file}...")
    rows = [json.loads(line) for line in input_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    clarify_rows = [r for r in rows if r.get("qtype") == "clarify"]
    scorable_clarify = [r for r in clarify_rows if str(r.get("gold", "")).strip()]
    print(f"Processing {len(rows)} rows ({len(clarify_rows)} clarify, {len(scorable_clarify)} scorable)...")

    processed = []
    flagged_count = 0
    slot_breakdown: dict[str, int] = defaultdict(int)

    for i, row in enumerate(rows):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}...", flush=True)

        row = apply_detector(row)
        if row.get("detector_flag"):
            flagged_count += 1
            for slot in row.get("missing_slots", []):
                slot_breakdown[slot] += 1
        processed.append(row)

    print(f"\nWriting {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for row in processed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    learned_clarify = sum(
        1 for r in scorable_clarify
        if r.get("predicted_qtype") == "clarify"
    )
    clarify_total = len(scorable_clarify)
    detector_caught = sum(
        1 for r in processed
        if r.get("qtype") == "clarify"
        and str(r.get("gold", "")).strip()
        and r.get("detector_flag")
    )
    combined = learned_clarify + detector_caught

    print(f"\n{'=' * 80}")
    print("STAGE 6C — POST-HOC CLARIFICATION DETECTOR")
    print(f"{'=' * 80}")
    print(f"Total rows: {len(rows)}")
    print(f"Scorable clarify-type rows: {clarify_total}")
    print(f"\nClarify handling (n={clarify_total}):")
    if clarify_total:
        print(f"  Model learned to clarify: {learned_clarify} ({100 * learned_clarify / clarify_total:.1f}%)")
        print(f"  Detector caught invented value: {detector_caught} ({100 * detector_caught / clarify_total:.1f}%)")
        print(f"  Combined safe handling: {combined} ({100 * combined / clarify_total:.1f}%)")
    print(f"\nFlagged by missing slot:")
    for slot, count in sorted(slot_breakdown.items()):
        print(f"  {slot}: {count}")
    print(f"\nDetector applied. Output: {output_file}")


if __name__ == "__main__":
    main()
