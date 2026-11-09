#!/usr/bin/env python3
"""Stage 7 — build the blinded 800-judgment rating sheet.

200 items (100 bn + 100 en, same underlying questions) x 4 arms = 800 rows.

Blinding:
  * No model/arm name anywhere in the rating sheet.
  * Rows are shuffled (seed 42) so the four outputs for one question are not
    adjacent.
  * The arm behind each judgment_id is written to a SEPARATE key file that
    must not be shared with annotators.

Columns: every scalar test-set column (nested fields serialised to JSON),
the model's answer, and empty rating columns.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

PROJECT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT / "data" / "human_eval" / "bilingual_100"
S6_PRED = PROJECT / "reports" / "s6" / "predictions"
S7_PRED_BN = PROJECT / "reports" / "s7" / "predictions_translated_bn"
OUT_DIR = PROJECT / "reports" / "s7"
SEED = 42

# Arm -> blinded label. Labels are only used in the key file.
ARMS = {
    "llama31_8b_zeroshot": "A",
    "llama31_8b_seed42": "B",
    "llama31_8b_seed42_clarify": "C",
    "mistral_nemo_seed42": "D",
}

# Test-set columns kept in the sheet (order matters). Nested -> JSON string.
TEST_COLS = [
    "row_id", "item_id", "source_id", "fact_id", "pair_id", "lang", "qtype",
    "category", "subtype", "origin", "gold_status", "answer_kind", "crop",
    "crop_bn", "crops", "scenario_hash", "scenario_hashes", "translation_status",
    "question", "answer",
]
RATING_COLS = ["rating_1_5", "would_give_farmer", "notes"]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def cell(v):
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def load_generations(arm: str) -> dict[str, str]:
    gens: dict[str, str] = {}
    for p in load_jsonl(S6_PRED / f"{arm}.jsonl"):
        gens[p["row_id"]] = p.get("generation") or ""
    extra = S7_PRED_BN / f"{arm}.jsonl"
    if extra.exists():
        for p in load_jsonl(extra):
            gens[p["row_id"]] = p.get("generation") or ""
    return gens


def main() -> None:
    items = load_jsonl(SAMPLE_DIR / "sample_100_bn.jsonl") + \
            load_jsonl(SAMPLE_DIR / "sample_100_en.jsonl")
    assert len(items) == 200, len(items)

    gens_by_arm = {arm: load_generations(arm) for arm in ARMS}
    missing = [(arm, it["row_id"]) for arm, g in gens_by_arm.items()
               for it in items if it["row_id"] not in g]
    if missing:
        raise SystemExit(
            f"{len(missing)} (arm,row) pairs have no generation. "
            f"Run scripts/s7_gen_translated_bn.py first. First: {missing[:3]}")

    judgments = []
    for it in items:
        for arm in ARMS:
            judgments.append((it, arm, gens_by_arm[arm][it["row_id"]]))
    assert len(judgments) == 800

    rng = random.Random(SEED)
    rng.shuffle(judgments)

    sheet_rows, key_rows = [], []
    for i, (it, arm, gen) in enumerate(judgments, start=1):
        jid = f"J{i:03d}"
        row = {"judgment_id": jid}
        for c in TEST_COLS:
            row["gold_answer" if c == "answer" else c] = cell(it.get(c))
        row["model_answer"] = gen
        for c in RATING_COLS:
            row[c] = ""
        sheet_rows.append(row)
        key_rows.append({
            "judgment_id": jid, "row_id": it["row_id"], "pair_id": it["pair_id"],
            "lang": it["lang"], "qtype": it["qtype"],
            "arm_label": ARMS[arm], "arm": arm,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = list(sheet_rows[0].keys())

    # ---- CSV (UTF-8 with BOM so Excel opens Bengali correctly)
    csv_path = OUT_DIR / "stage7_rating_sheet.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(sheet_rows)

    # ---- XLSX
    wb = Workbook()
    ws = wb.active
    ws.title = "ratings"
    ws.append(header)
    for r in sheet_rows:
        ws.append([r[h] for h in header])

    hdr_fill = PatternFill("solid", fgColor="DDDDDD")
    rate_fill = PatternFill("solid", fgColor="FFF2CC")
    for ci, h in enumerate(header, start=1):
        c = ws.cell(row=1, column=ci)
        c.font = Font(bold=True)
        c.fill = rate_fill if h in RATING_COLS else hdr_fill
        c.alignment = Alignment(vertical="top", wrap_text=True)
    widths = {"question": 60, "gold_answer": 60, "model_answer": 70, "notes": 40,
              "judgment_id": 10, "rating_1_5": 10, "would_give_farmer": 14}
    for ci, h in enumerate(header, start=1):
        ws.column_dimensions[get_column_letter(ci)].width = widths.get(h, 14)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = ws.cell(row=2, column=header.index("question") + 1)

    n = len(sheet_rows) + 1
    col = get_column_letter(header.index("rating_1_5") + 1)
    dv1 = DataValidation(type="whole", operator="between", formula1="1", formula2="5",
                         allow_blank=True, showErrorMessage=True,
                         error="Enter an integer 1-5")
    dv1.add(f"{col}2:{col}{n}")
    ws.add_data_validation(dv1)
    col = get_column_letter(header.index("would_give_farmer") + 1)
    dv2 = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    dv2.add(f"{col}2:{col}{n}")
    ws.add_data_validation(dv2)

    guide = wb.create_sheet("README")
    for line in [
        "STAGE 7 HUMAN RATING SHEET — 800 judgments (200 items x 4 model outputs)",
        "",
        "Each row = one question + one model's answer. The model is NOT identified.",
        "The same question appears 4 times in the sheet, in shuffled positions.",
        "",
        "Fill only the yellow columns:",
        "  rating_1_5        integer 1 (unusable) .. 5 (correct, complete, safe to give)",
        "  would_give_farmer yes / no  — would you hand this answer to a farmer as-is?",
        "  notes             optional, one line",
        "",
        "Do not edit any other column. Do not sort or delete rows; judgment_id is the join key.",
        "gold_answer is the reference answer for the question (hide the column if you",
        "want to rate blind to the reference).",
    ]:
        guide.append([line])
    guide.column_dimensions["A"].width = 100

    xlsx_path = OUT_DIR / "stage7_rating_sheet.xlsx"
    wb.save(xlsx_path)

    # ---- KEY (keep separate from annotators)
    key_path = OUT_DIR / "stage7_arm_key_DO_NOT_SHARE.csv"
    with key_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        w.writeheader()
        w.writerows(key_rows)

    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = {
        "seed": SEED,
        "items": 200,
        "arms": 4,
        "judgments": 800,
        "arm_labels": ARMS,
        "lang": dict(Counter(k["lang"] for k in key_rows)),
        "qtype": dict(Counter(k["qtype"] for k in key_rows)),
        "per_arm": dict(Counter(k["arm"] for k in key_rows)),
        "columns": header,
        "rating_sheet_csv": str(csv_path),
        "rating_sheet_xlsx": str(xlsx_path),
        "rating_sheet_csv_sha256": sha,
        "arm_key": str(key_path),
        "generation_sources": {
            "native_test_rows": "reports/s6/predictions/{arm}.jsonl",
            "authored_bn_rows": "reports/s7/predictions_translated_bn/{arm}.jsonl",
        },
    }
    (OUT_DIR / "stage7_sheet_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"judgments : {len(sheet_rows)}")
    print(f"per arm   : {manifest['per_arm']}")
    print(f"lang      : {manifest['lang']}")
    print(f"qtype     : {manifest['qtype']}")
    print(f"XLSX      : {xlsx_path}")
    print(f"CSV       : {csv_path}")
    print(f"KEY       : {key_path}   <- do not give to annotators")


if __name__ == "__main__":
    main()
