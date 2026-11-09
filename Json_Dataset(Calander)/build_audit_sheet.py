#!/usr/bin/env python3
"""
build_audit_sheet.py — Convert audit_sample.jsonl into audit_sheet.xlsx for
human verification before fine-tuning.

Columns: id, crop, lang, question_type, question, answer, source_file,
grounding_field, Reviewer1_verdict, Reviewer1_notes, Reviewer2_verdict,
Reviewer2_notes.

source_file is the crop's JSON filename (e.g. Aman_Rice.json), resolved by
scanning --jsondir and matching each file's internal crop_name_en field —
every row for a crop (factual, numeric_runtime, or abstain) comes from that
one file, so it applies to all question_types, not just factual.

grounding_field is grounding["field"] (only present for factual/interval
answers, which trace to one named field; numeric_runtime and abstain rows
are computed/out-of-scope and have no single field, so it's left blank).

Reviewer verdict columns get an Excel dropdown (Correct/Incorrect/Unsure)
via data validation. A second "Instructions" sheet is added.

Run:  python build_audit_sheet.py --in ./audit_sample.jsonl --jsondir . --out ./audit_sheet.xlsx
"""
import json, glob, os, argparse
from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

VERDICT_OPTIONS = ["Correct", "Incorrect", "Unsure"]

COLUMNS = [
    ("id", 30), ("crop", 14), ("lang", 6), ("question_type", 16),
    ("question", 55), ("answer", 55), ("source_file", 18),
    ("grounding_field", 22), ("Reviewer1_verdict", 16), ("Reviewer1_notes", 30),
    ("Reviewer2_verdict", 16), ("Reviewer2_notes", 30),
]

INSTRUCTIONS_TEXT = (
    "For each row, open the source_file JSON (and its matching crop image) and "
    "check whether the answer is actually correct given the calendar. Mark "
    "Correct/Incorrect/Unsure independently — do not discuss with the other "
    "reviewer until both are done."
)

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_crop_to_file(jsondir):
    mapping = {}
    for path in sorted(glob.glob(os.path.join(jsondir, "*.json"))):
        doc = json.load(open(path, encoding="utf-8"))
        crop = doc.get("crop_name_en")
        if crop:
            mapping[crop] = os.path.basename(path)
    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="./audit_sample.jsonl")
    ap.add_argument("--jsondir", dest="jsondir", default=".")
    ap.add_argument("--out", dest="outfile", default="./audit_sheet.xlsx")
    a = ap.parse_args()

    records = load_jsonl(a.infile)
    crop_to_file = build_crop_to_file(a.jsondir)

    missing_map = sorted({r["crop"] for r in records if r["crop"] not in crop_to_file})
    if missing_map:
        print(f"WARNING: no JSON file found for crop(s), source_file left blank: {missing_map}")

    wb = Workbook()
    ws = wb.active
    ws.title = "audit_sheet"

    header_font = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="DDEBF7")
    wrap = Alignment(wrap_text=True, vertical="top")

    for col_idx, (name, width) in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=name)
        cell.font = header_font
        cell.fill = header_fill
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    for row_idx, r in enumerate(records, start=2):
        grounding = r.get("grounding", {}) or {}
        values = {
            "id": r["id"],
            "crop": r["crop"],
            "lang": r["lang"],
            "question_type": r["question_type"],
            "question": r["messages"][0]["content"],
            "answer": r["messages"][1]["content"],
            "source_file": crop_to_file.get(r["crop"], ""),
            "grounding_field": grounding.get("field", ""),
            "Reviewer1_verdict": "",
            "Reviewer1_notes": "",
            "Reviewer2_verdict": "",
            "Reviewer2_notes": "",
        }
        for col_idx, (name, _) in enumerate(COLUMNS, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=values[name])
            if name in ("question", "answer"):
                cell.alignment = wrap

    n_rows = len(records)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{n_rows + 1}"

    dv = DataValidation(type="list", formula1=f'"{",".join(VERDICT_OPTIONS)}"',
                         allow_blank=True, showDropDown=False)
    dv.error = "Choose Correct, Incorrect, or Unsure."
    dv.errorTitle = "Invalid verdict"
    ws.add_data_validation(dv)
    r1_col = [name for name, _ in COLUMNS].index("Reviewer1_verdict") + 1
    r2_col = [name for name, _ in COLUMNS].index("Reviewer2_verdict") + 1
    dv.add(f"{get_column_letter(r1_col)}2:{get_column_letter(r1_col)}{n_rows + 1}")
    dv.add(f"{get_column_letter(r2_col)}2:{get_column_letter(r2_col)}{n_rows + 1}")

    # --- Instructions sheet -------------------------------------------------
    ins = wb.create_sheet("Instructions")
    ins.column_dimensions["A"].width = 100
    title = ins.cell(row=1, column=1, value="Human Verification Audit — Instructions")
    title.font = Font(bold=True, size=14)
    body = ins.cell(row=3, column=1, value=INSTRUCTIONS_TEXT)
    body.alignment = Alignment(wrap_text=True, vertical="top")
    ins.row_dimensions[3].height = 60
    ins.cell(row=5, column=1, value="Verdict options: Correct / Incorrect / Unsure").font = Font(bold=True)
    ins.cell(row=6, column=1,
             value="source_file = the crop's JSON file in Json_Dataset(Calander)/. "
                   "grounding_field = the field key that answer was retrieved from (factual/interval "
                   "rows only; numeric_runtime rows are computed from the ETc formula, not one field).")

    wb.save(a.outfile)
    print(f"Wrote {n_rows} rows to {a.outfile}")

if __name__ == "__main__":
    main()
