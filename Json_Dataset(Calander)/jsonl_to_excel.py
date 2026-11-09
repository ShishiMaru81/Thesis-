#!/usr/bin/env python3
"""
jsonl_to_excel.py — Convert qa_out/all.jsonl into qa_dataset.xlsx.

Columns: id, crop, lang, question_type, gold_kind, question, answer, source_file.
source_file is grounding["file"] where present (factual/interval answers, which
trace to a source screenshot); numeric_runtime/abstain records have no source
file (they're computed or out-of-scope) so the cell is left blank rather than
inventing one.

Run:  python jsonl_to_excel.py --in ./qa_out/all.jsonl --out ./qa_dataset.xlsx
"""
import json, argparse
import pandas as pd

def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "id": r["id"],
                "crop": r["crop"],
                "lang": r["lang"],
                "question_type": r["question_type"],
                "gold_kind": r["gold_kind"],
                "question": r["messages"][0]["content"],
                "answer": r["messages"][1]["content"],
                "source_file": r.get("grounding", {}).get("file", ""),
            })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="./qa_out/all.jsonl")
    ap.add_argument("--out", dest="outfile", default="./qa_dataset.xlsx")
    a = ap.parse_args()

    rows = load_rows(a.infile)
    df = pd.DataFrame(rows, columns=[
        "id", "crop", "lang", "question_type", "gold_kind",
        "question", "answer", "source_file",
    ])

    with pd.ExcelWriter(a.outfile, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="qa_dataset", index=False)
        ws = writer.sheets["qa_dataset"]
        widths = {"A": 30, "B": 16, "C": 6, "D": 18, "E": 12,
                   "F": 60, "G": 60, "H": 30}
        for col, w in widths.items():
            ws.column_dimensions[col].width = w

    print(f"Wrote {len(df)} rows to {a.outfile}")

if __name__ == "__main__":
    main()
