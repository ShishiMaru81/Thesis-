#!/usr/bin/env python3
"""
jsonl_to_csv.py — Convert a qa_out/*.jsonl dataset into a CSV file.

Columns: id, crop, lang, question_type, gold_kind, question, answer, source_file.
source_file is grounding["file"] where present (factual/interval answers, which
trace to a source screenshot); numeric_runtime/abstain records have no source
file (they're computed or out-of-scope) so the cell is left blank rather than
inventing one. Written with a UTF-8 BOM (utf-8-sig) so Bengali text displays
correctly when the CSV is opened directly in Excel.

Run:  python jsonl_to_csv.py --in ./qa_out/all.jsonl --out ./Final_QA_pair.csv
"""
import json, csv, argparse

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
    ap.add_argument("--out", dest="outfile", default="./Final_QA_pair.csv")
    a = ap.parse_args()

    rows = load_rows(a.infile)
    columns = ["id", "crop", "lang", "question_type", "gold_kind", "question", "answer", "source_file"]

    with open(a.outfile, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {a.outfile}")

if __name__ == "__main__":
    main()
