#!/usr/bin/env python3
"""
compute_kappa.py — Inter-rater agreement on the completed audit_sheet.xlsx.

Computes Cohen's Kappa between Reviewer1_verdict and Reviewer2_verdict,
prints the score with a Landis & Koch (1977) interpretation, lists every
row where the two reviewers disagreed, and lists every row either reviewer
marked "Incorrect" (candidates to fix or drop before fine-tuning).

Only rows where BOTH reviewers have filled in a verdict are used for the
kappa computation; rows still missing a verdict are reported separately.

Run:  python compute_kappa.py --sheet ./audit_sheet.xlsx
"""
import argparse, collections
import pandas as pd

LANDIS_KOCH = [
    (0.00, "Poor / no agreement"),
    (0.20, "Slight agreement"),
    (0.40, "Fair agreement"),
    (0.60, "Moderate agreement"),
    (0.80, "Substantial agreement"),
    (1.01, "Almost perfect agreement"),
]

def interpret(kappa):
    for i in range(len(LANDIS_KOCH) - 1):
        lo, label = LANDIS_KOCH[i]
        hi, _ = LANDIS_KOCH[i + 1]
        if lo <= kappa < hi or (kappa < 0 and i == 0):
            return f"{lo:.2f}-{hi if hi <= 1 else 1.00:.2f} = {label}"
    return "1.00 = Almost perfect agreement"

def cohens_kappa(r1, r2):
    n = len(r1)
    po = sum(1 for a, b in zip(r1, r2) if a == b) / n
    c1 = collections.Counter(r1)
    c2 = collections.Counter(r2)
    categories = set(c1) | set(c2)
    pe = sum((c1.get(c, 0) / n) * (c2.get(c, 0) / n) for c in categories)
    if pe == 1.0:
        return 1.0  # no variance in either rater; treat perfect observed agreement as kappa=1
    return (po - pe) / (1 - pe)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="./audit_sheet.xlsx")
    a = ap.parse_args()

    df = pd.read_excel(a.sheet, sheet_name="audit_sheet")
    for col in ("Reviewer1_verdict", "Reviewer2_verdict"):
        df[col] = df[col].astype("string").str.strip()

    both_done = df[df["Reviewer1_verdict"].notna() & (df["Reviewer1_verdict"] != "") &
                    df["Reviewer2_verdict"].notna() & (df["Reviewer2_verdict"] != "")]
    incomplete = df[~df.index.isin(both_done.index)]

    print(f"{len(both_done)}/{len(df)} rows have both verdicts filled in.")
    if len(incomplete) > 0:
        print(f"{len(incomplete)} row(s) still missing a verdict from at least one reviewer "
              f"(excluded from kappa):")
        for _, row in incomplete.iterrows():
            print(f"  {row['id']}: R1={row['Reviewer1_verdict']!r} R2={row['Reviewer2_verdict']!r}")
        print()

    if len(both_done) < 2:
        print("Not enough completed rows to compute kappa.")
        return

    kappa = cohens_kappa(both_done["Reviewer1_verdict"].tolist(), both_done["Reviewer2_verdict"].tolist())
    print(f"\nCohen's Kappa = {kappa:.3f}")
    print(f"Interpretation (Landis & Koch 1977): {interpret(kappa)}")

    disagreements = both_done[both_done["Reviewer1_verdict"] != both_done["Reviewer2_verdict"]]
    print(f"\n--- Disagreements ({len(disagreements)} row(s)) ---")
    for _, row in disagreements.iterrows():
        print(f"  {row['id']} [{row['crop']}/{row['question_type']}/{row['lang']}]: "
              f"R1={row['Reviewer1_verdict']} (notes: {row['Reviewer1_notes']!r})  "
              f"R2={row['Reviewer2_verdict']} (notes: {row['Reviewer2_notes']!r})")

    incorrect = both_done[(both_done["Reviewer1_verdict"] == "Incorrect") |
                           (both_done["Reviewer2_verdict"] == "Incorrect")]
    print(f"\n--- Marked 'Incorrect' by at least one reviewer ({len(incorrect)} row(s)) ---")
    for _, row in incorrect.iterrows():
        print(f"  {row['id']} [{row['crop']}/{row['question_type']}/{row['lang']}]: "
              f"R1={row['Reviewer1_verdict']} R2={row['Reviewer2_verdict']}")
        print(f"    Q: {row['question']}")
        print(f"    A: {row['answer']}")
        if row["Reviewer1_notes"]:
            print(f"    R1 notes: {row['Reviewer1_notes']}")
        if row["Reviewer2_notes"]:
            print(f"    R2 notes: {row['Reviewer2_notes']}")

if __name__ == "__main__":
    main()
