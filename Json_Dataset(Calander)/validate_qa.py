#!/usr/bin/env python3
"""
validate_qa.py — Independent check on qa_out/all.jsonl.

For every record with gold_kind == "computed", recompute net irrigation
depth (mm) and liters from record["grounding"]["inputs"] using the same
formula as generate_qa.py, then compare against the numbers actually
stated in the answer text (not against grounding["computed"], which was
written by the same code path being checked).

Run:  python validate_qa.py --qa ./qa_out/all.jsonl
"""
import json, re, argparse

# ----------------------------------------------------------------- helpers
BD = "০১২৩৪৫৬৭৮৯"
BD_TO_ASCII = {b: str(i) for i, b in enumerate(BD)}

def bn_to_ascii(s):
    return "".join(BD_TO_ASCII.get(c, c) for c in s)

# ----------------------------------------------------------------- ETc engine (same formula as generate_qa.py)
def effective_rain(mm):
    if mm <= 5:  return round(mm * 0.85, 2)
    if mm <= 15: return round(mm * 0.45, 2)
    return round(mm * 0.25, 2)

def net_irrigation_mm(kc, et0, rain_mm):
    net = kc * et0 - effective_rain(rain_mm)
    return round(max(net, 0.0), 2)

AREA = {"acre": 4046.86, "bigha": 1337.8, "katha": 66.89, "hectare": 10000.0, "sqm": 1.0}

# ----------------------------------------------------------------- extract stated numbers from answer text
MM_RE_EN = re.compile(r"=\s*([\d.]+)\s*mm\.")
LITERS_RE_EN = re.compile(r"about\s*([\d.]+)\s*liters")
MM_RE_BN = re.compile(r"=\s*([০-৯.]+)\s*মিমি")
LITERS_RE_BN = re.compile(r"প্রায়\s*([\d.]+)\s*লিটার")

def extract_stated(record):
    text = record["messages"][1]["content"]
    lang = record["lang"]
    if lang == "en":
        mm_m, li_m = MM_RE_EN.search(text), LITERS_RE_EN.search(text)
    else:
        mm_m, li_m = MM_RE_BN.search(text), LITERS_RE_BN.search(text)
    if not mm_m or not li_m:
        return None, None
    mm_str = bn_to_ascii(mm_m.group(1)) if lang == "bn" else mm_m.group(1)
    return mm_str, li_m.group(1)  # liters is always western digits, even in bn text

# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="./qa_out/all.jsonl")
    a = ap.parse_args()

    total = ok = 0
    mismatches = []

    with open(a.qa, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("gold_kind") != "computed":
                continue
            total += 1

            inp = rec["grounding"]["inputs"]
            net_mm = net_irrigation_mm(inp["kc"], inp["et0_mm"], inp["rain_mm"])
            area_sqm = inp["area_value"] * AREA[inp["area_unit"]]
            liters = round(net_mm / 1000 * area_sqm * 1000, 1)

            expected_mm_str = str(net_mm)
            expected_liters_str = f"{liters:.0f}"

            stated_mm_str, stated_liters_str = extract_stated(rec)
            if stated_mm_str is None:
                mismatches.append({"id": rec["id"], "lang": rec["lang"],
                                    "reason": "could not parse answer text"})
                continue

            if stated_mm_str == expected_mm_str and stated_liters_str == expected_liters_str:
                ok += 1
            else:
                mismatches.append({
                    "id": rec["id"], "lang": rec["lang"],
                    "recomputed_net_mm": expected_mm_str, "stated_net_mm": stated_mm_str,
                    "recomputed_liters": expected_liters_str, "stated_liters": stated_liters_str,
                })

    if mismatches:
        print(f"{len(mismatches)} MISMATCH(ES):")
        for m in mismatches:
            print(json.dumps(m, ensure_ascii=False))
        print()

    print(f"{ok}/{total} computed answers verified correct")

if __name__ == "__main__":
    main()
