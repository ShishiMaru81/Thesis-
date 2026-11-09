#!/usr/bin/env python3
"""
s6b_generate_clarify.py — Stage 6B: synthesize 300 clarify training examples.

Draws from numeric_runtime rows in train_augmented.jsonl, drops one required
slot per row, and pairs with a hand-authored clarification gold (no computed
numbers, no gold_trace).

Run:  python scripts/s6b_generate_clarify.py
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import irrigation_calc as ic  # noqa: E402

TRAIN_PATH = PROJECT_ROOT / "data" / "augmented" / "train_augmented.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "augmented" / "s6b_clarify_train.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports" / "s6b" / "generate_report.json"

SEED = 42
N_TOTAL = 300
PER_SLOT = 100
PER_LANG = 150

SLOT_TO_SUBTYPE = {
    "rainfall_mm": "missing_rainfall",
    "area_value": "missing_area",
    "stage": "missing_stage",
}

BN_DIGITS = "০১২৩৪৫৬৭৮৯"
ASCII_DIGITS = "0123456789"
_DIGIT_MAP = str.maketrans(dict(zip(BN_DIGITS, ASCII_DIGITS)))


def bn_num(n) -> str:
    s = str(n)
    return s.translate(_DIGIT_MAP) if any(c in BN_DIGITS for c in s) else s


AREA_UNIT_BN = {
    "acre": "একর",
    "bigha": "বিঘা",
    "katha": "কাঠা",
    "decimal": "শতক",
    "hectare": "হেক্টর",
    "sqm": "বর্গমিটার",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def clarify_question(sub: str, row: dict) -> str:
    """Rewrite a numeric question with one slot removed (mirrors s4 c2_question)."""
    scen = row["scenario"]
    crop = row.get("crop") or scen.get("crop", "")
    crop_bn = row.get("crop_bn") or crop
    stage = scen["stage"]
    lang = row["lang"]
    res = {
        "et0_mm": scen["et0_mm"],
        "rainfall_mm": scen["rainfall_mm"],
        "area_value": scen["area_value"],
        "area_unit": scen["area_unit"],
    }

    if lang == "en":
        if sub == "missing_rainfall":
            return (
                f"My {crop} field is at the {stage} stage. Today ET0 is "
                f"{res['et0_mm']} mm and the field is {res['area_value']} "
                f"{res['area_unit']}. How much irrigation water does it need today?"
            )
        if sub == "missing_area":
            return (
                f"My {crop} field is at the {stage} stage. Today ET0 is "
                f"{res['et0_mm']} mm and it rained {res['rainfall_mm']} mm. "
                f"How many liters of irrigation water does it need today?"
            )
        return (
            f"My {crop} field needs water. Today ET0 is {res['et0_mm']} mm, "
            f"it rained {res['rainfall_mm']} mm, and the field is "
            f"{res['area_value']} {res['area_unit']}. How much irrigation "
            f"water does it need today?"
        )

    sbn = ic.STAGE_BN[stage]
    ubn = AREA_UNIT_BN.get(res["area_unit"], res["area_unit"])
    if sub == "missing_rainfall":
        return (
            f"আমার {crop_bn} ক্ষেত {sbn} পর্যায়ে আছে। আজ ET0 "
            f"{bn_num(res['et0_mm'])} মিমি, জমির পরিমাণ "
            f"{bn_num(res['area_value'])} {ubn}। আজ কত পানি সেচ দিতে হবে?"
        )
    if sub == "missing_area":
        return (
            f"আমার {crop_bn} ক্ষেত {sbn} পর্যায়ে আছে। আজ ET0 "
            f"{bn_num(res['et0_mm'])} মিমি, বৃষ্টি হয়েছে "
            f"{bn_num(res['rainfall_mm'])} মিমি। আজ কত লিটার পানি সেচ দিতে হবে?"
        )
    return (
        f"আমার {crop_bn} ক্ষেতে পানি দিতে হবে। আজ ET0 "
        f"{bn_num(res['et0_mm'])} মিমি, বৃষ্টি হয়েছে "
        f"{bn_num(res['rainfall_mm'])} মিমি, জমির পরিমাণ "
        f"{bn_num(res['area_value'])} {ubn}। আজ কত পানি সেচ দিতে হবে?"
    )


def clarify_gold(sub: str, row: dict) -> str:
    crop = row.get("crop") or row["scenario"].get("crop", "crop")
    crop_bn = row.get("crop_bn") or crop
    lang = row["lang"]

    if lang == "en":
        if sub == "missing_rainfall":
            return (
                "I need to know this week's rainfall to calculate net irrigation. "
                "Could you provide it in millimeters?"
            )
        if sub == "missing_area":
            return (
                "I need your field size to calculate total water volume. "
                "Please tell me: acres, decimals, bigha, hectares, or square meters?"
            )
        return (
            f"Which growth stage is your {crop} in? "
            "(initial / mid-season / late-season) — this affects water needs."
        )

    if sub == "missing_rainfall":
        return (
            "এই সপ্তাহে কত বৃষ্টি হয়েছে? মিলিমিটারে জানান। তারপর আমি হিসাব করব।"
        )
    if sub == "missing_area":
        return (
            "মোট পানির পরিমাণ হিসাব করতে জমির মাপ জানতে হবে। "
            "অনুগ্রহ করে জানান: একর, শতক, বিঘা, হেক্টর, নাকি বর্গমিটার?"
        )
    return (
        f"আপনার {crop_bn} কোন বৃদ্ধির পর্যায়ে আছে? "
        "(প্রাথমিক / মধ্য-মৌসুম / শেষ-মৌসুম) — এতে পানির চাহিদা বদলায়।"
    )


def sample_rows(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    if len(pool) < n:
        raise SystemExit(f"pool too small: need {n}, have {len(pool)}")
    return rng.sample(pool, n)


def main() -> int:
    if OUT_PATH.exists():
        raise SystemExit(
            f"REFUSE: {OUT_PATH} already exists. Delete it to regenerate fresh."
        )
    if not TRAIN_PATH.exists():
        raise SystemExit(f"missing {TRAIN_PATH}")

    rows = load_jsonl(TRAIN_PATH)
    if len(rows) != 3552:
        print(f"WARNING: expected 3552 train rows, got {len(rows)}")

    numeric = [r for r in rows if r.get("qtype") == "numeric_runtime"]
    by_lang = {
        "bn": [r for r in numeric if r.get("lang") == "bn"],
        "en": [r for r in numeric if r.get("lang") == "en"],
    }
    print(f"numeric_runtime: {len(numeric)}  bn={len(by_lang['bn'])}  en={len(by_lang['en'])}")

    rng = random.Random(SEED)
    slots = ["rainfall_mm", "area_value", "stage"]
    out: list[dict] = []
    idx = 1

    for slot in slots:
        sub = SLOT_TO_SUBTYPE[slot]
        for lang in ("bn", "en"):
            picked = sample_rows(by_lang[lang], 50, rng)
            for parent in picked:
                rec = {
                    "row_id": f"clarify_{idx:03d}",
                    "qtype": "clarify",
                    "lang": lang,
                    "question": clarify_question(sub, parent),
                    "answer": clarify_gold(sub, parent),
                    "gold_status": "hand_authored",
                    "missing_slot": slot,
                    "subtype": sub,
                    "parent_row_id": parent["row_id"],
                    "crop": parent.get("crop"),
                    "crop_bn": parent.get("crop_bn"),
                    "stage": parent["scenario"].get("stage"),
                    "scenario": dict(parent["scenario"]),
                    "origin": "s6b_clarify",
                    "answer_kind": "clarification",
                    "provenance": {
                        "extraction_method": "s6b_synthetic_clarify",
                        "generator": "s6b_generate_clarify.py",
                        "seed": SEED,
                        "withheld_parameter": slot,
                    },
                }
                out.append(rec)
                idx += 1

    assert len(out) == N_TOTAL

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    langs = collections.Counter(r["lang"] for r in out)
    slots_c = collections.Counter(r["missing_slot"] for r in out)
    row_ids = [r["row_id"] for r in out]
    blanks = sum(1 for r in out if not r.get("answer", "").strip())

    report = {
        "rows_generated": len(out),
        "language_split": dict(langs),
        "slot_distribution": dict(slots_c),
        "all_row_ids_unique": len(row_ids) == len(set(row_ids)),
        "all_golds_non_empty": blanks == 0,
        "output": str(OUT_PATH),
        "seed": SEED,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("s6b_generate_clarify — REPORT")
    print("=" * 60)
    print(f"Rows generated     : {len(out)}")
    print(f"Language split     : {langs['bn']} bn / {langs['en']} en")
    print(f"Slot distribution  : rainfall={slots_c['rainfall_mm']}  "
          f"area={slots_c['area_value']}  stage={slots_c['stage']}")
    print(f"All row_ids unique : {'YES' if report['all_row_ids_unique'] else 'NO'}")
    print(f"All golds non-empty: {'YES' if report['all_golds_non_empty'] else 'NO'}")
    print(f"-> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
