#!/usr/bin/env python3
"""
regenerate_numeric_qa.py — build the v2 corpus.

WHAT THIS DOES
  KEEP      every factual + abstain record from qa_out/all.jsonl (2,712 rows).
            Their question/answer text is copied byte-for-byte; only group_id is
            rewritten (see below).
  DISCARD   all 1,008 v1 numeric_runtime records. Every one has a prose target
            and comes from a 3-scenario input space.
  GENERATE  1,800 fresh numeric_runtime records whose assistant turn is ONLY an
            <irrigation_calc> tool call, over ~900 independently sampled
            parameter tuples.

THE TWO BUGS THIS FIXES
  1. ET0 and rainfall were perfectly correlated across only 3 scenarios, so the
     model learned a lookup table. Here every variable is sampled independently.
  2. group_id carried a __bn/__en suffix, so the Bengali and English versions of
     one fact landed on opposite sides of the split. Here numeric group_ids bind
     the BN/EN pair of one tuple together, and the kept rows' group_ids have
     their language suffix stripped for the same reason.

All arithmetic comes from scripts/irrigation_calc.py — the same module the
runtime bridge imports. Nothing is reimplemented here.

Run:  python scripts/regenerate_numeric_qa.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import irrigation_calc as ic  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ALL = PROJECT_ROOT / "Json_Dataset(Calander)" / "qa_out" / "all.jsonl"
CROP_JSON_DIR = PROJECT_ROOT / "Json_Dataset(Calander)"
OUT_DIR = PROJECT_ROOT / "data" / "v2"
OUT_ALL = OUT_DIR / "all_v2.jsonl"

N_TUPLES = 900          # -> 1,800 records (one BN + one EN per tuple)
MIN_DISTINCT_TUPLES = 400
SEED = 42

RNG = random.Random(SEED)
random.seed(SEED)

BD_DIGITS = "০১২৩৪৫৬৭৮৯"


def bn_num(n) -> str:
    """Verbatim from generate_qa.py:61-62."""
    return "".join(BD_DIGITS[int(c)] if c.isdigit() else c for c in str(n))


def qhash(instr: str) -> str:
    return hashlib.sha1(instr.encode("utf-8")).hexdigest()[:12]


def fmt_num(v: float) -> str:
    """1.0 -> '1', 0.5 -> '0.5', 4.5 -> '4.5'."""
    return str(int(v)) if float(v) == int(v) else str(v)


# ----------------------------------------------------------------- area space
# Exactly the seven (value, unit) pairs specified. Units are the canonical keys
# in irrigation_calc.AREA; questions render varied surface forms of them and the
# model must normalize back to the canonical key.
AREA_CHOICES = [
    (1.0, "katha"),
    (5.0, "decimal"),
    (10.0, "decimal"),
    (0.5, "acre"),
    (1.0, "acre"),
    (0.5, "ha"),
    (1.0, "ha"),
]

UNIT_SURFACE = {
    "katha":   {"en_sing": ["katha"],           "en_plur": ["katha", "kathas"],
                "bn": ["কাঠা"]},
    "decimal": {"en_sing": ["decimal", "shatak"], "en_plur": ["decimal", "decimals", "shatak"],
                "bn": ["শতক", "ডেসিমেল"]},
    "acre":    {"en_sing": ["acre"],            "en_plur": ["acre", "acres"],
                "bn": ["একর"]},
    "ha":      {"en_sing": ["hectare", "ha"],   "en_plur": ["hectare", "hectares", "ha"],
                "bn": ["হেক্টর"]},
}


def unit_surface(unit: str, value: float, script: str) -> str:
    forms = UNIT_SURFACE[unit]
    if script == "bn":
        return RNG.choice(forms["bn"])
    key = "en_sing" if float(value) == 1.0 else "en_plur"
    return RNG.choice(forms[key])


# ----------------------------------------------------------------- templates
# 12 surface forms per language. Every template states ALL six parameters, so no
# question ever requires the model to invent a value it was not given.
TEMPLATES_EN = [
    "My {crop} field is at the {stage} stage. Today ET0 is {et0} mm, it rained {rain} mm, and the field is {aval} {aunit}. How much irrigation water does it need today?",
    "Tell me how much water my {crop} field needs today — it's at the {stage} stage, ET0 is {et0} mm, rainfall was {rain} mm, and the area is {aval} {aunit}.",
    "I want to know the irrigation amount for a {aval} {aunit} {crop} field at the {stage} stage, given ET0 of {et0} mm and {rain} mm of rain today.",
    "At the {stage} stage with ET0 {et0} mm and {rain} mm rain, how much should I water my {aval} {aunit} {crop} field today?",
    "My {crop} ({aval} {aunit}) is in its {stage} stage. It got {rain} mm rain and ET0 is {et0} mm — how much irrigation today?",
    "How much irrigation water does a {stage}-stage {crop} field of {aval} {aunit} need, with ET0 {et0} mm and {rain} mm of rainfall?",
    "Could you work out today's irrigation for {aval} {aunit} of {crop}? Stage is {stage}, ET0 {et0} mm, rain {rain} mm.",
    "Need the water figure for {crop} — {stage} stage, ET0 {et0} mm, rain {rain} mm, area {aval} {aunit}.",
    "Bhai, my {aval} {aunit} {crop} plot is in {stage} stage. ET0 {et0} mm and {rain} mm brishti today — koto water lagbe?",
    "Please calculate the irrigation requirement: crop {crop}, growth stage {stage}, reference evapotranspiration {et0} mm, rainfall {rain} mm, plot size {aval} {aunit}.",
    "What's today's watering for my {crop}? {stage} stage, {aval} {aunit}, ET0 came in at {et0} mm and we got {rain} mm of rain.",
    "Given ET0 = {et0} mm, rainfall = {rain} mm, area = {aval} {aunit}, and a {crop} crop in the {stage} stage, compute the irrigation need.",
]

TEMPLATES_BN = [
    "আমার {crop} ক্ষেত {stage} পর্যায়ে আছে। আজ ET0 {et0} মিমি, বৃষ্টি হয়েছে {rain} মিমি, জমির পরিমাণ {aval} {aunit}। আজ কত পানি সেচ দিতে হবে?",
    "আমার {crop} খেতে আজ কত পানি লাগবে বলুন — এটা {stage} পর্যায়ে আছে, ET0 {et0} মিমি, বৃষ্টি {rain} মিমি, জমি {aval} {aunit}।",
    "আমি জানতে চাই {aval} {aunit} {crop} খেতে {stage} পর্যায়ে, ET0 {et0} মিমি এবং আজ {rain} মিমি বৃষ্টিতে কত সেচ লাগবে।",
    "{stage} পর্যায়ে ET0 {et0} মিমি আর বৃষ্টি {rain} মিমি হলে আমার {aval} {aunit} {crop} খেতে আজ কতটুকু পানি দিমু?",
    "আমার {crop} ({aval} {aunit}) এখন {stage} পর্যায়ে। বৃষ্টি হয়েছে {rain} মিমি, ET0 {et0} মিমি — আজ কত সেচ লাগবে?",
    "{stage} পর্যায়ের {aval} {aunit} {crop} খেতে ET0 {et0} মিমি ও বৃষ্টি {rain} মিমি হলে কতটুকু সেচ লাগবে?",
    "ভাই, আমার {aval} {aunit} {crop} জমি — {stage} পর্যায়, ET0 {et0} মিমি, আজ {rain} মিমি বৃষ্টি। কত পানি দেওয়া লাগবে?",
    "দয়া করে হিসাব করে দিন: {crop}, {stage} পর্যায়, ET0 {et0} মিমি, বৃষ্টিপাত {rain} মিমি, জমি {aval} {aunit}।",
    "{crop} চাষ করি, {aval} {aunit} জমি। এখন {stage} পর্যায় চলছে, ET0 {et0} মিমি আর {rain} মিমি বৃষ্টি হইছে। সেচ কতটুকু?",
    "আজকে আমার {crop} খেতের সেচের হিসাব দরকার। পর্যায় {stage}, ET0 {et0} মিমি, বৃষ্টি {rain} মিমি, আয়তন {aval} {aunit}।",
    "{aval} {aunit} জমিতে {crop} আছে, {stage} স্টেজে। ET0 {et0} মিমি, রেইনফল {rain} মিমি। আজ কত লিটার পানি দিতে হবে?",
    "সেচের পরিমাণ বের করুন — ফসল {crop}, বৃদ্ধির পর্যায় {stage}, ET0 {et0} মিমি, বৃষ্টি {rain} মিমি, জমির আয়তন {aval} {aunit}।",
]


# ----------------------------------------------------------------- crop names
def load_crop_names() -> dict:
    """crop_name_en -> crop_name_bn, read from the BARC crop JSONs."""
    names = {}
    for p in sorted(CROP_JSON_DIR.glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        en = doc.get("crop_name_en")
        if en:
            names[en] = doc.get("crop_name_bn", en)
    return names


# ----------------------------------------------------------------- sampling
def sample_rainfall(tier: int) -> float:
    """Draw rainfall inside `tier`, rounding to 1 dp WITHOUT letting the rounded
    value fall into a neighbouring tier (e.g. a tier-2 draw of 5.02 would round
    to 5.0, which irrigation_calc scores as tier 1)."""
    lo, hi = {1: (0.0, 5.0), 2: (5.0, 15.0), 3: (15.0, 60.0)}[tier]
    for _ in range(200):
        v = round(RNG.uniform(lo, hi), 1)
        if ic.rain_tier(v) == tier and lo <= v <= hi:
            return v
    raise RuntimeError(f"could not draw rainfall in tier {tier}")


def build_tuples(crops: list[str]) -> list[dict]:
    """900 unique parameter tuples, balanced across every (crop, stage) cell and
    across the three rainfall tiers, with ET0 drawn independently of rainfall."""
    cells = [(c, s) for c in crops for s in ic.STAGES]          # 28 x 3 = 84
    reps = (N_TUPLES + len(cells) - 1) // len(cells)
    cell_slots = (cells * reps)[:N_TUPLES]
    RNG.shuffle(cell_slots)

    n1 = int(N_TUPLES * 0.33)
    n2 = int(N_TUPLES * 0.33)
    tier_slots = [1] * n1 + [2] * n2 + [3] * (N_TUPLES - n1 - n2)
    RNG.shuffle(tier_slots)

    area_reps = (N_TUPLES + len(AREA_CHOICES) - 1) // len(AREA_CHOICES)
    area_slots = (AREA_CHOICES * area_reps)[:N_TUPLES]
    RNG.shuffle(area_slots)

    out, seen = [], set()
    for (crop, stage), tier, (aval, aunit) in zip(cell_slots, tier_slots, area_slots):
        for _ in range(500):
            et0 = round(RNG.uniform(2.5, 6.5), 1)
            rain = sample_rainfall(tier)
            key = (crop, stage, et0, rain, aval, aunit)
            if key not in seen:
                seen.add(key)
                out.append({"crop": crop, "stage": stage, "et0_mm": et0,
                            "rainfall_mm": rain, "area_value": aval,
                            "area_unit": aunit, "tier": tier})
                break
        else:
            raise RuntimeError(f"could not find a unique tuple for {crop}/{stage}")
    return out


# ----------------------------------------------------------------- records
def gold_tool_call(params: dict) -> str:
    """The assistant turn, in full. Nothing precedes or follows it."""
    return f'<irrigation_calc>{json.dumps(params, ensure_ascii=False, sort_keys=True)}</irrigation_calc>'


def render_question(t: dict, lang: str, crop_bn: str,
                    numeral_script: str, unit_script: str) -> str:
    aunit_surface = unit_surface(t["area_unit"], t["area_value"],
                                 "bn" if unit_script == "bn" else "en")
    et0_s, rain_s, aval_s = fmt_num(t["et0_mm"]), fmt_num(t["rainfall_mm"]), fmt_num(t["area_value"])
    if numeral_script == "bn":
        et0_s, rain_s, aval_s = bn_num(et0_s), bn_num(rain_s), bn_num(aval_s)

    if lang == "en":
        return RNG.choice(TEMPLATES_EN).format(
            crop=t["crop"], stage=t["stage"], et0=et0_s, rain=rain_s,
            aval=aval_s, aunit=aunit_surface)
    return RNG.choice(TEMPLATES_BN).format(
        crop=crop_bn, stage=ic.STAGE_BN[t["stage"]], et0=et0_s, rain=rain_s,
        aval=aval_s, aunit=aunit_surface)


def make_numeric_records(tuples: list[dict], crop_names: dict) -> tuple[list, list]:
    records, skipped = [], []
    for t in tuples:
        params = {
            "area_unit": t["area_unit"],
            "area_value": t["area_value"],
            "crop": t["crop"],
            "et0_mm": t["et0_mm"],
            "rainfall_mm": t["rainfall_mm"],
            "stage": t["stage"],
        }
        try:
            res = ic.compute(**params)
        except ic.IrrigationError as e:
            # Never substitute a default. Drop the tuple and say why.
            skipped.append({"params": params, "reason": e.reason, "detail": str(e)})
            continue

        gold = gold_tool_call(params)
        gid = (t["crop"].replace(" ", "_") + "__num_" +
               hashlib.sha1(json.dumps(params, sort_keys=True,
                                       ensure_ascii=False).encode("utf-8")).hexdigest()[:8])

        grounding = {
            # --- keys required by the v2 spec ---
            "expected_net_mm": res["net_mm"],
            "expected_liters": res["liters"],
            "kc_used": res["kc_used"],
            "kc_source": res["kc_source_family"],
            "effective_rain_mm": res["effective_rain_mm"],
            "tier": res["tier"],
            "clamped": res["clamped"],
            # --- provenance / analysis extras ---
            "kc_source_verbose": res["kc_source"],
            "kc_stage_key": res["kc_stage_key"],
            "citation": res["citation"],
            "tier_factor": res["tier_factor"],
            "area_sqm": res["area_sqm"],
            "floored_at_zero": res["floored_at_zero"],
            "clamp_source": res["clamp_source"],
            "formula": res["formula"],
            "params": params,
            "calculator": "scripts/irrigation_calc.py::compute",
        }

        for lang in ("en", "bn"):
            numeral_script = "bn" if (lang == "bn" and RNG.random() < 0.5) else "en"
            unit_script = "bn" if (lang == "bn" and RNG.random() < 0.5) else "en"
            q = render_question(t, lang, crop_names.get(t["crop"], t["crop"]),
                                numeral_script, unit_script)
            g = dict(grounding, numeral_script=numeral_script, unit_script=unit_script)
            records.append({
                "id": f'{t["crop"]}_numeric_runtime_{lang}_{qhash(q)}',
                "crop": t["crop"],
                "lang": lang,
                "question_type": "numeric_runtime",
                "gold_kind": "computed",
                "group_id": gid,          # shared by the BN and EN twin
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": gold},
                ],
                "grounding": grounding | {"numeral_script": numeral_script,
                                          "unit_script": unit_script},
            })
    return records, skipped


def strip_lang_suffix(gid: str) -> str:
    """crop__topic__bn -> crop__topic. This is the cross-lingual leak fix for the
    kept rows: the BN and EN renderings of one BARC fact must share a group so
    they cannot land on opposite sides of the split."""
    for suf in ("__bn", "__en"):
        if gid.endswith(suf):
            return gid[: -len(suf)]
    return gid


# ----------------------------------------------------------------- main
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src = [json.loads(l) for l in SRC_ALL.open(encoding="utf-8") if l.strip()]
    kept, dropped = [], 0
    for r in src:
        if r["question_type"] == "numeric_runtime":
            dropped += 1
            continue
        r = dict(r)
        orig = r["group_id"]
        r["group_id"] = strip_lang_suffix(orig)
        if isinstance(r.get("grounding"), dict):
            r["grounding"] = dict(r["grounding"], orig_group_id=orig)
        kept.append(r)

    crop_names = load_crop_names()
    crops = sorted(c for c in crop_names
                   if ic.normalize_crop_name(c) in ic.KC_TABLE)
    no_kc = sorted(set(crop_names) - set(crops))
    if no_kc:
        print(f"SKIPPED (no Kc row, no numeric records generated): {no_kc}")

    tuples = build_tuples(crops)
    numeric, skipped = make_numeric_records(tuples, crop_names)
    if skipped:
        print(f"SKIPPED {len(skipped)} tuple(s) the calculator rejected:")
        for s in skipped[:20]:
            print("   ", s["reason"], s["detail"])

    corpus = kept + numeric

    # --- dedupe on question text (must not silently collide with kept rows) ---
    seen, deduped, dupes = set(), [], 0
    for r in corpus:
        q = r["messages"][0]["content"]
        if q in seen:
            dupes += 1
            continue
        seen.add(q)
        deduped.append(r)

    num_rows = [r for r in deduped if r["question_type"] == "numeric_runtime"]

    # ------------------------------------------------- hard gates
    distinct = {(g["params"]["crop"], g["params"]["stage"], g["params"]["et0_mm"],
                 g["params"]["rainfall_mm"], g["params"]["area_value"],
                 g["params"]["area_unit"])
                for g in (r["grounding"] for r in num_rows)}
    assert len(distinct) >= MIN_DISTINCT_TUPLES, (
        f"FAIL: only {len(distinct)} distinct numeric tuples, need >= {MIN_DISTINCT_TUPLES}")

    et0s = [r["grounding"]["params"]["et0_mm"] for r in num_rows]
    rains = [r["grounding"]["params"]["rainfall_mm"] for r in num_rows]
    pearson = statistics.correlation(et0s, rains)
    assert abs(pearson) < 0.10, f"FAIL: ET0/rainfall correlation {pearson:.4f} too high"

    tagged = [r for r in deduped if "<irrigation_calc>" in json.dumps(r, ensure_ascii=False)]
    assert all(r["question_type"] == "numeric_runtime" for r in tagged), \
        "FAIL: <irrigation_calc> leaked into a non-numeric record"
    assert len(tagged) == len(num_rows), \
        f"FAIL: {len(tagged)} tagged rows vs {len(num_rows)} numeric rows"

    ids = [r["id"] for r in deduped]
    assert len(ids) == len(set(ids)), "FAIL: duplicate id"

    with OUT_ALL.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------------------------- report
    print("\n" + "=" * 72)
    print("GATE 2 REPORT")
    print("=" * 72)
    print(f"source all.jsonl rows        : {len(src)}")
    print(f"  v1 numeric_runtime dropped : {dropped}")
    print(f"  kept (factual + abstain)   : {len(kept)}")
    print(f"  new numeric_runtime        : {len(numeric)}")
    print(f"  duplicate questions dropped: {dupes}")
    print(f"TOTAL v2 records             : {len(deduped)}")

    print("\n-- count per question_type --")
    for k, v in sorted(collections.Counter(r["question_type"] for r in deduped).items()):
        print(f"  {k:<16} {v}")
    print("\n-- count per lang --")
    for k, v in sorted(collections.Counter(r["lang"] for r in deduped).items()):
        print(f"  {k:<16} {v}")

    print(f"\n-- numeric coverage --")
    print(f"  distinct input tuples      : {len(distinct)}   (requirement >= {MIN_DISTINCT_TUPLES})")
    print(f"  distinct ET0 values        : {len(set(et0s))}")
    print(f"  distinct rainfall values   : {len(set(rains))}")
    print(f"  Pearson r(ET0, rainfall)   : {pearson:+.4f}")
    print(f"  distinct crops             : {len(set(r['crop'] for r in num_rows))}")
    print(f"  distinct stages            : {sorted(set(r['grounding']['params']['stage'] for r in num_rows))}")
    print(f"  distinct (area_value,unit) : {sorted(set((r['grounding']['params']['area_value'], r['grounding']['params']['area_unit']) for r in num_rows))}")
    print(f"  ET0 range                  : {min(et0s)} .. {max(et0s)}")
    print(f"  rainfall range             : {min(rains)} .. {max(rains)}")

    print("\n-- rainfall tier distribution (rows) --")
    tc = collections.Counter(r["grounding"]["tier"] for r in num_rows)
    for t in (1, 2, 3):
        print(f"  tier {t} (x{ic.TIER_FACTOR[t]}): {tc[t]:5d}  ({tc[t]/len(num_rows)*100:.1f}%)")

    print("\n-- numeral / unit script mix (BN rows) --")
    bn_rows = [r for r in num_rows if r["lang"] == "bn"]
    print(f"  bengali numerals : {sum(1 for r in bn_rows if r['grounding']['numeral_script']=='bn')}/{len(bn_rows)}")
    print(f"  bengali units    : {sum(1 for r in bn_rows if r['grounding']['unit_script']=='bn')}/{len(bn_rows)}")

    floored = sum(1 for r in num_rows if r["grounding"]["floored_at_zero"])
    print(f"\n-- outcome distribution --")
    print(f"  rows floored to 0 mm       : {floored} ({floored/len(num_rows)*100:.1f}%)")
    print(f"  rows clamped by BARC bounds: 0 (irrigation_rules.csv absent)")

    print("\n-- <irrigation_calc> containment --")
    print(f"  rows containing the tag    : {len(tagged)}")
    print(f"  ...of question_type numeric: {sum(1 for r in tagged if r['question_type']=='numeric_runtime')}")
    print(f"  ...any other type          : {sum(1 for r in tagged if r['question_type']!='numeric_runtime')}")

    print("\n" + "=" * 72)
    print("3 SAMPLE NUMERIC RECORDS")
    print("=" * 72)
    for r in RNG.sample(num_rows, 3):
        print(json.dumps(r, ensure_ascii=False, indent=2))
        print("-" * 72)

    print(f"\nWrote {OUT_ALL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
