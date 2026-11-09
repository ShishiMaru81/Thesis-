#!/usr/bin/env python3
"""
s0_build_seed_corpus.py — v3 STAGE 0: provenance-tagged seed corpus.

WHAT CHANGED FROM v2 AND WHY
----------------------------
v2's Json_Dataset(Calander)/generate_qa.py emitted 6 hand-written paraphrases
per fact per language, THEN split. Every paraphrase of a fact shared a
group_id, so the split itself was group-aware — but the corpus that went into
the split was already augmented. That is the generate-then-split ordering the
supervisor flagged.

v3 inverts it. THIS script emits exactly ONE canonical question per
(fact, language). Paraphrase / back-translation / synthetic / noise expansion
is Stage 2's job and touches train_seed.jsonl ONLY. Consequence: nothing in
val or test is ever a generated variant of anything. That is HARD RULE 1,
enforced structurally rather than by assertion.

Rebuilt from the ORIGINAL BARC extractions (Json_Dataset(Calander)/*.json,
28 crop calendar images). data/v2/ is never opened.

THREE ANSWER SOURCES, unchanged in spirit from v2:
  retrieved — answer text is a verified field value, or a deterministic
              composition of typed field values (see RENDERERS). No wording is
              invented beyond fixed connective text; no value is imputed.
  computed  — answer produced by irrigation_calc.compute(). The number is
              calculated here, never taken from an LLM.
  abstain   — out-of-calendar-scope questions whose gold answer is a refusal.

HARD RULE 3 (never impute): a field whose provenance is "absent" produces no
row at all. A field flagged exclude_from_qa_generation / needs_human_review /
numeric_backfill_declined is skipped and logged, never guessed.

HARD RULE 5 (seeded randomness): the only randomness here is (a) row_id
generation and (b) the numeric_runtime scenario tuples. Both are driven by
random.Random(42). Re-running this script byte-for-byte reproduces
seed_corpus.jsonl.

Run:  python s0_build_seed_corpus.py
"""
from __future__ import annotations

import argparse
import collections
import datetime
import glob
import hashlib
import json
import os
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import irrigation_calc as ic  # noqa: E402  (single source of arithmetic truth)

SEED = 42

V3_ROOT = Path(__file__).resolve().parent.parent
THESIS_ROOT = V3_ROOT.parent
BARC_DIR = THESIS_ROOT / "Json_Dataset(Calander)"
OUT_JSONL = V3_ROOT / "data" / "seed" / "seed_corpus.jsonl"
LOG_PATH = V3_ROOT / "logs" / "s0_build_seed_corpus.log"
GATE_JSON = V3_ROOT / "reports" / "gate0_seed_corpus.json"

# ----------------------------------------------------------------- helpers
BD = "০১২৩৪৫৬৭৮৯"


def bn_num(n) -> str:
    """ASCII digits -> Bengali digits. Non-digits (., -, /) pass through."""
    return "".join(BD[int(c)] if c.isdigit() else c for c in str(n))


def crop_slug(crop_en: str) -> str:
    return crop_en.strip().lower().replace(" ", "_")


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def scenario_hash(crop, area_unit, area_value, et0, rain, stage) -> str:
    """SHA-256 over the sorted (crop, area_unit, area_value, et0, rain, stage)
    tuple. The tuple mixes str and float, which is not sortable in Python, so
    each element is first stringified as "key=value" and the resulting six
    strings are sorted. Deterministic and order-independent, which is the point.
    """
    parts = sorted([
        f"crop={crop}",
        f"area_unit={area_unit}",
        f"area_value={area_value}",
        f"et0={et0}",
        f"rain={rain}",
        f"stage={stage}",
    ])
    return sha256_hex("|".join(parts))


class SeededIds:
    """uuid4-format row_ids from a seeded stream.

    A real uuid.uuid4() would break HARD RULE 5 — the corpus would differ on
    every run and no hash of it would mean anything. These carry the uuid4
    variant/version bits so they are valid UUIDs, but they are reproducible.
    """

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def next(self) -> str:
        raw = bytes(self._rng.getrandbits(8) for _ in range(16))
        return str(uuid.UUID(bytes=raw, version=4))


# ----------------------------------------------------------------- field map
# field_key -> (row_index, attribute slug, stage label, question EN, question BN)
#
# Four of the 24 extracted fields produce no seed rows and are recorded as
# deliberate skips in the Gate 0 report:
#   1_ফসলের_নাম          degenerate ("what is this crop called" answers itself)
#   4_আবহাওয়া_মাসিক_মানচিত্র  a 12-entry month wheel, not a text fact; rendering it
#                        as prose would be composition, not extraction
#   10_পানি_গভীরতা_মিমি    provenance "absent" in all 28 files
#   18_বৃষ্টি_সংবেদনশীল_থ্রেশহোল্ড  provenance "absent" in all 28 files
FIELDS = {
    "2_ফসলের_ধরন": (
        2, "crop_type", "na",
        "What type of crop is {crop}?",
        "{crop_bn} কোন ধরনের ফসল?"),
    "3_বপন_ঋতু": (
        3, "sowing_season", "na",
        "What is the sowing season for {crop}?",
        "{crop_bn} বপনের/রোপণের সময় কখন?"),
    "5_প্রথম_সেচ_সময়": (
        5, "first_irrigation_timing", "na",
        "When should the first irrigation be applied for {crop}?",
        "{crop_bn} ফসলে প্রথম সেচ কখন দিতে হবে?"),
    "6_প্রথম_সেচ_ধরন": (
        6, "first_irrigation_type", "na",
        "What kind of first irrigation does {crop} need?",
        "{crop_bn} ফসলে প্রথম সেচ কী ধরনের হওয়া উচিত?"),
    "7_সেচ_ব্যবধান_দিন": (
        7, "irrigation_interval_days", "na",
        "How often should {crop} be irrigated?",
        "{crop_bn} ফসলে কত দিন পর পর সেচ দিতে হবে?"),
    "8_সেচ_পদ্ধতি": (
        8, "irrigation_method", "na",
        "What irrigation method is recommended for {crop}?",
        "{crop_bn} ফসলে কোন সেচ পদ্ধতি সুপারিশ করা হয়েছে?"),
    "9_সংকটকাল_পর্যায়সমূহ": (
        9, "critical_stages", "na",
        "Which growth stages of {crop} are most sensitive to water stress?",
        "{crop_bn} ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?"),
    "11_পানি_গভীরতা_সেমি": (
        11, "water_depth_cm", "na",
        "What standing water depth in centimetres does the calendar give for {crop}?",
        "{crop_bn} খেতে পঞ্জিকা অনুযায়ী কত সেন্টিমিটার পানি রাখতে হবে?"),
    "12_বেসিন_জ্যামিতি": (
        12, "basin_geometry", "na",
        "What size basin should be made around a {crop} tree for irrigation?",
        "{crop_bn} গাছের গোড়ায় সেচের জন্য কত বড় বেসিন করতে হবে?"),
    "13_নিষ্কাশন_নিয়ম": (
        13, "drainage_rule", "na",
        "What should be done about drainage for {crop}?",
        "{crop_bn} ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত?"),
    "14_জল_চাহিদা_সংশোধক": (
        14, "water_demand_modifier", "na",
        "What practice changes the water requirement of {crop}?",
        "{crop_bn} ফসলের পানির চাহিদা কোন ব্যবস্থায় পরিবর্তন হয়?"),
    "15_মৌসুমী_শর্তাবলী": (
        15, "seasonal_conditions", "na",
        "Are there seasonal conditions affecting irrigation for {crop}?",
        "{crop_bn} ফসলের সেচে মৌসুমী কোন শর্ত প্রযোজ্য?"),
    "16_প্রতিষ্ঠা_পর্যায়_সেচ_ব্যবধান": (
        16, "establishment_interval", "establishment",
        "How often should {crop} be irrigated during the establishment stage?",
        "{crop_bn} ফসলের প্রতিষ্ঠা পর্যায়ে কত দিন পর পর সেচ দিতে হবে?"),
    "17_পর্যায়_সংক্রান্ত_সেচ_সূচী": (
        17, "stage_irrigation_schedule", "na",
        "What is the stage-by-stage irrigation schedule for {crop}?",
        "{crop_bn} ফসলের পর্যায় অনুযায়ী সেচের সূচী কী?"),
    "19_শুষ্ক_ঋতু_কৌশল": (
        19, "dry_season_strategy", "na",
        "What is the dry-season irrigation strategy for {crop}?",
        "{crop_bn} ফসলের শুষ্ক মৌসুমের সেচ কৌশল কী?"),
    "20_বর্ষা_ঋতু_কৌশল": (
        20, "monsoon_strategy", "na",
        "What is the rainy-season irrigation strategy for {crop}?",
        "{crop_bn} ফসলের বর্ষা মৌসুমের সেচ কৌশল কী?"),
    "21_বৃদ্ধি_পর্যায়_দিনের_সীমা": (
        21, "growth_stage_day_ranges", "na",
        "What are the day ranges of the growth stages of {crop}?",
        "{crop_bn} ফসলের বৃদ্ধির পর্যায়গুলো কত দিনের?"),
    "22_বয়স_নির্ভর_সেচ": (
        22, "age_dependent_irrigation", "na",
        "How does the age of the {crop} plant change its irrigation need?",
        "{crop_bn} গাছের বয়স অনুযায়ী সেচের প্রয়োজন কীভাবে বদলায়?"),
    "23_মাটির_ধরন_প্রভাব": (
        23, "soil_type_effect", "na",
        "How does soil type affect irrigation for {crop}?",
        "{crop_bn} ফসলের সেচে মাটির ধরনের প্রভাব কী?"),
    "24_জল_চাপ_সূচক": (
        24, "water_stress_indicator", "na",
        "What visible signs show that {crop} is short of water?",
        "{crop_bn} গাছে পানির অভাব হলে কী লক্ষণ দেখা যায়?"),
}

SKIPPED_FIELDS = {
    "1_ফসলের_নাম": "degenerate - the question restates the answer",
    "4_আবহাওয়া_মাসিক_মানচিত্র": "12-entry month wheel, not a text fact; prose rendering would be composition not extraction",
    "10_পানি_গভীরতা_মিমি": "provenance 'absent' in all 28 files",
    "18_বৃষ্টি_সংবেদনশীল_থ্রেশহোল্ড": "provenance 'absent' in all 28 files",
}


# ----------------------------------------------------------------- renderers
# Each renderer turns typed field values into one (answer_en, answer_bn) pair.
# The ONLY text a renderer contributes is fixed connective wording; every
# content word and every number comes from the extraction. Returns None when
# the field carries nothing renderable, so the caller can skip rather than
# invent (HARD RULE 3).

def _rng_str(lo, hi):
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def render_text(v, crop, crop_bn):
    en, bn = v.get("value_en"), v.get("value_bn")
    return (en, bn) if (en or bn) else None


def render_timing(v, crop, crop_bn):
    en, bn = v.get("timing_en"), v.get("timing_bn")
    return (en, bn) if (en or bn) else None


def render_soil_pref(v, crop, crop_bn):
    en, bn = v.get("soil_preference_en"), v.get("soil_preference_bn")
    return (en, bn) if (en or bn) else None


def render_interval_days(v, crop, crop_bn):
    lo, hi = v.get("value_days_min"), v.get("value_days_max")
    cond_en, cond_bn = v.get("condition_en"), v.get("condition_bn")
    if lo is not None and hi is not None:
        en = f"Irrigate {crop} every {_rng_str(lo, hi)} days."
        bn = f"{crop_bn} ফসলে প্রতি {bn_num(_rng_str(lo, hi))} দিন অন্তর সেচ দিতে হবে।"
        if cond_en:
            en += f" Condition: {cond_en}."
        if cond_bn:
            bn += f" শর্ত: {cond_bn}।"
        return en, bn
    if cond_en or cond_bn:
        # The calendar states a rule but no day count. Say exactly that; do not
        # substitute a plausible interval.
        en = (f"The calendar does not state a fixed number of days between "
              f"irrigations for {crop}. It states: {cond_en}.") if cond_en else None
        bn = (f"পঞ্জিকায় {crop_bn} ফসলের সেচের নির্দিষ্ট দিনসংখ্যা দেওয়া নেই। "
              f"এতে বলা হয়েছে: {cond_bn}।") if cond_bn else None
        return (en, bn)
    return None


def render_irrigation_count(v, crop, crop_bn):
    lo, hi = v.get("irrigations_min"), v.get("irrigations_max")
    if lo is None and hi is None:
        return None
    per_en, per_bn = v.get("period_en"), v.get("period_bn")
    en = f"Give {crop} {_rng_str(lo, hi)} irrigations"
    bn = f"{crop_bn} গাছে {bn_num(_rng_str(lo, hi))} বার সেচ দিতে হবে"
    if per_en:
        en += f" during the {per_en.lower()}"
    if per_bn:
        bn = f"{crop_bn} গাছে {per_bn} {bn_num(_rng_str(lo, hi))} বার সেচ দিতে হবে"
    return en + ".", bn + "।"


def render_depth_cm(v, crop, crop_bn):
    lo, hi = v.get("value_cm_min"), v.get("value_cm_max")
    if lo is None and hi is None:
        return None
    return (f"Keep {_rng_str(lo, hi)} cm of standing water in the {crop} field.",
            f"{crop_bn} খেতে {bn_num(_rng_str(lo, hi))} সেন্টিমিটার পানি ধরে রাখতে হবে।")


def render_basin(v, crop, crop_bn):
    r = v.get("radius_m")
    dlo, dhi = v.get("depth_cm_min"), v.get("depth_cm_max")
    if r is None and dlo is None:
        return None
    en_parts, bn_parts = [], []
    if r is not None:
        en_parts.append(f"make a basin of about {r} m radius around the tree")
        bn_parts.append(f"গাছের চারপাশে প্রায় {bn_num(r)} মিটার ব্যাসার্ধের বেসিন করতে হবে")
    if dlo is not None:
        en_parts.append(f"irrigate it to a depth of {_rng_str(dlo, dhi)} cm")
        bn_parts.append(f"তাতে {bn_num(_rng_str(dlo, dhi))} সেন্টিমিটার গভীর করে সেচ দিতে হবে")
    return (f"For {crop}, " + " and ".join(en_parts) + ".",
            f"{crop_bn} গাছের জন্য " + " এবং ".join(bn_parts) + "।")


STAGE_LIST_LEAD = {
    "critical_stages": ("The water-sensitive growth stages of {crop} are: ",
                        "{crop_bn} ফসলের পানি-সংবেদনশীল বৃদ্ধির পর্যায়গুলো হলো: "),
    "stage_irrigation_schedule": ("Irrigation schedule for {crop}: ",
                                  "{crop_bn} ফসলের সেচ সূচী: "),
    "growth_stage_day_ranges": ("Growth stage day ranges for {crop}: ",
                                "{crop_bn} ফসলের বৃদ্ধির পর্যায়ের দিনসীমা: "),
}


def render_stage_list(v, crop, crop_bn, attribute=None):
    stages = v.get("stages")
    if not stages:
        return None
    lead_en, lead_bn = STAGE_LIST_LEAD[attribute]
    items_en, items_bn = [], []
    for s in stages:
        en, bn = s.get("stage_en"), s.get("stage_bn")
        # growth_stage_day_ranges asks for DAY RANGES, so days_min/days_max have
        # to reach the answer. Some sources (Wheat) already spell the range out
        # inside stage_en; appending it again would duplicate, so only append
        # when the number is not already in the prose.
        dlo, dhi = s.get("days_min"), s.get("days_max")
        if dlo is not None:
            rng = _rng_str(dlo, dhi)
            if en and str(dlo) not in en:
                en = f"{en} ({rng} days)"
            if bn and bn_num(str(dlo)) not in bn:
                bn = f"{bn} ({bn_num(rng)} দিন)"
        if s.get("condition_en"):
            en = f"{en} — {s['condition_en']}"
        if s.get("condition_bn"):
            bn = f"{bn} — {s['condition_bn']}"
        if en:
            items_en.append(en)
        if bn:
            items_bn.append(bn)
    if not items_en and not items_bn:
        return None
    return (lead_en.format(crop=crop) + "; ".join(items_en) + "." if items_en else None,
            lead_bn.format(crop_bn=crop_bn) + "; ".join(items_bn) + "।" if items_bn else None)


# Renderer chain, tried in order. First one that returns a value wins, so a
# field carrying BOTH free text and parsed numerics (e.g. Maize field 5) uses
# the verified prose rather than a recomposition of it.
RENDERER_CHAIN = [
    ("text", render_text),
    ("timing", render_timing),
    ("soil_preference", render_soil_pref),
    ("interval_days", render_interval_days),
    ("irrigation_count", render_irrigation_count),
    ("depth_cm", render_depth_cm),
    ("basin", render_basin),
    ("stage_list", render_stage_list),
]


def usable(v) -> tuple[bool, str]:
    """v2's usable() gate, verbatim in behaviour, but reporting WHY."""
    if not isinstance(v, dict):
        return False, "not_a_dict"
    if v.get("provenance") in ("absent", None):
        return False, "provenance_absent"
    if v.get("exclude_from_qa_generation"):
        return False, "exclude_from_qa_generation"
    if v.get("needs_human_review"):
        return False, "needs_human_review"
    if v.get("numeric_backfill_declined"):
        return False, "numeric_backfill_declined"
    return True, "ok"


# ------------------------------------------------- numeric_runtime scenarios
# Stage 2C will sample synthetic scenarios from these same distributions and
# must reject any scenario_hash already present in val or test — so the seed
# scenarios have to come from the distribution, not from a fixed list. v2 used
# three hard-coded tuples shared by all 28 crops; that is why v2's numeric test
# items were near-identical to its numeric train items.
AREA_UNITS = ("acre", "bigha", "katha", "decimal", "hectare", "sqm")
RAIN_TIERS = (((0.0, 5.0), 0.40), ((5.0, 15.0), 0.35), ((15.0, 60.0), 0.25))

# Surface form of the unit in Bengali text. v2 interpolated the English unit
# token into its Bengali templates ("০.৮৪ bigha"), which made every Bengali
# numeric item silently code-switched. Stage 4 defines `code_switched` as a
# distinct adversarial test category, so the baseline bn items must not already
# be code-switched or that category measures nothing. The MACHINE-READABLE slot
# in `scenario` / `gold_trace` stays English — only the prose is translated.
AREA_UNIT_BN = {
    "acre": "একর",
    "bigha": "বিঘা",
    "katha": "কাঠা",
    "decimal": "শতক",
    "hectare": "হেক্টর",
    "sqm": "বর্গমিটার",
}


def sample_scenario(rng: random.Random) -> dict:
    et0 = round(rng.uniform(2.0, 9.0), 1)
    u = rng.random()
    acc = 0.0
    for (lo, hi), w in RAIN_TIERS:
        acc += w
        if u <= acc:
            rain = round(rng.uniform(lo, hi), 1)
            break
    else:
        rain = round(rng.uniform(*RAIN_TIERS[-1][0]), 1)
    area_unit = rng.choice(AREA_UNITS)
    area_value = 10 ** rng.uniform(-1.0, 1.0)          # LogUniform(0.1, 10)
    area_value = round(area_value, 2 if area_value < 1 else 1)
    return {"et0_mm": et0, "rainfall_mm": rain,
            "area_value": area_value, "area_unit": area_unit}


NUMERIC_Q_EN = ("My {crop} field is at the {stage} stage. Today ET0 is {et0} mm, "
                "it rained {rain} mm, and the field is {aval} {aunit}. "
                "How much irrigation water does it need today?")
NUMERIC_Q_BN = ("আমার {crop_bn} ক্ষেত {stage_bn} পর্যায়ে আছে। আজ ET0 {et0_bn} মিমি, "
                "বৃষ্টি হয়েছে {rain_bn} মিমি, জমির পরিমাণ {aval_bn} {aunit}। "
                "আজ কত পানি সেচ দিতে হবে?")


def numeric_answers(res: dict, crop: str, crop_bn: str) -> tuple[str, str]:
    """Answer wording kept identical to v2 generate_qa.py:428-433 so v2 and v3
    numeric gold strings stay comparable; only the inputs changed."""
    en = (f"Net irrigation depth = Kc×ET0 − effective rainfall "
          f"= {res['kc_used']}×{res['et0_mm']} − {res['effective_rain_mm']} "
          f"= {res['net_mm']} mm. For {res['area_value']} {res['area_unit']} "
          f"({res['area_sqm']:.0f} m²), that is about {res['liters']:.0f} liters today.")
    bn = (f"নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি "
          f"= {bn_num(res['kc_used'])}×{bn_num(res['et0_mm'])} − "
          f"{bn_num(res['effective_rain_mm'])} = {bn_num(res['net_mm'])} মিমি। "
          f"{bn_num(res['area_value'])} {AREA_UNIT_BN[res['area_unit']]} "
          f"({bn_num(f'{res['area_sqm']:.0f}')} বর্গমিটার)-এর জন্য আজ প্রায় "
          f"{bn_num(f'{res['liters']:.0f}')} লিটার।")
    return en, bn


# ----------------------------------------------------------------- abstain
# v2 picked ONE topic per crop at random. v3 emits all three so the abstain
# stratum is large enough to survive a 70/10/20 group split with a non-empty
# test cell for every crop.
ABSTAIN_TOPICS = [
    ("market_price",
     "What is the market price of {crop} today?",
     "{crop_bn} ফসলের আজকের বাজার দর কত?"),
    ("pesticide_brand",
     "Which pesticide brand is cheapest for {crop}?",
     "{crop_bn} ফসলের জন্য সবচেয়ে সস্তা কীটনাশক ব্র্যান্ড কোনটি?"),
    ("soil_ph",
     "What is the exact soil pH at my farm for {crop}?",
     "আমার খামারে {crop_bn} ফসলের সঠিক মাটির pH কত?"),
]
ABSTAIN_EN = ("The crop calendar does not specify this. I can only answer irrigation "
              "and cultivation questions grounded in the BARC calendar; please consult "
              "a market/soil source for this.")
ABSTAIN_BN = ("ফসল পঞ্জিকায় এই তথ্য নেই। আমি কেবল BARC পঞ্জিকা-ভিত্তিক সেচ ও চাষাবাদ "
              "সংক্রান্ত প্রশ্নের উত্তর দিতে পারি; এর জন্য বাজার/মাটি সংক্রান্ত উৎস দেখুন।")


# ----------------------------------------------------------------- build
def build(barc_dir: Path, out_path: Path) -> dict:
    ids = SeededIds(SEED)
    scen_rng = random.Random(SEED)

    files = sorted(glob.glob(str(barc_dir / "*.json")))
    rows: list[dict] = []
    skips: list[dict] = []
    lang_asymmetry: list[dict] = []
    kc_missing: list[str] = []

    for path in files:
        fname = os.path.basename(path)
        doc = json.load(open(path, encoding="utf-8"))
        crop = doc["crop_name_en"]
        crop_bn = doc.get("crop_name_bn", crop)
        image = doc["source_image"]
        img_sha = doc.get("source_image_sha256")
        feats = doc.get("irrigation_features", {})

        def emit(lang, question, answer, qtype, fact_id, source_id,
                 answer_kind, prov_extra, scen_hash=None, extra=None):
            rec = {
                "row_id": ids.next(),
                "source_id": source_id,
                "fact_id": fact_id,
                "scenario_hash": scen_hash,
                "question": question,
                "answer": answer,
                "lang": lang,
                "qtype": qtype,
                "crop": crop,
                "origin": "seed",
                "provenance": {
                    "source_file": fname,
                    "extraction_method": prov_extra.pop("extraction_method"),
                    "verified_by": None,
                    **prov_extra,
                },
                "crop_bn": crop_bn,
                "answer_kind": answer_kind,
            }
            if extra:
                rec.update(extra)
            rows.append(rec)

        # ---- (A) factual rows -------------------------------------------
        for fkey, (ridx, attribute, stage, q_en, q_bn) in FIELDS.items():
            v = feats.get(fkey)
            ok, why = usable(v)
            if not ok:
                if why != "provenance_absent":       # absent is expected, not notable
                    skips.append({"file": fname, "field": fkey, "reason": why,
                                  "quote": (v or {}).get("source_quote_bn")})
                continue

            rendered = None
            used_renderer = None
            for name, fn in RENDERER_CHAIN:
                out = fn(v, crop, crop_bn, attribute) if name == "stage_list" else fn(v, crop, crop_bn)
                if out and (out[0] or out[1]):
                    rendered, used_renderer = out, name
                    break
            if rendered is None:
                skips.append({"file": fname, "field": fkey,
                              "reason": "no_renderer_matched_field_shape",
                              "keys": sorted(v.keys())})
                continue

            a_en, a_bn = rendered
            if bool(a_en) != bool(a_bn):
                lang_asymmetry.append({"file": fname, "field": fkey,
                                       "has_en": bool(a_en), "has_bn": bool(a_bn)})

            fact_id = f"{crop_slug(crop)}_{stage}_{attribute}"
            source_id = f"{image}#f{ridx:02d}"
            prov = {
                "extraction_method": "gemini_vision",
                "source_image": image,
                "source_image_sha256": img_sha,
                "field_key": fkey,
                "field_provenance": v.get("provenance"),
                "field_source_section": v.get("source"),
                "source_quote_bn": v.get("source_quote_bn"),
                "renderer": used_renderer,
                "human_verified_upstream": bool(doc.get("human_verified")),
            }
            if a_en:
                emit("en", q_en.format(crop=crop), a_en, "factual",
                     fact_id, source_id, "retrieved", dict(prov))
            if a_bn:
                emit("bn", q_bn.format(crop_bn=crop_bn), a_bn, "factual",
                     fact_id, source_id, "retrieved", dict(prov))

        # ---- (B) numeric_runtime rows ------------------------------------
        for si, stage in enumerate(ic.STAGES, start=1):
            scen = sample_scenario(scen_rng)
            try:
                res = ic.compute(crop, stage, scen["et0_mm"], scen["rainfall_mm"],
                                 scen["area_value"], scen["area_unit"])
            except ic.UnknownCrop:
                if crop not in kc_missing:
                    kc_missing.append(crop)
                break
            a_en, a_bn = numeric_answers(res, crop, crop_bn)
            shash = scenario_hash(crop, scen["area_unit"], scen["area_value"],
                                  scen["et0_mm"], scen["rainfall_mm"], stage)
            fact_id = f"{crop_slug(crop)}_{stage}_net_irrigation"
            source_id = f"{image}#n{si:02d}"
            prov = {
                "extraction_method": "gemini_vision+deterministic_calc",
                "source_image": image,
                "source_image_sha256": img_sha,
                "field_key": "computable_link",
                "field_provenance": "computed",
                "field_source_section": None,
                "source_quote_bn": None,
                "renderer": "irrigation_calc.compute",
                "kc_source": res["kc_source"],
                "kc_citation": res["citation"],
                "human_verified_upstream": bool(doc.get("human_verified")),
            }
            extra = {"scenario": {**scen, "stage": stage, "crop": crop},
                     "gold_trace": res}
            emit("en", NUMERIC_Q_EN.format(crop=crop, stage=stage, et0=scen["et0_mm"],
                                           rain=scen["rainfall_mm"],
                                           aval=scen["area_value"], aunit=scen["area_unit"]),
                 a_en, "numeric_runtime", fact_id, source_id, "computed",
                 dict(prov), shash, extra)
            emit("bn", NUMERIC_Q_BN.format(crop_bn=crop_bn, stage_bn=ic.STAGE_BN[stage],
                                           et0_bn=bn_num(scen["et0_mm"]),
                                           rain_bn=bn_num(scen["rainfall_mm"]),
                                           aval_bn=bn_num(scen["area_value"]),
                                           aunit=AREA_UNIT_BN[scen["area_unit"]]),
                 a_bn, "numeric_runtime", fact_id, source_id, "computed",
                 dict(prov), shash, extra)

        # ---- (C) abstain rows ---------------------------------------------
        for ai, (topic, q_en, q_bn) in enumerate(ABSTAIN_TOPICS, start=1):
            fact_id = f"{crop_slug(crop)}_na_abstain_{topic}"
            source_id = f"{image}#x{ai:02d}"
            prov = {
                "extraction_method": "authored_policy",
                "source_image": image,
                "source_image_sha256": img_sha,
                "field_key": None,
                "field_provenance": "out_of_calendar_scope",
                "field_source_section": None,
                "source_quote_bn": None,
                "renderer": "abstain_policy",
                "abstain_topic": topic,
                "human_verified_upstream": bool(doc.get("human_verified")),
            }
            emit("en", q_en.format(crop=crop), ABSTAIN_EN, "abstain",
                 fact_id, source_id, "abstain", dict(prov))
            emit("bn", q_bn.format(crop_bn=crop_bn), ABSTAIN_BN, "abstain",
                 fact_id, source_id, "abstain", dict(prov))

    # ---- deterministic ordering, then write ------------------------------
    rows.sort(key=lambda r: (r["source_id"], r["qtype"], r["lang"], r["fact_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- GATE 0 report ----------------------------------------------------
    REQUIRED_PROV = ("source_file", "extraction_method", "verified_by")
    missing_prov = [
        {"row_id": r["row_id"], "missing": [k for k in REQUIRED_PROV
                                            if k not in r.get("provenance", {})]}
        for r in rows
        if any(k not in r.get("provenance", {}) for k in REQUIRED_PROV)
    ]
    dup_q = [q for q, n in collections.Counter(r["question"] for r in rows).items() if n > 1]

    report = {
        "stage": "GATE 0",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": SEED,
        "source_dir": str(barc_dir),
        "source_files": len(files),
        "output": str(out_path),
        "output_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "total_rows": len(rows),
        "rows_per_qtype": dict(sorted(collections.Counter(r["qtype"] for r in rows).items())),
        "rows_per_lang": dict(sorted(collections.Counter(r["lang"] for r in rows).items())),
        "rows_per_qtype_lang": {f"{q}/{l}": n for (q, l), n in sorted(
            collections.Counter((r["qtype"], r["lang"]) for r in rows).items())},
        "rows_per_answer_kind": dict(sorted(collections.Counter(r["answer_kind"] for r in rows).items())),
        "distinct_source_id": len({r["source_id"] for r in rows}),
        "distinct_fact_id": len({r["fact_id"] for r in rows}),
        "distinct_crop": len({r["crop"] for r in rows}),
        "distinct_scenario_hash": len({r["scenario_hash"] for r in rows if r["scenario_hash"]}),
        "rows_missing_provenance_field": missing_prov,
        "rows_missing_provenance_field_count": len(missing_prov),
        "duplicate_question_strings": len(dup_q),
        "rows_per_field": dict(sorted(collections.Counter(
            r["provenance"].get("field_key") or "-" for r in rows).items())),
        "rows_per_renderer": dict(sorted(collections.Counter(
            r["provenance"]["renderer"] for r in rows).items())),
        "skipped_fields_by_design": SKIPPED_FIELDS,
        "skipped_field_instances": skips,
        "skipped_field_instances_count": len(skips),
        "skip_reason_counts": dict(sorted(collections.Counter(s["reason"] for s in skips).items())),
        "language_asymmetry": lang_asymmetry,
        "crops_without_kc_row": kc_missing,
    }
    GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    GATE_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        f"s0_build_seed_corpus.py\nseed={SEED}\nrun_at={report['generated_at']}\n"
        f"rows={len(rows)}\nseed_corpus.sha256={report['output_sha256']}\n"
        f"skips={len(skips)}\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=str(BARC_DIR))
    ap.add_argument("--out", dest="outfile", default=str(OUT_JSONL))
    a = ap.parse_args()
    rep = build(Path(a.indir), Path(a.outfile))

    print("=" * 66)
    print("GATE 0 — PROVENANCE TAGGING")
    print("=" * 66)
    print(f"source files            : {rep['source_files']}")
    print(f"total rows              : {rep['total_rows']}")
    print(f"rows per qtype          : {rep['rows_per_qtype']}")
    print(f"rows per lang           : {rep['rows_per_lang']}")
    print(f"distinct source_id      : {rep['distinct_source_id']}")
    print(f"distinct fact_id        : {rep['distinct_fact_id']}")
    print(f"distinct crop           : {rep['distinct_crop']}")
    print(f"rows missing provenance : {rep['rows_missing_provenance_field_count']}")
    print(f"duplicate questions     : {rep['duplicate_question_strings']}")
    print(f"skipped field instances : {rep['skipped_field_instances_count']} "
          f"{rep['skip_reason_counts']}")
    print(f"seed_corpus.jsonl sha256: {rep['output_sha256']}")
    print(f"\nfull report -> {GATE_JSON}")
    print("STOP. Gate 0 requires review before Stage 1.")
