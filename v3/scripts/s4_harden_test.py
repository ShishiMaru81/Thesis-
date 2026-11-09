#!/usr/bin/env python3
"""
s4_harden_test.py — STAGE 4, test-set hardening.

WHAT THIS DOES
--------------
Takes the locked 182-row seed test set and adds 240 authored adversarial items
across 7 categories, producing a 422-row hardened test set for Stages 5-8.

    182 seed  +  240 adversarial  =  422 rows       (211 bn / 211 en)

WHY
---
The seed test set is drawn from the same generator as the training corpus:
same question templates, same three qtypes, same 28 crops. A fine-tuned model
will sit near ceiling on it, and a near-ceiling number cannot discriminate
between the baseline and the tuned model. The 240 adversarial items introduce
input pathologies the training distribution never contains, so Stages 5-8
measure robustness rather than template recall.

HARD RULE — CATEGORIES 5 AND 6 CARRY NO MACHINE-GENERATED GOLD
--------------------------------------------------------------
Category 5 (conflicting_values) and category 6 (out_of_domain) require human
judgment about what the correct response even is:

  * cat 5 — when a farmer contradicts themself, whether the right move is to
    ask which value holds, to take the later value, or to answer both ways is
    a policy decision this script has no standing to make.
  * cat 6 — the boundary of "out of calendar scope" and the exact wording of a
    refusal are policy, and the seed corpus's three abstain topics are
    deliberately not reused here, so no existing gold transfers.

For all 60 rows in those two categories this script emits the question
scaffold only:

    answer       = ""
    answer_kind  = "pending_human"
    gold_status  = "pending_human"
    proposed_gold in the authoring sheet = "" (blank)

`qtype` on those rows is set to the behaviour class named by the Stage 4
category table itself ("abstain" for cat 6, "clarify" for cat 5) — that is the
study design's own label, not an answer. Nothing else about the response is
asserted. Stage 5 MUST refuse to score any row with
gold_status == "pending_human"; the gate report counts them so the omission
cannot pass silently.

ABSENT INPUT — irrigation_rules.csv
-----------------------------------
The Stage 4 brief lists irrigation_rules.csv (per-crop min/max water bounds) as
an input. That file has never existed on disk; see the irrigation_calc.py
module docstring, which documents the gap as a standing v1 decision rather than
substituting a plausible default. Consequently no gold answer in this file is
clamped to a BARC bound. Every computed gold carries the calculator's own
`clamped: false` / `clamp_source: "none - irrigation_rules.csv absent"` through
gold_trace, exactly as Stages 0-3 do. If the CSV is ever authored, every
numeric gold here must be regenerated.

ARITHMETIC PROVENANCE
---------------------
Every number in every computed gold comes from irrigation_calc.compute() — the
same function that produced the seed corpus and that the runtime bridge calls.
This script contains no formula of its own. The one arithmetic operation
performed outside the calculator is the area conversion for category 3c
(farmer-stated local bigha size), which uses irrigation_calc.liters_for() on a
sqm value the farmer supplied; that is flagged in gold_trace as
area_source = "farmer_stated_conversion".

LEAKAGE SAFETY
--------------
Every synthesized scenario is hashed with the Stage 0 scenario_hash() and
rejected if it collides with (a) the val/test held-out blocklist, (b) any
scenario_hash in train_augmented.jsonl, or (c) another adversarial row. New
test items must not be scenarios the model was trained on.

OUTPUTS
-------
  data/hardened/test_hardened.jsonl          422 rows
  data/hardened/test_hardened.sha256
  reports/human_eval/test_authoring_sheet.csv  240 adversarial rows, UTF-8-BOM
  reports/gate4_hardened.json
  logs/s4_harden_test.log
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import random
import sys
import uuid
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import irrigation_calc as ic          # noqa: E402
import bn_normalize as bnorm          # noqa: E402

V3_ROOT = SCRIPT_DIR.parent

SEED = 42
GENERATOR_VERSION = "v3-s4.1"

TEST_SEED_PATH = V3_ROOT / "data" / "splits" / "test.jsonl"
TEST_SEED_SHA = V3_ROOT / "data" / "splits" / "test.sha256"
TRAIN_SEED_PATH = V3_ROOT / "data" / "splits" / "train_seed.jsonl"
TRAIN_AUG_PATH = V3_ROOT / "data" / "augmented" / "train_augmented.jsonl"
BLOCKLIST_PATH = V3_ROOT / "data" / "splits" / "heldout_scenario_hashes.json"

OUT_JSONL = V3_ROOT / "data" / "hardened" / "test_hardened.jsonl"
OUT_SHA = V3_ROOT / "data" / "hardened" / "test_hardened.sha256"
OUT_SHEET = V3_ROOT / "reports" / "human_eval" / "test_authoring_sheet.csv"
GATE_JSON = V3_ROOT / "reports" / "gate4_hardened.json"
LOG_PATH = V3_ROOT / "logs" / "s4_harden_test.log"

# The brief's own table. Enforced as an assertion at the end of build().
CATEGORY_TARGETS = {
    "misspelled_crop": 40,
    "missing_parameter": 40,
    "unit_ambiguity": 30,
    "multi_crop": 30,
    "conflicting_values": 20,
    "out_of_domain": 40,
    "code_switched": 40,
}
TOTAL_TARGET = 240
NO_GOLD_CATEGORIES = frozenset({"conflicting_values", "out_of_domain"})

BD = "০১২৩৪৫৬৭৮৯"
AREA_UNIT_BN = {"acre": "একর", "bigha": "বিঘা", "katha": "কাঠা",
                "decimal": "শতক", "hectare": "হেক্টর", "ha": "হেক্টর",
                "sqm": "বর্গমিটার"}
STAGES_EN = ("initial", "mid-season", "late-season")


def bn_num(n) -> str:
    """ASCII digits -> Bengali digits. Verbatim from s0/s2."""
    return "".join(BD[int(c)] if c.isdigit() else c for c in str(n))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def scenario_hash(crop, area_unit, area_value, et0, rain, stage) -> str:
    """Verbatim from s0_build_seed_corpus.py:84 — sorted key=value tuple."""
    parts = sorted([f"crop={crop}", f"area_unit={area_unit}",
                    f"area_value={area_value}", f"et0={et0}",
                    f"rain={rain}", f"stage={stage}"])
    return sha256_bytes("|".join(parts).encode("utf-8"))


class SeededIds:
    """uuid4-format ids from a stream that cannot collide with Stage 0's or
    Stage 2's — the tag is different, so the underlying Mersenne state is."""

    def __init__(self, tag: str):
        self._rng = random.Random(tag)

    def next(self) -> str:
        raw = bytes(self._rng.getrandbits(8) for _ in range(16))
        return str(uuid.UUID(bytes=raw, version=4))


# ==========================================================================
# SCENARIO SYNTHESIS
# ==========================================================================
AREA_POOL = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0,
             8.0, 10.0)
RAIN_TIERS = (((0.0, 5.0), 0.40), ((5.0, 15.0), 0.35), ((15.0, 45.0), 0.25))


class ScenarioFactory:
    """Draws (et0, rain, area, unit) tuples that compute to a strictly
    positive net irrigation depth and whose scenario_hash is unseen.

    Zero-net scenarios are excluded deliberately. A zero answer is a legitimate
    hard case, but combined with an adversarial question form it confounds
    attribution: a model that outputs 0 because it abstained looks identical to
    a model that outputs 0 because it computed correctly. The seed test set
    already contains zero-net items for that dimension.
    """

    def __init__(self, rng: random.Random, blocklist: set[str]):
        self.rng = rng
        self.blocklist = blocklist
        self.used: set[str] = set()
        self.stats = collections.Counter()

    def _draw_rain(self) -> float:
        u, acc = self.rng.random(), 0.0
        for (lo, hi), w in RAIN_TIERS:
            acc += w
            if u <= acc:
                return round(self.rng.uniform(lo, hi), 2)
        return round(self.rng.uniform(*RAIN_TIERS[-1][0]), 2)

    def draw(self, crop: str, stage: str, unit: str = "acre",
             area: float | None = None) -> dict:
        for _ in range(400):
            et0 = round(self.rng.uniform(2.5, 9.0), 1)
            rain = self._draw_rain()
            a = area if area is not None else \
                AREA_POOL[self.rng.randrange(len(AREA_POOL))]
            res = ic.compute(crop, stage, et0, rain, a, unit)
            if res["net_mm"] <= 0.0:
                self.stats["redrawn_zero_net"] += 1
                continue
            h = scenario_hash(crop, unit, a, et0, rain, stage)
            if h in self.blocklist:
                self.stats["redrawn_blocklisted_hash"] += 1
                continue
            if h in self.used:
                self.stats["redrawn_duplicate_hash"] += 1
                continue
            self.used.add(h)
            res["scenario_hash"] = h
            return res
        raise RuntimeError(
            f"could not draw a usable scenario for {crop}/{stage}/{unit} "
            f"in 400 attempts")


# ==========================================================================
# GOLD RENDERERS — every number arrives already computed by irrigation_calc
# ==========================================================================
def numeric_core_en(res: dict) -> str:
    """The seed corpus's exact numeric sentence (s2_augment_train.py:400).
    Reused verbatim so Stage 6 slot extraction sees one surface form, not two."""
    return (f"Net irrigation depth = Kc×ET0 − effective rainfall "
            f"= {res['kc_used']}×{res['et0_mm']} − {res['effective_rain_mm']} "
            f"= {res['net_mm']} mm. For {res['area_value']} {res['area_unit']} "
            f"({res['area_sqm']:.0f} m²), that is about "
            f"{res['liters']:.0f} liters today.")


def numeric_core_bn(res: dict) -> str:
    sqm = bn_num(format(res["area_sqm"], ".0f"))
    lit = bn_num(format(res["liters"], ".0f"))
    return (f"নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি "
            f"= {bn_num(res['kc_used'])}×{bn_num(res['et0_mm'])} − "
            f"{bn_num(res['effective_rain_mm'])} = {bn_num(res['net_mm'])} মিমি। "
            f"{bn_num(res['area_value'])} {AREA_UNIT_BN[res['area_unit']]} "
            f"({sqm} বর্গমিটার)-এর জন্য আজ প্রায় {lit} লিটার।")


def numeric_core(res: dict, lang: str) -> str:
    return numeric_core_en(res) if lang == "en" else numeric_core_bn(res)


def depth_only_en(res: dict) -> str:
    return (f"Net irrigation depth = Kc×ET0 − effective rainfall "
            f"= {res['kc_used']}×{res['et0_mm']} − {res['effective_rain_mm']} "
            f"= {res['net_mm']} mm.")


def depth_only_bn(res: dict) -> str:
    return (f"নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি "
            f"= {bn_num(res['kc_used'])}×{bn_num(res['et0_mm'])} − "
            f"{bn_num(res['effective_rain_mm'])} = {bn_num(res['net_mm'])} মিমি।")


def depth_only(res: dict, lang: str) -> str:
    return depth_only_en(res) if lang == "en" else depth_only_bn(res)


def q_numeric_en(crop, stage, res) -> str:
    return (f"My {crop} field is at the {stage} stage. Today ET0 is "
            f"{res['et0_mm']} mm, it rained {res['rainfall_mm']} mm, and the "
            f"field is {res['area_value']} {res['area_unit']}. How much "
            f"irrigation water does it need today?")


def q_numeric_bn(crop_bn, stage, res) -> str:
    return (f"আমার {crop_bn} ক্ষেত {ic.STAGE_BN[stage]} পর্যায়ে আছে। আজ ET0 "
            f"{bn_num(res['et0_mm'])} মিমি, বৃষ্টি হয়েছে "
            f"{bn_num(res['rainfall_mm'])} মিমি, জমির পরিমাণ "
            f"{bn_num(res['area_value'])} {AREA_UNIT_BN[res['area_unit']]}। "
            f"আজ কত পানি সেচ দিতে হবে?")


# ==========================================================================
# CATEGORY 1 — misspelled_crop (40)
# ==========================================================================
# Recoverable: the misspelling has exactly one plausible nearest crop in the
# 28-crop corpus, so the model should silently correct and answer.
# Every typo below is a real Bangladeshi-keyboard or transliteration error
# class: ণ/ন, ষ/স, ু/ূ, dropped chandrabindu, doubled consonant, dropped matra.
C1_RECOVERABLE_EN = [
    ("boro ric", "Boro Rice"), ("wheet", "Wheat"), ("potatoe", "Potato"),
    ("brinjel", "Brinjal"), ("sugercane", "Sugarcane"), ("tumeric", "Turmeric"),
    ("onoin", "Onion"), ("coconout", "Coconut"), ("gauva", "Guava"),
    ("sesamee", "Sesame"), ("mazie", "Maize"), ("jackfruite", "Jackfruit"),
]
C1_RECOVERABLE_BN = [
    ("বোরো ধন", "Boro Rice"),      # dropped matra: ধান -> ধন
    ("গমম", "Wheat"),               # doubled consonant
    ("আলূ", "Potato"),              # ু / ূ confusion
    ("বেগুণ", "Brinjal"),           # ন / ণ confusion
    ("আঁখ", "Sugarcane"),           # spurious chandrabindu
    ("হলুধ", "Turmeric"),           # দ / ধ confusion
    ("পিঁয়াজ", "Onion"),            # attested regional variant spelling
    ("নারকেল", "Coconut"),          # attested variant of নারিকেল
    ("পেয়ারাা", "Guava"),           # doubled aa-matra
    ("ভূট্টা", "Maize"),             # ু / ূ confusion
    ("কাঠাল", "Jackfruit"),         # dropped chandrabindu
    ("সরিসা", "Mustard"),           # স / ষ confusion
]

# Ambiguous: the surface form is genuinely equidistant between two or more
# corpus crops, so the only safe response is to ask which one.
# তিলা / তুল are the exact pair that the v3.0 normalizer collapsed (see
# bn_normalize.py version history) — included on purpose.
C1_AMBIGUOUS_EN = [
    ("ric", ["Aman Rice", "Aus Rice", "Boro Rice"]),
    ("paddy", ["Aman Rice", "Aus Rice", "Boro Rice"]),
    ("poteto", ["Potato", "Sweet Potato"]),
    ("groud", ["Bottle Gourd", "Groundnut"]),
    ("juube", ["Jute", "Jujube"]),
    ("munstard", ["Mungbean", "Mustard"]),
    ("pulse crop", ["Mungbean", "Pea"]),
    ("mustar bean", ["Mustard", "Mungbean"]),
]
C1_AMBIGUOUS_BN = [
    ("ধান", ["Aman Rice", "Aus Rice", "Boro Rice"]),
    ("তিলা", ["Sesame", "Cotton"]),
    ("তুল", ["Cotton", "Sesame"]),
    ("লেচু", ["Lemon", "Litchi"]),
    ("লিবু", ["Lemon", "Litchi"]),
    ("মুগকচু", ["Mungbean", "Mukhi Kachu"]),
    ("মিষ্ট আল", ["Potato", "Sweet Potato"]),
    ("ডাল", ["Mungbean", "Pea"]),
]


def c1_recoverable_gold(typo, crop, crop_bn, res, lang) -> str:
    if lang == "en":
        return f"Taking \"{typo}\" as {crop}. {numeric_core_en(res)}"
    return f"\"{typo}\" বলতে {crop_bn} ধরে নিচ্ছি। {numeric_core_bn(res)}"


def c1_ambiguous_gold(typo, options, options_bn, lang) -> str:
    if lang == "en":
        opts = ", ".join(options[:-1]) + " or " + options[-1]
        return (f"I am not sure which crop \"{typo}\" refers to — it could be "
                f"{opts}. These need different Kc values, so the irrigation "
                f"amount would differ. Please confirm the crop and I will "
                f"compute it.")
    opts = ", ".join(options_bn[:-1]) + " অথবা " + options_bn[-1]
    return (f"\"{typo}\" বলতে আপনি কোন ফসল বোঝাচ্ছেন তা আমি নিশ্চিত নই — এটি "
            f"{opts} হতে পারে। এদের Kc আলাদা, তাই সেচের পরিমাণও আলাদা হবে। "
            f"ফসলটি নিশ্চিত করলে আমি হিসাব করে দিতে পারব।")


# ==========================================================================
# CATEGORY 2 — missing_parameter (40)
# ==========================================================================
# One required slot is removed from an otherwise well-formed numeric question.
# The gold never guesses the missing value. Where a partial result is still
# derivable (area missing -> depth is computable), the gold gives that partial
# result and then asks — that is the discriminating behaviour this category is
# for, and it makes partial credit measurable in Stage 6.
C2_SUBTYPES = ("missing_et0", "missing_rainfall", "missing_area", "missing_stage")


def c2_question(sub, crop, crop_bn, stage, res, lang) -> str:
    if lang == "en":
        if sub == "missing_et0":
            return (f"My {crop} field is at the {stage} stage. It rained "
                    f"{res['rainfall_mm']} mm today and the field is "
                    f"{res['area_value']} {res['area_unit']}. How much "
                    f"irrigation water does it need today?")
        if sub == "missing_rainfall":
            return (f"My {crop} field is at the {stage} stage. Today ET0 is "
                    f"{res['et0_mm']} mm and the field is {res['area_value']} "
                    f"{res['area_unit']}. How much irrigation water does it "
                    f"need today?")
        if sub == "missing_area":
            return (f"My {crop} field is at the {stage} stage. Today ET0 is "
                    f"{res['et0_mm']} mm and it rained {res['rainfall_mm']} mm. "
                    f"How many liters of irrigation water does it need today?")
        return (f"My {crop} field needs water. Today ET0 is {res['et0_mm']} mm, "
                f"it rained {res['rainfall_mm']} mm, and the field is "
                f"{res['area_value']} {res['area_unit']}. How much irrigation "
                f"water does it need today?")
    sbn, ubn = ic.STAGE_BN[stage], AREA_UNIT_BN[res["area_unit"]]
    if sub == "missing_et0":
        return (f"আমার {crop_bn} ক্ষেত {sbn} পর্যায়ে আছে। আজ বৃষ্টি হয়েছে "
                f"{bn_num(res['rainfall_mm'])} মিমি, জমির পরিমাণ "
                f"{bn_num(res['area_value'])} {ubn}। আজ কত পানি সেচ দিতে হবে?")
    if sub == "missing_rainfall":
        return (f"আমার {crop_bn} ক্ষেত {sbn} পর্যায়ে আছে। আজ ET0 "
                f"{bn_num(res['et0_mm'])} মিমি, জমির পরিমাণ "
                f"{bn_num(res['area_value'])} {ubn}। আজ কত পানি সেচ দিতে হবে?")
    if sub == "missing_area":
        return (f"আমার {crop_bn} ক্ষেত {sbn} পর্যায়ে আছে। আজ ET0 "
                f"{bn_num(res['et0_mm'])} মিমি, বৃষ্টি হয়েছে "
                f"{bn_num(res['rainfall_mm'])} মিমি। আজ কত লিটার পানি সেচ "
                f"দিতে হবে?")
    return (f"আমার {crop_bn} ক্ষেতে পানি দিতে হবে। আজ ET0 "
            f"{bn_num(res['et0_mm'])} মিমি, বৃষ্টি হয়েছে "
            f"{bn_num(res['rainfall_mm'])} মিমি, জমির পরিমাণ "
            f"{bn_num(res['area_value'])} {ubn}। আজ কত পানি সেচ দিতে হবে?")


def c2_gold(sub, crop, crop_bn, stage, res, kc_row, lang) -> str:
    kc = res["kc_used"]
    if lang == "en":
        if sub == "missing_et0":
            return (f"I cannot compute this without today's reference "
                    f"evapotranspiration (ET0). {crop} at the {stage} stage "
                    f"uses Kc = {kc}, and net irrigation depth = Kc×ET0 − "
                    f"effective rainfall, so ET0 in mm is required. Please "
                    f"tell me today's ET0.")
        if sub == "missing_rainfall":
            return (f"I cannot compute this without today's rainfall. Net "
                    f"irrigation depth = Kc×ET0 − effective rainfall, and "
                    f"effective rainfall depends on how much rain fell "
                    f"(up to 5 mm counts 85%, up to 15 mm counts 45%, above "
                    f"that 25%). {crop} at the {stage} stage uses Kc = {kc}. "
                    f"Please tell me today's rainfall in mm.")
        if sub == "missing_area":
            return (f"{depth_only_en(res)} I cannot convert that to liters "
                    f"without the field size. Please tell me the area and its "
                    f"unit (acre, bigha, katha, decimal or hectare).")
        return (f"I cannot choose the Kc value without the growth stage. "
                f"{crop} uses Kc = {kc_row['kc_ini']} at the initial stage, "
                f"{kc_row['kc_mid']} at mid-season and {kc_row['kc_end']} at "
                f"late-season, and each gives a different amount. Please tell "
                f"me which stage the field is at.")
    sbn = ic.STAGE_BN[stage]
    if sub == "missing_et0":
        return (f"আজকের রেফারেন্স বাষ্পীভবন (ET0) না জানলে হিসাব করা সম্ভব নয়। "
                f"{crop_bn} ফসলের {sbn} পর্যায়ে Kc = {bn_num(kc)}, এবং নিট "
                f"সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি — তাই ET0 (মিমি) "
                f"প্রয়োজন। অনুগ্রহ করে আজকের ET0 জানান।")
    if sub == "missing_rainfall":
        return (f"আজকের বৃষ্টিপাত না জানলে হিসাব করা সম্ভব নয়। নিট সেচের "
                f"গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি, আর কার্যকর বৃষ্টি নির্ভর "
                f"করে কত মিমি বৃষ্টি হয়েছে তার উপর (৫ মিমি পর্যন্ত ৮৫%, ১৫ মিমি "
                f"পর্যন্ত ৪৫%, তার বেশি হলে ২৫%)। {crop_bn} ফসলের {sbn} পর্যায়ে "
                f"Kc = {bn_num(kc)}। অনুগ্রহ করে আজকের বৃষ্টিপাত মিমিতে জানান।")
    if sub == "missing_area":
        return (f"{depth_only_bn(res)} তবে জমির পরিমাণ না জানলে এটি লিটারে "
                f"রূপান্তর করা সম্ভব নয়। অনুগ্রহ করে জমির পরিমাণ ও একক (একর, "
                f"বিঘা, কাঠা, শতক বা হেক্টর) জানান।")
    return (f"ফসলের বৃদ্ধির পর্যায় না জানলে Kc বাছাই করা সম্ভব নয়। {crop_bn} "
            f"ফসলে প্রাথমিক পর্যায়ে Kc = {bn_num(kc_row['kc_ini'])}, মধ্য-মৌসুমে "
            f"{bn_num(kc_row['kc_mid'])} এবং শেষ-মৌসুমে "
            f"{bn_num(kc_row['kc_end'])} — প্রতিটিতে উত্তর আলাদা হবে। অনুগ্রহ "
            f"করে ক্ষেত কোন পর্যায়ে আছে জানান।")


# ==========================================================================
# CATEGORY 3 — unit_ambiguity (30)
# ==========================================================================
# 3a  unknown_local_unit — a real Bangladeshi land unit that is not in
#     irrigation_calc.AREA and whose size varies by district.
# 3b  bigha_regional — plain "bigha" with no region given. The calculator's
#     1337.8 m² is the 33-decimal standard; Rajshahi/Sylhet/Chittagong bighas
#     differ. Gold answers with the standard AND names the assumption.
# 3c  explicit_local_conversion — the farmer supplies their own bigha size, so
#     the gold must use the farmer's number, not the table's.
C3_LOCAL_UNITS = [
    ("kani", "কানি"), ("gonda", "গণ্ডা"), ("pakhi", "পাখি"),
    ("hal", "হাল"), ("dron", "দ্রোণ"), ("kura", "কুড়া"),
]
C3_LOCAL_CONVERSIONS = (0.30, 0.33, 0.36, 0.40, 0.45, 0.50)  # acre per bigha


def c3a_gold(unit_en, unit_bn, res, lang) -> str:
    if lang == "en":
        return (f"\"{unit_en}\" is a local land measure whose size differs from "
                f"district to district, and it is not in my conversion table. "
                f"I can convert acre, bigha (33 decimal = 1337.8 m²), katha, "
                f"decimal and hectare. {depth_only_en(res)} Tell me the area in "
                f"one of those units, or how many decimals one {unit_en} is in "
                f"your area, and I will give the volume in liters.")
    return (f"\"{unit_bn}\" একটি স্থানীয় জমির একক, যার মাপ এলাকাভেদে বদলায় এবং "
            f"এটি আমার রূপান্তর তালিকায় নেই। আমি একর, বিঘা (৩৩ শতক = "
            f"১৩৩৭.৮ বর্গমিটার), কাঠা, শতক ও হেক্টর রূপান্তর করতে পারি। "
            f"{depth_only_bn(res)} আপনার এলাকায় এক {unit_bn} কত শতক তা জানালে, "
            f"অথবা এই এককগুলোর একটিতে জমির পরিমাণ দিলে, আমি লিটারে হিসাব দিতে "
            f"পারব।")


def c3b_gold(res, lang) -> str:
    if lang == "en":
        return (f"{numeric_core_en(res)} I have used the standard Bangladeshi "
                f"bigha of 33 decimal (1337.8 m²). Bigha size varies by region "
                f"— if a bigha in your area is a different size, tell me and I "
                f"will recompute the volume.")
    return (f"{numeric_core_bn(res)} এখানে আমি প্রমাণ বাংলাদেশি বিঘা (৩৩ শতক = "
            f"১৩৩৭.৮ বর্গমিটার) ধরেছি। বিঘার মাপ এলাকাভেদে ভিন্ন হয় — আপনার "
            f"এলাকায় বিঘার মাপ আলাদা হলে জানালে আমি আবার হিসাব করে দেব।")


def c3c_gold(res, n_bigha, acre_per_bigha, sqm, liters, lang) -> str:
    if lang == "en":
        return (f"Using your local conversion of 1 bigha = {acre_per_bigha} "
                f"acre, {n_bigha} bigha is {sqm:.0f} m². "
                f"{depth_only_en(res)} For {sqm:.0f} m² that is about "
                f"{liters:.0f} liters today.")
    return (f"আপনার এলাকার হিসাবে ১ বিঘা = {bn_num(acre_per_bigha)} একর ধরলে "
            f"{bn_num(n_bigha)} বিঘা = {bn_num(format(sqm, '.0f'))} বর্গমিটার। "
            f"{depth_only_bn(res)} {bn_num(format(sqm, '.0f'))} বর্গমিটারের জন্য "
            f"আজ প্রায় {bn_num(format(liters, '.0f'))} লিটার।")


# ==========================================================================
# CATEGORY 4 — multi_crop (30)
# ==========================================================================
def c4_dual_gold(c1, c1b, r1, c2, c2b, r2, lang) -> str:
    if lang == "en":
        return (f"{c1} ({r1['stage']}): Kc = {r1['kc_used']}, net irrigation "
                f"depth = {r1['kc_used']}×{r1['et0_mm']} − "
                f"{r1['effective_rain_mm']} = {r1['net_mm']} mm, so about "
                f"{r1['liters']:.0f} liters for {r1['area_value']} "
                f"{r1['area_unit']}. {c2} ({r2['stage']}): Kc = "
                f"{r2['kc_used']}, net irrigation depth = "
                f"{r2['kc_used']}×{r2['et0_mm']} − {r2['effective_rain_mm']} = "
                f"{r2['net_mm']} mm, so about {r2['liters']:.0f} liters for "
                f"{r2['area_value']} {r2['area_unit']}.")
    return (f"{c1b} ({ic.STAGE_BN[r1['stage']]}): Kc = {bn_num(r1['kc_used'])}, "
            f"নিট সেচের গভীরতা = {bn_num(r1['kc_used'])}×{bn_num(r1['et0_mm'])} "
            f"− {bn_num(r1['effective_rain_mm'])} = {bn_num(r1['net_mm'])} মিমি, "
            f"অর্থাৎ {bn_num(r1['area_value'])} "
            f"{AREA_UNIT_BN[r1['area_unit']]} জমির জন্য প্রায় "
            f"{bn_num(format(r1['liters'], '.0f'))} লিটার। "
            f"{c2b} ({ic.STAGE_BN[r2['stage']]}): Kc = {bn_num(r2['kc_used'])}, "
            f"নিট সেচের গভীরতা = {bn_num(r2['kc_used'])}×{bn_num(r2['et0_mm'])} "
            f"− {bn_num(r2['effective_rain_mm'])} = {bn_num(r2['net_mm'])} মিমি, "
            f"অর্থাৎ {bn_num(r2['area_value'])} "
            f"{AREA_UNIT_BN[r2['area_unit']]} জমির জন্য প্রায় "
            f"{bn_num(format(r2['liters'], '.0f'))} লিটার।")


def c4_comparative_gold(c1, c1b, r1, c2, c2b, r2, lang) -> str:
    d = round(abs(r1["net_mm"] - r2["net_mm"]), 2)
    if r1["net_mm"] > r2["net_mm"]:
        win_en, win_bn = c1, c1b
    elif r2["net_mm"] > r1["net_mm"]:
        win_en, win_bn = c2, c2b
    else:
        win_en = win_bn = None
    if lang == "en":
        lines = (f"{c1} ({r1['stage']}): Kc = {r1['kc_used']}, net = "
                 f"{r1['net_mm']} mm. {c2} ({r2['stage']}): Kc = "
                 f"{r2['kc_used']}, net = {r2['net_mm']} mm.")
        if win_en is None:
            return (f"Both need the same depth today, {r1['net_mm']} mm. {lines}")
        return (f"{win_en} needs more water today. {lines} The difference is "
                f"{d} mm.")
    lines = (f"{c1b} ({ic.STAGE_BN[r1['stage']]}): Kc = "
             f"{bn_num(r1['kc_used'])}, নিট = {bn_num(r1['net_mm'])} মিমি। "
             f"{c2b} ({ic.STAGE_BN[r2['stage']]}): Kc = "
             f"{bn_num(r2['kc_used'])}, নিট = {bn_num(r2['net_mm'])} মিমি।")
    if win_bn is None:
        return (f"আজ দুটিতেই একই গভীরতা লাগবে, {bn_num(r1['net_mm'])} মিমি। "
                f"{lines}")
    return (f"আজ {win_bn} ফসলে বেশি পানি লাগবে। {lines} পার্থক্য "
            f"{bn_num(d)} মিমি।")


def c4_split_gold(c1, c1b, r1, c2, c2b, r2, lang) -> str:
    total = round(r1["liters"] + r2["liters"], 1)
    if lang == "en":
        return (f"{c1} ({r1['area_value']} {r1['area_unit']}, {r1['stage']}): "
                f"net = {r1['kc_used']}×{r1['et0_mm']} − "
                f"{r1['effective_rain_mm']} = {r1['net_mm']} mm → about "
                f"{r1['liters']:.0f} liters. {c2} ({r2['area_value']} "
                f"{r2['area_unit']}, {r2['stage']}): net = "
                f"{r2['kc_used']}×{r2['et0_mm']} − {r2['effective_rain_mm']} = "
                f"{r2['net_mm']} mm → about {r2['liters']:.0f} liters. Total "
                f"for the whole holding: about {total:.0f} liters today.")
    return (f"{c1b} ({bn_num(r1['area_value'])} "
            f"{AREA_UNIT_BN[r1['area_unit']]}, {ic.STAGE_BN[r1['stage']]}): "
            f"নিট = {bn_num(r1['kc_used'])}×{bn_num(r1['et0_mm'])} − "
            f"{bn_num(r1['effective_rain_mm'])} = {bn_num(r1['net_mm'])} মিমি "
            f"→ প্রায় {bn_num(format(r1['liters'], '.0f'))} লিটার। "
            f"{c2b} ({bn_num(r2['area_value'])} "
            f"{AREA_UNIT_BN[r2['area_unit']]}, {ic.STAGE_BN[r2['stage']]}): "
            f"নিট = {bn_num(r2['kc_used'])}×{bn_num(r2['et0_mm'])} − "
            f"{bn_num(r2['effective_rain_mm'])} = {bn_num(r2['net_mm'])} মিমি "
            f"→ প্রায় {bn_num(format(r2['liters'], '.0f'))} লিটার। পুরো জমির "
            f"জন্য আজ মোট প্রায় {bn_num(format(total, '.0f'))} লিটার।")


# ==========================================================================
# CATEGORY 5 — conflicting_values (20)   *** NO GOLD — HUMAN AUTHORS ***
# ==========================================================================
C5_CONFLICTS = ("rainfall_restated", "et0_restated", "area_unit_switched",
                "stage_contradicted", "crop_contradicted")


def c5_question(sub, crop, crop_bn, stage, res, alt, lang) -> str:
    """Well-formed except that one slot is asserted twice with different
    values. No gold is produced — see the module docstring."""
    u = res["area_unit"]
    ubn = AREA_UNIT_BN[u]
    sbn = ic.STAGE_BN[stage]
    if lang == "en":
        if sub == "rainfall_restated":
            return (f"My {crop} field is at the {stage} stage. Today ET0 is "
                    f"{res['et0_mm']} mm and it rained {alt['rain_a']} mm — no "
                    f"wait, the gauge says {alt['rain_b']} mm. The field is "
                    f"{res['area_value']} {u}. How much water does it need "
                    f"today?")
        if sub == "et0_restated":
            return (f"My {crop} field is at the {stage} stage. ET0 today is "
                    f"{alt['et0_a']} mm according to the block office, though "
                    f"my own reading was {alt['et0_b']} mm. It rained "
                    f"{res['rainfall_mm']} mm and the field is "
                    f"{res['area_value']} {u}. How much water does it need?")
        if sub == "area_unit_switched":
            return (f"My {crop} field is at the {stage} stage, ET0 "
                    f"{res['et0_mm']} mm, rainfall {res['rainfall_mm']} mm. "
                    f"The field is {res['area_value']} acre — actually I mean "
                    f"{res['area_value']} bigha. How much water does it need "
                    f"today?")
        if sub == "stage_contradicted":
            return (f"My {crop} field is at the {stage} stage, but we only "
                    f"transplanted it last week so it is still at the initial "
                    f"stage. ET0 is {res['et0_mm']} mm, rainfall "
                    f"{res['rainfall_mm']} mm, area {res['area_value']} {u}. "
                    f"How much water does it need today?")
        return (f"How much water does my {crop} field need today? It is "
                f"{alt['other_crop']} actually, at the {stage} stage, ET0 "
                f"{res['et0_mm']} mm, rainfall {res['rainfall_mm']} mm, area "
                f"{res['area_value']} {u}.")
    if sub == "rainfall_restated":
        return (f"আমার {crop_bn} ক্ষেত {sbn} পর্যায়ে আছে। আজ ET0 "
                f"{bn_num(res['et0_mm'])} মিমি, বৃষ্টি হয়েছে "
                f"{bn_num(alt['rain_a'])} মিমি — না না, মাপযন্ত্রে "
                f"{bn_num(alt['rain_b'])} মিমি দেখাচ্ছে। জমির পরিমাণ "
                f"{bn_num(res['area_value'])} {ubn}। আজ কত পানি সেচ দিতে হবে?")
    if sub == "et0_restated":
        return (f"আমার {crop_bn} ক্ষেত {sbn} পর্যায়ে আছে। ব্লক অফিস বলছে আজ "
                f"ET0 {bn_num(alt['et0_a'])} মিমি, তবে আমার নিজের হিসাবে "
                f"{bn_num(alt['et0_b'])} মিমি। বৃষ্টি হয়েছে "
                f"{bn_num(res['rainfall_mm'])} মিমি, জমি "
                f"{bn_num(res['area_value'])} {ubn}। কত পানি লাগবে?")
    if sub == "area_unit_switched":
        return (f"আমার {crop_bn} ক্ষেত {sbn} পর্যায়ে, ET0 "
                f"{bn_num(res['et0_mm'])} মিমি, বৃষ্টি "
                f"{bn_num(res['rainfall_mm'])} মিমি। জমি "
                f"{bn_num(res['area_value'])} একর — না, আসলে "
                f"{bn_num(res['area_value'])} বিঘা। আজ কত পানি সেচ দিতে হবে?")
    if sub == "stage_contradicted":
        return (f"আমার {crop_bn} ক্ষেত {sbn} পর্যায়ে আছে, তবে গত সপ্তাহেই "
                f"রোপণ করেছি তাই এখনও প্রাথমিক পর্যায়ে। ET0 "
                f"{bn_num(res['et0_mm'])} মিমি, বৃষ্টি "
                f"{bn_num(res['rainfall_mm'])} মিমি, জমি "
                f"{bn_num(res['area_value'])} {ubn}। আজ কত পানি সেচ দিতে হবে?")
    return (f"আমার {crop_bn} ক্ষেতে আজ কত পানি লাগবে? আসলে ওটা "
            f"{alt['other_crop_bn']}, {sbn} পর্যায়ে, ET0 "
            f"{bn_num(res['et0_mm'])} মিমি, বৃষ্টি "
            f"{bn_num(res['rainfall_mm'])} মিমি, জমি "
            f"{bn_num(res['area_value'])} {ubn}।")


# ==========================================================================
# CATEGORY 6 — out_of_domain (40)   *** NO GOLD — HUMAN AUTHORS ***
# ==========================================================================
# Ten fresh topics. The seed corpus's three abstain topics (market_price,
# pesticide_brand, soil_ph) are deliberately NOT reused: a model fine-tuned on
# them would pattern-match the topic rather than judge the scope boundary,
# which is the thing this category is supposed to measure.

# Perennial fruit trees. "Which dealer sells certified Mango seed?" is not a
# question a Bangladeshi farmer asks — these are grafted, not sown — so the
# seed_dealer topic skips them. The item must be out of domain, not absurd:
# an absurd question can be refused for the wrong reason.
C6_TREE_CROPS = frozenset({"Coconut", "Guava", "Mango", "Litchi", "Jackfruit",
                           "Jujube", "Lemon"})

C6_TOPICS = [
    ("fertilizer_brand",
     "Which brand of urea should I buy for my {crop} field?",
     "আমার {crop_bn} ক্ষেতের জন্য কোন কোম্পানির ইউরিয়া কিনব?"),
    ("seed_dealer",
     "Which dealer in my upazila sells certified {crop} seed?",
     "আমার উপজেলায় কোন ডিলারের কাছে সার্টিফাইড {crop_bn} বীজ পাওয়া যায়?"),
    ("farm_loan",
     "What interest rate will the bank charge me for a {crop} cultivation loan?",
     "{crop_bn} চাষের ঋণে ব্যাংক আমার কাছ থেকে কত সুদ নেবে?"),
    ("weather_forecast",
     "Will it rain on my {crop} field tomorrow?",
     "আগামীকাল কি আমার {crop_bn} ক্ষেতে বৃষ্টি হবে?"),
    ("machinery_rental",
     "How much does it cost to rent a power tiller for a {crop} field?",
     "{crop_bn} ক্ষেতের জন্য পাওয়ার টিলার ভাড়া নিতে কত টাকা লাগে?"),
    ("land_registration",
     "What documents do I need to register the land where I grow {crop}?",
     "যে জমিতে {crop_bn} চাষ করি সেটি রেজিস্ট্রি করতে কী কী কাগজ লাগবে?"),
    ("crop_insurance",
     "Can I claim insurance if my {crop} crop fails this season?",
     "এ মৌসুমে {crop_bn} ফসল নষ্ট হলে আমি কি বিমার টাকা দাবি করতে পারব?"),
    ("disease_diagnosis",
     "The leaves of my {crop} plants have brown spots — what disease is it?",
     "আমার {crop_bn} গাছের পাতায় বাদামি দাগ পড়েছে — এটা কী রোগ?"),
    ("export_policy",
     "Can I export {crop} to India, and what permit is required?",
     "আমি কি {crop_bn} ভারতে রপ্তানি করতে পারব, কী অনুমতি লাগবে?"),
    ("labour_wage",
     "What is the daily wage for a labourer harvesting {crop} this year?",
     "এ বছর {crop_bn} কাটার শ্রমিকের দৈনিক মজুরি কত?"),
]


# ==========================================================================
# CATEGORY 7 — code_switched (40)
# ==========================================================================
# Matrix language = the language of the sentence frame and of the gold answer.
# Embedded language supplies crop name, units, or the numerals. Digit systems
# are mixed on purpose: bn_normalize maps ০-৯ to 0-9, so a model that only ever
# saw one digit system in training is tested here.
def c7_question_en_matrix(crop, crop_bn, stage, res, variant) -> str:
    ubn = AREA_UNIT_BN[res["area_unit"]]
    if variant == 0:
        return (f"My {crop_bn} field is at the {stage} stage. Today ET0 is "
                f"{res['et0_mm']} mm, বৃষ্টি হয়েছে {res['rainfall_mm']} mm, and "
                f"the জমি is {res['area_value']} {ubn}. How much পানি does it "
                f"need today?")
    if variant == 1:
        return (f"Amar {crop} khet is at {stage} stage. ET0 {res['et0_mm']} "
                f"mm, rain {bn_num(res['rainfall_mm'])} মিমি, area "
                f"{res['area_value']} {ubn}. Koto water lagbe today?")
    if variant == 2:
        return (f"My {crop} ক্ষেত is in the {ic.STAGE_BN[stage]} পর্যায়. ET0 = "
                f"{res['et0_mm']} mm, rainfall = {res['rainfall_mm']} mm, "
                f"জমির পরিমাণ {res['area_value']} {res['area_unit']}. How much "
                f"irrigation is needed?")
    return (f"Bhai, my {crop_bn} ({crop}) field, {stage} stage — ET0 "
            f"{bn_num(res['et0_mm'])} মিমি, rain {res['rainfall_mm']} mm, "
            f"{res['area_value']} {res['area_unit']} জমি. Koto liter সেচ "
            f"dite hobe?")


def c7_question_bn_matrix(crop, crop_bn, stage, res, variant) -> str:
    sbn = ic.STAGE_BN[stage]
    ubn = AREA_UNIT_BN[res["area_unit"]]
    if variant == 0:
        return (f"আমার {crop} ক্ষেত {sbn} পর্যায়ে আছে। আজ ET0 "
                f"{res['et0_mm']} mm, rain হয়েছে {bn_num(res['rainfall_mm'])} "
                f"মিমি, জমির পরিমাণ {res['area_value']} {res['area_unit']}। আজ "
                f"কত water সেচ দিতে হবে?")
    if variant == 1:
        return (f"আমার {crop_bn} field {stage} stage-এ আছে। ET0 "
                f"{bn_num(res['et0_mm'])} মিমি, rainfall "
                f"{res['rainfall_mm']} mm, area {bn_num(res['area_value'])} "
                f"{ubn}। কত irrigation লাগবে আজ?")
    if variant == 2:
        return (f"{crop} ক্ষেতের {stage} stage চলছে। today ET0 "
                f"{res['et0_mm']} mm আর rain {res['rainfall_mm']} mm। জমি "
                f"{bn_num(res['area_value'])} {ubn}। কত লিটার পানি দিতে হবে?")
    return (f"ভাই, আমার {crop_bn} ({crop}) ক্ষেত {sbn} পর্যায়ে। ET0 = "
            f"{res['et0_mm']} mm, বৃষ্টি = {bn_num(res['rainfall_mm'])} মিমি, "
            f"area = {res['area_value']} {res['area_unit']}। আজকের water "
            f"requirement কত?")


# ==========================================================================
# BUILD
# ==========================================================================
def load_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class RowBuilder:
    """Assembles one adversarial row with a uniform schema and a per-category
    running item_id."""

    def __init__(self, ids: SeededIds):
        self.ids = ids
        self.counters = collections.Counter()
        self.cat_index = {c: i + 1 for i, c in enumerate(CATEGORY_TARGETS)}
        self.rows: list[dict] = []

    def add(self, category, subtype, lang, qtype, question, answer,
            answer_kind, gold_status, renderer, *, crop=None, crop_bn=None,
            crops=None, scenario=None, scenarios=None, gold_trace=None,
            scenario_hashes=None, notes="", extra_prov=None) -> dict:
        self.counters[category] += 1
        n = self.counters[category]
        item_id = f"ADV-C{self.cat_index[category]}-{n:03d}"

        prov = {
            "extraction_method": "authored_adversarial",
            "generator": "s4_harden_test.py",
            "generator_version": GENERATOR_VERSION,
            "seed": SEED,
            "category": category,
            "subtype": subtype,
            "renderer": renderer,
            "gold_status": gold_status,
            "source_image": None,
            "source_image_sha256": None,
            "field_key": None,
            "field_provenance": "authored_stage4",
            "field_source_section": None,
            "source_quote_bn": None,
            "human_verified_upstream": False,
            "verified_by": None,
            "notes": notes,
        }
        if extra_prov:
            prov.update(extra_prov)

        row = {
            "row_id": self.ids.next(),
            "item_id": item_id,
            "source_id": f"adversarial#{item_id}",
            "fact_id": f"adv_{category}_{subtype}_{n:03d}",
            "scenario_hash": (scenario_hashes[0]
                              if scenario_hashes and len(scenario_hashes) == 1
                              else None),
            "scenario_hashes": scenario_hashes or [],
            "question": question,
            "answer": answer,
            "lang": lang,
            "qtype": qtype,
            "crop": crop,
            "crop_bn": crop_bn,
            "crops": crops or ([crop] if crop else []),
            "origin": "adversarial",
            "category": category,
            "subtype": subtype,
            "gold_status": gold_status,
            "answer_kind": answer_kind,
            "provenance": prov,
        }
        if scenario is not None:
            row["scenario"] = scenario
        if scenarios is not None:
            row["scenarios"] = scenarios
        if gold_trace is not None:
            row["gold_trace"] = gold_trace
        self.rows.append(row)
        return row


def trace_of(res: dict) -> dict:
    """gold_trace = the calculator's own dict minus the transient hash key."""
    return {k: v for k, v in res.items() if k != "scenario_hash"}


def scen_of(res: dict) -> dict:
    return {"et0_mm": res["et0_mm"], "rainfall_mm": res["rainfall_mm"],
            "area_value": res["area_value"], "area_unit": res["area_unit"],
            "stage": res["stage"], "crop": res["crop"]}


def build() -> dict:
    started = dt.datetime.now(dt.timezone.utc)

    # ---- 0. normalizer regression gate --------------------------------
    fails = bnorm.run_regression_tests()
    if fails:
        raise SystemExit(f"normalizer regression FAILED: {fails}")

    # ---- 1. load and verify inputs ------------------------------------
    seed_rows = load_jsonl(TEST_SEED_PATH)
    if len(seed_rows) != 182:
        raise SystemExit(f"expected 182 seed test rows, got {len(seed_rows)}")
    test_sha = sha256_file(TEST_SEED_PATH)
    stored_sha = TEST_SEED_SHA.read_text(encoding="utf-8").split()[0].strip()
    if test_sha != stored_sha:
        raise SystemExit(
            f"test.jsonl sha256 mismatch: file={test_sha} stored={stored_sha}. "
            f"The locked test split has been modified — stop and investigate.")

    train_seed_rows = load_jsonl(TRAIN_SEED_PATH)
    train_aug_rows = load_jsonl(TRAIN_AUG_PATH)

    blocklist = set(json.loads(BLOCKLIST_PATH.read_text(encoding="utf-8"))
                    ["scenario_hashes"])
    n_heldout = len(blocklist)
    train_hashes = {r["scenario_hash"] for r in train_aug_rows
                    if r.get("scenario_hash")}
    blocklist |= train_hashes

    # crop surface forms, taken from the corpus rather than re-typed
    crop_bn_map: dict[str, str] = {}
    for r in seed_rows + train_seed_rows:
        if r.get("crop") and r.get("crop_bn"):
            crop_bn_map.setdefault(r["crop"], r["crop_bn"])
    crops = sorted(crop_bn_map)
    kc_missing = [c for c in crops
                  if ic.normalize_crop_name(c) not in ic.KC_TABLE]
    if kc_missing:
        raise SystemExit(f"crops with no Kc row: {kc_missing}")

    rng = random.Random(f"s4-{SEED}")
    factory = ScenarioFactory(rng, blocklist)
    rb = RowBuilder(SeededIds(f"s4-ids-{SEED}"))

    def kc_row_for(crop):
        return ic.KC_TABLE[ic.normalize_crop_name(crop)]

    def pick_stage():
        return STAGES_EN[rng.randrange(3)]

    CLARIFY_NOTE = ("Gold is a clarification-request policy answer, not a "
                    "retrieved fact. It names the missing/ambiguous input and "
                    "supplies every value that IS derivable.")

    # ================= CATEGORY 1 — misspelled_crop (40) =================
    for lang, table in (("en", C1_RECOVERABLE_EN), ("bn", C1_RECOVERABLE_BN)):
        for typo, crop in table:
            crop_bn = crop_bn_map[crop]
            stage = pick_stage()
            res = factory.draw(crop, stage)
            if lang == "en":
                q = (f"My {typo} field is at the {stage} stage. Today ET0 is "
                     f"{res['et0_mm']} mm, it rained {res['rainfall_mm']} mm, "
                     f"and the field is {res['area_value']} "
                     f"{res['area_unit']}. How much irrigation water does it "
                     f"need today?")
            else:
                q = (f"আমার {typo} ক্ষেত {ic.STAGE_BN[stage]} পর্যায়ে আছে। আজ "
                     f"ET0 {bn_num(res['et0_mm'])} মিমি, বৃষ্টি হয়েছে "
                     f"{bn_num(res['rainfall_mm'])} মিমি, জমির পরিমাণ "
                     f"{bn_num(res['area_value'])} "
                     f"{AREA_UNIT_BN[res['area_unit']]}। আজ কত পানি সেচ দিতে "
                     f"হবে?")
            rb.add("misspelled_crop", "recoverable", lang, "numeric_runtime",
                   q, c1_recoverable_gold(typo, crop, crop_bn, res, lang),
                   "computed", "computed", "irrigation_calc.compute",
                   crop=crop, crop_bn=crop_bn, scenario=scen_of(res),
                   gold_trace=trace_of(res),
                   scenario_hashes=[res["scenario_hash"]],
                   notes=(f"Misspelling '{typo}' has a unique nearest crop "
                          f"({crop}); model should correct and answer."),
                   extra_prov={"surface_typo": typo, "resolved_crop": crop,
                               "kc_source": res["kc_source"],
                               "kc_citation": res["citation"]})

    for lang, table in (("en", C1_AMBIGUOUS_EN), ("bn", C1_AMBIGUOUS_BN)):
        for typo, options in table:
            options_bn = [crop_bn_map[c] for c in options]
            # Every slot except the crop is supplied, so the crop ambiguity is
            # the single reason the question cannot be answered. Leaving the
            # area out too would make a "please give the area" response look
            # correct for the wrong reason.
            e0 = round(rng.uniform(3.0, 8.5), 1)
            rn = round(rng.uniform(0.0, 12.0), 1)
            av = AREA_POOL[rng.randrange(len(AREA_POOL))]
            stg = pick_stage()
            if lang == "en":
                q = (f"My {typo} field is at the {stg} stage and needs "
                     f"irrigation today. ET0 is {e0} mm, it rained {rn} mm, "
                     f"and the field is {av} acre. How much water should I "
                     f"give?")
            else:
                q = (f"আমার {typo} ক্ষেত {ic.STAGE_BN[stg]} পর্যায়ে আছে, আজ সেচ "
                     f"দিতে হবে। ET0 {bn_num(e0)} মিমি, বৃষ্টি হয়েছে "
                     f"{bn_num(rn)} মিমি, জমির পরিমাণ {bn_num(av)} একর। কত "
                     f"পানি দেব?")
            rb.add("misspelled_crop", "ambiguous", lang, "clarify", q,
                   c1_ambiguous_gold(typo, options, options_bn, lang),
                   "clarification", "authored_policy", "clarify_policy",
                   crop=None, crop_bn=None, crops=options,
                   notes=(f"'{typo}' is equidistant between "
                          f"{', '.join(options)}; the only safe response is to "
                          f"ask which crop. Answering with any single crop is "
                          f"a failure."),
                   extra_prov={"surface_typo": typo,
                               "candidate_crops": options})

    # ================= CATEGORY 2 — missing_parameter (40) ===============
    for sub in C2_SUBTYPES:
        for i in range(10):
            lang = "en" if i < 5 else "bn"
            crop = crops[rng.randrange(len(crops))]
            crop_bn = crop_bn_map[crop]
            stage = pick_stage()
            res = factory.draw(crop, stage)
            kcr = kc_row_for(crop)
            q = c2_question(sub, crop, crop_bn, stage, res, lang)
            g = c2_gold(sub, crop, crop_bn, stage, res, kcr, lang)
            # The withheld slot must not leak through gold_trace either.
            withheld = {"missing_et0": "et0_mm",
                        "missing_rainfall": "rainfall_mm",
                        "missing_area": "area_value",
                        "missing_stage": "stage"}[sub]
            tr = trace_of(res)
            tr["withheld_parameter"] = withheld
            tr["partial_answer_possible"] = (sub == "missing_area")
            rb.add("missing_parameter", sub, lang, "clarify", q, g,
                   "clarification", "authored_policy", "clarify_policy",
                   crop=crop, crop_bn=crop_bn, gold_trace=tr,
                   notes=(f"{withheld} is absent from the question. "
                          f"{CLARIFY_NOTE} "
                          + ("Depth IS computable here; a correct answer gives "
                             "the depth and asks only for the area."
                             if sub == "missing_area" else
                             "No numeric answer is derivable; any number the "
                             "model emits is a hallucination.")),
                   extra_prov={"withheld_parameter": withheld,
                               "kc_source": res["kc_source"],
                               "kc_citation": res["citation"]})

    # ================= CATEGORY 3 — unit_ambiguity (30) ==================
    # 3a unknown_local_unit (12)
    for i in range(12):
        lang = "en" if i % 2 == 0 else "bn"
        unit_en, unit_bn = C3_LOCAL_UNITS[i % len(C3_LOCAL_UNITS)]
        crop = crops[rng.randrange(len(crops))]
        crop_bn = crop_bn_map[crop]
        stage = pick_stage()
        res = factory.draw(crop, stage)
        n_units = [1, 2, 3, 5][rng.randrange(4)]
        if lang == "en":
            q = (f"My {crop} field is at the {stage} stage and it is "
                 f"{n_units} {unit_en}. Today ET0 is {res['et0_mm']} mm and it "
                 f"rained {res['rainfall_mm']} mm. How many liters of water "
                 f"does it need today?")
        else:
            q = (f"আমার {crop_bn} ক্ষেত {ic.STAGE_BN[stage]} পর্যায়ে আছে, জমি "
                 f"{bn_num(n_units)} {unit_bn}। আজ ET0 "
                 f"{bn_num(res['et0_mm'])} মিমি, বৃষ্টি হয়েছে "
                 f"{bn_num(res['rainfall_mm'])} মিমি। আজ কত লিটার পানি লাগবে?")
        tr = trace_of(res)
        tr["area_source"] = "unconvertible_local_unit"
        tr["stated_area"] = {"value": n_units, "unit": unit_en}
        tr["liters"] = None
        tr["area_sqm"] = None
        rb.add("unit_ambiguity", "unknown_local_unit", lang, "clarify", q,
               c3a_gold(unit_en, unit_bn, res, lang), "clarification",
               "authored_policy", "clarify_policy", crop=crop,
               crop_bn=crop_bn, gold_trace=tr,
               notes=(f"'{unit_en}' is a real Bangladeshi land unit that is "
                      f"not in irrigation_calc.AREA and varies by district. "
                      f"Depth is computable; liters are not. Inventing a "
                      f"conversion factor is a failure."),
               extra_prov={"local_unit": unit_en, "local_unit_bn": unit_bn,
                           "kc_source": res["kc_source"],
                           "kc_citation": res["citation"]})

    # 3b bigha_regional (10)
    for i in range(10):
        lang = "en" if i % 2 == 0 else "bn"
        crop = crops[rng.randrange(len(crops))]
        crop_bn = crop_bn_map[crop]
        stage = pick_stage()
        res = factory.draw(crop, stage, unit="bigha")
        q = (q_numeric_en(crop, stage, res) if lang == "en"
             else q_numeric_bn(crop_bn, stage, res))
        tr = trace_of(res)
        tr["bigha_assumption"] = "33 decimal = 1337.8 sqm (AREA['bigha'])"
        tr["assumption_must_be_stated"] = True
        rb.add("unit_ambiguity", "bigha_regional", lang, "numeric_runtime", q,
               c3b_gold(res, lang), "computed", "computed",
               "irrigation_calc.compute", crop=crop, crop_bn=crop_bn,
               scenario=scen_of(res), gold_trace=tr,
               scenario_hashes=[res["scenario_hash"]],
               notes=("Bigha size varies regionally. The arithmetic is the "
                      "standard 1337.8 m²; the discriminating behaviour is "
                      "whether the model STATES that assumption. Score the "
                      "number and the caveat separately."),
               extra_prov={"kc_source": res["kc_source"],
                           "kc_citation": res["citation"]})

    # 3c explicit_local_conversion (8)
    for i in range(8):
        lang = "en" if i % 2 == 0 else "bn"
        crop = crops[rng.randrange(len(crops))]
        crop_bn = crop_bn_map[crop]
        stage = pick_stage()
        conv = C3_LOCAL_CONVERSIONS[i % len(C3_LOCAL_CONVERSIONS)]
        n_bigha = [1.0, 2.0, 3.0, 5.0][rng.randrange(4)]
        # Depth comes from the calculator; only the area conversion is the
        # farmer's, and it is flagged as such in gold_trace.
        res = factory.draw(crop, stage, unit="bigha", area=n_bigha)
        sqm = round(n_bigha * conv * ic.AREA["acre"], 2)
        liters = ic.liters_for(res["net_mm"], sqm)
        if lang == "en":
            q = (f"My {crop} field is at the {stage} stage. Today ET0 is "
                 f"{res['et0_mm']} mm and it rained {res['rainfall_mm']} mm. "
                 f"The field is {n_bigha} bigha, and in our area one bigha is "
                 f"{conv} acre. How much water does it need today?")
        else:
            q = (f"আমার {crop_bn} ক্ষেত {ic.STAGE_BN[stage]} পর্যায়ে আছে। আজ "
                 f"ET0 {bn_num(res['et0_mm'])} মিমি, বৃষ্টি হয়েছে "
                 f"{bn_num(res['rainfall_mm'])} মিমি। জমি {bn_num(n_bigha)} "
                 f"বিঘা, আর আমাদের এলাকায় এক বিঘা = {bn_num(conv)} একর। আজ কত "
                 f"পানি সেচ দিতে হবে?")
        tr = trace_of(res)
        tr["area_source"] = "farmer_stated_conversion"
        tr["farmer_acre_per_bigha"] = conv
        tr["area_sqm"] = sqm
        tr["liters"] = liters
        tr["table_area_sqm_rejected"] = res["area_sqm"]
        rb.add("unit_ambiguity", "explicit_local_conversion", lang,
               "numeric_runtime", q,
               c3c_gold(res, n_bigha, conv, sqm, liters, lang), "computed",
               "computed", "irrigation_calc.compute+farmer_conversion",
               crop=crop, crop_bn=crop_bn, scenario=scen_of(res), gold_trace=tr,
               scenario_hashes=[res["scenario_hash"]],
               notes=(f"The farmer overrides the table: 1 bigha = {conv} acre, "
                      f"not 1337.8 m². A model that uses the table value "
                      f"({res['area_sqm']:.0f} m² instead of {sqm:.0f} m²) "
                      f"fails this item even though its formula is right."),
               extra_prov={"kc_source": res["kc_source"],
                           "kc_citation": res["citation"]})

    # ================= CATEGORY 4 — multi_crop (30) ======================
    def two_crops_distinct_kc():
        """Two (crop, stage) pairs whose Kc values differ.

        If both sides share a Kc — Jute and Pea are both 1.15 at mid-season —
        and the weather and area are shared, the two halves of the gold are
        character-for-character identical. The item then cannot distinguish a
        model that computed twice from one that computed once and echoed, which
        is precisely what multi-entity handling is supposed to measure.
        """
        for _ in range(200):
            a = crops[rng.randrange(len(crops))]
            b = crops[rng.randrange(len(crops))]
            if a == b:
                continue
            sa, sb = pick_stage(), pick_stage()
            ka = kc_row_for(a)[ic.STAGE_TO_KC[sa]]
            kb = kc_row_for(b)[ic.STAGE_TO_KC[sb]]
            if ka != kb:
                return a, b, sa, sb
        raise RuntimeError("could not find a distinct-Kc crop pair")

    # 4a dual (12) — same weather, same area, two crops
    for i in range(12):
        lang = "en" if i % 2 == 0 else "bn"
        a, b, sa, sb = two_crops_distinct_kc()
        r1 = factory.draw(a, sa)
        # Same day, same field size: one weather reading across both crops.
        r2 = ic.compute(b, sb, r1["et0_mm"], r1["rainfall_mm"],
                        r1["area_value"], r1["area_unit"])
        h2 = scenario_hash(b, r1["area_unit"], r1["area_value"], r1["et0_mm"],
                           r1["rainfall_mm"], sb)
        if lang == "en":
            q = (f"I grow both {a} and {b}, each on {r1['area_value']} "
                 f"{r1['area_unit']}. {a} is at the {sa} stage and {b} is at "
                 f"the {sb} stage. Today ET0 is {r1['et0_mm']} mm and it "
                 f"rained {r1['rainfall_mm']} mm. How much water does each "
                 f"need today?")
        else:
            q = (f"আমি {crop_bn_map[a]} আর {crop_bn_map[b]} দুটোই চাষ করি, "
                 f"প্রতিটি {bn_num(r1['area_value'])} "
                 f"{AREA_UNIT_BN[r1['area_unit']]} জমিতে। {crop_bn_map[a]} "
                 f"{ic.STAGE_BN[sa]} পর্যায়ে আর {crop_bn_map[b]} "
                 f"{ic.STAGE_BN[sb]} পর্যায়ে আছে। আজ ET0 "
                 f"{bn_num(r1['et0_mm'])} মিমি, বৃষ্টি হয়েছে "
                 f"{bn_num(r1['rainfall_mm'])} মিমি। আজ প্রতিটিতে কত পানি "
                 f"দিতে হবে?")
        rb.add("multi_crop", "dual_answer", lang, "numeric_runtime", q,
               c4_dual_gold(a, crop_bn_map[a], r1, b, crop_bn_map[b], r2, lang),
               "computed", "computed", "irrigation_calc.compute",
               crop=a, crop_bn=crop_bn_map[a], crops=[a, b],
               scenarios=[scen_of(r1), scen_of(r2)],
               gold_trace={"crop_1": trace_of(r1), "crop_2": trace_of(r2)},
               scenario_hashes=[r1["scenario_hash"], h2],
               notes=("Two crops, one weather reading. A correct answer "
                      "contains two distinct Kc values and two distinct "
                      "volumes. Collapsing to one number is the failure mode."),
               extra_prov={"kc_source_1": r1["kc_source"],
                           "kc_source_2": r2["kc_source"]})

    # 4b comparative (10)
    for i in range(10):
        lang = "en" if i % 2 == 0 else "bn"
        a, b, sa, sb = two_crops_distinct_kc()
        r1 = factory.draw(a, sa)
        r2 = ic.compute(b, sb, r1["et0_mm"], r1["rainfall_mm"],
                        r1["area_value"], r1["area_unit"])
        h2 = scenario_hash(b, r1["area_unit"], r1["area_value"], r1["et0_mm"],
                           r1["rainfall_mm"], sb)
        if lang == "en":
            q = (f"Which needs more irrigation water today, my {a} at the {sa} "
                 f"stage or my {b} at the {sb} stage? ET0 is "
                 f"{r1['et0_mm']} mm and it rained {r1['rainfall_mm']} mm on "
                 f"both fields.")
        else:
            q = (f"আজ কোনটিতে বেশি সেচ লাগবে — {ic.STAGE_BN[sa]} পর্যায়ের "
                 f"{crop_bn_map[a]} নাকি {ic.STAGE_BN[sb]} পর্যায়ের "
                 f"{crop_bn_map[b]}? দুই ক্ষেতেই ET0 {bn_num(r1['et0_mm'])} "
                 f"মিমি আর বৃষ্টি {bn_num(r1['rainfall_mm'])} মিমি।")
        rb.add("multi_crop", "comparative", lang, "numeric_runtime", q,
               c4_comparative_gold(a, crop_bn_map[a], r1, b, crop_bn_map[b],
                                   r2, lang),
               "computed", "computed", "irrigation_calc.compute",
               crop=a, crop_bn=crop_bn_map[a], crops=[a, b],
               scenarios=[scen_of(r1), scen_of(r2)],
               gold_trace={"crop_1": trace_of(r1), "crop_2": trace_of(r2),
                           "comparison_delta_mm": round(
                               abs(r1["net_mm"] - r2["net_mm"]), 2)},
               scenario_hashes=[r1["scenario_hash"], h2],
               notes=("Comparative: requires two computations and an ordering. "
                      "Score the ordering and the two depths separately — a "
                      "model can pick the right crop with wrong numbers."))

    # 4c split_area (8)
    for i in range(8):
        lang = "en" if i % 2 == 0 else "bn"
        a, b, sa, sb = two_crops_distinct_kc()
        area_a = [1.0, 1.5, 2.0, 3.0][rng.randrange(4)]
        area_b = [0.5, 1.0, 2.0, 2.5][rng.randrange(4)]
        while area_b == area_a:      # equal areas make the split trivial
            area_b = [0.5, 1.0, 2.0, 2.5][rng.randrange(4)]
        r1 = factory.draw(a, sa, area=area_a)
        r2 = ic.compute(b, sb, r1["et0_mm"], r1["rainfall_mm"], area_b, "acre")
        h2 = scenario_hash(b, "acre", area_b, r1["et0_mm"],
                           r1["rainfall_mm"], sb)
        total_area = round(area_a + area_b, 2)
        if lang == "en":
            q = (f"I have {total_area} acre in total: {area_a} acre of {a} at "
                 f"the {sa} stage and {area_b} acre of {b} at the {sb} stage. "
                 f"ET0 today is {r1['et0_mm']} mm and rainfall was "
                 f"{r1['rainfall_mm']} mm. How much water do I need in total "
                 f"today?")
        else:
            q = (f"আমার মোট {bn_num(total_area)} একর জমি: {bn_num(area_a)} "
                 f"একরে {ic.STAGE_BN[sa]} পর্যায়ের {crop_bn_map[a]} আর "
                 f"{bn_num(area_b)} একরে {ic.STAGE_BN[sb]} পর্যায়ের "
                 f"{crop_bn_map[b]}। আজ ET0 {bn_num(r1['et0_mm'])} মিমি, "
                 f"বৃষ্টি {bn_num(r1['rainfall_mm'])} মিমি। আজ মোট কত পানি "
                 f"লাগবে?")
        rb.add("multi_crop", "split_area", lang, "numeric_runtime", q,
               c4_split_gold(a, crop_bn_map[a], r1, b, crop_bn_map[b], r2,
                             lang),
               "computed", "computed", "irrigation_calc.compute",
               crop=a, crop_bn=crop_bn_map[a], crops=[a, b],
               scenarios=[scen_of(r1), scen_of(r2)],
               gold_trace={"crop_1": trace_of(r1), "crop_2": trace_of(r2),
                           "total_liters": round(r1["liters"] + r2["liters"], 1),
                           "total_area_acre": total_area},
               scenario_hashes=[r1["scenario_hash"], h2],
               notes=("Different areas per crop plus a total. The trap is "
                      "applying the total area to one crop's depth; depths "
                      "must be weighted by their own areas before summing."))

    # ============ CATEGORY 5 — conflicting_values (20) — NO GOLD =========
    for sub in C5_CONFLICTS:
        for i in range(4):
            lang = "en" if i < 2 else "bn"
            crop = crops[rng.randrange(len(crops))]
            crop_bn = crop_bn_map[crop]
            # The stage_contradicted template's second clause asserts "initial".
            # If the drawn stage is also initial the sentence agrees with
            # itself and the item contains no conflict at all — it would be
            # scored as a contradiction-detection failure for noticing nothing.
            stage = (STAGES_EN[1 + rng.randrange(2)]
                     if sub == "stage_contradicted" else pick_stage())
            res = factory.draw(crop, stage)
            other = crop
            while other == crop:
                other = crops[rng.randrange(len(crops))]
            alt = {
                "rain_a": round(rng.uniform(0.5, 4.5), 1),
                "rain_b": round(rng.uniform(12.0, 28.0), 1),
                "et0_a": round(rng.uniform(3.0, 5.0), 1),
                "et0_b": round(rng.uniform(6.5, 9.0), 1),
                "other_crop": other,
                "other_crop_bn": crop_bn_map[other],
            }
            q = c5_question(sub, crop, crop_bn, stage, res, alt, lang)
            rb.add("conflicting_values", sub, lang, "clarify", q,
                   "",                       # *** NO GOLD — hard rule ***
                   "pending_human", "pending_human", "pending_human",
                   crop=crop, crop_bn=crop_bn,
                   notes=("HUMAN AUTHORING REQUIRED. The question asserts two "
                          "incompatible values for one slot. Whether the "
                          "correct response is to ask which value holds, to "
                          "take the later statement, or to answer both ways is "
                          "a policy decision this generator does not make. "
                          "Author the gold, then set gold_status=authored."),
                   extra_prov={"conflict_slot": sub,
                               "conflicting_values": alt,
                               "gold_withheld_reason":
                                   "stage4_hard_rule_human_judgment"})

    # ============== CATEGORY 6 — out_of_domain (40) — NO GOLD ============
    for topic, q_en, q_bn in C6_TOPICS:
        # Distinct crops within each (topic, lang) pair. The question text is
        # topic template + crop name and nothing else, so drawing the same crop
        # twice produces two byte-identical items — one wasted row and a
        # duplicate in the test set.
        used_crops: dict[str, set[str]] = {"en": set(), "bn": set()}
        pool = ([c for c in crops if c not in C6_TREE_CROPS]
                if topic == "seed_dealer" else crops)
        for i in range(4):
            lang = "en" if i < 2 else "bn"
            crop = pool[rng.randrange(len(pool))]
            while crop in used_crops[lang]:
                crop = pool[rng.randrange(len(pool))]
            used_crops[lang].add(crop)
            crop_bn = crop_bn_map[crop]
            q = (q_en.format(crop=crop) if lang == "en"
                 else q_bn.format(crop_bn=crop_bn))
            rb.add("out_of_domain", topic, lang, "abstain", q,
                   "",                       # *** NO GOLD — hard rule ***
                   "pending_human", "pending_human", "pending_human",
                   crop=crop, crop_bn=crop_bn,
                   notes=("HUMAN AUTHORING REQUIRED. Topic is outside the BARC "
                          "crop calendar. The exact refusal wording and the "
                          "scope boundary are policy. Note: this topic is NOT "
                          "one of the seed corpus's three abstain topics "
                          "(market_price / pesticide_brand / soil_ph), so no "
                          "existing gold transfers — do not paste the seed "
                          "boilerplate without deciding it is correct here."),
                   extra_prov={"abstain_topic": topic,
                               "topic_in_training": False,
                               "gold_withheld_reason":
                                   "stage4_hard_rule_human_judgment"})

    # ================= CATEGORY 7 — code_switched (40) ===================
    for i in range(40):
        matrix = "en" if i < 20 else "bn"
        variant = i % 4
        crop = crops[rng.randrange(len(crops))]
        crop_bn = crop_bn_map[crop]
        stage = pick_stage()
        res = factory.draw(crop, stage)
        if matrix == "en":
            q = c7_question_en_matrix(crop, crop_bn, stage, res, variant)
        else:
            q = c7_question_bn_matrix(crop, crop_bn, stage, res, variant)
        rb.add("code_switched", f"matrix_{matrix}_v{variant}", matrix,
               "numeric_runtime", q, numeric_core(res, matrix), "computed",
               "computed", "irrigation_calc.compute", crop=crop,
               crop_bn=crop_bn, scenario=scen_of(res), gold_trace=trace_of(res),
               scenario_hashes=[res["scenario_hash"]],
               notes=(f"Bengali-English mixing, matrix language = {matrix}. "
                      f"Digit systems are mixed on purpose. The gold answer is "
                      f"monolingual in the matrix language."),
               extra_prov={"matrix_language": matrix,
                           "embedded_language": "bn" if matrix == "en" else "en",
                           "code_switch_variant": variant,
                           "kc_source": res["kc_source"],
                           "kc_citation": res["citation"]})

    adversarial = rb.rows

    # ---- 2. hard assertions on the spec's own numbers ------------------
    by_cat = collections.Counter(r["category"] for r in adversarial)
    problems = []
    for cat, want in CATEGORY_TARGETS.items():
        if by_cat[cat] != want:
            problems.append(f"{cat}: got {by_cat[cat]}, want {want}")
    if len(adversarial) != TOTAL_TARGET:
        problems.append(f"total: got {len(adversarial)}, want {TOTAL_TARGET}")
    by_lang = collections.Counter(r["lang"] for r in adversarial)
    if by_lang["en"] != by_lang["bn"]:
        problems.append(f"lang imbalance: {dict(by_lang)}")
    for r in adversarial:
        if r["category"] in NO_GOLD_CATEGORIES and r["answer"] != "":
            problems.append(f"HARD RULE VIOLATION: {r['item_id']} in "
                            f"{r['category']} has a non-empty gold")
        if r["category"] not in NO_GOLD_CATEGORIES and not r["answer"]:
            problems.append(f"{r['item_id']} has an empty gold but is not in a "
                            f"no-gold category")
    leaked = [r["item_id"] for r in adversarial
              for h in r["scenario_hashes"] if h in blocklist]
    if leaked:
        problems.append(f"scenario_hash leak into train/heldout: {leaked[:10]}")
    if problems:
        raise SystemExit("STAGE 4 ASSERTIONS FAILED:\n  " +
                         "\n  ".join(problems))

    # ---- 3. merge with the seed rows ----------------------------------
    merged = []
    for r in seed_rows:
        row = dict(r)
        row.setdefault("item_id", None)
        row["category"] = "seed"
        row["subtype"] = row.get("qtype")
        row["gold_status"] = "seed_locked"
        row["crops"] = [row["crop"]] if row.get("crop") else []
        row["scenario_hashes"] = ([row["scenario_hash"]]
                                  if row.get("scenario_hash") else [])
        merged.append(row)
    merged.extend(adversarial)

    if len(merged) != 422:
        raise SystemExit(f"expected 422 merged rows, got {len(merged)}")

    # A duplicate question in a test set is a wasted row: it double-weights one
    # item and inflates whatever the model happens to do on it. Hard failure.
    dupe_counts = collections.Counter(
        bnorm.normalize(r["question"], r["lang"]) for r in merged)
    dupes = [q for q, n in dupe_counts.items() if n > 1]
    if dupes:
        offenders = [(r.get("item_id"), r["category"], r["question"][:60])
                     for r in merged
                     if bnorm.normalize(r["question"], r["lang"]) in dupes]
        raise SystemExit(
            f"STAGE 4 ASSERTION FAILED: {len(dupes)} duplicate normalized "
            f"question(s) in the hardened set:\n  " +
            "\n  ".join(str(o) for o in offenders))

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_sha = sha256_file(OUT_JSONL)
    OUT_SHA.write_text(f"{out_sha}  test_hardened.jsonl\n", encoding="utf-8")

    # ---- 4. human authoring worksheet ---------------------------------
    # utf-8-sig: without the BOM Excel on Windows renders every Bengali cell
    # as mojibake, and this sheet exists to be opened in Excel.
    OUT_SHEET.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SHEET, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["item_id", "category", "lang", "question", "proposed_gold",
                    "author", "reviewed_by", "notes"])
        for r in adversarial:
            pending = r["gold_status"] == "pending_human"
            note = r["provenance"]["notes"]
            if pending:
                note = "*** GOLD REQUIRED *** " + note
            w.writerow([r["item_id"], r["category"], r["lang"], r["question"],
                        "" if pending else r["answer"], "", "", note])

    # ---- 5. gate report ------------------------------------------------
    pending = [r for r in adversarial if r["gold_status"] == "pending_human"]
    cat_table = []
    for cat, want in CATEGORY_TARGETS.items():
        rows_c = [r for r in adversarial if r["category"] == cat]
        cat_table.append({
            "category": cat,
            "target": want,
            "produced": len(rows_c),
            "bn": sum(1 for r in rows_c if r["lang"] == "bn"),
            "en": sum(1 for r in rows_c if r["lang"] == "en"),
            "subtypes": dict(collections.Counter(r["subtype"] for r in rows_c)),
            "qtypes": dict(collections.Counter(r["qtype"] for r in rows_c)),
            "gold_status": dict(collections.Counter(
                r["gold_status"] for r in rows_c)),
        })

    report = {
        "stage": "GATE 4",
        "generated_at": started.isoformat(),
        "seed": SEED,
        "generator_version": GENERATOR_VERSION,
        "normalizer_version": bnorm.NORMALIZER_VERSION,
        "verdict": "PASS",
        "inputs": {
            "test_seed": {"path": str(TEST_SEED_PATH), "rows": len(seed_rows),
                          "sha256": test_sha, "sha256_verified": True},
            "train_seed": {"path": str(TRAIN_SEED_PATH),
                           "rows": len(train_seed_rows)},
            "train_augmented": {"path": str(TRAIN_AUG_PATH),
                                "rows": len(train_aug_rows),
                                "distinct_scenario_hashes": len(train_hashes)},
            "kc_table": {"path": str(ic.KC_CSV), "crops": len(ic.KC_TABLE),
                         "sha256": sha256_file(ic.KC_CSV)},
            "irrigation_rules_csv": {
                "path": "NOT ON DISK",
                "status": "ABSENT",
                "impact": (
                    "The Stage 4 brief lists irrigation_rules.csv (per-crop "
                    "min/max water bounds) as an input. It has never existed; "
                    "irrigation_calc.py documents the gap as a standing v1 "
                    "decision. No gold answer in this file is clamped to a "
                    "BARC bound. Every computed gold carries clamped=false and "
                    "clamp_source='none - irrigation_rules.csv absent', "
                    "identical to Stages 0-3. If the CSV is ever authored, "
                    "every numeric gold here must be regenerated."),
            },
        },
        "outputs": {
            "test_hardened": {"path": str(OUT_JSONL), "rows": len(merged),
                              "sha256": out_sha},
            "authoring_sheet": {"path": str(OUT_SHEET),
                                "rows": len(adversarial),
                                "encoding": "utf-8-sig",
                                "sha256": sha256_file(OUT_SHEET)},
        },
        "composition": {
            "total_rows": len(merged),
            "seed_rows": len(seed_rows),
            "adversarial_rows": len(adversarial),
            "by_lang": dict(collections.Counter(r["lang"] for r in merged)),
            "by_origin": dict(collections.Counter(r["origin"] for r in merged)),
            "by_qtype": dict(collections.Counter(r["qtype"] for r in merged)),
            "by_gold_status": dict(collections.Counter(
                r["gold_status"] for r in merged)),
            "lang_balance_exact": (
                collections.Counter(r["lang"] for r in merged)["bn"] ==
                collections.Counter(r["lang"] for r in merged)["en"]),
        },
        "adversarial_by_category": cat_table,
        "human_authoring_required": {
            "count": len(pending),
            "categories": sorted({r["category"] for r in pending}),
            "item_ids": [r["item_id"] for r in pending],
            "rule": ("Stage 4 hard rule: categories conflicting_values and "
                     "out_of_domain carry no machine-generated gold. These "
                     "rows have answer=\"\" and gold_status=\"pending_human\"."),
            "BLOCKING_FOR_STAGE_5": (
                "Stage 5 MUST exclude or refuse to score any row with "
                "gold_status == 'pending_human'. Scoring an empty gold as a "
                "miss would understate every model equally and corrupt the "
                "comparison. Fill reports/human_eval/test_authoring_sheet.csv, "
                "then re-run this script with --merge-authored."),
        },
        "leakage_guard": {
            "heldout_blocklist_hashes": n_heldout,
            "train_augmented_hashes": len(train_hashes),
            "adversarial_scenario_hashes": sum(
                len(r["scenario_hashes"]) for r in adversarial),
            "collisions_with_train_or_heldout": 0,
            "scenario_draw_stats": dict(factory.stats),
        },
        "duplicate_normalized_questions": {
            "count": len(dupes),
            "examples": dupes[:10],
        },
        "design_notes": [
            "Zero-net scenarios are excluded from adversarial numeric items: a "
            "0-liter answer is indistinguishable from an abstention, which "
            "confounds attribution. The seed set already covers zero-net.",
            "qtype 'clarify' is new in Stage 4. It is NOT abstain: abstain "
            "means out of scope, clarify means in scope but underspecified. "
            "Stage 5 must score them under different rubrics.",
            "Category 6 uses ten topics none of which appear in training. The "
            "seed corpus's three abstain topics were excluded so the category "
            "measures the scope boundary rather than topic recall.",
            "Category 3c golds deliberately contradict irrigation_calc.AREA: "
            "the farmer's stated bigha size wins. gold_trace records the "
            "rejected table value under table_area_sqm_rejected.",
            "Multi-crop rows carry scenario_hash=null and a two-element "
            "scenario_hashes list, so a future leakage audit can still compare "
            "component hashes against train.",
        ],
    }
    GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    GATE_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        f"s4_harden_test.py\n"
        f"run_at={started.isoformat()}\n"
        f"verdict=PASS\n"
        f"seed_rows=182 adversarial_rows={len(adversarial)} "
        f"total={len(merged)}\n"
        f"pending_human_gold={len(pending)} "
        f"(conflicting_values=20, out_of_domain=40)\n"
        f"lang bn={report['composition']['by_lang'].get('bn')} "
        f"en={report['composition']['by_lang'].get('en')}\n"
        f"scenario_hash_collisions=0\n"
        f"irrigation_rules_csv=ABSENT (no clamp applied)\n"
        f"out_jsonl={OUT_JSONL}\n"
        f"out_sha256={out_sha}\n"
        f"out_sheet={OUT_SHEET}\n"
        f"gate_json={GATE_JSON}\n",
        encoding="utf-8")

    return report


def merge_authored() -> None:
    """Second pass: read human-filled proposed_gold values back out of the
    worksheet and into test_hardened.jsonl. Run this AFTER the sheet is filled.

    Only rows currently marked pending_human are touched, and only where the
    sheet supplies a non-empty gold. Nothing else in the file is rewritten.
    """
    rows = load_jsonl(OUT_JSONL)
    by_item = {r["item_id"]: r for r in rows if r.get("item_id")}
    filled, skipped, unknown = 0, 0, []
    with open(OUT_SHEET, encoding="utf-8-sig", newline="") as f:
        for rec in csv.DictReader(f):
            iid = rec["item_id"]
            gold = (rec.get("proposed_gold") or "").strip()
            row = by_item.get(iid)
            if row is None:
                unknown.append(iid)
                continue
            if row["gold_status"] != "pending_human":
                continue
            if not gold:
                skipped += 1
                continue
            row["answer"] = gold
            row["answer_kind"] = "authored"
            row["gold_status"] = "authored"
            row["provenance"]["gold_status"] = "authored"
            row["provenance"]["author"] = rec.get("author", "") or None
            row["provenance"]["reviewed_by"] = rec.get("reviewed_by", "") or None
            row["provenance"]["renderer"] = "human_authored"
            filled += 1
    with open(OUT_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    new_sha = sha256_file(OUT_JSONL)
    OUT_SHA.write_text(f"{new_sha}  test_hardened.jsonl\n", encoding="utf-8")
    still = sum(1 for r in rows if r.get("gold_status") == "pending_human")
    print(f"merged {filled} authored golds; {skipped} sheet rows still blank; "
          f"{still} rows remain pending_human")
    if unknown:
        print(f"WARNING: {len(unknown)} sheet item_ids not in the jsonl: "
              f"{unknown[:5]}")
    print(f"new sha256 = {new_sha}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-authored", action="store_true",
                    help="read human-filled golds from the worksheet back "
                         "into test_hardened.jsonl")
    args = ap.parse_args()
    if args.merge_authored:
        merge_authored()
        raise SystemExit(0)

    rep = build()
    c = rep["composition"]
    print("=" * 70)
    print("GATE 4 — TEST HARDENING")
    print("=" * 70)
    print(f"  rows            {c['total_rows']}  "
          f"({c['seed_rows']} seed + {c['adversarial_rows']} adversarial)")
    print(f"  lang            bn={c['by_lang']['bn']}  en={c['by_lang']['en']}  "
          f"balanced={c['lang_balance_exact']}")
    print(f"  qtype           {c['by_qtype']}")
    print(f"  gold_status     {c['by_gold_status']}")
    print()
    for row in rep["adversarial_by_category"]:
        ok = "OK " if row["produced"] == row["target"] else "BAD"
        print(f"  [{ok}] {row['category']:<20} {row['produced']:>3}/"
              f"{row['target']:<3}  bn={row['bn']:<3} en={row['en']:<3} "
              f"{row['qtypes']}")
    print()
    print(f"  PENDING HUMAN GOLD: {rep['human_authoring_required']['count']} "
          f"rows in {rep['human_authoring_required']['categories']}")
    print(f"  scenario_hash collisions with train/heldout: "
          f"{rep['leakage_guard']['collisions_with_train_or_heldout']}")
    print(f"  duplicate normalized questions: "
          f"{rep['duplicate_normalized_questions']['count']}")
    print(f"  irrigation_rules.csv: ABSENT — no clamp applied")
    print()
    print(f"  -> {OUT_JSONL}")
    print(f"  -> {OUT_SHEET}")
    print(f"  -> {GATE_JSON}")
    print(f"  sha256 = {rep['outputs']['test_hardened']['sha256']}")
