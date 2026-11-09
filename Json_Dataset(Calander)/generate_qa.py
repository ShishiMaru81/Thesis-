#!/usr/bin/env python3
"""
generate_qa.py  —  Build a bilingual instruction-tuning dataset from the
repaired BARC crop-calendar JSONs.

CORE RULE (this is what protects your thesis): every emitted answer must trace
to a real field in a real file. Three answer sources, in decreasing safety:

  1. RETRIEVED   answer text == a verified field value (value_bn / value_en).
                 Used for factual "what does the calendar say" questions.
  2. COMPUTED    answer produced by the deterministic ETc formula in code.
                 Used for runtime-conditioned numeric questions (thesis H1).
                 The number is calculated here, never taken from an LLM.
  3. ABSTAIN     a small set of out-of-scope questions whose gold answer is
                 "the calendar does not specify this" — teaches the model to
                 hand off to RAG instead of hallucinating.

Nothing is invented. Fields flagged exclude_from_qa_generation, needs_human_review,
or numeric_backfill_declined are skipped or routed to review, never emitted as
confident training truth.

PARAPHRASE GROUPS: each fact (one crop + one field/stage/topic + one language)
is emitted as several differently-worded questions sharing one answer and one
"group_id". The train/val/test split is done by group_id, not by row — this is
what stops a paraphrase of a train-set question from leaking into val/test and
inflating eval numbers. See build() for the split + leakage assertion.

Run:  python generate_qa.py --in ./json --out ./qa_out
"""
import json, glob, os, argparse, random, hashlib, collections, datetime, csv

random.seed(42)  # reproducible splits — panels ask about this

# ----------------------------------------------------------------- Kc lookup (real, per-crop)
def normalize_crop_name(name):
    """kc_values_hybrid.csv uses underscores (Aman_Rice); JSON crop_name_en uses
    spaces (Aman Rice). Normalize both to the same key so the same real Kc row
    is found either way — no value is invented, only the join key is fixed."""
    return name.replace("_", " ").strip().casefold()

def load_kc_table():
    """normalized crop name -> {kc_ini, kc_mid, kc_end, kc_source, citation} from kc_values_hybrid.csv."""
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kc_values_hybrid.csv")
    table = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[normalize_crop_name(row["crop_name_en"])] = {
                "kc_ini": float(row["kc_ini"]),
                "kc_mid": float(row["kc_mid"]),
                "kc_end": float(row["kc_end"]),
                "kc_source": row["kc_source"],
                "citation": row["citation"],
            }
    return table

KC_TABLE = load_kc_table()
SKIPPED_KC_CROPS = []  # crops present in the JSON folder with no matching Kc row

# ----------------------------------------------------------------- helpers
BD = "০১২৩৪৫৬৭৮৯"
def bn_num(n):
    return "".join(BD[int(c)] if c.isdigit() else c for c in str(n))

def field(feat, key):
    v = feat.get(key)
    return v if isinstance(v, dict) and v.get("provenance") not in ("absent", None) else None

def usable(v):
    """A field is usable for a RETRIEVED answer only if it is grounded and clean."""
    if not v:
        return False
    if v.get("exclude_from_qa_generation"):
        return False
    if v.get("needs_human_review") or v.get("numeric_backfill_declined"):
        return False
    return True

def qhash(instr):
    return hashlib.sha1(instr.encode("utf-8")).hexdigest()[:12]

def group_id_for(crop, field_key, lang):
    """Every paraphrase of one underlying fact (same crop + field/stage/topic +
    language) shares this id. build() splits by group_id so paraphrases of the
    same fact never end up split across train/val/test."""
    return f"{crop.replace(' ', '_')}__{field_key}__{lang}"

# ----------------------------------------------------------------- ETc engine
# Canonical formula from computable_link. All numbers COMPUTED, not retrieved.
def effective_rain(mm):
    if mm <= 5:  return round(mm * 0.85, 2)
    if mm <= 15: return round(mm * 0.45, 2)
    return round(mm * 0.25, 2)

def net_irrigation_mm(kc, et0, rain_mm, lo=None, hi=None):
    net = kc * et0 - effective_rain(rain_mm)
    net = max(net, 0.0)
    if lo is not None: net = max(net, lo)
    if hi is not None: net = min(net, hi)
    return round(net, 2)

AREA = {"acre": 4046.86, "bigha": 1337.8, "katha": 66.89, "hectare": 10000.0, "sqm": 1.0}

# ----------------------------------------------------------------- QA templates
def qa(instruction, answer, crop, qtype, lang, source, gold_kind, group_id, extra=None):
    rec = {
        "id": f"{crop}_{qtype}_{lang}_{qhash(instruction)}",
        "crop": crop,
        "lang": lang,
        "question_type": qtype,       # factual | numeric_runtime | abstain
        "gold_kind": gold_kind,       # retrieved | computed | abstain
        "group_id": group_id,         # shared by every paraphrase of this fact
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": answer},
        ],
        "grounding": source,          # file + field, or formula trace
    }
    if extra:
        rec.update(extra)
    return rec

# --- (A) factual, retrieved from verified field values -----------------------
# Each field has a LIST of paraphrases per language: direct / imperative /
# indirect / casual phrasing, varied vocabulary. The answer is identical for
# every paraphrase in a group — only the question wording changes.
FACTUAL_TEMPLATES = [
    # (field_key, [en paraphrases], [bn paraphrases])
    ("3_বপন_ঋতু", [
        "What is the sowing season for {crop}?",
        "When should {crop} be planted?",
        "Tell me the planting season for {crop}.",
        "I want to know when to seed {crop}.",
        "Which months are best for sowing {crop}?",
        "When do I plant {crop}?",
     ], [
        "{crop_bn} বপনের/রোপণের সময় কখন?",
        "{crop_bn} কোন মাসে বপন করতে হয়?",
        "{crop_bn} চাষের জন্য বীজ বোনার উপযুক্ত সময় বলুন।",
        "আমি জানতে চাই {crop_bn} কখন লাগাতে হয়।",
        "ভাই, {crop_bn} কবে লাগামু?",
        "{crop_bn} রোপণের মৌসুম কোনটি?",
     ]),
    ("5_প্রথম_সেচ_সময়", [
        "When should the first irrigation be applied for {crop}?",
        "Tell me when to give the first watering to {crop}.",
        "I want to know the timing of the first irrigation for {crop}.",
        "How soon after planting should I irrigate {crop} the first time?",
        "When's the first watering due for {crop}?",
        "What is the correct time for the initial irrigation of {crop}?",
     ], [
        "{crop_bn} ফসলে প্রথম সেচ কখন দিতে হবে?",
        "{crop_bn}-এ প্রথমবার পানি কখন দিতে হয় বলুন।",
        "আমি জানতে চাই {crop_bn} ফসলে প্রথম সেচের সময় কখন।",
        "{crop_bn} লাগানোর পর প্রথম সেচ কবে দিমু?",
        "প্রথম সেচের সঠিক সময় কী {crop_bn} ফসলের জন্য?",
        "{crop_bn} খেতে প্রথম পানি সেচ কোন সময় দেওয়া উচিত?",
     ]),
    ("6_প্রথম_সেচ_ধরন", [
        "What kind of first irrigation does {crop} need?",
        "Tell me what type of first watering {crop} requires.",
        "I want to know the type of the first irrigation for {crop}.",
        "What sort of initial watering should {crop} get?",
        "How should the first irrigation of {crop} be done?",
        "What form does the first irrigation for {crop} take?",
     ], [
        "{crop_bn} ফসলে প্রথম সেচ কী ধরনের হওয়া উচিত?",
        "{crop_bn}-এর প্রথম সেচ কেমন হতে হবে বলুন।",
        "আমি জানতে চাই {crop_bn} ফসলে প্রথম সেচ কোন ধরনের।",
        "{crop_bn} খেতে প্রথম পানি কেমনে দিমু?",
        "প্রথম সেচ {crop_bn} ফসলে কী রকম হওয়া দরকার?",
        "{crop_bn} ফসলের প্রথম সেচের ধরন কী?",
     ]),
    ("8_সেচ_পদ্ধতি", [
        "What irrigation method is recommended for {crop}?",
        "Tell me the recommended watering method for {crop}.",
        "I want to know which irrigation method suits {crop}.",
        "How should {crop} be irrigated — which method?",
        "What's the best way to water {crop}?",
        "Which irrigation technique works for {crop}?",
     ], [
        "{crop_bn} ফসলে কোন সেচ পদ্ধতি সুপারিশ করা হয়েছে?",
        "{crop_bn}-এর জন্য কোন সেচ পদ্ধতি ভালো বলুন।",
        "আমি জানতে চাই {crop_bn} ফসলে কী পদ্ধতিতে সেচ দিতে হয়।",
        "{crop_bn} খেতে কেমনে পানি দিমু, কোন পদ্ধতিতে?",
        "{crop_bn} ফসলের সেচের উপযুক্ত পদ্ধতি কোনটি?",
        "কোন কৌশলে {crop_bn} ফসলে সেচ দেওয়া উচিত?",
     ]),
    ("9_সংকটকাল_পর্যায়সমূহ", [
        "Which growth stages of {crop} are most sensitive to water stress?",
        "Tell me which stages of {crop} growth are critical for water.",
        "I want to know the water-sensitive stages of {crop}.",
        "At which growth phase does {crop} need water the most?",
        "When is {crop} most vulnerable to a lack of water?",
        "What are the critical irrigation stages for {crop}?",
     ], [
        "{crop_bn} ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?",
        "{crop_bn}-এর কোন পর্যায়ে পানির অভাব সবচেয়ে ক্ষতিকর বলুন।",
        "আমি জানতে চাই {crop_bn} ফসলের সংকটকালীন পর্যায় কোনগুলো।",
        "{crop_bn} খেতে পানি না দিলে কোন সময় সবচেয়ে ক্ষতি হয়?",
        "{crop_bn} ফসলের কোন সময়টা পানির জন্য সবচেয়ে জরুরি?",
        "কোন বৃদ্ধির ধাপে {crop_bn} ফসল পানির সংকটে বেশি ভোগে?",
     ]),
    ("13_নিষ্কাশন_নিয়ম", [
        "What should be done about drainage for {crop}?",
        "Tell me the drainage requirements for {crop}.",
        "I want to know how to manage drainage for {crop}.",
        "How should excess water be drained from a {crop} field?",
        "What's the drainage advice for {crop}?",
        "How do I handle water drainage in {crop} cultivation?",
     ], [
        "{crop_bn} ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত?",
        "{crop_bn}-এর জমিতে পানি নিষ্কাশনের নিয়ম বলুন।",
        "আমি জানতে চাই {crop_bn} ক্ষেতে বাড়তি পানি কীভাবে বের করতে হয়।",
        "{crop_bn} খেতে বেশি পানি জমলে কী করমু?",
        "{crop_bn} ফসলের নিষ্কাশন ব্যবস্থাপনা কেমন হওয়া উচিত?",
        "জমিতে পানি জমে গেলে {crop_bn} ফসলে কী ব্যবস্থা নিতে হবে?",
     ]),
    ("15_মৌসুমী_শর্তাবলী", [
        "Are there seasonal conditions affecting irrigation for {crop}?",
        "Tell me about any seasonal conditions for irrigating {crop}.",
        "I want to know if the season affects irrigation for {crop}.",
        "Does the time of year change how {crop} should be irrigated?",
        "What seasonal factors matter for watering {crop}?",
        "How does the season impact {crop} irrigation?",
     ], [
        "{crop_bn} ফসলের সেচে মৌসুমী কোন শর্ত প্রযোজ্য?",
        "{crop_bn}-এর সেচে ঋতুভেদে কী পরিবর্তন হয় বলুন।",
        "আমি জানতে চাই {crop_bn} ফসলে মৌসুম অনুযায়ী সেচের কোনো শর্ত আছে কিনা।",
        "{crop_bn} খেতে ঋতু ভেদে সেচ কি অন্যরকম হয়?",
        "{crop_bn} ফসলের সেচে মৌসুমের প্রভাব কেমন?",
        "কোন ঋতুতে {crop_bn} ফসলের সেচ ভিন্ন হওয়া উচিত?",
     ]),
    ("19_শুষ্ক_ঋতু_কৌশল", [
        "What is the dry-season irrigation strategy for {crop}?",
        "Tell me how to irrigate {crop} during the dry season.",
        "I want to know the dry-season watering plan for {crop}.",
        "How should I water {crop} when it's dry outside?",
        "What's the irrigation approach for {crop} in dry weather?",
        "How is {crop} irrigated in the dry season?",
     ], [
        "{crop_bn} ফসলের শুষ্ক মৌসুমের সেচ কৌশল কী?",
        "{crop_bn}-এ শুষ্ক মৌসুমে কীভাবে সেচ দিতে হয় বলুন।",
        "আমি জানতে চাই শুষ্ক মৌসুমে {crop_bn} ফসলে সেচ কেমনে দিতে হয়।",
        "খরার সময় {crop_bn} খেতে পানি কেমনে দিমু?",
        "{crop_bn} ফসলে শুকনো মৌসুমে সেচের পদ্ধতি কী?",
        "শুষ্ক ঋতুতে {crop_bn} ফসলের সেচ ব্যবস্থাপনা কেমন?",
     ]),
    ("20_বর্ষা_ঋতু_কৌশল", [
        "What is the rainy-season irrigation strategy for {crop}?",
        "Tell me how to manage {crop} irrigation during the rainy season.",
        "I want to know the monsoon watering plan for {crop}.",
        "How should {crop} be irrigated when it's raining a lot?",
        "What's the irrigation approach for {crop} in the monsoon?",
        "How is {crop} irrigated during heavy rains?",
     ], [
        "{crop_bn} ফসলের বর্ষা মৌসুমের সেচ কৌশল কী?",
        "{crop_bn}-এ বর্ষাকালে সেচ ব্যবস্থাপনা কেমন বলুন।",
        "আমি জানতে চাই বর্ষা মৌসুমে {crop_bn} ফসলে সেচ কেমনে দিতে হয়।",
        "বৃষ্টির সময় {crop_bn} খেতে পানি দেওয়া লাগে কিনা?",
        "{crop_bn} ফসলে বর্ষাকালীন সেচের কৌশল কী?",
        "বর্ষা ঋতুতে {crop_bn} ফসলের সেচ পরিকল্পনা কেমন?",
     ]),
    ("23_মাটির_ধরন_প্রভাব", [
        "How does soil type affect irrigation for {crop}?",
        "Tell me how the soil type changes irrigation needs for {crop}.",
        "I want to know the effect of soil type on watering {crop}.",
        "Does the kind of soil matter for irrigating {crop}?",
        "How should soil type influence {crop} watering?",
        "What role does soil play in {crop} irrigation?",
     ], [
        "{crop_bn} ফসলের সেচে মাটির ধরনের প্রভাব কী?",
        "{crop_bn}-এর সেচে মাটির প্রকারভেদ কীভাবে প্রভাব ফেলে বলুন।",
        "আমি জানতে চাই মাটির ধরন {crop_bn} ফসলের সেচে কী প্রভাব ফেলে।",
        "মাটি ভেদে {crop_bn} খেতে পানি দেওয়া কি ভিন্ন হয়?",
        "{crop_bn} ফসলে মাটির প্রকার সেচকে কীভাবে প্রভাবিত করে?",
        "কোন মাটিতে {crop_bn} ফসলের সেচ বেশি লাগে?",
     ]),
]

def factual_pairs(doc):
    feat = doc["irrigation_features"]
    crop = doc["crop_name_en"]
    crop_bn = doc.get("crop_name_bn", crop)
    out = []
    for key, qens, qbns in FACTUAL_TEMPLATES:
        v = field(feat, key)
        if not usable(v):
            continue
        aen = v.get("value_en")
        abn = v.get("value_bn")
        src = {"file": doc["source_image"], "field": key,
               "provenance": v.get("provenance"),
               "source_quote_bn": v.get("source_quote_bn")}
        if aen:
            gid = group_id_for(crop, key, "en")
            for qen in qens:
                out.append(qa(qen.format(crop=crop), aen, crop, "factual", "en", src, "retrieved", gid))
        if abn:
            gid = group_id_for(crop, key, "bn")
            for qbn in qbns:
                out.append(qa(qbn.format(crop_bn=crop_bn), abn, crop, "factual", "bn", src, "retrieved", gid))
    return out

# --- (B) interval question from typed numeric day fields ---------------------
# Field 7 is the general irrigation interval; field 16 is specifically about
# the establishment stage — worded distinctly so they stay two separate facts
# instead of colliding into identical question text.
INTERVAL_TEMPLATES = {
    "7_সেচ_ব্যবধান_দিন": ([
        "How often should {crop} be irrigated?",
        "Tell me the irrigation interval for {crop}.",
        "I want to know how many days between irrigations for {crop}.",
        "How many days should I wait between watering {crop}?",
        "What's the gap between irrigations for {crop}?",
        "How frequently does {crop} need watering?",
     ], [
        "{crop_bn} ফসলে কত দিন পর পর সেচ দিতে হবে?",
        "{crop_bn}-এর সেচের ব্যবধান কত দিন বলুন।",
        "আমি জানতে চাই {crop_bn} ফসলে কত দিন অন্তর সেচ লাগে।",
        "{crop_bn} খেতে কয়দিন পরপর পানি দেওয়া লাগে?",
        "{crop_bn} ফসলে সেচের ফ্রিকোয়েন্সি কেমন?",
        "কতদিন পরপর {crop_bn} ফসলে পানি দিমু?",
     ]),
    "16_প্রতিষ্ঠা_পর্যায়_সেচ_ব্যবধান": ([
        "How often should {crop} be irrigated during the establishment stage?",
        "Tell me the irrigation interval for {crop} while it's getting established.",
        "I want to know how many days between waterings during {crop}'s establishment period.",
        "In the early establishment phase, how many days should I wait between irrigations for {crop}?",
        "What's the watering gap for {crop} right after planting, during establishment?",
        "How frequently does {crop} need irrigation while establishing?",
     ], [
        "{crop_bn} ফসলের প্রতিষ্ঠা পর্যায়ে কত দিন পর পর সেচ দিতে হবে?",
        "{crop_bn} প্রতিষ্ঠাকালীন সময়ে সেচের ব্যবধান কত দিন বলুন।",
        "আমি জানতে চাই {crop_bn} ফসলের প্রতিষ্ঠা পর্যায়ে কত দিন অন্তর সেচ লাগে।",
        "{crop_bn} নতুন লাগানোর পর প্রতিষ্ঠাকালে কয়দিন পরপর পানি দেওয়া লাগে?",
        "{crop_bn} ফসল প্রতিষ্ঠার সময় সেচের ফ্রিকোয়েন্সি কেমন?",
        "প্রতিষ্ঠা পর্যায়ে কতদিন পরপর {crop_bn} ফসলে পানি দিমু?",
     ]),
}

def interval_pairs(doc):
    feat = doc["irrigation_features"]
    crop = doc["crop_name_en"]; crop_bn = doc.get("crop_name_bn", crop)
    out = []
    for key in ("7_সেচ_ব্যবধান_দিন", "16_প্রতিষ্ঠা_পর্যায়_সেচ_ব্যবধান"):
        v = field(feat, key)
        if not usable(v):
            continue
        lo, hi = v.get("value_days_min"), v.get("value_days_max")
        if lo is None:
            continue
        rng_en = f"{lo}" if lo == hi else f"{lo}-{hi}"
        rng_bn = bn_num(lo) if lo == hi else f"{bn_num(lo)}-{bn_num(hi)}"
        cond_en = (" " + v["condition_en"]) if v.get("condition_en") else ""
        cond_bn = (v["condition_bn"] + " ") if v.get("condition_bn") else ""
        aen = f"Irrigate {crop} every {rng_en} days.{(' Condition:'+cond_en) if cond_en else ''}"
        abn = f"{cond_bn}{crop_bn} ফসলে প্রতি {rng_bn} দিন অন্তর সেচ দিতে হবে।"
        src = {"file": doc["source_image"], "field": key, "provenance": v.get("provenance"),
               "value_days_min": lo, "value_days_max": hi,
               "source_quote_bn": v.get("source_quote_bn")}
        qens, qbns = INTERVAL_TEMPLATES[key]
        gid_en = group_id_for(crop, key, "en")
        gid_bn = group_id_for(crop, key, "bn")
        for qen in qens:
            out.append(qa(qen.format(crop=crop), aen, crop, "factual", "en", src, "retrieved", gid_en))
        for qbn in qbns:
            out.append(qa(qbn.format(crop_bn=crop_bn), abn, crop, "factual", "bn", src, "retrieved", gid_bn))
    return out

# --- (C) runtime-conditioned numeric — the H1 anchor -------------------------
# These require Kc + ET0 + rainfall + area at runtime. RAG structurally cannot
# retrieve them; only the fine-tuned model + tool/formula can answer. We emit a
# worked, deterministic answer so the model learns the REASONING SHAPE, and we
# tag the exact inputs so the eval harness can regenerate gold on the fly.
# Kc itself is looked up per-crop from kc_values_hybrid.csv (KC_TABLE), never
# hardcoded — each tuple below only names which Kc stage the scenario uses.
RUNTIME_SCENARIOS = [
    # (kc_stage_key, et0_mm, rain_mm, area_val, area_unit, stage_en, stage_bn)
    ("kc_ini", 5.0, 0.0, 1, "acre",  "initial",     "প্রাথমিক"),
    ("kc_mid", 6.2, 8.0, 1, "bigha", "mid-season",  "মধ্য-মৌসুম"),
    ("kc_end", 4.8, 20.0, 0.5, "acre", "late-season", "শেষ-মৌসুম"),
]

RUNTIME_Q_TEMPLATES_EN = [
    "My {crop} field is at the {stg_en} stage. Today ET0 is {et0} mm, it rained {rain} mm, "
    "and the field is {aval} {aunit}. How much irrigation water does it need today?",
    "Tell me how much water my {crop} field needs today — it's at the {stg_en} stage, "
    "ET0 is {et0} mm, rainfall was {rain} mm, and the area is {aval} {aunit}.",
    "I want to know the irrigation amount for a {aval} {aunit} {crop} field at the {stg_en} "
    "stage, given ET0 of {et0} mm and {rain} mm of rain today.",
    "At the {stg_en} stage with ET0 {et0} mm and {rain} mm rain, how much should I water my "
    "{aval} {aunit} {crop} field today?",
    "My {crop} ({aval} {aunit}) is in its {stg_en} stage. It got {rain} mm rain and ET0 is "
    "{et0} mm — how much irrigation today?",
    "How much irrigation water does a {stg_en}-stage {crop} field of {aval} {aunit} need, "
    "with ET0 {et0} mm and {rain} mm of rainfall?",
]
RUNTIME_Q_TEMPLATES_BN = [
    "আমার {crop_bn} ক্ষেত {stg_bn} পর্যায়ে আছে। আজ ET0 {et0_bn} মিমি, বৃষ্টি হয়েছে {rain_bn} মিমি, "
    "জমির পরিমাণ {aval_bn} {aunit}। আজ কত পানি সেচ দিতে হবে?",
    "আমার {crop_bn} খেতে আজ কত পানি লাগবে বলুন — এটা {stg_bn} পর্যায়ে আছে, ET0 {et0_bn} মিমি, "
    "বৃষ্টি {rain_bn} মিমি, জমি {aval_bn} {aunit}।",
    "আমি জানতে চাই {aval_bn} {aunit} {crop_bn} খেতে {stg_bn} পর্যায়ে, ET0 {et0_bn} মিমি এবং আজ "
    "{rain_bn} মিমি বৃষ্টিতে কত সেচ লাগবে।",
    "{stg_bn} পর্যায়ে ET0 {et0_bn} মিমি আর বৃষ্টি {rain_bn} মিমি হলে আমার {aval_bn} {aunit} "
    "{crop_bn} খেতে আজ কতটুকু পানি দিমু?",
    "আমার {crop_bn} ({aval_bn} {aunit}) এখন {stg_bn} পর্যায়ে। বৃষ্টি হয়েছে {rain_bn} মিমি, "
    "ET0 {et0_bn} মিমি — আজ কত সেচ লাগবে?",
    "{stg_bn} পর্যায়ের {aval_bn} {aunit} {crop_bn} খেতে ET0 {et0_bn} মিমি ও বৃষ্টি {rain_bn} মিমি "
    "হলে কতটুকু সেচ লাগবে?",
]

def runtime_pairs(doc):
    feat = doc["irrigation_features"]
    crop = doc["crop_name_en"]; crop_bn = doc.get("crop_name_bn", crop)
    kc_row = KC_TABLE.get(normalize_crop_name(crop))
    if kc_row is None:
        if crop not in SKIPPED_KC_CROPS:
            SKIPPED_KC_CROPS.append(crop)
        return []
    out = []
    for stage_key, et0, rain, aval, aunit, stg_en, stg_bn in RUNTIME_SCENARIOS:
        kc = kc_row[stage_key]
        net_mm = net_irrigation_mm(kc, et0, rain)
        area_sqm = aval * AREA[aunit]
        liters = round(net_mm / 1000 * area_sqm * 1000, 1)  # mm→m depth × m² → liters
        eff = effective_rain(rain)
        a_en = (f"Net irrigation depth = Kc×ET0 − effective rainfall "
                f"= {kc}×{et0} − {eff} = {net_mm} mm. "
                f"For {aval} {aunit} ({area_sqm:.0f} m²), that is about {liters:.0f} liters today.")
        a_bn = (f"নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি "
                f"= {bn_num(kc)}×{bn_num(et0)} − {bn_num(eff)} = {bn_num(net_mm)} মিমি। "
                f"{bn_num(aval)} {aunit} ({area_sqm:.0f} বর্গমিটার)-এর জন্য আজ প্রায় {liters:.0f} লিটার।")
        trace = {"formula": "net_mm = Kc*ET0 - effective_rain(rain); liters = net_mm/1000*area_sqm*1000",
                 "inputs": {"kc": kc, "et0_mm": et0, "rain_mm": rain,
                            "area_value": aval, "area_unit": aunit, "stage": stg_en},
                 "computed": {"effective_rain_mm": eff, "net_mm": net_mm, "liters": liters},
                 "kc_stage": stage_key,
                 "kc_source": kc_row["kc_source"],
                 "citation": kc_row["citation"],
                 "note": "Kc is looked up per-crop/per-stage from kc_values_hybrid.csv (see kc_source/citation); "
                         "ET0/rain/area are worked-example scenario inputs; production ET0 comes from Open-Meteo."}
        gid_en = group_id_for(crop, stage_key, "en")
        gid_bn = group_id_for(crop, stage_key, "bn")
        for tmpl in RUNTIME_Q_TEMPLATES_EN:
            q_en = tmpl.format(crop=crop, stg_en=stg_en, et0=et0, rain=rain, aval=aval, aunit=aunit)
            out.append(qa(q_en, a_en, crop, "numeric_runtime", "en", trace, "computed", gid_en))
        for tmpl in RUNTIME_Q_TEMPLATES_BN:
            q_bn = tmpl.format(crop_bn=crop_bn, stg_bn=stg_bn, et0_bn=bn_num(et0), rain_bn=bn_num(rain),
                                aval_bn=bn_num(aval), aunit=aunit)
            out.append(qa(q_bn, a_bn, crop, "numeric_runtime", "bn", trace, "computed", gid_bn))
    return out

# --- (D) abstain / out-of-scope — teaches RAG hand-off -----------------------
# One topic is chosen at random per crop (as before), but that topic now gets
# a full list of paraphrases instead of a single fixed sentence.
ABSTAIN_TEMPLATES = [
    ("market_price", [
        "What is the market price of {crop} today?",
        "Tell me today's market price for {crop}.",
        "I want to know how much {crop} is selling for right now.",
        "What's the current price of {crop} in the market?",
        "How much does {crop} cost today?",
        "What rate is {crop} going for today?",
     ], [
        "{crop_bn} ফসলের আজকের বাজার দর কত?",
        "{crop_bn}-এর আজকের বাজারমূল্য বলুন।",
        "আমি জানতে চাই আজ {crop_bn} এর দাম কত।",
        "আইজকা {crop_bn} এর দাম কত যাইতেছে?",
        "বাজারে {crop_bn} এখন কত টাকা?",
        "{crop_bn} বেচলে আজ কত পামু?",
     ]),
    ("pesticide_brand", [
        "Which pesticide brand is cheapest for {crop}?",
        "Tell me the cheapest pesticide brand for {crop}.",
        "I want to know which pesticide brand costs least for {crop}.",
        "What's the most affordable pesticide for {crop}?",
        "Which pesticide brand should I buy cheaply for {crop}?",
        "What cheap pesticide works for {crop}?",
     ], [
        "{crop_bn} ফসলের জন্য সবচেয়ে সস্তা কীটনাশক ব্র্যান্ড কোনটি?",
        "{crop_bn}-এর জন্য কম দামি কীটনাশক ব্র্যান্ড বলুন।",
        "আমি জানতে চাই {crop_bn} ফসলে কোন কীটনাশক সস্তায় পাওয়া যায়।",
        "{crop_bn} খেতে কোন কীটনাশক সস্তা হইবো?",
        "সবচেয়ে কম দামে {crop_bn} এর জন্য কোন কীটনাশক ভালো?",
        "{crop_bn} ফসলে কোন কীটনাশক ব্র্যান্ড কিনলে খরচ কম হইবো?",
     ]),
    ("soil_ph", [
        "What is the exact soil pH at my farm for {crop}?",
        "Tell me the exact soil pH reading needed for {crop} on my farm.",
        "I want to know my farm's precise soil pH for growing {crop}.",
        "What's the soil pH on my specific plot for {crop}?",
        "Can you tell me the exact pH of my farm's soil for {crop}?",
        "What is my farm's soil pH level for {crop}?",
     ], [
        "আমার খামারে {crop_bn} ফসলের সঠিক মাটির pH কত?",
        "আমার জমির {crop_bn}-এর জন্য সঠিক মাটির pH বলুন।",
        "আমি জানতে চাই আমার খেতের মাটির pH ঠিক কত {crop_bn}-এর জন্য।",
        "আমার জমির মাটির pH কত, {crop_bn} চাষের জন্য?",
        "আমার খামারের মাটির সঠিক pH মান কত {crop_bn}-এর ক্ষেত্রে?",
        "{crop_bn} চাষে আমার জমির মাটির pH কতটুকু?",
     ]),
]
ABSTAIN_EN = ("The crop calendar does not specify this. I can only answer irrigation "
              "and cultivation questions grounded in the BARC calendar; please consult "
              "a market/soil source for this.")
ABSTAIN_BN = ("ফসল পঞ্জিকায় এই তথ্য নেই। আমি কেবল BARC পঞ্জিকা-ভিত্তিক সেচ ও চাষাবাদ "
              "সংক্রান্ত প্রশ্নের উত্তর দিতে পারি; এর জন্য বাজার/মাটি সংক্রান্ত উৎস দেখুন।")

def abstain_pairs(doc):
    crop = doc["crop_name_en"]; crop_bn = doc.get("crop_name_bn", crop)
    out = []
    topic_key, qens, qbns = random.choice(ABSTAIN_TEMPLATES)
    src = {"reason": "out_of_calendar_scope", "expected_behavior": "route_to_RAG_or_abstain"}
    gid_en = group_id_for(crop, topic_key, "en")
    gid_bn = group_id_for(crop, topic_key, "bn")
    for qen in qens:
        out.append(qa(qen.format(crop=crop), ABSTAIN_EN, crop, "abstain", "en", src, "abstain", gid_en))
    for qbn in qbns:
        out.append(qa(qbn.format(crop_bn=crop_bn), ABSTAIN_BN, crop, "abstain", "bn", src, "abstain", gid_bn))
    return out

# ----------------------------------------------------------------- main
def build(indir, outdir):
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(indir, "*.json")))
    all_pairs, review = [], []
    stats = collections.Counter()
    for path in files:
        doc = json.load(open(path, encoding="utf-8"))
        feat = doc.get("irrigation_features", {})
        # collect anything that needs a human before it can be used
        for k, v in feat.items():
            if isinstance(v, dict) and (v.get("needs_human_review") or v.get("numeric_backfill_declined")):
                review.append({"file": os.path.basename(path), "field": k,
                               "reason": v.get("numeric_backfill_declined") or "needs_human_review",
                               "quote": v.get("source_quote_bn")})
        pairs = (factual_pairs(doc) + interval_pairs(doc) +
                 runtime_pairs(doc) + abstain_pairs(doc))
        for p in pairs:
            stats[(p["question_type"], p["lang"])] += 1
        all_pairs.extend(pairs)

    # dedupe identical instructions
    seen, deduped = set(), []
    for p in all_pairs:
        k = p["messages"][0]["content"]
        if k in seen:
            continue
        seen.add(k); deduped.append(p)
    duplicates_dropped = len(all_pairs) - len(deduped)

    # --- group-aware stratified split ---------------------------------------
    # Collect groups and, for each, the single question_type all its members
    # share (a group is only ever produced by one of the four pair-builders
    # above, so this is a true invariant, not just "usually true").
    groups = collections.defaultdict(list)
    group_qtype = {}
    for p in deduped:
        gid = p["group_id"]
        groups[gid].append(p)
        group_qtype[gid] = p["question_type"]

    by_type_groups = collections.defaultdict(list)
    for gid, qtype in group_qtype.items():
        by_type_groups[qtype].append(gid)

    split_rng = random.Random(42)
    train_groups, val_groups, test_groups = [], [], []
    for qtype in sorted(by_type_groups):
        gids = sorted(by_type_groups[qtype])  # deterministic order before shuffling
        split_rng.shuffle(gids)
        n = len(gids); a_n, b_n = int(n * 0.8), int(n * 0.9)
        train_groups += gids[:a_n]; val_groups += gids[a_n:b_n]; test_groups += gids[b_n:]

    train_group_set, val_group_set, test_group_set = set(train_groups), set(val_groups), set(test_groups)

    train, val, test = [], [], []
    for gid in train_groups: train.extend(groups[gid])
    for gid in val_groups:   val.extend(groups[gid])
    for gid in test_groups:  test.extend(groups[gid])
    split_rng.shuffle(train); split_rng.shuffle(val); split_rng.shuffle(test)

    # --- leakage check: no group_id may appear in more than one split -------
    assert train_group_set.isdisjoint(val_group_set), "LEAKAGE CHECK FAILED: group(s) in both train and val"
    assert train_group_set.isdisjoint(test_group_set), "LEAKAGE CHECK FAILED: group(s) in both train and test"
    assert val_group_set.isdisjoint(test_group_set), "LEAKAGE CHECK FAILED: group(s) in both val and test"
    assert len(train_group_set) + len(val_group_set) + len(test_group_set) == len(group_qtype), \
        "LEAKAGE CHECK FAILED: group count mismatch after split"
    print("LEAKAGE CHECK PASSED")

    def dump(name, rows):
        with open(os.path.join(outdir, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("train.jsonl", train)
    dump("val.jsonl", val)
    dump("test.jsonl", test)
    dump("all.jsonl", deduped)
    with open(os.path.join(outdir, "needs_human_review.json"), "w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=False, indent=2)

    n_groups = len(group_qtype)
    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_files": len(files),
        "total_pairs": len(deduped),
        "duplicates_dropped": duplicates_dropped,
        "unique_groups": n_groups,
        "avg_paraphrases_per_group": round(len(deduped) / n_groups, 2) if n_groups else 0,
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "split_groups": {"train": len(train_group_set), "val": len(val_group_set), "test": len(test_group_set)},
        "by_type_lang": {f"{t}/{l}": c for (t, l), c in sorted(stats.items())},
        "gold_kind_counts": dict(collections.Counter(p["gold_kind"] for p in deduped)),
        "fields_needing_human_review": len(review),
    }
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if SKIPPED_KC_CROPS:
        print(f"WARNING: no Kc row in kc_values_hybrid.csv for {len(SKIPPED_KC_CROPS)} "
              f"crop(s) — numeric_runtime questions were NOT generated for: "
              f"{', '.join(sorted(SKIPPED_KC_CROPS))}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="./json")
    ap.add_argument("--out", dest="outdir", default="./qa_out")
    a = ap.parse_args()
    build(a.indir, a.outdir)
