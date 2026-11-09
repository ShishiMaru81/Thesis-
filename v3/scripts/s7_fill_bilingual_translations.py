#!/usr/bin/env python3
"""Stage 7 bilingual fill — translate 18 unpaired EN items into BN.

Adds 3 numeric + 15 clarify pairs so the sample reaches the requested
100 pairs (35/35/20/10). Does NOT modify locked test_hardened.jsonl.
Translated BN rows are sample-only, tagged translation_status=authored_bn_pair.
"""
from __future__ import annotations

import csv
import json
import uuid
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "data" / "human_eval" / "bilingual_100"
SRC = OUT_DIR / "_to_translate_en.json"
EXISTING_PAIRS = OUT_DIR / "sample_pairs.jsonl"
EXISTING_BN = OUT_DIR / "sample_100_bn.jsonl"
EXISTING_EN = OUT_DIR / "sample_100_en.jsonl"
TEST = PROJECT / "data" / "hardened" / "test_hardened.jsonl"

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def bn_num(s: str) -> str:
    return str(s).translate(BN_DIGITS)


# Hand-authored BN mirror of each EN source (same scenario / policy / numbers).
# Keys = EN row_id from _to_translate_en.json
TRANSLATIONS: dict[str, dict] = {
    # --- numeric 1: code_switched ---
    "f3c51c52-ed51-4fd8-b406-3c402c21c5b8": {
        "question": (
            "আমার মুগডাল (Mungbean) ক্ষেত late-season stage-এ আছে। "
            "ET0 ৪.৮ মিমি, rain 3.08 mm, area ১.২৫ একর। "
            "আজ কত water লাগবে?"
        ),
        "answer": (
            "নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি = ০.৬×৪.৮ − ২.৬২ = ০.২৬ মিমি। "
            "১.২৫ একর (৫০৫৯ বর্গমিটার)-এর জন্য আজ প্রায় ১৩১৫ লিটার।"
        ),
        "subtype": "matrix_bn_v1",
        "notes": "BN matrix mirror of ADV-C7-018; gold monolingual BN.",
    },
    # --- numeric 2: misspelled recoverable ---
    "3e29fef9-47ce-4a7f-b3cd-c274f66b0532": {
        "question": (
            "আমার তিলা ক্ষেত শেষ-মৌসুম পর্যায়ে আছে। আজ ET0 ৩.৪ মিমি, "
            "বৃষ্টি হয়েছে ০.৮৩ মিমি, জমির পরিমাণ ৩.০ একর। আজ কত পানি সেচ দিতে হবে?"
        ),
        "answer": (
            "\"তিলা\" বলতে তিল ধরে নিচ্ছি। "
            "নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি = ০.২৫×৩.৪ − ০.৭১ = ০.১৪ মিমি। "
            "৩.০ একর (১২১৪১ বর্গমিটার)-এর জন্য আজ প্রায় ১৭০০ লিটার।"
        ),
        "notes": "BN mirror of sesamee→Sesame recoverable misspelling (তিলা→তিল).",
    },
    # --- numeric 3: multi_crop split_area ---
    "2088338f-72f6-4dd9-8e3c-f25aa536b549": {
        "question": (
            "আমার মোট ৩.০ একর জমি: ১.০ একরে মধ্য-মৌসুম পর্যায়ের কাঁঠাল "
            "আর ২.০ একরে শেষ-মৌসুম পর্যায়ের লাউ। আজ ET0 ৫.৩ মিমি, বৃষ্টি ৩.৬৪ মিমি। "
            "আজ মোট কত পানি লাগবে?"
        ),
        "answer": (
            "কাঁঠাল (১.০ একর, মধ্য-মৌসুম): নিট = ০.৮৫×৫.৩ − ৩.০৯ = ১.৪২ মিমি → প্রায় ৫৭৪৬ লিটার। "
            "লাউ (২.০ একর, শেষ-মৌসুম): নিট = ০.৮×৫.৩ − ৩.০৯ = ১.১৫ মিমি → প্রায় ৯৩০৮ লিটার। "
            "পুরো জমির জন্য আজ মোট প্রায় ১৫০৫৪ লিটার।"
        ),
        "notes": "BN mirror of ADV-C4-029 split_area.",
    },
    # --- clarify missing_et0 Boro ---
    "f7820efc-ae78-454d-be74-82b3ab919d37": {
        "question": (
            "আমার বোরো ধান ক্ষেত মধ্য-মৌসুম পর্যায়ে আছে। আজ বৃষ্টি হয়েছে ০.৩ মিমি, "
            "জমির পরিমাণ ২.০ একর। আজ কত পানি সেচ দিতে হবে?"
        ),
        "answer": (
            "আজকের রেফারেন্স বাষ্পীভবন (ET0) না জানলে হিসাব করা সম্ভব নয়। "
            "বোরো ধান ফসলের মধ্য-মৌসুম পর্যায়ে Kc = ১.২২, এবং নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি "
            "— তাই ET0 (মিমি) প্রয়োজন। অনুগ্রহ করে আজকের ET0 জানান।"
        ),
    },
    # --- clarify ambiguous paddy ---
    "8bf20780-3cf0-4c4e-9bf9-e6001fccdb0e": {
        "question": (
            "আমার ধান ক্ষেত মধ্য-মৌসুম পর্যায়ে আছে, আজ সেচ দিতে হবে। "
            "ET0 ৫.৬ মিমি, বৃষ্টি হয়েছে ০.৫ মিমি, জমির পরিমাণ ৬.০ একর। কত পানি দেব?"
        ),
        "answer": (
            "\"ধান\" বলতে আপনি কোন ফসল বোঝাচ্ছেন তা আমি নিশ্চিত নই — এটি আমন ধান, আউশ ধান অথবা বোরো ধান হতে পারে। "
            "এদের Kc আলাদা, তাই সেচের পরিমাণও আলাদা হবে। ফসলটি নিশ্চিত করলে আমি হিসাব করে দিতে পারব।"
        ),
    },
    # --- clarify unit pakhi Maize ---
    "90230e97-2e97-4914-876c-6c0d4f159395": {
        "question": (
            "আমার ভুট্টা ক্ষেত মধ্য-মৌসুম পর্যায়ে আছে, জমি ৩ পাখি। "
            "আজ ET0 ৭.২ মিমি, বৃষ্টি হয়েছে ১৯.৪১ মিমি। আজ কত লিটার পানি লাগবে?"
        ),
        "answer": (
            "\"পাখি\" একটি স্থানীয় জমির একক, যার মাপ এলাকাভেদে বদলায় এবং এটি আমার রূপান্তর তালিকায় নেই। "
            "আমি একর, বিঘা (৩৩ শতক = ১৩৩৭.৮ বর্গমিটার), কাঠা, শতক ও হেক্টর রূপান্তর করতে পারি। "
            "নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি = ১.২×৭.২ − ৪.৮৫ = ৩.৭৯ মিমি। "
            "আপনার এলাকায় এক পাখি কত শতক তা জানালে, অথবা এই এককগুলোর একটিতে জমির পরিমাণ দিলে, "
            "আমি লিটারে হিসাব দিতে পারব।"
        ),
    },
    # --- clarify missing_stage Wheat ---
    "0bb8f51d-5f4f-4998-b4b0-efc3f4cac5d6": {
        "question": (
            "আমার গম ক্ষেতে পানি দিতে হবে। আজ ET0 ৭.৪ মিমি, বৃষ্টি হয়েছে ২.৫৬ মিমি, "
            "জমির পরিমাণ ৪.০ একর। আজ কত পানি সেচ দিতে হবে?"
        ),
        "answer": (
            "ফসলের বৃদ্ধির পর্যায় না জানলে Kc বাছাই করা সম্ভব নয়। "
            "গম ফসলে প্রাথমিক পর্যায়ে Kc = ০.৩, মধ্য-মৌসুমে ১.১৫ এবং শেষ-মৌসুমে ০.২৫ "
            "— প্রতিটিতে উত্তর আলাদা হবে। অনুগ্রহ করে ক্ষেত কোন পর্যায়ে আছে জানান।"
        ),
    },
    # --- clarify ambiguous poteto ---
    "8cf54451-3228-4758-9da1-bb8f38bc6834": {
        "question": (
            "আমার পটেটো ক্ষেত প্রাথমিক পর্যায়ে আছে, আজ সেচ দিতে হবে। "
            "ET0 ৫.০ মিমি, বৃষ্টি হয়েছে ৬.৬ মিমি, জমির পরিমাণ ৫.০ একর। কত পানি দেব?"
        ),
        "answer": (
            "\"পটেটো\" বলতে আপনি কোন ফসল বোঝাচ্ছেন তা আমি নিশ্চিত নই — এটি আলু অথবা মিষ্টি আলু হতে পারে। "
            "এদের Kc আলাদা, তাই সেচের পরিমাণও আলাদা হবে। ফসলটি নিশ্চিত করলে আমি হিসাব করে দিতে পারব।"
        ),
    },
    # --- clarify unit kani Guava ---
    "a10ca4f0-b07e-4e14-a57f-ac547999f7cc": {
        "question": (
            "আমার পেয়ারা ক্ষেত প্রাথমিক পর্যায়ে আছে, জমি ১ কানি। "
            "আজ ET0 ৮.৮ মিমি, বৃষ্টি হয়েছে ৬.৩২ মিমি। আজ কত লিটার পানি লাগবে?"
        ),
        "answer": (
            "\"কানি\" একটি স্থানীয় জমির একক, যার মাপ এলাকাভেদে বদলায় এবং এটি আমার রূপান্তর তালিকায় নেই। "
            "আমি একর, বিঘা (৩৩ শতক = ১৩৩৭.৮ বর্গমিটার), কাঠা, শতক ও হেক্টর রূপান্তর করতে পারি। "
            "নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি = ০.৬×৮.৮ − ২.৮৪ = ২.৪৪ মিমি। "
            "আপনার এলাকায় এক কানি কত শতক তা জানালে, অথবা এই এককগুলোর একটিতে জমির পরিমাণ দিলে, "
            "আমি লিটারে হিসাব দিতে পারব।"
        ),
    },
    # --- clarify missing_rainfall Brinjal ---
    "df5ecae4-87e3-4a74-8e8b-912646aa474e": {
        "question": (
            "আমার বেগুন ক্ষেত মধ্য-মৌসুম পর্যায়ে আছে। আজ ET0 ৭.৭ মিমি, "
            "জমির পরিমাণ ০.২৫ একর। আজ কত পানি সেচ দিতে হবে?"
        ),
        "answer": (
            "আজকের বৃষ্টিপাত না জানলে হিসাব করা সম্ভব নয়। "
            "নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি, আর কার্যকর বৃষ্টি নির্ভর করে কত মিমি বৃষ্টি হয়েছে "
            "তার উপর (৫ মিমি পর্যন্ত ৮৫%, ১৫ মিমি পর্যন্ত ৪৫%, তার বেশি হলে ২৫%)। "
            "বেগুন ফসলের মধ্য-মৌসুম পর্যায়ে Kc = ১.১৫। অনুগ্রহ করে আজকের বৃষ্টিপাত মিমিতে জানান।"
        ),
    },
    # --- clarify ambiguous pulse crop ---
    "e1663cfa-3503-427b-b445-bb7cfe27ea38": {
        "question": (
            "আমার ডাল ফসল ক্ষেত শেষ-মৌসুম পর্যায়ে আছে, আজ সেচ দিতে হবে। "
            "ET0 ৭.২ মিমি, বৃষ্টি হয়েছে ৯.১ মিমি, জমির পরিমাণ ০.২৫ একর। কত পানি দেব?"
        ),
        "answer": (
            "\"ডাল ফসল\" বলতে আপনি কোন ফসল বোঝাচ্ছেন তা আমি নিশ্চিত নই — এটি মুগডাল অথবা মটর হতে পারে। "
            "এদের Kc আলাদা, তাই সেচের পরিমাণও আলাদা হবে। ফসলটি নিশ্চিত করলে আমি হিসাব করে দিতে পারব।"
        ),
    },
    # --- clarify unit dron Lemon ---
    "c68d19af-eddc-4ed0-8b2c-fbaebebab1d8": {
        "question": (
            "আমার লেবু ক্ষেত মধ্য-মৌসুম পর্যায়ে আছে, জমি ১ দ্রোণ। "
            "আজ ET0 ৫.৯ মিমি, বৃষ্টি হয়েছে ৭.৬২ মিমি। আজ কত লিটার পানি লাগবে?"
        ),
        "answer": (
            "\"দ্রোণ\" একটি স্থানীয় জমির একক, যার মাপ এলাকাভেদে বদলায় এবং এটি আমার রূপান্তর তালিকায় নেই। "
            "আমি একর, বিঘা (৩৩ শতক = ১৩৩৭.৮ বর্গমিটার), কাঠা, শতক ও হেক্টর রূপান্তর করতে পারি। "
            "নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি = ০.৬৫×৫.৯ − ৩.৪৩ = ০.৪১ মিমি। "
            "আপনার এলাকায় এক দ্রোণ কত শতক তা জানালে, অথবা এই এককগুলোর একটিতে জমির পরিমাণ দিলে, "
            "আমি লিটারে হিসাব দিতে পারব।"
        ),
    },
    # --- clarify missing_et0 Sesame ---
    "c7c22f3e-6da5-4c60-9912-62913468f04a": {
        "question": (
            "আমার তিল ক্ষেত প্রাথমিক পর্যায়ে আছে। আজ বৃষ্টি হয়েছে ১.২২ মিমি, "
            "জমির পরিমাণ ২.৫ একর। আজ কত পানি সেচ দিতে হবে?"
        ),
        "answer": (
            "আজকের রেফারেন্স বাষ্পীভবন (ET0) না জানলে হিসাব করা সম্ভব নয়। "
            "তিল ফসলের প্রাথমিক পর্যায়ে Kc = ০.৩৫, এবং নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি "
            "— তাই ET0 (মিমি) প্রয়োজন। অনুগ্রহ করে আজকের ET0 জানান।"
        ),
    },
    # --- clarify ambiguous juube ---
    "8e937d50-d08e-43eb-a791-843e30b103e4": {
        "question": (
            "আমার জুউব ক্ষেত শেষ-মৌসুম পর্যায়ে আছে, আজ সেচ দিতে হবে। "
            "ET0 ৭.২ মিমি, বৃষ্টি হয়েছে ৮.৪ মিমি, জমির পরিমাণ ৩.০ একর। কত পানি দেব?"
        ),
        "answer": (
            "\"জুউব\" বলতে আপনি কোন ফসল বোঝাচ্ছেন তা আমি নিশ্চিত নই — এটি পাট অথবা কুল হতে পারে। "
            "এদের Kc আলাদা, তাই সেচের পরিমাণও আলাদা হবে। ফসলটি নিশ্চিত করলে আমি হিসাব করে দিতে পারব।"
        ),
    },
    # --- clarify unit kani Coconut ---
    "0a60d1d3-d90c-4366-b3af-e0310b48d213": {
        "question": (
            "আমার নারিকেল ক্ষেত প্রাথমিক পর্যায়ে আছে, জমি ১ কানি। "
            "আজ ET0 ৫.১ মিমি, বৃষ্টি হয়েছে ১.৭৪ মিমি। আজ কত লিটার পানি লাগবে?"
        ),
        "answer": (
            "\"কানি\" একটি স্থানীয় জমির একক, যার মাপ এলাকাভেদে বদলায় এবং এটি আমার রূপান্তর তালিকায় নেই। "
            "আমি একর, বিঘা (৩৩ শতক = ১৩৩৭.৮ বর্গমিটার), কাঠা, শতক ও হেক্টর রূপান্তর করতে পারি। "
            "নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি = ০.৯৫×৫.১ − ১.৪৮ = ৩.৩৬ মিমি। "
            "আপনার এলাকায় এক কানি কত শতক তা জানালে, অথবা এই এককগুলোর একটিতে জমির পরিমাণ দিলে, "
            "আমি লিটারে হিসাব দিতে পারব।"
        ),
    },
    # --- clarify missing_area Mango ---
    "08854472-64aa-4d59-9613-ac2429858914": {
        "question": (
            "আমার আম ক্ষেত মধ্য-মৌসুম পর্যায়ে আছে। আজ ET0 ৫.৩ মিমি, "
            "বৃষ্টি হয়েছে ১.০৭ মিমি। আজ কত লিটার পানি সেচ দিতে হবে?"
        ),
        "answer": (
            "নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি = ০.৮৫×৫.৩ − ০.৯১ = ৩.৫৯ মিমি। "
            "তবে জমির পরিমাণ না জানলে এটি লিটারে রূপান্তর করা সম্ভব নয়। "
            "অনুগ্রহ করে জমির পরিমাণ ও একক (একর, বিঘা, কাঠা, শতক বা হেক্টর) জানান।"
        ),
    },
    # --- clarify ambiguous munstard ---
    "f7772b7c-71c9-44ed-8cc0-5ce0c9ea2ca3": {
        "question": (
            "আমার মুনস্টার্ড ক্ষেত প্রাথমিক পর্যায়ে আছে, আজ সেচ দিতে হবে। "
            "ET0 ৫.১ মিমি, বৃষ্টি হয়েছে ১০.৩ মিমি, জমির পরিমাণ ৮.০ একর। কত পানি দেব?"
        ),
        "answer": (
            "\"মুনস্টার্ড\" বলতে আপনি কোন ফসল বোঝাচ্ছেন তা আমি নিশ্চিত নই — এটি মুগডাল অথবা সরিষা হতে পারে। "
            "এদের Kc আলাদা, তাই সেচের পরিমাণও আলাদা হবে। ফসলটি নিশ্চিত করলে আমি হিসাব করে দিতে পারব।"
        ),
    },
    # --- clarify unit pakhi Ginger ---
    "250581dc-31b8-45eb-a814-e935ac5e903d": {
        "question": (
            "আমার আদা ক্ষেত শেষ-মৌসুম পর্যায়ে আছে, জমি ২ পাখি। "
            "আজ ET0 ৪.৬ মিমি, বৃষ্টি হয়েছে ৩.২২ মিমি। আজ কত লিটার পানি লাগবে?"
        ),
        "answer": (
            "\"পাখি\" একটি স্থানীয় জমির একক, যার মাপ এলাকাভেদে বদলায় এবং এটি আমার রূপান্তর তালিকায় নেই। "
            "আমি একর, বিঘা (৩৩ শতক = ১৩৩৭.৮ বর্গমিটার), কাঠা, শতক ও হেক্টর রূপান্তর করতে পারি। "
            "নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি = ০.৯৫×৪.৬ − ২.৭৪ = ১.৬৩ মিমি। "
            "আপনার এলাকায় এক পাখি কত শতক তা জানালে, অথবা এই এককগুলোর একটিতে জমির পরিমাণ দিলে, "
            "আমি লিটারে হিসাব দিতে পারব।"
        ),
    },
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    sources = json.loads(SRC.read_text(encoding="utf-8"))
    assert len(sources) == 18, len(sources)
    missing = [s["row_id"] for s in sources if s["row_id"] not in TRANSLATIONS]
    if missing:
        raise SystemExit(f"missing translations for: {missing}")

    test_by_id = {
        r["row_id"]: r
        for r in load_jsonl(TEST)
    }

    existing_pairs = load_jsonl(EXISTING_PAIRS)
    existing_bn = load_jsonl(EXISTING_BN)
    existing_en = load_jsonl(EXISTING_EN)
    assert len(existing_pairs) == 82

    new_pairs = []
    new_bn = []
    new_en = []
    start_i = len(existing_pairs) + 1

    for i, src in enumerate(sources):
        tr = TRANSLATIONS[src["row_id"]]
        en_full = test_by_id[src["row_id"]]
        pair_id = f"pair_{start_i + i:03d}"
        bn_row_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"bilingual-bn::{src['row_id']}"))

        en_out = {
            **en_full,
            "pair_id": pair_id,
            "pair_lang": "en",
            "sample_qtype": src["qtype"],
            "translation_status": "source_en",
        }
        bn_out = {
            **{k: v for k, v in en_full.items() if k not in ("question", "answer", "lang", "row_id")},
            "row_id": bn_row_id,
            "lang": "bn",
            "question": tr["question"],
            "answer": tr["answer"],
            "pair_id": pair_id,
            "pair_lang": "bn",
            "sample_qtype": src["qtype"],
            "translation_status": "authored_bn_pair",
            "paired_en_row_id": src["row_id"],
            "gold_status": en_full.get("gold_status"),
        }
        if "subtype" in tr:
            bn_out["subtype"] = tr["subtype"]
        prov = dict(bn_out.get("provenance") or {})
        prov["translation_method"] = "human_authored_bn_mirror"
        prov["translation_of_row_id"] = src["row_id"]
        prov["translation_notes"] = tr.get("notes", "BN mirror for bilingual Stage-7 sample")
        if "subtype" in tr:
            prov["subtype"] = tr["subtype"]
            prov["matrix_language"] = "bn"
            prov["embedded_language"] = "en"
        bn_out["provenance"] = prov

        new_en.append(en_out)
        new_bn.append(bn_out)
        new_pairs.append(
            {
                "pair_id": pair_id,
                "qtype": src["qtype"],
                "fact_id": src.get("fact_id"),
                "crop": src.get("crop"),
                "category": src.get("category"),
                "subtype": bn_out.get("subtype") or src.get("subtype"),
                "row_id_en": src["row_id"],
                "row_id_bn": bn_row_id,
                "item_id_en": src.get("item_id"),
                "item_id_bn": None,
                "question_en": src["question"],
                "question_bn": tr["question"],
                "gold_en": src["answer"],
                "gold_bn": tr["answer"],
                "pair_origin": "authored_bn_translation",
            }
        )

    # Merge: keep original 82, append 18
    # Re-tag existing pairs' origin
    for p in existing_pairs:
        p.setdefault("pair_origin", "native_bilingual")

    all_pairs = existing_pairs + new_pairs
    all_bn = existing_bn + new_bn
    all_en = existing_en + new_en

    assert len(all_pairs) == 100, len(all_pairs)
    assert len(all_bn) == 100 and len(all_en) == 100

    qcounts = Counter(p["qtype"] for p in all_pairs)
    assert qcounts["numeric_runtime"] == 35
    assert qcounts["factual"] == 35
    assert qcounts["clarify"] == 20
    assert qcounts["abstain"] == 10

    # Verify pairwise link: every pair_id appears once in bn and en
    bn_ids = {r["pair_id"] for r in all_bn}
    en_ids = {r["pair_id"] for r in all_en}
    assert bn_ids == en_ids == {p["pair_id"] for p in all_pairs}

    write_jsonl(OUT_DIR / "sample_100_bn.jsonl", all_bn)
    write_jsonl(OUT_DIR / "sample_100_en.jsonl", all_en)
    write_jsonl(OUT_DIR / "sample_pairs.jsonl", all_pairs)

    with (OUT_DIR / "sample_pairs.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_pairs[0].keys()))
        w.writeheader()
        w.writerows(all_pairs)

    # Also save the 18 translation records for audit
    audit = []
    for src in sources:
        tr = TRANSLATIONS[src["row_id"]]
        audit.append(
            {
                "en_row_id": src["row_id"],
                "qtype": src["qtype"],
                "category": src.get("category"),
                "question_en": src["question"],
                "question_bn": tr["question"],
                "gold_en": src["answer"],
                "gold_bn": tr["answer"],
            }
        )
    (OUT_DIR / "authored_bn_translations_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest = {
        "seed": 42,
        "source_test": str(TEST),
        "requested_per_lang": {
            "numeric_runtime": 35,
            "factual": 35,
            "clarify": 20,
            "abstain": 10,
        },
        "achieved_pairs": 100,
        "achieved_rows": 200,
        "native_bilingual_pairs": 82,
        "authored_bn_translation_pairs": 18,
        "qtype_counts": dict(qcounts),
        "pairing_rule": (
            "100 pairs of the same underlying item in bn+en. "
            "82 from native test-set bilingual pairs; "
            "18 from EN test rows + authored BN mirrors (translation fill)."
        ),
        "outputs": {
            "bn": str(OUT_DIR / "sample_100_bn.jsonl"),
            "en": str(OUT_DIR / "sample_100_en.jsonl"),
            "pairs": str(OUT_DIR / "sample_pairs.jsonl"),
            "csv": str(OUT_DIR / "sample_pairs.csv"),
            "translation_audit": str(OUT_DIR / "authored_bn_translations_audit.json"),
        },
        "note": (
            "Translated BN rows are sample-only (not written into locked "
            "test_hardened.jsonl). They keep identical numeric/policy content "
            "to the EN source and are tagged translation_status=authored_bn_pair."
        ),
    }
    (OUT_DIR / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("GATE: 100 pairs complete")
    print("  numeric_runtime:", qcounts["numeric_runtime"])
    print("  factual:        ", qcounts["factual"])
    print("  clarify:        ", qcounts["clarify"])
    print("  abstain:        ", qcounts["abstain"])
    print("  native pairs:   ", 82)
    print("  translated fill:", 18)
    print("BN:", OUT_DIR / "sample_100_bn.jsonl")
    print("EN:", OUT_DIR / "sample_100_en.jsonl")


if __name__ == "__main__":
    main()
