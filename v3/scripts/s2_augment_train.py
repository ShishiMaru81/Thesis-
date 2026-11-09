#!/usr/bin/env python3
"""
s2_augment_train.py — v3 STAGE 2: augment the training split, and only it.

INPUT   data/splits/train_seed.jsonl
OUTPUT  data/augmented/train_augmented.jsonl

HARD RULE 1 — the held-out splits are never opened
--------------------------------------------------
Three independent mechanisms, all active on every run:

  1. A static self-scan of this file's own source. The two held-out
     filenames are assembled at runtime from their stems, so the literals do
     not appear anywhere in this script. If either literal is ever typed into
     the source, the scan finds it and the script raises before doing
     anything.
  2. A runtime guard wrapping builtins.open, io.open and pathlib.Path.open.
     Any attempt to open a file whose basename is a held-out split raises
     HeldOutAccessError, including from inside a library.
  3. Family C needs to reject scenario_hashes that exist in the held-out
     splits, which looks like it requires reading them. It does not:
     s1b_export_heldout_hashes.py is the one place those files are opened,
     and it exports a blocklist containing scenario_hash strings and nothing
     else. This script reads only that derived blocklist.

FOUR FAMILIES
-------------
  A paraphrase       2 per seed question, Gemini, temperature 0.8
  B backtranslation  1 per seed question, round trip through the other language
  C synthetic        new numeric scenarios, gold from irrigation_calc
  D noise            character typos in crop and unit words, 10% of the pool

Every generated row carries "origin" and "parent_row_id". Seed rows keep
origin "seed".

APPROVED DECISIONS APPLIED HERE
------------------------------
  Decision 2  Family C rejects zero-gold scenarios until a 70% non-zero quota
              is filled; the remaining 30% may be zero (a real drought
              signal). Seed numeric rows that compute to zero are KEPT and
              flagged "zero_gold": true rather than dropped.
  Decision 4  All text comparison goes through bn_normalize v3.1. The v3.0
              regex destroyed Bengali matras; the entity-preservation checks
              in families A and B would have been meaningless under it,
              because তিল and তুলা compared equal.

SEEDING
-------
seed=42 for every stream. row_id generation uses a stream seeded with the
string "s2-rowid-42" rather than the integer 42, because Stage 0's row_ids
came from random.Random(42) and reusing that stream would regenerate the
same UUIDs. Asserted explicitly at the end of build().

Run:  python s2_augment_train.py                 (all four families)
      python s2_augment_train.py --families CD   (no API calls)
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# HARD RULE 1 GUARD — installed before anything else can touch the filesystem
# --------------------------------------------------------------------------
import builtins
import io
import pathlib

_HELD_OUT_BASENAMES = tuple(stem + "." + "jsonl" for stem in ("val", "test"))


class HeldOutAccessError(RuntimeError):
    """Raised on any attempt to read a frozen held-out split from Stage 2."""


def _check_path(p) -> None:
    base = str(p).replace("\\", "/").rsplit("/", 1)[-1].lower()
    if base in _HELD_OUT_BASENAMES:
        raise HeldOutAccessError(
            f"HARD RULE 1 VIOLATION: Stage 2 attempted to open {base!r}. "
            f"The held-out splits are frozen. Family C's scenario constraint "
            f"is served by data/splits/heldout_scenario_hashes.json, which is "
            f"produced by s1b_export_heldout_hashes.py.")


_real_open = builtins.open


def _guarded_open(file, *args, **kwargs):
    _check_path(file)
    return _real_open(file, *args, **kwargs)


_real_path_open = pathlib.Path.open


def _guarded_path_open(self, *args, **kwargs):
    _check_path(self)
    return _real_path_open(self, *args, **kwargs)


builtins.open = _guarded_open
io.open = _guarded_open
pathlib.Path.open = _guarded_path_open


def _selfscan_source() -> None:
    """Mechanism 1. Uses the unguarded opener so it can read this file."""
    src = _real_open(__file__, encoding="utf-8").read()
    for name in _HELD_OUT_BASENAMES:
        if name in src:
            raise HeldOutAccessError(
                f"HARD RULE 1 VIOLATION: the literal {name!r} appears in "
                f"{__file__}. Stage 2 must not name the held-out splits.")


_selfscan_source()

# --------------------------------------------------------------------------
import argparse            # noqa: E402
import collections         # noqa: E402
import copy                # noqa: E402
import datetime            # noqa: E402
import hashlib             # noqa: E402
import json                # noqa: E402
import math                # noqa: E402
import os                  # noqa: E402
import random              # noqa: E402
import re                  # noqa: E402
import sys                 # noqa: E402
import time                # noqa: E402
import unicodedata         # noqa: E402
import uuid                # noqa: E402
from pathlib import Path   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import irrigation_calc as ic                              # noqa: E402
from bn_normalize import NORMALIZER_VERSION, normalize, run_regression_tests  # noqa: E402

SEED = 42
NUMERIC_TARGET = 3000
NONZERO_QUOTA = 0.70
NOISE_FRACTION = 0.10
PARAPHRASES_PER_ROW = 2
PEARSON_TOLERANCE = 0.05
PEARSON_MAX_ATTEMPTS = 8

GEMINI_MODEL = "gemini-2.5-flash"
PARAPHRASE_TEMPERATURE = 0.8
# Back-translation runs cooler than paraphrase on purpose. The round trip is
# already a lexical perturbation; at 0.8 the extra drift mostly corrupts
# numbers and crop names, which the entity check then discards, so the calls
# are billed and thrown away. 0.5 keeps enough variation to be useful.
TRANSLATE_TEMPERATURE = 0.5

V3_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SEED = V3_ROOT / "data" / "splits" / "train_seed.jsonl"
BLOCKLIST = V3_ROOT / "data" / "splits" / "heldout_scenario_hashes.json"
OUT_PATH = V3_ROOT / "data" / "augmented" / "train_augmented.jsonl"
GATE_JSON = V3_ROOT / "reports" / "gate2_augment.json"
LOG_PATH = V3_ROOT / "logs" / "s2_augment_train.log"
CACHE_PATH = V3_ROOT / "logs" / "gemini_cache.jsonl"

AREA_UNIT_BN = {"acre": "একর", "bigha": "বিঘা", "katha": "কাঠা",
                "decimal": "শতক", "hectare": "হেক্টর", "sqm": "বর্গমিটার"}
# Decision-approved unit set for family C. sqm is deliberately absent: it is
# not a unit a Bangladeshi farmer states a plot in.
SYNTH_AREA_UNITS = ("acre", "bigha", "katha", "decimal", "hectare")
RAIN_TIERS = (((0.0, 5.0), 0.40), ((5.0, 15.0), 0.35), ((15.0, 60.0), 0.25))

BD = "০১২৩৪৫৬৭৮৯"


def bn_num(n) -> str:
    return "".join(BD[int(c)] if c.isdigit() else c for c in str(n))


class SeededIds:
    """uuid4-format ids from a stream that cannot collide with Stage 0's."""

    def __init__(self, tag: str):
        self._rng = random.Random(tag)

    def next(self) -> str:
        raw = bytes(self._rng.getrandbits(8) for _ in range(16))
        return str(uuid.UUID(bytes=raw, version=4))


def scenario_hash(crop, area_unit, area_value, et0, rain, stage) -> str:
    parts = sorted([f"crop={crop}", f"area_unit={area_unit}",
                    f"area_value={area_value}", f"et0={et0}",
                    f"rain={rain}", f"stage={stage}"])
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


# ==========================================================================
# ENTITY PRESERVATION — explicit code, per the Stage 2 requirement
# ==========================================================================
UNIT_TOKENS = {
    "en": ["mm", "cm", "days", "day", "liters", "litres", "ET0", "Kc",
           "acre", "bigha", "katha", "decimal", "hectare", "sqm", "m"],
    "bn": ["মিমি", "সেমি", "মিটার", "বর্গমিটার", "দিন", "লিটার", "ET0", "Kc",
           "একর", "বিঘা", "কাঠা", "শতক", "হেক্টর"],
}
STAGE_TOKENS = {
    "en": ["initial", "mid-season", "late-season", "establishment"],
    "bn": ["প্রাথমিক", "মধ্য-মৌসুম", "শেষ-মৌসুম", "প্রতিষ্ঠা"],
}


def _num_multiset(norm_text: str) -> collections.Counter:
    """Digit runs in normalized text. bn_normalize has already mapped Bengali
    digits to ASCII, so ৬.৩ and 6.3 both arrive as the tokens 6 and 3 and
    compare equal across languages."""
    return collections.Counter(re.findall(r"\d+", norm_text))


def extract_entities(row: dict) -> dict:
    """Everything in the ORIGINAL question that a paraphrase must carry over."""
    lang = row["lang"]
    nq = normalize(row["question"], lang)
    crop_surface = row["crop"] if lang == "en" else row.get("crop_bn", row["crop"])
    ncrop = normalize(crop_surface, lang)
    return {
        "crop_norm": ncrop,
        "crop_required": bool(ncrop) and ncrop in nq,
        "units": [u for u in UNIT_TOKENS[lang] if normalize(u, lang) in nq],
        "stages": [s for s in STAGE_TOKENS[lang] if normalize(s, lang) in nq],
        "numbers": _num_multiset(nq),
        "norm_original": nq,
    }


def entities_preserved(ents: dict, candidate: str, lang: str) -> tuple[bool, str]:
    """Returns (ok, reason). Every failure reason is recorded at Gate 2."""
    if not candidate or not candidate.strip():
        return False, "empty"
    nc = normalize(candidate, lang)
    if not nc:
        return False, "empty_after_normalization"
    if ents["crop_required"] and ents["crop_norm"] not in nc:
        return False, "crop_name_lost"
    for u in ents["units"]:
        if normalize(u, lang) not in nc:
            return False, f"unit_lost:{u}"
    for s in ents["stages"]:
        if normalize(s, lang) not in nc:
            return False, f"stage_lost:{s}"
    if _num_multiset(nc) != ents["numbers"]:
        return False, "numbers_changed"
    if nc == ents["norm_original"]:
        return False, "identical_to_source"
    return True, "ok"


# ==========================================================================
# GEMINI
# ==========================================================================
class MissingApiKey(RuntimeError):
    pass


class GeminiClient:
    """Thin wrapper with an on-disk response cache.

    The cache is keyed by (model, temperature, prompt) and makes a re-run free.
    Families A and B issue roughly 1,900 calls on a full run; without the cache
    every debugging pass would re-bill all of them.
    """

    def __init__(self, model=GEMINI_MODEL, cache_path: Path = CACHE_PATH,
                 max_retries: int = 4):
        key = (os.environ.get("GEMINI_API_KEY")
               or os.environ.get("GOOGLE_API_KEY")
               or os.environ.get("GOOGLE_AI_STUDIO_KEY"))
        if not key:
            raise MissingApiKey(
                "No Gemini key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) "
                "in the environment. Families C and D need no key and can be "
                "run now with:  python s2_augment_train.py --families CD")
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=key)
        self.model = model
        self.max_retries = max_retries
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        self.stats = collections.Counter()
        if cache_path.exists():
            for line in _real_open(cache_path, encoding="utf-8"):
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["k"]] = rec["v"]

    def _key(self, prompt: str, temperature: float, json_mode: bool) -> str:
        blob = f"{self.model}|{temperature}|{int(json_mode)}|{prompt}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, temperature: float,
                 json_mode: bool = False) -> str | None:
        k = self._key(prompt, temperature, json_mode)
        if k in self.cache:
            self.stats["cache_hit"] += 1
            return self.cache[k]

        from google.genai import types
        cfg = {"temperature": temperature}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = list[str]

        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt,
                    config=types.GenerateContentConfig(**cfg))
                text = (resp.text or "").strip()
                if not text:
                    raise RuntimeError("empty response")
                self.cache[k] = text
                with _real_open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "v": text}, ensure_ascii=False) + "\n")
                self.stats["api_call"] += 1
                return text
            except Exception as exc:                       # noqa: BLE001
                self.stats[f"retry:{type(exc).__name__}"] += 1
                if attempt == self.max_retries - 1:
                    self.stats["failed"] += 1
                    return None
                time.sleep(delay)
                delay *= 2
        return None


LANG_NAME = {"en": "English", "bn": "Bengali"}

PARAPHRASE_PROMPT = """You are building a bilingual agricultural question-answering dataset for Bangladeshi farmers.

Rewrite the QUESTION below in exactly {n} different ways.

Rules, all mandatory:
- Write in {lang_name}. Do not switch language or script.
- Preserve VERBATIM: the crop name, every number, every unit word, and the growth stage name.
- Do not answer the question and do not add or remove any information.
- Vary the sentence structure and the register: one more formal, one more colloquial.
- Each rewrite must differ from the original and from each other.

Return a JSON array of exactly {n} strings and nothing else.

QUESTION: {question}"""

TRANSLATE_PROMPT = """Translate the following {src_name} text into {tgt_name}.

Rules:
- Return ONLY the translation. No commentary, no quotes, no explanation.
- Preserve every number, unit, crop name and growth stage name exactly.
- It is a farmer's question. Keep it a natural question in {tgt_name}.

TEXT: {text}"""


def gemini_paraphrases(client, row, n) -> list[str]:
    raw = client.generate(
        PARAPHRASE_PROMPT.format(n=n, lang_name=LANG_NAME[row["lang"]],
                                 question=row["question"]),
        temperature=PARAPHRASE_TEMPERATURE, json_mode=True)
    if not raw:
        return []
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [s for s in out if isinstance(s, str)] if isinstance(out, list) else []


def gemini_roundtrip(client, row) -> str | None:
    src, tgt = row["lang"], ("en" if row["lang"] == "bn" else "bn")
    mid = client.generate(
        TRANSLATE_PROMPT.format(src_name=LANG_NAME[src], tgt_name=LANG_NAME[tgt],
                                text=row["question"]),
        temperature=TRANSLATE_TEMPERATURE)
    if not mid:
        return None
    return client.generate(
        TRANSLATE_PROMPT.format(src_name=LANG_NAME[tgt], tgt_name=LANG_NAME[src],
                                text=mid.strip()),
        temperature=TRANSLATE_TEMPERATURE)


# ==========================================================================
# FAMILY C — synthetic numeric scenarios
# ==========================================================================
NUMERIC_Q_EN = ("My {crop} field is at the {stage} stage. Today ET0 is {et0} mm, "
                "it rained {rain} mm, and the field is {aval} {aunit}. "
                "How much irrigation water does it need today?")
NUMERIC_Q_BN = ("আমার {crop_bn} ক্ষেত {stage_bn} পর্যায়ে আছে। আজ ET0 {et0_bn} মিমি, "
                "বৃষ্টি হয়েছে {rain_bn} মিমি, জমির পরিমাণ {aval_bn} {aunit}। "
                "আজ কত পানি সেচ দিতে হবে?")


def numeric_answers(res: dict, crop: str, crop_bn: str) -> tuple[str, str]:
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


def _draw_rain(rng) -> float:
    u, acc = rng.random(), 0.0
    for (lo, hi), w in RAIN_TIERS:
        acc += w
        if u <= acc:
            return round(rng.uniform(lo, hi), 2)
    return round(rng.uniform(*RAIN_TIERS[-1][0]), 2)


def generate_scenarios(n_scenarios, crop_pairs, blocklist, rng):
    """Sample n_scenarios scenarios meeting the 70% non-zero quota.

    NO (et0, rain) DRAW IS EVER DISCARDED. That is the whole design.

    Decision 2 wants 70% non-zero gold; requirement 6 wants
    |r(et0, rainfall)| <= 0.05. Any scheme that inspects a weather draw and
    throws it away for having the wrong zero-status breaks the second to serve
    the first. Both directions of the quota reject the same corner of the
    plane: when a non-zero row is needed, (low et0, high rain) has no crop that
    can produce one; when a zero row is needed, (high et0, low rain) has none
    either. Dropping both corners leaves the (low, low) and (high, high)
    quadrants over-represented and pushes r positive. Measured at +0.063 on all
    eight seeds — systematic, not sampling noise.

    So the weather is drawn once and kept:

      1. Draw n independent (et0, rain) pairs. Never reject. The marginals stay
         exactly Uniform(2, 9) and the specified rain tiers, and independence is
         preserved because nothing is conditioned on the pair.
      2. For each pair, ask which zero-statuses are ACHIEVABLE across the 84
         (crop, stage) options. Some pairs can only ever be non-zero, some only
         zero, most either.
      3. Fill the quota by assignment: forced points take their only option,
         then flexible points are allocated to reach 70/30. Assignment moves no
         point, so it cannot touch the correlation.
      4. Pick a crop uniformly among those achieving the assigned status.

    A scenario_hash collision (held-out blocklist or an internal duplicate) is
    resolved by redrawing the AREA only. Area is not one of the correlated
    variables, so that redraw is free.
    """
    kc_of = {(c, s): ic.KC_TABLE[ic.normalize_crop_name(c)][ic.STAGE_TO_KC[s]]
             for c, _, s in crop_pairs}
    stats = collections.Counter()

    # ---- 1. independent weather, kept unconditionally ---------------------
    weather = [(round(rng.uniform(2.0, 9.0), 2), _draw_rain(rng))
               for _ in range(n_scenarios)]

    # ---- 2. feasibility per point -----------------------------------------
    feasible = []
    for et0, rain in weather:
        nz, z = [], []
        for c, cbn, s in crop_pairs:
            (nz if ic.net_irrigation_mm(kc_of[(c, s)], et0, rain) > 0 else z
             ).append((c, cbn, s))
        feasible.append({"et0": et0, "rain": rain, "nz": nz, "z": z})

    # ---- 3. quota by assignment -------------------------------------------
    target_nz = int(round(NONZERO_QUOTA * n_scenarios))
    idx_forced_nz = [i for i, f in enumerate(feasible) if f["nz"] and not f["z"]]
    idx_forced_z = [i for i, f in enumerate(feasible) if f["z"] and not f["nz"]]
    idx_flex = [i for i, f in enumerate(feasible) if f["nz"] and f["z"]]
    rng.shuffle(idx_flex)

    stats["points_forced_nonzero"] = len(idx_forced_nz)
    stats["points_forced_zero"] = len(idx_forced_z)
    stats["points_flexible"] = len(idx_flex)

    want_nz = {i: True for i in idx_forced_nz}
    want_nz.update({i: False for i in idx_forced_z})
    remaining_nz = max(0, target_nz - len(idx_forced_nz))
    for j, i in enumerate(idx_flex):
        want_nz[i] = j < remaining_nz
    if len(idx_forced_nz) > target_nz:
        stats["quota_infeasible_too_many_forced_nonzero"] = \
            len(idx_forced_nz) - target_nz
    if remaining_nz > len(idx_flex):
        stats["quota_infeasible_too_few_flexible"] = remaining_nz - len(idx_flex)

    # ---- 4. crop choice and area ------------------------------------------
    out, seen = [], set()
    for i, f in enumerate(feasible):
        options = f["nz"] if want_nz[i] else f["z"]
        crop, crop_bn, stage = options[rng.randrange(len(options))]
        for _ in range(64):
            area = round(10 ** rng.uniform(-1.0, 1.0), 2)   # LogUniform(0.1, 10)
            unit = SYNTH_AREA_UNITS[rng.randrange(len(SYNTH_AREA_UNITS))]
            shash = scenario_hash(crop, unit, area, f["et0"], f["rain"], stage)
            if shash in blocklist:
                stats["area_redrawn_heldout_scenario_hash"] += 1
                continue
            if shash in seen:
                stats["area_redrawn_duplicate_scenario_hash"] += 1
                continue
            break
        else:
            stats["dropped_unresolvable_scenario_hash"] += 1
            continue
        res = ic.compute(crop, stage, f["et0"], f["rain"], area, unit)
        seen.add(shash)
        out.append({"crop": crop, "crop_bn": crop_bn, "stage": stage,
                    "et0_mm": f["et0"], "rainfall_mm": f["rain"],
                    "area_value": area, "area_unit": unit,
                    "scenario_hash": shash, "res": res})
        stats["accepted_nonzero" if res["net_mm"] > 0 else "accepted_zero"] += 1
    return out, stats


# ==========================================================================
# FAMILY D — noise injection
# ==========================================================================
def _is_mark(ch: str) -> bool:
    return unicodedata.category(ch).startswith("M")


def _typo(word: str, rng) -> str | None:
    """One character operation on one word.

    A Bengali combining mark cannot stand without a base consonant. An
    operation that leaves one stranded produces a dotted-circle placeholder
    (◌ি), which is a malformed grapheme cluster, not the human misspelling this
    family is meant to teach. Two rules keep every result well-formed:

      DROP  a mark is always droppable — "ধান" -> "ধন" is precisely the
            misspelling class Stage 4's misspelled_crop items test. A BASE
            character is droppable only when no mark follows it, otherwise
            "তিল" -> "িল" strands the ি.
      SWAP  both characters must be bases, since moving a mark ahead of its
            base strands it the same way.

    DUPLICATE is unrestricted: doubling a base keeps the following mark
    attached to the copy, and doubling a mark is itself a plausible typo.
    """
    if len(word) < 2:
        return None
    droppable = [i for i in range(len(word))
                 if _is_mark(word[i])
                 or not (i + 1 < len(word) and _is_mark(word[i + 1]))]
    swappable = [i for i in range(len(word) - 1)
                 if not _is_mark(word[i]) and not _is_mark(word[i + 1])]
    ops = ["duplicate"]
    if droppable:
        ops.append("drop")
    if swappable:
        ops.append("swap")
    op = ops[rng.randrange(len(ops))]
    if op == "swap":
        i = swappable[rng.randrange(len(swappable))]
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        i = droppable[rng.randrange(len(droppable))]
        return word[:i] + word[i + 1:]
    i = rng.randrange(len(word))
    return word[:i] + word[i] + word[i:]


def inject_noise(row: dict, rng) -> tuple[str, str] | None:
    """Typo the crop name or a unit word in the question. Answer untouched."""
    q, lang = row["question"], row["lang"]
    crop_surface = row["crop"] if lang == "en" else row.get("crop_bn", row["crop"])
    targets = []
    for token in [crop_surface] + UNIT_TOKENS[lang]:
        if token and token in q and len(token) >= 2:
            targets.append(token)
    if not targets:
        return None
    token = targets[rng.randrange(len(targets))]
    bad = _typo(token, rng)
    if not bad or bad == token:
        return None
    return q.replace(token, bad, 1), token


# ==========================================================================
# BUILD
# ==========================================================================
def build(families: str, numeric_target: int) -> dict:
    fails = run_regression_tests()
    if fails:
        raise SystemExit(f"bn_normalize regression FAILED: {fails}")

    ids = SeededIds(f"s2-rowid-{SEED}")
    seed_rows = [json.loads(l) for l in _real_open(TRAIN_SEED, encoding="utf-8")]
    blocklist_doc = json.loads(_real_open(BLOCKLIST, encoding="utf-8").read())
    blocklist = set(blocklist_doc["scenario_hashes"])

    drops = collections.Counter()
    notes = {}

    def child(parent: dict, origin: str, question: str, **extra) -> dict:
        r = copy.deepcopy(parent)
        r["row_id"] = ids.next()
        r["parent_row_id"] = parent["row_id"]
        r["origin"] = origin
        r["question"] = question
        r.update(extra)
        return r

    # ---- seed rows: Decision 2 zero-gold flagging --------------------------
    pool: list[dict] = []
    for r in seed_rows:
        r = copy.deepcopy(r)
        r["parent_row_id"] = None
        if r["qtype"] == "numeric_runtime":
            r["zero_gold"] = r.get("gold_trace", {}).get("net_mm", None) == 0
        pool.append(r)
    seed_zero_gold = sum(1 for r in pool if r.get("zero_gold"))

    # ---- families A and B -------------------------------------------------
    client = None
    if "A" in families or "B" in families:
        client = GeminiClient()

    seen_norm = {normalize(r["question"], r["lang"]) for r in pool}

    if "A" in families:
        for r in seed_rows:
            ents = extract_entities(r)
            for cand in gemini_paraphrases(client, r, PARAPHRASES_PER_ROW):
                ok, why = entities_preserved(ents, cand, r["lang"])
                if not ok:
                    drops[f"paraphrase:{why.split(':')[0]}"] += 1
                    continue
                n = normalize(cand, r["lang"])
                if n in seen_norm:
                    drops["paraphrase:duplicate"] += 1
                    continue
                seen_norm.add(n)
                pool.append(child(r, "paraphrase", cand.strip()))

    if "B" in families:
        for r in seed_rows:
            ents = extract_entities(r)
            cand = gemini_roundtrip(client, r)
            if cand is None:
                drops["backtranslation:api_failed"] += 1
                continue
            ok, why = entities_preserved(ents, cand.strip(), r["lang"])
            if not ok:
                drops[f"backtranslation:{why.split(':')[0]}"] += 1
                continue
            n = normalize(cand, r["lang"])
            if n in seen_norm:
                drops["backtranslation:duplicate"] += 1
                continue
            seen_norm.add(n)
            pool.append(child(r, "backtranslation", cand.strip()))

    # ---- family C ---------------------------------------------------------
    pearson_attempts = []
    synth_stats = collections.Counter()
    if "C" in families:
        crop_pairs = sorted({(r["crop"], r.get("crop_bn", r["crop"]), s)
                             for r in seed_rows for s in ic.STAGES
                             if ic.normalize_crop_name(r["crop"]) in ic.KC_TABLE})
        # A synthetic scenario yields one bn row and one en row, so the number
        # of scenarios is half the row deficit. Family D then copies a further
        # NOISE_FRACTION of the numeric pool, so aim below the target by that
        # factor rather than overshooting it.
        have_numeric = sum(1 for r in pool if r["qtype"] == "numeric_runtime")
        pre_noise_target = numeric_target / (1.0 + NOISE_FRACTION) if "D" in families \
            else numeric_target
        n_scen = max(0, math.ceil((pre_noise_target - have_numeric) / 2))

        from scipy.stats import pearsonr
        scenarios = []
        for attempt in range(PEARSON_MAX_ATTEMPTS):
            rng = random.Random(f"s2-synth-{SEED}-{attempt}")
            scenarios, synth_stats = generate_scenarios(
                n_scen, crop_pairs, blocklist, rng)
            et0s = [s["et0_mm"] for s in scenarios]
            rains = [s["rainfall_mm"] for s in scenarios]
            r_val, p_val = pearsonr(et0s, rains) if len(scenarios) > 2 else (0.0, 1.0)
            pearson_attempts.append({"attempt": attempt, "n": len(scenarios),
                                     "r": round(float(r_val), 5),
                                     "p": round(float(p_val), 5),
                                     # scipy returns np.float64; the comparison
                                     # yields np.bool_, which json cannot encode
                                     "within_tolerance": bool(abs(r_val) <= PEARSON_TOLERANCE)})
            if abs(r_val) <= PEARSON_TOLERANCE:
                break

        seed_numeric_by_fact = {}
        for r in seed_rows:
            if r["qtype"] == "numeric_runtime":
                seed_numeric_by_fact.setdefault(r["fact_id"], r["row_id"])

        for s in scenarios:
            res = s["res"]
            a_en, a_bn = numeric_answers(res, s["crop"], s["crop_bn"])
            fact_id = (f"{s['crop'].lower().replace(' ', '_')}_{s['stage']}"
                       f"_net_irrigation")
            common = {
                "source_id": None,
                "fact_id": fact_id,
                "scenario_hash": s["scenario_hash"],
                "qtype": "numeric_runtime",
                "crop": s["crop"], "crop_bn": s["crop_bn"],
                "origin": "synthetic",
                "parent_row_id": seed_numeric_by_fact.get(fact_id),
                "answer_kind": "computed",
                "zero_gold": res["net_mm"] == 0,
                "scenario": {"et0_mm": s["et0_mm"], "rainfall_mm": s["rainfall_mm"],
                             "area_value": s["area_value"], "area_unit": s["area_unit"],
                             "stage": s["stage"], "crop": s["crop"]},
                "gold_trace": res,
                "provenance": {
                    "source_file": None,
                    "extraction_method": "synthetic_scenario+deterministic_calc",
                    "verified_by": None,
                    "field_key": "computable_link",
                    "field_provenance": "computed",
                    "renderer": "irrigation_calc.compute",
                    "kc_source": res["kc_source"],
                    "kc_citation": res["citation"],
                    "synthetic_of_fact_id": fact_id,
                    "parent_fact_in_train_seed": fact_id in seed_numeric_by_fact,
                },
            }
            q_en = NUMERIC_Q_EN.format(crop=s["crop"], stage=s["stage"],
                                       et0=s["et0_mm"], rain=s["rainfall_mm"],
                                       aval=s["area_value"], aunit=s["area_unit"])
            q_bn = NUMERIC_Q_BN.format(crop_bn=s["crop_bn"],
                                       stage_bn=ic.STAGE_BN[s["stage"]],
                                       et0_bn=bn_num(s["et0_mm"]),
                                       rain_bn=bn_num(s["rainfall_mm"]),
                                       aval_bn=bn_num(s["area_value"]),
                                       aunit=AREA_UNIT_BN[s["area_unit"]])
            for lang, q, a in (("en", q_en, a_en), ("bn", q_bn, a_bn)):
                n = normalize(q, lang)
                if n in seen_norm:
                    drops["synthetic:duplicate"] += 1
                    continue
                seen_norm.add(n)
                pool.append({"row_id": ids.next(), "question": q, "answer": a,
                             "lang": lang, **copy.deepcopy(common)})

    # ---- family D ---------------------------------------------------------
    if "D" in families:
        # 10% of train rows, taken as 10% of EACH qtype. This is exactly the
        # specified 10% overall, and it removes the sampling variance of an
        # unstratified draw.
        #
        # IT DOES NOT rebalance the noise across qtypes, and it was a mistake to
        # think it would: 10% of every stratum reproduces the pool's own
        # proportions. After family C the pool is ~84% numeric, so the noise
        # rows are ~84% numeric either way. Only ~38 noise rows land on factual
        # questions.
        #
        # That matters because this family is meant to be the train-side
        # counterpart to Stage 4's misspelled_crop items, which attack factual
        # retrieval. The cause is the 3000-row numeric target dwarfing the ~430
        # factual rows, not anything family D does. Flagged at Gate 2 rather
        # than silently reinterpreted — over-weighting factual here would no
        # longer be "10% of train rows".
        rng = random.Random(f"s2-noise-{SEED}")
        noise_log = {}
        for qtype in sorted({r["qtype"] for r in pool}):
            eligible = [r for r in pool
                        if r["origin"] != "noise" and r["qtype"] == qtype]
            n_noise = int(round(NOISE_FRACTION * len(eligible)))
            order = list(range(len(eligible)))
            rng.shuffle(order)
            made = 0
            for idx in order:
                if made >= n_noise:
                    break
                src = eligible[idx]
                res = inject_noise(src, rng)
                if res is None:
                    drops["noise:no_target_token"] += 1
                    continue
                q_noisy, token = res
                n = normalize(q_noisy, src["lang"])
                if n in seen_norm:
                    drops["noise:duplicate"] += 1
                    continue
                seen_norm.add(n)
                pool.append(child(src, "noise", q_noisy, noise_target_token=token))
                made += 1
            noise_log[qtype] = {"eligible": len(eligible), "requested": n_noise,
                                "made": made}
        notes["noise_by_qtype"] = noise_log

    # ---- 50/50 bn/en balance within each qtype ----------------------------
    # Downsampling removes generated rows before seed rows: seed rows are the
    # only ones traceable to a verified calendar field, so they are the last
    # thing to discard.
    rng = random.Random(f"s2-balance-{SEED}")
    balance_log = []
    kept: list[dict] = []
    ORIGIN_DROP_ORDER = {"noise": 0, "backtranslation": 1, "paraphrase": 2,
                         "synthetic": 3, "seed": 4}
    for qtype in sorted({r["qtype"] for r in pool}):
        rows_q = [r for r in pool if r["qtype"] == qtype]
        by_lang = {"bn": [r for r in rows_q if r["lang"] == "bn"],
                   "en": [r for r in rows_q if r["lang"] == "en"]}
        target = min(len(by_lang["bn"]), len(by_lang["en"]))
        entry = {"qtype": qtype, "bn_before": len(by_lang["bn"]),
                 "en_before": len(by_lang["en"]), "target_each": target}
        for lang, rows_l in by_lang.items():
            drop_n = len(rows_l) - target
            if drop_n > 0:
                shuffled = list(rows_l)
                rng.shuffle(shuffled)
                shuffled.sort(key=lambda r: ORIGIN_DROP_ORDER.get(r["origin"], 9))
                dropped = shuffled[:drop_n]
                for d in dropped:
                    drops[f"balance:{d['origin']}"] += 1
                entry[f"{lang}_dropped_by_origin"] = dict(collections.Counter(
                    d["origin"] for d in dropped))
                keep_ids = {id(r) for r in shuffled[drop_n:]}
                rows_l = [r for r in rows_l if id(r) in keep_ids]
            kept.extend(rows_l)
        balance_log.append(entry)
    pool = kept

    # ---- integrity assertions ---------------------------------------------
    seed_ids = {r["row_id"] for r in seed_rows}
    gen_ids = [r["row_id"] for r in pool if r["origin"] != "seed"]
    assert not (set(gen_ids) & seed_ids), \
        "row_id collision between generated rows and Stage 0 seed rows"
    assert len(set(r["row_id"] for r in pool)) == len(pool), "duplicate row_id"
    for r in pool:
        if r["origin"] != "seed":
            assert "parent_row_id" in r, f"{r['row_id']} missing parent_row_id"
    synth_hashes = {r["scenario_hash"] for r in pool
                    if r["origin"] == "synthetic" and r.get("scenario_hash")}
    assert not (synth_hashes & blocklist), \
        "HARD CONSTRAINT VIOLATION: synthetic scenario_hash present in a held-out split"

    # ---- write -------------------------------------------------------------
    pool.sort(key=lambda r: (r["qtype"], r["origin"], r["lang"], r["row_id"]))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _real_open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- GATE 2 ------------------------------------------------------------
    def pct(rows, key, val):
        return round(100.0 * sum(1 for r in rows if r[key] == val) / len(rows), 1) \
            if rows else 0.0

    table = []
    for origin in ("seed", "paraphrase", "backtranslation", "synthetic", "noise"):
        rows_o = [r for r in pool if r["origin"] == origin]
        if not rows_o:
            continue
        table.append({
            "origin": origin, "rows": len(rows_o),
            "pct_bn": pct(rows_o, "lang", "bn"), "pct_en": pct(rows_o, "lang", "en"),
            "pct_numeric": pct(rows_o, "qtype", "numeric_runtime"),
            "pct_factual": pct(rows_o, "qtype", "factual"),
            "pct_abstain": pct(rows_o, "qtype", "abstain"),
        })

    synth_rows = [r for r in pool if r["origin"] == "synthetic"]
    synth_zero = sum(1 for r in synth_rows if r.get("zero_gold"))
    final_pearson = pearson_attempts[-1] if pearson_attempts else None
    numeric_rows = [r for r in pool if r["qtype"] == "numeric_runtime"]
    exposed = sum(1 for r in synth_rows
                  if not r["provenance"].get("parent_fact_in_train_seed"))

    report = {
        "stage": "GATE 2",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": SEED,
        "families_run": families,
        "normalizer_version": NORMALIZER_VERSION,
        "input": str(TRAIN_SEED),
        "input_rows": len(seed_rows),
        "input_sha256": hashlib.sha256(
            _real_open(TRAIN_SEED, "rb").read()).hexdigest(),
        "blocklist": {
            "path": str(BLOCKLIST),
            "distinct_hashes": len(blocklist),
            "source_sha256": {k: v["sha256"]
                              for k, v in blocklist_doc["sources"].items()},
        },
        "output": str(OUT_PATH),
        "output_rows": len(pool),
        "output_sha256": hashlib.sha256(_real_open(OUT_PATH, "rb").read()).hexdigest(),
        "table_by_origin": table,
        "rows_by_qtype": dict(sorted(collections.Counter(
            r["qtype"] for r in pool).items())),
        "rows_by_lang": dict(sorted(collections.Counter(
            r["lang"] for r in pool).items())),
        "lang_balance_exact": len({
            v for v in collections.Counter(r["lang"] for r in pool).values()}) == 1,
        "balance_log": balance_log,
        "numeric_rows_total": len(numeric_rows),
        "numeric_target": numeric_target,
        "zero_gold": {
            "seed_numeric_zero_gold": seed_zero_gold,
            "synthetic_rows": len(synth_rows),
            "synthetic_zero_gold": synth_zero,
            "synthetic_zero_gold_pct": round(100.0 * synth_zero / len(synth_rows), 1)
            if synth_rows else 0.0,
            "quota_target_pct": round(100 * (1 - NONZERO_QUOTA), 1),
        },
        "pearson_et0_rainfall": final_pearson,
        "pearson_attempts": pearson_attempts,
        "pearson_tolerance": PEARSON_TOLERANCE,
        "synthetic_sampling_stats": dict(synth_stats),
        "noise_by_qtype": notes.get("noise_by_qtype", {}),
        "synthetic_crop_stage_exposure": {
            "rows_whose_fact_is_not_in_train_seed": exposed,
            "pct": round(100.0 * exposed / len(synth_rows), 1) if synth_rows else 0.0,
            "note": "Family C samples (crop, stage) across all 28 crops, as the "
                    "spec's only stated constraint is scenario_hash. Rows counted "
                    "here teach a Kc whose seed fact sits in a held-out split.",
        },
        "dropped_by_check": dict(sorted(drops.items())),
        "dropped_total": sum(drops.values()),
        "gemini": dict(client.stats) if client else {"skipped": True},
        "held_out_guard": {
            "static_source_scan": "pass",
            "runtime_open_guard": "installed",
            "held_out_files_opened": 0,
        },
    }
    GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    GATE_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    LOG_PATH.write_text(
        f"s2_augment_train.py\nseed={SEED}\nfamilies={families}\n"
        f"run_at={report['generated_at']}\nnormalizer={NORMALIZER_VERSION}\n"
        f"in={len(seed_rows)} out={len(pool)}\n"
        f"train_augmented.sha256={report['output_sha256']}\n"
        f"pearson={final_pearson}\ndropped={sum(drops.values())}\n",
        encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="ABCD",
                    help="subset of ABCD; CD needs no API key")
    ap.add_argument("--numeric-target", type=int, default=NUMERIC_TARGET)
    a = ap.parse_args()
    rep = build(a.families.upper(), a.numeric_target)

    W = [17, 8, 8, 8, 11, 11, 10]
    print("=" * 74)
    print(f"GATE 2 — AUGMENT TRAIN ONLY   (families run: {rep['families_run']})")
    print("=" * 74)
    print("".join(h.ljust(w) for h, w in zip(
        ["origin", "rows", "%bn", "%en", "%numeric", "%factual", "%abstain"], W)))
    print("-" * sum(W))
    for t in rep["table_by_origin"]:
        print("".join(str(c).ljust(w) for c, w in zip(
            [t["origin"], t["rows"], t["pct_bn"], t["pct_en"],
             t["pct_numeric"], t["pct_factual"], t["pct_abstain"]], W)))
    print("-" * sum(W))
    print(f"{'TOTAL'.ljust(W[0])}{str(rep['output_rows']).ljust(W[1])}"
          f"{'50.0'.ljust(W[2])}{'50.0'.ljust(W[3])}")

    print(f"\nrows by qtype        : {rep['rows_by_qtype']}")
    print(f"rows by lang         : {rep['rows_by_lang']}  "
          f"exact 50/50: {rep['lang_balance_exact']}")
    print(f"numeric rows         : {rep['numeric_rows_total']} "
          f"(target ~{rep['numeric_target']})")
    z = rep["zero_gold"]
    print(f"zero-gold synthetic  : {z['synthetic_zero_gold']}/{z['synthetic_rows']} "
          f"= {z['synthetic_zero_gold_pct']}%  (quota allows {z['quota_target_pct']}%)")
    print(f"zero-gold seed rows  : {z['seed_numeric_zero_gold']} (kept, flagged)")
    p = rep["pearson_et0_rainfall"]
    if p:
        verdict = "PASS" if p["within_tolerance"] else "FAIL"
        print(f"Pearson r(et0, rain) : {p['r']:+.5f}  p={p['p']:.4f}  n={p['n']}  "
              f"[tolerance ±{rep['pearson_tolerance']}]  -> {verdict}")
        print(f"                       attempts: {len(rep['pearson_attempts'])}")
    print(f"dropped by checks    : {rep['dropped_total']}")
    for k, v in sorted(rep["dropped_by_check"].items()):
        print(f"    {k:42s} {v}")
    print(f"\nheld-out guard       : {rep['held_out_guard']}")
    print(f"train_augmented.sha256: {rep['output_sha256']}")
    print(f"\nfull report -> {GATE_JSON}")
    print("STOP. Gate 2 requires approval before Stage 3.")
