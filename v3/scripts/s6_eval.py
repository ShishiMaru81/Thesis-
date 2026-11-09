#!/usr/bin/env python3
"""
s6_eval.py — Stage 6: evaluate the 16-arm matrix on the hardened test set.

THE MATRIX
----------
  4 base models x zero-shot          (base weights, no adapter, n=1 per model)
  4 base models x fine-tuned x 3 seeds

Arms are named {model}_zeroshot and {model}_seed{seed}. One arm per model load;
the model is freed and CUDA cache emptied between arms.

SCOPE — 362 OF 422 ROWS ARE SCORABLE
-------------------------------------
60 rows carry gold_status == "pending_human" (40 out_of_domain / abstain,
20 conflicting_values / clarify). Their `answer` field is the empty string
because the gold has not been authored yet. Scoring an empty gold as a miss
would depress all 16 arms by the same amount and corrupt the comparison, so
they are filtered out of every metric at load time and the count is asserted.

Predictions are still GENERATED for all 422 rows and written to disk. When the
authoring sheet is filled the 60 can be scored by re-running with --score_only,
without paying for inference again.

WHAT v3 ANSWERS ACTUALLY LOOK LIKE — AND WHY THE PARSER IS A PROSE PARSER
-------------------------------------------------------------------------
The Stage 6 brief asks for an <irrigation_calc> JSON block and a parse_success
rate over it. That block belongs to v2. s2b_render_train.py:126-130 refuses to
render any corpus containing it, and this file verifies the same thing on the
test set at startup: 0 of 422 v3 answers carry the tag. v3's numeric target is
the worked answer with the arithmetic shown:

  en: Net irrigation depth = Kc×ET0 - effective rainfall = 0.96x8.8 - 2.99 = 5.46 mm.
      For 3.0 acre (12141 m2), that is about 66288 liters today.
  bn: নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি = ০.৮×৭.৯ − ৬.৩৩ = ০.০ মিমি।
      ৪.১ একর (১৬৫৯২ বর্গমিটার)-এর জন্য আজ প্রায় ০ লিটার।

So <irrigation_calc> IS still extracted and reported (calc_block_rate,
calc_block_parse_ok) because a base model emitting it would be a real finding —
but it cannot be the headline parse metric, because the correct rate for a
well-behaved v3 model is zero. The headline parse_success is net_extract_rate:
did a net-irrigation depth come out of the prose at all.

BENGALI DIGITS ARE NOT bn_normalize's JOB
------------------------------------------
bn_normalize.normalize() maps punctuation to a space, so "৬.৩ মিমি" becomes
"6 3 মিমি" — the decimal point is destroyed. That is correct for text overlap
metrics and fatal for number extraction. Numeric parsing therefore uses its own
digit-only transliteration (to_ascii_digits) that touches nothing else.
bn_normalize is used for, and only for, the text-similarity metrics.

METRIC GATING
-------------
Every metric asserts the qtype of the row it is handed. numeric metrics on a
factual row, or Dice on a clarify row, raise rather than silently averaging.
The gate is enforced in score_row(), not by convention.

M5 SEMANTIC JUDGE IS DEFERRED — no LLM-as-judge runs here. Stage 7 human eval
covers it. Reported as a stated absence, not omitted silently.
"""
from __future__ import annotations

import argparse
import builtins
import collections
import datetime
import gc
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

TEST_PATH = PROJECT_ROOT / "data" / "hardened" / "test_hardened.jsonl"
TEST_SHA_PATH = PROJECT_ROOT / "data" / "hardened" / "test_hardened.sha256"
ADAPTER_ROOT = PROJECT_ROOT / "models" / "s5"
S6E_ADAPTER_ROOT = PROJECT_ROOT / "models" / "s6e"
REPORT_DIR = PROJECT_ROOT / "reports" / "s6"
PRED_DIR = REPORT_DIR / "predictions"
LOG_PATH = PROJECT_ROOT / "logs" / "s6_eval.log"

EXPECTED_TOTAL = 422
EXPECTED_SCORABLE = 362
EXPECTED_SCORABLE_QTYPES = {"numeric_runtime": 146, "factual": 114,
                            "clarify": 68, "abstain": 34}

MODEL_REGISTRY = {
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral_nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "phi35_mini": "microsoft/Phi-3.5-mini-instruct",
    "phi4_mini": "microsoft/Phi-4-mini-instruct",
}
SEEDS = [42, 1337, 2024]

# Generation. max_seq_length is 2048 here, not training's 768: the prompt alone
# is short but a zero-shot base model will happily run the full 512 new tokens,
# and 2048 is inside every one of these models' native context so no RoPE
# scaling is introduced by the change.
GEN_MAX_NEW_TOKENS = 512
INFER_MAX_SEQ_LEN = 2048

log = logging.getLogger("s6")


# ===========================================================================
# ENTRY GATE
# ===========================================================================
# The brief requires that train_augmented.jsonl and val.jsonl are never opened
# by this script. A comment cannot enforce that, so the guard is live: both
# builtins.open and Path.open are wrapped for the whole run and raise on the
# forbidden basenames. Everything else (model weights, tokenizers, caches)
# passes through untouched.
FORBIDDEN_BASENAMES = {"train_augmented.jsonl", "val.jsonl", "train.jsonl"}
_real_open = builtins.open
_real_path_open = Path.open


def _check_forbidden(path) -> None:
    try:
        name = os.path.basename(os.fspath(path))
    except (TypeError, ValueError):
        return
    if name in FORBIDDEN_BASENAMES:
        raise RuntimeError(
            f"FORBIDDEN READ: s6_eval.py attempted to open {name!r}. "
            "Stage 6 evaluates on the hardened test set only; touching the "
            "training or validation corpus here would contaminate the result.")


def _guarded_open(file, *a, **kw):
    _check_forbidden(file)
    return _real_open(file, *a, **kw)


def _guarded_path_open(self, *a, **kw):
    _check_forbidden(self)
    return _real_path_open(self, *a, **kw)


def install_read_guard() -> None:
    builtins.open = _guarded_open
    Path.open = _guarded_path_open


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with _real_open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry_gate(skip_manifest: bool = False) -> dict:
    """Verify S5 adapters and the test-set hash before anything expensive."""
    import subprocess

    log.info("=" * 78)
    log.info("ENTRY GATE")
    log.info("=" * 78)
    info = {}

    # [1] S5 manifests
    if skip_manifest:
        log.warning("[1] verify_s5_manifests.py SKIPPED by flag")
        info["manifest_verify"] = "skipped"
    else:
        script = PROJECT_ROOT / "scripts" / "verify_s5_manifests.py"
        proc = subprocess.run([sys.executable, str(script)],
                              capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        if proc.returncode != 0:
            log.error(proc.stdout[-4000:])
            log.error(proc.stderr[-4000:])
            raise SystemExit("ABORT: verify_s5_manifests.py failed — adapters are not trustworthy.")
        log.info("[1] verify_s5_manifests.py PASS (12/12 cells)")
        info["manifest_verify"] = "pass"

    # [2] test-set hash
    expected = _real_open(TEST_SHA_PATH, encoding="utf-8").read().split()[0].strip().lower()
    actual = sha256_of(TEST_PATH)
    if expected != actual:
        raise SystemExit(
            f"ABORT: test_hardened.jsonl hash mismatch.\n"
            f"  expected {expected}\n  actual   {actual}\n"
            "The test set on disk is not the one Stage 4 sealed.")
    log.info("[2] test_hardened.sha256 PASS  %s", actual)
    info["test_sha256"] = actual

    # [3] read guard
    install_read_guard()
    for name in sorted(FORBIDDEN_BASENAMES):
        try:
            _guarded_open(PROJECT_ROOT / "data" / "augmented" / name)
        except RuntimeError:
            pass
        else:
            raise SystemExit(f"ABORT: read guard did not fire for {name}")
    log.info("[3] read guard ARMED and self-tested: %s", sorted(FORBIDDEN_BASENAMES))
    info["read_guard"] = sorted(FORBIDDEN_BASENAMES)

    # [4] normalizer version
    import bn_normalize
    fails = bn_normalize.run_regression_tests()
    if fails:
        raise SystemExit(f"ABORT: bn_normalize regression failures: {fails}")
    log.info("[4] bn_normalize %s PASS (%d cases)",
             bn_normalize.NORMALIZER_VERSION, len(bn_normalize.REGRESSION_CASES))
    info["normalizer_version"] = bn_normalize.NORMALIZER_VERSION

    return info


# ===========================================================================
# DATA
# ===========================================================================
def load_test_rows() -> tuple[list[dict], list[dict]]:
    with _real_open(TEST_PATH, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    if len(rows) != EXPECTED_TOTAL:
        raise SystemExit(f"expected {EXPECTED_TOTAL} test rows, got {len(rows)}")

    tagged = sum(1 for r in rows if "<irrigation_calc>" in str(r.get("answer", "")))
    if tagged:
        raise SystemExit(
            f"{tagged} gold answers contain <irrigation_calc>; this script's "
            "prose parser assumes the v3 worked-answer format")

    scorable = [r for r in rows if r.get("gold_status") != "pending_human"]
    assert len(scorable) == EXPECTED_SCORABLE, \
        f"expected {EXPECTED_SCORABLE} scorable, got {len(scorable)}"

    # The join key must be row_id, NOT item_id. item_id is null on all 182 seed
    # rows (only the 240 adversarial rows carry one), so keying on it collapses
    # every seed row onto a single dict entry and scores 182 predictions against
    # one arbitrary gold. That is exactly what happened on the first smoke arm:
    # factual dropped to 0 and abstain inflated to 182. It also slipped past a
    # bare len()==362 assert by arithmetic coincidence (182 collapsed + 180
    # correctly mapped = 362), which is why the distribution is asserted too.
    ids = [r["row_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("row_id is not unique — cannot join predictions to golds")

    got = collections.Counter(r["qtype"] for r in scorable)
    if dict(got) != EXPECTED_SCORABLE_QTYPES:
        raise SystemExit(f"scorable qtype mix changed: {dict(got)} "
                         f"!= {EXPECTED_SCORABLE_QTYPES}")

    pending = [r for r in rows if r.get("gold_status") == "pending_human"]
    by_cat = collections.Counter(r["category"] for r in pending)
    log.info("test rows: %d total | %d scorable | %d pending_human %s",
             len(rows), len(scorable), len(pending), dict(by_cat))
    log.info("calc-block gold rate: 0/%d (v3 uses worked prose, not tool calls)", len(rows))
    return rows, scorable


# ===========================================================================
# NUMERIC / SLOT EXTRACTION
# ===========================================================================
BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_DIGIT_MAP = {ord(b): a for b, a in zip(BN_DIGITS, "0123456789")}

NUM = r"[-+]?\d+(?:\.\d+)?"
MM = r"(?:mm|মিমি|মিলিমিটার)"
LIT = r"(?:liters?|litres?|লিটার)"
SQM = r"(?:m2|m²|sq\.?\s*m|square\s*met(?:er|re)s?|বর্গমিটার|বর্গ\s*মিটার)"
MUL = r"[×xX*]"
MINUS = r"[-−–—]"

STAGE_SURFACE = {
    "initial":     ["initial", "প্রাথমিক"],
    "mid-season":  ["mid-season", "mid season", "midseason", "মধ্য-মৌসুম", "মধ্য মৌসুম", "মধ্যমৌসুম"],
    "late-season": ["late-season", "late season", "lateseason", "শেষ-মৌসুম", "শেষ মৌসুম", "শেষমৌসুম"],
}

# Refusal / clarification surface forms, taken from the authored golds.
ABSTAIN_MARKERS = [
    "crop calendar does not specify", "does not specify this",
    "i can only answer", "i can only help",
    "outside", "not something i can", "cannot help with",
    "পঞ্জিকায় এই তথ্য নেই", "আমি কেবল", "তথ্য নেই",
    "সংক্রান্ত উৎস দেখুন", "consult a market",
]
CLARIFY_MARKERS = [
    "please tell me", "please confirm", "could you tell", "can you tell me",
    "i cannot compute this without", "cannot compute", "i am not sure which",
    "i cannot choose", "cannot convert that to liters", "tell me the area",
    "which stage", "please provide", "i need to know",
    "অনুগ্রহ করে", "জানান", "নিশ্চিত করলে", "নিশ্চিত নই",
    "হিসাব করা সম্ভব নয়", "সম্ভব নয়", "জানালে", "বলতে আপনি কোন",
]
GENERIC_CLARIFY_MARKERS = [
    "more information", "more info", "more details", "further information",
    "additional information", "need more", "insufficient information",
    "আরও তথ্য", "অতিরিক্ত তথ্য", "বিস্তারিত জানান",
]

# subtype -> keywords that name the specific missing slot. A clarify response
# scores 1.0 only if it names the slot; 0.5 for a generic request; 0.0 else.
CLARIFY_SLOT_KEYWORDS = {
    "missing_et0":        ["et0", "et 0", "eto", "evapotranspiration",
                           "বাষ্পীভবন", "বাষ্পীয়", "ইটিও"],
    "missing_rainfall":   ["rain", "rainfall", "rained", "precipitation",
                           "বৃষ্টি", "বৃষ্টিপাত"],
    "missing_area":       ["area", "field size", "size of", "how large", "how big",
                           "acre", "bigha", "katha", "hectare", "decimal",
                           "জমির পরিমাণ", "জমির মাপ", "আয়তন", "একক",
                           "একর", "বিঘা", "কাঠা", "হেক্টর", "শতক"],
    "missing_stage":      ["stage", "growth stage", "initial", "mid-season", "late-season",
                           "পর্যায়", "মৌসুম", "বৃদ্ধির"],
    "ambiguous":          ["which crop", "which variety", "crop", "aman", "aus", "boro",
                           "ফসল", "আমন", "আউশ", "বোরো", "কোন ফসল"],
    "unknown_local_unit": ["kani", "gonda", "unit", "convert", "conversion",
                           "acre", "bigha", "katha", "decimal", "hectare",
                           "একক", "গণ্ডা", "কানি", "রূপান্তর",
                           "একর", "বিঘা", "কাঠা", "শতক", "হেক্টর"],
}


def to_ascii_digits(s: str) -> str:
    """Bengali digits -> ASCII. Nothing else is touched; decimal points and
    the arithmetic operators must survive for the number regexes."""
    return unicodedata.normalize("NFC", s).translate(_DIGIT_MAP)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def extract_calc_block(text: str) -> tuple[bool, bool, dict | None]:
    """v2's <irrigation_calc> JSON block. Expected absent in v3 — reported so
    that a base model emitting one shows up as a finding rather than a bug."""
    m = re.search(r"<irrigation_calc>(.*?)</irrigation_calc>", text, re.S)
    if not m:
        return False, False, None
    try:
        return True, True, json.loads(m.group(1).strip())
    except Exception:
        return True, False, None


def extract_numeric_fields(text: str) -> dict:
    """Pull the irrigation slots out of a worked-prose answer, bn or en.

    Returns per-field values plus `net_mm_all`, the ordered list of every net
    depth stated — multi_crop answers state two, one per crop.
    """
    t = to_ascii_digits(text)
    out = {"net_mm": None, "net_mm_all": [], "liters": None, "liters_all": [],
           "kc": None, "kc_all": [], "et0_mm": None, "eff_rain_mm": None,
           "area_sqm": None}

    # The full worked chain: "... = <kc>x<et0> - <effrain> = <net> mm".
    # Non-greedy is safe here: the intermediate "=" are followed by an
    # expression, never by "<number> mm", so the first match of "= N mm" is
    # always the net depth of that clause.
    chain = re.compile(
        rf"({NUM})\s*{MUL}\s*({NUM})\s*{MINUS}\s*({NUM})\s*=\s*({NUM})\s*{MM}", re.I)
    chain_hits = chain.findall(t)
    if chain_hits:
        out["kc_all"] = [_f(h[0]) for h in chain_hits]
        out["net_mm_all"] = [_f(h[3]) for h in chain_hits]
        out["kc"] = _f(chain_hits[0][0])
        out["et0_mm"] = _f(chain_hits[0][1])
        out["eff_rain_mm"] = _f(chain_hits[0][2])
        out["net_mm"] = _f(chain_hits[0][3])

    # Fallback: any "= <number> mm" (model showed a shorter working).
    if not out["net_mm_all"]:
        eq_mm = re.findall(rf"=\s*({NUM})\s*{MM}", t, re.I)
        if eq_mm:
            out["net_mm_all"] = [_f(x) for x in eq_mm]
            out["net_mm"] = _f(eq_mm[-1])

    # Last resort: a bare "<number> mm" that is not the quoted ET0/rainfall
    # input. Only used when nothing above fired.
    if out["net_mm"] is None:
        bare = re.findall(rf"({NUM})\s*{MM}", t, re.I)
        if bare:
            out["net_mm"] = _f(bare[-1])
            out["net_mm_all"] = [out["net_mm"]]

    if out["kc"] is None:
        kc = re.findall(rf"kc\s*(?:=|:|\s+is\s+)\s*({NUM})", t, re.I)
        if kc:
            out["kc_all"] = [_f(x) for x in kc]
            out["kc"] = _f(kc[0])
    if out["et0_mm"] is None:
        e = re.search(rf"et\s*0?\s*(?:=|:|\s+is\s+)?\s*({NUM})\s*{MM}", t, re.I)
        if e:
            out["et0_mm"] = _f(e.group(1))

    lit = re.findall(rf"({NUM})\s*{LIT}", t, re.I)
    if lit:
        out["liters_all"] = [_f(x) for x in lit]
        out["liters"] = _f(lit[0])

    sq = re.findall(rf"({NUM})\s*{SQM}", t, re.I)
    if sq:
        out["area_sqm"] = _f(sq[0])

    return out


def close(a, b, tol) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def gold_traces(row: dict) -> list[dict]:
    """Normalise single-crop and multi_crop gold_trace into a list of traces."""
    gt = row.get("gold_trace") or {}
    if "crop_1" in gt:
        return [gt[k] for k in sorted(gt) if k.startswith("crop_")]
    return [gt] if gt else []


# ===========================================================================
# TEXT METRICS (factual only)
# ===========================================================================
def ngrams(s: str, n: int) -> collections.Counter:
    return collections.Counter(s[i:i + n] for i in range(len(s) - n + 1))


def dice_counter(a: collections.Counter, b: collections.Counter) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = sum((a & b).values())
    return 2.0 * inter / (sum(a.values()) + sum(b.values()))


def dice_word(pred_n: str, gold_n: str) -> float:
    return dice_counter(collections.Counter(pred_n.split()),
                        collections.Counter(gold_n.split()))


def dice_char(pred_n: str, gold_n: str, n: int) -> float:
    return dice_counter(ngrams(pred_n.replace(" ", ""), n),
                        ngrams(gold_n.replace(" ", ""), n))


def rouge_l(pred_n: str, gold_n: str) -> float:
    """LCS-based F1 on normalized tokens.

    rouge_score's own tokenizer is English-only — it strips non-ASCII, which
    reduces every Bengali reference to the empty string. So ROUGE-L is computed
    here on bn_normalize output, which is script-correct for both languages.
    """
    p, g = pred_n.split(), gold_n.split()
    if not p or not g:
        return 0.0
    prev = [0] * (len(g) + 1)
    for i in range(1, len(p) + 1):
        cur = [0] * (len(g) + 1)
        pi = p[i - 1]
        for j in range(1, len(g) + 1):
            cur[j] = prev[j - 1] + 1 if pi == g[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec)


# ===========================================================================
# RESPONSE CLASSIFICATION (answer / clarify / abstain)
# ===========================================================================
def has_marker(text_low: str, markers) -> bool:
    return any(m in text_low for m in markers)


def classify_response(gen: str) -> str:
    """Bucket a generation into answer / clarify / abstain.

    Order matters. A unit_ambiguity clarify gold contains BOTH a computed depth
    and a request for the unit, so "contains a number" cannot by itself mean
    "answer" — the request markers are tested first.

    The asking markers alone are still not enough, because the bigha_regional
    numeric golds deliver a COMPLETE result and then append a courtesy caveat
    ("...if a bigha is a different size in your area, জানালে I will recompute").
    Five such golds classified as clarify on markers alone. What separates them
    from a real clarify is delivery: they state both the depth AND the volume in
    litres, whereas every genuine clarify gold withholds the volume — the
    unknown_local_unit and missing_area golds give a depth and stop, because the
    area is exactly what they are missing. So a response that delivers the full
    result is an answer-with-caveat, not a request.

    This tightening also makes invented_value_rate stricter rather than looser:
    a model that emits litres for a row where the area was never given has
    invented the area, and is now counted as such.
    """
    if not gen.strip():
        return "empty"
    low = to_ascii_digits(gen).lower()
    if has_marker(low, ABSTAIN_MARKERS):
        return "abstain"
    if has_marker(low, CLARIFY_MARKERS) or has_marker(low, GENERIC_CLARIFY_MARKERS):
        ext = extract_numeric_fields(gen)
        delivered = ext["net_mm"] is not None and ext["liters"] is not None
        return "answer" if delivered else "clarify"
    return "answer"


GOLD_CLASS = {"numeric_runtime": "answer", "factual": "answer",
              "clarify": "clarify", "abstain": "abstain"}


def score_clarify(gen: str, subtype: str) -> tuple[float, str]:
    """1.0 names the specific missing slot, 0.5 generic ask, 0.0 neither."""
    low = to_ascii_digits(gen).lower()
    kws = CLARIFY_SLOT_KEYWORDS.get(subtype, [])
    asked = has_marker(low, CLARIFY_MARKERS) or has_marker(low, GENERIC_CLARIFY_MARKERS)
    named = any(k.lower() in low for k in kws)
    if asked and named:
        return 1.0, "named_slot"
    if asked:
        return 0.5, "generic_ask"
    # Naming WITHOUT asking earns nothing. The worked-answer format itself
    # contains the slot words — "Net irrigation depth = Kc×ET0 - effective
    # rainfall" literally contains "ET0" and "rainfall" — so a model that
    # ignores the missing parameter and computes anyway would otherwise collect
    # half credit for emitting the formula. Observed on llama31_8b_seed42:
    # every one of its 24 "partials" was this string, not a request.
    return 0.0, "named_no_ask" if named else "no_clarify"


# ===========================================================================
# PER-ROW SCORING — GATED BY QTYPE
# ===========================================================================
SLOTS = ["crop", "stage", "kc", "et0_mm", "rainfall_mm", "area_sqm"]

TOL = {"kc": 0.005, "et0_mm": 0.05, "rainfall_mm": 0.05, "area_sqm": 1.0, "net": 0.05}


def score_row(row: dict, gen: str, normalize) -> dict:
    """Score one generation. Raises if a metric family meets the wrong qtype."""
    qtype, lang = row["qtype"], row["lang"]
    gold = str(row.get("answer", ""))
    rec = {
        "item_id": row.get("item_id"), "row_id": row.get("row_id"),
        "qtype": qtype, "lang": lang, "category": row["category"],
        "subtype": row.get("subtype"), "gen_chars": len(gen),
        "gen_empty": not gen.strip(),
    }

    has_block, block_ok, _ = extract_calc_block(gen)
    rec["calc_block_present"] = has_block
    rec["calc_block_parse_ok"] = block_ok

    pred_cls = classify_response(gen)
    rec["pred_class"] = pred_cls
    rec["gold_class"] = GOLD_CLASS[qtype]

    # ---------------------------------------------------------- numeric
    if qtype == "numeric_runtime":
        traces = gold_traces(row)
        if not traces:
            raise AssertionError(f"numeric_runtime row {row.get('item_id')} has no gold_trace")
        ext = extract_numeric_fields(gen)
        rec["net_extracted"] = ext["net_mm"] is not None
        rec["is_multi"] = len(traces) > 1

        gold_nets = [t.get("net_mm") for t in traces]
        rec["gold_net_mm"] = gold_nets[0] if len(traces) == 1 else gold_nets
        rec["gold_zero"] = (len(traces) == 1 and gold_nets[0] == 0)

        if len(traces) == 1:
            pred_net = ext["net_mm"]
            rec["pred_net_mm"] = pred_net
            if pred_net is not None:
                err = pred_net - gold_nets[0]
                rec["abs_err"] = abs(err)
                rec["sq_err"] = err * err
                rec["em_0p1"] = int(abs(err) <= 0.1)
                rec["em_1p0"] = int(abs(err) <= 1.0)
                if gold_nets[0] != 0:
                    rec["ape"] = abs(err) / abs(gold_nets[0]) * 100.0
        else:
            # multi_crop: pair each gold net with the model's stated nets in
            # order. Kept out of the single-crop MAE so the two are never
            # silently averaged; reported as its own n.
            preds = ext["net_mm_all"]
            rec["pred_net_mm"] = preds
            pair_errs, pair_hits = [], []
            for i, gnet in enumerate(gold_nets):
                p = preds[i] if i < len(preds) else None
                if p is None:
                    pair_hits.append(0)
                    continue
                pair_errs.append(abs(p - gnet))
                pair_hits.append(int(abs(p - gnet) <= 0.1))
            rec["multi_abs_errs"] = pair_errs
            rec["multi_n_values"] = len(gold_nets)
            rec["multi_all_correct"] = int(len(pair_hits) == len(gold_nets) and all(pair_hits))

        # slots. multi_crop is conjunctive: every gold crop/stage/kc must appear.
        #
        # crop and stage are marked N/A unless the GOLD answer states them. v3's
        # numeric format is bare arithmetic — "Net irrigation depth = 0.8x7.9 -
        # 6.33 = 0.0 mm. For 4.1 acre (16592 m2)..." names neither the crop nor
        # the stage. Only 18.5% of gold numeric answers state the crop and 20.5%
        # the stage (the misspelled_crop and multi_crop rows, which restate them
        # by design). Scoring those on every row measures a metric whose ceiling
        # is 0.185 and penalises a model for reproducing the gold exactly.
        #
        # The rule is therefore uniform across all six slots: a slot is scorable
        # on a row IFF the GOLD answer itself states it — tested by running this
        # same extractor over the gold and requiring it to recover the
        # gold_trace value. N/A rows leave BOTH the P and R denominators.
        #
        # The goldoracle arm enforces this: it feeds the gold answers back in as
        # predictions and must score 1.000 on every slot. Anything below 1.000
        # there is a ceiling artifact rather than a model result — which is how
        # area_sqm was caught (multi_crop golds say "for 3.0 acre" and never
        # state m2), along with et0/rainfall on 10 further rows.
        gen_low = to_ascii_digits(gen).lower()
        gold_low = to_ascii_digits(gold).lower()
        gold_ext = extract_numeric_fields(gold)
        gold_has = {
            "kc": all(any(close(g, t.get("kc_used"), TOL["kc"])
                          for g in (gold_ext["kc_all"] or [gold_ext["kc"]])) for t in traces),
            "et0_mm": close(gold_ext["et0_mm"], traces[0].get("et0_mm"), TOL["et0_mm"]),
            "rainfall_mm": close(gold_ext["eff_rain_mm"],
                                 traces[0].get("effective_rain_mm"), TOL["rainfall_mm"]),
            "area_sqm": (traces[0].get("area_sqm") is not None
                         and close(gold_ext["area_sqm"], traces[0].get("area_sqm"),
                                   max(TOL["area_sqm"], abs(traces[0]["area_sqm"]) * 0.005))),
        }
        for slot in SLOTS:
            if slot in gold_has and not gold_has[slot]:
                rec[f"slot_{slot}_na"] = True
            pred_present, correct = False, False
            if slot == "crop":
                names = []
                for t in traces:
                    cn = t.get("crop")
                    names.append([x for x in
                                  ([cn.lower()] if cn else []) +
                                  ([row["crop_bn"].lower()] if len(traces) == 1
                                   and row.get("crop_bn") else []) if x])
                groups = [g for g in names if g]
                if not all(any(n in gold_low for n in g) for g in groups):
                    rec["slot_crop_na"] = True
                pred_present = any(any(n in gen_low for n in g) for g in groups)
                correct = all(any(n in gen_low for n in g) for g in groups)
            elif slot == "stage":
                if not all(any(f.lower() in gold_low for f in
                               STAGE_SURFACE.get(t.get("stage"), [])) for t in traces):
                    rec["slot_stage_na"] = True
                pred_present = any(
                    s in gen_low for forms in STAGE_SURFACE.values() for s in forms)
                correct = all(
                    any(f.lower() in gen_low for f in STAGE_SURFACE.get(t.get("stage"), []))
                    for t in traces)
            elif slot == "kc":
                pred_present = bool(ext["kc_all"]) or ext["kc"] is not None
                got = ext["kc_all"] or ([ext["kc"]] if ext["kc"] is not None else [])
                correct = all(
                    any(close(g, t.get("kc_used"), TOL["kc"]) for g in got) for t in traces)
            elif slot == "et0_mm":
                pred_present = ext["et0_mm"] is not None
                correct = close(ext["et0_mm"], traces[0].get("et0_mm"), TOL["et0_mm"])
            elif slot == "rainfall_mm":
                # The answer states EFFECTIVE rainfall (the equation term), not
                # the raw input, so that is what is scored. See report note.
                pred_present = ext["eff_rain_mm"] is not None
                correct = close(ext["eff_rain_mm"], traces[0].get("effective_rain_mm"),
                                TOL["rainfall_mm"])
            elif slot == "area_sqm":
                # Relative tolerance. Gold areas span 6 m2 to 57000 m2, so a
                # fixed +/-1 m2 is 17% error on the smallest and 0.002% on the
                # largest — it was failing 16609-vs-16592 (0.1% off) while
                # passing 16591. 0.5% or 1 m2, whichever is larger.
                gold_area = traces[0].get("area_sqm")
                pred_present = ext["area_sqm"] is not None
                if gold_area is None:
                    rec[f"slot_{slot}_na"] = True
                    correct = False
                else:
                    correct = close(ext["area_sqm"], gold_area,
                                    max(TOL["area_sqm"], abs(gold_area) * 0.005))
            rec[f"slot_{slot}_pred"] = int(bool(pred_present))
            rec[f"slot_{slot}_correct"] = int(bool(correct))

    # ---------------------------------------------------------- factual
    elif qtype == "factual":
        pn, gn = normalize(gen, lang), normalize(gold, lang)
        rec["pred_norm"], rec["gold_norm"] = pn, gn
        rec["dice_word"] = dice_word(pn, gn)
        rec["dice_char3"] = dice_char(pn, gn, 3)
        rec["dice_char5"] = dice_char(pn, gn, 5)
        rec["rouge_l"] = rouge_l(pn, gn)
        rec["exact_match"] = int(pn == gn)
        # chrF++ and BERTScore are filled in batch by score_arm().

    # ---------------------------------------------------------- clarify
    elif qtype == "clarify":
        s, label = score_clarify(gen, row.get("subtype", ""))
        rec["clarify_score"] = s
        rec["clarify_label"] = label
        rec["clarify_exact"] = int(s == 1.0)
        rec["clarify_partial"] = int(s == 0.5)
        ext = extract_numeric_fields(gen)
        # "Invented a value": produced a net depth while failing to ask for the
        # parameter it did not have. For missing_parameter rows the number is
        # unjustifiable on its face — there was no ET0/rain/area/stage to use.
        rec["emitted_number"] = int(ext["net_mm"] is not None or ext["liters"] is not None)
        rec["invented_value"] = int(pred_cls == "answer")
        rec["invented_value_strict"] = int(
            row["category"] == "missing_parameter"
            and ext["net_mm"] is not None and s < 1.0)

    # ---------------------------------------------------------- abstain
    elif qtype == "abstain":
        rec["abstain_correct"] = int(pred_cls == "abstain")
        ext = extract_numeric_fields(gen)
        rec["emitted_number"] = int(ext["net_mm"] is not None or ext["liters"] is not None)

    else:
        raise AssertionError(f"unknown qtype {qtype!r}")

    # Metric gate: assert nothing leaked across qtype boundaries.
    if qtype in ("clarify", "abstain"):
        for k in rec:
            if k.startswith("slot_") or k in ("dice_word", "dice_char3", "dice_char5",
                                              "rouge_l", "abs_err", "ape"):
                raise AssertionError(
                    f"METRIC GATE: {k} computed on qtype={qtype} "
                    f"(item {row.get('item_id')}) — Dice/slot metrics are "
                    "not defined for clarify/abstain rows")
    if qtype == "factual":
        for k in rec:
            if k.startswith("slot_") or k in ("abs_err", "ape", "clarify_score"):
                raise AssertionError(f"METRIC GATE: {k} computed on qtype=factual")
    if qtype == "numeric_runtime":
        for k in ("dice_word", "rouge_l", "clarify_score"):
            if k in rec:
                raise AssertionError(f"METRIC GATE: {k} computed on qtype=numeric_runtime")
    return rec


# ===========================================================================
# EXTRACTOR SELF-TEST — run the parser against the GOLD answers
# ===========================================================================
def selftest(rows: list[dict]) -> int:
    """If the parser cannot recover gold_trace values from the gold answer text,
    it will silently under-score every model. Validate it before touching a GPU."""
    print("=" * 78)
    print("EXTRACTOR SELF-TEST — parsing the GOLD answers")
    print("=" * 78)

    nr = [r for r in rows if r["qtype"] == "numeric_runtime"]
    single = [r for r in nr if len(gold_traces(r)) == 1]
    multi = [r for r in nr if len(gold_traces(r)) > 1]

    stats = collections.Counter()
    failures = []
    for r in single:
        gt = gold_traces(r)[0]
        ext = extract_numeric_fields(r["answer"])
        checks = {
            "net_mm": close(ext["net_mm"], gt.get("net_mm"), 0.005),
            "kc": close(ext["kc"], gt.get("kc_used"), 0.005),
            "et0_mm": close(ext["et0_mm"], gt.get("et0_mm"), 0.005),
            "eff_rain": close(ext["eff_rain_mm"], gt.get("effective_rain_mm"), 0.005),
            "area_sqm": (close(ext["area_sqm"], gt.get("area_sqm"), 1.0)
                         if gt.get("area_sqm") is not None else None),
            "liters": (close(ext["liters"], gt.get("liters"), max(1.0, abs(gt["liters"]) * 0.001))
                       if gt.get("liters") is not None else None),
        }
        for k, v in checks.items():
            if v is None:
                stats[f"{k}:n/a"] += 1
            elif v:
                stats[f"{k}:ok"] += 1
            else:
                stats[f"{k}:FAIL"] += 1
                if len(failures) < 10:
                    failures.append((r["item_id"], r["lang"], k,
                                     ext.get(k if k != "eff_rain" else "eff_rain_mm"),
                                     gt.get({"net_mm": "net_mm", "kc": "kc_used",
                                             "et0_mm": "et0_mm",
                                             "eff_rain": "effective_rain_mm",
                                             "area_sqm": "area_sqm",
                                             "liters": "liters"}[k]),
                                     r["answer"][:130]))

    print(f"\nsingle-crop numeric rows: {len(single)}")
    for field in ["net_mm", "kc", "et0_mm", "eff_rain", "area_sqm", "liters"]:
        ok, bad, na = stats[f"{field}:ok"], stats[f"{field}:FAIL"], stats[f"{field}:n/a"]
        tot = ok + bad
        pct = 100.0 * ok / tot if tot else float("nan")
        print(f"  {field:10s} ok={ok:4d} FAIL={bad:4d} n/a={na:3d}  recovery={pct:6.2f}%")

    m_ok = 0
    for r in multi:
        gts = gold_traces(r)
        ext = extract_numeric_fields(r["answer"])
        want = [t.get("net_mm") for t in gts]
        got = ext["net_mm_all"]
        if len(got) == len(want) and all(close(g, w, 0.005) for g, w in zip(got, want)):
            m_ok += 1
    print(f"\nmulti_crop rows: {len(multi)}  both nets recovered: {m_ok} "
          f"({100.0 * m_ok / len(multi):.1f}%)")

    # classification of the gold answers themselves
    print("\nresponse classifier on GOLD answers (should be near-perfect):")
    conf = collections.Counter()
    for r in rows:
        if r.get("gold_status") == "pending_human":
            continue
        conf[(GOLD_CLASS[r["qtype"]], classify_response(r["answer"]))] += 1
    for gold_c in ["answer", "clarify", "abstain"]:
        tot = sum(v for (g, _), v in conf.items() if g == gold_c)
        hit = conf[(gold_c, gold_c)]
        row_s = "  ".join(f"{p}={conf[(gold_c, p)]:3d}"
                          for p in ["answer", "clarify", "abstain", "empty"])
        print(f"  gold={gold_c:8s} n={tot:3d}  {row_s}   acc={100.0 * hit / tot:5.1f}%")

    print("\nclarify slot-naming on GOLD answers (should be ~100% named_slot):")
    lab = collections.Counter()
    for r in rows:
        if r["qtype"] != "clarify" or r.get("gold_status") == "pending_human":
            continue
        _, label = score_clarify(r["answer"], r.get("subtype", ""))
        lab[(r["subtype"], label)] += 1
    for st in sorted(set(k[0] for k in lab)):
        parts = {k[1]: v for k, v in lab.items() if k[0] == st}
        print(f"  {st:20s} {parts}")

    if failures:
        print("\nfirst extraction failures:")
        for f in failures:
            print(f"  {f[0]} [{f[1]}] {f[2]}: got={f[3]} want={f[4]}")
            print(f"       {f[5]}")

    n_fail = sum(v for k, v in stats.items() if k.endswith(":FAIL"))
    print("\n" + "-" * 78)
    print(f"SELF-TEST: {n_fail} field-level failures")
    return 0 if n_fail == 0 else 1


# ===========================================================================
# INFERENCE
# ===========================================================================
def arm_spec(arm: str) -> tuple[str, int | None, str | None]:
    """Return (model_key, seed_or_None, run_suffix_or_None)."""
    if arm.endswith("_zeroshot"):
        return arm[:-len("_zeroshot")], None, None
    m = re.match(r"^(.*)_seed(\d+)_clarify$", arm)
    if m:
        return m.group(1), int(m.group(2)), "clarify"
    m = re.match(r"^(.*)_seed(\d+)$", arm)
    if not m:
        raise SystemExit(f"unrecognised arm {arm!r}")
    return m.group(1), int(m.group(2)), None


def all_arms() -> list[str]:
    arms = [f"{m}_zeroshot" for m in MODEL_REGISTRY]
    for m in MODEL_REGISTRY:
        for s in SEEDS:
            arms.append(f"{m}_seed{s}")
    return arms


def count_lora_modules(model) -> int:
    return sum(1 for n, _ in model.named_modules() if "lora_A" in n)


def load_arm(arm: str, max_seq_len: int):
    """Load base 4-bit; attach the S5 adapter for fine-tuned arms.

    Asserts the adapter is actually live. An adapter that silently fails to
    attach would make a fine-tuned arm score exactly like zero-shot, which is
    the one failure mode that looks like a legitimate result.
    """
    from unsloth import FastLanguageModel

    model_key, seed, suffix = arm_spec(arm)
    base_id = MODEL_REGISTRY[model_key]

    if seed is None:
        target, kind = base_id, "base weights (zero-shot)"
    else:
        tag = f"{model_key}_seed{seed}"
        if suffix:
            tag += f"_{suffix}"
        candidates = [ADAPTER_ROOT / tag / "final_adapter"]
        if suffix:
            candidates.insert(0, S6E_ADAPTER_ROOT / tag / "final_adapter")
        adapter = next(
            (p for p in candidates if (p / "adapter_model.safetensors").exists()),
            None,
        )
        if adapter is None:
            raise SystemExit(f"missing adapter: tried {[str(p) for p in candidates]}")
        target, kind = str(adapter), f"S5 adapter {adapter.relative_to(PROJECT_ROOT)}"

    log.info("loading %s  <- %s", arm, kind)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=target,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    n_lora = count_lora_modules(model)
    if seed is None:
        if n_lora:
            raise SystemExit(f"{arm}: zero-shot arm has {n_lora} LoRA modules attached")
        log.info("  adapter check OK: 0 LoRA modules (pure base weights)")
    else:
        if n_lora == 0:
            raise SystemExit(
                f"{arm}: adapter did not attach — 0 LoRA modules found. "
                "This arm would have scored identically to zero-shot.")
        log.info("  adapter check OK: %d LoRA modules live", n_lora)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"     # required for batched generation
    return model, tokenizer, n_lora


def build_prompt(tokenizer, question: str) -> str:
    """Exactly the Stage 5 training shape: a single user turn, no system
    preamble (see s2b_render_train.py). Zero-shot arms get the identical
    prompt, so the only variable between arms is the weights."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)


def generate_arm(arm: str, rows: list[dict], batch_size: int,
                 max_new_tokens: int, max_seq_len: int) -> tuple[list[dict], dict]:
    import torch

    model, tokenizer, n_lora = load_arm(arm, max_seq_len)
    prompts = [build_prompt(tokenizer, r["question"]) for r in rows]

    # Sort by token length so each batch pads to a similar width. Deterministic
    # (stable sort on a fixed list), and the original order is restored below.
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(rows)), key=lambda i: (-lengths[i], i))
    log.info("  prompt tokens: min=%d max=%d mean=%.1f",
             min(lengths), max(lengths), sum(lengths) / len(lengths))

    gens: list[str | None] = [None] * len(rows)
    t0 = time.time()
    n_batches = (len(order) + batch_size - 1) // batch_size

    for bi in range(n_batches):
        idxs = order[bi * batch_size:(bi + 1) * batch_size]
        batch = [prompts[i] for i in idxs]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy == temperature 0
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        for k, i in enumerate(idxs):
            gens[i] = tokenizer.decode(new_tokens[k], skip_special_tokens=True).strip()

        if bi % 5 == 0 or bi == n_batches - 1:
            done = min((bi + 1) * batch_size, len(order))
            el = time.time() - t0
            log.info("  [%s] %d/%d rows  %.1fs elapsed  eta %.1f min",
                     arm, done, len(order), el, (el / done) * (len(order) - done) / 60)

    wall = time.time() - t0
    peak = torch.cuda.max_memory_reserved() / 1024 ** 3
    log.info("  generation done in %.1f min  peak VRAM %.2f GB", wall / 60, peak)

    preds = []
    for r, g in zip(rows, gens):
        preds.append({
            "item_id": r.get("item_id"), "row_id": r.get("row_id"),
            "qtype": r["qtype"], "lang": r["lang"], "category": r["category"],
            "subtype": r.get("subtype"), "gold_status": r.get("gold_status"),
            "question": r["question"], "gold": r.get("answer", ""),
            "generation": g,
        })

    meta = {"arm": arm, "n_lora_modules": n_lora, "wall_seconds": round(wall, 1),
            "peak_vram_gb": round(peak, 3), "batch_size": batch_size,
            "max_new_tokens": max_new_tokens, "rows": len(rows)}

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return preds, meta


# ===========================================================================
# ARM SCORING + AGGREGATION
# ===========================================================================
def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def pct(xs):
    m = mean(xs)
    return None if m is None else 100.0 * m


def score_arm(arm: str, preds: list[dict], rows_by_id: dict,
              use_bertscore: bool = True) -> list[dict]:
    import bn_normalize
    normalize = bn_normalize.normalize

    recs = []
    seen_ids = set()
    for p in preds:
        row = rows_by_id[p["row_id"]]       # row_id, never item_id — see load_test_rows
        if row.get("gold_status") == "pending_human":
            continue          # scope rule: never scored
        if p["row_id"] in seen_ids:
            raise AssertionError(f"{arm}: duplicate row_id {p['row_id']} in predictions")
        seen_ids.add(p["row_id"])
        if p["qtype"] != row["qtype"]:
            raise AssertionError(
                f"{arm}: prediction/gold qtype mismatch on row_id {p['row_id']} "
                f"({p['qtype']} vs {row['qtype']}) — the join is wrong")
        rec = score_row(row, p["generation"] or "", normalize)
        rec["arm"] = arm
        recs.append(rec)

    # ---- batch text metrics for factual rows
    fac = [r for r in recs if r["qtype"] == "factual"]
    if fac:
        import sacrebleu
        chrf = sacrebleu.CHRF(word_order=2)          # chrF++
        for r in fac:
            r["chrf_pp"] = chrf.sentence_score(r["pred_norm"], [r["gold_norm"]]).score

        if use_bertscore:
            try:
                from bert_score import BERTScorer
                scorer = BERTScorer(model_type="bert-base-multilingual-cased",
                                    num_layers=9, rescale_with_baseline=False,
                                    batch_size=32)
                cands = [r["pred_norm"] if r["pred_norm"] else " " for r in fac]
                refs = [r["gold_norm"] for r in fac]
                _, _, F = scorer.score(cands, refs)
                for r, f in zip(fac, F):
                    r["bertscore_f1"] = float(f)
                del scorer
            except Exception as e:                    # noqa: BLE001
                log.warning("BERTScore unavailable (%s) — column left empty", e)
        for r in fac:
            r.pop("pred_norm", None)
            r.pop("gold_norm", None)
    return recs


def summarise(recs: list[dict]) -> dict:
    """Headline numbers for one arm (or one arm x slice)."""
    n = lambda qt: [r for r in recs if r["qtype"] == qt]      # noqa: E731
    numeric, factual, clar, abst = n("numeric_runtime"), n("factual"), n("clarify"), n("abstain")
    single = [r for r in numeric if not r.get("is_multi")]
    multi = [r for r in numeric if r.get("is_multi")]
    extracted = [r for r in single if r.get("pred_net_mm") is not None]

    s = {"n_scored": len(recs), "n_numeric": len(numeric), "n_factual": len(factual),
         "n_clarify": len(clar), "n_abstain": len(abst)}

    # ---- parse / coverage
    s["net_extract_rate"] = pct([r.get("net_extracted", 0) for r in numeric]) if numeric else None
    s["calc_block_rate"] = pct([r["calc_block_present"] for r in recs])
    s["calc_block_parse_ok"] = pct([r["calc_block_parse_ok"] for r in recs
                                    if r["calc_block_present"]]) if any(
        r["calc_block_present"] for r in recs) else None
    s["empty_gen_rate"] = pct([r["gen_empty"] for r in recs])
    s["mean_gen_chars"] = mean([r["gen_chars"] for r in recs])

    # ---- slots
    f1s = []
    for slot in SLOTS:
        # A row where the gold never states the slot is not evidence about the
        # model either way, so it leaves BOTH denominators.
        elig = [r for r in numeric if not r.get(f"slot_{slot}_na")]
        tp = sum(r[f"slot_{slot}_correct"] for r in elig)
        pr = sum(r[f"slot_{slot}_pred"] for r in elig)
        P = tp / pr if pr else 0.0
        R = tp / len(elig) if elig else 0.0
        F = 2 * P * R / (P + R) if (P + R) else 0.0
        s[f"slot_{slot}_P"], s[f"slot_{slot}_R"], s[f"slot_{slot}_F1"] = P, R, F
        s[f"slot_{slot}_n"] = len(elig)
        if elig:
            f1s.append(F)
    s["slot_macro_F1"] = mean(f1s)

    # ---- numeric regression (single-crop only; multi reported separately)
    s["n_single"] = len(single)
    s["n_single_extracted"] = len(extracted)
    s["MAE"] = mean([r["abs_err"] for r in extracted])
    s["RMSE"] = (math.sqrt(mean([r["sq_err"] for r in extracted]))
                 if extracted else None)
    nz = [r for r in extracted if r.get("ape") is not None]
    s["MAPE"] = mean([r["ape"] for r in nz])
    s["MAPE_n"] = len(nz)
    s["zero_gold_frac"] = pct([r["gold_zero"] for r in single]) if single else None
    s["exact_numeric_0p1"] = pct([r.get("em_0p1", 0) for r in single]) if single else None
    s["exact_numeric_1p0"] = pct([r.get("em_1p0", 0) for r in single]) if single else None

    all_errs = [r["abs_err"] for r in extracted]
    for r in multi:
        all_errs.extend(r.get("multi_abs_errs", []))
    s["n_multi"] = len(multi)
    s["multi_both_correct"] = pct([r.get("multi_all_correct", 0) for r in multi]) if multi else None
    s["MAE_all_values"] = mean(all_errs)
    s["MAE_all_values_n"] = len(all_errs)

    # ---- factual
    for k in ["dice_word", "dice_char3", "dice_char5", "rouge_l", "chrf_pp", "bertscore_f1"]:
        s[k] = mean([r.get(k) for r in factual]) if factual else None
    s["exact_factual"] = pct([r["exact_match"] for r in factual]) if factual else None

    # ---- clarify
    s["clarify_score_mean"] = mean([r["clarify_score"] for r in clar]) if clar else None
    s["exact_clarify"] = pct([r["clarify_exact"] for r in clar]) if clar else None
    s["clarify_partial_n"] = sum(r["clarify_partial"] for r in clar)
    s["invented_value_rate"] = pct([r["invented_value"] for r in clar]) if clar else None
    s["invented_value_strict_n"] = sum(r["invented_value_strict"] for r in clar)

    # ---- abstain
    s["abstain_accuracy"] = pct([r["abstain_correct"] for r in abst]) if abst else None
    return s


def confusion(recs: list[dict]) -> dict:
    c = collections.Counter((r["gold_class"], r["pred_class"]) for r in recs)
    classes = ["answer", "clarify", "abstain", "empty"]
    return {g: {p: c[(g, p)] for p in classes} for g in ["answer", "clarify", "abstain"]}


# ===========================================================================
# OUTPUT
# ===========================================================================
LONG_METRIC_KEYS = [
    "abs_err", "sq_err", "ape", "em_0p1", "em_1p0",
    "dice_word", "dice_char3", "dice_char5", "rouge_l", "chrf_pp",
    "bertscore_f1", "exact_match", "clarify_score", "clarify_exact",
    "invented_value", "invented_value_strict", "abstain_correct",
    "net_extracted", "calc_block_present", "multi_all_correct",
] + [f"slot_{s}_correct" for s in SLOTS]


def write_master(recs: list[dict], path: Path, append: bool) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not (append and path.exists())
    with _real_open(path, "a" if append else "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arm", "item_id", "row_id", "qtype", "lang",
                        "category", "subtype", "metric", "value"])
        for r in recs:
            for k in LONG_METRIC_KEYS:
                if k in r and r[k] is not None:
                    w.writerow([r["arm"], r["item_id"], r["row_id"], r["qtype"],
                                r["lang"], r["category"], r.get("subtype"), k, r[k]])


def write_table(path: Path, rows: list[dict], append: bool) -> None:
    import csv
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    new = not (append and path.exists())
    with _real_open(path, "a" if append else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(v, nd=3):
    if v is None:
        return "  --  "
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def print_arm_report(arm: str, s: dict, conf: dict, meta: dict) -> None:
    print("\n" + "=" * 78)
    print(f"ARM REPORT — {arm}")
    print("=" * 78)
    print(f"rows scored : {s['n_scored']}  "
          f"(numeric {s['n_numeric']} / factual {s['n_factual']} / "
          f"clarify {s['n_clarify']} / abstain {s['n_abstain']})")
    print(f"inference   : {meta['wall_seconds'] / 60:.1f} min, "
          f"peak VRAM {meta['peak_vram_gb']:.2f} GB, "
          f"{meta['n_lora_modules']} LoRA modules")

    print("\n-- PARSING")
    print(f"  net_extract_rate (headline parse_success) : {fmt(s['net_extract_rate'], 1)}%")
    print(f"  <irrigation_calc> block emitted           : {fmt(s['calc_block_rate'], 1)}%  "
          f"(v3 gold rate is 0% — see report note)")
    print(f"  empty generations                         : {fmt(s['empty_gen_rate'], 1)}%")
    print(f"  mean generation length                    : {fmt(s['mean_gen_chars'], 0)} chars")

    print("\n-- SLOTS (numeric_runtime only; n = rows whose GOLD states the slot)")
    print(f"  {'slot':14s}{'P':>8}{'R':>8}{'F1':>8}{'n':>6}")
    for slot in SLOTS:
        print(f"  {slot:14s}{fmt(s[f'slot_{slot}_P']):>8}"
              f"{fmt(s[f'slot_{slot}_R']):>8}{fmt(s[f'slot_{slot}_F1']):>8}"
              f"{s.get(f'slot_{slot}_n', 0):>6}")
    print(f"  {'MACRO':14s}{'':>8}{'':>8}{fmt(s['slot_macro_F1']):>8}")

    print("\n-- NUMERIC (net irrigation mm)")
    print(f"  single-crop rows        : {s['n_single']}  "
          f"(value extracted from {s['n_single_extracted']})")
    print(f"  MAE                     : {fmt(s['MAE'])} mm   (n={s['n_single_extracted']})")
    print(f"  RMSE                    : {fmt(s['RMSE'])} mm")
    print(f"  MAPE                    : {fmt(s['MAPE'], 1)}%   "
          f"(n={s['MAPE_n']}, non-zero golds only)")
    print(f"  zero-gold fraction      : {fmt(s['zero_gold_frac'], 1)}%  "
          f"<- context for every exact-match number below")
    print(f"  exact match +/-0.1mm    : {fmt(s['exact_numeric_0p1'], 1)}%")
    print(f"  exact match +/-1.0mm    : {fmt(s['exact_numeric_1p0'], 1)}%")
    print(f"  multi_crop rows         : {s['n_multi']}  "
          f"both values correct: {fmt(s['multi_both_correct'], 1)}%")
    print(f"  MAE over all values     : {fmt(s['MAE_all_values'])} mm  "
          f"(n={s['MAE_all_values_n']} incl. multi_crop)")

    print("\n-- FACTUAL (text similarity)")
    print(f"  dice_word    : {fmt(s['dice_word'])}")
    print(f"  dice_char3   : {fmt(s['dice_char3'])}")
    print(f"  dice_char5   : {fmt(s['dice_char5'])}")
    print(f"  chrF++       : {fmt(s['chrf_pp'], 2)}")
    print(f"  ROUGE-L      : {fmt(s['rouge_l'])}")
    print(f"  BERTScore F1 : {fmt(s['bertscore_f1'])}")
    print("  M5 semantic judge: DEFERRED to Stage 7 human eval (not computed here)")

    print("\n-- CLARIFY / ABSTAIN")
    print(f"  exact_clarify (named the missing slot) : {fmt(s['exact_clarify'], 1)}%")
    print(f"  partial credit (0.5) count             : {s['clarify_partial_n']}")
    print(f"  clarify score mean                     : {fmt(s['clarify_score_mean'])}")
    print(f"  INVENTED_VALUE_RATE (clarify->answer)  : {fmt(s['invented_value_rate'], 1)}%")
    print(f"  invented, strict (missing_parameter)   : {s['invented_value_strict_n']}")
    print(f"  abstain accuracy                       : {fmt(s['abstain_accuracy'], 1)}%")

    print("\n-- EXACT MATCH, THREE COLUMNS (never blended)")
    print(f"  exact_match_factual% : {fmt(s['exact_factual'], 1)}")
    print(f"  exact_match_numeric% : {fmt(s['exact_numeric_0p1'], 1)}   (+/-0.1mm)")
    print(f"  exact_match_clarify% : {fmt(s['exact_clarify'], 1)}")

    print("\n-- CONFUSION (gold rows x predicted class)")
    print(f"  {'gold\\pred':12s}{'answer':>9}{'clarify':>9}{'abstain':>9}{'empty':>8}")
    for g in ["answer", "clarify", "abstain"]:
        r = conf[g]
        print(f"  {g:12s}{r['answer']:>9}{r['clarify']:>9}{r['abstain']:>9}{r['empty']:>8}")


# ===========================================================================
# MAIN
# ===========================================================================
def slice_rows(recs, key, value):
    return [r for r in recs if r[key] == value]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["llama31_8b_zeroshot"],
                    help="arm names, or 'all'")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=GEN_MAX_NEW_TOKENS)
    ap.add_argument("--max_seq_len", type=int, default=INFER_MAX_SEQ_LEN)
    ap.add_argument("--limit", type=int, default=0,
                    help="sanity run on the first N rows; never write scored output")
    ap.add_argument("--selftest", action="store_true",
                    help="validate the extractor against the gold answers and exit")
    ap.add_argument("--score_only", action="store_true",
                    help="re-score existing predictions without inference")
    ap.add_argument("--no_bertscore", action="store_true")
    ap.add_argument("--skip_manifest_verify", action="store_true",
                    help="skip S5 manifest check (use for one-off clarify retrain eval)")
    ap.add_argument("--append", action="store_true",
                    help="append to results CSVs instead of overwriting")
    args = ap.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    # huggingface_hub logs one INFO line per cache HEAD/GET. Across 16 arms that
    # buries the eval log — which is a Gate 6 deliverable — under HTTP chatter.
    for noisy in ("httpx", "huggingface_hub", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.selftest:
        with _real_open(TEST_PATH, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        return selftest(rows)

    gate = entry_gate(skip_manifest=args.skip_manifest_verify)
    rows, scorable = load_test_rows()
    rows_by_id = {r["row_id"]: r for r in rows}

    arms = all_arms() if args.arms == ["all"] else args.arms
    log.info("arms this run: %s", arms)

    gen_rows = rows[:args.limit] if args.limit else rows
    if args.limit:
        log.warning("LIMIT MODE: %d rows, scored output will NOT be written", len(gen_rows))

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows, perlang_rows, percat_rows = [], [], []
    conf_all = {}

    for arm in arms:
        pred_path = PRED_DIR / f"{arm}.jsonl"

        if args.score_only:
            with _real_open(pred_path, encoding="utf-8") as f:
                preds = [json.loads(l) for l in f if l.strip()]
            meta = {"wall_seconds": 0.0, "peak_vram_gb": 0.0, "n_lora_modules": -1}
            log.info("score_only: loaded %d predictions from %s", len(preds), pred_path.name)
        else:
            preds, meta = generate_arm(arm, gen_rows, args.batch_size,
                                       args.max_new_tokens, args.max_seq_len)
            # Raw generations land on disk BEFORE any scoring runs, so a bug in
            # the metrics never costs the inference time again.
            with _real_open(pred_path, "w", encoding="utf-8") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            log.info("predictions -> %s (%d rows)",
                     pred_path.relative_to(PROJECT_ROOT), len(preds))
            with _real_open(REPORT_DIR / f"{arm}_meta.json", "w", encoding="utf-8") as f:
                json.dump({**meta, "entry_gate": gate,
                           "timestamp": datetime.datetime.now().isoformat()},
                          f, indent=2)

        if args.limit:
            log.warning("LIMIT MODE: skipping scoring/aggregation for %s", arm)
            sample = [p for p in preds if p["gold_status"] != "pending_human"][:6]
            for p in sample:
                print("\n" + "-" * 78)
                print(f"[{p['qtype']}/{p['lang']}/{p['category']}] Q: {p['question'][:140]}")
                print(f"GOLD: {p['gold'][:220]}")
                print(f"GEN : {(p['generation'] or '')[:400]}")
            continue

        recs = score_arm(arm, preds, rows_by_id, use_bertscore=not args.no_bertscore)
        assert len(recs) == EXPECTED_SCORABLE, \
            f"{arm}: scored {len(recs)} rows, expected {EXPECTED_SCORABLE}"
        # A bare count assert is not enough: a bad join can hit 362 by accident.
        got = dict(collections.Counter(r["qtype"] for r in recs))
        assert got == EXPECTED_SCORABLE_QTYPES, \
            f"{arm}: scored qtype mix {got} != {EXPECTED_SCORABLE_QTYPES}"

        s = summarise(recs)
        conf = confusion(recs)
        conf_all[arm] = {"overall": conf}
        print_arm_report(arm, s, conf, meta)

        summary_rows.append({"arm": arm, **{k: v for k, v in s.items()}})
        for lang in ["bn", "en"]:
            sub = slice_rows(recs, "lang", lang)
            perlang_rows.append({"arm": arm, "lang": lang, **summarise(sub)})
            conf_all[arm][f"lang_{lang}"] = confusion(sub)
        for cat in sorted(set(r["category"] for r in recs)):
            sub = slice_rows(recs, "category", cat)
            percat_rows.append({"arm": arm, "category": cat, **summarise(sub)})

        write_master(recs, REPORT_DIR / "results_master.csv", args.append or arm != arms[0])

    if not args.limit:
        first = not args.append
        write_table(REPORT_DIR / "summary_by_arm.csv", summary_rows, not first)
        write_table(REPORT_DIR / "per_language.csv", perlang_rows, not first)
        write_table(REPORT_DIR / "per_category.csv", percat_rows, not first)

        cpath = REPORT_DIR / "confusion_matrices.json"
        existing = {}
        if args.append and cpath.exists():
            with _real_open(cpath, encoding="utf-8") as f:
                existing = json.load(f)
        existing.update(conf_all)
        with _real_open(cpath, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        # bn/en gap table (Table B)
        print("\n" + "=" * 78)
        print("TABLE B — bn/en GAP")
        print("=" * 78)
        for arm in arms:
            bn = next(r for r in perlang_rows if r["arm"] == arm and r["lang"] == "bn")
            en = next(r for r in perlang_rows if r["arm"] == arm and r["lang"] == "en")
            print(f"\n{arm}")
            print(f"  {'metric':24s}{'bn':>10}{'en':>10}{'gap(en-bn)':>12}")
            for k in ["net_extract_rate", "slot_macro_F1", "MAE", "MAPE",
                      "exact_numeric_0p1", "dice_word", "chrf_pp", "bertscore_f1",
                      "exact_factual", "exact_clarify", "invented_value_rate",
                      "abstain_accuracy"]:
                b, e = bn.get(k), en.get(k)
                gap = (e - b) if (b is not None and e is not None) else None
                print(f"  {k:24s}{fmt(b):>10}{fmt(e):>10}{fmt(gap):>12}")

        log.info("wrote reports to %s", REPORT_DIR.relative_to(PROJECT_ROOT))

    print("\n" + "=" * 78)
    print("SCOPE REMINDER")
    print("=" * 78)
    print(f"  {EXPECTED_SCORABLE} of {EXPECTED_TOTAL} rows scored; "
          f"60 pending_human golds excluded")
    print("  excluded: 40 out_of_domain (abstain), 20 conflicting_values (clarify)")
    print("  predictions for all 422 rows are on disk — re-run with --score_only")
    print("  M5 semantic judge: DEFERRED to Stage 7 human eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
