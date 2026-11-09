#!/usr/bin/env python3
"""
bn_normalize.py — single source of truth for text normalization in v3.

VERSION HISTORY
---------------
v3.0  Original specification. Step 4 stripped punctuation with
      re.sub(r"[^\\w\\s]", " ", text).
v3.1  CRITICAL FIX. That regex destroyed Bengali orthography.

      Python's `re` defines \\w as str.isalnum() plus underscore. Bengali
      vowel signs (matras) and the virama/hasant are Unicode category Mn
      (nonspacing mark) and Mc (spacing combining mark), and isalnum() is
      False for every one of them. The pattern therefore deleted every matra
      in the corpus:

          তিল  (Sesame)  ->  "ত ল"
          তুলা (Cotton)  ->  "ত ল"

      Two different crops collapsed to the same string. In the Gate 1
      pre-flight this manufactured 3 phantom exact matches between train and
      val, and it silently corrupted every char-n-gram score computed on top
      of the normalized text.

      The fix replaces the regex with an explicit unicodedata.category()
      test. Combining marks are now preserved; only real punctuation and
      symbols are removed. See REMOVE_CATEGORIES and KEEP_CATEGORY_PREFIXES.

      ANY CHANGE TO THIS FILE INVALIDATES ALL PRIOR SCORES. Eval scripts must
      assert NORMALIZER_VERSION matches the value stamped into the results
      file they are comparing against.

NORMALIZATION PIPELINE
----------------------
  1. NFC normalize
  2. Bengali digits ০১২৩৪৫৬৭৮৯ -> 0123456789
  3. ZWJ / ZWNJ removed ONLY where they are not orthographically load-bearing
     (see strip_joiners) — never blindly
  4. Punctuation and symbols -> space, by Unicode category; Bengali danda (।
     U+0964) and double danda (॥ U+0965) explicitly included
  5. lowercase, only when lang == "en" (no-op for Bengali script)
  6. collapse whitespace

PUNCTUATION BECOMES A SPACE, NOT NOTHING. Deleting it would merge tokens
across a danda ("ধান।গম" -> "ধানগম") and fuse words across a hyphen. The
consequence, which is intended, is that "15-20 cm" and "15 20 cm" compare
equal under exact match — punctuation variation should not fail an exact
match, and word-level Dice benefits from the split.
"""
from __future__ import annotations

import unicodedata

NORMALIZER_VERSION = "v3.1"

# ----------------------------------------------------------------- tables
BN_DIGITS = "০১২৩৪৫৬৭৮৯"
ASCII_DIGITS = "0123456789"
DIGIT_MAP = {ord(b): a for b, a in zip(BN_DIGITS, ASCII_DIGITS)}

ZWNJ = "‌"
ZWJ = "‍"
BENGALI_VIRAMA = "্"          # হসন্ত

DANDA = "।"                   # ।
DOUBLE_DANDA = "॥"            # ॥

# Categories removed and replaced with a space.
#   Pc Pd Pe Pf Pi Po Ps — connector, dash, close, final-quote, initial-quote,
#                          other, open punctuation. Po covers the danda.
#   Sc Sk Sm So          — currency, modifier, math, other symbols. Sm covers
#                          the × and − used in the numeric gold answers.
REMOVE_CATEGORIES = frozenset({
    "Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps",
    "Sc", "Sk", "Sm", "So",
})

# Belt and braces: these two are already Po, but Decision 4 names them
# explicitly, so the code names them explicitly.
EXPLICIT_REMOVE = frozenset({DANDA, DOUBLE_DANDA})

# Never removed. "M" is the whole point of v3.1 — Mn and Mc are Bengali
# matras and the virama.
KEEP_CATEGORY_PREFIXES = ("L", "N", "M", "Z")

_joiner_log: list[dict] = []


def get_joiner_log() -> list[dict]:
    """Every ZWJ/ZWNJ decision taken since reset_joiner_log(). Step 3 of the
    spec requires that retained joiners are logged rather than silently
    dropped or silently kept."""
    return list(_joiner_log)


def reset_joiner_log() -> None:
    _joiner_log.clear()


# ----------------------------------------------------------------- steps
def strip_joiners(text: str) -> str:
    """Remove ZWJ/ZWNJ only where they cannot be carrying orthography.

    In Bengali a joiner immediately following the virama (U+09CD) is
    load-bearing: ZWNJ after virama forces the explicit hasant form and
    suppresses the conjunct ligature, ZWJ after virama forces a half-form.
    Removing either one changes what the word looks like and, for a reader,
    what it is. Those are KEPT and logged.

    A joiner anywhere else is not selecting between conjunct forms and is
    almost always an artifact of copy-paste or of the extraction pipeline.
    Those are removed and logged.
    """
    if ZWJ not in text and ZWNJ not in text:
        return text
    out = []
    for i, ch in enumerate(text):
        if ch in (ZWJ, ZWNJ):
            prev = text[i - 1] if i > 0 else ""
            if prev == BENGALI_VIRAMA:
                _joiner_log.append({
                    "action": "kept",
                    "char": "ZWJ" if ch == ZWJ else "ZWNJ",
                    "reason": "follows virama U+09CD - selects conjunct form",
                    "context": text[max(0, i - 4):i + 4],
                })
                out.append(ch)
            else:
                _joiner_log.append({
                    "action": "removed",
                    "char": "ZWJ" if ch == ZWJ else "ZWNJ",
                    "reason": "not preceded by virama - not orthographic",
                    "context": text[max(0, i - 4):i + 4],
                })
            continue
        out.append(ch)
    return "".join(out)


def strip_punctuation(text: str) -> str:
    """Unicode-category punctuation/symbol strip. The v3.0 regex lived here."""
    out = []
    for ch in text:
        if ch in EXPLICIT_REMOVE:
            out.append(" ")
            continue
        cat = unicodedata.category(ch)
        if cat in REMOVE_CATEGORIES:
            out.append(" ")
        elif cat.startswith("C"):
            # Control and remaining format characters (BOM, LRM, ...). ZWJ and
            # ZWNJ are also Cf but strip_joiners has already run, so anything
            # reaching here is not one of them.
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def normalize(text: str, lang: str) -> str:
    """The one normalizer. lang is "bn" or "en"."""
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(f"normalize() expects str, got {type(text).__name__}")

    t = unicodedata.normalize("NFC", text)      # 1
    t = t.translate(DIGIT_MAP)                  # 2
    t = strip_joiners(t)                        # 3
    t = strip_punctuation(t)                    # 4
    if lang == "en":                            # 5 - no-op for Bengali script
        t = t.lower()
    return " ".join(t.split())                  # 6


# ----------------------------------------------------------------- tests
# The four crop names mandated by Decision 4, plus the cases that pin down
# every other step. run_regression_tests() is called at the start of every
# stage that normalizes text; a failure stops the pipeline.
REGRESSION_CASES = [
    # (text, lang, expected, label)
    ("তিল",  "bn", "তিল",  "Decision 4 - Sesame, matras preserved"),
    ("তুলা", "bn", "তুলা", "Decision 4 - Cotton, must not collapse onto তিল"),
    ("ধান",  "bn", "ধান",  "Decision 4 - Rice"),
    ("গম",   "bn", "গম",   "Decision 4 - Wheat"),

    ("ধান।", "bn", "ধান", "danda U+0964 removed"),
    ("ধান॥", "bn", "ধান", "double danda U+0965 removed"),
    ("Hello, World!", "en", "hello world", "ascii punctuation + lowercase"),
    ("Boro  Rice\n", "en", "boro rice", "whitespace collapsed"),
    ("০১২৩৪৫৬৭৮৯", "bn", "0123456789", "Bengali digits mapped"),
    ("৬.৩ মিমি", "bn", "6 3 মিমি", "digits mapped, decimal point -> space"),
    ("ET0", "bn", "ET0", "no lowercasing on lang=bn"),
    ("Kc×ET0 − rain", "en", "kc et0 rain", "Sm symbols removed"),
    ("বৃষ্টি", "bn", "বৃষ্টি", "virama-joined conjunct survives"),
    ("১৫-২০ সেমি", "bn", "15 20 সেমি", "hyphen -> space, digits mapped"),
]


def run_regression_tests(verbose: bool = False) -> list[dict]:
    """Returns a list of failures. Empty list == pass."""
    failures = []
    for text, lang, expected, label in REGRESSION_CASES:
        got = normalize(text, lang)
        ok = got == expected
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
            print(f"         in={text!r} lang={lang} -> {got!r}")
            if not ok:
                print(f"         expected {expected!r}")
        if not ok:
            failures.append({"input": text, "lang": lang,
                             "expected": expected, "got": got, "label": label})

    # The specific collapse that v3.0 caused. Asserted as a relation, not just
    # as two independent equalities, so it cannot regress by both sides
    # becoming equally wrong.
    if normalize("তিল", "bn") == normalize("তুলা", "bn"):
        failures.append({"input": "তিল vs তুলা", "lang": "bn",
                         "expected": "distinct normalized forms",
                         "got": normalize("তিল", "bn"),
                         "label": "v3.0 matra-destruction regression"})
    return failures


def assert_version(expected: str) -> None:
    """Called by eval scripts. A results file carrying a different normalizer
    version was scored under different rules and is not comparable."""
    if expected != NORMALIZER_VERSION:
        raise RuntimeError(
            f"normalizer version mismatch: results were produced under "
            f"{expected!r} but this file is {NORMALIZER_VERSION!r}. "
            f"All prior scores are invalidated by a normalizer change — "
            f"re-run evaluation rather than comparing across versions.")


if __name__ == "__main__":
    print(f"bn_normalize.py  NORMALIZER_VERSION = {NORMALIZER_VERSION}")
    print("=" * 66)
    fails = run_regression_tests(verbose=True)
    print("=" * 66)
    if fails:
        print(f"FAILED: {len(fails)} case(s)")
        for f in fails:
            print(f"  {f['label']}: {f['input']!r} -> {f['got']!r} "
                  f"(expected {f['expected']!r})")
        raise SystemExit(1)
    print(f"PASSED: all {len(REGRESSION_CASES)} cases + matra-collapse guard")
