#!/usr/bin/env python3
"""
irrigation_calc.py — the single source of arithmetic truth for the irrigation
system.

WHY THIS FILE EXISTS
--------------------
Before v2 there was no calculator module. The ETc logic lived inline inside
Json_Dataset(Calander)/generate_qa.py (functions effective_rain /
net_irrigation_mm / the AREA dict), and nothing else could import it. That is
how the v1 corpus and any future runtime code were able to drift apart without
anyone noticing.

Every formula below is lifted VERBATIM from generate_qa.py lines 89-101 and 426.
The only additions are:
  * two area units (see AREA), and
  * introspection helpers (rain_tier, compute) that report intermediate values
    the callers need for grounding metadata.

No formula has been altered. If you change one here, you have changed it for
both the training-target generator and the runtime bridge simultaneously — which
is the entire point.

BARC MIN/MAX CLAMP — DELIBERATELY ABSENT
----------------------------------------
net_irrigation_mm() keeps its lo/hi parameters, but no caller passes them,
exactly as in v1. The per-crop bounds are referenced by every crop JSON as
`computable_link.irrigation_bounds_ref = "irrigation_rules.csv#crop=..."`, but
that CSV was never created and is not on disk. Rather than substitute a
plausible default, compute() reports clamped=False with an explicit
clamp_source explaining the gap. The only bound actually applied is the
physical floor max(net, 0.0), reported separately as floored_at_zero.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

# ONLY EDIT vs v2 scripts/irrigation_calc.py (sha256
# d7d2b43fb2adfc96249b24a4c3803329c805912be48395806c6ff2cbe7cf0a0b):
# the KC_CSV location. v2 lived at Thesis/scripts/, so parent.parent was
# Thesis/. v3 lives at Thesis/v3/scripts/, so parent.parent is Thesis/v3/ and
# the hard-coded path no longer resolves. _find_kc_csv() walks upward instead.
# NO FORMULA, CONSTANT, OR BOUND BELOW THIS POINT WAS TOUCHED — verified by
# diffing everything from `class IrrigationError` onward against the v2 file.


def _find_kc_csv() -> Path:
    """Walk up from this file until kc_values_hybrid.csv is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "kc_values_hybrid.csv"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"kc_values_hybrid.csv not found in any parent of {here}")


KC_CSV = _find_kc_csv()
PROJECT_ROOT = KC_CSV.parent


# ----------------------------------------------------------------- exceptions
class IrrigationError(ValueError):
    """Base class. Every failure carries a machine-readable `reason`."""

    reason = "irrigation_error"


class UnknownCrop(IrrigationError):
    reason = "unknown_crop"


class UnknownStage(IrrigationError):
    reason = "unknown_stage"


class UnknownAreaUnit(IrrigationError):
    reason = "unknown_area_unit"


class OutOfRange(IrrigationError):
    reason = "out_of_range"


# ----------------------------------------------------------------- Kc lookup
def normalize_crop_name(name: str) -> str:
    """kc_values_hybrid.csv uses underscores (Aman_Rice); JSON crop_name_en uses
    spaces (Aman Rice). Normalize both to the same key. Verbatim from
    generate_qa.py:35-39."""
    return name.replace("_", " ").strip().casefold()


def load_kc_table(csv_path: os.PathLike | str | None = None) -> dict:
    """normalized crop name -> Kc row. Verbatim from generate_qa.py:41-54."""
    path = Path(csv_path) if csv_path else KC_CSV
    table = {}
    with open(path, newline="", encoding="utf-8") as f:
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

# Stage strings confirmed from the live corpus, not invented. All 1,008 v1
# numeric records use exactly these three, mapped to the three Kc columns that
# actually exist in kc_values_hybrid.csv. FAO-56's "crop development" stage has
# no Kc column here and is therefore NOT offered.
STAGE_TO_KC = {
    "initial": "kc_ini",
    "mid-season": "kc_mid",
    "late-season": "kc_end",
}
STAGES = tuple(STAGE_TO_KC)

STAGE_BN = {
    "initial": "প্রাথমিক",
    "mid-season": "মধ্য-মৌসুম",
    "late-season": "শেষ-মৌসুম",
}


# ----------------------------------------------------------------- ETc engine
def effective_rain(mm: float) -> float:
    """Verbatim from generate_qa.py:89-92. Boundaries are inclusive-upper:
    5.0 mm is tier 1, 15.0 mm is tier 2."""
    if mm <= 5:
        return round(mm * 0.85, 2)
    if mm <= 15:
        return round(mm * 0.45, 2)
    return round(mm * 0.25, 2)


def rain_tier(mm: float) -> int:
    """Which effective-rainfall tier `mm` falls in. Mirrors effective_rain's
    branch structure exactly so the two can never disagree."""
    if mm <= 5:
        return 1
    if mm <= 15:
        return 2
    return 3


TIER_FACTOR = {1: 0.85, 2: 0.45, 3: 0.25}


def net_irrigation_mm(kc: float, et0: float, rain_mm: float,
                      lo: float | None = None, hi: float | None = None) -> float:
    """Verbatim from generate_qa.py:94-99. lo/hi remain unused — see the module
    docstring on the absent irrigation_rules.csv."""
    net = kc * et0 - effective_rain(rain_mm)
    net = max(net, 0.0)
    if lo is not None:
        net = max(net, lo)
    if hi is not None:
        net = min(net, hi)
    return round(net, 2)


# Area conversions in m². The first five are verbatim from generate_qa.py:101.
#   decimal — 1 decimal is by definition 1/100 acre (the standard Bangladeshi
#             "shatak"). Derived from the existing acre constant, not invented:
#             4046.86 / 100 = 40.4686.
#   ha      — alias for the hectare entry that was already present.
AREA = {
    "acre": 4046.86,
    "bigha": 1337.8,
    "katha": 66.89,
    "decimal": 40.4686,
    "ha": 10000.0,
    "hectare": 10000.0,
    "sqm": 1.0,
}

# Plausibility bounds for parsed input. These reject nonsense (negative rain,
# ET0 of 900) — they are input validation, NOT the absent BARC output clamp.
INPUT_BOUNDS = {
    "et0_mm": (0.0, 20.0),
    "rainfall_mm": (0.0, 500.0),
    "area_value": (0.0, 10000.0),
}


def area_sqm(area_value: float, area_unit: str) -> float:
    unit = str(area_unit).strip().casefold()
    if unit not in AREA:
        raise UnknownAreaUnit(
            f"unknown area unit {area_unit!r}; known: {sorted(AREA)}")
    return area_value * AREA[unit]


def liters_for(net_mm: float, sqm: float) -> float:
    """Verbatim from generate_qa.py:426 — mm depth -> m -> m³ -> liters.
    Algebraically net_mm * sqm, matching computable_link.volume_conversion."""
    return round(net_mm / 1000 * sqm * 1000, 1)


def kc_family(kc_source: str) -> str:
    """Collapse the CSV's verbose kc_source into the two-value family label.
    The verbose original is always carried alongside; nothing is discarded."""
    return "Islam 2021" if kc_source.startswith("Islam") else "FAO-56 Table 12"


def compute(crop: str, stage: str, et0_mm: float, rainfall_mm: float,
            area_value: float, area_unit: str) -> dict:
    """Full deterministic evaluation. Raises an IrrigationError subclass on any
    unusable input; never returns a silently wrong number."""
    kc_row = KC_TABLE.get(normalize_crop_name(str(crop)))
    if kc_row is None:
        raise UnknownCrop(f"no Kc row for crop {crop!r}")

    stage_key = str(stage).strip().casefold()
    if stage_key not in STAGE_TO_KC:
        raise UnknownStage(
            f"unknown stage {stage!r}; known: {list(STAGE_TO_KC)}")

    for name, value in (("et0_mm", et0_mm), ("rainfall_mm", rainfall_mm),
                        ("area_value", area_value)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise OutOfRange(f"{name}={value!r} is not a number")
        low, high = INPUT_BOUNDS[name]
        if not (low <= value <= high):
            raise OutOfRange(f"{name}={value} outside plausible [{low}, {high}]")

    et0_mm, rainfall_mm, area_value = float(et0_mm), float(rainfall_mm), float(area_value)

    kc_col = STAGE_TO_KC[stage_key]
    kc = kc_row[kc_col]
    eff = effective_rain(rainfall_mm)
    etc = round(kc * et0_mm, 2)
    raw = round(etc - eff, 2)
    net_mm = net_irrigation_mm(kc, et0_mm, rainfall_mm)
    sqm = area_sqm(area_value, area_unit)
    liters = liters_for(net_mm, sqm)

    return {
        "crop": crop,
        "stage": stage_key,
        "et0_mm": et0_mm,
        "rainfall_mm": rainfall_mm,
        "area_value": area_value,
        "area_unit": str(area_unit).strip().casefold(),
        "etc": etc,
        "kc_used": kc,
        "kc_stage_key": kc_col,
        "kc_source": kc_row["kc_source"],
        "kc_source_family": kc_family(kc_row["kc_source"]),
        "citation": kc_row["citation"],
        "effective_rain_mm": eff,
        "tier": rain_tier(rainfall_mm),
        "tier_factor": TIER_FACTOR[rain_tier(rainfall_mm)],
        "net_mm": net_mm,
        "area_sqm": round(sqm, 2),
        "liters": liters,
        # Honest reporting of what was and was not bounded:
        "floored_at_zero": raw < 0.0,
        "clamped": False,
        "clamp_source": "none - irrigation_rules.csv absent",
        "formula": "net_mm = max(Kc*ET0 - effective_rain(rain), 0); liters = net_mm * area_sqm",
    }


if __name__ == "__main__":
    import json
    demo = compute("Boro Rice", "mid-season", 4.5, 25.0, 1.0, "acre")
    print(json.dumps(demo, ensure_ascii=False, indent=2))
