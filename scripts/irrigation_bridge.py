#!/usr/bin/env python3
"""
irrigation_bridge.py — parses a model's <irrigation_calc> tool call, runs it
through the deterministic calculator, and templates the result into prose.

This is the ONLY place a computed number becomes a sentence. The model never
writes a number itself (see scripts/train_v2.py's target-architecture
rationale); everything downstream of the tag is arithmetic, not generation.

Nothing here raises. Every function returns a structured failure instead of
propagating an exception, and every failure carries a machine-readable reason
so a caller (or a human reading logs) can tell WHY a row didn't parse or
compute, never just THAT it didn't.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import irrigation_calc as ic  # noqa: E402

TAG_RE = re.compile(r"<irrigation_calc>(.*?)</irrigation_calc>", re.DOTALL)
REQUIRED_KEYS = ("area_unit", "area_value", "crop", "et0_mm", "rainfall_mm", "stage")
NUMERIC_KEYS = ("et0_mm", "rainfall_mm", "area_value")


def _log(msg: str) -> None:
    print(f"[irrigation_bridge] {msg}", file=sys.stderr)


def _parse_tool_call_detailed(model_output: str) -> tuple[dict | None, str | None]:
    """Core parse logic, returning (params, error_reason). error_reason is None
    iff params is not None. Never raises."""
    if not isinstance(model_output, str):
        msg = f"model_output is not a string (got {type(model_output).__name__})"
        _log(msg)
        return None, msg

    match = TAG_RE.search(model_output)
    if not match:
        msg = "<irrigation_calc> block not found"
        _log(msg)
        return None, msg

    inner = match.group(1).strip()
    try:
        obj = json.loads(inner)
    except json.JSONDecodeError as e:
        msg = f"JSON decode error: {e}"
        _log(msg)
        return None, msg

    if not isinstance(obj, dict):
        msg = f"JSON decode error: expected an object, got {type(obj).__name__}"
        _log(msg)
        return None, msg

    missing = sorted(k for k in REQUIRED_KEYS if k not in obj)
    if missing:
        msg = "missing required key: " + ", ".join(missing)
        _log(msg)
        return None, msg

    extra = sorted(set(obj) - set(REQUIRED_KEYS))
    if extra:
        _log(f"ignoring unexpected key(s): {', '.join(extra)}")

    params: dict = {}
    for key in REQUIRED_KEYS:
        value = obj[key]
        if key in NUMERIC_KEYS:
            try:
                value = float(value)
            except (TypeError, ValueError):
                msg = f"key {key!r} is not numeric: {obj[key]!r}"
                _log(msg)
                return None, msg
        else:
            value = str(value)
        params[key] = value

    return params, None


def parse_tool_call(model_output: str) -> dict | None:
    """Extract and parse the <irrigation_calc> JSON block.
    Returns None if absent or malformed. Never raises."""
    params, _ = _parse_tool_call_detailed(model_output)
    return params


def answer_numeric(model_output: str, calculator_func: Callable[..., dict] = ic.compute) -> dict:
    """Parse -> call the calculator -> return a structured result.
    Never raises; a bad model output or a bad crop/stage/value produces a
    populated *_error field, not an exception."""
    params, parse_error = _parse_tool_call_detailed(model_output)

    if params is None:
        return {
            "parsed": False,
            "parse_error": parse_error,
            "params": None,
            "net_mm": None,
            "liters": None,
            "calc_error": None,
            "grounding_applied": None,
        }

    try:
        calc = calculator_func(**params)
    except ic.UnknownCrop:
        calc = None
        calc_error = f"unknown crop: {params['crop']}"
    except ic.UnknownStage:
        calc = None
        calc_error = f"unknown stage: {params['stage']}"
    except ic.UnknownAreaUnit:
        calc = None
        calc_error = f"unknown area unit: {params['area_unit']}"
    except ic.OutOfRange as e:
        calc = None
        calc_error = f"value out of range: {e}"
    except ic.IrrigationError as e:
        calc = None
        calc_error = f"{e.reason}: {e}"
    except Exception as e:  # calculator_func may be a caller-supplied stand-in
        calc = None
        reason = getattr(e, "reason", type(e).__name__)
        calc_error = f"{reason}: {e}"
    else:
        calc_error = None

    if calc is None:
        _log(f"calculator error: {calc_error}")
        return {
            "parsed": True,
            "parse_error": None,
            "params": params,
            "net_mm": None,
            "liters": None,
            "calc_error": calc_error,
            "grounding_applied": None,
        }

    return {
        "parsed": True,
        "parse_error": None,
        "params": params,
        "net_mm": calc["net_mm"],
        "liters": calc["liters"],
        "calc_error": None,
        "grounding_applied": {
            "kc": calc["kc_used"],
            "etc": calc["etc"],
            "effective_rain_mm": calc["effective_rain_mm"],
            "tier": calc["tier"],
            "floored": calc["floored_at_zero"],
        },
    }


_MISSING_KEY_RE = re.compile(r"missing required key: (.+)")


def _fallback_prose(result: dict, lang: str) -> str:
    reason = result.get("parse_error") or result.get("calc_error") or ""
    m = _MISSING_KEY_RE.match(reason)
    missing = [k.strip() for k in m.group(1).split(",")] if m else None

    if lang == "bn":
        if missing:
            return f"পানির চাহিদা হিসাব করা যায়নি; দয়া করে {', '.join(missing)} তথ্য দিন।"
        return ("পানির চাহিদা হিসাব করা যায়নি; দয়া করে ফসল, বৃদ্ধির পর্যায়, ET0, বৃষ্টিপাত "
                "এবং জমির পরিমাণ সঠিকভাবে দিন।")
    if missing:
        return f"I could not compute the water requirement; please provide {', '.join(missing)}."
    return ("I could not compute the water requirement; please provide the crop, growth "
            "stage, ET0, rainfall, and field area.")


def render_prose(result: dict, lang: str) -> str:
    """Template a successful answer_numeric() result into one sentence.
    lang in {'bn', 'en'}. If result['parsed'] is False or calc_error is set,
    returns a neutral fallback naming what's missing when that's known."""
    if not result.get("parsed") or result.get("calc_error"):
        return _fallback_prose(result, lang)

    g = result["grounding_applied"]
    p = result["params"]
    area_value = p["area_value"]
    area_str = f"{area_value:g}"

    if lang == "en":
        return (
            f"Based on {g['kc']:.2f} Kc and {g['etc']:.2f} mm ETc, effective rainfall "
            f"accounts for {g['effective_rain_mm']:.1f} mm, leaving a net requirement of "
            f"{result['net_mm']:.1f} mm ({result['liters']:.0f} liters for your "
            f"{area_str} {p['area_unit']} field)."
        )
    return (
        f"{g['kc']:.2f} Kc এবং {g['etc']:.2f} মিমি ETc হিসাবে, কার্যকর বৃষ্টিপাত "
        f"{g['effective_rain_mm']:.1f} মিমি বাদ দিলে নিট চাহিদা দাঁড়ায় {result['net_mm']:.1f} মিমি "
        f"(আপনার {area_str} {p['area_unit']} জমির জন্য প্রায় {result['liters']:.0f} লিটার)।"
    )


# ----------------------------------------------------------------- test harness
def _run_tests() -> None:
    cases = [
        (
            "Case 1 — success (in-distribution)",
            '<irrigation_calc>{"area_unit": "decimal", "area_value": 5.0, "crop": "Jute", '
            '"et0_mm": 3.8, "rainfall_mm": 35.4, "stage": "initial"}</irrigation_calc>',
        ),
        (
            "Case 2 — missing block",
            "If there is not enough water in the field, irrigate at the 3-4 leaf stage.",
        ),
        (
            "Case 3 — malformed JSON",
            "<irrigation_calc>{broken json}</irrigation_calc>",
        ),
        (
            "Case 4 — missing key (incomplete params)",
            '<irrigation_calc>{"crop":"Boro"}</irrigation_calc>',
        ),
        (
            "Case 5 — unknown crop",
            '<irrigation_calc>{"crop":"FakeCrop", "stage":"mid-season", "et0_mm": 4.0, '
            '"rainfall_mm": 10.0, "area_value": 1.0, "area_unit": "acre"}</irrigation_calc>',
        ),
        (
            # Spec used the shorthand "Boro" for this case; substituted with the real
            # crop name "Boro Rice" so the crop lookup SUCCEEDS and the stage lookup
            # is what actually fails — otherwise UnknownCrop would fire first and the
            # test wouldn't exercise the stage-validation path at all.
            "Case 6 — unknown stage (crop corrected to 'Boro Rice' so crop lookup "
            "succeeds and the stage error is what's actually exercised)",
            '<irrigation_calc>{"crop":"Boro Rice", "stage":"FakeStage", "et0_mm": 4.0, '
            '"rainfall_mm": 10.0, "area_value": 1.0, "area_unit": "acre"}</irrigation_calc>',
        ),
    ]

    for label, model_output in cases:
        print("=" * 88)
        print(label)
        print("=" * 88)
        print(f"input: {model_output}")

        result = answer_numeric(model_output)
        print(f"\nanswer_numeric() result:\n{json.dumps(result, ensure_ascii=False, indent=2)}")

        prose_en = render_prose(result, "en")
        prose_bn = render_prose(result, "bn")
        print(f"\nrender_prose(en): {prose_en}")
        print(f"render_prose(bn): {prose_bn}")
        print()


if __name__ == "__main__":
    _run_tests()
