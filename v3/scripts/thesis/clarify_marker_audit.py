#!/usr/bin/env python3
"""
clarify_marker_audit.py — post-hoc, descriptive audit of the clarify scorer.

Found while writing the thesis (2026-09-24): on missing-stage rows the
clarify-trained models answer "Which growth stage is your <crop> in? ...",
a correct request, but the registered request markers in s6_eval.py include
"which stage" and not "which growth stage", so classify_response() returns
"answer" and score_clarify() returns 0. The registered metrics are NOT changed.

This script counts, for every arm, the clarify rows that the registered scorer
did not credit (clarify exact < 1) but whose response contains a question mark
and states no net depth and no volume, a conservative sign that the model asked
instead of computing. It reads saved predictions only (read-only) and writes
thesis/generated/clarify_marker_audit.json, which make_numbers.py turns into
macros. The counts are descriptive and are never used in a significance test.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from common import GEN, MODELS, PRED_DIR, ROOT, load_jsonl, rel, verify_test_sha, write_json

sys.path.insert(0, str(ROOT / "scripts"))
from s6_eval import extract_numeric_fields, score_clarify  # noqa: E402


def audit(arm: str) -> dict:
    rows = [r for r in load_jsonl(PRED_DIR / f"{arm}.jsonl") if r["qtype"] == "clarify" and str(r.get("gold", "")).strip()]
    exact, unrecognised = 0, collections.Counter()
    example = {}
    for r in rows:
        g = r.get("generation") or ""
        if score_clarify(g, r.get("subtype", ""))[0] == 1.0:
            exact += 1
            continue
        ext = extract_numeric_fields(g)
        asks = ("?" in g or "？" in g) and ext["net_mm"] is None and ext["liters"] is None
        if asks:
            unrecognised[r.get("subtype", "")] += 1
            example.setdefault(r.get("subtype", ""), {"row_id": r["row_id"], "generation": g[:300]})
    return {"n": len(rows), "exact": exact, "asks_unrecognised": sum(unrecognised.values()),
            "asks_unrecognised_by_subtype": dict(sorted(unrecognised.items())), "examples": example}


def main() -> None:
    verify_test_sha()
    out = {"note": __doc__.strip().splitlines()[2], "arms": {}}
    for key, _, _ in MODELS:
        for v in ("zeroshot", "seed42", "seed1337", "seed2024", "seed42_clarify"):
            arm = f"{key}_{v}"
            out["arms"][arm] = audit(arm)
    write_json(GEN / "clarify_marker_audit.json", out)
    for arm, d in out["arms"].items():
        print(f"{arm:28s} exact {d['exact']:2d}/{d['n']}  asks-unrecognised {d['asks_unrecognised']:2d} {d['asks_unrecognised_by_subtype']}")
    print("->", rel(GEN / "clarify_marker_audit.json"))


if __name__ == "__main__":
    main()
