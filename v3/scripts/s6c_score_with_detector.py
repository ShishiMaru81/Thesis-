#!/usr/bin/env python3
"""Stage 6C — rescore clarify rows with post-hoc detector applied."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def score_with_detector(pred_file: Path) -> None:
    rows = [
        json.loads(line)
        for line in pred_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    clarify_rows = [
        r for r in rows
        if r.get("qtype") == "clarify" and str(r.get("gold", "")).strip()
    ]

    learned_correct = 0
    detector_correct = 0
    still_invented = 0

    for row in clarify_rows:
        pred_qtype = row.get("predicted_qtype")
        detector_flag = row.get("detector_flag")

        if pred_qtype == "clarify":
            learned_correct += 1
        elif detector_flag:
            detector_correct += 1
        else:
            still_invented += 1

    total = len(clarify_rows)
    combined = learned_correct + detector_correct

    print(f"\n{'=' * 80}")
    print("HYBRID CLARIFICATION SCORING")
    print(f"{'=' * 80}")
    print(f"Clarify rows (n={total}):")
    if total:
        print(f"  Model learned: {learned_correct:3d} ({100 * learned_correct / total:5.1f}%)")
        print(f"  Detector caught: {detector_correct:3d} ({100 * detector_correct / total:5.1f}%)")
        print(f"  Combined safe:   {combined:3d} ({100 * combined / total:5.1f}%)")
        print(f"  Still invented:  {still_invented:3d} ({100 * still_invented / total:5.1f}%)")
    print(f"\nStage 6C complete")


def main() -> None:
    pred_file = PROJECT_ROOT / "reports/s6/predictions/llama31_8b_seed42_clarify_with_detector.jsonl"
    if not pred_file.exists():
        raise SystemExit(f"missing input: {pred_file}")
    score_with_detector(pred_file)


if __name__ == "__main__":
    main()
