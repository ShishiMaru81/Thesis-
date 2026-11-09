#!/usr/bin/env python3
"""Stage 6E — before/after clarify training impact for 3 models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT_ROOT / "reports" / "s6" / "predictions"
SUMMARY = PROJECT_ROOT / "reports" / "s6" / "summary_by_arm.csv"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from s6_eval import classify_response  # noqa: E402

MODELS = [
    ("mistral_nemo", "Mistral-NeMo-12B"),
    ("phi35_mini", "Phi-3.5-mini"),
    ("phi4_mini", "Phi-4-mini"),
]
LLAMA_CLARIFY_ARM = "llama31_8b_seed42_clarify"


def pred_stats(path: Path) -> tuple[int, int, int, float, float]:
    """Return learned, invented, total, exact_clarify%, invented%."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    clarify = [r for r in rows if r.get("qtype") == "clarify" and str(r.get("gold", "")).strip()]
    if not clarify:
        return 0, 0, 0, 0.0, 0.0
    learned = sum(1 for r in clarify if classify_response(r.get("generation") or "") == "clarify")
    invented = sum(1 for r in clarify if classify_response(r.get("generation") or "") == "answer")
    total = len(clarify)
    return learned, invented, total, 100 * learned / total, 100 * invented / total


def summary_metrics(arm: str) -> tuple[float | None, float | None]:
    if not SUMMARY.exists():
        return None, None
    df = pd.read_csv(SUMMARY)
    row = df[df["arm"] == arm]
    if row.empty:
        return None, None
    r = row.iloc[0]
    return float(r["exact_clarify"]), float(r["invented_value_rate"])


def main() -> None:
    results = []

    print("\n" + "=" * 80)
    print("STAGE 6E — CLARIFY TRAINING IMPACT ANALYSIS")
    print("=" * 80)

    for model_id, model_name in MODELS:
        before_path = PRED_DIR / f"{model_id}_seed42.jsonl"
        after_path = PRED_DIR / f"{model_id}_seed42_clarify.jsonl"
        if not before_path.exists() or not after_path.exists():
            print(f"SKIP {model_name}: missing predictions")
            continue

        _, inv_b, n, learn_pct_a, inv_pct_a = pred_stats(after_path)
        learn_pct_b, inv_pct_b = pred_stats(before_path)[3], pred_stats(before_path)[4]
        exact_after, inv_rate_after = summary_metrics(f"{model_id}_seed42_clarify")
        exact_before, inv_rate_before = summary_metrics(f"{model_id}_seed42")

        results.append({
            "Model": model_name,
            "Clarify before": f"{learn_pct_b:.1f}%",
            "Clarify after": f"{learn_pct_a:.1f}%",
            "Clarify Δ": f"+{learn_pct_a - learn_pct_b:.1f} pp",
            "Exact clarify before": f"{exact_before:.1f}%" if exact_before is not None else "—",
            "Exact clarify after": f"{exact_after:.1f}%" if exact_after is not None else "—",
            "Invented before": f"{inv_pct_b:.1f}%",
            "Invented after": f"{inv_pct_a:.1f}%",
            "Invented Δ": f"-{inv_pct_b - inv_pct_a:.1f} pp",
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    llama_exact, llama_inv = summary_metrics(LLAMA_CLARIFY_ARM)

    print("\n" + "=" * 80)
    print("HEADLINE RESULTS")
    print("=" * 80)
    for r in results:
        print(f"{r['Model']:18s}  clarify {r['Clarify after']:>6s}  "
              f"exact {r['Exact clarify after']:>6s}  invented {r['Invented after']:>6s}")

    if llama_exact is not None:
        print(f"\nLlama (Stage 6B):     exact clarify {llama_exact:.1f}%  invented {llama_inv:.1f}%")

    out = PROJECT_ROOT / "reports" / "s6" / "s6e_clarify_impact.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print("\nStage 6E analysis complete")


if __name__ == "__main__":
    main()
