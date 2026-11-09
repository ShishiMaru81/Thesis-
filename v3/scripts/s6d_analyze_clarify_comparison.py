#!/usr/bin/env python3
"""Stage 6D — compare clarify performance across all 4 models."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT_ROOT / "reports" / "s6" / "predictions"

MODELS = ["llama31_8b", "mistral_nemo", "phi35_mini", "phi4_mini"]
SEEDS = ["42", "1337", "2024"]


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def clarify_stats(rows: list[dict]) -> tuple[int, int, int]:
    clarify_rows = [
        r for r in rows
        if r.get("qtype") == "clarify" and str(r.get("gold", "")).strip()
    ]
    if not clarify_rows:
        return 0, 0, 0

    # Use pred_class if present (from prior scoring), else classify from generation.
    from s6_eval import classify_response

    correct = 0
    invented = 0
    for r in clarify_rows:
        pred = r.get("pred_class") or classify_response(r.get("generation") or "")
        if pred == "clarify":
            correct += 1
        elif pred == "answer":
            invented += 1
    return correct, invented, len(clarify_rows)


def main() -> None:
    results = []

    for model in MODELS:
        zs_file = PRED_DIR / f"{model}_zeroshot.jsonl"
        rows = load_rows(zs_file)
        if rows:
            correct, invented, total = clarify_stats(rows)
            results.append({
                "model": model,
                "variant": "zeroshot",
                "clarify_correct": correct,
                "invented_value_count": invented,
                "clarify_total": total,
                "clarify_percent": 100 * correct / total if total else 0.0,
                "invented_percent": 100 * invented / total if total else 0.0,
            })

        for seed in SEEDS:
            ft_file = PRED_DIR / f"{model}_seed{seed}.jsonl"
            rows = load_rows(ft_file)
            if not rows:
                continue
            correct, invented, total = clarify_stats(rows)
            results.append({
                "model": model,
                "variant": f"seed_{seed}",
                "clarify_correct": correct,
                "invented_value_count": invented,
                "clarify_total": total,
                "clarify_percent": 100 * correct / total if total else 0.0,
                "invented_percent": 100 * invented / total if total else 0.0,
            })

    # Llama clarify-retrained variant (Stage 6B)
    clarify_file = PRED_DIR / "llama31_8b_seed42_clarify.jsonl"
    rows = load_rows(clarify_file)
    if rows:
        correct, invented, total = clarify_stats(rows)
        results.append({
            "model": "llama31_8b",
            "variant": "seed_42_clarify",
            "clarify_correct": correct,
            "invented_value_count": invented,
            "clarify_total": total,
            "clarify_percent": 100 * correct / total if total else 0.0,
            "invented_percent": 100 * invented / total if total else 0.0,
        })

    print("\n" + "=" * 80)
    print("CLARIFY PERFORMANCE ACROSS ALL 4 MODELS")
    print("=" * 80)
    if not results:
        print("No prediction files found.")
        return

    header = (
        f"{'model':<16} {'variant':<18} {'clarify%':>9} {'invented%':>10} "
        f"{'correct':>8} {'invented':>9} {'total':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<16} {r['variant']:<18} "
            f"{r['clarify_percent']:9.1f} {r['invented_percent']:10.1f} "
            f"{r['clarify_correct']:8d} {r['invented_value_count']:9d} {r['clarify_total']:6d}"
        )

    print("\n" + "=" * 80)
    print("SUMMARY BY MODEL (mean across seeds)")
    print("=" * 80)
    for model in MODELS:
        model_rows = [r for r in results if r["model"] == model]
        if not model_rows:
            continue
        seed_rows = [r for r in model_rows if r["variant"].startswith("seed_") and "clarify" not in r["variant"]]
        print(f"\n{model}:")
        if seed_rows:
            mean_clarify = sum(r["clarify_percent"] for r in seed_rows) / len(seed_rows)
            mean_invented = sum(r["invented_percent"] for r in seed_rows) / len(seed_rows)
            print(f"  Clarify learned (mean):  {mean_clarify:.1f}%")
            print(f"  Invented value (mean):   {mean_invented:.1f}%")
        zs = [r for r in model_rows if r["variant"] == "zeroshot"]
        if zs:
            print(f"  Zero-shot invented:      {zs[0]['invented_percent']:.1f}%")

    out = PROJECT_ROOT / "reports" / "s6" / "clarify_comparison.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print("\nAnalysis complete")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    main()
