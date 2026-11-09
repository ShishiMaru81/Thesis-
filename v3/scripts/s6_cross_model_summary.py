#!/usr/bin/env python3
"""Build cross-model comparison markdown from reports/s6/."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports" / "s6"
PRED_DIR = REPORT_DIR / "predictions"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from s6_eval import classify_response  # noqa: E402

MODELS = ["llama31_8b", "mistral_nemo", "phi35_mini", "phi4_mini"]
SEEDS = ["42", "1337", "2024"]

METRICS = [
    ("net_extract_rate", "%", 1),
    ("MAE", "mm", 3),
    ("RMSE", "mm", 3),
    ("exact_numeric_0p1", "%", 1),
    ("exact_numeric_1p0", "%", 1),
    ("slot_macro_F1", "", 3),
    ("slot_kc_F1", "", 3),
    ("slot_et0_mm_F1", "", 3),
    ("slot_rainfall_mm_F1", "", 3),
    ("slot_area_sqm_F1", "", 3),
    ("abstain_accuracy", "%", 1),
    ("exact_factual", "%", 1),
    ("dice_word", "", 3),
    ("chrf_pp", "", 1),
    ("bertscore_f1", "", 3),
    ("multi_both_correct", "%", 1),
]

METRIC_LABELS = {
    "net_extract_rate": "net_extract_rate",
    "MAE": "MAE",
    "RMSE": "RMSE",
    "exact_numeric_0p1": "exact ±0.1mm",
    "exact_numeric_1p0": "exact ±1.0mm",
    "slot_macro_F1": "slot macro F1",
    "slot_kc_F1": "kc F1",
    "slot_et0_mm_F1": "et0 F1",
    "slot_rainfall_mm_F1": "rainfall F1",
    "slot_area_sqm_F1": "area_sqm F1",
    "abstain_accuracy": "abstain accuracy",
    "exact_factual": "exact_factual",
    "dice_word": "dice_word",
    "chrf_pp": "chrF++",
    "bertscore_f1": "BERTScore F1",
    "multi_both_correct": "multi_crop both-correct",
}


def fmt_val(val: float | None, kind: str, decimals: int) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    if kind == "%":
        return f"{val:.{decimals}f}"
    return f"{val:.{decimals}f}"


def fmt_mean_std(vals: list[float], kind: str, decimals: int) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return fmt_val(vals[0], kind, decimals)
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt_val(m, kind, decimals)} ± {fmt_val(s, kind, decimals)}"


def load_16_arm_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[~df["arm"].str.contains("_clarify", na=False)]
    expected = {f"{m}_zeroshot" for m in MODELS} | {
        f"{m}_seed{s}" for m in MODELS for s in SEEDS
    }
    df = df[df["arm"].isin(expected)].copy()
    return df


def aggregate_table(df: pd.DataFrame, group_col: str | None = None) -> list[str]:
    """Build markdown rows. group_col=None for overall; 'lang' for bn/en split."""
    lines = []
    header = "| Model | Variant | " + " | ".join(METRIC_LABELS[m] for m, _, _ in METRICS) + " |"
    sep = "|---|---|" + "|".join("---:" for _ in METRICS) + "|"

    groups = [None] if group_col is None else sorted(df[group_col].unique())

    for i, grp in enumerate(groups):
        if group_col:
            lines.append(f"\n### Language: **{grp}**\n")
            sub_all = df[df[group_col] == grp]
        else:
            sub_all = df

        lines.extend([header, sep])

        for model in MODELS:
            # Zero-shot
            zs = sub_all[sub_all["arm"] == f"{model}_zeroshot"]
            if not zs.empty:
                row = zs.iloc[0]
                cells = [fmt_val(row[m], kind, d) for m, kind, d in METRICS]
                lines.append(f"| {model} | zero-shot | " + " | ".join(cells) + " |")

            # Fine-tuned mean ± std
            ft = sub_all[sub_all["arm"].str.match(rf"^{model}_seed\d+$", na=False)]
            if not ft.empty:
                cells = []
                for m, kind, d in METRICS:
                    vals = ft[m].dropna().astype(float).tolist()
                    cells.append(fmt_mean_std(vals, kind, d))
                lines.append(f"| {model} | fine-tuned | " + " | ".join(cells) + " |")

    return lines


def non_answer_clarify_zeroshot() -> list[str]:
    lines = [
        "\n## Non-Llama zero-shot clarify rows NOT classified as `answer`\n",
        "These are the rare zero-shot cases where the model did not invent a numeric answer.\n",
    ]
    models = ["mistral_nemo", "phi35_mini", "phi4_mini"]

    for model in models:
        pred_path = PRED_DIR / f"{model}_zeroshot.jsonl"
        if not pred_path.exists():
            continue
        rows = [
            json.loads(line)
            for line in pred_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        hits = []
        for r in rows:
            if r.get("qtype") != "clarify":
                continue
            if not str(r.get("gold", "")).strip():
                continue
            pred = classify_response(r.get("generation") or "")
            if pred != "answer":
                hits.append((r, pred))

        lines.append(f"\n### {model}_zeroshot ({len(hits)} rows)\n")
        if not hits:
            lines.append("_None._\n")
            continue
        lines.append("| row_id | pred_class | question | generation |")
        lines.append("|---|---|---|---|")
        for r, pred in hits:
            q = (r.get("question") or "").replace("|", "\\|").replace("\n", " ")
            g = (r.get("generation") or "").replace("|", "\\|").replace("\n", " ")
            if len(q) > 120:
                q = q[:117] + "..."
            if len(g) > 160:
                g = g[:157] + "..."
            lines.append(f"| `{r.get('row_id')}` | {pred} | {q} | {g} |")

    return lines


def main() -> None:
    overall_df = load_16_arm_df(REPORT_DIR / "summary_by_arm.csv")
    lang_df = load_16_arm_df(REPORT_DIR / "per_language.csv")

    out: list[str] = [
        "# Stage 6 — Cross-Model Comparison (16 arms)\n",
        "362 scorable rows per arm (60 `pending_human` excluded). ",
        "Fine-tuned = mean ± std across seeds 42, 1337, 2024.\n",
        "\n## Overall (all languages)\n",
    ]
    out.extend(aggregate_table(overall_df))
    out.append("\n## By language\n")
    out.extend(aggregate_table(lang_df, group_col="lang"))
    out.extend(non_answer_clarify_zeroshot())

    text = "\n".join(out) + "\n"
    out_path = REPORT_DIR / "cross_model_summary.md"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
