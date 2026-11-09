#!/usr/bin/env python3
"""Stage 9 Step 3: tests F1-F6 and descriptive CIs (s8_statlib imported unchanged).

  python stats_frontier.py

Reads results/stage9_frontier/scores/per_row.csv; writes stats/tests.csv and stats/descriptive_ci.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT  # noqa: F401  (also puts v3/scripts on sys.path)
import s8_statlib as sl

PLAN = json.loads((Path(__file__).resolve().parent / "plan.json").read_text(encoding="utf-8"))
SCORES = OUT / "scores"
STATS = OUT / "stats"
ARM_OF = {"frontier ZS": "gptoss_ZS", "frontier ZS+P": "gptoss_ZS+P", "llama31_8b_seed42": "llama_FT",
          "llama31_8b_seed42_clarify": "llama_CL"}
METRIC_COL = {"confident_error_a": "ce_a", "confident_error_b": "ce_b"}
SUBSET_COL = {"clarify_68": "in_clarify_68", "numeric_146": "in_numeric_146", "numeric_single_116": "in_single_116"}
CI_METRICS = [
    ("invented_value", "in_clarify_68"), ("clarify_exact", "in_clarify_68"), ("clarify_learned", "in_clarify_68"),
    ("ce_a", "in_clarify_68"), ("ce_b", "in_clarify_68"), ("ce_a_512", "in_clarify_68"), ("ce_b_512", "in_clarify_68"),
    ("truncated_primary", "in_clarify_68"), ("over_512", "in_clarify_68"), ("over_512", "all_288"),
    ("em_0p1", "in_single_116"), ("exact_0p005", "in_single_116"), ("em_1p0", "in_single_116"),
    ("over_asking", "in_numeric_146"), ("abstain_correct", "in_abstain_seed_34"), ("abstain_correct", "in_ood_hard_40"),
    ("response_bn", "bn_rows"),
]


def vec(df: pd.DataFrame, arm: str, metric: str, subset_col: str) -> pd.Series:
    s = df[(df["arm"] == arm) & (df[subset_col] == 1)].set_index("row_id")[metric]
    return s.sort_index()


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for t in PLAN["tests"]:
        if t.get("role") == "DESCRIPTIVE":
            continue
        col = SUBSET_COL[t["rows"]]
        m = METRIC_COL.get(t["metric"], t["metric"])
        b = vec(df, ARM_OF[t["B"]], m, col)
        a = vec(df, ARM_OF[t["A"]], m, col)
        assert list(b.index) == list(a.index) and b.notna().all() and a.notna().all(), t["id"]
        mc = sl.mcnemar_exact(b.values, a.values)
        diff, lo, hi = sl.paired_boot_diff(b.values, a.values)
        recs.append({"test_id": t["id"], "holm_family": t["holm_family"], "metric": t["metric"], "rows": t["rows"],
                     "B": t["B"], "A": t["A"], "model": "openai/gpt-oss-120b", "n": mc["n"],
                     "rate_b": mc["rate_b"], "rate_a": mc["rate_a"], "k_b": int(b.sum()), "k_a": int(a.sum()),
                     "b_only": mc["b"], "a_only": mc["c"], "both": mc["both"], "neither": mc["neither"],
                     "diff": diff, "diff_ci_low": lo, "diff_ci_high": hi, "p_raw": mc["p"],
                     "odds_ratio": mc["odds_ratio"], "or_lo": mc["or_lo"], "or_hi": mc["or_hi"],
                     "not_testable": mc.get("not_testable")})
    out = pd.DataFrame(recs)
    out["p_holm"] = None
    for fam, g in out.groupby("holm_family"):
        adj = sl.holm(list(g["p_raw"]))
        out.loc[g.index, "p_holm"] = adj
    out["verdict"] = [
        sl.verdict(r.p_holm, int(np.sign(r.diff)) if r.diff else 0, None,
                   sl.fmt_ci(r.diff_ci_low, r.diff_ci_high, pct=True), r.B, r.A, not_testable=r.not_testable)
        for r in out.itertuples()]
    return out


def descriptive(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for arm, g in df.groupby("arm", sort=False):
        for metric, subset in CI_METRICS:
            if metric not in g:
                continue
            s = g if subset == "all_288" else (g[g["lang"] == "bn"] if subset == "bn_rows" else g[g[subset] == 1])
            v = s[metric].dropna().astype(float).values
            if len(v) == 0:
                continue
            pt, lo, hi = sl.boot_ci(v)
            recs.append({"arm": arm, "metric": metric, "subset": subset, "n": len(v), "k": int(v.sum()),
                         "rate": pt, "ci_low": lo, "ci_high": hi, "n_missing": int(len(s) - len(v))})
    f6 = next(t for t in PLAN["tests"] if t["id"] == "F6")
    for r in recs:
        r["f6_arm"] = int(r["metric"] == "em_0p1" and r["subset"] == "in_single_116"
                          and r["arm"] in {ARM_OF[a] for a in f6["arms"]})
    return pd.DataFrame(recs)


def main() -> int:
    df = pd.read_csv(SCORES / "per_row.csv", encoding="utf-8")
    STATS.mkdir(parents=True, exist_ok=True)
    tests = run_tests(df)
    tests.to_csv(STATS / "tests.csv", index=False, encoding="utf-8")
    desc = descriptive(df)
    desc.to_csv(STATS / "descriptive_ci.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 250)
    print(tests[["test_id", "metric", "B", "A", "n", "rate_b", "rate_a", "b_only", "a_only", "diff",
                 "diff_ci_low", "diff_ci_high", "p_raw", "p_holm", "verdict"]].to_string(index=False))
    print("\nF6 (descriptive):")
    print(desc[desc["f6_arm"] == 1][["arm", "n", "k", "rate", "ci_low", "ci_high"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
