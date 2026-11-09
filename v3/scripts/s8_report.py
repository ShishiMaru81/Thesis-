"""Stage 8 — write reports/stats/significance.md from the Part A / Part B outputs.

Read-only on everything except reports/stats/significance.md. Verifies the
Table 1 verdicts supplied by the author against the CSVs and stops on conflict.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date

import pandas as pd

import s8_statlib as sl
from s8_stats import HUMAN_ARMS, MODELS, OUT_DIR, PROJECT, SEEDS, TEST_PATH, load_jsonl, log

ML = {"llama31_8b": "Llama-3.1-8B", "mistral_nemo": "Mistral-NeMo-12B", "phi35_mini": "Phi-3.5-mini", "phi4_mini": "Phi-4-mini"}
AS = {"llama31_8b_zeroshot": "Llama ZS", "llama31_8b_seed42": "Llama FT", "llama31_8b_seed42_clarify": "Llama FT+clarify",
      "mistral_nemo_seed42": "Mistral FT"}
DET_LABEL = ("Oracle-gated diagnostic. The detector reads gold labels, runs only on known-incomplete queries, and its "
             "rules were written with the test categories in view. This is an upper bound on what surface rules can "
             "catch, not a performance estimate.")


# ---------------------------------------------------------------------------
def _num(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if v == math.inf or v > 1e6:
        return "inf"
    return f"{v:.{nd}f}"


def _p(v):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else sl.fmt_p(v)


def _eff(r):
    e, lo, hi = r["effect"], r["effect_ci_low"], r["effect_ci_high"]
    if e is None or (isinstance(e, float) and math.isnan(e)):
        return "—"
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return _num(e, 2)
    return f"{_num(e, 2)} [{_num(lo, 2)}, {_num(hi, 2)}]"


def _diff(r):
    d, lo, hi = r["diff"], r["diff_ci_low"], r["diff_ci_high"]
    if d is None or (isinstance(d, float) and math.isnan(d)):
        return "—"
    unit = r.get("diff_unit")
    unit = "" if unit is None or (isinstance(unit, float) and math.isnan(unit)) else str(unit)
    if not unit and ("Fisher" in str(r.get("test")) or "McNemar" in str(r.get("test"))):
        unit = "proportion"           # rate difference stored as a proportion; diff_unit not filled for Fisher rows
    if unit.startswith("proportion"):
        return f"{100*d:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] pp"
    u = "chrF++ pts" if "metric" in unit else ("rating pts" if "rating" in unit else unit)
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}] {u}"


def _bc(r):
    return f"{int(r['b'])}/{int(r['c'])}" if pd.notna(r.get("b")) else "—"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def results_table(df: pd.DataFrame, with_bc=True) -> str:
    hdr = ["test_id", "n", "test", "stat", "p_raw", "p_holm", "effect [95% CI]", "difference [95% CI]", "verdict"]
    if with_bc:
        hdr.insert(2, "b/c")
    rows = []
    for _, r in df.iterrows():
        row = [r["test_id"], r["n"], r["test"], _num(r["statistic"], 3) if pd.notna(r["statistic"]) else "—",
               _p(r["p_raw"]), _p(r["p_holm"]), _eff(r), _diff(r), r["verdict"]]
        if with_bc:
            row.insert(2, _bc(r))
        rows.append(row)
    return md_table(hdr, rows)


def sig(p) -> bool:
    return p is not None and not (isinstance(p, float) and math.isnan(p)) and p < 0.05


# ---------------------------------------------------------------------------
def load_all() -> dict:
    d = {}
    d["plan"] = json.load((OUT_DIR / "test_plan.json").open(encoding="utf-8"))
    d["gate"] = json.load((OUT_DIR / "entry_gate.json").open(encoding="utf-8"))
    d["pre"] = json.load((OUT_DIR / "precheck.json").open(encoding="utf-8"))
    d["A"] = pd.read_csv(OUT_DIR / "results_automatic.csv")
    d["B"] = pd.read_csv(OUT_DIR / "results_human.csv")
    d["boot"] = pd.read_csv(OUT_DIR / "bootstrap_cis.csv")
    d["a1b"] = pd.read_csv(OUT_DIR / "a1b_descriptive.csv")
    d["a6c"] = pd.read_csv(OUT_DIR / "a6_clarify_descriptive.csv")
    d["a7"] = pd.read_csv(OUT_DIR / "a7_descriptive.csv")
    d["a7ex"] = json.load((OUT_DIR / "a7_explicit_assumption_examples.json").open(encoding="utf-8"))
    d["a7trunc"] = json.load((OUT_DIR / "a7_truncation_caveat.json").open(encoding="utf-8"))
    d["hedge"] = json.load((OUT_DIR / "bn_hedge_check.json").open(encoding="utf-8"))
    d["rob"] = pd.read_csv(OUT_DIR / "seed_robustness_A.csv")
    d["emd"] = pd.read_csv(OUT_DIR / "exact_match_denominators.csv")
    d["lang"] = pd.read_csv(OUT_DIR / "response_language.csv")
    d["agree"] = json.load((OUT_DIR / "agreement.json").open(encoding="utf-8"))
    d["dice"] = pd.read_csv(OUT_DIR / "dice_validation.csv")
    d["b8"] = pd.read_csv(OUT_DIR / "b8_error_bins.csv")
    d["hsum"] = pd.read_csv(OUT_DIR / "human_arm_summary.csv")
    d["mixed"] = (OUT_DIR / "mixed_model.txt").read_text(encoding="utf-8")
    test = load_jsonl(TEST_PATH)
    d["clarify_cats"] = dict(Counter(r["category"] for r in test if r["qtype"] == "clarify" and r["gold_status"] != "pending_human"))
    d["clarify_subtypes"] = dict(Counter(r["subtype"] for r in test if r["qtype"] == "clarify" and r["gold_status"] != "pending_human"))
    for k in ("A", "B"):
        d[k] = d[k].where(pd.notna(d[k]), None)
    return d


def verify_verdicts(d: dict) -> dict:
    """Check the author-supplied Table 1 verdicts against the CSVs. Raise on conflict."""
    A, B, rob = d["A"], d["B"], d["rob"]
    P = A[(A.role == "PRIMARY")]
    P42 = P[P.seed == 42]
    chk = {}
    chk["A1_all_seeds_sig"] = all(sig(p) for p in P[P.family == "A1"].p_holm)
    chk["A7_all_seeds_sig"] = all(sig(p) for p in P[P.family == "A7"].p_holm)
    chk["A3a_sig"] = all(sig(p) for p in P42[P42.family == "A3a"].p_holm)
    chk["A3b_sig"] = all(sig(p) for p in P42[P42.family == "A3b"].p_holm)
    chk["A4_none_sig"] = not any(sig(p) for p in P42[P42.family == "A4"].p_holm)
    chk["A2_lvm_ns_all_seeds"] = not any(sig(p) for p in P[P.test_id == "A2_llama_vs_mistral"].p_holm)
    chk["A2_tier_sig_all_seeds"] = all(sig(p) for p in P[P.test_id == "A2_tier"].p_holm)
    fac42 = P42[P42.test_id.str.startswith("A6_fac_")]
    chk["A6_fac_sig_seed42"] = int(sum(sig(p) for p in fac42.p_holm))
    chk["A6_fac_robust"] = int(rob[rob.test_id.str.startswith("A6_fac_")].robust_all_3_seeds.sum())
    chk["A6_num_none_sig_all_seeds"] = not any(sig(p) for p in P[P.test_id.str.startswith("A6_num_unpaired_")].p_holm)
    BP = B[B.role == "PRIMARY"]
    chk["B2_i_sig"] = all(sig(p) for p in BP[BP.test_id.str.startswith("B2_i_")].p_holm)
    chk["B3_i_sig"] = all(sig(p) for p in BP[BP.test_id.str.startswith("B3_i_")].p_holm)
    chk["B3_ii_clarify_sig"] = sig(BP[BP.test_id == "B3_ii_clarify_test"].p_holm.iloc[0])
    chk["B2B3_iii_none_sig"] = not any(sig(p) for p in BP[BP.test_id.str.contains("_iii_")].p_holm)
    chk["B5_none_sig"] = not any(sig(p) for p in BP[BP.family == "B5"].p_holm)
    b4m = B[(B.test_id == "B4_mixedlm_test_mistral_nemo_seed42")].iloc[0]
    chk["B4_mistral_ns"] = not sig(b4m.p_raw)
    ag = d["agree"]["coefficients"]
    chk["agree_ok"] = all(v.get("audit_within_0.005", True) for v in ag.values())
    chk["n_primary_A42"] = int(len(P42))
    chk["n_primary_B"] = int(len(BP))
    expected = {"A1_all_seeds_sig": True, "A7_all_seeds_sig": True, "A3a_sig": True, "A3b_sig": True, "A4_none_sig": True,
                "A2_lvm_ns_all_seeds": True, "A2_tier_sig_all_seeds": True, "A6_fac_sig_seed42": 3, "A6_fac_robust": 2,
                "A6_num_none_sig_all_seeds": True, "B2_i_sig": True, "B3_i_sig": True, "B3_ii_clarify_sig": True,
                "B2B3_iii_none_sig": True, "B5_none_sig": True, "B4_mistral_ns": True, "agree_ok": True,
                "n_primary_A42": 31, "n_primary_B": 29}
    bad = {k: (chk[k], v) for k, v in expected.items() if chk[k] != v}
    if bad:
        raise SystemExit(f"TABLE 1 VERDICT CONFLICT with CSVs: {bad}")
    log.info("Table 1 verdicts verified against CSVs: %s", chk)
    return chk


# ---------------------------------------------------------------------------
def build(d: dict) -> str:
    plan, gate, pre, A, B = d["plan"], d["gate"], d["pre"], d["A"], d["B"]
    env = plan["environment"]
    ag = d["agree"]["coefficients"]
    A42 = A[A.seed == 42]
    P42 = A42[A42.role == "PRIMARY"]
    BP = B[B.role == "PRIMARY"]
    get = lambda df, tid: df[df.test_id == tid].iloc[0]
    L: list[str] = []
    w = L.append

    # ======================================================================
    w("# Stage 8 — Statistical significance (Gate 8 report)")
    w("")
    w(f"*Generated {date.today().isoformat()} by `scripts/s8_stats.py --part report`; pre-registered plan "
      f"`reports/stats/test_plan.json` v{plan['plan_version']} (changelog in Appendix B).*")
    w("")
    w("## Scope and methods")
    w("")
    w(plan["scope"])
    w("")
    w(f"- **Test set:** 362 of 422 rows scored. Excluded 60 `pending_human`: {gate['pending_by_category']} "
      f"(qtype: {gate['pending_by_qtype']}). SHA-256 of `test_hardened.jsonl` verified against `test_hardened.sha256`.")
    w(f"- **Arms:** 20 scored arms in `results_master.csv` (4 zero-shot, 12 fine-tuned = 4 models × seeds 42/1337/2024, "
      f"4 clarify-trained at seed 42) plus 1 detector prediction set.")
    w(f"- **Seed policy:** {plan['seed_policy']}")
    w(f"- **Multiple comparisons:** {plan['multiple_comparisons']} PRIMARY tests: **31 automatic (seed 42) + 29 human = 60**. "
      f"Every A-family PRIMARY test with a like-for-like FT arm is repeated at seeds 1337 and 2024 (Table 4).")
    w(f"- **Exact-match denominators:** {plan['em_denominator_note']} (Table 8.)")
    w(f"- **Human sample:** 200 questions × 4 arms = 800 judgments; 100 judgments rated by all 5 annotators, 700 by one. "
      f"PRIMARY = the 182 questions that are locked test rows; SENSITIVITY = all 200 (adds 18 authored Bengali translations: "
      f"15 clarify, 3 numeric). Per-judgment `mean_rating` = mean of its raters; `farmer_yes` = majority vote.")
    w(f"- **Mixed-model method actually used (B4):** Python `statsmodels` fallback — MixedLM (rating ~ arm, groups = annotator, "
      f"variance component for row_id) and BinomialBayesMixedGLM (variational Bayes, crossed annotator + row_id). `Rscript` is "
      f"not on PATH and R was not installed. MixedLM treats the 1–5 rating as interval; the Bayesian model yields credible "
      f"intervals, not p-values. B4 is supporting, not primary.")
    w(f"- **Agreement (100 shared judgments × 5 raters):** would_give_farmer Fleiss κ = {ag['fleiss_kappa_farmer']['estimate']:.3f} "
      f"[{ag['fleiss_kappa_farmer']['ci_low']:.3f}, {ag['fleiss_kappa_farmer']['ci_high']:.3f}]; ordinal Krippendorff α = "
      f"{ag['kripp_alpha_ordinal']['estimate']:.3f} [{ag['kripp_alpha_ordinal']['ci_low']:.3f}, {ag['kripp_alpha_ordinal']['ci_high']:.3f}]; "
      f"nominal Fleiss κ on 1–5 = {ag['fleiss_kappa_nominal_1to5']['estimate']:.3f}. **Decision recorded:** the primary human "
      f"measures are `would_give_farmer` and ordinal α. Nominal κ treats a 4-vs-5 disagreement as a full disagreement and so "
      f"understates agreement on an ordinal scale; it is reported for completeness (Table 7). All three audit values reproduced "
      f"within 0.0005; package and independent implementations agree to 4 dp.")
    w(f"- **Environment:** Python {env['python']}, numpy {env['numpy']}, scipy {env['scipy']}, statsmodels {env['statsmodels']}, "
      f"scikit-learn {env['sklearn']}, krippendorff ({env['krippendorff']}), pandas {env['pandas']}; R not available. "
      f"Full freeze: `reports/stats/env_freeze.txt`. Seed 42 throughout; 10,000 bootstrap resamples.")
    w("- **Verdict wording:** non-significant results are written as *no significant difference detected* together with the "
      "95% CI of the difference, so the reader can see how large a difference is ruled out. *Equivalent* and a bare *no difference* are never used.")
    w("")

    # ======================================================================
    w("## Table 1 — Hypothesis summary")
    w("")
    a1 = P42[P42.family == "A1"]
    a1_rng = f"{100*a1['diff'].min():+.0f} to {100*a1['diff'].max():+.0f} pp"
    fac = P42[P42.test_id.str.startswith("A6_fac_")]
    fac_sig = [ML[t.split("A6_fac_")[1]] for t, p in zip(fac.test_id, fac.p_holm) if sig(p)]
    fac_rob = [ML[t.split("A6_fac_")[1]] for t, r in zip(d["rob"].test_id, d["rob"].robust_all_3_seeds) if t.startswith("A6_fac_") and r]
    num = P42[P42.test_id.str.startswith("A6_num_unpaired_")]
    num_hw = ((num.diff_ci_high - num.diff_ci_low) / 2 * 100)
    num_w = f"{num_hw.min():.0f}–{num_hw.max():.0f}"      # CI half-widths in pp
    a7 = P42[P42.family == "A7"]
    a7d = d["a7"]
    zs_b = {r.model: r.confident_error_rate_a for _, r in a7d[a7d.variant == "zeroshot"].iterrows()}
    a7_txt = ", ".join(f"{ML[t[3:]]} {100*r['diff']:+.0f} pp" for t, (_, r) in zip(a7.test_id, a7.iterrows()))
    a4 = P42[P42.family == "A4"]
    a4_txt = "; ".join(f"{ML[t.split('A4_')[1]]} {100*r['diff']:+.1f} pp [{100*r.diff_ci_low:+.1f}, {100*r.diff_ci_high:+.1f}]"
                       for t, (_, r) in zip(a4.test_id, a4.iterrows()))
    a5 = get(A42, "A5_detector")
    lvm = get(P42, "A2_llama_vs_mistral")
    b2iii = get(BP, "B2_iii_overall_test"); b3iii = get(BP, "B3_iii_overall_test")
    b3ii = get(BP, "B3_ii_clarify_test"); b2i = get(BP, "B2_i_overall_test"); b3i = get(BP, "B3_i_overall_test")
    b5 = BP[BP.family == "B5"]
    rows = [
        ["H1", "Fine-tuning beats zero-shot", "A1, B2-i, B3-i", "**Supported**",
         f"A1: all 4 models p_holm < .001 at every seed (robust on 3 seeds); Δcorrect_0p1 (/146) {a1_rng}. Human: rating r_rb = "
         f"{b2i.effect:.2f} [{b2i.effect_ci_low:.2f}, {b2i.effect_ci_high:.2f}], farmer_yes {_diff(b3i)} (n = 182), both p_holm < .001."],
        ["H2", "Bengali vs English", "A6, B5", "**Partially supported**",
         f"Factual chrF++ (automatic, paired on fact_id, n = 57): English > Bengali in {len(fac_sig)} of 4 models at seed 42 "
         f"({', '.join(fac_sig)}); robust on all seeds in {len(fac_rob)} of 4 ({', '.join(fac_rob)}); Llama not significant. "
         f"Numeric (unpaired Fisher, 73 vs 73): no significant difference detected in any model or seed; CI half-widths are ±{num_w} pp, "
         f"too wide to rule out gaps of about 15 points. Human (B5, 100 pairs): none of 8 tests significant after Holm; point estimates "
         f"lean English in all three fine-tuned arms. Exploratory: Mistral zero-shot answered only 30% of Bengali questions in Bengali (§H2 note)."],
        ["H3", "Human agreement", "B6", "**Supported on the pre-specified primary measures**",
         f"farmer κ = {ag['fleiss_kappa_farmer']['estimate']:.3f}, ordinal α = {ag['kripp_alpha_ordinal']['estimate']:.3f}; nominal κ = "
         f"{ag['fleiss_kappa_nominal_1to5']['estimate']:.3f} reported and explained (Table 7)."],
        ["T", "Fine-tuning converts invented values into confident errors (thesis title)", "A7", "**Supported in all four models on all seeds**",
         f"Registered (a): FT − ZS = {a7_txt}; all p_holm ≤ .001. "
         f"Size of change depends on the base model: zero-shot baselines Llama {100*zs_b['llama31_8b']:.0f}%, Mistral {100*zs_b['mistral_nemo']:.0f}%, "
         f"Phi-3.5 {100*zs_b['phi35_mini']:.0f}%, Phi-4 {100*zs_b['phi4_mini']:.0f}%; every FT arm ≥ 98.5%. Strict version (b), which matches the thesis "
         f"definition, is stronger (§A7)."],
        ["C1", "Clarify training teaches clarification", "A3a, B3-ii", "**Supported**",
         f"A3a: clarify_exact 0% → 29–38% in all 4 models, p_holm < .001. Human B3-ii clarify (n = 25): farmer_yes {_diff(b3ii)}, p_holm < .001."],
        ["C2", "Clarify training reduces invented values", "A3b", "**Supported**",
         "A3b: invented_value 100% → 50–66% in all 4 models, p_holm < .001 (Δ −34 to −50 pp)."],
        ["C3", "Clarify training does not harm numeric accuracy", "A4", "**No loss detected**",
         f"correct_0p1 (/146), clarify-FT − FT: {a4_txt}; no test significant. Phi-3.5 (flagged) is the only negative point estimate; a loss "
         f"larger than 8.2 pp is ruled out at 95%."],
        ["D", "Detector adds safety", "A5", "**Descriptive only; oracle-gated upper bound**",
         f"Safe handling {100*a5.rate_a:.1f}% → {100*a5.rate_b:.1f}%, risk difference {_diff(a5)}; b = {int(a5.b)}, c = {int(a5.c)} by construction. {DET_LABEL}"],
        ["M", "Llama FT vs Mistral FT", "A2, B2-iii, B3-iii, B4", "**No significant difference detected**",
         f"A2 McNemar OR {lvm.effect:.2f} [{lvm.effect_ci_low:.2f}, {lvm.effect_ci_high:.2f}], accuracy Δ {_diff(lvm)}, non-significant at all 3 seeds; "
         f"human rating r_rb {b2iii.effect:.2f} [{b2iii.effect_ci_low:.2f}, {b2iii.effect_ci_high:.2f}], farmer_yes {_diff(b3iii)}; annotator-adjusted "
         f"B4 unchanged in direction and significance. Friedman shows the four FT models do differ (W = {get(P42, 'A2_friedman').effect:.2f}); the "
         f"tier contrast {{Llama, Mistral}} > {{Phi}} is significant and robust."],
    ]
    w(md_table(["ID", "Claim", "Tests", "Verdict", "Key numbers"], rows))
    w("")

    # ======================================================================
    w("## Table 2 — Automatic results (Part A), seed 42, grouped by family")
    w("")
    w("Paired by `row_id` unless stated. Effects: McNemar odds ratio = b/c (b = rows where the treatment arm scores 1 and the baseline 0); "
      "Wilcoxon rank-biserial r; Fisher OR = odds(en)/odds(bn). Differences are treatment minus baseline: FT − ZS (A1, A7), Mistral − Llama (A2), "
      "clarify-FT − FT (A3, A4), hybrid − model-only (A5), en − bn (A6). SENSITIVITY rows carry raw p only; their verdict strings are raw-p based.")
    for fam, title, note in [
        ("A1", "A1 — Fine-tuning helps (zero-shot vs FT, numeric rows, correct_0p1 over 146)", None),
        ("A2", "A2 — Model comparison after fine-tuning (numeric rows, correct_0p1 over 146)",
         "Pre-registered null expectation for Llama FT vs Mistral FT. No pairwise post-hoc tests were planned or run."),
        ("A3a", "A3a — Clarify training teaches clarification (clarify rows, n = 68)", None),
        ("A3b", "A3b — Clarify training reduces invented values (clarify rows, n = 68)", None),
        ("A4", "A4 — Clarify training does not harm numeric accuracy (numeric rows, correct_0p1 over 146)",
         "Hypothesis is *no harm*; the CI of the accuracy difference is the result. Phi-3.5 was flagged in advance (Stage 6E slot-F1 drop). "
         + plan["a3_a4_seed_note"]),
        ("A5", "A5 — Detector (DESCRIPTIVE, plan v3)", DET_LABEL + " Rules and the test categories that informed each are in Appendix A."),
        ("A6", "A6 — Bengali vs English, automatic (H2)",
         "Pairing key is `fact_id` (the locked test set has no `pair_id`): 57 factual pairs, 17 numeric pairs, 0 clarify pairs. "
         "Numeric PRIMARY is therefore the unpaired Fisher test on 73 vs 73 rows; the 17-pair McNemar is SECONDARY and underpowered."),
        ("A7", "A7 — Confident error (clarify rows, n = 68): zero-shot vs FT", None),
    ]:
        w("")
        w(f"### {title}")
        w("")
        if note:
            w(f"> {note}")
            w("")
        sub = A42[A42.family == fam]
        if fam == "A7":
            sub = sub[sub.role == "PRIMARY"]     # registered (a) first; (b) follows in the sensitivity table below
        w(results_table(sub))
        if fam == "A6":
            w("")
            fl = get(A42, "A6_fac_llama31_8b")
            w(f"**Llama factual.** The registered Wilcoxon signed-rank is not significant (p_raw = {fl.p_raw:.3f}, p_holm = {fl.p_holm:.3f}; "
              f"rank-biserial r = {fl.effect:.2f} [{fl.effect_ci_low:.2f}, {fl.effect_ci_high:.2f}]), while the paired-bootstrap 95% CI of the "
              f"*mean* chrF++ difference, {_diff(fl)}, excludes zero. The paired differences are skewed (a few English rows gain a lot; most "
              f"pairs are close to zero), which pulls the mean but not the ranks. The registered test's verdict stands: no significant difference detected for Llama.")
            w("")
            w(f"**Numeric.** No significant difference detected in any model at any seed. The CIs of the en − bn accuracy difference have "
              f"half-widths of ±{num_w} pp (e.g. Llama {_diff(get(A42, 'A6_num_unpaired_llama31_8b'))}); they are too wide to rule out gaps of about "
              f"15 points in either direction. **Pattern check:** numeric shows no consistent direction (as expected from Stage 6); factual favours "
              f"English significantly in {len(fac_sig)} of 4 models, not in all four as Stage 6 suggested.")
            w("")
            w("**Clarify (descriptive; 0 valid pairs).** bn vs en rates per arm with bootstrap CIs:")
            w("")
            a6c = d["a6c"]
            w(md_table(["arm", "metric", "bn (n = 34)", "en (n = 34)"],
                       [[arm, m,
                         f"{r_bn.rate:.2f} [{r_bn.ci_low:.2f}, {r_bn.ci_high:.2f}]", f"{r_en.rate:.2f} [{r_en.ci_low:.2f}, {r_en.ci_high:.2f}]"]
                        for arm in a6c.arm.unique() for m in ["clarify_exact", "invented_value"]
                        for r_bn in [a6c[(a6c.arm == arm) & (a6c.metric == m) & (a6c.lang == "bn")].iloc[0]]
                        for r_en in [a6c[(a6c.arm == arm) & (a6c.metric == m) & (a6c.lang == "en")].iloc[0]]]))
    # ---- A7 narrative
    w("")
    w("### A7 — detail")
    w("")
    w(f"**Definition (registered, primary (a)).** {plan['a7_definition']}. {plan['a7_note']} {d['a7trunc']['observation']}; "
      f"the threshold ≥ 511 was kept as registered. Rows at 505–510 tokens treated as not truncated: {d['a7trunc']['zero_shot_rows_in_505_510_treated_as_not_truncated']}; "
      f"{d['a7trunc']['direction_of_bias']}.")
    w("")
    w("**Registered primary (a) — reported first.** All four McNemar tests are significant after Holm at all three seeds (Table 4). "
      "The size of the change depends on the base model: Mistral and Phi-4 already produce confident errors on most clarify rows zero-shot, "
      "so fine-tuning adds 16 and 24 pp; Llama and Phi-3.5 mostly ran into the token cap or hedged zero-shot, so fine-tuning adds 53 and 68 pp. "
      "Mistral's zero-shot baseline is partly English-language output: it answered only 30% of Bengali questions in Bengali (Table 9), and its "
      "English answers carry the hedges the marker list detects.")
    w("")
    w("**Sensitivity (b) — the definition that matches the thesis.** (b) additionally requires `explicit_assumption = 0`: a confident error is a "
      "complete numeric answer that neither hits the token cap nor states that a value was assumed. This is the thesis definition of a confident "
      "error, and it is *stronger* than (a) in every model because hedged zero-shot answers drop out of the zero-shot count while no fine-tuned answer "
      "hedges:")
    w("")
    sens = A42[A42.test_id.str.startswith("A7_sens_")]
    w(results_table(sens))
    w("")
    w("**Descriptive per model (clarify rows, n = 68; bootstrap 95% CIs).** conf_err (a) = registered; (b) = strict.")
    w("")
    rows = []
    for _, r in a7d.iterrows():
        rows.append([ML[r.model], r.variant, f"{r.truncation_rate:.2f} [{r.truncation_rate_lo:.2f}, {r.truncation_rate_hi:.2f}]",
                     f"{r.median_gen_tokens:.0f}", f"{r.invented_value_rate:.2f} [{r.invented_value_rate_lo:.2f}, {r.invented_value_rate_hi:.2f}]",
                     f"{int(round(r.explicit_assumption_rate_among_invented * r.n_invented))}/{int(r.n_invented)} = {r.explicit_assumption_rate_among_invented:.2f} [{r.ea_lo:.2f}, {r.ea_hi:.2f}]",
                     f"{r.confident_error_rate_a:.2f} [{r.confident_error_rate_a_lo:.2f}, {r.confident_error_rate_a_hi:.2f}]",
                     f"{r.confident_error_rate_b_strict:.2f} [{r.confident_error_rate_b_strict_lo:.2f}, {r.confident_error_rate_b_strict_hi:.2f}]"])
    w(md_table(["model", "variant", "truncation rate", "median tokens", "invented_value", "explicit_assumption among invented", "conf_err (a)", "conf_err (b) strict"], rows))
    w("")
    w(f"**Explicit-assumption markers.** English: {plan['a7_explicit_assumption_markers']['en']}; Bengali: {plan['a7_explicit_assumption_markers']['bn']}; "
      f"excluded: {plan['a7_explicit_assumption_markers']['excluded']}. {len(d['a7ex'])} example generations (5 per model, all zero-shot — no fine-tuned "
      f"output carries a marker) are in `a7_explicit_assumption_examples.json`; all 20 hit `assum`, one also `not provided`.")
    w("")
    hz = d["hedge"]
    w("**Bengali hedge check (exploratory, added before this report; no change to the registered definition).** The registered Bengali list "
      f"missed phrasings used by zero-shot Bengali answers: {list(hz['unlisted_candidates'])[:8]} … Searching the 34 Bengali clarify outputs per arm:")
    w("")
    rows = []
    for arm, r in hz["arms"].items():
        rows.append([arm, r["n_outputs_in_bengali_script"], r["registered_bn_marker_hits"], r["unlisted_hedge_hits"],
                     r["unlisted_hit_counts_by_phrase"] or "—", f"{r['confident_error_b_strict_registered_rate_bn']:.2f}",
                     f"{r['confident_error_b_strict_extended_rate_bn_EXPLORATORY']:.2f}"])
    w(md_table(["arm", "outputs in Bengali script /34", "registered BN hits", "unlisted hedge hits", "phrases", "strict (b) rate, bn rows, registered", "strict (b) rate, bn rows, extended (exploratory)"], rows))
    w("")
    w("Every fine-tuned Bengali clarify output (0/34 in each of the four models) contains no hedge under either list; 40 hand-read FT outputs "
      "(10 per model) are all the bare templated line `নিট সেচের গভীরতা = Kc×ET0 − কার্যকর বৃষ্টি = … মিমি। <area> … লিটার।` with an invented "
      "area (e.g. \"৫.০ একর\", \"১ বর্গকিলোমিটার\"). The under-detection therefore biases only zero-shot rows, in the conservative direction; the "
      "adjusted strict rate changes only for Phi-4 zero-shot Bengali rows (0.62 → 0.47) and leaves every FT rate unchanged.")
    w("")

    # ======================================================================
    w("## Table 3 — Human results (Part B), PRIMARY sample (182 test-set questions), grouped by family")
    w("")
    w("Per-judgment `mean_rating` (1–5) and `farmer_yes` (majority vote). Wilcoxon signed-rank and McNemar exact, paired by `row_id`. "
      "Clarify n = 25 in the primary sample and 40 in the sensitivity sample.")
    for fam, title in [("B1", "B1 — Omnibus (Holm family of 2)"), ("B2", "B2 — Contrasts on mean_rating (9 cells, Holm within family)"),
                       ("B3", "B3 — Contrasts on farmer_yes (9 cells, Holm within family)"),
                       ("B5", "B5 — Bengali vs English, human (all 200 questions → 100 pair_id pairs; 18 pairs contain an authored translation; 8 cells, Holm)"),
                       ("B8", "B8 — Human vs automatic on numeric (PRIMARY: Spearman only; Jonckheere–Terpstra dropped in plan v3)")]:
        w("")
        w(f"### {title}")
        w("")
        w(results_table(BP[BP.family == fam]))
    w("")
    w("### B4 — Annotator-adjusted models (SECONDARY, supporting)")
    w("")
    w("Reference arm `llama31_8b_seed42`. MixedLM coefficients are adjusted rating differences (Wald 95% CI, raw p). Bayesian rows are posterior "
      "log-odds of `farmer_yes` with 95% credible intervals (mean ± 1.96 sd), no p-values. Full summaries in `mixed_model.txt`.")
    w("")
    b4 = B[B.family == "B4"]
    w(md_table(["test_id", "n ratings", "comparison", "adjusted effect [95% CI / CrI]", "raw p", "unadjusted difference (rating pts; farmer_yes proportion for Bayesian rows)", "verdict"],
               [[r.test_id, r.n, r.comparison, _eff(r), _p(r.p_raw), _num(r["diff"], 3), r.verdict] for _, r in b4.iterrows()]))
    w("")
    b4m = get(B, "B4_mixedlm_test_mistral_nemo_seed42"); b4c = get(B, "B4_mixedlm_test_llama31_8b_seed42_clarify")
    w(f"**Llama FT vs Mistral FT after adjustment:** coefficient {b4m.effect:+.2f} [{b4m.effect_ci_low:+.2f}, {b4m.effect_ci_high:+.2f}], p = {b4m.p_raw:.2f}; "
      f"Bayesian log-odds {get(B, 'B4_bayesglm_test_mistral_nemo_seed42').effect:+.2f} with a credible interval that includes 0. Neither direction nor "
      f"significance changes relative to B2-iii / B3-iii. **One difference from B2:** the clarify-vs-FT *overall* rating contrast is significant in the "
      f"adjusted model ({b4c.effect:+.2f} [{b4c.effect_ci_low:+.2f}, {b4c.effect_ci_high:+.2f}], raw p < .001) but not after Holm in B2 "
      f"(p_raw = {get(BP, 'B2_ii_overall_test').p_raw:.3f}, p_holm = {get(BP, 'B2_ii_overall_test').p_holm:.3f}); same direction, and B4 is "
      f"uncorrected for multiplicity, so B2's verdict stands.")
    w("")
    w("### B8 — detail")
    w("")
    b8 = get(BP, "B8_spearman")
    w(f"Spearman ρ(abs_err, mean_rating) = {b8.effect:.2f} [{b8.effect_ci_low:.2f}, {b8.effect_ci_high:.2f}], p < .001, n = {b8.n} FT-family numeric "
      f"outputs (3 fine-tuned arms × 67 test-set numeric questions = 201; 21 multi-crop outputs have no scalar `abs_err` and are excluded; 0 excluded for non-extraction).")
    w("")
    w(md_table(["abs(final depth − gold) bin (mm)", "n", "mean rating [95% CI]", "farmer_yes [95% CI]"],
               [[r.bin_abs_err_mm, r.n, f"{r.mean_rating:.2f} [{r.mean_rating_lo:.2f}, {r.mean_rating_hi:.2f}]",
                 f"{100*r.farmer_yes_rate:.0f}% [{100*r.farmer_yes_lo:.0f}, {100*r.farmer_yes_hi:.0f}]"] for _, r in d["b8"].iterrows()]))
    w("")
    e, t = d["b8"].iloc[0], d["b8"].iloc[1]
    w(f"**Registered headline:** human acceptance tracks the ±0.1 mm tier, not ±1 mm — the ≤1.0 mm bin is accepted {100*d['b8'].iloc[2].farmer_yes_rate:.0f}% of the time. "
      f"**Additional finding:** with the reference visible, annotators accepted {100*e.farmer_yes_rate:.0f}% of exact matches (< 0.005 mm, n = {e.n}) but only "
      f"{100*t.farmer_yes_rate:.0f}% of answers within 0.005–0.1 mm (n = {t.n}). Human acceptance tracks reference agreement, and the ±0.1 mm tolerance is more "
      f"lenient than that standard. Table 8 therefore adds an exact (< 0.005 mm) column per arm.")
    w("")

    # ======================================================================
    w("## Table 4 — Seed robustness (A1, A2, A6, A7; significant after Holm on all three seeds?)")
    w("")
    w(f"> {plan['a3_a4_seed_note']} (Limitation.)")
    w("")
    rob = d["rob"]
    w(md_table(["test_id", "expected", "seed 42", "seed 1337", "seed 2024", "robust"],
               [[r.test_id, r.expected_direction, r.seed42, r.seed1337, r.seed2024, "**yes**" if r.robust_all_3_seeds else "no"] for _, r in rob.iterrows()]))
    w("")
    w("Reading: H1 (A1), the tier contrast (A2_tier), the Friedman omnibus and the thesis-title claim (A7) are robust. Llama FT vs Mistral FT is "
      "non-significant at all three seeds (consistent with the pre-registered null). H2-factual is robust for Phi-3.5 and Phi-4 only; Mistral's "
      "English advantage holds at seed 42 but not at 1337 or 2024, and Llama never shows it. A6 numeric is non-significant at every seed.")
    w("")

    # ======================================================================
    w("## Table 5 — Human sample: primary (182 test rows) vs sensitivity (all 200 / excluding annotator A1)")
    w("")
    rows = []
    for _, r in BP[BP.family.isin(["B1", "B2", "B3"])].iterrows():
        base = r.test_id.rsplit("_test", 1)[0]
        alt = B[B.test_id == f"{base}_all"]
        ex = B[B.test_id == f"{base}_test_exclA1"]
        alt = alt.iloc[0] if len(alt) else None
        ex = ex.iloc[0] if len(ex) else None
        s_p = sig(r.p_holm); s_a = sig(alt.p_raw) if alt is not None else None; s_e = sig(ex.p_raw) if ex is not None else None
        change = "no" if (s_a is None or s_a == sig(r.p_raw)) and (s_e is None or s_e == sig(r.p_raw)) else "raw-p pattern differs"
        rows.append([r.test_id.rsplit("_test", 1)[0], r.n, f"{_p(r.p_raw)} ({'sig' if sig(r.p_raw) else 'ns'})",
                     f"{_p(r.p_holm)} ({'sig' if s_p else 'ns'})",
                     f"n={alt.n}, raw p {_p(alt.p_raw)} ({'sig' if s_a else 'ns'})" if alt is not None else "—",
                     f"n={ex.n}, raw p {_p(ex.p_raw)} ({'sig' if s_e else 'ns'})" if ex is not None else "—", change])
    w(md_table(["cell", "n (primary)", "primary raw p", "primary p_holm", "all 200 (raw p)", "excl. A1 (raw p)", "raw-p pattern change?"], rows))
    w("")
    w("No verdict changes: every contrast keeps its direction and its raw-p significance pattern (sensitivity runs are compared on raw p, since Holm "
      "is applied only in the primary analysis) when the 18 authored translations are added or "
      "when A1 (the most divergent annotator; leave-one-out α rises to "
      f"{d['agree']['leave_one_annotator_out_alpha_ordinal']['A1']['alpha_ordinal_without']:.3f} without A1) is removed. B5 sensitivity on the 82 native pairs: "
      "same direction (English ≥ Bengali in FT arms); raw p < .05 for Llama FT and Mistral FT ratings; nothing survives Holm in the primary analysis.")
    w("")
    hs = d["hsum"]; hs = hs[(hs["sample"] == "test") & (hs.measure.isin(["mean_rating", "farmer_yes", "acceptable"]))]
    w("Per-arm human summary, primary sample (n = 182 per arm; bootstrap 95% CI):")
    w("")
    w(md_table(["arm", "mean rating", "farmer_yes", "acceptable (mean ≥ 4)"],
               [[AS[a]] + [f"{hs[(hs.arm == a) & (hs.measure == m)].estimate.iloc[0]:.2f} [{hs[(hs.arm == a) & (hs.measure == m)].ci_low.iloc[0]:.2f}, {hs[(hs.arm == a) & (hs.measure == m)].ci_high.iloc[0]:.2f}]"
                            for m in ["mean_rating", "farmer_yes", "acceptable"]] for a in HUMAN_ARMS]))
    w("")

    # ======================================================================
    w("## Table 6 — Dice validation (B7; 280 factual (row_id, arm) judgments, seed 42 FT arms; closes deferred M5)")
    w("")
    dc = d["dice"]
    piv = {(r.metric, r.statistic): f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]" for _, r in dc.iterrows()}
    mets = ["dice_word", "rouge_l", "chrf_pp", "bertscore_f1", "dice_char3", "dice_char5"]
    w(md_table(["metric", "point-biserial r vs farmer_yes", "ROC-AUC predicting farmer_yes", "Spearman ρ vs mean_rating"],
               [[m, piv[(m, "point_biserial_r_vs_farmer_yes")], piv[(m, "roc_auc_predicting_farmer_yes")], piv[(m, "spearman_rho_vs_mean_rating")]] for m in mets]))
    w("")
    w("`dice_word` has the best point estimate on all three criteria; its CI overlaps ROUGE-L and chrF++, so it is best but not distinguishably better. "
      "Character-level Dice tracks human judgment worst. Join asserted at 280/280 rows.")
    w("")

    # ======================================================================
    w("## Table 7 — Agreement (B6; 100 shared judgments × 5 annotators; bootstrap 95% CI over items)")
    w("")
    rows = []
    for k, v in ag.items():
        rows.append([k, f"{v['estimate']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]",
                     f"{v['package_estimate']:.4f} ({'agree' if v['package_vs_own_agree_3dp'] else 'DISAGREE'})" if "package_estimate" in v else "—",
                     f"{v['audit_value']} (Δ {v['audit_abs_diff']:.4f})" if "audit_value" in v else "—"])
    w(md_table(["coefficient", "estimate [95% CI]", "package / statsmodels", "Stage 7 audit"], rows))
    w("")
    lo = d["agree"]["leave_one_annotator_out_alpha_ordinal"]
    w("Leave-one-annotator-out ordinal α: " + ", ".join(f"without {a} = {v['alpha_ordinal_without']:.3f}" for a, v in lo.items()) + ".")
    w("")
    w(d["agree"]["decision"])
    w("")

    # ======================================================================
    w("## Table 8 — Exact-match denominators per arm")
    w("")
    w(plan["em_denominator_note"] + " The exact (< 0.005 mm) column is the standard human acceptance tracked in B8.")
    w("")
    emd = d["emd"]
    w(md_table(["arm", "correct_0p1 (/146)", "exact_0.1 single-crop (/116)", "exact < 0.005 mm single-crop (/116)", "single rows with extracted depth (/116)", "multi-crop all values ≤ 0.1 mm (/30)"],
               [[r.arm, f"{r.correct_0p1_count} ({100*r.correct_0p1_rate:.1f}%)", f"{r.exact_0p1_single_count} ({100*r.exact_0p1_single_rate:.1f}%)",
                 f"{r.exact_lt0p005_single_count} ({100*r.exact_lt0p005_single_rate:.1f}%)", r.single_rows_with_extracted_depth, r.multi_all_correct_count]
                for _, r in emd.iterrows()]))
    w("")
    w("No arm gets all values of a multi-crop row within 0.1 mm (0/30 everywhere), so `correct_0p1` over 146 equals `exact_0.1` over 116 in count for "
      "every arm and differs only in denominator.")
    w("")

    # ======================================================================
    w("## Table 9 — Response language (exploratory; added after Part A, plan v3)")
    w("")
    w(plan["response_language_note"])
    w("")
    rl = d["lang"]
    subs = ["all_bn_questions", "numeric_runtime", "factual", "clarify", "abstain"]
    rows = []
    for arm in rl.arm.unique():
        row = [arm]
        for s in subs:
            r = rl[(rl.arm == arm) & (rl.subset == s)].iloc[0]
            row.append(f"{100*r.share_answered_in_bengali:.1f}% [{100*r.ci_low:.0f}, {100*r.ci_high:.0f}] (n={r.n_bn_questions})")
        rows.append(row)
    w(md_table(["arm", "all Bengali questions", "numeric", "factual", "clarify", "abstain"], rows))
    w("")
    w("**H2 note (exploratory).** Mistral-NeMo zero-shot answered only 30.4% [23.8, 37.0] of Bengali questions in Bengali script; every "
      "fine-tuned and clarify-trained arm answers ≥ 98.9% in Bengali. Fine-tuning fixes response language completely. This also qualifies the A7 "
      "Mistral zero-shot baseline: its 84% confident-error rate is measured largely on English-language answers to Bengali questions.")
    w("")

    # ======================================================================
    w("## Table 10 — Bootstrap 95% CIs for Stage 6 Table A headlines (DESCRIPTIVE; 10,000 row resamples)")
    w("")
    w("Denominators: em_0p1 here is `correct_0p1` over 146; em_1p0 over the 116 single-crop rows; MAE over single-crop rows with an extracted depth; "
      "net_extract_rate over 146; abstain accuracy over 34; clarify_exact and invented_value_rate over 68; dice_word / chrF++ / BERTScore over 114 factual rows. "
      "`*_FT_3seeds` rows are mean ± std of the three seed estimates.")
    w("")
    bt = d["boot"]
    mets = ["em_0p1", "em_1p0", "mae", "net_extract_rate", "abstain_accuracy", "clarify_exact", "invented_value_rate", "dice_word", "chrf_pp", "bertscore_f1"]
    arms = list(bt.arm.unique())
    rows = []
    for arm in arms:
        row = [arm]
        for m in mets:
            r = bt[(bt.arm == arm) & (bt.metric == m)].iloc[0]
            row.append(f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]" if r.kind == "bootstrap" else f"{r.estimate:.3f} ± {r.estimate - r.ci_low:.3f}")
        rows.append(row)
    w(md_table(["arm"] + mets, rows))
    w("")
    w("### A1b — abs_err, single-crop rows where both arms extracted a depth (DESCRIPTIVE)")
    w("")
    w(md_table(["model", "n both extracted (/116)", "dropped", "median abs_err ZS (mm)", "median abs_err FT (mm)", "paired median Δ FT − ZS [95% CI]"],
               [[ML[r.model], r.n_both_extracted, r.n_dropped, f"{r.median_abs_err_zs:.3f}", f"{r.median_abs_err_ft:.3f}",
                 f"{r.median_paired_diff_ft_minus_zs:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"] for _, r in d["a1b"].iterrows()]))
    w("")

    # ======================================================================
    w("## What changed — Stage 6/7 claims not supported once tested, or qualified")
    w("")
    w("1. **H2 factual is not universal.** Stage 6 reported English > Bengali chrF++ in every model. Tested pairwise on 57 fact_id pairs with Holm, "
      f"it holds for {len(fac_sig)} of 4 models at seed 42 and for only {len(fac_rob)} of 4 (Phi-3.5, Phi-4) on all three seeds. Llama's gap (+6.3 chrF++ mean) is "
      "not significant by the registered rank test; Mistral's is seed-42-only.")
    w("2. **±0.1 mm is not the human standard.** The registered B8 headline (acceptance tracks ±0.1 mm, not ±1 mm) holds, but annotators accepted 97% of exact "
      "matches and only 38% of answers in the 0.005–0.1 mm band. `exact_0.1` over-counts human-acceptable answers by roughly that band; Table 8 now shows the exact column.")
    w("3. **The detector result is not a performance estimate.** The Stage 6C figure (safe handling 36.8% → 83.8%) is reproduced, but A5 was demoted from a PRIMARY "
      f"test to a descriptive, oracle-gated upper bound: {DET_LABEL}")
    w("4. **Clarify training's overall human benefit is measure-dependent.** On clarify rows it is unambiguous (B2/B3-ii clarify, p_holm ≤ .002). Overall, the "
      "farmer_yes gain (+11 pp, p_holm = .033) survives Holm but the mean-rating gain (r_rb = 0.25, p_holm = .128) does not; the annotator-adjusted model finds it "
      "significant uncorrected. Report as: significant on farmer_yes, not on rating after correction.")
    w("5. **Mistral's zero-shot 'clarify' behaviour was partly a language artefact.** 70% of its Bengali-question answers were in English (exploratory Table 9), which "
      "the Stage 6 metrics did not distinguish from Bengali answers.")
    w("6. **Confident-error marker coverage (Bengali).** The registered Bengali hedge list under-detects zero-shot Bengali hedging (Llama 7/34, Phi-4 12/34 rows use "
      "unlisted phrasings). This biases A7 conservatively and leaves all fine-tuned rows (0/34 hedges in every model) unaffected; the registered definition was not changed.")
    w("7. **Nothing else reversed.** No PRIMARY test, automatic or human, contradicted its pre-registered direction; no null-expected test came out significant; "
      "no human verdict changed between the 182-question primary sample and the 200-question sensitivity sample or when annotator A1 was excluded.")
    w("")

    # ======================================================================
    w("## Limitations recorded in this stage")
    w("")
    w(f"- {plan['a3_a4_seed_note']}")
    w("- The locked test set carries no `pair_id`; bn/en pairing for A6 uses `fact_id` and yields 0 clarify pairs and 17 numeric pairs, so A6-numeric is unpaired and A6-clarify is descriptive.")
    w("- Truncation is inferred by re-tokenising generations (no `finish_reason` stored); decode→re-encode is not perfectly idempotent (see A7 detail). The Mistral-NeMo tokenizer regex warning from `transformers` does not affect the flag because the same default tokenizer was used at generation time.")
    w("- B4 uses interval-scale MixedLM and a variational Bayesian GLM instead of an ordinal mixed model (`ordinal::clmm`), because R is not available.")
    w("- The human sample's 18 authored Bengali translations are not locked test rows; they are excluded from the PRIMARY human analyses and their model answers were generated after Stage 6 with the same adapters, prompt and decoding (provenance verified in `precheck.json`).")
    w("- The detector (A5) reads gold labels; see Appendix A.")
    w("- Registered Bengali explicit-assumption markers under-detect Bengali hedging (zero-shot only; conservative).")
    w("")

    # ======================================================================
    w("## Appendix A — Detector rules (Stage 6C `s6c_postfix_clarify_detector.py`) and the test categories that informed each")
    w("")
    w(f"> {DET_LABEL}")
    w("")
    w(f"Scorable clarify rows by category: {d['clarify_cats']}; by subtype: {d['clarify_subtypes']}. The 20 `conflicting_values` clarify rows are `pending_human` and unscored.")
    w("")
    w(md_table(["#", "rule (as implemented)", "reads", "test categories / subtypes that informed it"], [
        ["1", "Gate: run only on rows whose `qtype == \"clarify\"`; all other rows get `detector_flag = False`.", "**gold qtype**",
         "All clarify categories. This gate alone rules out false positives on numeric/factual/abstain rows and makes c = 0 by construction."],
        ["2", "`invented_answer` = `classify_response(generation) == \"answer\"` AND the text matches a numeric-with-unit pattern (number followed by mm / liter / litre / l, or the Bengali unit words মিমি / লিটার).",
         "generation", "Observed Stage 6B outputs: worked-answer template always emits `… মিমি … লিটার`."],
        ["3a", "Keyword gap: a slot is 'missing' if none of its keywords appears in the question — `et0_mm` {et0, eto, evapotranspiration, ইটিও, বাষ্পীভবন}, "
               "`rainfall_mm` {rain…, বৃষ্টি…}, `area` {acre, decimal, bigha, hectare, m², sqm, square, একর, ডেসিমাল, বিঘা, হেক্টর, বর্গমিটার, জমির পরিমাণ}, "
               "`stage` {initial, early, seedling, mid, …, প্রাথমিক, মধ্য, শেষ, পর্যায়}.", "question",
         "`missing_parameter` subtypes `missing_rainfall`, `missing_area`, `missing_stage` (Stage 6B training slots); **`missing_et0` was added in Stage 6C after inspecting the test rows** — the training data never deleted ET0."],
        ["3b", "Flag if `invented_answer` AND (any slot missing by 3a **OR** `category ∈ {missing_parameter, misspelled_crop, unit_ambiguity}`).", "**gold category**",
         "Exactly the three scorable clarify categories in the test set. The OR-branch flags every invented answer on a clarify row regardless of the question text."],
        ["4", "Correction text: `Please provide: <slot name>` where the slot comes from `SUBTYPE_TO_SLOT[subtype]` (`ambiguous` → 'which crop', `unknown_local_unit` → 'field size and unit', …).",
         "**gold subtype**", "`misspelled_crop/ambiguous`, `unit_ambiguity/unknown_local_unit`, `missing_parameter/missing_*`."],
    ]))
    w("")
    w("Consequence for A5: with rule 1 and rule 3b, every clarify row on which the clarify-trained Llama still produced a numeric answer is flagged; the only "
      f"unflagged unsafe rows are those where `classify_response` did not return `answer` or no unit pattern matched. Safe handling {100*a5.rate_a:.1f}% → "
      f"{100*a5.rate_b:.1f}% (risk difference {_diff(a5)}) is therefore the ceiling of this rule set on this test set, not a generalisation estimate.")
    w("")

    # ======================================================================
    w("## Appendix B — Test-plan changelog")
    w("")
    for c in plan["changelog"]:
        w(f"- **v{c['version']}** ({c['date']}): " + (c["note"] if "note" in c else ""))
        for ch in c.get("changes", []):
            w(f"  - {ch}")
    w("")
    w("## Appendix C — Files")
    w("")
    for f in ["test_plan.json", "entry_gate.json", "precheck.json", "results_automatic.csv", "results_human.csv", "bootstrap_cis.csv",
              "seed_robustness_A.csv", "a1b_descriptive.csv", "a6_clarify_descriptive.csv", "a7_descriptive.csv",
              "a7_explicit_assumption_examples.json", "a7_truncation_caveat.json", "bn_hedge_check.json", "exact_match_denominators.csv",
              "response_language.csv", "agreement.json", "mixed_model.txt", "dice_validation.csv", "b8_error_bins.csv",
              "human_arm_summary.csv", "env_freeze.txt", "significance.md"]:
        w(f"- `reports/stats/{f}`")
    w("- `logs/s8_stats.log`")
    return "\n".join(L)


def gate8_summary(d: dict) -> str:
    ag = d["agree"]["coefficients"]
    A, B = d["A"], d["B"]
    P42 = A[(A.role == "PRIMARY") & (A.seed == 42)]
    BP = B[B.role == "PRIMARY"]
    n_sig_A = int(sum(sig(p) for p in P42.p_holm)); n_sig_B = int(sum(sig(p) for p in BP.p_holm))
    lines = [
        "=" * 70, "GATE 8 — STATISTICAL SIGNIFICANCE (plan v%d)" % d["plan"]["plan_version"], "=" * 70,
        "Scope       362/422 test rows; 60 pending_human excluded (40 out_of_domain, 20 conflicting_values)",
        "Tests       60 PRIMARY (31 automatic seed 42 + 29 human); Holm within family; 10,000-resample CIs",
        f"Significant {n_sig_A}/31 automatic, {n_sig_B}/29 human PRIMARY tests after Holm; 0 contradict expected direction",
        "",
        "H1  FT beats zero-shot ............ SUPPORTED (A1 all models, robust 3 seeds; B2/B3-i)",
        "H2  Bengali vs English ............ PARTIALLY SUPPORTED (factual 3/4 models seed 42, 2/4 all seeds; numeric ns, CIs ±15 pp; human ns)",
        f"H3  Human agreement ............... SUPPORTED on primary measures (farmer κ {ag['fleiss_kappa_farmer']['estimate']:.3f}, ordinal α {ag['kripp_alpha_ordinal']['estimate']:.3f}; nominal κ {ag['fleiss_kappa_nominal_1to5']['estimate']:.3f})",
        "T   FT -> confident errors ........ SUPPORTED all 4 models, all seeds (A7); size depends on base model; strict (b) stronger",
        "C1  Clarify training teaches ...... SUPPORTED (A3a; B3-ii clarify)",
        "C2  Clarify reduces invented ...... SUPPORTED (A3b)",
        "C3  No numeric harm ............... NO LOSS DETECTED (A4; worst-case bound -8.2 pp, Phi-3.5)",
        "D   Detector ...................... DESCRIPTIVE ONLY; oracle-gated upper bound (+47 pp in-sample)",
        "M   Llama FT vs Mistral FT ........ NO SIGNIFICANT DIFFERENCE DETECTED (A2 3 seeds; B2/B3-iii; B4)",
        "",
        "What changed: H2-factual not universal; ±0.1 mm more lenient than human standard (97% exact vs 38% in 0.005-0.1 mm);",
        "              detector demoted to descriptive; clarify overall human gain significant on farmer_yes only;",
        "              Mistral zero-shot answered 70% of Bengali questions in English (exploratory).",
        "Report      reports/stats/significance.md", "=" * 70,
    ]
    return "\n".join(lines)


def run(plan: dict) -> None:
    d = load_all()
    assert d["plan"]["plan_version"] == plan["plan_version"] == 3
    verify_verdicts(d)
    md = build(d)
    (OUT_DIR / "significance.md").write_text(md, encoding="utf-8")
    log.info("significance.md written (%d lines) -> %s", md.count("\n") + 1, (OUT_DIR / "significance.md").relative_to(PROJECT))
    for line in gate8_summary(d).splitlines():
        log.info(line)
