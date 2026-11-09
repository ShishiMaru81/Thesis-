"""Stage 8 — Part B: human-evaluation tests on Stage 7 data. Read-only upstream.

Imported lazily by s8_stats.main(). Test definitions come from test_plan.json v3.
"""
from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import s8_statlib as sl
from s8_part_a import Data, RESULT_COLS, _row, finalize, mcnemar_row, wilcoxon_row
from s8_stats import (ARM_KEY_CANDIDATES, HUMAN_ARMS, OUT_DIR, PROJECT, RATINGS, first_existing, human_frame, log)

ARM_SHORT = {"llama31_8b_zeroshot": "Llama ZS", "llama31_8b_seed42": "Llama FT",
             "llama31_8b_seed42_clarify": "Llama FT+clarify", "mistral_nemo_seed42": "Mistral FT"}
CONTRASTS = {
    "i": ("llama31_8b_zeroshot", "llama31_8b_seed42", ["overall", "numeric_runtime", "factual"], +1, "FT > zero-shot"),
    "ii": ("llama31_8b_seed42", "llama31_8b_seed42_clarify", ["overall", "clarify", "numeric_runtime"], +1, "clarify-trained >= FT"),
    "iii": ("llama31_8b_seed42", "mistral_nemo_seed42", ["overall", "numeric_runtime", "factual"], 0, "not significant (null expected)"),
}
AUDIT = {"fleiss_kappa_nominal_1to5": 0.392, "kripp_alpha_ordinal": 0.721, "fleiss_kappa_farmer": 0.647}
REF_ARM = "llama31_8b_seed42"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class Human:
    def __init__(self):
        r = pd.read_csv(RATINGS, dtype=str, keep_default_na=False)
        r["rating"] = r["rating_1_5"].astype(int)
        r["farmer"] = (r["would_give_farmer"].str.strip().str.lower() == "yes").astype(int)
        hq = human_frame().set_index("row_id")
        r["in_test_set"] = r["row_id"].map(hq["in_test_set"])
        r["translation_status"] = r["row_id"].map(hq["translation_status"])
        self.ratings = r
        self.hq = hq
        self.judg = self.aggregate(r)

    @staticmethod
    def aggregate(r: pd.DataFrame) -> pd.DataFrame:
        """Per-judgment mean_rating, farmer_yes (majority), acceptable."""
        g = r.groupby("judgment_id")
        j = g.agg(arm=("arm", "first"), row_id=("row_id", "first"), pair_id=("pair_id", "first"),
                  qtype=("qtype", "first"), lang=("lang", "first"), in_test_set=("in_test_set", "first"),
                  n_raters=("annotator", "nunique"), mean_rating=("rating", "mean"), farmer_share=("farmer", "mean"))
        j["farmer_yes"] = (j["farmer_share"] > 0.5).astype(int)     # 5 raters -> majority; 1 rater -> that rater
        j["acceptable"] = (j["mean_rating"] >= 4).astype(int)
        return j.reset_index()

    def wide(self, col: str, judg: pd.DataFrame | None = None, subset: str = "test", qtype: str | None = None) -> pd.DataFrame:
        """row_id x arm matrix of `col`, restricted to sample subset / qtype. Rows with any missing arm dropped."""
        j = self.judg if judg is None else judg
        if subset == "test":
            j = j[j["in_test_set"]]
        if qtype and qtype != "overall":
            j = j[j["qtype"] == qtype]
        w = j.pivot(index="row_id", columns="arm", values=col)[HUMAN_ARMS]
        return w.dropna()

    def judg_excluding(self, annotator: str) -> pd.DataFrame:
        return self.aggregate(self.ratings[self.ratings["annotator"] != annotator])

    def overlap_matrix(self, col: str) -> tuple[pd.DataFrame, list[str]]:
        """items x annotators matrix on the 100 shared judgments."""
        ov = self.ratings[self.ratings["is_overlap"] == "yes"]
        m = ov.pivot(index="judgment_id", columns="annotator", values=col)
        assert m.shape == (100, 5) and m.notna().all().all()
        return m, list(m.columns)


# ---------------------------------------------------------------------------
# agreement coefficients (independent implementations)
# ---------------------------------------------------------------------------
def fleiss_kappa_own(mat: np.ndarray) -> float:
    """mat: items x raters (integer categories)."""
    cats = np.unique(mat)
    n_ij = np.stack([(mat == c).sum(axis=1) for c in cats], axis=1).astype(float)   # items x cats
    n = mat.shape[1]
    P_i = ((n_ij ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_j = n_ij.sum(axis=0) / (n_ij.shape[0] * n)
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else float("nan")


def kripp_alpha_own(mat: np.ndarray, level: str) -> float:
    """Krippendorff's alpha, coincidence-matrix form. mat: items x raters, NaN allowed."""
    vals = np.unique(mat[~np.isnan(mat)])
    k = len(vals)
    idx = {v: i for i, v in enumerate(vals)}
    C = np.zeros((mat.shape[0], k))
    for u in range(mat.shape[0]):
        for v in mat[u][~np.isnan(mat[u])]:
            C[u, idx[v]] += 1
    m_u = C.sum(axis=1)
    keep = m_u >= 2
    C, m_u = C[keep], m_u[keep]
    O = (C / (m_u - 1)[:, None]).T @ C - np.diag((C / (m_u - 1)[:, None]).sum(axis=0))
    n_c = O.sum(axis=1)
    n = n_c.sum()
    if level == "nominal":
        delta = 1.0 - np.eye(k)
    elif level == "interval":
        delta = (vals[:, None] - vals[None, :]) ** 2
    elif level == "ordinal":
        delta = np.zeros((k, k))
        for c in range(k):
            for kk in range(k):
                lo, hi = min(c, kk), max(c, kk)
                delta[c, kk] = (n_c[lo:hi + 1].sum() - (n_c[c] + n_c[kk]) / 2) ** 2
    else:
        raise ValueError(level)
    D_o = (O * delta).sum() / n
    D_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return float(1 - D_o / D_e) if D_e > 0 else float("nan")


def kripp_alpha_pkg(mat: np.ndarray, level: str) -> float:
    import krippendorff
    return float(krippendorff.alpha(reliability_data=mat.T, level_of_measurement=level))


def boot_items(mat: np.ndarray, fn, n_boot=sl.N_BOOT, seed=sl.SEED):
    rng = sl.rng_for(seed)
    n = mat.shape[0]
    vals = np.array([fn(mat[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def section_B6(H: Human) -> dict:
    log.info("B6 agreement on the 100 shared judgments ...")
    R, annots = H.overlap_matrix("rating")
    F, _ = H.overlap_matrix("farmer")
    Rm = R.values.astype(float)
    Fm = F.values.astype(float)
    Rc = np.where(Rm <= 2, 0, np.where(Rm == 3, 1, 2)).astype(float)   # bad/mid/good
    out = {"n_items": 100, "n_annotators": 5, "annotators": annots, "coefficients": {}}

    def add(name, mat, fn, pkg_fn=None):
        pt = fn(mat)
        lo, hi = boot_items(mat, fn)
        rec = {"estimate": pt, "ci_low": lo, "ci_high": hi}
        if pkg_fn is not None:
            rec["package_estimate"] = pkg_fn(mat)
            rec["package_vs_own_agree_3dp"] = bool(round(rec["package_estimate"], 3) == round(pt, 3))
        if name in AUDIT:
            rec["audit_value"] = AUDIT[name]
            rec["audit_abs_diff"] = abs(pt - AUDIT[name])
            rec["audit_within_0.005"] = bool(rec["audit_abs_diff"] <= 0.005)
        out["coefficients"][name] = rec
        log.info("  %-34s %.4f [%.3f, %.3f]%s%s", name, pt, lo, hi,
                 f"  pkg={rec['package_estimate']:.4f} agree3dp={rec['package_vs_own_agree_3dp']}" if pkg_fn else "",
                 f"  audit={AUDIT[name]} ok={rec['audit_within_0.005']}" if name in AUDIT else "")

    from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa as sm_fleiss
    sm_fk = lambda m: float(sm_fleiss(aggregate_raters(m.astype(int))[0]))
    add("fleiss_kappa_nominal_1to5", Rm, fleiss_kappa_own, sm_fk)
    add("kripp_alpha_ordinal", Rm, lambda m: kripp_alpha_own(m, "ordinal"), lambda m: kripp_alpha_pkg(m, "ordinal"))
    add("kripp_alpha_interval", Rm, lambda m: kripp_alpha_own(m, "interval"), lambda m: kripp_alpha_pkg(m, "interval"))
    add("fleiss_kappa_bad_mid_good", Rc, fleiss_kappa_own, sm_fk)
    add("fleiss_kappa_farmer", Fm, fleiss_kappa_own, sm_fk)
    add("kripp_alpha_farmer_nominal", Fm, lambda m: kripp_alpha_own(m, "nominal"), lambda m: kripp_alpha_pkg(m, "nominal"))
    loao = {}
    for j, a in enumerate(annots):
        sub = np.delete(Rm, j, axis=1)
        loao[a] = {"alpha_ordinal_without": kripp_alpha_own(sub, "ordinal"), "package": kripp_alpha_pkg(sub, "ordinal")}
        log.info("  LOAO alpha_ordinal without %s: %.4f", a, loao[a]["alpha_ordinal_without"])
    out["leave_one_annotator_out_alpha_ordinal"] = loao
    # stop rules
    bad = [k for k, v in out["coefficients"].items() if "package_vs_own_agree_3dp" in v and not v["package_vs_own_agree_3dp"]]
    if bad:
        raise SystemExit(f"B6 STOP: package and independent implementation disagree at 3 dp for {bad}")
    bad = [k for k, v in out["coefficients"].items() if "audit_within_0.005" in v and not v["audit_within_0.005"]]
    if bad:
        raise SystemExit(f"B6 STOP: recomputed agreement differs from audit by > 0.005 for {bad}: "
                         + str({k: out['coefficients'][k] for k in bad}))
    out["decision"] = ("Primary human measures: would_give_farmer (kappa %.3f) and ordinal Krippendorff alpha (%.3f). "
                       "Nominal kappa on the 1-5 scale (%.3f) is reported for completeness; it treats a 4-vs-5 "
                       "disagreement as a full disagreement and so understates agreement on an ordinal scale."
                       % (out["coefficients"]["fleiss_kappa_farmer"]["estimate"], out["coefficients"]["kripp_alpha_ordinal"]["estimate"],
                          out["coefficients"]["fleiss_kappa_nominal_1to5"]["estimate"]))
    return out


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------
def fam_B1(H: Human) -> list[dict]:
    from statsmodels.stats.contingency_tables import cochrans_q
    out = []
    for role, subset, tag in [("PRIMARY", "test", "test"), ("SENSITIVITY", "all", "all")]:
        w = H.wide("mean_rating", subset=subset)
        fr = sl.friedman(w.values)
        wlo, whi = sl.kendall_w_boot(w.values)
        out.append(_row(family="B1", test_id=f"B1_friedman_{tag}", comparison=" / ".join(ARM_SHORT[a] for a in HUMAN_ARMS),
                        subset=f"human sample {tag} ({len(w)} questions)", n=len(w), test="Friedman on mean_rating",
                        statistic=fr["statistic"], p_raw=fr["p"], effect_name="kendall_w", effect=fr["kendall_w"],
                        effect_ci_low=wlo, effect_ci_high=whi, role=role, holm_family="B1" if role == "PRIMARY" else None,
                        seed=42, expected_direction="any difference",
                        note="arm means: " + ", ".join(f"{ARM_SHORT[a]}={v:.2f}" for a, v in zip(HUMAN_ARMS, fr["col_means"])),
                        _expected=None, _sign=1, _ci_txt=f"Kendall's W {sl.fmt_ci(wlo, whi)}", _lb="arms", _la="each other"))
        f = H.wide("farmer_yes", subset=subset)
        q = cochrans_q(f.values)
        out.append(_row(family="B1", test_id=f"B1_cochranq_{tag}", comparison=" / ".join(ARM_SHORT[a] for a in HUMAN_ARMS),
                        subset=f"human sample {tag} ({len(f)} questions)", n=len(f), test="Cochran's Q on farmer_yes",
                        statistic=float(q.statistic), p_raw=float(q.pvalue), effect_name="none (omnibus)", effect=None,
                        effect_ci_low=None, effect_ci_high=None, role=role, holm_family="B1" if role == "PRIMARY" else None,
                        seed=42, expected_direction="any difference",
                        note="arm farmer_yes rates: " + ", ".join(f"{ARM_SHORT[a]}={f[a].mean():.2f}" for a in HUMAN_ARMS),
                        _expected=None, _sign=1, _ci_txt="", _lb="arms", _la="each other"))
    return out


def fam_B2_B3(H: Human) -> list[dict]:
    out = []
    j_exclA1 = H.judg_excluding("A1")
    variants = [("PRIMARY", "test", None, "test", None), ("SENSITIVITY", "all", None, "all", None),
                ("SENSITIVITY", "test", j_exclA1, "test_exclA1", "A1's ratings excluded")]
    for c, (a, b, subsets, expected, direction) in CONTRASTS.items():
        for sub in subsets:
            for role, sset, judg, tag, note in variants:
                hf = role == "PRIMARY"
                wr = H.wide("mean_rating", judg=judg, subset=sset, qtype=sub)
                out.append(wilcoxon_row(f"B2_{c}_{sub}_{tag}", "B2", "B2" if hf else None, role, wr[b].values, wr[a].values,
                                        ARM_SHORT[b], ARM_SHORT[a], f"human sample {tag}; {sub} (n={len(wr)})", expected,
                                        expected_direction=direction, note=note, diff_unit="rating points (1-5)"))
                wf = H.wide("farmer_yes", judg=judg, subset=sset, qtype=sub)
                out.append(mcnemar_row(f"B3_{c}_{sub}_{tag}", "B3", "B3" if hf else None, role, wf[b].values, wf[a].values,
                                       ARM_SHORT[b], ARM_SHORT[a], f"human sample {tag}; {sub} (n={len(wf)})", expected,
                                       expected_direction=direction, note=note, test_name="McNemar exact on farmer_yes"))
    return out


def fam_B4(H: Human) -> tuple[list[dict], str]:
    import statsmodels.formula.api as smf
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    lines, rows = [], []
    lines.append("STAGE 8 — FAMILY B4: ANNOTATOR-ADJUSTED MODELS (supporting, not primary)")
    lines.append("Method: Python statsmodels fallback. Rscript is not on PATH; R was not installed (policy).")
    lines.append("Limitation: MixedLM treats the 1-5 rating as interval. The registered specification uses groups=annotator "
                 "with a variance component for row_id, i.e. row effects nested within annotator rather than fully crossed.")
    lines.append("BinomialBayesMixedGLM is variational Bayes: posterior means and 95% credible intervals (mean ± 1.96 sd); no p-values.")
    lines.append(f"Reference arm: {REF_ARM}")
    for tag, sset in [("test", "test"), ("all", "all")]:
        d = H.ratings if sset == "all" else H.ratings[H.ratings["in_test_set"]]
        d = d.copy()
        d["arm_c"] = pd.Categorical(d["arm"], categories=[REF_ARM] + [a for a in HUMAN_ARMS if a != REF_ARM])
        lines.append("\n" + "=" * 78 + f"\nSAMPLE: {tag}  (n ratings = {len(d)}, questions = {d.row_id.nunique()}, annotators = {d.annotator.nunique()})\n" + "=" * 78)
        # ---- MixedLM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm("rating ~ C(arm_c)", d, groups=d["annotator"], vc_formula={"row": "0 + C(row_id)"})
            mfit = md.fit(reml=True, method=["lbfgs", "powell"], maxiter=2000)
        lines.append("\n--- MixedLM: rating ~ C(arm), groups=annotator, vc={row: 0 + C(row_id)} (REML) ---")
        lines.append(str(mfit.summary()))
        raw_means = d.groupby("arm")["rating"].mean()
        for a in HUMAN_ARMS:
            if a == REF_ARM:
                continue
            name = f"C(arm_c)[T.{a}]"
            coef, se = float(mfit.params[name]), float(mfit.bse[name])
            lo, hi = coef - 1.96 * se, coef + 1.96 * se
            p = float(mfit.pvalues[name])
            rows.append(_row(family="B4", test_id=f"B4_mixedlm_{tag}_{a}", comparison=f"{ARM_SHORT[a]} vs {ARM_SHORT[REF_ARM]} (adjusted)",
                             subset=f"individual ratings, human sample {tag} (n={len(d)})", n=len(d),
                             test="MixedLM fixed-effect coefficient (Wald)", statistic=float(mfit.tvalues[name]), p_raw=p,
                             effect_name="adjusted_rating_difference", effect=coef, effect_ci_low=lo, effect_ci_high=hi,
                             role="SECONDARY", holm_family=None, seed=42, expected_direction="same direction as B2",
                             rate_a=float(raw_means[REF_ARM]), rate_b=float(raw_means[a]), diff=float(raw_means[a] - raw_means[REF_ARM]),
                             diff_unit="rating points (1-5); diff = raw unadjusted mean difference",
                             note="treats rating as interval", _expected=None, _sign=1 if coef > 0 else -1,
                             _ci_txt=f"adjusted diff {sl.fmt_ci(lo, hi)} rating points", _lb=ARM_SHORT[a], _la=ARM_SHORT[REF_ARM]))
            lines.append(f"  {a:28s} adjusted coef {coef:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]  p={p:.4f}   raw mean diff {raw_means[a]-raw_means[REF_ARM]:+.3f}")
        # ---- Bayes GLM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bm = BinomialBayesMixedGLM.from_formula("farmer ~ C(arm_c)", {"annotator": "0 + C(annotator)", "row": "0 + C(row_id)"}, d)
            bfit = bm.fit_vb()
        lines.append("\n--- BinomialBayesMixedGLM (VB): farmer ~ C(arm) + (1|annotator) + (1|row_id) ---")
        lines.append(str(bfit.summary()))
        raw_f = d.groupby("arm")["farmer"].mean()
        names = list(bfit.model.exog_names)
        for a in HUMAN_ARMS:
            if a == REF_ARM:
                continue
            name = f"C(arm_c)[T.{a}]"
            k = names.index(name)
            mean, sd = float(bfit.fe_mean[k]), float(bfit.fe_sd[k])
            lo, hi = mean - 1.96 * sd, mean + 1.96 * sd
            rows.append(_row(family="B4", test_id=f"B4_bayesglm_{tag}_{a}", comparison=f"{ARM_SHORT[a]} vs {ARM_SHORT[REF_ARM]} (adjusted)",
                             subset=f"individual ratings, human sample {tag} (n={len(d)})", n=len(d),
                             test="BinomialBayesMixedGLM VB posterior (no p-value)", statistic=None, p_raw=None,
                             effect_name="adjusted_log_odds_farmer_yes", effect=mean, effect_ci_low=lo, effect_ci_high=hi,
                             role="SECONDARY", holm_family=None, seed=42, expected_direction="same direction as B3",
                             rate_a=float(raw_f[REF_ARM]), rate_b=float(raw_f[a]), diff=float(raw_f[a] - raw_f[REF_ARM]),
                             diff_unit="proportion (x100 = pp); diff = raw unadjusted farmer_yes difference",
                             note="95% credible interval (posterior mean ± 1.96 sd), variational approximation"))
            rows[-1]["verdict"] = (f"[SECONDARY, Bayesian] adjusted log-odds {mean:+.2f}, 95% credible interval [{lo:+.2f}, {hi:+.2f}]"
                                   + (" — excludes 0" if lo > 0 or hi < 0 else " — includes 0"))
            lines.append(f"  {a:28s} posterior log-odds {mean:+.3f} [95% CrI {lo:+.3f}, {hi:+.3f}]  raw farmer_yes diff {raw_f[a]-raw_f[REF_ARM]:+.3f}")
    return rows, "\n".join(lines)


def fam_B5(H: Human) -> list[dict]:
    out = []
    j = H.judg.copy()
    j["translation_status"] = j["row_id"].map(H.hq["translation_status"])
    authored_pairs = set(j.loc[j["translation_status"] == "authored_bn_pair", "pair_id"])
    for arm in HUMAN_ARMS:
        ja = j[j["arm"] == arm]
        for role, tag, pairs_filter in [("PRIMARY", "all100", None), ("SENSITIVITY", "native82", authored_pairs)]:
            sub = ja if pairs_filter is None else ja[~ja["pair_id"].isin(pairs_filter)]
            wr = sub.pivot(index="pair_id", columns="lang", values="mean_rating").dropna()
            wf = sub.pivot(index="pair_id", columns="lang", values="farmer_yes").dropna()
            note = f"{len(authored_pairs)} of 100 pairs contain an authored translation" if tag == "all100" else "pairs with no authored translation"
            out.append(wilcoxon_row(f"B5_rating_{arm}_{tag}", "B5", "B5" if role == "PRIMARY" else None, role, wr["en"].values, wr["bn"].values,
                                    "en", "bn", f"{ARM_SHORT[arm]}; {len(wr)} bn/en pairs (pair_id)", None,
                                    expected_direction="English >= Bengali", note=note, diff_unit="rating points (1-5)"))
            out.append(mcnemar_row(f"B5_farmer_{arm}_{tag}", "B5", "B5" if role == "PRIMARY" else None, role, wf["en"].values, wf["bn"].values,
                                   "en", "bn", f"{ARM_SHORT[arm]}; {len(wf)} bn/en pairs (pair_id)", None,
                                   expected_direction="English >= Bengali", note=note, test_name="McNemar exact on farmer_yes"))
    return out


def section_B7(H: Human, D: Data) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    j = H.judg[(H.judg["qtype"] == "factual") & (H.judg["in_test_set"])]
    recs_join = []
    for _, r in j.iterrows():
        w = D.wide[r["arm"]]
        if r["row_id"] in w.index:
            m = w.loc[r["row_id"]]
            recs_join.append({"row_id": r["row_id"], "arm": r["arm"], "mean_rating": r["mean_rating"], "farmer_yes": r["farmer_yes"],
                              **{k: m.get(k) for k in ["dice_word", "dice_char3", "dice_char5", "chrf_pp", "rouge_l", "bertscore_f1"]}})
    df = pd.DataFrame(recs_join)
    assert len(df) == 280, f"B7 join matched {len(df)} rows, expected 280"
    log.info("B7 join: %d factual (row_id, arm) rows matched (expected 280)", len(df))
    rng = sl.rng_for()
    idx = rng.integers(0, len(df), (sl.N_BOOT, len(df)))
    y = df["farmer_yes"].values.astype(float)
    mr = df["mean_rating"].values
    recs = []
    for metric in ["dice_word", "dice_char3", "dice_char5", "chrf_pp", "rouge_l", "bertscore_f1"]:
        x = df[metric].values.astype(float)
        pb = float(stats.pearsonr(x, y)[0])
        auc = float(roc_auc_score(y, x))
        rho = float(stats.spearmanr(x, mr)[0])
        pbs, aucs, rhos = [], [], []
        for i in idx:
            xi, yi, mi = x[i], y[i], mr[i]
            if yi.min() == yi.max():
                pbs.append(np.nan); aucs.append(np.nan)
            else:
                pbs.append(stats.pearsonr(xi, yi)[0]); aucs.append(roc_auc_score(yi, xi))
            rhos.append(stats.spearmanr(xi, mi)[0])
        for stat, pt, vals in [("point_biserial_r_vs_farmer_yes", pb, pbs), ("roc_auc_predicting_farmer_yes", auc, aucs),
                               ("spearman_rho_vs_mean_rating", rho, rhos)]:
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            recs.append({"metric": metric, "statistic": stat, "n": len(df), "estimate": pt, "ci_low": float(lo), "ci_high": float(hi)})
    out = pd.DataFrame(recs)
    best = out[out.statistic == "roc_auc_predicting_farmer_yes"].sort_values("estimate", ascending=False).iloc[0]
    log.info("B7 best metric by ROC-AUC: %s = %.3f [%.3f, %.3f]", best.metric, best.estimate, best.ci_low, best.ci_high)
    return out


def section_B8(H: Human, D: Data) -> tuple[list[dict], pd.DataFrame]:
    ft_arms = ["llama31_8b_seed42", "llama31_8b_seed42_clarify", "mistral_nemo_seed42"]
    j = H.judg[(H.judg["qtype"] == "numeric_runtime") & (H.judg["in_test_set"]) & (H.judg["arm"].isin(ft_arms))]
    recs = []
    for _, r in j.iterrows():
        w = D.wide[r["arm"]]
        ae = w.loc[r["row_id"]].get("abs_err") if r["row_id"] in w.index else np.nan
        recs.append({"row_id": r["row_id"], "arm": r["arm"], "mean_rating": r["mean_rating"], "farmer_yes": r["farmer_yes"],
                     "abs_err": ae, "is_multi": r["row_id"] in D.multi})
    df = pd.DataFrame(recs)
    n_all = len(df)
    have = df.dropna(subset=["abs_err"]).copy()
    log.info("B8: %d FT-family numeric outputs in test set; %d with abs_err (single-crop, depth extracted); %d dropped "
             "(%d multi-crop, %d no extracted depth)", n_all, len(have), n_all - len(have), int(df["is_multi"].sum()),
             int((~df["is_multi"] & df["abs_err"].isna()).sum()))
    rho, p = stats.spearmanr(have["abs_err"], have["mean_rating"])
    rng = sl.rng_for()
    idx = rng.integers(0, len(have), (sl.N_BOOT, len(have)))
    ae, mr = have["abs_err"].values, have["mean_rating"].values
    rhos = np.array([stats.spearmanr(ae[i], mr[i])[0] for i in idx])
    lo, hi = np.nanpercentile(rhos, [2.5, 97.5])
    row = _row(family="B8", test_id="B8_spearman", comparison="abs_err vs mean_rating", subset=f"FT-family numeric outputs in test set with abs_err (n={len(have)} of {n_all})",
               n=len(have), test="Spearman rho", statistic=float(rho), p_raw=float(p), effect_name="spearman_rho", effect=float(rho),
               effect_ci_low=float(lo), effect_ci_high=float(hi), role="PRIMARY", holm_family="B8", seed=42, expected_direction="rho < 0",
               diff_unit="n/a", _expected=-1, _sign=1 if rho > 0 else -1, _ci_txt=f"rho {sl.fmt_ci(lo, hi)}", _lb="abs_err", _la="rating")
    bins = pd.cut(have["abs_err"], bins=[-np.inf, 0.005, 0.1, 1.0, np.inf], labels=["exact", "<=0.1", "<=1.0", ">1.0"])
    have["bin"] = bins
    brecs = []
    for b in ["exact", "<=0.1", "<=1.0", ">1.0"]:
        s = have[have["bin"] == b]
        mr_e, mr_lo, mr_hi = sl.boot_ci(s["mean_rating"].values)
        fy_e, fy_lo, fy_hi = sl.boot_ci(s["farmer_yes"].values.astype(float))
        brecs.append({"bin_abs_err_mm": b, "n": len(s), "mean_rating": mr_e, "mean_rating_lo": mr_lo, "mean_rating_hi": mr_hi,
                      "farmer_yes_rate": fy_e, "farmer_yes_lo": fy_lo, "farmer_yes_hi": fy_hi})
    return [row], pd.DataFrame(brecs)


def arm_summary(H: Human) -> pd.DataFrame:
    recs = []
    for sset in ["test", "all"]:
        j = H.judg if sset == "all" else H.judg[H.judg["in_test_set"]]
        for arm in HUMAN_ARMS:
            ja = j[j["arm"] == arm]
            for col in ["mean_rating", "farmer_yes", "acceptable"]:
                e, lo, hi = sl.boot_ci(ja[col].values.astype(float))
                recs.append({"sample": sset, "arm": arm, "n": len(ja), "measure": col, "estimate": e, "ci_low": lo, "ci_high": hi})
            for q in ["numeric_runtime", "factual", "clarify", "abstain"]:
                jq = ja[ja["qtype"] == q]
                for col in ["mean_rating", "farmer_yes"]:
                    e, lo, hi = sl.boot_ci(jq[col].values.astype(float))
                    recs.append({"sample": sset, "arm": arm, "n": len(jq), "measure": f"{col}/{q}", "estimate": e, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
def _fx(r) -> str:
    e, lo, hi = r["effect"], r["effect_ci_low"], r["effect_ci_high"]
    if e is None or (isinstance(e, float) and math.isnan(e)):
        return "—"
    f = lambda v: "inf" if v == math.inf else f"{v:.2f}"
    return f"{f(e)} [{f(lo)}, {f(hi)}]" if lo is not None else f(e)


def print_table(df: pd.DataFrame, title: str):
    log.info("")
    log.info(title)
    log.info("-" * len(title))
    log.info("%-38s %4s %8s %8s %8s  %-26s %s", "test_id", "n", "stat", "p_raw", "p_holm", "effect [95% CI]", "verdict")
    for _, r in df.iterrows():
        st = "—" if r["statistic"] is None or (isinstance(r["statistic"], float) and math.isnan(r["statistic"])) else f"{r['statistic']:.3g}"
        log.info("%-38s %4s %8s %8s %8s  %-26s %s", r["test_id"], r["n"], st, sl.fmt_p(r["p_raw"]), sl.fmt_p(r["p_holm"]), _fx(r), r["verdict"])


def run(plan: dict) -> None:
    assert plan.get("plan_version") == 3
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 78)
    log.info("PART B — HUMAN EVALUATION (plan v3)")
    log.info("=" * 78)
    H = Human()
    D = Data()
    log.info("judgments: %d; per arm %s; in_test_set %d/%d", len(H.judg), dict(H.judg.arm.value_counts()),
             int(H.judg.in_test_set.sum()), len(H.judg))
    agreement = section_B6(H)          # runs first: stop rules
    (OUT_DIR / "agreement.json").write_text(json.dumps(agreement, indent=2), encoding="utf-8")

    rows = []
    rows += fam_B1(H)
    rows += fam_B2_B3(H)
    rows += fam_B5(H)
    b8_rows, b8_bins = section_B8(H, D)
    rows += b8_rows
    b4_rows, b4_txt = fam_B4(H)
    rows += b4_rows
    df = finalize(rows)
    df.to_csv(OUT_DIR / "results_human.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "mixed_model.txt").write_text(b4_txt, encoding="utf-8")
    b8_bins.to_csv(OUT_DIR / "b8_error_bins.csv", index=False)
    dice = section_B7(H, D); dice.to_csv(OUT_DIR / "dice_validation.csv", index=False)
    summ = arm_summary(H); summ.to_csv(OUT_DIR / "human_arm_summary.csv", index=False)

    prim = df[df.role == "PRIMARY"]
    contra = prim[prim.verdict.str.contains("CONTRADICT")]
    log.info("")
    log.info("PRIMARY HUMAN TESTS CONTRADICTING EXPECTED DIRECTION: %d", len(contra))
    if len(contra):
        print_table(contra, "contradictions")
    for fam in ["B1", "B2", "B3", "B5", "B8"]:
        print_table(prim[prim.family == fam], f"FAMILY {fam} — PRIMARY")
    print_table(df[(df.role == "SENSITIVITY") & (df.family.isin(["B1", "B2", "B3"]))], "B1-B3 SENSITIVITY (all 200 / excl. A1) — raw p")
    print_table(df[(df.role == "SENSITIVITY") & (df.family == "B5")], "B5 SENSITIVITY (82 native pairs) — raw p")
    print_table(df[df.family == "B4"], "B4 annotator-adjusted (SECONDARY)")
    log.info("")
    log.info("B8 error bins (FT-family numeric, test set):")
    for _, r in b8_bins.iterrows():
        log.info("  %-6s n=%3d  mean rating %.2f [%.2f, %.2f]  farmer_yes %.2f [%.2f, %.2f]", r.bin_abs_err_mm, r.n, r.mean_rating,
                 r.mean_rating_lo, r.mean_rating_hi, r.farmer_yes_rate, r.farmer_yes_lo, r.farmer_yes_hi)
    log.info("")
    log.info("B7 dice validation (factual, 280 rows):")
    for _, r in dice.iterrows():
        log.info("  %-13s %-32s %.3f [%.3f, %.3f]", r.metric, r.statistic, r.estimate, r.ci_low, r.ci_high)
    log.info("")
    log.info("Part B outputs -> reports/stats: results_human.csv, agreement.json, mixed_model.txt, dice_validation.csv, "
             "b8_error_bins.csv, human_arm_summary.csv")
