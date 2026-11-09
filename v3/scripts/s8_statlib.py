"""Stage 8 statistical primitives. Pure functions; no I/O.

Conventions
-----------
Paired comparisons are expressed as B vs A with effect = B - A:
  b = rows where B=1 and A=0  (B better)
  c = rows where A=1 and B=0  (A better)
  odds ratio = b / c  (> 1 favours B)
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

SEED = 42
N_BOOT = 10_000


def rng_for(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def boot_ci(x, statfn=np.mean, n_boot: int = N_BOOT, seed: int = SEED, axis_fn=True):
    """Percentile 95% CI of statfn over row resamples of a 1-D array.

    If axis_fn, statfn must accept axis=1 on a (n_boot, n) matrix (fast path
    for mean/median). Otherwise it is applied per resample (slow path).
    Returns (point, lo, hi). NaN CI when n == 0.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng_for(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    if axis_fn:
        vals = statfn(x[idx], axis=1)
    else:
        vals = np.array([statfn(x[i]) for i in idx])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(statfn(x)), float(lo), float(hi)


def paired_boot_diff(b, a, statfn=np.mean, n_boot: int = N_BOOT, seed: int = SEED):
    """Percentile 95% CI of statfn(b) - statfn(a) under paired row resampling."""
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    assert b.shape == a.shape
    n = len(b)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng_for(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = statfn(b[idx], axis=1) - statfn(a[idx], axis=1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(statfn(b) - statfn(a)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# McNemar exact
# ---------------------------------------------------------------------------
def mcnemar_exact(b_arr, a_arr) -> dict:
    """Exact (binomial) McNemar test for paired binary outcomes B vs A."""
    b_arr = np.asarray(b_arr, dtype=int)
    a_arr = np.asarray(a_arr, dtype=int)
    assert b_arr.shape == a_arr.shape
    n = len(b_arr)
    b = int(((b_arr == 1) & (a_arr == 0)).sum())
    c = int(((b_arr == 0) & (a_arr == 1)).sum())
    both = int(((b_arr == 1) & (a_arr == 1)).sum())
    neither = int(((b_arr == 0) & (a_arr == 0)).sum())
    out = {"n": n, "b": b, "c": c, "both": both, "neither": neither,
           "rate_a": float(a_arr.mean()) if n else float("nan"),
           "rate_b": float(b_arr.mean()) if n else float("nan")}
    disc = b + c
    if disc == 0:
        out.update(p=None, statistic=0, odds_ratio=None, or_lo=None, or_hi=None,
                   not_testable="identical outcomes (0 discordant pairs)")
        return out
    out["statistic"] = min(b, c)
    out["p"] = float(stats.binomtest(min(b, c), disc, 0.5, alternative="two-sided").pvalue)
    # exact conditional CI for OR via Clopper-Pearson on b/(b+c)
    ci = stats.binomtest(b, disc, 0.5).proportion_ci(confidence_level=0.95, method="exact")
    pl, pu = ci.low, ci.high

    def _or(p):
        if p >= 1.0:
            return math.inf
        return p / (1 - p)

    out["odds_ratio"] = math.inf if c == 0 else b / c
    out["or_lo"] = _or(pl)
    out["or_hi"] = _or(pu)
    return out


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank + rank-biserial
# ---------------------------------------------------------------------------
def rank_biserial(d) -> float:
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return float("nan")
    r = stats.rankdata(np.abs(d))
    pos = r[d > 0].sum()
    neg = r[d < 0].sum()
    return float((pos - neg) / (pos + neg))


def wilcoxon_paired(b, a, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Wilcoxon signed-rank on d = b - a (zeros discarded), rank-biserial r with
    bootstrap CI (row resampling)."""
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    d = b - a
    n = len(d)
    nz = int((d != 0).sum())
    out = {"n": n, "n_nonzero": nz, "median_diff": float(np.median(d)) if n else float("nan"),
           "mean_a": float(a.mean()) if n else float("nan"), "mean_b": float(b.mean()) if n else float("nan")}
    if nz == 0:
        out.update(p=None, statistic=None, rb=None, rb_lo=None, rb_hi=None,
                   not_testable="identical outcomes (all paired differences are zero)")
        return out
    res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
    out["statistic"] = float(res.statistic)
    out["p"] = float(res.pvalue)
    out["rb"] = rank_biserial(d)
    rng = rng_for(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.array([rank_biserial(d[i]) for i in idx])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    out["rb_lo"], out["rb_hi"] = float(lo), float(hi)
    # median paired difference CI (useful for chrF++ etc.)
    mv = np.median(d[idx], axis=1)
    out["median_diff_lo"], out["median_diff_hi"] = [float(v) for v in np.percentile(mv, [2.5, 97.5])]
    return out


# ---------------------------------------------------------------------------
# Friedman + Kendall's W
# ---------------------------------------------------------------------------
def friedman(matrix) -> dict:
    """matrix: (n_blocks, k_arms)."""
    m = np.asarray(matrix, dtype=float)
    n, k = m.shape
    res = stats.friedmanchisquare(*[m[:, j] for j in range(k)])
    chi2 = float(res.statistic)
    return {"n": n, "k": k, "statistic": chi2, "p": float(res.pvalue),
            "kendall_w": chi2 / (n * (k - 1)) if n * (k - 1) else float("nan"),
            "col_means": [float(v) for v in m.mean(axis=0)]}


def kendall_w_boot(matrix, n_boot: int = N_BOOT, seed: int = SEED):
    m = np.asarray(matrix, dtype=float)
    n, k = m.shape
    rng = rng_for(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub = m[idx]
        # Friedman chi2 with tie correction via scipy
        try:
            chi2 = stats.friedmanchisquare(*[sub[:, j] for j in range(k)]).statistic
        except ValueError:
            chi2 = float("nan")
        vals.append(chi2 / (n * (k - 1)))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Fisher exact (unpaired 2x2)
# ---------------------------------------------------------------------------
def fisher_2x2(x_grp1, x_grp2) -> dict:
    """Fisher's exact on [[grp1 success, grp1 fail],[grp2 success, grp2 fail]].
    OR = odds(success | grp1) / odds(success | grp2); Woolf log CI with 0.5
    continuity shift when any cell is zero (statsmodels Table2x2 default)."""
    from statsmodels.stats.contingency_tables import Table2x2
    x1 = np.asarray(x_grp1, dtype=int)
    x2 = np.asarray(x_grp2, dtype=int)
    tbl = np.array([[x1.sum(), len(x1) - x1.sum()], [x2.sum(), len(x2) - x2.sum()]])
    res = stats.fisher_exact(tbl, alternative="two-sided")
    t = Table2x2(tbl, shift_zeros=True)
    lo, hi = t.oddsratio_confint(0.05)
    return {"n": int(len(x1) + len(x2)), "n1": int(len(x1)), "n2": int(len(x2)),
            "table": tbl.tolist(), "statistic": float(res.statistic), "p": float(res.pvalue),
            "odds_ratio": float(t.oddsratio), "or_lo": float(lo), "or_hi": float(hi),
            "rate1": float(x1.mean()), "rate2": float(x2.mean())}


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------
def holm(pvals: list[float | None]) -> list[float | None]:
    """Holm step-down; None (not testable) entries are excluded and returned as None."""
    from statsmodels.stats.multitest import multipletests
    idx = [i for i, p in enumerate(pvals) if p is not None and not (isinstance(p, float) and math.isnan(p))]
    out: list[float | None] = [None] * len(pvals)
    if not idx:
        return out
    adj = multipletests([pvals[i] for i in idx], method="holm")[1]
    for i, a in zip(idx, adj):
        out[i] = float(a)
    return out


# ---------------------------------------------------------------------------
# verdict wording
# ---------------------------------------------------------------------------
def fmt_p(p):
    if p is None:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_ci(lo, hi, nd=2, pct=False):
    if lo is None or hi is None:
        return "—"
    if pct:
        return f"[{100*lo:+.1f}, {100*hi:+.1f}] pp"
    f = lambda v: "inf" if v == math.inf else f"{v:.{nd}f}"
    return f"[{f(lo)}, {f(hi)}]"


def verdict(p_adj, effect_sign: int | None, expected: int | None, ci_text: str,
            label_b: str, label_a: str, alpha: float = 0.05, not_testable: str | None = None,
            metric_word: str = "difference") -> str:
    """expected: +1 (B > A expected), -1 (B < A expected), 0 (null expected), None (any).
    effect_sign: observed sign of the effect (B - A), or None."""
    if not_testable:
        return f"not testable, {not_testable}"
    if p_adj is None:
        return "not run"
    sig = p_adj < alpha
    if not sig:
        base = f"no significant {metric_word} detected"
        if expected == 0:
            return f"{base}; consistent with the pre-registered null expectation; 95% CI of the difference {ci_text}"
        return f"{base}; 95% CI of the difference {ci_text}"
    direction = f"{label_b} > {label_a}" if effect_sign and effect_sign > 0 else f"{label_b} < {label_a}"
    if expected == 0:
        return f"SIGNIFICANT difference ({direction}) — CONTRADICTS the pre-registered null expectation; CI {ci_text}"
    if expected is None:
        return f"significant ({direction}); CI {ci_text}"
    if effect_sign is not None and effect_sign == expected:
        return f"supported: {direction}, significant after Holm; CI {ci_text}"
    return f"SIGNIFICANT IN THE OPPOSITE DIRECTION ({direction}) — CONTRADICTS expected direction; CI {ci_text}"
