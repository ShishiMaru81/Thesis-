"""Stage 8 — Part A: automatic-metric tests on Stage 6 data. Read-only upstream.

Imported lazily by s8_stats.main(). All test definitions come from
reports/stats/test_plan.json (plan_version 2); nothing here adds a test.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd

import s6_eval
import s8_statlib as sl
from s8_stats import (DETECTOR_ARM, MODELS, OUT_DIR, PROJECT, S6_MASTER, S6_PRED, SEEDS, TEST_PATH,
                      TRUNC_TOKENS, load_jsonl, load_tokenizer, log)

MODEL_LABEL = {"llama31_8b": "Llama", "mistral_nemo": "Mistral", "phi35_mini": "Phi-3.5", "phi4_mini": "Phi-4"}
EA_MARKERS_EN = ["assum", "suppos", "let's say", "let us say", "if we take", "taking a typical", "typical value",
                 "not provided", "not given", "not mentioned"]
EA_MARKERS_BN = ["ধরে", "ধরা যাক", "অনুমান", "দেওয়া নেই", "উল্লেখ নেই"]

RESULT_COLS = ["family", "test_id", "comparison", "subset", "n", "test", "statistic", "p_raw", "p_holm",
               "effect_name", "effect", "effect_ci_low", "effect_ci_high", "verdict",
               "role", "holm_family", "seed", "expected_direction", "b", "c", "rate_a", "rate_b",
               "diff", "diff_ci_low", "diff_ci_high", "diff_unit", "n_nonzero", "note"]
BN_BLOCK = (0x0980, 0x09FF)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class Data:
    def __init__(self):
        rows = load_jsonl(TEST_PATH)
        self.test = {r["row_id"]: r for r in rows if r["gold_status"] != "pending_human"}
        ids = list(self.test)
        q = lambda t: [i for i in ids if self.test[i]["qtype"] == t]
        self.numeric = q("numeric_runtime")
        self.factual = q("factual")
        self.clarify = q("clarify")
        self.abstain = q("abstain")
        self.single = [i for i in self.numeric if "scenario" in self.test[i]]
        self.multi = [i for i in self.numeric if "scenarios" in self.test[i]]
        assert len(self.numeric) == 146 and len(self.single) == 116 and len(self.multi) == 30
        assert len(self.clarify) == 68 and len(self.factual) == 114 and len(self.abstain) == 34
        m = pd.read_csv(S6_MASTER)
        m["value"] = pd.to_numeric(m["value"].map({"True": 1, "False": 0}).fillna(m["value"]), errors="coerce")
        self.wide = {arm: g.pivot(index="row_id", columns="metric", values="value") for arm, g in m.groupby("arm")}
        self.arms = sorted(self.wide)
        self._preds: dict[str, dict] = {}
        self._cls: dict[str, pd.Series] = {}
        self._tok: dict[str, pd.Series] = {}
        self._tokenizers: dict = {}

    # -- per-row vectors -------------------------------------------------
    def metric(self, arm: str, metric: str, ids: list[str], fill=None) -> pd.Series:
        s = self.wide[arm].reindex(ids)[metric] if metric in self.wide[arm] else pd.Series(np.nan, index=ids)
        return s.fillna(fill) if fill is not None else s

    def correct_0p1(self, arm: str) -> pd.Series:
        """146 numeric rows: em_0p1 | multi_all_correct | 0 (not extracted)."""
        w = self.wide[arm].reindex(self.numeric)
        s = w["em_0p1"].copy()
        s = s.fillna(w["multi_all_correct"]) if "multi_all_correct" in w else s
        return s.fillna(0).astype(int)

    def preds(self, arm: str) -> dict[str, dict]:
        if arm not in self._preds:
            self._preds[arm] = {p["row_id"]: p for p in load_jsonl(S6_PRED / f"{arm}.jsonl") if p["row_id"] in self.test}
        return self._preds[arm]

    def gen(self, arm: str, rid: str) -> str:
        return self.preds(arm)[rid].get("generation") or ""

    def pred_class(self, arm: str, ids: list[str]) -> pd.Series:
        if arm not in self._cls:
            self._cls[arm] = pd.Series({i: s6_eval.classify_response(self.gen(arm, i)) for i in self.test})
        return self._cls[arm].reindex(ids)

    def tokenizer(self, arm: str):
        mk = next(m for m in MODELS if arm.startswith(m))
        if mk not in self._tokenizers:
            self._tokenizers[mk] = load_tokenizer(mk)
        return self._tokenizers[mk]

    def gen_tokens(self, arm: str, ids: list[str]) -> pd.Series:
        if arm not in self._tok:
            tok = self.tokenizer(arm)
            self._tok[arm] = pd.Series({i: len(tok(self.gen(arm, i), add_special_tokens=False)["input_ids"]) for i in ids})
        return self._tok[arm].reindex(ids)

    def net_extracted_any(self, arm: str, ids: list[str]) -> pd.Series:
        return pd.Series({i: int(s6_eval.extract_numeric_fields(self.gen(arm, i))["net_mm"] is not None) for i in ids})

    def explicit_assumption(self, arm: str, ids: list[str]) -> tuple[pd.Series, dict[str, list[str]]]:
        flags, hits = {}, {}
        for i in ids:
            low = self.gen(arm, i).lower()
            h = [k for k in EA_MARKERS_EN if k in low] + [k for k in EA_MARKERS_BN if k in self.gen(arm, i)]
            flags[i] = int(bool(h))
            hits[i] = h
        return pd.Series(flags), hits

    def confident_error(self, arm: str, strict: bool = False) -> pd.Series:
        ids = self.clarify
        cls = self.pred_class(arm, ids) == "answer"
        trunc = self.gen_tokens(arm, ids) >= TRUNC_TOKENS
        net = self.net_extracted_any(arm, ids) == 1
        ce = cls & ~trunc & net
        if strict:
            ea, _ = self.explicit_assumption(arm, ids)
            ce = ce & (ea == 0)
        return ce.astype(int)

    def lang(self, ids):
        return pd.Series({i: self.test[i]["lang"] for i in ids})

    def fact_pairs(self, ids: list[str]) -> list[tuple[str, str]]:
        """(bn_row_id, en_row_id) pairs by fact_id within ids."""
        by = defaultdict(dict)
        for i in ids:
            by[self.test[i]["fact_id"]][self.test[i]["lang"]] = i
        return [(v["bn"], v["en"]) for v in by.values() if "bn" in v and "en" in v]


# ---------------------------------------------------------------------------
# result-row builders
# ---------------------------------------------------------------------------
def _row(**kw) -> dict:
    r = {c: None for c in RESULT_COLS}
    r.update(kw)
    return r


def mcnemar_row(test_id, family, holm_family, role, b_vec, a_vec, label_b, label_a, subset, expected,
                seed=42, expected_direction="", note=None, test_name="McNemar exact") -> dict:
    m = sl.mcnemar_exact(b_vec, a_vec)
    d, dlo, dhi = sl.paired_boot_diff(np.asarray(b_vec, float), np.asarray(a_vec, float))
    ci_txt = sl.fmt_ci(dlo, dhi, pct=True)
    sign = None if m["odds_ratio"] is None else (1 if (m["b"] > m["c"]) else -1 if m["b"] < m["c"] else 0)
    return _row(family=family, test_id=test_id, comparison=f"{label_b} vs {label_a}", subset=subset, n=m["n"],
                test=test_name, statistic=m["statistic"], p_raw=m["p"], effect_name="odds_ratio_b_over_c",
                effect=m["odds_ratio"], effect_ci_low=m["or_lo"], effect_ci_high=m["or_hi"],
                role=role, holm_family=holm_family, seed=seed, expected_direction=expected_direction,
                b=m["b"], c=m["c"], rate_a=m["rate_a"], rate_b=m["rate_b"], diff=d, diff_ci_low=dlo, diff_ci_high=dhi,
                diff_unit="proportion (x100 = pp)",
                note=note, _expected=expected, _sign=sign, _ci_txt=ci_txt, _lb=label_b, _la=label_a,
                _nt=m.get("not_testable"))


def wilcoxon_row(test_id, family, holm_family, role, b_vec, a_vec, label_b, label_a, subset, expected,
                 seed=42, expected_direction="", note=None, ci_pct=False, diff_unit=None) -> dict:
    w = sl.wilcoxon_paired(b_vec, a_vec)
    d, dlo, dhi = sl.paired_boot_diff(np.asarray(b_vec, float), np.asarray(a_vec, float))
    unit = diff_unit or ("proportion (x100 = pp)" if ci_pct else "metric points")
    ci_txt = (f"rank-biserial r {sl.fmt_ci(w.get('rb_lo'), w.get('rb_hi'))}; mean diff "
              + sl.fmt_ci(dlo, dhi, pct=ci_pct, nd=2) + ("" if ci_pct else f" {unit}"))
    sign = None if w.get("rb") is None else (1 if w["rb"] > 0 else -1 if w["rb"] < 0 else 0)
    return _row(family=family, test_id=test_id, comparison=f"{label_b} vs {label_a}", subset=subset, n=w["n"],
                test="Wilcoxon signed-rank", statistic=w.get("statistic"), p_raw=w.get("p"),
                effect_name="rank_biserial_r", effect=w.get("rb"), effect_ci_low=w.get("rb_lo"), effect_ci_high=w.get("rb_hi"),
                role=role, holm_family=holm_family, seed=seed, expected_direction=expected_direction,
                rate_a=w["mean_a"], rate_b=w["mean_b"], diff=d, diff_ci_low=dlo, diff_ci_high=dhi, diff_unit=unit,
                n_nonzero=w["n_nonzero"],
                note=note, _expected=expected, _sign=sign, _ci_txt=ci_txt, _lb=label_b, _la=label_a, _nt=w.get("not_testable"))


def finalize(rows: list[dict]) -> pd.DataFrame:
    """Holm within (holm_family, seed) over PRIMARY rows; verdicts; drop private keys."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r["role"] == "PRIMARY" and r["holm_family"]:
            groups[(r["holm_family"], r["seed"])].append(i)
    for idxs in groups.values():
        adj = sl.holm([rows[i]["p_raw"] for i in idxs])
        for i, a in zip(idxs, adj):
            rows[i]["p_holm"] = a
    for r in rows:
        if r.get("verdict"):                         # pre-set (descriptive rows)
            for k in ("_sign", "_expected", "_ci_txt", "_lb", "_la", "_nt"):
                r.pop(k, None)
            continue
        p_for_verdict = r["p_holm"] if r["role"] == "PRIMARY" else r["p_raw"]
        r["verdict"] = sl.verdict(p_for_verdict, r.pop("_sign", None), r.pop("_expected", None), r.pop("_ci_txt", ""),
                                  r.pop("_lb", "B"), r.pop("_la", "A"), not_testable=r.pop("_nt", None))
        if r["role"] != "PRIMARY" and r["p_raw"] is not None:
            r["verdict"] = f"[{r['role']}, raw p only] " + r["verdict"]
    return pd.DataFrame(rows)[RESULT_COLS]


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------
def fam_A1(D: Data, seed: int) -> list[dict]:
    out = []
    for m in MODELS:
        zs, ft = f"{m}_zeroshot", f"{m}_seed{seed}"
        out.append(mcnemar_row(f"A1_{m}", "A1", "A1", "PRIMARY", D.correct_0p1(ft), D.correct_0p1(zs),
                               f"{MODEL_LABEL[m]} FT", f"{MODEL_LABEL[m]} ZS", "numeric_runtime (146)", +1,
                               seed=seed, expected_direction="FT > zero-shot", test_name="McNemar exact on correct_0p1"))
    return out


def desc_A1b(D: Data) -> pd.DataFrame:
    recs = []
    for m in MODELS:
        zs, ft = f"{m}_zeroshot", f"{m}_seed42"
        a = D.metric(zs, "abs_err", D.single)
        b = D.metric(ft, "abs_err", D.single)
        ok = a.notna() & b.notna()
        d, lo, hi = sl.paired_boot_diff(b[ok].values, a[ok].values, statfn=np.median)
        recs.append({"model": m, "n_single_crop": len(D.single), "n_both_extracted": int(ok.sum()),
                     "n_dropped": int((~ok).sum()), "median_abs_err_zs": float(a[ok].median()),
                     "median_abs_err_ft": float(b[ok].median()), "median_paired_diff_ft_minus_zs": d,
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(recs)


def fam_A2(D: Data, seed: int) -> list[dict]:
    out = []
    arms = [f"{m}_seed{seed}" for m in MODELS]
    mat = np.column_stack([D.correct_0p1(a).values for a in arms])
    fr = sl.friedman(mat)
    wlo, whi = sl.kendall_w_boot(mat)
    out.append(_row(family="A2", test_id="A2_friedman", comparison=" / ".join(MODEL_LABEL[m] + " FT" for m in MODELS),
                    subset="numeric_runtime (146)", n=fr["n"], test="Friedman on correct_0p1", statistic=fr["statistic"],
                    p_raw=fr["p"], effect_name="kendall_w", effect=fr["kendall_w"], effect_ci_low=wlo, effect_ci_high=whi,
                    role="PRIMARY", holm_family="A2", seed=seed, expected_direction="any difference",
                    note="col means: " + ", ".join(f"{MODEL_LABEL[m]}={v:.3f}" for m, v in zip(MODELS, fr["col_means"])),
                    _expected=None, _sign=1, _ci_txt=f"Kendall's W {sl.fmt_ci(wlo, whi)}", _lb="arms", _la="each other"))
    out.append(mcnemar_row("A2_llama_vs_mistral", "A2", "A2", "PRIMARY", D.correct_0p1(f"mistral_nemo_seed{seed}"),
                           D.correct_0p1(f"llama31_8b_seed{seed}"), "Mistral FT", "Llama FT", "numeric_runtime (146)", 0,
                           seed=seed, expected_direction="not significant (null expected)", test_name="McNemar exact on correct_0p1"))
    large = np.mean([D.correct_0p1(f"{m}_seed{seed}").values for m in ["llama31_8b", "mistral_nemo"]], axis=0)
    small = np.mean([D.correct_0p1(f"{m}_seed{seed}").values for m in ["phi35_mini", "phi4_mini"]], axis=0)
    out.append(wilcoxon_row("A2_tier", "A2", "A2", "PRIMARY", large, small, "tier {Llama,Mistral}", "tier {Phi-3.5,Phi-4}",
                            "numeric_runtime (146), per-row tier-mean correctness", +1, seed=seed,
                            expected_direction="large tier > small tier", ci_pct=True))
    return out


def fam_A3_A4(D: Data) -> list[dict]:
    out = []
    for m in MODELS:
        ft, cl = f"{m}_seed42", f"{m}_seed42_clarify"
        out.append(mcnemar_row(f"A3a_{m}", "A3a", "A3a", "PRIMARY", D.metric(cl, "clarify_exact", D.clarify, 0),
                               D.metric(ft, "clarify_exact", D.clarify, 0), f"{MODEL_LABEL[m]} clarify-FT", f"{MODEL_LABEL[m]} FT",
                               "clarify (68)", +1, expected_direction="clarify-trained > FT", test_name="McNemar exact on clarify_exact"))
        out.append(mcnemar_row(f"A3b_{m}", "A3b", "A3b", "PRIMARY", D.metric(cl, "invented_value", D.clarify, 0),
                               D.metric(ft, "invented_value", D.clarify, 0), f"{MODEL_LABEL[m]} clarify-FT", f"{MODEL_LABEL[m]} FT",
                               "clarify (68)", -1, expected_direction="clarify-trained < FT", test_name="McNemar exact on invented_value"))
        out.append(mcnemar_row(f"A4_{m}", "A4", "A4", "PRIMARY", D.correct_0p1(cl), D.correct_0p1(ft),
                               f"{MODEL_LABEL[m]} clarify-FT", f"{MODEL_LABEL[m]} FT", "numeric_runtime (146)", 0,
                               expected_direction="no harm (null expected)", test_name="McNemar exact on correct_0p1",
                               note="FLAG: Phi-3.5 slot F1 reported dropping in Stage 6E" if m == "phi35_mini" else None))
    return out


def fam_A5(D: Data, caveat: str) -> list[dict]:
    arm = "llama31_8b_seed42_clarify"
    det = {p["row_id"]: p for p in load_jsonl(S6_PRED / f"{DETECTOR_ARM}.jsonl") if p["row_id"] in D.test}
    cls = D.pred_class(arm, D.clarify)
    model_safe = cls.isin(["clarify", "abstain"]).astype(int)
    flag = pd.Series({i: int(bool(det[i].get("detector_flag"))) for i in D.clarify})
    hybrid_safe = ((model_safe == 1) | (flag == 1)).astype(int)
    m = sl.mcnemar_exact(hybrid_safe.values, model_safe.values)   # cell counts only; no test (plan v3)
    d, lo, hi = sl.paired_boot_diff(hybrid_safe.values.astype(float), model_safe.values.astype(float))
    r = _row(family="A5", test_id="A5_detector", comparison="clarify-FT + detector vs clarify-FT", subset="clarify (68)",
             n=m["n"], test="DESCRIPTIVE: risk difference, paired bootstrap (no test; plan v3)", statistic=None, p_raw=None,
             effect_name="risk_difference_safe_handling", effect=d, effect_ci_low=lo, effect_ci_high=hi,
             role="DESCRIPTIVE", holm_family=None, seed=42, expected_direction="n/a",
             b=m["b"], c=m["c"], rate_a=m["rate_a"], rate_b=m["rate_b"], diff=d, diff_ci_low=lo, diff_ci_high=hi,
             diff_unit="proportion (x100 = pp)", note=caveat)
    r["verdict"] = (f"DESCRIPTIVE (oracle-gated diagnostic): safe handling {100*m['rate_a']:.1f}% -> {100*m['rate_b']:.1f}%, "
                    f"risk difference {sl.fmt_ci(lo, hi, pct=True)}; b={m['b']}, c={m['c']} by construction; no test")
    return [r]


def desc_em_denominators(D: Data) -> pd.DataFrame:
    recs = []
    for arm in D.arms:
        c146 = D.correct_0p1(arm)
        em116 = D.metric(arm, "em_0p1", D.single, 0)
        ext116 = D.metric(arm, "net_extracted", D.single, 0)
        mac = D.metric(arm, "multi_all_correct", D.multi, 0)
        ae = D.metric(arm, "abs_err", D.single)
        exact = int((ae < 0.005).sum())            # exact agreement with the reference (single-crop, extracted)
        recs.append({"arm": arm, "correct_0p1_count": int(c146.sum()), "correct_0p1_denominator": 146,
                     "correct_0p1_rate": float(c146.mean()),
                     "exact_0p1_single_count": int(em116.sum()), "exact_0p1_single_denominator": 116,
                     "exact_0p1_single_rate": float(em116.mean()),
                     "exact_lt0p005_single_count": exact, "exact_lt0p005_single_rate": exact / 116,
                     "single_rows_with_extracted_depth": int(ext116.sum()),
                     "multi_all_correct_count": int(mac.sum()), "multi_denominator": 30})
    return pd.DataFrame(recs)


def is_bengali_script(text: str) -> int | None:
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return None
    bn = sum(1 for ch in alpha if BN_BLOCK[0] <= ord(ch) <= BN_BLOCK[1])
    return int(bn / len(alpha) >= 0.5)


def desc_response_language(D: Data) -> pd.DataFrame:
    bn_rows = [i for i in D.test if D.test[i]["lang"] == "bn"]
    arms = [f"{m}_zeroshot" for m in MODELS] + [f"{m}_seed42" for m in MODELS] + [f"{m}_seed42_clarify" for m in MODELS]
    recs = []
    for arm in arms:
        flags = {i: is_bengali_script(D.gen(arm, i)) for i in bn_rows}
        for subset, ids in [("all_bn_questions", bn_rows)] + [(q, [i for i in bn_rows if D.test[i]["qtype"] == q])
                                                                for q in ["numeric_runtime", "factual", "clarify", "abstain"]]:
            v = np.array([flags[i] if flags[i] is not None else 0 for i in ids], dtype=float)
            n_empty = sum(1 for i in ids if flags[i] is None)
            est, lo, hi = sl.boot_ci(v)
            recs.append({"arm": arm, "subset": subset, "n_bn_questions": len(ids), "n_no_alphabetic_chars": n_empty,
                         "share_answered_in_bengali": est, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(recs)


def fam_A6(D: Data, seed: int) -> list[dict]:
    out = []
    num_pairs = D.fact_pairs(D.numeric)
    fac_pairs = D.fact_pairs(D.factual)
    lang = D.lang(D.numeric)
    for m in MODELS:
        arm = f"{m}_seed{seed}"
        c = D.correct_0p1(arm)
        en, bn = c[lang == "en"].values, c[lang == "bn"].values
        f = sl.fisher_2x2(en, bn)
        d, dlo, dhi = float(en.mean() - bn.mean()), None, None
        # unpaired bootstrap of the rate difference
        rng = sl.rng_for()
        boots = rng.choice(en, (sl.N_BOOT, len(en))).mean(1) - rng.choice(bn, (sl.N_BOOT, len(bn))).mean(1)
        dlo, dhi = [float(v) for v in np.percentile(boots, [2.5, 97.5])]
        out.append(_row(family="A6", test_id=f"A6_num_unpaired_{m}", comparison=f"{MODEL_LABEL[m]} FT: en vs bn",
                        subset="numeric_runtime, 73 en vs 73 bn", n=f["n"], test="Fisher exact on correct_0p1",
                        statistic=f["statistic"], p_raw=f["p"], effect_name="odds_ratio_en_over_bn", effect=f["odds_ratio"],
                        effect_ci_low=f["or_lo"], effect_ci_high=f["or_hi"], role="PRIMARY", holm_family="A6", seed=seed,
                        expected_direction="no consistent direction", rate_a=f["rate2"], rate_b=f["rate1"],
                        diff=d, diff_ci_low=dlo, diff_ci_high=dhi, note=f"table [[en_ok,en_no],[bn_ok,bn_no]]={f['table']}",
                        _expected=None, _sign=1 if d > 0 else -1 if d < 0 else 0, _ci_txt=sl.fmt_ci(dlo, dhi, pct=True),
                        _lb="en", _la="bn"))
        bn_ids, en_ids = zip(*num_pairs)
        out.append(mcnemar_row(f"A6_num_paired_{m}", "A6", None, "SECONDARY", c[list(en_ids)].values, c[list(bn_ids)].values,
                               "en", "bn", f"numeric_runtime paired on fact_id ({len(num_pairs)})", None, seed=seed,
                               expected_direction="n/a", note="underpowered; not used for inference",
                               test_name="McNemar exact on correct_0p1"))
        chrf = D.metric(arm, "chrf_pp", D.factual)
        bn_ids, en_ids = zip(*fac_pairs)
        out.append(wilcoxon_row(f"A6_fac_{m}", "A6", "A6", "PRIMARY", chrf[list(en_ids)].values, chrf[list(bn_ids)].values,
                                "en", "bn", f"factual chrF++ paired on fact_id ({len(fac_pairs)}); {MODEL_LABEL[m]} FT", +1,
                                seed=seed, expected_direction="English > Bengali"))
    return out


def desc_A6_clarify(D: Data) -> pd.DataFrame:
    recs = []
    lang = D.lang(D.clarify)
    for arm in [f"{m}_seed42" for m in MODELS] + [f"{m}_seed42_clarify" for m in MODELS]:
        for metric in ["clarify_exact", "invented_value"]:
            v = D.metric(arm, metric, D.clarify, 0)
            for lg in ["bn", "en"]:
                est, lo, hi = sl.boot_ci(v[lang == lg].values)
                recs.append({"arm": arm, "metric": metric, "lang": lg, "n": int((lang == lg).sum()),
                             "rate": est, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(recs)


def fam_A7(D: Data, seed: int) -> list[dict]:
    out = []
    for m in MODELS:
        zs, ft = f"{m}_zeroshot", f"{m}_seed{seed}"
        out.append(mcnemar_row(f"A7_{m}", "A7", "A7", "PRIMARY", D.confident_error(ft), D.confident_error(zs),
                               f"{MODEL_LABEL[m]} FT", f"{MODEL_LABEL[m]} ZS", "clarify (68)", +1, seed=seed,
                               expected_direction="FT > zero-shot", test_name="McNemar exact on confident_error"))
        if seed == 42:
            out.append(mcnemar_row(f"A7_sens_{m}", "A7", None, "SENSITIVITY", D.confident_error(ft, strict=True),
                                   D.confident_error(zs, strict=True), f"{MODEL_LABEL[m]} FT", f"{MODEL_LABEL[m]} ZS",
                                   "clarify (68); confident_error_strict (explicit_assumption=0)", +1, seed=seed,
                                   expected_direction="FT > zero-shot", test_name="McNemar exact on confident_error_strict"))
    return out


def desc_A7(D: Data) -> tuple[pd.DataFrame, list[dict]]:
    recs, examples = [], []
    for m in MODELS:
        ex_m = []
        for variant, arm in [("zeroshot", f"{m}_zeroshot"), ("ft_seed42", f"{m}_seed42")]:
            ids = D.clarify
            tokens = D.gen_tokens(arm, ids)
            trunc = (tokens >= TRUNC_TOKENS).astype(int)
            inv = D.metric(arm, "invented_value", ids, 0).astype(int)
            ea, hits = D.explicit_assumption(arm, ids)
            ce = D.confident_error(arm)
            ces = D.confident_error(arm, strict=True)
            r = {"model": m, "variant": variant, "arm": arm, "n_clarify": len(ids)}
            for name, v in [("truncation_rate", trunc), ("invented_value_rate", inv),
                            ("confident_error_rate_a", ce), ("confident_error_rate_b_strict", ces)]:
                est, lo, hi = sl.boot_ci(v.values)
                r[name], r[name + "_lo"], r[name + "_hi"] = est, lo, hi
            est, lo, hi = sl.boot_ci(tokens.values, statfn=np.median)
            r["median_gen_tokens"], r["median_gen_tokens_lo"], r["median_gen_tokens_hi"] = est, lo, hi
            inv_ids = inv[inv == 1].index
            if len(inv_ids):
                est, lo, hi = sl.boot_ci(ea[inv_ids].values)
            else:
                est, lo, hi = float("nan"), float("nan"), float("nan")
            r["n_invented"] = int(len(inv_ids))
            r["explicit_assumption_rate_among_invented"], r["ea_lo"], r["ea_hi"] = est, lo, hi
            r["explicit_assumption_n_all_clarify"] = int(ea.sum())
            recs.append(r)
            for i in ids:
                if ea[i] == 1 and len(ex_m) < 5:
                    g = D.gen(arm, i)
                    ex_m.append({"model": m, "arm": arm, "row_id": i, "lang": D.test[i]["lang"],
                                 "category": D.test[i]["category"], "markers_hit": hits[i],
                                 "invented_value": int(inv[i]), "gen_tokens": int(tokens[i]),
                                 "generation_snippet": g[:400] + ("…" if len(g) > 400 else "")})
        examples.extend(ex_m)
    return pd.DataFrame(recs), examples


def bootstrap_cis(D: Data) -> pd.DataFrame:
    spec = {
        "em_0p1": ("numeric", lambda a: D.correct_0p1(a).values),
        "em_1p0": ("numeric", lambda a: D.metric(a, "em_1p0", D.single, 0).values),
        "mae": ("numeric_single_extracted", lambda a: D.metric(a, "abs_err", D.single).dropna().values),
        "net_extract_rate": ("numeric", lambda a: D.metric(a, "net_extracted", D.numeric, 0).values),
        "abstain_accuracy": ("abstain", lambda a: D.metric(a, "abstain_correct", D.abstain, 0).values),
        "clarify_exact": ("clarify", lambda a: D.metric(a, "clarify_exact", D.clarify, 0).values),
        "invented_value_rate": ("clarify", lambda a: D.metric(a, "invented_value", D.clarify, 0).values),
        "dice_word": ("factual", lambda a: D.metric(a, "dice_word", D.factual).dropna().values),
        "chrf_pp": ("factual", lambda a: D.metric(a, "chrf_pp", D.factual).dropna().values),
        "bertscore_f1": ("factual", lambda a: D.metric(a, "bertscore_f1", D.factual).dropna().values),
    }
    recs = []
    for arm in D.arms:
        for metric, (subset, fn) in spec.items():
            v = fn(arm)
            est, lo, hi = sl.boot_ci(v)
            recs.append({"arm": arm, "metric": metric, "subset": subset, "n": int(len(v)),
                         "estimate": est, "ci_low": lo, "ci_high": hi, "kind": "bootstrap"})
    df = pd.DataFrame(recs)
    for m in MODELS:
        for metric in spec:
            vals = df[(df.arm.isin([f"{m}_seed{s}" for s in SEEDS])) & (df.metric == metric)]["estimate"]
            recs.append({"arm": f"{m}_FT_3seeds", "metric": metric, "subset": spec[metric][0], "n": 3,
                         "estimate": float(vals.mean()), "ci_low": float(vals.mean() - vals.std(ddof=1)),
                         "ci_high": float(vals.mean() + vals.std(ddof=1)), "kind": "mean±std across seeds 42/1337/2024"})
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _fx(r) -> str:
    e, lo, hi = r["effect"], r["effect_ci_low"], r["effect_ci_high"]
    if e is None or (isinstance(e, float) and math.isnan(e)):
        return "—"
    f = lambda v: "inf" if v == math.inf else f"{v:.2f}"
    return f"{f(e)} [{f(lo)}, {f(hi)}]" if lo is not None else f(e)


def print_primary_table(df: pd.DataFrame, title: str):
    log.info("")
    log.info(title)
    log.info("-" * len(title))
    log.info("%-26s %5s %9s %8s %8s  %-28s %s", "test_id", "n", "stat", "p_raw", "p_holm", "effect [95% CI]", "verdict")
    for _, r in df.iterrows():
        st = "—" if r["statistic"] is None or (isinstance(r["statistic"], float) and math.isnan(r["statistic"])) else f"{r['statistic']:.3g}"
        log.info("%-26s %5s %9s %8s %8s  %-28s %s", r["test_id"], r["n"], st, sl.fmt_p(r["p_raw"]), sl.fmt_p(r["p_holm"]),
                 _fx(r), r["verdict"])


def seed_robustness(df: pd.DataFrame) -> pd.DataFrame:
    prim = df[(df.role == "PRIMARY") & (df.family.isin(["A1", "A2", "A6", "A7"]))]
    recs = []
    for tid, g in prim.groupby("test_id", sort=False):
        r = {"test_id": tid, "family": g.iloc[0]["family"], "expected_direction": g.iloc[0]["expected_direction"]}
        sig_all = True
        for s in SEEDS:
            gs = g[g.seed == s]
            if gs.empty:
                r[f"seed{s}"] = "not run"; sig_all = False; continue
            row = gs.iloc[0]
            sig = row["p_holm"] is not None and row["p_holm"] < 0.05
            sig_all &= bool(sig)
            r[f"seed{s}"] = f"p_holm={sl.fmt_p(row['p_holm'])} eff={_fx(row).split(' [')[0]} {'SIG' if sig else 'ns'}"
        r["robust_all_3_seeds"] = sig_all
        recs.append(r)
    return pd.DataFrame(recs)


def run(plan: dict) -> None:
    assert plan.get("plan_version") == 3, "Part A requires test_plan.json plan_version 3"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    D = Data()
    log.info("=" * 78)
    log.info("PART A — AUTOMATIC METRICS (plan v%d, seed %d, %d bootstrap resamples)", plan["plan_version"], sl.SEED, sl.N_BOOT)
    log.info("=" * 78)
    log.info("correct_0p1 definition: %s", plan["correct_0p1_definition"])

    rows: list[dict] = []
    for s in SEEDS:
        rows += fam_A1(D, s)
    rows += fam_A3_A4(D)
    for s in SEEDS:
        rows += fam_A2(D, s)
    rows += fam_A5(D, plan["a5_in_sample_caveat"])
    for s in SEEDS:
        rows += fam_A6(D, s)
    for s in SEEDS:
        rows += fam_A7(D, s)
    df = finalize(rows)
    df.to_csv(OUT_DIR / "results_automatic.csv", index=False, encoding="utf-8-sig")

    a1b = desc_A1b(D); a1b.to_csv(OUT_DIR / "a1b_descriptive.csv", index=False)
    a6c = desc_A6_clarify(D); a6c.to_csv(OUT_DIR / "a6_clarify_descriptive.csv", index=False)
    a7d, a7ex = desc_A7(D); a7d.to_csv(OUT_DIR / "a7_descriptive.csv", index=False)
    (OUT_DIR / "a7_explicit_assumption_examples.json").write_text(json.dumps(a7ex, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("computing bootstrap CIs for 20 arms x 10 metrics ...")
    bci = bootstrap_cis(D); bci.to_csv(OUT_DIR / "bootstrap_cis.csv", index=False)
    rob = seed_robustness(df); rob.to_csv(OUT_DIR / "seed_robustness_A.csv", index=False)
    emd = desc_em_denominators(D); emd.to_csv(OUT_DIR / "exact_match_denominators.csv", index=False)
    rl = desc_response_language(D); rl.to_csv(OUT_DIR / "response_language.csv", index=False)
    n_prim42 = int(((df.role == "PRIMARY") & (df.seed == 42)).sum())
    log.info("PRIMARY tests at seed 42: %d (plan v3; A5 demoted to DESCRIPTIVE)", n_prim42)
    log.info("")
    log.info("EXACT-MATCH DENOMINATORS per arm: correct_0p1 /146 | exact_0.1 /116 single-crop")
    for _, r in emd.iterrows():
        log.info("  %-28s %3d/146 = %.3f | %3d/116 = %.3f (extracted %3d/116) | multi %2d/30", r.arm, r.correct_0p1_count,
                 r.correct_0p1_rate, r.exact_0p1_single_count, r.exact_0p1_single_rate, r.single_rows_with_extracted_depth,
                 r.multi_all_correct_count)
    log.info("")
    log.info("RESPONSE LANGUAGE (exploratory): share of 181 Bengali questions answered in Bengali script")
    for _, r in rl[rl.subset == "all_bn_questions"].iterrows():
        log.info("  %-28s %.3f [%.3f, %.3f]  (no-alphabetic: %d)", r.arm, r.share_answered_in_bengali, r.ci_low, r.ci_high,
                 r.n_no_alphabetic_chars)

    # ---- console report
    prim42 = df[(df.role == "PRIMARY") & (df.seed == 42)]
    contra = prim42[prim42.verdict.str.contains("CONTRADICT")]
    log.info("")
    log.info("=" * 78)
    log.info("PRIMARY TESTS CONTRADICTING THEIR EXPECTED DIRECTION: %d", len(contra))
    log.info("=" * 78)
    if len(contra):
        print_primary_table(contra, "contradictions (seed 42)")
    for fam in ["A1", "A2", "A3a", "A3b", "A4", "A5", "A6", "A7"]:
        print_primary_table(prim42[prim42.family == fam], f"FAMILY {fam} — PRIMARY (seed 42)")
    sec = df[(df.role != "PRIMARY") & (df.seed == 42)]
    print_primary_table(sec, "SECONDARY / SENSITIVITY (seed 42, raw p only)")

    log.info("")
    log.info("A1b DESCRIPTIVE — abs_err, single-crop rows where both arms extracted (FT − ZS median, 95% CI)")
    for _, r in a1b.iterrows():
        log.info("  %-14s n=%3d dropped=%2d  median ZS=%.3f FT=%.3f  paired median diff=%+.3f [%+.3f, %+.3f] mm",
                 r.model, r.n_both_extracted, r.n_dropped, r.median_abs_err_zs, r.median_abs_err_ft,
                 r.median_paired_diff_ft_minus_zs, r.ci_low, r.ci_high)

    log.info("")
    log.info("A7 DESCRIPTIVE — clarify rows (n=68), zero-shot vs FT seed 42")
    log.info("%-10s %-9s %-16s %-9s %-16s %-14s %-16s %-16s", "model", "variant", "trunc_rate", "med_tok", "invented", "EA|invented",
             "conf_err (a)", "conf_err (b)")
    for _, r in a7d.iterrows():
        log.info("%-10s %-9s %.2f [%.2f,%.2f] %5.0f    %.2f [%.2f,%.2f] %s/%-3d (%.2f)  %.2f [%.2f,%.2f] %.2f [%.2f,%.2f]",
                 MODEL_LABEL[r.model], r.variant, r.truncation_rate, r.truncation_rate_lo, r.truncation_rate_hi, r.median_gen_tokens,
                 r.invented_value_rate, r.invented_value_rate_lo, r.invented_value_rate_hi,
                 int(round(r.explicit_assumption_rate_among_invented * r.n_invented)) if r.n_invented else 0, r.n_invented,
                 r.explicit_assumption_rate_among_invented if r.n_invented else float("nan"),
                 r.confident_error_rate_a, r.confident_error_rate_a_lo, r.confident_error_rate_a_hi,
                 r.confident_error_rate_b_strict, r.confident_error_rate_b_strict_lo, r.confident_error_rate_b_strict_hi)
    log.info("")
    log.info("A7 explicit_assumption examples (%d):", len(a7ex))
    for e in a7ex:
        log.info("  [%s | %s | %s | markers=%s | invented=%d | tokens=%d]", e["arm"], e["lang"], e["category"], e["markers_hit"],
                 e["invented_value"], e["gen_tokens"])
        log.info("     %s", e["generation_snippet"].replace("\n", " ")[:300])

    log.info("")
    log.info("SEED ROBUSTNESS — A1, A2, A6, A7 (significant after Holm on all three seeds?)")
    for _, r in rob.iterrows():
        log.info("  %-26s %-48s %-48s %-48s robust=%s", r.test_id, r.seed42, r.seed1337, r.seed2024, r.robust_all_3_seeds)

    log.info("")
    log.info("Part A outputs -> %s", OUT_DIR.relative_to(PROJECT))
    for f in ["results_automatic.csv", "bootstrap_cis.csv", "a1b_descriptive.csv", "a6_clarify_descriptive.csv",
              "a7_descriptive.csv", "a7_explicit_assumption_examples.json", "seed_robustness_A.csv"]:
        log.info("  %s", f)
