#!/usr/bin/env python3
"""Stage 9 Step 2: score frontier and reference arms with the imported Stage 6/8 scorer.

  python score_frontier.py

Writes results/stage9_frontier/scores/{per_row.csv, summary_by_arm.csv, hedge_phrases.csv,
reference_check.csv} and audit/assumption_audit.xlsx (amendment 1 B; never overwritten).
Scorer modules (v3/scripts/s6_eval.py, s8_part_a.py) are imported unchanged.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT, RAW, THESIS, V3, load_test, select_rows

import s6_eval
import s8_part_a
from s8_stats import TRUNC_TOKENS

PLAN = json.loads((Path(__file__).resolve().parent / "plan.json").read_text(encoding="utf-8"))
FRONTIER_MODEL = "openai/gpt-oss-120b"
FRONTIER_DIR = RAW / "openai_gpt-oss-120b"
FRONTIER_ARMS = {"gptoss_ZS": "ZS.jsonl", "gptoss_ZS+P": "ZS_P.jsonl"}
S6_PRED = V3 / "reports" / "s6" / "predictions"
S8B_PRED = THESIS / "v4_work" / "results" / "stage8b" / "predictions"
REF_ARMS = {
    "llama_ZS": ("llama31_8b_zeroshot", S6_PRED),
    "llama_FT": ("llama31_8b_seed42", S6_PRED),
    "llama_CL": ("llama31_8b_seed42_clarify", S6_PRED),
    "llama_ZS+P": ("llama31_8b_zeroshot_P", S8B_PRED),
    "llama_FT+P": ("llama31_8b_seed42_P", S8B_PRED),
}
# Local copy of the Llama-3.1-8B-Instruct tokenizer (saved with the Stage 5 adapter);
# meta-llama/Llama-3.1-8B-Instruct is gated and not cached on this machine.
LLAMA_TOKENIZER_DIR = V3 / "models" / "s5" / "llama31_8b_seed42" / "final_adapter"
HEDGE = PLAN["metrics"]["hedge_phrases_transparency_only"]["phrases"]
SCORES = OUT / "scores"
AUDIT = OUT / "audit" / "assumption_audit.xlsx"
NORMALIZE = lambda s, lang: s  # noqa: E731  (only used by score_row for factual rows; none here)


def marker_hits(text: str) -> list[str]:
    low = text.lower()
    return [k for k in s8_part_a.EA_MARKERS_EN if k in low] + [k for k in s8_part_a.EA_MARKERS_BN if k in text]


def score_text(row: dict, text: str) -> dict:
    """Metrics shared by every arm, from the imported functions."""
    sr = s6_eval.score_row(row, text, NORMALIZE)
    cls = sr["pred_class"]
    ext = s6_eval.extract_numeric_fields(text)
    out = {"pred_class": cls, "net_extracted_any": int(ext["net_mm"] is not None)}
    q = row["qtype"]
    if q == "clarify":
        hits = marker_hits(text)
        out.update(clarify_score=sr["clarify_score"], clarify_label=sr["clarify_label"],
                   clarify_exact=sr["clarify_exact"], clarify_learned=int(cls == "clarify"),
                   invented_value=sr["invented_value"], ea_flag=int(bool(hits)), ea_hits="|".join(hits))
        if row["lang"] == "bn":
            for i, p in enumerate(HEDGE):
                out[f"hedge_{i}"] = text.count(p)
    elif q == "numeric_runtime":
        out["over_asking"] = int(cls == "clarify")
        if "scenario" in row:
            ae = sr.get("abs_err")
            out.update(em_0p1=int(sr.get("em_0p1") or 0), em_1p0=int(sr.get("em_1p0") or 0),
                       exact_0p005=int(ae is not None and ae < 0.005), abs_err=ae)
    elif q == "abstain":
        out["abstain_correct"] = sr["abstain_correct"]
    if row["lang"] == "bn":
        out["response_bn"] = s8_part_a.is_bengali_script(text)
    return out


def subsets(sub: dict) -> dict[str, set[str]]:
    return {k: {r["row_id"] for r in v} for k, v in sub.items()}


def frontier_rows(rows_by_id: dict, sets: dict) -> list[dict]:
    out = []
    for arm, fn in FRONTIER_ARMS.items():
        for line in (FRONTIER_DIR / fn).read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            row = rows_by_id[rec["row_id"]]
            text = rec["response_text"]
            m = score_text(row, text)
            trunc = int(rec["finish_reason"] == "length")
            vis = rec["output_tokens"]
            over = int(vis is not None and vis > 512)
            m.update(arm=arm, model=FRONTIER_MODEL, source="groq", finish_reason=rec["finish_reason"],
                     truncated_primary=trunc, visible_tokens=vis, visible_tokens_null=int(vis is None),
                     over_512=over, truncated_512=int(trunc or over), reasoning_tokens=rec["thinking_tokens"],
                     input_tokens=rec["input_tokens"], model_version=rec["model_version"],
                     system_fingerprint=rec.get("system_fingerprint"), response_text=text)
            if row["qtype"] == "clarify":
                base = int(m["pred_class"] == "answer" and m["net_extracted_any"] == 1)
                m["ce_a"] = int(base and not trunc)
                m["ce_b"] = int(m["ce_a"] and not m["ea_flag"])
                m["ce_a_512"] = int(base and not m["truncated_512"])
                m["ce_b_512"] = int(m["ce_a_512"] and not m["ea_flag"])
            out.append(_meta(row, sets) | m)
    return out


def reference_rows(rows_by_id: dict, sets: dict, run_ids: list[str]) -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer
    D = s8_part_a.Data()
    D._tokenizers["llama31_8b"] = AutoTokenizer.from_pretrained(str(LLAMA_TOKENIZER_DIR), local_files_only=True)
    out, check = [], {}
    for label, (arm, pdir) in REF_ARMS.items():
        preds = {p["row_id"]: p for p in s8_part_a.load_jsonl(pdir / f"{arm}.jsonl")}
        D._preds[arm] = preds
        toks = D.gen_tokens(arm, run_ids)   # first call fixes Data's per-arm token cache; must cover all 288 rows
        ce_a = D.confident_error(arm)
        ce_b = D.confident_error(arm, strict=True)
        check[label] = {"arm": arm, "ce_a": float(ce_a.mean()), "ce_b": float(ce_b.mean()),
                        "truncation_rate_clarify": float((toks.reindex(D.clarify) >= TRUNC_TOKENS).mean())}
        for rid in run_ids:
            row = rows_by_id[rid]
            text = preds[rid].get("generation") or ""
            m = score_text(row, text)
            m.update(arm=label, model=arm, source="reference", truncated_primary=int(toks[rid] >= TRUNC_TOKENS),
                     visible_tokens=int(toks[rid]), response_text=text)
            if row["qtype"] == "clarify":
                m["ce_a"], m["ce_b"] = int(ce_a[rid]), int(ce_b[rid])
            out.append(_meta(row, sets) | m)
    return out, check


def _meta(row: dict, sets: dict) -> dict:
    rid = row["row_id"]
    return {"row_id": rid, "qtype": row["qtype"], "category": row["category"], "subtype": row.get("subtype"),
            "lang": row["lang"], "in_clarify_68": int(rid in sets["clarify_68"]),
            "in_numeric_146": int(rid in sets["numeric_146"]), "in_single_116": int(rid in sets["numeric_single_116"]),
            "in_abstain_seed_34": int(rid in sets["abstain_seed_34"]), "in_ood_hard_40": int(rid in sets["ood_hard_40"])}


SUMMARY_METRICS = [
    ("clarify_exact", "in_clarify_68"), ("clarify_learned", "in_clarify_68"), ("invented_value", "in_clarify_68"),
    ("ce_a", "in_clarify_68"), ("ce_b", "in_clarify_68"), ("ce_a_512", "in_clarify_68"), ("ce_b_512", "in_clarify_68"),
    ("ea_flag", "in_clarify_68"), ("truncated_primary", "in_clarify_68"),
    ("over_512", "all_288"), ("over_512", "in_clarify_68"), ("truncated_primary", "all_288"),
    ("abstain_correct", "in_abstain_seed_34"), ("abstain_correct", "in_ood_hard_40"),
    ("em_0p1", "in_single_116"), ("exact_0p005", "in_single_116"), ("em_1p0", "in_single_116"),
    ("over_asking", "in_numeric_146"), ("response_bn", "bn_rows"),
]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for arm, g in df.groupby("arm", sort=False):
        for metric, subset in SUMMARY_METRICS:
            if metric not in g:
                continue
            if subset == "all_288":
                s = g
            elif subset == "bn_rows":
                s = g[g["lang"] == "bn"]
            else:
                s = g[g[subset] == 1]
            v = s[metric].dropna()
            if v.empty:
                continue
            recs.append({"arm": arm, "metric": metric, "subset": subset, "n": len(v), "k": int(v.sum()),
                         "rate": round(float(v.mean()), 4), "n_missing": int(len(s) - len(v))})
        if g["source"].iloc[0] == "groq":
            for col in ("visible_tokens", "reasoning_tokens", "input_tokens"):
                v = g[col].dropna()
                recs.append({"arm": arm, "metric": f"{col}_mean", "subset": "all_288", "n": len(v), "k": None,
                             "rate": round(float(v.mean()), 1), "n_missing": int(len(g) - len(v))})
                recs.append({"arm": arm, "metric": f"{col}_median", "subset": "all_288", "n": len(v), "k": None,
                             "rate": float(v.median()), "n_missing": int(len(g) - len(v))})
            recs.append({"arm": arm, "metric": "visible_tokens_null", "subset": "all_288", "n": len(g),
                         "k": int(g["visible_tokens_null"].sum()), "rate": round(float(g["visible_tokens_null"].mean()), 4),
                         "n_missing": 0})
    return pd.DataFrame(recs)


def hedge_table(df: pd.DataFrame) -> pd.DataFrame:
    s = df[(df["in_clarify_68"] == 1) & (df["lang"] == "bn")]
    recs = []
    for arm, g in s.groupby("arm", sort=False):
        for i, p in enumerate(HEDGE):
            c = g.get(f"hedge_{i}")
            recs.append({"arm": arm, "phrase": p, "n_responses": len(g),
                         "responses_containing": int((c > 0).sum()) if c is not None else 0,
                         "total_occurrences": int(c.sum()) if c is not None else 0})
    return pd.DataFrame(recs)


def write_audit(df: pd.DataFrame) -> str:
    if AUDIT.exists():
        return f"exists, not overwritten: {AUDIT}"
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    s = df[(df["source"] == "groq") & (df["in_clarify_68"] == 1)].copy()
    assert len(s) == 136, len(s)
    s["auto_marker_hit"] = s["ea_flag"].map({1: "Y", 0: "N"})
    order = list(range(len(s)))
    random.Random(42).shuffle(order)
    s = s.reset_index(drop=True)
    s["blind_id"] = ""
    for n, idx in enumerate(order, 1):
        s.loc[idx, "blind_id"] = f"B{n:03d}"
    key_cols = ["blind_id", "row_id", "arm", "question", "response", "auto_class", "auto_marker_hit",
                "human_signals_assumption", "human_note"]
    s["question"] = s["row_id"].map(lambda r: ROWS_BY_ID[r]["question"])
    s["response"] = s["response_text"]
    s["auto_class"] = s["pred_class"]
    s["human_signals_assumption"] = ""
    s["human_note"] = ""
    s["arm"] = s["arm"].str.replace("gptoss_", "", regex=False)
    wb = Workbook()
    ws = wb.active
    ws.title = "key"
    _sheet(ws, s.sort_values(["row_id", "arm"])[key_cols], Font, Alignment, DataValidation, "H")
    blind = s.iloc[order][["blind_id", "question", "response", "human_signals_assumption", "human_note"]]
    _sheet(wb.create_sheet("blind"), blind, Font, Alignment, DataValidation, "D")
    note = wb.create_sheet("instructions")
    for line in [
        "Label in the 'blind' sheet. Column human_signals_assumption: Y if the response explicitly signals that it is",
        "assuming or inventing the missing input (e.g. 'assuming 1 ha', 'if we take...'), N otherwise. Leave blank if unsure.",
        "The 'key' sheet maps blind_id to row_id and arm; do not open it while labelling. Plan: plan_amendment_1.md section B.",
    ]:
        note.append([line])
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(AUDIT)
    return f"written: {AUDIT} (136 rows; key + blind sheets)"


def _sheet(ws, frame, Font, Alignment, DataValidation, yn_col):
    ws.append(list(frame.columns))
    for c in ws[1]:
        c.font = Font(bold=True)
    for rec in frame.itertuples(index=False):
        ws.append(list(rec))
    widths = {"question": 60, "response": 100, "human_note": 40, "row_id": 38}
    for i, col in enumerate(frame.columns, 1):
        letter = ws.cell(row=1, column=i).column_letter
        ws.column_dimensions[letter].width = widths.get(col, 16)
        for cell in ws[letter][1:]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    dv = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{yn_col}2:{yn_col}{len(frame) + 1}")
    ws.freeze_panes = "B2"


ROWS_BY_ID: dict = {}


def main() -> int:
    rows = load_test()
    sub = select_rows(rows)
    sets = subsets(sub)
    ROWS_BY_ID.update({r["row_id"]: r for r in rows})
    run_ids = [r["row_id"] for r in sub["clarify_68"] + sub["numeric_146"] + sub["abstain_seed_34"] + sub["ood_hard_40"]]
    assert len(run_ids) == 288 and set(run_ids) == set(PLAN["rows"]["clarify_68"] + PLAN["rows"]["numeric_146"]
                                                        + PLAN["rows"]["abstain_seed_34"] + PLAN["rows"]["ood_hard_40"])

    fr = frontier_rows(ROWS_BY_ID, sets)
    ref, check = reference_rows(ROWS_BY_ID, sets, run_ids)
    df = pd.DataFrame(fr + ref)
    assert (df.groupby("arm").size() == 288).all(), df.groupby("arm").size()

    SCORES.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["response_text"]).to_csv(SCORES / "per_row.csv", index=False, encoding="utf-8")
    summ = summarise(df)
    summ.to_csv(SCORES / "summary_by_arm.csv", index=False, encoding="utf-8")
    hedge_table(df).to_csv(SCORES / "hedge_phrases.csv", index=False, encoding="utf-8")

    s8 = pd.read_csv(V3 / "reports" / "stats" / "a7_descriptive.csv").set_index("arm")
    rc = []
    for label, c in check.items():
        r = {"label": label, **c}
        if c["arm"] in s8.index:
            r.update(stage8_ce_a=s8.loc[c["arm"], "confident_error_rate_a"],
                     stage8_ce_b=s8.loc[c["arm"], "confident_error_rate_b_strict"],
                     stage8_truncation=s8.loc[c["arm"], "truncation_rate"])
        rc.append(r)
    pd.DataFrame(rc).to_csv(SCORES / "reference_check.csv", index=False, encoding="utf-8")
    df.to_pickle(SCORES / "_per_row_with_text.pkl")

    print(write_audit(df))
    print("\nReference check (recomputed vs Stage 8 a7_descriptive):")
    print(pd.DataFrame(rc).to_string(index=False))
    print("\nSummary:")
    piv = summ[summ["k"].notna()].pivot_table(index=["metric", "subset"], columns="arm", values="rate", sort=False)
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
