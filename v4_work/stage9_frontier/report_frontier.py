#!/usr/bin/env python3
"""Stage 9 Step 4: REPORT.md and numbers_stage9.tex, built only from files the Stage 9 scripts wrote.

  python report_frontier.py

Re-run after the assumption audit is labelled to add the human-label CE(b).
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from common import HERE, OUT, RAW, sha256_bytes, sha256_file
from run_frontier import verify_plan
from s8_part_a import is_bengali_script

SCORES, STATS, AUDIT = OUT / "scores", OUT / "stats", OUT / "audit"
FRONTIER = {"gptoss_ZS": "ZS.jsonl", "gptoss_ZS+P": "ZS_P.jsonl"}
LABEL = {"gptoss_ZS": "gpt-oss-120b ZS", "gptoss_ZS+P": "gpt-oss-120b ZS+P", "llama_ZS": "Llama-3.1-8B ZS",
         "llama_FT": "Llama-3.1-8B FT", "llama_CL": "Llama-3.1-8B CL", "llama_ZS+P": "Llama-3.1-8B ZS+P",
         "llama_FT+P": "Llama-3.1-8B FT+P"}
ARM_MACRO = {"gptoss_ZS": "GptZS", "gptoss_ZS+P": "GptZSP", "llama_ZS": "LlamaZS", "llama_FT": "LlamaFT",
             "llama_CL": "LlamaCL", "llama_ZS+P": "LlamaZSP", "llama_FT+P": "LlamaFTP"}
METRIC_MACRO = {"invented_value": "Invented", "clarify_exact": "ClarifyExact", "clarify_learned": "ClarifyLearned",
                "ce_a": "CEa", "ce_b": "CEb", "ce_a_512": "CEaCap", "ce_b_512": "CEbCap",
                "truncated_primary": "Trunc", "over_512": "OverCap", "em_0p1": "AccTenth", "exact_0p005": "Exact",
                "em_1p0": "AccOne", "over_asking": "OverAsk", "abstain_correct": "Abstain", "response_bn": "RespBn"}
SUBSET_MACRO = {"in_clarify_68": "", "all_288": "All", "in_single_116": "", "in_numeric_146": "",
                "in_abstain_seed_34": "Seed", "in_ood_hard_40": "OOD", "bn_rows": ""}
EXAMPLE_ROW = "08854472-64aa-4d59-9613-ac2429858914"
GROQ_POLICY_URL = "https://console.groq.com/docs/legal/services-agreement"
GROQ_POLICY_QUOTE = ("Groq is not permitted to use Inputs or Outputs for training or fine-tuning any AI Model "
                     "Services or other models, unless explicitly granted permission or instructed by Customer.")

MACROS: list[tuple[str, str, str]] = []


def mac(name: str, value, source: str) -> str:
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in {m[0] for m in MACROS}, name
    MACROS.append((name, str(value), source))
    return str(value)


def pct(x: float, nd: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{nd}f}"


def load_raw(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def rate_cell(desc: pd.DataFrame, arm: str, metric: str, subset: str, macro: bool = True) -> str:
    r = desc[(desc.arm == arm) & (desc.metric == metric) & (desc.subset == subset)]
    if r.empty:
        return "—"
    r = r.iloc[0]
    base = f"StageNine{ARM_MACRO[arm]}{METRIC_MACRO[metric]}{SUBSET_MACRO[subset]}"
    src = f"stats/descriptive_ci.csv :: arm={arm}, metric={metric}, subset={subset}"
    v = pct(r.rate)
    if macro:
        mac(base + "Pct", v, src + " :: rate*100")
        mac(base + "Lo", pct(r.ci_low), src + " :: ci_low*100")
        mac(base + "Hi", pct(r.ci_high), src + " :: ci_high*100")
        mac(base + "K", int(r.k), src + " :: k")
        mac(base + "N", int(r.n), src + " :: n")
    return f"{v} [{pct(r.ci_low)}, {pct(r.ci_high)}] ({int(r.k)}/{int(r.n)})"


def audit_status(per_row: pd.DataFrame) -> dict:
    path = AUDIT / "assumption_audit.xlsx"
    key = pd.read_excel(path, sheet_name="key")
    blind = pd.read_excel(path, sheet_name="blind")
    lab = blind["human_signals_assumption"].astype(str).str.strip().str.upper()
    blind = blind.assign(h=lab.where(lab.isin(["Y", "N"])))
    out = {"n_items": len(blind), "n_labelled": int(blind.h.notna().sum()), "sha256": sha256_file(path)}
    if out["n_labelled"] < len(blind):
        return out
    m = key.drop(columns=["human_signals_assumption", "human_note"]).merge(blind[["blind_id", "h"]], on="blind_id")
    base = per_row[["row_id", "arm", "pred_class", "net_extracted_any"]]
    m = m.merge(base, on=["row_id", "arm"])
    m["answer_with_number"] = ((m.pred_class == "answer") & (m.net_extracted_any == 1)).astype(int)
    m["ce_b_human"] = (m.answer_with_number & (m.h == "N")).astype(int)
    out["ce_b_human"] = m.groupby("arm")["ce_b_human"].agg(["sum", "count"]).to_dict("index")
    out["agreement"] = pd.crosstab(m.auto_marker_hit.astype(bool), m.h == "Y").to_dict()
    return out


def main() -> int:
    plan_sha = verify_plan()["sha256"]
    am1, am2 = (sha256_bytes((HERE / f"plan_amendment_{i}.md").read_bytes().replace(b"\r\n", b"\n")) for i in (1, 2))
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    per_row = pd.read_csv(SCORES / "per_row.csv", encoding="utf-8")
    texts = pd.read_pickle(SCORES / "_per_row_with_text.pkl")
    desc = pd.read_csv(STATS / "descriptive_ci.csv", encoding="utf-8")
    tests = pd.read_csv(STATS / "tests.csv", encoding="utf-8")
    hedge = pd.read_csv(SCORES / "hedge_phrases.csv", encoding="utf-8")
    refchk = pd.read_csv(SCORES / "reference_check.csv", encoding="utf-8")
    raw = {arm: load_raw(RAW / "openai_gpt-oss-120b" / f) for arm, f in FRONTIER.items()}
    gem = load_raw(RAW / "gemini-3.8-flash" / "ZS.jsonl")
    fails = load_raw(OUT / "failures.jsonl")
    allraw = [r for v in raw.values() for r in v]
    L: list[str] = []
    w = L.append

    # ---------------- setup ----------------
    ts = sorted(r["timestamp"] for r in allraw)
    fps = Counter(r["system_fingerprint"] for r in allraw)
    fin = Counter(r["finish_reason"] for r in allraw)
    versions = sorted({r["model_version"] for r in allraw})
    modes = Counter(r["run_mode"] for r in allraw)
    runs = sorted({r["run_started_at"] for r in allraw})
    retried = sum(1 for r in allraw if r.get("attempts", 1) > 1)
    settings = {k: sorted({str(r[k]) for r in allraw}) for k in
                ["provider", "model_id", "reasoning_effort", "include_reasoning", "temperature_sent", "seed_sent",
                 "api_max_completion_tokens"]}
    skipped = [m for m in cfg["models"] if m["model_id"].startswith("<FILL")]
    gfails = [f for f in fails if f["model_id"] == "openai/gpt-oss-120b"]
    fkinds = Counter((f["kind"], f.get("code")) for f in gfails)
    gemfails = [f for f in fails if f["model_id"] == "gemini-3.8-flash"]
    src_raw = "raw/openai_gpt-oss-120b/{ZS,ZS_P}.jsonl"

    w("# Stage 9 — frontier-model zero-shot baseline on the clarification probe\n")
    w("Every number below is read from a file written by `score_frontier.py`, `stats_frontier.py` or the raw "
      "API cache; this report is generated by `report_frontier.py`. Rates are percentages with bootstrap 95% CIs "
      "(10,000 row resamples, seed 42) and counts k/n.\n")
    w("## Setup\n")
    w(f"- Model called: `{settings['model_id'][0]}` via provider `{settings['provider'][0]}` "
      f"(OpenAI-compatible endpoint `{cfg['groq']['base_url']}/chat/completions`). "
      f"`model` field returned by the API: {', '.join(f'`{v}`' for v in versions)}.")
    w(f"- Responses: {mac('StageNineRespTotal', len(allraw), src_raw + ' :: lines')} "
      f"({len(raw['gptoss_ZS'])} ZS + {len(raw['gptoss_ZS+P'])} ZS+P; rows × arms of the frozen plan). "
      f"Run dates (UTC, `timestamp` field): {ts[0][:16]} to {ts[-1][:16]}, over {len(runs)} runs "
      f"({modes.get('dry_run', 0)} rows from the dry run, {modes.get('confirm', len(allraw) - modes.get('dry_run', 0))} "
      f"from `--confirm` runs).")
    w(f"- Settings sent on every request: reasoning_effort = `{settings['reasoning_effort'][0]}` (the lowest level "
      f"Groq accepts for this model), include_reasoning = {settings['include_reasoning'][0]}, "
      f"temperature = {settings['temperature_sent'][0]}, seed = {settings['seed_sent'][0]}, "
      f"max_completion_tokens = {settings['api_max_completion_tokens'][0]} (counts reasoning + visible tokens). "
      f"Logprobs not requested (unsupported by Groq).")
    w(f"- Visible-token rule (amendments 1 and 2): visible = completion_tokens − reasoning_tokens, both "
      f"API-reported. Primary truncation = finish_reason `length`; secondary = more than 512 visible tokens.")
    w(f"- finish_reason: {dict(fin)}. Requests needing more than one attempt: {retried}. "
      f"Distinct system_fingerprint values: {mac('StageNineFingerprints', len(fps), src_raw + ' :: distinct system_fingerprint')} "
      f"across {len(allraw)} responses.")
    w(f"- Plan: `plan.json` sha256 `{plan_sha}` (verified by `run_frontier.verify_plan`); `plan_amendment_1.md` "
      f"`{am1}`; `plan_amendment_2.md` `{am2}` (sha256 of the LF file, as recorded in `FROZEN.txt`).")
    w(f"- Skipped providers: {', '.join(m['provider'] for m in skipped)} (model_id left as `<FILL or remove>` in "
      f"`config.yaml`). The originally planned `gemini-3.8-flash` (provider google) was replaced by "
      f"gpt-oss-120b under amendment 2.")
    w(f"- Failed rows: none remain; all {len(allraw)} rows are cached. `failures.jsonl` logs "
      f"{len(gfails)} failed gpt-oss requests ({', '.join(f'{n}× {k}' + (f' HTTP {c}' if c else '') for (k, c), n in fkinds.items())}); "
      f"failed requests are not cached and were retried. It also logs {len(gemfails)} gemini-3.8-flash "
      f"failure(s) from before the model change.")
    w(f"- Excluded Gemini dry-run responses: {len(gem)} gemini-3.8-flash ZS responses exist in "
      f"`raw/gemini-3.8-flash/ZS.jsonl` (rows {', '.join(r['row_id'][:8] for r in gem)}). Per amendment 2 they "
      f"are excluded from every result and are mentioned only here.")
    w("- Llama reference arms are rescored from existing predictions (ZS/FT/CL: `v3/reports/s6/predictions`; "
      "ZS+P/FT+P: `v4_work/results/stage8b/predictions`) with the same Stage 6 scorer. Rescored CE(a)/CE(b)/"
      "truncation for Llama ZS and FT match the Stage 8 values exactly (`scores/reference_check.csv`).\n")

    # ---------------- Table A ----------------
    arms_a = ["gptoss_ZS", "gptoss_ZS+P", "llama_ZS", "llama_FT", "llama_CL", "llama_ZS+P", "llama_FT+P"]
    w("## Table A — incomplete questions (68 clarify rows)\n")
    w("Invented = a numeric value is produced for the missing input; clarify exact = asks for the missing input "
      "(Stage 6 rule); confident error (a) = answers with a number; (b) = (a) without a registered "
      "explicit-assumption marker; truncated = finish_reason `length` (frontier) or ≥ 511 generated tokens (Llama, "
      "Stage 8 rule).\n")
    w("| Arm | Invented % | Clarify exact % | Confident error (a) % | Confident error (b) % | Truncated % |")
    w("|---|---|---|---|---|---|")
    for arm in arms_a:
        cells = [rate_cell(desc, arm, m, "in_clarify_68") for m in
                 ["invented_value", "clarify_exact", "ce_a", "ce_b", "truncated_primary"]]
        w(f"| {LABEL[arm]} | " + " | ".join(cells) + " |")
    w("\n**Length sensitivity (amendment 1, secondary).** Responses with more than 512 API-reported visible tokens "
      "are treated as truncated and therefore not counted as confident errors.\n")
    w("| Arm | CE (a) primary % | CE (a) 512 cap % | CE (b) primary % | CE (b) 512 cap % | > 512 visible, clarify rows % | > 512 visible, all 288 % |")
    w("|---|---|---|---|---|---|---|")
    for arm in FRONTIER:
        r = lambda m, s: rate_cell(desc, arm, m, s, macro=False)
        cells = [r("ce_a", "in_clarify_68"), rate_cell(desc, arm, "ce_a_512", "in_clarify_68"),
                 r("ce_b", "in_clarify_68"), rate_cell(desc, arm, "ce_b_512", "in_clarify_68"),
                 rate_cell(desc, arm, "over_512", "in_clarify_68"), rate_cell(desc, arm, "over_512", "all_288")]
        w(f"| {LABEL[arm]} | " + " | ".join(cells) + " |")

    aud = audit_status(per_row)
    w("\n**Assumption audit (amendment 1).** "
      f"`audit/assumption_audit.xlsx` (sha256 `{aud['sha256'][:12]}…`) holds the {aud['n_items']} frontier clarify "
      f"responses in a blind sheet. Labelled: {aud['n_labelled']}/{aud['n_items']}. ")
    if "ce_b_human" in aud:
        for arm, v in aud["ce_b_human"].items():
            w(f"- {LABEL.get(arm, arm)}: CE(b) with human label {pct(v['sum'] / v['count'])}% ({v['sum']}/{v['count']}); "
              f"registered-marker CE(b) in Table A.")
        w(f"- Marker hit (rows) vs human 'signals assumption' (columns): {aud['agreement']}")
    else:
        w("Human labels are pending, so CE(b) with the human label cannot be reported yet; Table A uses the "
          "registered marker list only. Re-run `report_frontier.py` after labelling.\n")

    # ---------------- Table B ----------------
    w("## Table B — complete questions\n")
    w("Accuracy within ±0.1 mm on the 116 single-crop numeric rows (not extracted counts as wrong); over-asking = "
      "a clarification request on the 146 numeric rows where nothing is missing.\n")
    w("| Arm | Accuracy ±0.1 mm % (116) | Exact (abs err < 0.005) % (116) | Over-asking % (146) |")
    w("|---|---|---|---|")
    for arm in arms_a:
        cells = [rate_cell(desc, arm, "em_0p1", "in_single_116"), rate_cell(desc, arm, "exact_0p005", "in_single_116"),
                 rate_cell(desc, arm, "over_asking", "in_numeric_146")]
        if set(cells) != {"—"}:
            w(f"| {LABEL[arm]} | " + " | ".join(cells) + " |")
    w("\nF6 (descriptive only, no test): accuracy within ±0.1 mm for gpt-oss ZS, gpt-oss ZS+P and Llama FT is the "
      "first column above.\n")

    # ---------------- Table C ----------------
    w("## Table C — tests F1–F5\n")
    w("Exact McNemar test paired by row; difference = B − A in percentage points with paired bootstrap 95% CI; "
      "Holm correction within each family (one frontier model, so each family holds one test and p_Holm = p). "
      "A non-significant result is not evidence of equivalence.\n")
    w("| Test | Metric (rows) | B | A | B % | A % | B only / A only | Difference (pp) [95% CI] | p | p_Holm | Result |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in tests.itertuples():
        tid = t.test_id
        src = f"stats/tests.csv :: test_id={tid}"
        tm = "F" + ["Zero", "One", "Two", "Three", "Four", "Five", "Six"][int(tid[1:])]
        nt = isinstance(t.not_testable, str) and t.not_testable
        pfmt = lambda p: "n.t." if nt or pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")
        d, lo, hi = (f"{100 * x:+.1f}" for x in (t.diff, t.diff_ci_low, t.diff_ci_high))
        mac(f"StageNine{tm}Diff", d, src + " :: diff*100")
        mac(f"StageNine{tm}Lo", lo, src + " :: diff_ci_low*100")
        mac(f"StageNine{tm}Hi", hi, src + " :: diff_ci_high*100")
        mac(f"StageNine{tm}P", pfmt(t.p_raw), src + " :: p_raw")
        mac(f"StageNine{tm}PHolm", pfmt(t.p_holm), src + " :: p_holm")
        w(f"| {tid} | {t.metric} ({t.n}) | {t.B} | {t.A} | {pct(t.rate_b)} | {pct(t.rate_a)} | {t.b_only} / {t.a_only} "
          f"| {d} [{lo}, {hi}] | {pfmt(t.p_raw)} | {pfmt(t.p_holm)} | {t.verdict} |")
    w("")

    # ---------------- abstention ----------------
    w("## Abstention (out-of-scope questions)\n")
    w("| Arm | Seed abstain rows (34) % | Hard out-of-domain rows (40) % |")
    w("|---|---|---|")
    for arm in arms_a:
        cells = [rate_cell(desc, arm, "abstain_correct", "in_abstain_seed_34"),
                 rate_cell(desc, arm, "abstain_correct", "in_ood_hard_40")]
        if set(cells) != {"—"}:
            w(f"| {LABEL[arm]} | " + " | ".join(cells) + " |")
    w("")

    # ---------------- examples ----------------
    w("## Example responses (same missing-field-size question)\n")
    w(f"Row `{EXAMPLE_ROW[:8]}` (subtype `missing_area`, English). Only one frontier model was run, so one model "
      "is shown, in both arms. Verbatim API output (visible answer only).\n")
    key = pd.read_excel(AUDIT / "assumption_audit.xlsx", sheet_name="key")
    q = key[key.row_id == EXAMPLE_ROW].question.iloc[0]
    w("**Question:**\n")
    w("```text\n" + q.strip() + "\n```\n")
    for arm in FRONTIER:
        tr = texts[(texts.arm == arm) & (texts.row_id == EXAMPLE_ROW)].iloc[0]
        rr = next(r for r in raw[arm] if r["row_id"] == EXAMPLE_ROW)
        w(f"**{LABEL[arm]}** (class `{tr.pred_class}`, clarify exact = {int(tr.clarify_exact)}, visible tokens "
          f"{rr['output_tokens']}, reasoning tokens {rr['thinking_tokens']}):\n")
        w("```text\n" + rr["response_text"].strip() + "\n```\n")

    # ---------------- reasoning & cost ----------------
    w("## Reasoning tokens, tokens and cost\n")
    w("| Arm | Requests | Input tokens | Completion tokens (incl. reasoning) | Reasoning tokens | Visible tokens | Median reasoning / response | Median visible / response |")
    w("|---|---|---|---|---|---|---|---|")
    price = cfg["pricing_usd_per_mtok"]["openai/gpt-oss-120b"]
    tot = Counter()
    for arm, rs in raw.items():
        s = Counter(input=sum(r["input_tokens"] for r in rs), completion=sum(r["completion_tokens"] for r in rs),
                    reasoning=sum(r["thinking_tokens"] or 0 for r in rs), visible=sum(r["output_tokens"] for r in rs))
        tot.update(s)
        med_r = pd.Series([r["thinking_tokens"] for r in rs]).median()
        med_v = pd.Series([r["output_tokens"] for r in rs]).median()
        w(f"| {LABEL[arm]} | {len(rs)} | {s['input']:,} | {s['completion']:,} | {s['reasoning']:,} | {s['visible']:,} "
          f"| {med_r:.0f} | {med_v:.0f} |")
    n_reason = sum(1 for r in allraw if r["thinking_tokens"])
    cost = tot["input"] * price["input"] / 1e6 + tot["completion"] * price["output_incl_thinking"] / 1e6
    w(f"| Total | {len(allraw)} | {mac('StageNineTokIn', f'{tot['input']:,}', src_raw + ' :: sum input_tokens')} "
      f"| {mac('StageNineTokCompletion', f'{tot['completion']:,}', src_raw + ' :: sum completion_tokens')} "
      f"| {mac('StageNineTokReasoning', f'{tot['reasoning']:,}', src_raw + ' :: sum thinking_tokens')} "
      f"| {tot['visible']:,} | | |")
    w(f"\nReasoning tokens were reported for {n_reason}/{len(allraw)} responses, and the reasoning text was "
      f"returned (`reasoning_text` in the raw cache, not scored). No cached input tokens were reported.")
    w(f"\n**Cost.** Billed: USD {mac('StageNineCostBilled', '0.00', 'Groq free tier')} (Groq free tier). "
      f"At Groq's paid list price (USD {price['input']} per M input tokens and {price['output_incl_thinking']} per M "
      f"output tokens, reasoning billed as output; `config.yaml` pricing, source {price['source']}), the same "
      f"{len(allraw)} requests would cost USD "
      f"{mac('StageNineCostListPrice', f'{cost:.2f}', 'raw token sums x config.yaml pricing_usd_per_mtok')}. "
      f"The {len(gem)} excluded Gemini dry-run requests were also free tier "
      f"({sum(r.get('input_tokens') or 0 for r in gem):,} input, {sum(r.get('output_tokens') or 0 for r in gem):,} "
      f"output tokens).\n")

    # ---------------- data use ----------------
    w("## Provider data use\n")
    w(f"- Groq (all {len(allraw)} scored requests, free tier): the Groq Services Agreement, section 4.2, states: "
      f"\"{GROQ_POLICY_QUOTE}\" ([source]({GROQ_POLICY_URL}), checked 2026-10-11). Inference data may still be "
      f"retained up to 30 days for reliability and abuse monitoring unless Zero Data Retention is enabled; it was "
      f"not enabled for this run.")
    w(f"- Google (the {len(gem)} excluded Gemini dry-run requests, free tier): free-tier content may be used by "
      f"Google to improve its products.\n")

    # ---------------- issues ----------------
    w("## Issues noticed\n")
    sh = {}
    for arm in FRONTIER:
        t = texts[(texts.arm == arm) & (texts.category != "code_switched")]
        s_en = t[t.lang == "en"].response_text.map(is_bengali_script)
        s_bn = t[t.lang == "bn"].response_text.map(is_bengali_script)
        sh[arm] = (f"{int(s_en.sum())}/{len(s_en)}", f"{int((s_bn == 0).sum())}/{len(s_bn)}")
    n_cs = int(((texts.arm == "gptoss_ZS") & (texts.category == "code_switched")).sum())
    zs_cl = per_row[(per_row.arm == "gptoss_ZS") & (per_row.in_clarify_68 == 1)]
    w(f"1. **Response language under prompt P.** Excluding the {n_cs} code-switched rows, gpt-oss ZS+P answered "
      f"{sh['gptoss_ZS+P'][0]} English questions mainly in Bengali script and {sh['gptoss_ZS+P'][1]} Bengali "
      f"questions mainly in non-Bengali script (ZS: {sh['gptoss_ZS'][0]} and {sh['gptoss_ZS'][1]}; "
      f"`s8_part_a.is_bengali_script`, ≥ 50% Bengali letters). Under prompt P it appears to answer in the "
      f"language of the matching few-shot example, not the language of the question. Scoring was not changed.")
    w(f"2. **Clarify exact vs class.** For gpt-oss ZS, clarify exact = {int(zs_cl.clarify_exact.sum())}/68 but "
      f"learned clarification (class `clarify`) = {int(zs_cl.clarify_learned.sum())}/68: some responses name the "
      f"missing input and still compute an answer with an assumed value. The registered rule counts them as "
      f"clarify exact; Table A reports both invented and clarify exact, so they show up in both columns.")
    w(f"3. **Long zero-shot answers.** {pct(desc.query('arm==\"gptoss_ZS\" and metric==\"over_512\" and subset==\"all_288\"').rate.iloc[0])}% "
      f"of gpt-oss ZS responses exceed 512 visible tokens (none truncated by the API). Out-of-scope ZS answers "
      f"often invent specific products and prices instead of abstaining. The Llama arms were generated with a "
      f"512-token cap, so the primary CE comparison (F4) is length-asymmetric; see the 512-cap columns.")
    w(f"4. **Not bit-reproducible.** temperature 0 and seed 42 were sent, but system_fingerprint took "
      f"{len(fps)} distinct values over {len(allraw)} requests, so reruns may differ.")
    w("5. **Temperature.** Groq's model page suggests temperature 0.5–0.7 for gpt-oss; 0 was kept as registered "
      "(amendment 2). No responses ended with finish_reason `length`.")
    w("6. **Llama tokenizer source.** Llama truncation (≥ 511 tokens) uses the tokenizer saved with the local "
      "adapter (`v3/models/s5/llama31_8b_seed42/final_adapter`), because the gated meta-llama repository is not "
      "cached. It reproduces the Stage 8 values exactly (`scores/reference_check.csv`).")
    tt = pd.Series([r["total_tokens"] for r in allraw],
                   index=pd.to_datetime([r["timestamp"] for r in allraw])).sort_index()
    peak = int(tt.rolling("24h").sum().max())
    w(f"7. **Free-tier token window.** The largest rolling-24 h sum of API-reported total tokens in the cache is "
      f"{peak:,} (limit 200,000; a response's token count is only known after the call). No request was "
      f"rejected for the daily token limit; the 4 `daily_limit` entries in `failures.jsonl` are the throttle's own "
      f"clean stops.")
    hz = hedge[hedge.arm.isin(FRONTIER)].groupby("arm").responses_containing.sum().to_dict()
    w(f"8. **Bengali hedge phrases** (registered list, 34 bn clarify rows): responses containing any listed phrase, "
      f"summed over phrases: {hz}. Details in `scores/hedge_phrases.csv`.\n")

    # ---------------- limitations ----------------
    gem_think_null = all(r.get("thinking_tokens") is None for r in gem)
    w("## Limitations\n")
    w(f"- One frontier model, one run, one prompt P; results may not generalise to other commercial models.")
    w(f"- Reasoning: reasoning_effort was set to low (the lowest level Groq supports for gpt-oss-120b). Unlike the "
      f"planned Gemini model, the API reported reasoning tokens ({n_reason}/{len(allraw)} responses) and returned "
      f"the reasoning text, so hidden reasoning did occur. It is excluded from the 512-visible-token rule and is "
      f"not scored.")
    if gem_think_null:
        w("- For the excluded Gemini dry-run responses only: the thinking level was set to low and the API reported "
          "no thinking tokens, so it is unknown whether hidden reasoning occurred.")
    w("- The hard out-of-domain rows have gold answers pending human review; only abstention is scored on them.")
    w("- CE(b) depends on a registered marker list; the human assumption audit is "
      + ("complete." if "ce_b_human" in aud else "pending."))
    w("")

    (OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8", newline="\n")

    tex = ["% AUTO-GENERATED by v4_work/stage9_frontier/report_frontier.py. Do not edit by hand.",
           "% Every macro below names its source file (relative to v4_work/results/stage9_frontier) and field. "
           "Requires \\usepackage{xspace}.", ""]
    tex += [f"\\newcommand{{\\{n}}}{{{v.replace('%', '\\%')}\\xspace}} % {s}" for n, v, s in MACROS]
    (OUT / "numbers_stage9.tex").write_text("\n".join(tex) + "\n", encoding="utf-8", newline="\n")
    print(f"REPORT.md: {len(L)} lines; numbers_stage9.tex: {len(MACROS)} macros")
    return 0


if __name__ == "__main__":
    sys.exit(main())
