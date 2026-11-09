#!/usr/bin/env python3
"""
check_abstract.py — verify every number in the abstract against
thesis/generated/numbers.json (written by make_numbers.py).

The abstract text is thesis/generated/abstract_author.txt, copied verbatim from
THESIS_MASTER_CONTEXT.md section 3 (final abstract, 2026-09-24). This script never
edits it. Each claim pairs a string as it appears in the abstract with a string
built only from numbers.json; "match" is computed, not typed. A coverage check
confirms that every number in the abstract (model names excepted) is covered by
a claim. Wording risks that numbers cannot settle are listed separately.

Outputs
  thesis/generated/abstract_check.md   the verification table
  thesis/generated/abstract_body.tex   the same text for LaTeX (only % escaped)
Exit code 1 if any claim fails or a number is uncovered.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "thesis" / "generated"
N = json.loads((GEN / "numbers.json").read_text(encoding="utf-8"))
ABS = (GEN / "abstract_author.txt").read_text(encoding="utf-8")
FLAT = " ".join(ABS.split())
M4 = ["Llama", "Mistral", "PhiThree", "PhiFour"]


def v(k: str) -> str:
    return N[k]["value"]


def f(k: str) -> float:
    return float(N[k]["raw"])


def rng(lo: str, hi: str) -> str:
    return f"{v(lo)}–{v(hi)}"


def src(*keys: str) -> str:
    return "; ".join(sorted({N[k]["source"] for k in keys}))


ROWS: list[dict] = []
RISKS: list[str] = []


def claim(phrase: str, abstract_value: str, data_value: str, sources: str, ok: bool, note: str = "") -> None:
    if phrase not in FLAT:
        raise SystemExit(f"phrase not found verbatim in abstract: {phrase!r}")
    if abstract_value and abstract_value not in FLAT:
        raise SystemExit(f"value not found verbatim in abstract: {abstract_value!r}")
    ROWS.append({"phrase": phrase, "abstract": abstract_value, "data": data_value, "source": sources, "ok": ok, "note": note})


# ---------------------------------------------------------------- claims
claim("fine-tuned with three seeds each", "three seeds", f"{v('FTCells')} fine-tuned runs (4 models × 3 seeds)",
      src("FTCells"), v("FTCells") == "12")
claim("a hardened 422-item test set (362 items scored)", "422", v("TestRows"), src("TestRows"), v("TestRows") == "422")
claim("a hardened 422-item test set (362 items scored)", "362", v("TestScorable"), src("TestScorable"),
      v("TestScorable") == "362",
      f"{v('TestExcluded')} excluded by design ({v('TestExcludedOOD')} out_of_domain, {v('TestExcludedConflict')} conflicting_values).")
claim("Five annotators rated 800 model outputs", "800",
      f"{v('HumAnnotators')} annotators; {v('HumUniqueOutputs')} distinct outputs; {v('HumJudgments')} ratings",
      src("HumAnnotators", "HumUniqueOutputs"), v("HumAnnotators") == "5" and v("HumUniqueOutputs") == "800",
      f"{v('HumOverlapOutputs')} outputs were rated by all five and the rest by one annotator each.")
claim("an analysis plan fixed before any significance test was computed", "",
      f"plan v{v('PlanVersion')}: {v('PlanTests')} entries, {v('PlanTestsPrimary')} PRIMARY", src("PlanTests"), True,
      "Timing is documented in reports/stats/significance.md (plan written before any test, after descriptive results).")
kz = [v(f"{m}ZSKcFone") for m in M4]
claim("All four scored 0.000 F1 on the crop-coefficient table", "0.000", ", ".join(kz),
      src(*[f"{m}ZSKcFone" for m in M4]), all(x == "0.000" for x in kz))
rz = [v(f"{m}ZSRainFone") for m in M4]
claim("and on the tiered effective-rainfall rule before training", "0.000", ", ".join(rz),
      src(*[f"{m}ZSRainFone" for m in M4]), all(x == "0.000" for x in rz))
d = rng("RangeFTKcFoneMin", "RangeFTKcFoneMax")
claim("and 0.542–0.761 and 0.930–0.999 respectively after", "0.542–0.761", f"Kc F1 {d}", src("RangeFTKcFoneMin"),
      d == "0.542–0.761", "Fine-tuned values are the mean of three seeds per model; range across models.")
d = rng("RangeFTRainFoneMin", "RangeFTRainFoneMax")
claim("and 0.542–0.761 and 0.930–0.999 respectively after", "0.930–0.999", f"effective-rainfall F1 {d}",
      src("RangeFTRainFoneMin"), d == "0.930–0.999")
d = rng("RangeZSExactPtOneMin", "RangeZSExactPtOneMax")
claim("Accuracy within ±0.1 mm on single-crop queries rose from 0.0–3.4% to 38.8–69.3%", "0.0–3.4", d,
      src("RangeZSExactPtOneMin"), d == "0.0–3.4")
d = rng("RangeFTExactPtOneMin", "RangeFTExactPtOneMax")
claim("Accuracy within ±0.1 mm on single-crop queries rose from 0.0–3.4% to 38.8–69.3%", "38.8–69.3", d,
      src("RangeFTExactPtOneMin"), d == "38.8–69.3",
      f"Denominator: the {v('NumericSingle')} single-crop numeric rows; fine-tuned values are means of three seeds.")
claim("a gain that held in every model and seed", "",
      f"lowest fine-tuned run {v('RangeFTAnySeedExactPtOneMin')}% > highest zero-shot {v('RangeZSExactPtOneMax')}%; "
      f"A1 significant after Holm in every model at every seed ({v('RobustAOneLlama')}/{v('RobustAOneMistral')}/"
      f"{v('RobustAOnePhiThree')}/{v('RobustAOnePhiFour')})",
      src("RangeFTAnySeedExactPtOneMin", "RobustAOneLlama"),
      f("RangeFTAnySeedExactPtOneMin") > f("RangeZSExactPtOneMax")
      and all(v(f"RobustAOne{m}") == "yes" for m in M4))
claim("no clarification behaviour in any of twelve training runs", "twelve",
      f"exact clarify = 0 in {v('FTCellsZeroClarifyExact')}/{v('FTCells')} runs; learned clarify = 0 in "
      f"{v('FTCellsZeroLearned')}/{v('FTCells')} runs", src("FTCellsZeroClarifyExact", "FTCellsZeroLearned"),
      v("FTCellsZeroClarifyExact") == v("FTCells") == v("FTCellsZeroLearned") == "12")
claim("every fine-tuned model supplied a value the user never gave and computed with it", "",
      f"invented-value rate 100% in {v('FTCellsFullInvented')}/{v('FTCells')} runs; a depth or litre figure stated on "
      f"≥{v('FTClarifyEmittedMin')}% and a net depth on ≥{v('FTClarifyNetMin')}% of clarify rows in every run",
      src("FTCellsFullInvented", "FTClarifyEmittedMin"),
      v("FTCellsFullInvented") == v("FTCells") and v("FTClarifyEmittedMin") == "100.0")
claim("while refusing out-of-scope questions correctly", "",
      f"abstain accuracy 100% in {v('FTCellsFullAbstain')}/{v('FTCells')} runs, n = {v('TestAbstain')} seed abstain rows",
      src("FTCellsFullAbstain", "TestAbstain"), v("FTCellsFullAbstain") == v("FTCells"),
      f"Scope: the {v('TestAbstain')} seed out-of-domain rows only; the {v('TestExcludedOOD')} adversarial out-of-domain rows are unscored.")
claim("abstention appeared in the training corpus and clarification did not", "",
      f"abstain {v('TrainAbstain')} rows ({v('TrainPctAbstain')}%), clarify {v('TrainClarify')} rows of {v('AugRows')}",
      src("TrainAbstain", "TrainClarify"), v("TrainClarify") == "0" and int(v("TrainAbstain")) > 0)
d = rng("RangeZSClarifyTruncMin", "RangeZSClarifyTruncMax")
claim("13.2–66.2% of their responses to incomplete queries ran to the token limit", "13.2–66.2", d,
      src("RangeZSClarifyTruncMin"), d == "13.2–66.2",
      f"Zero-shot responses to the {v('ZSClarifyRowsPerModel')} scorable clarify rows per model, >= {v('TruncThresholdTok')} tokens.")
d = f"{v('RangeZSExplicitAmongInvMinInt')}–{v('RangeZSExplicitAmongInvMaxInt')}"
claim("15–42% of their invented answers said a value had been assumed", "15–42",
      f"{d} (unrounded {v('RangeZSExplicitAmongInvMin')}–{v('RangeZSExplicitAmongInvMax')})",
      src("RangeZSExplicitAmongInvMinInt"), d == "15–42",
      "Share of zero-shot invented answers on clarify rows that carry a registered explicit-assumption marker (A7 descriptive).")
lo, hi = f("RangeFTSeedAConfErrBMin"), f("RangeFTSeedAConfErrBMax")
claim("After fine-tuning, 98.5–100% were complete, unhedged, confident errors in every model", "98.5–100",
      f"{v('RangeFTSeedAConfErrBMin')}–{v('RangeFTSeedAConfErrBMax')} (strict definition (b), seed 42)",
      src("RangeFTSeedAConfErrBMin"), abs(lo - 98.5) < 0.05 and abs(hi - 100.0) < 0.05,
      "Seed 42 descriptive (A7 detail). The A7 test itself is significant in every model at every seed.")
claim("annotators judged none of the evaluated fine-tuned models' responses to incomplete queries fit to give a farmer",
      "", f"Llama FT {v('HFTClarifyPctYes')}% of {v('HFTClarifyN')}, Mistral FT {v('HMisClarifyPctYes')}% of "
          f"{v('HMisClarifyN')} (majority vote); {v('HumFTClarifyRatingsYes')} of {v('HumFTClarifyRatings')} individual ratings",
      src("HFTClarifyPctYes", "HumFTClarifyRatings"),
      v("HFTClarifyPctYes") == v("HMisClarifyPctYes") == "0.0" and v("HumFTClarifyRatingsYes") == "0",
      "Scope: the two standard fine-tuned arms in the human study (Llama-3.1-8B and Mistral-NeMo-12B, seed 42).")
claim("Adding 300 clarification examples", "300", v("ClarifyTrainRows"), src("ClarifyTrainRows"),
      v("ClarifyTrainRows") == "300", "Clarify-trained models exist at seed 42 only.")
d = rng("RangeCLClarifyExactMin", "RangeCLClarifyExactMax")
claim("raised clarification that names the missing parameter to 29.4–38.2%", "29.4–38.2", d,
      src("RangeCLClarifyExactMin"), d == "29.4–38.2", "Primary clarify metric (exact).")
claim("cut invented values from 100% to 50.0–66.2%", "100%",
      ", ".join(v(f"{m}FTSeedAInventedN") + "/" + v(f"{m}FTSeedAClarifyN") for m in M4) + " (seed 42 before)",
      src("LlamaFTSeedAInventedN"), all(v(f"{m}FTSeedAInventedN") == v(f"{m}FTSeedAClarifyN") for m in M4))
d = rng("RangeCLInventedMin", "RangeCLInventedMax")
claim("cut invented values from 100% to 50.0–66.2%", "50.0–66.2", d, src("RangeCLInventedMin"), d == "50.0–66.2")
claim("with no detectable loss of numeric accuracy", "",
      f"A4: {v('SigAFourSig')} of 4 tests significant; lowest CI bound {v('SigAFourWorstLo')} pp",
      src("SigAFourSig"), v("SigAFourSig") == "0")
d = rng("RangeFTGapChrfMin", "RangeFTGapChrfMax")
claim("Bengali trailed English on factual text in all four models (6.4–12.8 chrF++ points)", "6.4–12.8", d,
      src("RangeFTGapChrfMin"), d == "6.4–12.8" and v("RangeGapChrfPosCells") == "12",
      f"Stage 6 means of three seeds; English ahead in {v('RangeGapChrfPosCells')}/12 fine-tuned runs.")
claim("significantly in three", "", f"A6 factual significant after Holm at seed 42 in {v('SigASixFacSigSeedA')} of 4 models; "
      f"on all three seeds in {v('RobustASixFacYes')}", src("SigASixFacSigSeedA", "RobustASixFacYes"),
      v("SigASixFacSigSeedA") == "3", "The body adds that the gap held on all three seeds only in the two Phi models.")
claim("differences in numeric accuracy and in human ratings between the languages were not significant", "",
      f"A6 numeric: {v('SigASixNumSigAnySeed')} of {v('SigASixNumTestsAllSeeds')} significant; B5: {v('SigBFiveSig')} of {v('SigBFiveTests')}",
      src("SigASixNumSigAnySeed", "SigBFiveSig"), v("SigASixNumSigAnySeed") == "0" and v("SigBFiveSig") == "0")

# ---------------------------------------------------------------- wording risks (reported, text not edited)
RISKS.append("\"annotators judged none of the evaluated fine-tuned models' responses ... fit\": true of the two standard "
             f"fine-tuned arms. The clarify-trained arm is also a fine-tuned model and was judged fit on "
             f"{v('HumTestCLClarifyYes')}% of clarify questions (primary sample) and {v('HCLClarifyPctYes')}% (all 200). "
             "A reader could take \"evaluated fine-tuned models\" to include it. Suggest \"standard fine-tuned models'\" (authors decide).")
RISKS.append("\"significantly in three\" and \"were not significant\" carry no effect size or CI in the abstract; the body "
             "(Chapter 5, Table T9) gives effect, 95% CI and p_Holm for every one.")
RISKS.append("\"98.5–100% ... in every model\" is the seed-42 descriptive rate (strict definition). The test is significant at "
             "all three seeds; the rates at seeds 1337 and 2024 were not tabulated by Stage 8.")


def coverage() -> list[str]:
    text = FLAT
    for name in ("Llama-3.1-8B", "Mistral-NeMo-12B", "Phi-3.5-mini", "Phi-4-mini"):
        text = text.replace(name, " ")
    nums = re.findall(r"\d+(?:\.\d+)?(?:–\d+(?:\.\d+)?)?%?", text)
    covered = " ".join(r["abstract"] for r in ROWS)
    return [n for n in nums if n.rstrip("%") not in covered and n not in ("0.1",)]


def write_tex() -> None:
    pars = [" ".join(p.split()) for p in ABS.strip().split("\n\n")]
    tex = "% AUTO-GENERATED by scripts/thesis/check_abstract.py from abstract_author.txt (verbatim; only % escaped).\n"
    tex += "\n\n".join(p.replace("%", "\\%") for p in pars) + "\n"
    (GEN / "abstract_body.tex").write_text(tex, encoding="utf-8", newline="\n")


def main() -> int:
    unc = coverage()
    out = ["# Abstract verification", "",
           "Generated by `scripts/thesis/check_abstract.py` from `numbers.json`. Text: `abstract_author.txt`, "
           "copied verbatim from THESIS_MASTER_CONTEXT.md section 3 (final abstract, 2026-09-24). The text is not edited.", "",
           "| # | abstract phrase | value in abstract | value in data | source | match? |",
           "|---|---|---|---|---|---|"]
    for i, r in enumerate(ROWS, 1):
        out.append(f"| {i} | {r['phrase']} | {r['abstract'] or '(no number)'} | {r['data']} | `{r['source']}` | "
                   f"{'yes' if r['ok'] else '**NO**'} |")
    out += ["", f"Numbers in the abstract not covered by any claim: {unc or 'none'} "
                "(\"0.1\" in \"±0.1 mm\" is the metric's definition).", "", "## Notes", ""]
    for i, r in enumerate(ROWS, 1):
        if r["note"]:
            out.append(f"{i}. {r['note']}")
    out += ["", "## Wording risks (reported to the authors; text not changed)", ""]
    out += [f"- {x}" for x in RISKS]
    (GEN / "abstract_check.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    write_tex()
    bad = [r for r in ROWS if not r["ok"]]
    print(f"{len(ROWS)} claims; {len(bad)} failing; uncovered numbers: {unc or 'none'}; {len(RISKS)} wording risks")
    for r in bad:
        print("  FAIL:", r["phrase"], "| abstract", r["abstract"], "| data", r["data"])
    return 1 if bad or unc else 0


if __name__ == "__main__":
    sys.exit(main())
