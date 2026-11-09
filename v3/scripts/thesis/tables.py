#!/usr/bin/env python3
"""
tables.py — LaTeX tables T1–T13 (booktabs), numbers from macros only.

Writes thesis/generated/tables/tNN_<name>.tex. Each file is a complete table
float (caption, label, tabular). A validator checks, for every file, that
  * every non-LaTeX control sequence is a macro defined in numbers.json, and
  * no digit appears outside a macro, except in an allow-list of fixed tokens
    (model names, "±0.1 mm"-style metric names, seed numbers, category ids).
Run make_numbers.py first.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "thesis" / "generated"
OUT = GEN / "tables"
N = json.loads((GEN / "numbers.json").read_text(encoding="utf-8"))
M4 = [("Llama", "Llama-3.1-8B"), ("Mistral", "Mistral-NeMo-12B"), ("PhiThree", "Phi-3.5-mini"), ("PhiFour", "Phi-4-mini")]

LATEX_CMDS = {
    "begin", "end", "caption", "label", "centering", "small", "footnotesize", "toprule", "midrule", "bottomrule",
    "cmidrule", "multicolumn", "multirow", "textbf", "emph", "texttt", "textit", "pm", "times", "to", "le", "ge",
    "addlinespace", "rightarrow", "STAGEEIGHTB", "PENDINGREVIEW", "MISSING", "CITENEEDED", "ldots", "textsubscript",
    "mathrm", "leftarrow", "hspace", "vspace", "raggedright", "arraybackslash", "tabularnewline", "newline",
    "setlength", "tabcolsep", "renewcommand", "arraystretch", "S", "si", "bn", "par", "makecell", "Delta", "phantom",
    "resizebox", "linewidth", "textwidth", "hline", "ref", "cite", "nobreakdash", "enspace", "quad", "qquad",
    "checkmark", "text", "textsuperscript", "mbox", "noalign", "rule", "relax", "dots", "cdot", "sim", "approx",
    "sffamily", "ttfamily", "footnote", "tablenotes", "item", "includegraphics", "hfill", "strut", "textsc",
    "scriptsize", "cite",
}
# Tokens containing digits that are names, not measured numbers.
ALLOWED_DIGIT_TOKENS = [
    "Llama-3.1-8B", "Mistral-NeMo-12B", "Phi-3.5-mini", "Phi-4-mini", "Phi-3.5", "Phi-4", "Llama-3.1",
    "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-Nemo-Instruct-2407", "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-4-mini-instruct", "Mistral-Nemo-Instruct-2407", "Llama-3.1-8B-Instruct", "Phi-3.5-mini-instruct", "Phi-4-mini-instruct", "chrF++", "FAO-56", "ET0", "ET_0", "±0.1", "±1.0", "0.1", "1.0", "seed 42",
    "seed 1337", "seed 2024", "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "A1", "A2", "A3", "A4", "A5",
    "A6", "A7", "B3", "B6", "4-bit", "32bit", "SHA-256", "bf16", "NF4", "5-gram", "L12", "v2", "v3", "1–5", "4–5",
    "1–2", "fp16", "MiniLM-L12-v2", "BERTScore-F1", "F1", "tab:", "fig:", "t01", "t02", "t03", "t04", "t05", "t06",
    "t07", "t08", "t09", "t10", "t11", "t12", "t13", "p50", "(1)", "(2)", "(3)", "(4)", "{2}", "{3}", "{4}", "{5}",
    "{6}", "{7}", "{8}", "{9}", "{10}", "{11}", "{12}", "{13}", "2-", "3-", "4-", "5-", "6-", "7-", "8-", "9-", "10-",
    "11-", "12-", "13-", "-2", "-3", "-4", "-5", "-6", "-7", "-8", "-9", "-10", "-11", "-12", "-13", "0pt", "1pt",
    "2pt", "3pt", "4pt", "0.9", "1.15", "1.1", "\\\\[2pt]", "95\\%", "0.005", "t14", "t15",
    "H1 &", "H2 &", "H3 &", "C1 &", "C2 &", "C3 &",
]


def v(name: str) -> str:
    if name not in N:
        raise SystemExit(f"table references unknown macro {name}")
    return "\\" + name + "{}"


def ms(stem: str) -> str:
    return f"{v(stem)}\\,$\\pm$\\,{v(stem + 'SD')}"


TABLES: dict[str, str] = {}


def table(key: str, caption: str, label: str, colspec: str, header: str, body: str, size: str = "\\small",
          note: str = "") -> None:
    TABLES[key] = (
        "\\begin{table}[tbp]\n\\centering\n" + size + "\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
        + (f"\\par\\vspace{{2pt}}\\parbox{{0.95\\textwidth}}{{\\footnotesize {note}}}\n" if note else "")
        + "\\end{table}\n")


# --------------------------------------------------------------------------- T1
def t01() -> None:
    body = "\n".join([
        f"Seed corpus (S0) & {v('SeedRows')} & {v('SeedSourceIds')} & {v('SeedNumeric')} & {v('SeedFactual')} & {v('SeedAbstain')} & -- \\\\",
        f"\\quad train split (S1) & {v('SplitTrainSeedRows')} & {v('SplitTrainSeedGroups')} & {v('SplitTrainSeedNumeric')} & {v('SplitTrainSeedFactual')} & {v('SplitTrainSeedAbstain')} & -- \\\\",
        f"\\quad validation split (S1) & {v('SplitValRows')} & {v('SplitValGroups')} & {v('SplitValNumeric')} & {v('SplitValFactual')} & {v('SplitValAbstain')} & -- \\\\",
        f"\\quad test split (S1) & {v('SplitTestSeedRows')} & {v('SplitTestSeedGroups')} & {v('SplitTestSeedNumeric')} & {v('SplitTestSeedFactual')} & {v('SplitTestSeedAbstain')} & -- \\\\",
        "\\addlinespace",
        f"Training corpus (S2) & {v('AugRows')} & -- & {v('TrainNumeric')} & {v('TrainFactual')} & {v('TrainAbstain')} & {v('TrainClarify')} \\\\",
        f"Clarify-augmented corpus (S6B) & {v('ClarifyCorpusRows')} & -- & {v('TrainNumeric')} & {v('TrainFactual')} & {v('TrainAbstain')} & {v('ClarifyTrainRows')} \\\\",
        "\\addlinespace",
        f"Hardened test set (S4), all rows & {v('TestRows')} & -- & {v('TestAllNumeric')} & {v('TestAllFactual')} & {v('TestAllAbstain')} & {v('TestAllClarify')} \\\\",
        f"Hardened test set, scored rows & {v('TestScorable')} & -- & {v('TestNumeric')} & {v('TestFactual')} & {v('TestAbstain')} & {v('TestClarify')} \\\\",
    ])
    table("t01_data", "Dataset and split statistics. Rows by stage and question type. The seed corpus was split by "
          "source group before any augmentation; each source group holds one Bengali and one English row. "
          "Clarification rows exist only in the hardened test set and in the clarify-augmented corpus.",
          "tab:data", "lrrrrrr",
          "Stage & Rows & Groups & Numeric & Factual & Abstain & Clarify \\\\", body)


# --------------------------------------------------------------------------- T2
def t02() -> None:
    body = "\n".join([
        "A & Paraphrase of seed questions by an external LLM & not run & -- \\\\",
        "B & Back-translation through the other language & not run & -- \\\\",
        f"C & Synthetic numeric scenarios, gold from \\texttt{{irrigation\\_calc}} & run & {v('AugRowsSynthetic')} \\\\",
        f"D & Character noise in crop and unit words & run & {v('AugRowsNoise')} \\\\",
        f"-- & Seed rows carried over unchanged & -- & {v('AugRowsSeed')} \\\\",
        "\\midrule",
        f"\\multicolumn{{3}}{{l}}{{Training corpus}} & {v('AugRows')} \\\\",
    ])
    table("t02_augmentation", f"Augmentation families. Families A and B required calls to an external language model and were "
          f"not run; the training corpus comes from the seed rows plus families C and D. Family D produced "
          f"{v('AugNoiseMade')} rows, of which {v('AugDropBalance')} were removed to keep the two languages exactly balanced.",
          "tab:augmentation", "c>{\\raggedright\\arraybackslash}p{0.5\\textwidth}lr", "Family & Method & Status & Rows kept \\\\", body)


# --------------------------------------------------------------------------- T3
def t03() -> None:
    body = "\n".join([
        f"Exact & identical normalised question & -- & {v('LeakExactCount')} pairs & pass \\\\",
        f"Lexical & character 5-gram Jaccard & {v('LeakJaccardThreshold')} & {v('LeakJaccardPairs')} pairs above threshold; {v('LeakJaccardHitsRefined')} under the refined rule & pass \\\\",
        f"Semantic & multilingual MiniLM cosine & {v('LeakSemThreshold')} & {v('LeakSemCandidates')} candidates; {v('LeakSemDecisionsRecorded')} reviewed & pass, unreviewed \\\\",
        f"Scenario & shared scenario hash & -- & {v('LeakScenarioOverlap')} of {v('LeakScenarioHeldout')} held-out hashes & pass \\\\",
    ])
    table("t03_leakage", "Leakage detectors and outcomes (Stage 3), training corpus against the validation and test splits. "
          f"The {v('LeakJaccardHitsRefined')} lexical pairs share a crop--stage fact but not a scenario; they were accepted as a "
          f"documented limitation. The semantic detector passed with no human review recorded; the reviewed rate is "
          f"\\PENDINGREVIEW{{semantic leakage sample}}.",
          "tab:leakage", ">{\\raggedright\\arraybackslash}p{0.13\\textwidth}>{\\raggedright\\arraybackslash}p{0.22\\textwidth}l>{\\raggedright\\arraybackslash}p{0.3\\textwidth}l",
          "Detector & Signal & Threshold & Finding & Verdict \\\\", body, size="\\footnotesize")


# --------------------------------------------------------------------------- T4
def t04() -> None:
    body = "\n".join([
        f"Misspelled crop & {v('AdvMisspelled')} & {v('AdvMisspelledNumeric')} numeric, {v('AdvMisspelledClarify')} clarify & recoverable ({v('AdvMisspelledSubRecoverable')}), ambiguous ({v('AdvMisspelledSubAmbiguous')}) & yes \\\\",
        f"Missing parameter & {v('AdvMissingParam')} & {v('AdvMissingParamClarify')} clarify & ET0, rainfall, area, stage ({v('AdvMissingEt')} each) & yes \\\\",
        f"Unit ambiguity & {v('AdvUnitAmbig')} & {v('AdvUnitAmbigNumeric')} numeric, {v('AdvUnitAmbigClarify')} clarify & unknown local unit ({v('AdvUnitAmbigSubUnknownLocalUnit')}), regional bigha ({v('AdvUnitAmbigSubBighaRegional')}), stated conversion ({v('AdvUnitAmbigSubExplicitLocalConversion')}) & yes \\\\",
        f"Multi-crop & {v('AdvMultiCrop')} & {v('AdvMultiCropNumeric')} numeric & two answers ({v('AdvMultiCropSubDualAnswer')}), comparison ({v('AdvMultiCropSubComparative')}), split area ({v('AdvMultiCropSubSplitArea')}) & yes \\\\",
        f"Code-switched & {v('AdvCodeSwitched')} & {v('AdvCodeSwitchedNumeric')} numeric & {v('AdvCodeSwitchedSubtypes')} matrix-language variants & yes \\\\",
        f"Conflicting values & {v('AdvConflicting')} & {v('AdvConflictingClarify')} clarify & restated or contradicted inputs & no \\\\",
        f"Out of domain & {v('AdvOutOfDomain')} & {v('AdvOutOfDomainAbstain')} abstain & {v('AdvOutOfDomainSubtypes')} topics absent from training & no \\\\",
        "\\midrule",
        f"Adversarial total & {v('TestRowsAdv')} & & & \\\\",
    ])
    table("t04_hardening", "Test hardening categories (Stage 4). Each category is half Bengali and half English. "
          f"The two unscored categories ({v('TestExcluded')} rows) have no machine-verifiable gold answer and were excluded "
          "by design, so conflicting-value behaviour is not evaluated and abstention is measured on seed rows only.",
          "tab:hardening", ">{\\raggedright\\arraybackslash}p{0.16\\textwidth}r>{\\raggedright\\arraybackslash}p{0.19\\textwidth}>{\\raggedright\\arraybackslash}p{0.36\\textwidth}l",
          "Category & Rows & Question type & Subtypes & Scored \\\\", body, size="\\footnotesize")


# --------------------------------------------------------------------------- T5
def t05() -> None:
    rows = []
    ids = {"Llama": "meta-llama/Llama-3.1-8B-Instruct", "Mistral": "mistralai/Mistral-Nemo-Instruct-2407",
           "PhiThree": "microsoft/Phi-3.5-mini-instruct", "PhiFour": "microsoft/Phi-4-mini-instruct"}
    for M, lab in M4:
        org, name = ids[M].split("/")
        rows.append(f"{lab} & \\makecell[l]{{\\texttt{{{org}/}}\\\\\\texttt{{{name}}}}} & {v(M + 'LoraTargets')} & "
                    f"{v(M + 'LoraLayersAdapted')} & {v(M + 'TrainableParamsM')} & {v(M + 'PeakVramMax')} & {v(M + 'WallMinMean')} \\\\")
    table("t05_models", "Models and adapters. Targets are the projection modules that receive a LoRA adapter; every decoder "
          "layer is adapted. Phi-4-mini exposes fused query--key--value and gate--up projections, so its adapter covers the "
          "same paths through fewer, fused modules. Trainable parameters are in millions and identical across seeds. Peak "
          "VRAM is the maximum over three seeds; time is the mean training time per run on one "
          f"{v('GpuName')}.",
          "tab:models", "llrrrrr",
          "Model & Checkpoint & Targets & Layers & \\makecell{Trainable\\\\ parameters (M)} & \\makecell{VRAM\\\\ (GB)} & Minutes \\\\",
          "\n".join(rows), size="\\footnotesize\\setlength{\\tabcolsep}{4pt}")


# --------------------------------------------------------------------------- T6
def t06() -> None:
    body = "\n".join([
        f"Quantisation & 4-bit base weights, LoRA in bf16 \\\\",
        f"LoRA rank / alpha / dropout & {v('LoraRank')} / {v('LoraAlpha')} / {v('LoraDropout')} \\\\",
        f"Epochs & {v('HpEpochs')} (fixed; no early stopping) \\\\",
        f"Learning rate & ${v('HpLearningRate')}$, {v('HpScheduler')} schedule, {v('HpWarmup')} warm-up steps \\\\",
        f"Optimiser & \\texttt{{{v('HpOptim')}}}, weight decay {v('HpWeightDecay')}, gradient clip {v('HpMaxGradNorm')} \\\\",
        f"Effective batch & {v('HpEffectiveBatch')} (per-device batch $\\times$ accumulation) \\\\",
        f"\\quad Llama / Mistral / Phi-3.5 / Phi-4 & {v('LlamaBatchSize')}$\\times${v('LlamaGradAccum')} / {v('MistralBatchSize')}$\\times${v('MistralGradAccum')} / {v('PhiThreeBatchSize')}$\\times${v('PhiThreeGradAccum')} / {v('PhiFourBatchSize')}$\\times${v('PhiFourGradAccum')} \\\\",
        f"Maximum sequence length & {v('MaxSeqLen')} tokens (longest training example {v('SeqLenGlobalMax')}) \\\\",
        f"Loss & completion tokens only (masking checked on {v('MaskingProbes')} examples) \\\\",
        f"Validation loss & every {v('HpEvalSteps')} steps, monitoring only \\\\",
        f"Seeds & 42, 1337, 2024 (clarify-trained models: 42 only) \\\\",
        f"Decoding & greedy, at most {v('GenMaxNewTokens')} new tokens \\\\",
    ])
    table("t06_hyperparameters", "Training and inference settings, identical for all four models except the per-device batch. "
          "Values are read from the training script and the run manifests.",
          "tab:hyperparameters", "l>{\\raggedright\\arraybackslash}p{0.55\\textwidth}", "Setting & Value \\\\", body)


# --------------------------------------------------------------------------- T7
def t07() -> None:
    body = "\n".join([
        f"Numeric & {v('TestNumeric')} & net depth within $\\pm$0.1 mm ({v('NumericSingle')} single-crop rows; "
        f"Stage 8 tests use all {v('NumericAll')} rows) & exact ($<$0.005 mm) and within $\\pm$1.0 mm, always reported with the "
        f"headline; MAE; RMSE; slot F1 for Kc, ET0, effective rainfall, area; both values correct on {v('NumericMulti')} multi-crop rows \\\\",
        f"Factual & {v('TestFactual')} & chrF++ & exact match; Dice; ROUGE-L; BERTScore F1 (unrescaled) \\\\",
        f"Clarify & {v('TestClarify')} & clarify exact: asks and names the missing parameter & learned clarification (asks, withholds the result); invented-value rate \\\\",
        f"Abstain & {v('TestAbstain')} & abstain accuracy & -- \\\\",
    ])
    table("t07_metrics", "Metric definitions by question type. Each metric asserts the question type of the row it scores and "
          "raises an error on any other type, so no metric is averaged across types. Slot tolerances: Kc "
          f"$\\pm${v('TolKc')}, ET0 and effective rainfall $\\pm${v('TolEt')} mm, area the larger of {v('TolAreaAbs')} m\\textsuperscript{{2}} "
          f"and {v('TolAreaRelPct')}\\%.",
          "tab:metrics", "lr>{\\raggedright\\arraybackslash}p{0.3\\textwidth}>{\\raggedright\\arraybackslash}p{0.36\\textwidth}",
          "Type & Rows & Headline metric & Other metrics \\\\", body, size="\\footnotesize")


# --------------------------------------------------------------------------- T8
def t08() -> None:
    rows = []
    for M, lab in M4:
        rows.append(f"\\multirow{{3}}{{*}}{{{lab}}} & zero-shot & {v(M + 'ZSExactStrict')} & {v(M + 'ZSExactPtOne')} & "
                    f"{v(M + 'ZSExactOne')} & {v(M + 'ZSClarifyExact')} & {v(M + 'ZSAbstainAcc')} & {v(M + 'ZSChrf')} \\\\")
        rows.append(f" & fine-tuned & {ms(M + 'FTExactStrict')} & {ms(M + 'FTExactPtOne')} & {ms(M + 'FTExactOne')} & "
                    f"{ms(M + 'FTClarifyExact')} & {ms(M + 'FTAbstainAcc')} & {ms(M + 'FTChrf')} \\\\")
        rows.append(f" & clarify-trained & {v(M + 'CLExactStrict')} & {v(M + 'CLExactPtOne')} & {v(M + 'CLExactOne')} & "
                    f"{v(M + 'CLClarifyExact')} & {v(M + 'CLAbstainAcc')} & {v(M + 'CLChrf')} \\\\")
        rows.append("\\addlinespace")
    header = ("& & \\multicolumn{3}{c}{Numeric, " + v("NumericSingle") + " single-crop rows (\\%)} & Clarify & Abstain & Factual \\\\\n"
              "\\cmidrule(lr){3-5}\n"
              "Model & Arm & exact & $\\pm$0.1 mm & $\\pm$1.0 mm & exact (\\%) & (\\%) & chrF++ \\\\")
    table("t08_headline", f"Headline results per arm. Numeric accuracy at three tolerances on the same {v('NumericSingle')} "
          f"single-crop rows: exact ($<$0.005 mm), within $\\pm$0.1 mm (the headline, fixed before results) and within "
          f"$\\pm$1.0 mm. Clarify exact: asks and names the missing parameter, {v('TestClarify')} rows. Abstain: "
          f"{v('TestAbstain')} seed out-of-domain rows. Factual chrF++: {v('TestFactual')} rows. Fine-tuned values are mean "
          "$\\pm$ standard deviation over three seeds; zero-shot is a single run; clarify-trained is seed 42 only. "
          "No value is averaged across question types.",
          "tab:headline", "llrrrrrr", header, "\n".join(rows[:-1]), size="\\footnotesize")


# --------------------------------------------------------------------------- T9
def ci(stem: str) -> str:
    return f"{v(stem + 'Diff')} [{v(stem + 'DiffLo')}, {v(stem + 'DiffHi')}]"


def t09() -> None:
    rows = []
    for M, lab in M4:
        n, f = f"SigASixNum{M}", f"SigASixFac{M}"
        rows.append(f"{lab} & {v(n + 'RateA')} & {v(n + 'RateB')} & {ci(n)} & {v(n + 'Pholm')} & "
                    f"{v(f + 'RateA')} & {v(f + 'RateB')} & {ci(f)} & {v(f + 'Pholm')} & {v('RobustASixFac' + M)} \\\\")
    header = ("& \\multicolumn{4}{c}{Numeric, $\\pm$0.1 mm (\\%), " + v("SigASixNumLlamaN") + " rows} & "
              "\\multicolumn{5}{c}{Factual chrF++, " + v("SigASixFacLlamaN") + " paired facts} \\\\\n"
              "\\cmidrule(lr){2-5}\\cmidrule(lr){6-10}\n"
              "Model & bn & en & en $-$ bn [95\\% CI] & $p_\\text{Holm}$ & bn & en & en $-$ bn [95\\% CI] & $p_\\text{Holm}$ & all seeds \\\\")
    table("t09_language", "Bengali versus English, fine-tuned models, seed 42 (test family A6). Numeric: share of the "
          f"{v('NumericAll')} numeric rows correct within $\\pm$0.1 mm, half in each language, compared with an unpaired Fisher "
          "exact test. Factual: chrF++ on facts asked in both languages, compared with a Wilcoxon signed-rank test; the "
          f"difference is in chrF++ points. CIs are {v('PlanBootstrap')}-resample bootstrap intervals. ``All seeds'' says whether the "
          "factual difference is significant after Holm correction at seeds 42, 1337 and 2024.",
          "tab:language", "lrrlrrrlrl", header, "\n".join(rows), size="\\scriptsize")


# --------------------------------------------------------------------------- T10
def t10() -> None:
    rows = []
    for M, lab in M4:
        rows.append(f"{lab} & {v(M + 'FTClarifyExactSeedA')} & {v(M + 'CLClarifyExact')} & {ci('SigAThreeA' + M)} & "
                    f"{v(M + 'FTSeedAClarifyLearned')} & {v(M + 'CLClarifyLearned')} & "
                    f"{v(M + 'FTInventedSeedA')} & {v(M + 'CLInvented')} & {ci('SigAThreeB' + M)} \\\\")
    header = ("& \\multicolumn{3}{c}{Clarify exact (primary)} & \\multicolumn{2}{c}{Clarify learned} & "
              "\\multicolumn{3}{c}{Invented value} \\\\\n"
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}\n"
              "Model & FT & CL & CL $-$ FT [95\\% CI] & FT & CL & FT & CL & CL $-$ FT [95\\% CI] \\\\")
    table("t10_clarify", f"Clarify training, per model: seed 42 fine-tuned (FT) against seed 42 clarify-trained (CL), "
          f"{v('TestClarify')} clarify rows, values in per cent and differences in percentage points. Clarify exact "
          "(asks and names the missing parameter) is the primary measure; clarify learned (any request that withholds "
          "the result) is secondary and untested. Every difference in both tested families (A3a, A3b) has "
          f"$p_\\text{{Holm}}$ {v('SigAThreeALlamaPholm')}. Clarify-trained models exist at seed 42 only.",
          "tab:clarify", "lrrlrrrrl", header, "\n".join(rows), size="\\scriptsize")


# --------------------------------------------------------------------------- T11
def t11() -> None:
    body = "\n".join([
        f"Questions & {v('HumQuestions')}: {v('HumInTest')} from the locked test set, {v('HumAuthored')} authored Bengali translations \\\\",
        f"Question types & numeric {v('HumQNumeric')}, factual {v('HumQFactual')}, clarify {v('HumQClarify')}, abstain {v('HumQAbstain')} \\\\",
        f"Arms & {v('HumArms')}: Llama zero-shot, Llama fine-tuned, Llama clarify-trained, Mistral fine-tuned (seed 42) \\\\",
        f"Outputs rated & {v('HumUniqueOutputs')} (every question once per arm) \\\\",
        f"Annotators & {v('HumAnnotators')}, {v('HumPerAnnotator')} ratings each, {v('HumJudgments')} in total \\\\",
        f"Shared outputs & {v('HumOverlapOutputs')}, rated by all annotators ({v('HumOverlapRatings')} ratings) \\\\",
        "Blinding & arm identity hidden behind letter codes; gold answer shown \\\\",
        "Scales & rating 1--5; would give to a farmer (yes/no) \\\\",
    ])
    table("t11_human_design", "Human evaluation design (Stage 7). The authored translations were generated with the same "
          "adapters and decoding as the test-set outputs.", "tab:human-design", "l>{\\raggedright\\arraybackslash}p{0.62\\textwidth}",
          "Element & Value \\\\", body)


# --------------------------------------------------------------------------- T12
def t12() -> None:
    a = lambda s: f"{v('Agr' + s)} & [{v('Agr' + s + 'Lo')}, {v('Agr' + s + 'Hi')}]"
    body = "\n".join([
        f"Would give to a farmer & Fleiss' $\\kappa$ (primary) & {a('KappaFarmer')} \\\\",
        f"Would give to a farmer & Krippendorff's $\\alpha$, nominal & {a('AlphaFarmer')} \\\\",
        f"Rating 1--5 & Krippendorff's $\\alpha$, ordinal (primary) & {a('AlphaOrdinal')} \\\\",
        f"Rating 1--5 & Krippendorff's $\\alpha$, interval & {a('AlphaInterval')} \\\\",
        f"Rating, three bands & Fleiss' $\\kappa$ & {a('KappaBands')} \\\\",
        f"Rating 1--5 & Fleiss' $\\kappa$, nominal & {a('KappaNominal')} \\\\",
        "\\addlinespace",
        f"Rating, leave one out & ordinal $\\alpha$ without A1 / A2 / A3 / A4 / A5 & \\multicolumn{{2}}{{l}}{{{v('AgreeAlphaWithoutOneRecomputed')} / "
        f"{v('AgreeAlphaWithoutTwoRecomputed')} / {v('AgreeAlphaWithoutThreeRecomputed')} / {v('AgreeAlphaWithoutFourRecomputed')} / "
        f"{v('AgreeAlphaWithoutFiveRecomputed')}}} \\\\",
    ])
    table("t12_agreement", f"Agreement on the {v('HumOverlapOutputs')} outputs rated by all five annotators, with 95\\% "
          f"bootstrap CIs over items. The two primary measures were named before significance testing. Nominal $\\kappa$ "
          "treats adjacent ratings (four versus five) as full disagreement and so understates agreement on an ordinal scale; it is "
          "reported for completeness. Leave-one-out values were recomputed and reproduce the audit report's all-annotator "
          "and without-A1 values.",
          "tab:agreement", "llrl", "Measure & Statistic & Value & 95\\% CI \\\\", body, size="\\footnotesize")


# --------------------------------------------------------------------------- T13
def t13() -> None:
    rows = []
    c = lambda s: f"{v(s)} [{v(s + 'Lo')}, {v(s + 'Hi')}]"
    for A, lab in (("ZS", "Llama zero-shot"), ("FT", "Llama fine-tuned"), ("CL", "Llama clarify-trained"),
                   ("Mis", "Mistral fine-tuned")):
        rows.append(f"{lab} & {c('HumTest' + A + 'Rating')} & {c('HumTest' + A + 'Yes')} & {c('HumTest' + A + 'Good')} & "
                    f"{v('HumTest' + A + 'ClarifyYes')} & {v('HumAll' + A + 'Yes')} \\\\")
    table("t13_human_results", f"Human results per arm on the primary sample of {v('HumTestFTN')} test-set questions, with "
          f"95\\% bootstrap CIs. A shared output's score is the mean of its five ratings and its farmer decision is the "
          f"majority vote. ``Acceptable'' means a mean rating of at least four. The clarify column covers the "
          f"{v('HumTestFTClarifyN')} clarify questions in the primary sample. The last column adds the {v('HumAuthored')} "
          "authored Bengali translations (sensitivity sample). Raw means do not adjust for which annotator rated which output; "
          "the annotator-adjusted models (B4) leave every contrast's direction and significance unchanged.",
          "tab:human-results", "lllllr",
          f"Arm & Mean rating (1--5) & Farmer yes (\\%) & Acceptable (\\%) & Clarify yes (\\%) & All {v('HumQuestions')}: yes (\\%) \\\\",
          "\n".join(rows), size="\\scriptsize")


# --------------------------------------------------------------------------- T14
def t14() -> None:
    body = "\n".join([
        f"H1 & Fine-tuning beats zero-shot & A1, B2-i, B3-i & Supported; robust on all three seeds & "
        f"A1: {v('RangeSigAOneDiffMin')} to {v('RangeSigAOneDiffMax')} pp, every $p_\\text{{Holm}}$ {v('SigAOneLlamaPholm')}; "
        f"farmer yes {ci('SigBThreeFtVsZsOverall')} pp \\\\",
        f"H2 & Bengali trails English & A6, B5 & Partially supported & factual: significant in {v('SigASixFacSigSeedA')} of four models "
        f"at seed 42, {v('RobustASixFacYes')} of four on all seeds; numeric: {v('SigASixNumSigAnySeed')} of "
        f"{v('SigASixNumTestsAllSeeds')} tests; human: {v('SigBFiveSig')} of {v('SigBFiveTests')} \\\\",
        f"H3 & Annotators agree & B6 & Supported on the primary measures & farmer $\\kappa$ {v('AgrKappaFarmer')} "
        f"[{v('AgrKappaFarmerLo')}, {v('AgrKappaFarmerHi')}]; ordinal $\\alpha$ {v('AgrAlphaOrdinal')} "
        f"[{v('AgrAlphaOrdinalLo')}, {v('AgrAlphaOrdinalHi')}] \\\\",
        f"T & Fine-tuning turns invented values into confident errors & A7 & Supported in all four models on all seeds; "
        f"size depends on the base model & {v('RangeSigASevenDiffMin')} to {v('RangeSigASevenDiffMax')} pp; fine-tuned strict "
        f"rate {v('RangeFTSeedAConfErrBMin')}--{v('RangeFTSeedAConfErrBMax')}\\% \\\\",
        f"C1 & Clarify training teaches clarification & A3a, B3-ii & Supported & +{v('RangeSigAThreeADiffMin')} to "
        f"+{v('RangeSigAThreeADiffMax')} pp; human clarify cell {ci('SigBThreeClVsFtClarify')} pp \\\\",
        f"C2 & Clarify training reduces invented values & A3b & Supported & $-${v('RangeSigAThreeBDiffMin')} to "
        f"$-${v('RangeSigAThreeBDiffMax')} pp \\\\",
        f"C3 & Clarify training does not harm numeric accuracy & A4 & No loss detected & {v('SigAFourSig')} of four significant; "
        f"lowest CI bound {v('SigAFourWorstLo')} pp \\\\",
        f"D & A detector adds safety & A5 & Descriptive only: oracle-gated upper bound & {ci('SigAFiveDetector')} pp, no test \\\\",
        f"M & Llama FT differs from Mistral FT & A2, B2-iii, B3-iii, B4 & No significant difference detected & "
        f"{ci('SigATwoLlamaVsMistral')} pp; farmer yes {ci('SigBThreeMisVsFtOverall')} pp \\\\",
    ])
    table("t14_verdicts", "Hypothesis verdicts under the analysis plan fixed before significance testing (plan v3). "
          "Automatic tests use seed 42 with Holm correction within each family; human tests use the primary sample of "
          f"{v('HumInTest')} test-set questions. pp = percentage points; intervals are 95\\% bootstrap CIs.",
          "tab:verdicts", "l>{\\raggedright\\arraybackslash}p{0.2\\textwidth}>{\\raggedright\\arraybackslash}p{0.12\\textwidth}>{\\raggedright\\arraybackslash}p{0.2\\textwidth}>{\\raggedright\\arraybackslash}p{0.3\\textwidth}",
          "ID & Claim & Tests & Verdict & Key numbers \\\\", body, size="\\scriptsize")


# --------------------------------------------------------------------------- T15
def t15() -> None:
    Y, N_ = "yes", "--"
    rows = [
        ("QuestBench~\\cite{li2025questbench}", "logic, planning, maths", "en", N_, "yes (one missing variable)", N_),
        ("AbstentionBench~\\cite{kirichenko2025abstentionbench}", "general QA", "en", "yes (reasoning-tuned vs base)", "yes (underspecified)", N_),
        ("Missing premise~\\cite{fan2025missing}", "maths", "en", "yes (reasoning-tuned vs base)", "yes (missing premise)", N_),
        ("Gekhman et al.~\\cite{gekhman2024finetuning}", "closed-book QA", "en", Y, "no (unknown facts)", N_),
        ("Kang et al.~\\cite{kang2025unfamiliar}", "QA, biographies", "en", Y, "no (unfamiliar queries)", N_),
        ("R-Tuning~\\cite{zhang2024rtuning}", "QA", "en", Y, "no (unknown facts)", N_),
        ("CLAM~\\cite{kuhn2022clam}", "ambiguous QA", "en", "no (prompting)", "yes (ambiguity)", N_),
        ("STaR-GATE~\\cite{andukuri2024stargate}", "personalised tasks", "en", Y, "yes (unstated preferences)", N_),
        ("ClarifyCoder~\\cite{wu2025clarifycoder}", "code generation", "en", Y, "yes (ambiguous requirements)", N_),
        ("Balaguer et al.~\\cite{balaguer2024rag}", "agriculture", "en", "yes (vs RAG)", N_, N_),
        ("Farmer.Chat~\\cite{singh2024farmerchat}", "agriculture", "multilingual", N_, N_, "yes (deployment)"),
        ("KrishokBondhu~\\cite{krishokbondhu2025}", "agriculture", "bn", "no (RAG)", N_, N_),
        ("KrishokChat~\\cite{krishokchat2026}", "agriculture", "bn", "yes (one model)", N_, N_),
        ("This thesis", "irrigation advisory", "bn, en", "yes (four models $\\times$ three seeds)", "yes (designed clarify probe)", "yes (five annotators)"),
    ]
    body = "\n".join(" & ".join(r) + " \\\\" + ("\n\\midrule" if i == len(rows) - 2 else "") for i, r in enumerate(rows))
    table("t15_related", "Related work compared on the dimensions this thesis studies. Entries record only what each work's "
          "abstract states; ``--'' means the abstract does not state it, not that the work lacks it. ``Missing-input probe'' "
          "means the evaluation includes questions that cannot be answered without information the user did not give.",
          "tab:related", ">{\\raggedright\\arraybackslash}p{0.2\\textwidth}>{\\raggedright\\arraybackslash}p{0.13\\textwidth}"
          "l>{\\raggedright\\arraybackslash}p{0.15\\textwidth}>{\\raggedright\\arraybackslash}p{0.17\\textwidth}"
          ">{\\raggedright\\arraybackslash}p{0.12\\textwidth}",
          "Work & Domain & Language & Fine-tuning studied & Missing-input probe & Human evaluation \\\\", body,
          size="\\scriptsize")


# --------------------------------------------------------------------------- validation
LAYOUT_PATTERNS = [
    r"\\cite\{[^}]*\}",
    r"p\{[0-9.]+\\textwidth\}", r"\\cmidrule\([a-z]*\)\{\d+-\d+\}", r"\\multicolumn\{\d+\}", r"\\multirow\{\d+\}",
    r"\\(vspace|hspace)\{[0-9.]+pt\}", r"\\parbox\{[0-9.]+\\textwidth\}",
    r"Stage \d[A-E]?", r"\(S\d[A-E]?\)", r"seeds? 42(, 1337,? (and )?2024)?", r"42, 1337, 2024", r"models: 42 only",
    r"[AB]\d[ab]?(/[AB]?\d[ab]?)*", r"\d--\d",
]


def validate(name: str, text: str) -> list[str]:
    problems = []
    for cs in re.findall(r"\\([A-Za-z]+)", text):
        if cs in LATEX_CMDS or cs in {"quad", "addlinespace", "alpha", "kappa", "parbox"}:
            continue
        if cs not in N:
            problems.append(f"{name}: unknown control sequence \\{cs}")
    stripped = text
    for pat in LAYOUT_PATTERNS:
        stripped = re.sub(pat, " ", stripped)
    for tok in sorted(ALLOWED_DIGIT_TOKENS, key=len, reverse=True):
        stripped = stripped.replace(tok, " ")
    stripped = re.sub(r"\\[A-Za-z]+", " ", stripped)
    for m in re.finditer(r"\d", stripped):
        ctx = stripped[max(0, m.start() - 25): m.end() + 25].replace("\n", " ")
        problems.append(f"{name}: bare digit in '{ctx}'")
    return problems


def main() -> None:
    for f in (t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15):
        f()
    OUT.mkdir(parents=True, exist_ok=True)
    problems = []
    for key, text in TABLES.items():
        (OUT / f"{key}.tex").write_text("% AUTO-GENERATED by scripts/thesis/tables.py\n" + text, encoding="utf-8", newline="\n")
        problems += validate(key, text)
    print(f"{len(TABLES)} tables -> {OUT.relative_to(ROOT).as_posix()}")
    for p in problems:
        print("  PROBLEM", p)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
