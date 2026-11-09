#!/usr/bin/env python3
"""
make_numbers.py — the single source of every number in the thesis.

Reads repo files (read-only) and writes:
  thesis/generated/numbers.tex   LaTeX macros, one per number, each with a
                                 comment naming its source file and field
  thesis/generated/numbers.json  the same values, for figures.py and checks
  thesis/generated/facts.md      human-readable table of every macro

Rules enforced here:
  * The locked test set SHA-256 is verified before any prediction is read.
  * No training file is opened. Training-corpus composition comes from the
    Stage 2 / 2b / 6B gate reports.
  * Stage 8 was released on 2026-09-24 (reports/stats/significance.md, plan v3).
    Its outputs are read in section 13 below; before release they were excluded
    (the exclusion is recorded in BUILD_REPORT.md).
  * A number that cannot be sourced becomes \\MISSING{what}{where looked}.
  * Output is deterministic: no timestamps, fixed ordering, fixed rounding.

Seed naming in macro names (LaTeX macro names cannot contain digits):
  SeedA = 42, SeedB = 1337, SeedC = 2024.
Model naming: Llama = llama31_8b, Mistral = mistral_nemo,
  PhiThree = phi35_mini, PhiFour = phi4_mini.
Variant naming: ZS = zero-shot, FT = fine-tuned (mean of 3 seeds unless a
  Seed suffix is present), CL = clarify-trained (seed 42 only).
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis" / "generated"
sys.path.insert(0, str(ROOT / "scripts"))

MODELS = [("llama31_8b", "Llama", "Llama-3.1-8B"),
          ("mistral_nemo", "Mistral", "Mistral-NeMo-12B"),
          ("phi35_mini", "PhiThree", "Phi-3.5-mini"),
          ("phi4_mini", "PhiFour", "Phi-4-mini")]
SEEDS = [(42, "SeedA"), (1337, "SeedB"), (2024, "SeedC")]

# Files that must never be read by this script.
FORBIDDEN = {"train_augmented.jsonl", "train_augmented_clarify.jsonl", "train.jsonl",
             "train_clarify.jsonl", "train_seed.jsonl", "val.jsonl",
             "s6b_clarify_train.jsonl"}

MACROS: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
SECTION = ["general"]
EVAL_SUMMARY: list[tuple] = []


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def src(path: str) -> Path:
    p = ROOT / path
    if p.name in FORBIDDEN:
        raise SystemExit(f"ABORT: make_numbers.py must not read {path}")
    return p


def load_json(path: str):
    return json.loads(src(path).read_text(encoding="utf-8"))


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(src(path))


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in src(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def section(name: str) -> None:
    SECTION[0] = name


def fmt(v, nd: int | None) -> str:
    if nd is None:
        return str(v)
    if isinstance(v, str):
        return v
    if nd == 0:
        return f"{int(round(float(v))):d}"
    return f"{float(v):.{nd}f}"


def put(name: str, value, source: str, field: str, nd: int | None = None,
        unit: str = "") -> None:
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"illegal macro name {name!r}")
    if name in MACROS:
        raise ValueError(f"duplicate macro {name}")
    text = fmt(value, nd)
    if text in ("nan", "None"):
        raise ValueError(f"{name}: value is {text}")
    MACROS[name] = {"value": text, "raw": value if not isinstance(value, float) else round(value, 10),
                    "source": source, "field": field, "unit": unit,
                    "section": SECTION[0], "missing": False}


def missing(name: str, what: str, where: str) -> None:
    if name in MACROS:
        raise ValueError(f"duplicate macro {name}")
    MACROS[name] = {"value": f"\\MISSING{{{what}}}{{{where}}}", "raw": None,
                    "source": "(none)", "field": where, "unit": "",
                    "section": SECTION[0], "missing": True, "what": what}


def pct(n: int, d: int) -> float:
    return 100.0 * n / d


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grab(text: str, pattern: str, what: str, flags=0) -> re.Match:
    m = re.search(pattern, text, flags)
    if not m:
        raise SystemExit(f"pattern for {what!r} not found: {pattern}")
    return m


WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10}


# ===========================================================================
# 0. Locked test set
# ===========================================================================
def test_set_gate() -> None:
    section("test set")
    f = src("data/hardened/test_hardened.jsonl")
    want = src("data/hardened/test_hardened.sha256").read_text(encoding="utf-8").split()[0]
    got = sha256(f)
    if got != want:
        raise SystemExit(f"ABORT: test_hardened.jsonl SHA-256 {got} != locked {want}")
    put("TestShaShort", got[:12], "data/hardened/test_hardened.sha256", "first 12 hex chars (verified)")
    f2 = src("data/splits/test.jsonl")
    want2 = src("data/splits/test.sha256").read_text(encoding="utf-8").split()[0]
    got2 = sha256(f2)
    if got2 != want2:
        raise SystemExit("ABORT: data/splits/test.jsonl SHA-256 mismatch")
    put("SeedTestShaShort", got2[:12], "data/splits/test.sha256", "first 12 hex chars (verified)")


# ===========================================================================
# 1. Data: digitisation, split, augmentation, render
# ===========================================================================
def tab_rows_check(split: str) -> int:
    return {r["split"]: r["rows"] for r in load_json("reports/gate1_split.json")["table"]}[split]


def data_numbers() -> None:
    section("data")
    g0 = load_json("reports/gate0_seed_corpus.json")
    s = "reports/gate0_seed_corpus.json"
    put("SourceFiles", g0["source_files"], s, "source_files", 0)
    put("SeedRows", g0["total_rows"], s, "total_rows", 0)
    put("SeedRowsBn", g0["rows_per_lang"]["bn"], s, "rows_per_lang.bn", 0)
    put("SeedRowsEn", g0["rows_per_lang"]["en"], s, "rows_per_lang.en", 0)
    put("SeedFacts", g0["distinct_fact_id"], s, "distinct_fact_id", 0)
    put("SeedSourceIds", g0["distinct_source_id"], s, "distinct_source_id", 0)
    put("SeedCrops", g0["distinct_crop"], s, "distinct_crop", 0)
    put("SeedScenarios", g0["distinct_scenario_hash"], s, "distinct_scenario_hash", 0)
    put("SeedNumeric", g0["rows_per_qtype"]["numeric_runtime"], s, "rows_per_qtype.numeric_runtime", 0)
    put("SeedFactual", g0["rows_per_qtype"]["factual"], s, "rows_per_qtype.factual", 0)
    put("SeedAbstain", g0["rows_per_qtype"]["abstain"], s, "rows_per_qtype.abstain", 0)
    put("SeedSkippedFieldTypes", len(g0["skipped_fields_by_design"]), s, "len(skipped_fields_by_design)", 0)
    put("SeedSkippedInstances", g0["skipped_field_instances_count"], s, "skipped_field_instances_count", 0)

    # BARC calendar images (counts only). Folder lies outside v3.
    img_dir = ROOT.parent / "crop_calender_images"
    if img_dir.is_dir():
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        put("CalendarImageFiles", len(imgs), rel(img_dir), "count of .jpg/.png files", 0)
        by_hash = collections.defaultdict(list)
        for p in imgs:
            by_hash[sha256(p)].append(p)
        put("CalendarImageUnique", len(by_hash), rel(img_dir), "distinct SHA-256 among image files", 0)
        put("CalendarImageDuplicates", len(imgs) - len(by_hash), rel(img_dir), "files minus distinct SHA-256", 0)
        # A named (non-screenshot) image whose file name matches none of the study crops.
        import difflib
        crops = [c.lower().replace(" ", "") for c in load_json("reports/gate1_split.json")["crops_per_split"]["train"]]
        outside = 0
        for ps in by_hash.values():
            stem = re.sub(r"[^a-z]", "", ps[0].stem.lower())
            if stem.startswith("screenshot"):
                continue
            best = max(difflib.SequenceMatcher(None, stem, c).ratio() for c in crops)
            outside += best < 0.8
        put("CalendarImageOutsideStudy", outside, rel(img_dir),
            "distinct named images whose file name matches no study crop (SequenceMatcher < 0.8)", 0)
    else:
        missing("CalendarImageFiles", "BARC calendar image count", "../crop_calender_images")
    json_dir = ROOT.parent / "Json_Dataset(Calander)"
    if json_dir.is_dir():
        put("CalendarJsonFiles", len(list(json_dir.glob("*.json"))), rel(json_dir), "count of *.json", 0)

    g1 = load_json("reports/gate1_split.json")
    s = "reports/gate1_split.json"
    put("SplitGroups", g1["input_groups"], s, "input_groups", 0)
    put("SplitRatioTrain", g1["ratios"]["train"] * 100, s, "ratios.train x100", 0, "%")
    put("SplitRatioVal", g1["ratios"]["val"] * 100, s, "ratios.val x100", 0, "%")
    put("SplitRatioTest", g1["ratios"]["test"] * 100, s, "ratios.test x100", 0, "%")
    tab = {r["split"]: r for r in g1["table"]}
    put("TrainSeedRows", tab["train_seed"]["rows"], s, "table[train_seed].rows", 0)
    put("ValRows", tab["val"]["rows"], s, "table[val].rows", 0)
    put("TestSeedRows", tab["test"]["rows"], s, "table[test].rows", 0)
    put("ValPctFactual", tab["val"]["pct_factual"], s, "table[val].pct_factual", 1, "%")
    put("ValPctNumeric", tab["val"]["pct_numeric"], s, "table[val].pct_numeric", 1, "%")
    for k in ("source_id", "fact_id", "scenario_hash"):
        tot = sum(g1["overlap_counts"][k].values())
        name = {"source_id": "SplitOverlapSource", "fact_id": "SplitOverlapFact",
                "scenario_hash": "SplitOverlapScenario"}[k]
        put(name, tot, s, f"sum(overlap_counts.{k})", 0)
    put("PreflightLexTestPctAbove", g1["preflight_lexical_summary"]["test"]["pct_at_or_above_threshold"],
        s, "preflight_lexical_summary.test.pct_at_or_above_threshold", 1, "%")

    # Per-split counts by question type (split metadata only; no row content is read).
    sa = load_csv("data/splits/split_assignment.csv")
    s = "data/splits/split_assignment.csv"
    QN = {"numeric_runtime": "Numeric", "factual": "Factual", "abstain": "Abstain"}
    SN = {"train": "TrainSeed", "val": "Val", "test": "TestSeed"}
    for sp, S in SN.items():
        for q, Q in QN.items():
            put(f"Split{S}{Q}", int(sa[(sa["split"] == sp) & (sa["qtype"] == q)]["n_rows"].sum()), s,
                f"sum(n_rows) where split={sp}, qtype={q}", 0)
        put(f"Split{S}Rows", int(sa[sa["split"] == sp]["n_rows"].sum()), s, f"sum(n_rows) where split={sp}", 0)
        put(f"Split{S}Groups", int((sa["split"] == sp).sum()), s, f"source_ids in split {sp}", 0)
    assert int(sa[sa["split"] == "train"]["n_rows"].sum()) == tab_rows_check("train_seed")

    g2 = load_json("reports/gate2_augment.json")
    s = "reports/gate2_augment.json"
    put("AugFamiliesRun", len(g2["families_run"]), s, "len(families_run) ('CD')", 0)
    s2 = src("scripts/s2_augment_train.py").read_text(encoding="utf-8")
    put("AugFamiliesPlanned", WORDNUM[grab(s2, r"\n(\w+) FAMILIES\n", "families").group(1).lower()],
        "scripts/s2_augment_train.py", "docstring 'FOUR FAMILIES'", 0)
    put("AugFamiliesNotRun", WORDNUM[grab(s2, r"\n(\w+) FAMILIES\n", "families").group(1).lower()] - len(g2["families_run"]),
        "scripts/s2_augment_train.py + reports/gate2_augment.json", "planned families minus families_run", 0)
    put("AugGeminiSkipped", int(bool(g2["gemini"]["skipped"])), s, "gemini.skipped (1 = yes)", 0)
    dbc = g2["dropped_by_check"]
    put("AugDropBalance", dbc.get("balance:noise", 0), s, "dropped_by_check['balance:noise']", 0)
    put("AugDropDuplicate", dbc.get("noise:duplicate", 0), s, "dropped_by_check['noise:duplicate']", 0)
    put("AugDropNoTarget", dbc.get("noise:no_target_token", 0), s, "dropped_by_check['noise:no_target_token']", 0)
    put("AugNoiseMade", sum(v["made"] for v in g2["noise_by_qtype"].values()), s, "sum(noise_by_qtype[*].made)", 0)
    put("AugSynthNumericPct", g2["table_by_origin"][1]["pct_numeric"], s, "table_by_origin[synthetic].pct_numeric", 1, "%")
    put("AugZeroQuotaPct", g2["zero_gold"]["quota_target_pct"], s, "zero_gold.quota_target_pct", 1, "%")
    put("AugSeedZeroGold", g2["zero_gold"]["seed_numeric_zero_gold"], s, "zero_gold.seed_numeric_zero_gold", 0)
    put("AugInputRows", g2["input_rows"], s, "input_rows", 0)
    put("AugRows", g2["output_rows"], s, "output_rows", 0)
    orig = {r["origin"]: r["rows"] for r in g2["table_by_origin"]}
    put("AugRowsSeed", orig["seed"], s, "table_by_origin[seed].rows", 0)
    put("AugRowsSynthetic", orig["synthetic"], s, "table_by_origin[synthetic].rows (family C)", 0)
    put("AugRowsNoise", orig["noise"], s, "table_by_origin[noise].rows (family D)", 0)
    q = g2["rows_by_qtype"]
    put("TrainNumeric", q["numeric_runtime"], s, "rows_by_qtype.numeric_runtime", 0)
    put("TrainFactual", q["factual"], s, "rows_by_qtype.factual", 0)
    put("TrainAbstain", q["abstain"], s, "rows_by_qtype.abstain", 0)
    put("TrainClarify", q.get("clarify", 0), s, "rows_by_qtype.clarify (absent key = 0)", 0)
    tot = g2["output_rows"]
    put("TrainPctNumeric", pct(q["numeric_runtime"], tot), s, "numeric_runtime / output_rows", 1, "%")
    put("TrainPctFactual", pct(q["factual"], tot), s, "factual / output_rows", 1, "%")
    put("TrainPctAbstain", pct(q["abstain"], tot), s, "abstain / output_rows", 1, "%")
    put("TrainPctClarify", pct(q.get("clarify", 0), tot), s, "clarify / output_rows", 1, "%")
    put("TrainRowsBn", g2["rows_by_lang"]["bn"], s, "rows_by_lang.bn", 0)
    put("TrainRowsEn", g2["rows_by_lang"]["en"], s, "rows_by_lang.en", 0)
    put("AugSynthZeroGoldPct", g2["zero_gold"]["synthetic_zero_gold_pct"], s, "zero_gold.synthetic_zero_gold_pct", 1, "%")
    put("AugSynthZeroGold", g2["zero_gold"]["synthetic_zero_gold"], s, "zero_gold.synthetic_zero_gold", 0)
    put("AugPearsonR", g2["pearson_et0_rainfall"]["r"], s, "pearson_et0_rainfall.r", 3)
    put("AugPearsonTol", g2["pearson_tolerance"], s, "pearson_tolerance", 2)
    put("AugHeldoutFactRows", g2["synthetic_crop_stage_exposure"]["rows_whose_fact_is_not_in_train_seed"],
        s, "synthetic_crop_stage_exposure.rows_whose_fact_is_not_in_train_seed", 0)
    put("AugHeldoutFactPct", g2["synthetic_crop_stage_exposure"]["pct"], s, "synthetic_crop_stage_exposure.pct", 1, "%")
    put("AugDropped", g2["dropped_total"], s, "dropped_total", 0)
    put("AugNoiseNumeric", g2["noise_by_qtype"]["numeric_runtime"]["made"], s, "noise_by_qtype.numeric_runtime.made", 0)
    put("AugNoiseFactual", g2["noise_by_qtype"]["factual"]["made"], s, "noise_by_qtype.factual.made", 0)
    put("AugNoiseAbstain", g2["noise_by_qtype"]["abstain"]["made"], s, "noise_by_qtype.abstain.made", 0)
    put("AugBlocklistHashes", g2["blocklist"]["distinct_hashes"], s, "blocklist.distinct_hashes", 0)

    g2b = load_json("reports/gate2b_render.json")
    put("RenderTrainRows", g2b["splits"]["train"]["rows"], "reports/gate2b_render.json", "splits.train.rows", 0)
    put("RenderValRows", g2b["splits"]["val"]["rows"], "reports/gate2b_render.json", "splits.val.rows", 0)

    gb = load_json("reports/s6b/generate_report.json")
    s = "reports/s6b/generate_report.json"
    put("ClarifyTrainRows", gb["rows_generated"], s, "rows_generated", 0)
    put("ClarifyTrainBn", gb["language_split"]["bn"], s, "language_split.bn", 0)
    put("ClarifyTrainEn", gb["language_split"]["en"], s, "language_split.en", 0)
    put("ClarifyTrainSlotRain", gb["slot_distribution"]["rainfall_mm"], s, "slot_distribution.rainfall_mm", 0)
    put("ClarifyTrainSlotArea", gb["slot_distribution"]["area_value"], s, "slot_distribution.area_value", 0)
    put("ClarifyTrainSlotStage", gb["slot_distribution"]["stage"], s, "slot_distribution.stage", 0)
    put("ClarifyTrainSlotEt", gb["slot_distribution"].get("et0_mm", 0), s, "slot_distribution.et0_mm (absent = 0)", 0)
    gm = load_json("reports/s6b/merge_report.json")
    put("ClarifyCorpusRows", gm["total"], "reports/s6b/merge_report.json", "total", 0)
    put("ClarifyCorpusPctClarify", pct(gm["qtype_distribution"]["clarify"], gm["total"]),
        "reports/s6b/merge_report.json", "qtype_distribution.clarify / total", 1, "%")


# ===========================================================================
# 2. Leakage audit
# ===========================================================================
def leakage_numbers() -> None:
    section("leakage")
    L = load_json("reports/leakage/leakage_report.json")
    s = "reports/leakage/leakage_report.json"
    put("LeakExactCount", L["detector_1_exact"]["count"], s, "detector_1_exact.count", 0)
    d2 = L["detector_2_jaccard"]
    put("LeakJaccardThreshold", d2["threshold"], s, "detector_2_jaccard.threshold", 2)
    put("LeakJaccardPairs", d2["n_pairs_at_or_above_threshold"], s, "detector_2_jaccard.n_pairs_at_or_above_threshold", 0)
    put("LeakJaccardHitsOriginal", d2["fail_rule_hits_original"], s, "detector_2_jaccard.fail_rule_hits_original", 0)
    put("LeakJaccardHitsRefined", d2["fail_rule_hits"], s, "detector_2_jaccard.fail_rule_hits (F3)", 0)
    put("LeakJaccardAbstainExcluded", d2["gold_only_excluded"], s, "detector_2_jaccard.gold_only_excluded", 0)
    d3 = L["detector_3_semantic"]
    put("LeakSemThreshold", d3["threshold"], s, "detector_3_semantic.threshold", 2)
    put("LeakSemCandidates", d3["n_candidates_at_or_above_threshold"], s, "detector_3_semantic.n_candidates_at_or_above_threshold", 0)
    put("LeakSemPending", d3["n_pending_human_review"], s, "detector_3_semantic.n_pending_human_review", 0)
    put("LeakSemMarkedLeak", d3["n_human_marked_leakage"], s, "detector_3_semantic.n_human_marked_leakage", 0)
    put("LeakSemMarkedLegit", d3["n_human_marked_legitimate"], s, "detector_3_semantic.n_human_marked_legitimate", 0)
    dec = load_json("reports/leakage/semantic_human_decisions.json")
    put("LeakSemDecisionsRecorded", len(dec.get("decisions", {})), "reports/leakage/semantic_human_decisions.json", "len(decisions)", 0)
    frame = {r["pair_key"] for r in d3["candidates"] + d3["pending_review"]}
    put("LeakReviewFrame", len(frame), s, "distinct pair_key in detector_3_semantic.candidates + pending_review", 0)
    rv = "reports/leakage/review/semantic_review_sample.csv"
    if (ROOT / rv).exists():
        sheet = pd.read_csv(ROOT / rv, encoding="utf-8-sig", dtype=str).fillna("")
        put("LeakReviewSample", len(sheet), rv, "rows", 0)
        both = sheet[(sheet["reviewer_1_decision"] != "") & (sheet["reviewer_2_decision"] != "")]
        put("LeakReviewLabelled", len(both), rv, "rows labelled by both reviewers", 0)
    d4 = L["detector_4_scenario"]
    put("LeakScenarioHeldout", d4["heldout_hash_count"], s, "detector_4_scenario.heldout_hash_count", 0)
    put("LeakScenarioOverlap", d4["overlap_count"], s, "detector_4_scenario.overlap_count", 0)


# ===========================================================================
# 3. Test hardening
# ===========================================================================
CAT_MACRO = {"misspelled_crop": "Misspelled", "missing_parameter": "MissingParam",
             "unit_ambiguity": "UnitAmbig", "multi_crop": "MultiCrop",
             "conflicting_values": "Conflicting", "out_of_domain": "OutOfDomain",
             "code_switched": "CodeSwitched"}


def hardening_numbers() -> None:
    section("test hardening")
    g4 = load_json("reports/gate4_hardened.json")
    s = "reports/gate4_hardened.json"
    c = g4["composition"]
    put("TestRows", c["total_rows"], s, "composition.total_rows", 0)
    put("TestRowsSeed", c["seed_rows"], s, "composition.seed_rows", 0)
    put("TestRowsAdv", c["adversarial_rows"], s, "composition.adversarial_rows", 0)
    put("TestRowsBn", c["by_lang"]["bn"], s, "composition.by_lang.bn", 0)
    put("TestRowsEn", c["by_lang"]["en"], s, "composition.by_lang.en", 0)
    for q, m in (("numeric_runtime", "Numeric"), ("factual", "Factual"),
                 ("clarify", "Clarify"), ("abstain", "Abstain")):
        put(f"TestAll{m}", c["by_qtype"][q], s, f"composition.by_qtype.{q}", 0)
    put("TestPending", c["by_gold_status"]["pending_human"], s, "composition.by_gold_status.pending_human", 0)
    QM = {"numeric_runtime": "Numeric", "factual": "Factual", "clarify": "Clarify", "abstain": "Abstain"}
    for row in g4["adversarial_by_category"]:
        m = CAT_MACRO[row["category"]]
        cat = row["category"]
        put(f"Adv{m}", row["produced"], s, f"adversarial_by_category[{cat}].produced", 0)
        put(f"Adv{m}Target", row["target"], s, f"adversarial_by_category[{cat}].target", 0)
        put(f"Adv{m}Subtypes", len(row["subtypes"]), s, f"len(adversarial_by_category[{cat}].subtypes)", 0)
        for q, Q in QM.items():
            put(f"Adv{m}{Q}", row["qtypes"].get(q, 0), s, f"adversarial_by_category[{cat}].qtypes.{q}", 0)
        digit_word = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
        for sub, n in row["subtypes"].items():
            spelled = "".join(digit_word.get(ch, ch) for ch in sub)
            name = "".join(w[:1].upper() + w[1:] for w in re.sub(r"[^A-Za-z_]", "", spelled).split("_") if w)
            put(f"Adv{m}Sub{name}", n, s, f"adversarial_by_category[{cat}].subtypes.{sub}", 0)
    put("KcTableCrops", g4["inputs"]["kc_table"]["crops"], s, "inputs.kc_table.crops", 0)
    put("IrrigationRulesPresent", 0 if g4["inputs"]["irrigation_rules_csv"]["status"] == "ABSENT" else 1,
        s, "inputs.irrigation_rules_csv.status == ABSENT -> 0", 0)
    put("AdvScenarioCollisions", g4["leakage_guard"]["collisions_with_train_or_heldout"], s,
        "leakage_guard.collisions_with_train_or_heldout", 0)
    put("AdvRedrawnZeroNet", g4["leakage_guard"]["scenario_draw_stats"]["redrawn_zero_net"], s,
        "leakage_guard.scenario_draw_stats.redrawn_zero_net", 0)
    miss = next(r for r in g4["adversarial_by_category"] if r["category"] == "missing_parameter")
    for k, m in (("missing_et0", "Et"), ("missing_rainfall", "Rain"), ("missing_area", "Area"), ("missing_stage", "Stage")):
        put(f"AdvMissing{m}", miss["subtypes"][k], s, f"adversarial_by_category[missing_parameter].subtypes.{k}", 0)

    eg = load_json("reports/stats/entry_gate.json")
    s = "reports/stats/entry_gate.json"
    sc = eg["scorable_by_qtype"]
    scorable = sum(sc.values())
    put("TestScorable", scorable, s, "sum(scorable_by_qtype)", 0)
    put("TestExcluded", sum(eg["pending_by_category"].values()), s, "sum(pending_by_category)", 0)
    put("TestExcludedOOD", eg["pending_by_category"]["out_of_domain"], s, "pending_by_category.out_of_domain", 0)
    put("TestExcludedConflict", eg["pending_by_category"]["conflicting_values"], s, "pending_by_category.conflicting_values", 0)
    put("TestNumeric", sc["numeric_runtime"], s, "scorable_by_qtype.numeric_runtime", 0)
    put("TestFactual", sc["factual"], s, "scorable_by_qtype.factual", 0)
    put("TestClarify", sc["clarify"], s, "scorable_by_qtype.clarify", 0)
    put("TestAbstain", sc["abstain"], s, "scorable_by_qtype.abstain", 0)
    put("TestPctClarifyScorable", pct(sc["clarify"], scorable), s, "clarify / scorable", 1, "%")
    put("TestPctClarifyAll", pct(c["by_qtype"]["clarify"], c["total_rows"]),
        "reports/gate4_hardened.json", "composition.by_qtype.clarify / total_rows", 1, "%")
    put("TestPctAbstainScorable", pct(sc["abstain"], scorable), s, "abstain / scorable", 1, "%")
    put("TestPctNumericScorable", pct(sc["numeric_runtime"], scorable), s, "numeric_runtime / scorable", 1, "%")
    put("TestPctFactualScorable", pct(sc["factual"], scorable), s, "factual / scorable", 1, "%")
    put("TestClarifyPerLang", eg["scorable_by_qtype_lang"]["clarify/bn"], s, "scorable_by_qtype_lang.clarify/bn", 0)
    put("TestNumericPerLang", eg["scorable_by_qtype_lang"]["numeric_runtime/bn"], s, "scorable_by_qtype_lang.numeric_runtime/bn", 0)
    put("TestFactualPerLang", eg["scorable_by_qtype_lang"]["factual/bn"], s, "scorable_by_qtype_lang.factual/bn", 0)
    put("PairsFactual", eg["bn_en_pairing_key_options"]["fact_id"]["factual"], s, "bn_en_pairing_key_options.fact_id.factual", 0)
    put("PairsNumeric", eg["bn_en_pairing_key_options"]["fact_id"]["numeric_runtime"], s, "bn_en_pairing_key_options.fact_id.numeric_runtime", 0)

    # provenance recorded on the locked test rows (seed rows carry the Stage 0 provenance block)
    t = "data/hardened/test_hardened.jsonl"
    rows = load_jsonl(t)
    seed = [x for x in rows if x.get("origin") == "seed"]
    put("ProvSeedRows", len(seed), t, "rows with origin == seed", 0)
    put("ProvGeminiRows", sum("gemini_vision" in str((x.get("provenance") or {}).get("extraction_method", "")) for x in seed), t,
        "seed rows whose provenance.extraction_method names gemini_vision", 0)
    put("ProvHumanVerifiedRows", sum(bool((x.get("provenance") or {}).get("human_verified_upstream")) for x in seed), t,
        "seed rows with provenance.human_verified_upstream == true", 0)
    nets = [x["gold_trace"]["net_mm"] for x in rows if x["qtype"] == "numeric_runtime" and x.get("gold_status") != "pending_human"
            and isinstance(x.get("gold_trace"), dict) and "net_mm" in x["gold_trace"] and len(x.get("crops", [])) <= 1]
    nz = [v for v in nets if v > 0]
    put("GoldNetSingleN", len(nets), t, "single-crop numeric rows with a gold net_mm", 0)
    put("GoldNetMedianNonzero", statistics.median(nz), t, "median gold net_mm among single-crop rows with net_mm > 0", 2, "mm")
    put("GoldTolShareOfMedian", 100 * 0.1 / statistics.median(nz), t, "0.1 mm as a share of the median non-zero gold depth", 1, "%")

    sm = load_csv("reports/s6/summary_by_arm.csv").set_index("arm")
    r = sm.loc["llama31_8b_seed42"]
    put("NumericSingle", r["n_single"], "reports/s6/summary_by_arm.csv", "n_single (any arm)", 0)
    put("NumericMulti", r["n_multi"], "reports/s6/summary_by_arm.csv", "n_multi (any arm)", 0)
    put("ZeroGoldPct", r["zero_gold_frac"], "reports/s6/summary_by_arm.csv", "zero_gold_frac (single-crop numeric)", 1, "%")
    put("ZeroGoldN", round(r["zero_gold_frac"] / 100 * r["n_single"]), "reports/s6/summary_by_arm.csv",
        "zero_gold_frac x n_single", 0)


# ===========================================================================
# 4. Models and training
# ===========================================================================
def training_numbers() -> None:
    section("models and training")
    pf = load_json("reports/s5/preflight_models.json")
    put("GpuName", pf["gpu"], "reports/s5/preflight_models.json", "gpu")
    pre = {r: v for r, v in pf["results"].items()}
    seq = load_json("reports/eval/preflight_seqlen.json")
    put("MaxSeqLen", pf["max_seq_len"], "reports/s5/preflight_models.json", "max_seq_len", 0)
    put("SeqLenGlobalMax", seq["global_max_tokens"], "reports/eval/preflight_seqlen.json", "global_max_tokens", 0)
    wall_total = 0.0
    for key, M, _ in MODELS:
        mid = pre[key]["model_id"]
        st = seq["results"][mid]
        put(f"{M}TrainTokPfifty", st["p50"], "reports/eval/preflight_seqlen.json", f"results[{mid}].p50", 0)
        put(f"{M}TrainTokMax", st["max"], "reports/eval/preflight_seqlen.json", f"results[{mid}].max", 0)
        put(f"{M}VramLoad", pre[key]["vram_gb_after_load"], "reports/s5/preflight_models.json",
            f"results.{key}.vram_gb_after_load", 2, "GB")
        losses, evals, walls, vrams, steps, params = [], [], [], [], [], set()
        for seed, S in SEEDS:
            f = f"reports/s5/{key}_seed{seed}.json"
            man = load_json(f)
            assert man["seed"] == seed and man["model_key"] == key and man["masking_gate"] == "PASS"
            params.add(man["trainable_params"])
            ev = man["eval_loss_by_step"]
            last_eval = ev[max(ev, key=int)]
            min_step = min(ev, key=lambda k_: (ev[k_], int(k_)))
            put(f"{M}{S}EvalMinStep", int(min_step), f, "step of minimum eval_loss", 0)
            put(f"{M}{S}EvalMin", ev[min_step], f, "minimum eval_loss", 3)
            put(f"{M}{S}EvalRise", last_eval - ev[min_step], f, "final eval_loss minus minimum", 3)
            put(f"{M}{S}EvalMinEpoch", 1 + int(int(min_step) > man["global_step"] / 3) + int(int(min_step) > 2 * man["global_step"] / 3),
                f, "epoch containing the minimum eval_loss (global_step/3 per epoch)", 0)
            EVAL_SUMMARY.append((key, seed, int(min_step), man["global_step"], last_eval - ev[min_step]))
            put(f"{M}{S}FinalTrainLoss", man["final_training_loss"], f, "final_training_loss", 3)
            put(f"{M}{S}FinalEvalLoss", last_eval, f, "eval_loss_by_step[last]", 3)
            put(f"{M}{S}Steps", man["global_step"], f, "global_step", 0)
            put(f"{M}{S}WallMin", man["wall_seconds"] / 60, f, "wall_seconds / 60", 1, "min")
            put(f"{M}{S}PeakVram", man["peak_vram_gb_reserved"], f, "peak_vram_gb_reserved", 1, "GB")
            losses.append(man["final_training_loss"]); evals.append(last_eval)
            walls.append(man["wall_seconds"]); vrams.append(man["peak_vram_gb_reserved"]); steps.append(man["global_step"])
            wall_total += man["wall_seconds"]
            if seed == 42:
                lora = man["lora"]
                put(f"{M}LoraTargets", len(man["lora_target_modules"]), f, "len(lora_target_modules)", 0)
                per_target = set(man["lora_modules_attached"].values())
                assert len(per_target) == 1
                # assert_targets_attached() counts named modules containing "lora_A":
                # the lora_A ModuleDict and its "default" Linear, i.e. two per layer.
                put(f"{M}LoraLayersAdapted", per_target.pop() // 2, f,
                    "lora_modules_attached per target / 2 (two lora_A entries per layer)", 0)
                if key == "llama31_8b":
                    put("LoraRank", lora["r"], f, "lora.r", 0)
                    put("LoraAlpha", lora["lora_alpha"], f, "lora.lora_alpha", 0)
                    put("LoraDropout", lora["lora_dropout"], f, "lora.lora_dropout", 2)
                    put("TrainRowsManifest", man["train_rows"], f, "train_rows", 0)
                    put("ValRowsManifest", man["val_rows"], f, "val_rows", 0)
        assert len(params) == 1, f"{key}: trainable params differ across seeds"
        p = params.pop()
        put(f"{M}TrainableParams", f"{p:,}", f"reports/s5/{key}_seed42.json", "trainable_params (identical x3 seeds)")
        put(f"{M}TrainableParamsM", p / 1e6, f"reports/s5/{key}_seed42.json", "trainable_params / 1e6", 1, "M")
        put(f"{M}StepsPerRun", steps[0], f"reports/s5/{key}_seed42.json", "global_step", 0)
        put(f"{M}WallMinMean", statistics.mean(walls) / 60, "reports/s5/{key}_seed*.json".replace("{key}", key),
            "mean(wall_seconds)/60", 1, "min")
        put(f"{M}PeakVramMax", max(vrams), "reports/s5/{key}_seed*.json".replace("{key}", key),
            "max(peak_vram_gb_reserved)", 1, "GB")
        put(f"{M}FinalEvalLossMean", statistics.mean(evals), "reports/s5/{key}_seed*.json".replace("{key}", key),
            "mean(eval_loss_by_step[last])", 3)
        # clarify-trained cell
        fc = f"reports/s5/{key}_seed42_clarify.json"
        mc = load_json(fc)
        put(f"{M}CLTrainRows", mc["train_rows"], fc, "train_rows", 0)
        put(f"{M}CLSteps", mc["global_step"], fc, "global_step", 0)
        put(f"{M}CLWallMin", mc["wall_seconds"] / 60, fc, "wall_seconds / 60", 1, "min")
    put("TrainGpuHoursTwelve", wall_total / 3600, "reports/s5/*_seed{42,1337,2024}.json", "sum(wall_seconds)/3600", 1, "h")
    put("EvalMinBeforeFinalRuns", sum(r_[4] > 1e-9 for r_ in EVAL_SUMMARY), "reports/s5/*_seed*.json",
        "runs whose final eval_loss is above their minimum", 0)
    put("EvalMinInLastEpochRuns", sum(r_[2] > 2 * r_[3] / 3 for r_ in EVAL_SUMMARY), "reports/s5/*_seed*.json",
        "runs whose minimum eval_loss falls in the third epoch", 0)
    put("EvalRiseMax", max(r_[4] for r_ in EVAL_SUMMARY), "reports/s5/*_seed*.json",
        "largest final-minus-minimum eval_loss over 12 runs", 3)
    put("PhiFourBrokenAdapterParams", f"{pre['phi4_mini']['trainable_params']:,}", "reports/s5/preflight_models.json",
        "results.phi4_mini.trainable_params (seven-module list, pre-fix)")

    # Hyperparameters are constants in the training script.
    code = src("scripts/s5_train.py").read_text(encoding="utf-8")
    s = "scripts/s5_train.py"
    put("HpEpochs", int(grab(code, r"num_train_epochs=(\d+)", "epochs").group(1)), s, "SFTConfig.num_train_epochs", 0)
    lr = grab(code, r"learning_rate=([0-9.eE-]+)", "lr").group(1)
    put("HpLearningRate", f"{float(lr):.0e}".replace("e-0", "\\times10^{-") + "}", s, "SFTConfig.learning_rate")
    put("HpScheduler", grab(code, r'lr_scheduler_type="(\w+)"', "sched").group(1), s, "SFTConfig.lr_scheduler_type")
    put("HpWarmup", int(grab(code, r"else (\d+),\s*\n\s*optim=", "warmup").group(1)), s, "SFTConfig.warmup_steps (full run)", 0)
    put("HpOptim", grab(code, r'optim="([\w]+)"', "optim").group(1), s, "SFTConfig.optim")
    put("HpWeightDecay", float(grab(code, r"weight_decay=([0-9.]+)", "wd").group(1)), s, "SFTConfig.weight_decay", 3)
    put("HpMaxGradNorm", float(grab(code, r"max_grad_norm=([0-9.]+)", "gn").group(1)), s, "SFTConfig.max_grad_norm", 1)
    put("HpEvalSteps", int(grab(code, r"eval_steps = 25 if smoke else (\d+)", "eval_steps").group(1)), s, "eval_steps", 0)
    put("HpLoggingSteps", int(grab(code, r"logging_steps=1 if smoke else (\d+)", "log").group(1)), s, "logging_steps", 0)
    put("HpEffectiveBatch", int(grab(code, r"effective batch = batch \* grad_accum = (\d+)", "eff").group(1)), s,
        "comment on MODEL_BATCH", 0)
    for key, M, _ in MODELS:
        m = grab(code, rf'"{key}": \{{"per_device_train_batch_size": (\d+), "gradient_accumulation_steps": (\d+)\}}', key)
        put(f"{M}BatchSize", int(m.group(1)), s, f"MODEL_BATCH[{key}].per_device_train_batch_size", 0)
        put(f"{M}GradAccum", int(m.group(2)), s, f"MODEL_BATCH[{key}].gradient_accumulation_steps", 0)
    put("MaskingProbes", int(WORDNUM.get(grab(code, r"pulls (\d+|\w+) real examples", "probes").group(1),
                                          None) or grab(code, r"pulls (\d+) real", "probes").group(1)),
        s, "docstring: masking gate probe count", 0)

    ev = src("scripts/s6_eval.py").read_text(encoding="utf-8")
    put("GenMaxNewTokens", int(grab(ev, r"GEN_MAX_NEW_TOKENS = (\d+)", "gen").group(1)), "scripts/s6_eval.py", "GEN_MAX_NEW_TOKENS", 0)
    put("InferMaxSeqLen", int(grab(ev, r"INFER_MAX_SEQ_LEN = (\d+)", "inf").group(1)), "scripts/s6_eval.py", "INFER_MAX_SEQ_LEN", 0)
    m = grab(ev, r'TOL = \{"kc": ([0-9.]+), "et0_mm": ([0-9.]+), "rainfall_mm": ([0-9.]+), "area_sqm": ([0-9.]+)', "tol")
    put("TolKc", float(m.group(1)), "scripts/s6_eval.py", "TOL.kc", 3)
    put("TolEt", float(m.group(2)), "scripts/s6_eval.py", "TOL.et0_mm", 2)
    put("TolRain", float(m.group(3)), "scripts/s6_eval.py", "TOL.rainfall_mm", 2)
    put("TolAreaAbs", float(m.group(4)), "scripts/s6_eval.py", "TOL.area_sqm", 0)
    put("TolAreaRelPct", 0.005 * 100, "scripts/s6_eval.py", "area tolerance max(1, 0.005 x gold)", 1, "%")


# ===========================================================================
# 5. Stage 6 automatic metrics
# ===========================================================================
METRICS = [  # (csv column, macro stem, decimals, unit)
    ("exact_numeric_0p1", "ExactPtOne", 1, "%"),
    ("exact_numeric_1p0", "ExactOne", 1, "%"),
    ("MAE", "Mae", 3, "mm"),
    ("RMSE", "Rmse", 3, "mm"),
    ("net_extract_rate", "NetExtract", 1, "%"),
    ("slot_macro_F1", "SlotMacro", 3, ""),
    ("slot_kc_F1", "KcFone", 3, ""),
    ("slot_et0_mm_F1", "EtFone", 3, ""),
    ("slot_rainfall_mm_F1", "RainFone", 3, ""),
    ("slot_area_sqm_F1", "AreaFone", 3, ""),
    ("abstain_accuracy", "AbstainAcc", 1, "%"),
    ("exact_factual", "ExactFactual", 1, "%"),
    ("dice_word", "DiceWord", 3, ""),
    ("chrf_pp", "Chrf", 1, ""),
    ("bertscore_f1", "BertF", 3, ""),
    ("exact_clarify", "ClarifyExact", 1, "%"),
    ("invented_value_rate", "Invented", 1, "%"),
    ("multi_both_correct", "MultiBoth", 1, "%"),
    ("mean_gen_chars", "GenChars", 0, "chars"),
]
S6 = "reports/s6/summary_by_arm.csv"
S6L = "reports/s6/per_language.csv"


def ms(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def stage6_numbers() -> dict:
    section("stage 6 results")
    sm = load_csv(S6).set_index("arm")
    pl = load_csv(S6L)
    ranges = collections.defaultdict(dict)
    for key, M, _ in MODELS:
        for col, stem, nd, unit in METRICS:
            zs = sm.loc[f"{key}_zeroshot", col]
            put(f"{M}ZS{stem}", zs, S6, f"{key}_zeroshot.{col}", nd, unit)
            vals = []
            for seed, S in SEEDS:
                v = sm.loc[f"{key}_seed{seed}", col]
                vals.append(v)
                put(f"{M}FT{stem}{S}", v, S6, f"{key}_seed{seed}.{col}", nd, unit)
            mu, sd = ms(vals)
            put(f"{M}FT{stem}", mu, S6, f"mean({key}_seed{{42,1337,2024}}.{col})", nd, unit)
            put(f"{M}FT{stem}SD", sd, S6, f"stdev(ddof=1)({key}_seed*.{col})", nd, unit)
            put(f"{M}FT{stem}Min", min(vals), S6, f"min over seeds({col})", nd, unit)
            put(f"{M}FT{stem}Max", max(vals), S6, f"max over seeds({col})", nd, unit)
            cl = sm.loc[f"{key}_seed42_clarify", col]
            put(f"{M}CL{stem}", cl, S6, f"{key}_seed42_clarify.{col}", nd, unit)
            ranges[stem].setdefault("ZS", []).append(zs)
            ranges[stem].setdefault("FT", []).append(mu)
            ranges[stem].setdefault("CL", []).append(cl)
            ranges[stem].setdefault("FTseed", []).extend(vals)
            ranges[stem]["nd"] = nd
            ranges[stem]["unit"] = unit
        # per language
        for lang, Lg in (("bn", "Bn"), ("en", "En")):
            sub = pl[pl["lang"] == lang].set_index("arm")
            for col, stem, nd, unit in METRICS:
                if stem in ("GenChars", "MultiBoth"):
                    continue
                zs = sub.loc[f"{key}_zeroshot", col]
                put(f"{M}ZS{stem}{Lg}", zs, S6L, f"{key}_zeroshot[{lang}].{col}", nd, unit)
                vals = [sub.loc[f"{key}_seed{seed}", col] for seed, _ in SEEDS]
                mu, sd = ms(vals)
                put(f"{M}FT{stem}{Lg}", mu, S6L, f"mean({key}_seed*[{lang}].{col})", nd, unit)
                put(f"{M}FT{stem}{Lg}SD", sd, S6L, f"stdev({key}_seed*[{lang}].{col})", nd, unit)
                put(f"{M}CL{stem}{Lg}", sub.loc[f"{key}_seed42_clarify", col], S6L,
                    f"{key}_seed42_clarify[{lang}].{col}", nd, unit)
        # en minus bn gaps, FT means and per seed
        for col, stem, nd in (("chrf_pp", "Chrf", 1), ("exact_numeric_0p1", "ExactPtOne", 1),
                              ("exact_numeric_1p0", "ExactOne", 1), ("bertscore_f1", "BertF", 3),
                              ("exact_factual", "ExactFactual", 1)):
            bn = pl[(pl["lang"] == "bn")].set_index("arm")
            en = pl[(pl["lang"] == "en")].set_index("arm")
            gaps = [en.loc[f"{key}_seed{s}", col] - bn.loc[f"{key}_seed{s}", col] for s, _ in SEEDS]
            mu = statistics.mean(gaps)
            put(f"{M}FT{stem}GapEnBn", mu, S6L, f"mean over seeds(en - bn).{col}", nd)
            put(f"{M}FT{stem}GapPosSeeds", sum(g > 0 for g in gaps), S6L,
                f"count seeds with en > bn on {col}", 0)
            put(f"{M}FT{stem}GapNegSeeds", sum(g < 0 for g in gaps), S6L,
                f"count seeds with en < bn on {col}", 0)
            ranges[f"Gap{stem}"].setdefault("FT", []).append(mu)
            ranges[f"Gap{stem}"]["nd"] = nd
            ranges[f"Gap{stem}"]["unit"] = ""
            ranges[f"Gap{stem}"].setdefault("pos", 0)
            ranges[f"Gap{stem}"]["pos"] += sum(g > 0 for g in gaps)
            ranges[f"Gap{stem}"].setdefault("neg", 0)
            ranges[f"Gap{stem}"]["neg"] += sum(g < 0 for g in gaps)

    section("stage 6 ranges across models")
    for stem, d in ranges.items():
        nd, unit = d["nd"], d["unit"]
        for v in ("ZS", "FT", "CL", "FTseed"):
            if v in d:
                lab = {"ZS": "ZS", "FT": "FT", "CL": "CL", "FTseed": "FTAnySeed"}[v]
                put(f"Range{lab}{stem}Min", min(d[v]), S6 if not stem.startswith("Gap") else S6L,
                    f"min across models ({v}) of {stem}", nd, unit)
                put(f"Range{lab}{stem}Max", max(d[v]), S6 if not stem.startswith("Gap") else S6L,
                    f"max across models ({v}) of {stem}", nd, unit)
        if "pos" in d:
            put(f"Range{stem}PosCells", d["pos"], S6L, f"FT cells (of 12) with en > bn on {stem}", 0)
            put(f"Range{stem}NegCells", d["neg"], S6L, f"FT cells (of 12) with en < bn on {stem}", 0)

    # count FT cells with zero clarify / 100% invented / 100% abstain
    ft_arms = [f"{k}_seed{s}" for k, _, _ in MODELS for s, _ in SEEDS]
    put("FTCells", len(ft_arms), S6, "count of fine-tuned arms", 0)
    put("FTCellsZeroClarifyExact", int((sm.loc[ft_arms, "exact_clarify"] == 0).sum()), S6,
        "count FT arms with exact_clarify == 0", 0)
    put("FTCellsFullInvented", int((sm.loc[ft_arms, "invented_value_rate"] == 100).sum()), S6,
        "count FT arms with invented_value_rate == 100", 0)
    put("FTCellsFullAbstain", int((sm.loc[ft_arms, "abstain_accuracy"] == 100).sum()), S6,
        "count FT arms with abstain_accuracy == 100", 0)
    put("FTCellsZeroMultiBoth", int((sm.loc[ft_arms, "multi_both_correct"] == 0).sum()), S6,
        "count FT arms with multi_both_correct == 0", 0)
    put("ZSRuns", 1, "reports/s6/*_zeroshot_meta.json", "one zero-shot run per model (n=1)", 0)
    return {}


# ===========================================================================
# 6. Confusion matrices
# ===========================================================================
def confusion_numbers() -> None:
    section("confusion matrices")
    cm = load_json("reports/s6/confusion_matrices.json")
    s = "reports/s6/confusion_matrices.json"
    for arm, A in (("llama31_8b_zeroshot", "LlamaZS"), ("llama31_8b_seed42", "LlamaFTSeedA"),
                   ("llama31_8b_seed42_clarify", "LlamaCL")):
        o = cm[arm]["overall"]
        for g, G in (("answer", "Ans"), ("clarify", "Cla"), ("abstain", "Abs")):
            for p, P in (("answer", "Ans"), ("clarify", "Cla"), ("abstain", "Abs"), ("empty", "Emp")):
                put(f"Cm{A}{G}To{P}", o[g][p], s, f"{arm}.overall.{g}.{p}", 0)


# ===========================================================================
# 7. Clarification: learned vs exact, cross-tab, detector
# ===========================================================================
def clarify_numbers() -> None:
    section("clarification")
    from s6_eval import classify_response, score_clarify  # stdlib-only module

    from s6_eval import extract_numeric_fields
    master = load_csv("reports/s6/results_master.csv")
    master["value"] = pd.to_numeric(master["value"], errors="coerce")
    ft_learned_zero, ft_emitted, ft_net = 0, [], []
    for key, M, _ in MODELS:
        arms = [(f"{key}_zeroshot", "ZS"), (f"{key}_seed42", "FTSeedA"), (f"{key}_seed1337", "FTSeedB"),
                (f"{key}_seed2024", "FTSeedC"), (f"{key}_seed42_clarify", "CL")]
        for arm, V in arms:
            f = f"reports/s6/predictions/{arm}.jsonl"
            rows = [r for r in load_jsonl(f) if r["qtype"] == "clarify" and str(r.get("gold", "")).strip()]
            n = len(rows)
            cls = [classify_response(r.get("generation") or "") for r in rows]
            sc = [score_clarify(r.get("generation") or "", r.get("subtype", ""))[0] for r in rows]
            learned = sum(c == "clarify" for c in cls)
            invented = sum(c == "answer" for c in cls)
            exact = sum(s == 1.0 for s in sc)
            both = sum(c == "answer" and s == 1.0 for c, s in zip(cls, sc))
            exact_not_learned = sum(c != "clarify" and s == 1.0 for c, s in zip(cls, sc))
            learned_not_exact = sum(c == "clarify" and s < 1.0 for c, s in zip(cls, sc))
            # cross-check exact against results_master
            mm = master[(master["arm"] == arm) & (master["metric"] == "clarify_exact")]
            assert int(mm["value"].sum()) == exact and len(mm) == n, f"{arm}: clarify_exact mismatch"
            put(f"{M}{V}ClarifyN", n, f, "clarify rows with non-empty gold", 0)
            put(f"{M}{V}ClarifyLearnedN", learned, f, "count classify_response == clarify", 0)
            put(f"{M}{V}ClarifyLearned", pct(learned, n), f, "pct classify_response == clarify", 1, "%")
            put(f"{M}{V}ClarifyExactN", exact, f, "count score_clarify == 1.0", 0)
            put(f"{M}{V}InventedN", invented, f, "count classify_response == answer", 0)
            put(f"{M}{V}ExactAndInventedN", both, f, "count exact==1 and class==answer", 0)
            put(f"{M}{V}ExactNotLearnedN", exact_not_learned, f, "count exact==1 and class!=clarify", 0)
            put(f"{M}{V}LearnedNotExactN", learned_not_exact, f, "count class==clarify and exact<1", 0)
            ext = [extract_numeric_fields(r.get("generation") or "") for r in rows]
            emitted = sum(e["net_mm"] is not None or e["liters"] is not None for e in ext)
            net = sum(e["net_mm"] is not None for e in ext)
            if V in ("CL", "FTSeedA"):
                SUB = {"missing_et0": "MissEt", "missing_rainfall": "MissRain", "missing_area": "MissArea",
                       "missing_stage": "MissStage", "ambiguous": "AmbigCrop", "unknown_local_unit": "UnknownUnit"}
                for sub, S_ in SUB.items():
                    idx = [i for i, r in enumerate(rows) if r.get("subtype") == sub]
                    put(f"{M}{V}Exact{S_}N", sum(sc[i] == 1.0 for i in idx), f, f"score_clarify == 1.0 on subtype {sub}", 0)
                    if M == "Llama" and V == "CL":
                        put(f"ClarifySub{S_}Rows", len(idx), f, f"clarify rows with subtype {sub}", 0)
            put(f"{M}{V}ClarifyEmittedPct", pct(emitted, n), f, "pct clarify rows stating a net depth or litres", 1, "%")
            put(f"{M}{V}ClarifyNetPct", pct(net, n), f, "pct clarify rows stating a net depth (mm)", 1, "%")
            if V.startswith("FTSeed"):
                ft_learned_zero += learned == 0
                ft_emitted.append(pct(emitted, n))
                ft_net.append(pct(net, n))
    put("FTCellsZeroLearned", ft_learned_zero, "reports/s6/predictions/*_seed{42,1337,2024}.jsonl",
        "FT arms with zero clarify-classified responses on clarify rows", 0)
    put("FTClarifyEmittedMin", min(ft_emitted), "reports/s6/predictions/*_seed{42,1337,2024}.jsonl",
        "min over 12 FT arms of pct clarify rows stating a depth or litres", 1, "%")
    put("FTClarifyNetMin", min(ft_net), "reports/s6/predictions/*_seed{42,1337,2024}.jsonl",
        "min over 12 FT arms of pct clarify rows stating a net depth", 1, "%")
    # ranges for clarify-trained arms
    for stem in ("ClarifyLearned",):
        vals = [float(MACROS[f"{M}CL{stem}"]["value"]) for _, M, _ in MODELS]
        put(f"RangeCL{stem}Min", min(vals), "reports/s6/predictions/*_seed42_clarify.jsonl", f"min across models of {stem}", 1, "%")
        put(f"RangeCL{stem}Max", max(vals), "reports/s6/predictions/*_seed42_clarify.jsonl", f"max across models of {stem}", 1, "%")

    # cross-check against the Stage 6D/6E summaries
    cc = load_json("reports/s6/clarify_comparison.json")
    ll = next(r for r in cc if r["model"] == "llama31_8b" and r["variant"] == "seed_42_clarify")
    assert abs(ll["clarify_percent"] - float(MACROS["LlamaCLClarifyLearned"]["value"])) < 0.05
    imp = load_json("reports/s6/s6e_clarify_impact.json")
    name_to_M = {"Mistral-NeMo-12B": "Mistral", "Phi-3.5-mini": "PhiThree", "Phi-4-mini": "PhiFour"}
    for r in imp:
        M = name_to_M[r["Model"]]
        assert r["Clarify after"] == MACROS[f"{M}CLClarifyLearned"]["value"] + "%", r
        assert r["Exact clarify after"] == MACROS[f"{M}CLClarifyExact"]["value"] + "%", r
    put("ClarifySummariesAgree", 1, "reports/s6/clarify_comparison.json + s6e_clarify_impact.json",
        "recomputed values match both files (1 = yes)", 0)

    # zero-shot non-answer clarify rows (cross_model_summary lists them)
    section("detector (Stage 6C)")
    f = "reports/s6/predictions/llama31_8b_seed42_clarify_with_detector.jsonl"
    rows = load_jsonl(f)
    put("DetRowsTotal", len(rows), f, "rows", 0)
    cl = [r for r in rows if r.get("qtype") == "clarify" and str(r.get("gold", "")).strip()]
    n = len(cl)
    learned = sum(r.get("predicted_qtype") == "clarify" for r in cl)
    caught = sum(r.get("predicted_qtype") != "clarify" and bool(r.get("detector_flag")) for r in cl)
    still = n - learned - caught
    put("DetClarifyN", n, f, "scorable clarify rows", 0)
    put("DetLearnedN", learned, f, "predicted_qtype == clarify", 0)
    put("DetLearnedPct", pct(learned, n), f, "predicted_qtype == clarify / n", 1, "%")
    put("DetCaughtN", caught, f, "detector_flag and not learned", 0)
    put("DetCaughtPct", pct(caught, n), f, "detector_flag and not learned / n", 1, "%")
    put("DetCombinedN", learned + caught, f, "learned + caught", 0)
    put("DetCombinedPct", pct(learned + caught, n), f, "(learned + caught) / n", 1, "%")
    put("DetStillInventedN", still, f, "n - learned - caught", 0)
    put("DetStillInventedPct", pct(still, n), f, "(n - learned - caught) / n", 1, "%")
    flagged_nonclarify = sum(bool(r.get("detector_flag")) for r in rows if r.get("qtype") != "clarify")
    put("DetFlaggedNonClarify", flagged_nonclarify, f, "detector_flag on rows whose gold qtype != clarify", 0)
    code = src("scripts/s6c_postfix_clarify_detector.py").read_text(encoding="utf-8")
    cats = grab(code, r'row\.get\("category"\) in \{(.*?)\}', "detector categories", re.S).group(1)
    put("DetGoldCategoriesUsed", len(re.findall(r'"(\w+)"', cats)), "scripts/s6c_postfix_clarify_detector.py",
        "number of gold categories in the detector's flag rule", 0)
    put("DetSlotRules", len(re.findall(r'^\s+"(\w+)": \[', grab(code, r"REQUIRED_SLOTS = \{(.*?)\n\}", "slots", re.S).group(1), re.M)),
        "scripts/s6c_postfix_clarify_detector.py", "len(REQUIRED_SLOTS)", 0)


# ===========================================================================
# 8. Output form: generation length and truncation (descriptive only)
# ===========================================================================
def output_form_numbers() -> None:
    section("output form (descriptive)")
    pc = load_json("reports/stats/precheck.json")
    s = "reports/stats/precheck.json"
    put("FinishReasonStored", int(pc["finish_reason_stored"]), s, "finish_reason_stored (0 = no)", 0)
    put("TokenCountStored", int(pc["token_count_stored"]), s, "token_count_stored (0 = no)", 0)
    for arm, A in (("llama31_8b_zeroshot", "LlamaZS"), ("llama31_8b_seed42", "LlamaFTSeedA")):
        d = pc["token_count_distribution"][arm]
        put(f"{A}AllRowsMedianTok", d["median"], s, f"token_count_distribution.{arm}.median", 0, "tokens")
        put(f"{A}AllRowsPctCap", d["pct_ge_511"], s, f"token_count_distribution.{arm}.pct_ge_511", 1, "%")
        put(f"{A}ClarifyAtCapN", d["clarify_rows_ge_511"], s, f"token_count_distribution.{arm}.clarify_rows_ge_511", 0)
    a7 = load_csv("reports/stats/a7_descriptive.csv")
    s = "reports/stats/a7_descriptive.csv"
    for key, M, _ in MODELS:
        for var, V in (("zeroshot", "ZS"), ("ft_seed42", "FTSeedA")):
            r = a7[(a7["model"] == key) & (a7["variant"] == var)].iloc[0]
            put(f"{M}{V}ClarifyTruncPct", r["truncation_rate"] * 100, s, f"{key}/{var}.truncation_rate x100 (point value only)", 1, "%")
            put(f"{M}{V}ClarifyMedianTok", r["median_gen_tokens"], s, f"{key}/{var}.median_gen_tokens (point value only)", 1, "tokens")
            put(f"{M}{V}ExplicitAssumeN", r["explicit_assumption_n_all_clarify"], s,
                f"{key}/{var}.explicit_assumption_n_all_clarify", 0)
    tc = load_json("reports/stats/a7_truncation_caveat.json")
    put("TruncThresholdTok", int(grab(tc["threshold"], r">= (\d+)", "thr").group(1)),
        "reports/stats/a7_truncation_caveat.json", "threshold", 0, "tokens")
    # character counts on clarify rows, all arms, straight from predictions
    for key, M, _ in MODELS:
        for arm, V in ((f"{key}_zeroshot", "ZS"), (f"{key}_seed42", "FTSeedA"), (f"{key}_seed42_clarify", "CL")):
            f = f"reports/s6/predictions/{arm}.jsonl"
            rows = [r for r in load_jsonl(f) if r["qtype"] == "clarify" and str(r.get("gold", "")).strip()]
            lens = [len(r.get("generation") or "") for r in rows]
            put(f"{M}{V}ClarifyMedianChars", statistics.median(lens), f, "median len(generation) on clarify rows", 0, "chars")

    # Token counts for every arm and seed (scripts/thesis/gen_tokens.py; same method as
    # the Stage 8 A7 descriptive, reproduced exactly for the arms that file covers).
    g = "thesis/generated/gen_tokens.csv"
    chk = json.loads((OUT / "gen_tokens_check.json").read_text(encoding="utf-8"))
    assert chk["all_match"], "gen_tokens.csv does not reproduce reports/stats point values"
    put("GenTokensReproduced", int(chk["all_match"]), "thesis/generated/gen_tokens_check.json",
        "all_match (1 = recomputed counts equal a7_descriptive.csv and precheck.json)", 0)
    gt = pd.read_csv(OUT / "gen_tokens.csv")
    cl = gt[gt["qtype"] == "clarify"]
    ft_trunc_rows, ft_max = 0, 0
    zs_trunc, zs_trunc_mp = [], []
    for key, M, _ in MODELS:
        z = cl[cl["arm"] == f"{key}_zeroshot"]
        zs_trunc.append(100 * z["truncated"].mean())
        put(f"{M}ZSClarifyTruncN", int(z["truncated"].sum()), g, f"{key}_zeroshot clarify rows >= threshold", 0)
        zm = z[z["category"] == "missing_parameter"]
        zs_trunc_mp.append(100 * zm["truncated"].mean())
        put(f"{M}ZSMissingParamTruncPct", 100 * zm["truncated"].mean(), g,
            f"{key}_zeroshot missing_parameter rows (n={len(zm)}) >= threshold", 1, "%")
        put(f"{M}ZSAllRowsTruncPct", 100 * gt[gt["arm"] == f"{key}_zeroshot"]["truncated"].mean(), g,
            f"{key}_zeroshot all 362 scorable rows >= threshold", 1, "%")
        for seed, S in SEEDS:
            s = cl[cl["arm"] == f"{key}_seed{seed}"]
            put(f"{M}FT{S}ClarifyTruncPctRecomputed", 100 * s["truncated"].mean(), g,
                f"{key}_seed{seed} clarify rows >= threshold", 1, "%")
            put(f"{M}FT{S}ClarifyMedianTokRecomputed", float(s["gen_tokens"].median()), g,
                f"{key}_seed{seed} clarify median tokens", 1, "tokens")
            ft_trunc_rows += int(s["truncated"].sum())
            ft_max = max(ft_max, int(s["gen_tokens"].max()))
        c = cl[cl["arm"] == f"{key}_seed42_clarify"]
        put(f"{M}CLClarifyTruncPct", 100 * c["truncated"].mean(), g, f"{key}_seed42_clarify clarify rows >= threshold", 1, "%")
        put(f"{M}CLClarifyMedianTok", float(c["gen_tokens"].median()), g, f"{key}_seed42_clarify clarify median tokens", 1, "tokens")
    put("RangeZSClarifyTruncMin", min(zs_trunc), g, "min across models, zero-shot clarify truncation", 1, "%")
    put("RangeZSClarifyTruncMax", max(zs_trunc), g, "max across models, zero-shot clarify truncation", 1, "%")
    put("RangeZSMissingParamTruncMin", min(zs_trunc_mp), g, "min across models, zero-shot missing_parameter truncation", 1, "%")
    put("RangeZSMissingParamTruncMax", max(zs_trunc_mp), g, "max across models, zero-shot missing_parameter truncation", 1, "%")
    put("FTClarifyTruncRowsAllSeeds", ft_trunc_rows, g, "truncated clarify rows summed over the 12 FT arms", 0)
    put("FTClarifyMaxTokAllSeeds", ft_max, g, "max generated tokens on clarify rows over the 12 FT arms", 0, "tokens")
    put("ZSClarifyRowsPerModel", int((cl["arm"] == "llama31_8b_zeroshot").sum()), g, "clarify rows per arm", 0)
    put("ZSMissingParamRowsPerModel", int(((cl["arm"] == "llama31_8b_zeroshot") & (cl["category"] == "missing_parameter")).sum()),
        g, "missing_parameter clarify rows per arm", 0)


# ===========================================================================
# 9. Human evaluation (Stage 7)
# ===========================================================================
ARMS_H = [("llama31_8b_zeroshot", "HZS"), ("llama31_8b_seed42", "HFT"),
          ("llama31_8b_seed42_clarify", "HCL"), ("mistral_nemo_seed42", "HMis")]
QT = [("numeric_runtime", "Numeric"), ("factual", "Factual"), ("clarify", "Clarify"), ("abstain", "Abstain")]


def human_numbers() -> None:
    section("human evaluation")
    f = "reports/human_eval/stage7_ratings_merged.csv"
    r = load_csv(f)
    put("HumJudgments", len(r), f, "rows", 0)
    put("HumAnnotators", r["annotator"].nunique(), f, "nunique(annotator)", 0)
    per_ann = r.groupby("annotator").size()
    assert per_ann.nunique() == 1
    put("HumPerAnnotator", int(per_ann.iloc[0]), f, "rows per annotator", 0)
    uniq = r.drop_duplicates("judgment_id")
    put("HumUniqueOutputs", len(uniq), f, "nunique(judgment_id)", 0)
    put("HumQuestions", uniq["row_id"].nunique(), f, "nunique(row_id)", 0)
    put("HumArms", uniq["arm"].nunique(), f, "nunique(arm)", 0)
    ov = r[r["is_overlap"] == "yes"]
    put("HumOverlapOutputs", ov["judgment_id"].nunique(), f, "nunique(judgment_id | is_overlap)", 0)
    put("HumOverlapRatings", len(ov), f, "rows with is_overlap == yes", 0)
    put("HumSoloOutputs", r[r["is_overlap"] == "no"]["judgment_id"].nunique(), f, "nunique(judgment_id | solo)", 0)
    ovu = ov.drop_duplicates("judgment_id")
    for arm, A in ARMS_H:
        put(f"{A}OverlapN", int((ovu["arm"] == arm).sum()), f, f"overlap outputs for {arm}", 0)
    for q, Q in QT:
        put(f"HumQ{Q}", int(uniq[uniq["arm"] == ARMS_H[0][0]]["qtype"].eq(q).sum()), f, f"questions with qtype {q}", 0)
    put("HumAuditFlags", int(r["audit_flag"].notna().sum()), f, "non-null audit_flag", 0)

    # per-judgment aggregation: mean rating; farmer yes by majority on overlap rows
    r = r.copy()
    r["yes"] = (r["would_give_farmer"].str.lower() == "yes").astype(int)
    g = r.groupby("judgment_id").agg(arm=("arm", "first"), qtype=("qtype", "first"), lang=("lang", "first"),
                                     rating=("rating_1_5", "mean"), yes=("yes", "mean"), k=("yes", "size"))
    g["yes_major"] = (g["yes"] > 0.5).astype(int)
    assert not ((g["k"] % 2 == 0) & (g["yes"] == 0.5)).any(), "tie in majority vote"
    for arm, A in ARMS_H:
        sub = g[g["arm"] == arm]
        put(f"{A}N", len(sub), f, f"judgments for {arm}", 0)
        put(f"{A}MeanRating", sub["rating"].mean(), f, "mean of per-judgment mean rating", 2)
        put(f"{A}PctGood", pct(int((sub["rating"] >= 4).sum()), len(sub)), f, "per-judgment mean >= 4", 1, "%")
        put(f"{A}PctBad", pct(int((sub["rating"] <= 2).sum()), len(sub)), f, "per-judgment mean <= 2", 1, "%")
        put(f"{A}PctYes", pct(int(sub["yes_major"].sum()), len(sub)), f, "majority would_give_farmer", 1, "%")
        for q, Q in QT:
            s2 = sub[sub["qtype"] == q]
            put(f"{A}{Q}N", len(s2), f, f"{arm} x {q} judgments", 0)
            put(f"{A}{Q}PctYes", pct(int(s2["yes_major"].sum()), len(s2)), f, f"{arm} x {q} majority yes", 1, "%")
            put(f"{A}{Q}MeanRating", s2["rating"].mean(), f, f"{arm} x {q} mean rating", 2)
        for lang, Lg in (("bn", "Bn"), ("en", "En")):
            s2 = sub[sub["lang"] == lang]
            put(f"{A}{Lg}N", len(s2), f, f"{arm} x {lang} judgments", 0)
            put(f"{A}{Lg}PctYes", pct(int(s2["yes_major"].sum()), len(s2)), f, f"{arm} x {lang} majority yes", 1, "%")
    rej = g[(g["qtype"] == "clarify") & g["arm"].isin(["llama31_8b_seed42", "mistral_nemo_seed42"])]
    rr = r[(r["qtype"] == "clarify") & r["arm"].isin(["llama31_8b_seed42", "mistral_nemo_seed42"])]
    put("HumFTClarifyRatings", len(rr), f, "individual ratings: FT arms (Llama, Mistral) x clarify", 0)
    put("HumFTClarifyRatingsYes", int(rr["yes"].sum()), f, "of those, would_give_farmer == yes", 0)

    # rubric drift: 1 vs 2 on zero-shot numeric; 1 on Llama FT clarify
    for ann in sorted(r["annotator"].unique()):
        a = r[(r["annotator"] == ann)]
        z = a[(a["arm"] == "llama31_8b_zeroshot") & (a["qtype"] == "numeric_runtime")]
        A = "Ann" + {"A1": "One", "A2": "Two", "A3": "Three", "A4": "Four", "A5": "Five"}[ann]
        put(f"{A}ZSNumN", len(z), f, f"{ann}: zero-shot numeric ratings", 0)
        put(f"{A}ZSNumOnes", int((z["rating_1_5"] == 1).sum()), f, f"{ann}: zero-shot numeric rated 1", 0)
        put(f"{A}ZSNumTwos", int((z["rating_1_5"] == 2).sum()), f, f"{ann}: zero-shot numeric rated 2", 0)
        c = a[(a["arm"] == "llama31_8b_seed42") & (a["qtype"] == "clarify")]
        put(f"{A}FTClaN", len(c), f, f"{ann}: Llama FT clarify ratings", 0)
        put(f"{A}FTClaOnes", int((c["rating_1_5"] == 1).sum()), f, f"{ann}: Llama FT clarify rated 1", 0)

    # agreement: taken from the Stage 7 audit report, as instructed
    a = "reports/human_eval/stage7_audit_report.md"
    t = src(a).read_text(encoding="utf-8")
    put("AgreeFarmerKappa", float(grab(t, r"would_give_farmer\*\* — Fleiss' κ \| \*\*([0-9.]+)", "k").group(1)), a, "§3 would_give_farmer Fleiss kappa", 3)
    put("AgreeFarmerAlpha", float(grab(t, r"would_give_farmer\*\* — Krippendorff's α \| \*\*([0-9.]+)", "a").group(1)), a, "§3 would_give_farmer Krippendorff alpha", 3)
    put("AgreeRatingOrdAlpha", float(grab(t, r"rating 1–5\*\* — Krippendorff's α, ordinal \| \*\*([0-9.]+)", "o").group(1)), a, "§3 rating ordinal alpha", 3)
    put("AgreeRatingIntAlpha", float(grab(t, r"rating 1–5 — Krippendorff's α, interval \| ([0-9.]+)", "i").group(1)), a, "§3 rating interval alpha", 3)
    put("AgreeRatingCollapsedKappa", float(grab(t, r"bad \(1–2\) / mid \(3\) / good \(4–5\) — Fleiss' κ \| ([0-9.]+)", "c").group(1)), a, "§3 collapsed 3-level Fleiss kappa", 3)
    put("AgreeRatingNominalKappa", float(grab(t, r"rating 1–5 — Fleiss' κ, nominal \| ([0-9.]+)", "n").group(1)), a, "§3 nominal Fleiss kappa", 3)
    put("AgreeThreshold", float(grab(t, r"Threshold ([0-9.]+)", "thr").group(1)), a, "§3 table header threshold", 1)
    put("HumAuditCorrections", len(re.findall(r"^\| A\d \| J\d+ \|.*\*\*Corrected to (?:yes|no)\*\*", t, re.M)), a,
        "§2 cells returned for correction and corrected", 0)
    m = grab(t, r"within one point on (\d+)–(\d+)%", "w1")
    put("AgreeWithinOneMin", int(m.group(1)), a, "§3 pairwise within-one-point min", 0, "%")
    put("AgreeWithinOneMax", int(m.group(2)), a, "§3 pairwise within-one-point max", 0, "%")
    m = grab(t, r"Removing A1 raises ordinal α from ([0-9.]+) to ([0-9.]+)", "loo")
    put("AgreeAlphaWithoutAOne", float(m.group(2)), a, "§3 ordinal alpha without A1", 3)
    m = grab(t, r"mean \(([0-9.]+); the others are ([0-9.]+)–([0-9.]+)\)", "corr")
    put("AgreeAOneCorr", float(m.group(1)), a, "§3 A1 correlation with others' mean", 2)
    put("AgreeOthersCorrMin", float(m.group(2)), a, "§3 others' correlation min", 2)
    put("AgreeOthersCorrMax", float(m.group(3)), a, "§3 others' correlation max", 2)
    m = grab(t, r"A3 ([0-9.]+), A4 ([0-9.]+), A5 ([0-9.]+)\)", "sev")
    put("SeverityAThree", float(m.group(1)), a, "§7 overlap mean A3", 2)
    put("SeverityAFour", float(m.group(2)), a, "§7 overlap mean A4", 2)
    put("SeverityAFive", float(m.group(3)), a, "§7 overlap mean A5", 2)

    # Leave-one-out ordinal alpha for every annotator (the audit reports only A1).
    # Computed with the krippendorff package; the all-five and without-A1 values
    # must reproduce the audit report exactly.
    import krippendorff
    ov = r[r["is_overlap"] == "yes"]
    Mx = ov.pivot_table(index="annotator", columns="judgment_id", values="rating_1_5", aggfunc="first")
    alpha = lambda mm: krippendorff.alpha(reliability_data=mm.values.astype(float), level_of_measurement="ordinal")
    assert f"{alpha(Mx):.3f}" == MACROS["AgreeRatingOrdAlpha"]["value"], "ordinal alpha does not reproduce the audit"
    assert f"{alpha(Mx.drop(index='A1')):.3f}" == MACROS["AgreeAlphaWithoutAOne"]["value"]
    for ann in Mx.index:
        A = {"A1": "One", "A2": "Two", "A3": "Three", "A4": "Four", "A5": "Five"}[ann]
        put(f"AgreeAlphaWithout{A}Recomputed", alpha(Mx.drop(index=ann)), f,
            f"krippendorff ordinal alpha on overlap ratings without {ann}", 3)
    put("AgreeLooRaisers", sum(alpha(Mx.drop(index=a_)) > alpha(Mx) for a_ in Mx.index), f,
        "annotators whose removal raises ordinal alpha", 0)
    # overlap-row rating means per annotator (severity)
    for ann in Mx.index:
        A = {"A1": "One", "A2": "Two", "A3": "Three", "A4": "Four", "A5": "Five"}[ann]
        put(f"Ann{A}OverlapMean", Mx.loc[ann].mean(), f, f"mean overlap rating of {ann}", 2)

    # Human rating by distance from gold (fine-tuned numeric answers, single-crop rows).
    mm = load_csv("reports/s6/results_master.csv")
    mm = mm[mm["metric"] == "abs_err"].copy()
    mm["value"] = pd.to_numeric(mm["value"])
    err = mm.set_index(["arm", "row_id"])["value"]
    bins = [("Exact", lambda e: e < 1e-6), ("PtOne", lambda e: 1e-6 <= e <= 0.1 + 1e-9),
            ("One", lambda e: 0.1 + 1e-9 < e <= 1.0 + 1e-9), ("Over", lambda e: e > 1.0 + 1e-9)]
    pooled = {b: [] for b, _ in bins}
    for arm in ("llama31_8b_seed42", "mistral_nemo_seed42", "llama31_8b_seed42_clarify"):
        sub = g[(g["arm"] == arm) & (g["qtype"] == "numeric_runtime")]
        for jid, row in sub.iterrows():
            key = (arm, r.loc[r["judgment_id"] == jid, "row_id"].iloc[0])
            if key not in err.index:
                continue
            e = err[key]
            for b, fn in bins:
                if fn(e):
                    pooled[b].append(row["rating"])
    for b, _ in bins:
        v = pooled[b]
        put(f"HumDist{b}N", len(v), "reports/human_eval/stage7_ratings_merged.csv + reports/s6/results_master.csv",
            f"fine-tuned numeric outputs in distance bin {b}", 0)
        if v:
            put(f"HumDist{b}Mean", statistics.mean(v), "reports/human_eval/stage7_ratings_merged.csv + reports/s6/results_master.csv",
                f"mean per-output rating in distance bin {b}", 2)

    # human sample composition
    eg = load_json("reports/stats/entry_gate.json")
    put("HumInTest", eg["human_rows_in_scorable_test"], "reports/stats/entry_gate.json", "human_rows_in_scorable_test", 0)
    put("HumAuthored", eg["human_rows_not_in_test_set"], "reports/stats/entry_gate.json", "human_rows_not_in_test_set", 0)
    for ann, arms in eg["solo_arm_by_annotator"].items():
        A = "Ann" + {"A1": "One", "A2": "Two", "A3": "Three", "A4": "Four", "A5": "Five"}[ann]
        for arm, H in ARMS_H:
            put(f"{A}Solo{H}", arms[arm], "reports/stats/entry_gate.json", f"solo_arm_by_annotator.{ann}.{arm}", 0)
    pc = load_json("reports/stats/precheck.json")
    au = pc["authored_18"]
    put("AuthoredClarify", au["by_qtype"]["clarify"], "reports/stats/precheck.json", "authored_18.by_qtype.clarify", 0)
    put("AuthoredNumeric", au["by_qtype"]["numeric_runtime"], "reports/stats/precheck.json", "authored_18.by_qtype.numeric_runtime", 0)
    for q, Q in QT:
        hs = pc["human_sample_by_qtype"][q]
        put(f"HumQ{Q}InTest", hs["in_test"], "reports/stats/precheck.json", f"human_sample_by_qtype.{q}.in_test", 0)
        put(f"HumQ{Q}Authored", hs["authored"], "reports/stats/precheck.json", f"human_sample_by_qtype.{q}.authored", 0)
    s7 = [p for p in sorted(src("reports/s7/predictions_translated_bn").glob("*.jsonl"))]
    put("AuthoredPredFiles", len(s7), "reports/s7/predictions_translated_bn/", "count of *.jsonl", 0)
    put("AuthoredRowsPerArm", len(load_jsonl(rel(s7[0]))), rel(s7[0]), "rows", 0)
    same = all(v["match"] for v in pc["provenance_18"]["arms"].values())
    put("AuthoredProvenanceMatch", int(same), "reports/stats/precheck.json", "all(provenance_18.arms[*].match)", 0)


# ===========================================================================
# 10. Stage 8 plan (the plan only; no results)
# ===========================================================================
def plan_numbers() -> None:
    section("stage 8 plan (fixed before significance testing)")
    p = load_json("reports/stats/test_plan.json")
    s = "reports/stats/test_plan.json"
    put("PlanTests", p["summary"]["n_tests"], s, "summary.n_tests", 0)
    for role, R in (("PRIMARY", "Primary"), ("SECONDARY", "Secondary"), ("SENSITIVITY", "Sensitivity"),
                    ("DESCRIPTIVE", "Descriptive")):
        put(f"PlanTests{R}", p["summary"]["by_role"][role], s, f"summary.by_role.{role}", 0)
    put("PlanHolmFamilies", len(p["summary"]["primary_per_holm_family"]), s, "len(summary.primary_per_holm_family)", 0)
    put("PlanBootstrap", f"{p['n_bootstrap']:,}", s, "n_bootstrap")
    put("PlanHypothesisRows", len(p["report_structure"]["table1_hypotheses"]), s, "len(report_structure.table1_hypotheses)", 0)
    put("PlanVersion", p["plan_version"], s, "plan_version", 0)
    a3 = next(t for t in p["tests"] if t["test_id"] == "A3a_llama31_8b")
    assert "clarify_exact" in a3["test"]
    put("PlanClarifyMetricIsExact", 1, s, "tests[A3a_*].test uses clarify_exact (1 = yes)", 0)


# ===========================================================================
# 11. Disclosed changes: numbers stated in the scripts themselves
# ===========================================================================
def change_numbers() -> None:
    section("disclosed changes (from script docstrings/comments)")
    ev = src("scripts/s6_eval.py").read_text(encoding="utf-8")
    s = "scripts/s6_eval.py"
    m = grab(ev, r"(\w+) such golds classified as clarify on markers alone", "five")
    put("ChangeBighaGolds", WORDNUM[m.group(1).lower()], s, "classify_response docstring: bigha_regional golds misclassified", 0)
    put("ChangeNamedNoAskPartials", int(grab(ev, r"every one of its (\d+) \"partials\"", "partials").group(1)), s,
        "score_clarify comment: llama31_8b_seed42 half-credit rows", 0)
    m = grab(ev, r"Only ([0-9.]+)% of gold numeric answers state the crop and ([0-9.]+)%", "ceil")
    put("ChangeCropCeilingPct", float(m.group(1)), s, "slot N/A comment: gold states crop", 1, "%")
    put("ChangeStageCeilingPct", float(m.group(2)), s, "slot N/A comment: gold states stage", 1, "%")
    put("ChangeOracleExtraRows", int(grab(ev, r"et0/rainfall on (\d+) further rows", "oracle").group(1)), s,
        "gold-oracle comment: et0/rainfall ceiling rows", 0)
    m = grab(ev, r"(\d+) collapsed \+ (\d+)\s*#?\s*correctly mapped = (\d+)", "join")
    put("JoinBugCollapsed", int(m.group(1)), s, "load_test_rows comment", 0)
    put("JoinBugMapped", int(m.group(2)), s, "load_test_rows comment", 0)
    put("CalcTagGoldRate", 0, "logs/s6_eval.log", "'calc-block gold rate: 0/422'", 0)
    lg = src("logs/s6_eval.log").read_text(encoding="utf-8", errors="replace")
    put("BnNormCases", int(grab(lg, r"bn_normalize v3\.1 PASS \((\d+) cases\)", "bn").group(1)), "logs/s6_eval.log",
        "entry gate bn_normalize PASS (n cases)", 0)
    put("ManifestCheckSkippedRuns", len(re.findall(r"verify_s5_manifests\.py SKIPPED by flag", lg)), "logs/s6_eval.log",
        "count of entry gates with manifest check skipped", 0)
    # The oracle score before the N/A rule is not recorded anywhere and the repo has no
    # version history, so no "before" macro exists (decision recorded in BUILD_REPORT.md).
    put("ChangeOracleAfter", grab(ev, r"must score ([0-9.]+) on every slot", "oracle").group(1), s,
        "score_row comment: gold-oracle arm must score this on every slot")


# ===========================================================================
# 12. Explicitly missing sources
# ===========================================================================
def tokenizer_numbers() -> None:
    section("tokenizer cost (descriptive)")
    s = "thesis/generated/tokenizer_cost.json"
    t = json.loads((OUT / "tokenizer_cost.json").read_text(encoding="utf-8"))
    assert t["test_sha256"] == sha256(src("data/hardened/test_hardened.jsonl"))
    put("TokPairs", t["n_pairs"], s, "n_pairs", 0)
    for q, Q in (("factual", "Factual"), ("numeric_runtime", "Numeric"), ("abstain", "Abstain"), ("clarify", "Clarify")):
        put(f"TokPairs{Q}", t["pairs_by_qtype"][q], s, f"pairs_by_qtype.{q}", 0)
    nboot = int(re.search(r"(\d+) resamples", t["ci"]).group(1))
    put("TokBootstrap", f"{nboot:,}", s, "ci (resamples)")
    meds = []
    for key, M, _ in MODELS:
        d = t["models"][key]
        put(f"{M}TokRatio", d["median_ratio_bn_over_en"], s, f"models.{key}.median_ratio_bn_over_en", 2)
        put(f"{M}TokRatioLo", d["ci95_low"], s, f"models.{key}.ci95_low", 2)
        put(f"{M}TokRatioHi", d["ci95_high"], s, f"models.{key}.ci95_high", 2)
        put(f"{M}TokPerCharBn", d["tokens_per_char_bn"], s, f"models.{key}.tokens_per_char_bn", 3)
        put(f"{M}TokPerCharEn", d["tokens_per_char_en"], s, f"models.{key}.tokens_per_char_en", 3)
        put(f"{M}TokMedianBn", d["median_tokens_bn"], s, f"models.{key}.median_tokens_bn", 1)
        put(f"{M}TokMedianEn", d["median_tokens_en"], s, f"models.{key}.median_tokens_en", 1)
        meds.append(d["median_ratio_bn_over_en"])
    put("RangeTokRatioMin", min(meds), s, "min over models of median ratio", 2)
    put("RangeTokRatioMax", max(meds), s, "max over models of median ratio", 2)


def verification_numbers() -> None:
    section("adapter verification")
    s = "thesis/generated/adapter_verification.json"
    a = json.loads((OUT / "adapter_verification.json").read_text(encoding="utf-8"))
    assert not a["problems"], a["problems"]
    put("AdapterArms", a["n_adapter_arms"], s, "n_adapter_arms", 0)
    put("AdapterArmsPass", a["n_adapter_pass"], s, "n_adapter_pass", 0)
    put("ZeroShotArmsBase", sum(r["adapter"] is None and r["status"].startswith("PASS") for r in a["arms"]
                                if "adapter" in r), s, "zero-shot arms that loaded base weights only", 0)
    put("EvalArms", a["n_arms"], s, "n_arms", 0)
    put("EvalManifestCheckSkippedArms", a["n_eval_manifest_check_skipped"], s, "n_eval_manifest_check_skipped", 0)
    put("AdapterSidecars", sum("sidecar" in r.get("records", {}) for r in a["arms"]), s, "arms with adapter.sha256 sidecar", 0)


# ===========================================================================
# 13. Stage 8 inferential results (released 2026-09-24, plan v3)
# ===========================================================================
MODEL_TOK = [("llama31_8b", "LLAMA"), ("mistral_nemo", "MISTRAL"), ("phi35_mini", "PHITHREE"), ("phi4_mini", "PHIFOUR")]
TOK_WORD = {
    "A1": "AOne", "A2": "ATwo", "A3a": "AThreeA", "A3b": "AThreeB", "A4": "AFour", "A5": "AFive", "A6": "ASix",
    "A7": "ASeven", "B1": "BOne", "B2": "BTwo", "B3": "BThree", "B4": "BFour", "B5": "BFive", "B8": "BEight",
    "LLAMA": "Llama", "MISTRAL": "Mistral", "PHITHREE": "PhiThree", "PHIFOUR": "PhiFour",
    "llama": "Llama", "mistral": "Mistral", "vs": "Vs", "friedman": "Friedman", "tier": "Tier",
    "detector": "Detector", "num": "Num", "fac": "Fac", "unpaired": "", "paired": "Paired", "sens": "Sens",
    "i": "FtVsZs", "ii": "ClVsFt", "iii": "MisVsFt", "overall": "Overall", "numeric": "Numeric", "runtime": "",
    "factual": "Factual", "clarify": "Clarify", "test": "", "all": "All", "exclA1": "ExclAOne", "rating": "Rating",
    "farmer": "Farmer", "zeroshot": "ZS", "FTARM": "FT", "CLARM": "CL", "all100": "", "native82": "Native",
    "spearman": "Rho", "cochranq": "Cochran", "mixedlm": "Mixed", "bayesglm": "Bayes",
}
SEED_SUFFIX = {42: "", 1337: "SeedB", 2024: "SeedC"}
S8A = "reports/stats/results_automatic.csv"
S8H = "reports/stats/results_human.csv"


def test_stem(test_id: str) -> str:
    s = test_id
    for key, tok in MODEL_TOK:
        s = s.replace(key, tok)
    s = s.replace("seed42_clarify", "CLARM").replace("seed42", "FTARM")
    return "Sig" + "".join(TOK_WORD[t] for t in s.split("_"))


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<~.001"
    return "=~" + f"{p:.3f}".lstrip("0")


def signed(v: float, nd: int) -> str:
    t = f"{abs(v):.{nd}f}"
    if float(t) == 0:
        return f"{0:.{nd}f}"
    return ("+" if v > 0 else "−") + t


def fmt_eff(v: float, nd: int = 2) -> str:
    if v == float("inf"):
        return "\\ensuremath{\\infty}"
    return f"{v:.{nd}f}".replace("-", "−")


def stage8_numbers() -> None:
    section("stage 8 inferential results")
    for path in (S8A, S8H):
        df = load_csv(path)
        for _, r in df.iterrows():
            stem = test_stem(r["test_id"]) + SEED_SUFFIX[int(r["seed"])]
            where = f"{r['test_id']} seed {int(r['seed'])}"
            unit = str(r["diff_unit"]) if pd.notna(r["diff_unit"]) else ""
            # A6 numeric (Fisher) rows carry no diff_unit; their rates and difference are proportions.
            prop = unit.startswith("proportion") or (not unit and "correct_0p1" in str(r["test"]))
            put(stem + "N", int(r["n"]), path, f"{where}: n", 0)
            if pd.notna(r["statistic"]):
                put(stem + "Stat", r["statistic"], path, f"{where}: statistic", 1 if abs(r["statistic"]) >= 1 else 2)
            if pd.notna(r["p_raw"]):
                put(stem + "P", fmt_p(r["p_raw"]), path, f"{where}: p_raw (formatted)")
            if pd.notna(r["p_holm"]):
                put(stem + "Pholm", fmt_p(r["p_holm"]), path, f"{where}: p_holm (formatted)")
            if pd.notna(r["effect"]):
                put(stem + "Eff", fmt_eff(r["effect"]), path, f"{where}: effect ({r['effect_name']})")
            if pd.notna(r["effect_ci_low"]):
                put(stem + "EffLo", fmt_eff(r["effect_ci_low"]), path, f"{where}: effect_ci_low", )
                put(stem + "EffHi", fmt_eff(r["effect_ci_high"]), path, f"{where}: effect_ci_high")
            if pd.notna(r["diff"]):
                k, nd = (100.0, 1) if prop else ((1.0, 1) if unit.startswith("metric") else (1.0, 2))
                put(stem + "Diff", signed(r["diff"] * k, nd), path, f"{where}: diff x{k:g} ({unit})")
                if pd.notna(r["diff_ci_low"]):
                    put(stem + "DiffLo", signed(r["diff_ci_low"] * k, nd), path, f"{where}: diff_ci_low x{k:g}")
                    put(stem + "DiffHi", signed(r["diff_ci_high"] * k, nd), path, f"{where}: diff_ci_high x{k:g}")
            for col, suf in (("rate_a", "RateA"), ("rate_b", "RateB")):
                if pd.notna(r[col]):
                    k, nd = (100.0, 1) if prop else (1.0, 2)
                    put(stem + suf, r[col] * k, path, f"{where}: {col} x{k:g}", nd)
            for col, suf in (("b", "B"), ("c", "C")):
                if pd.notna(r[col]):
                    put(stem + suf, int(r[col]), path, f"{where}: {col}", 0)

    # counts of PRIMARY tests and significant ones, seed 42 (the registered primary seed)
    a = load_csv(S8A); h = load_csv(S8H)
    pa = a[(a["role"] == "PRIMARY") & (a["seed"] == 42)]
    ph = h[(h["role"] == "PRIMARY") & (h["seed"] == 42)]
    put("SigAutoPrimaryTotal", len(pa), S8A, "PRIMARY rows at seed 42", 0)
    put("SigAutoPrimarySig", int((pa["p_holm"] < 0.05).sum()), S8A, "PRIMARY rows at seed 42 with p_holm < .05", 0)
    put("SigHumPrimaryTotal", len(ph), S8H, "PRIMARY rows", 0)
    put("SigHumPrimarySig", int((ph["p_holm"] < 0.05).sum()), S8H, "PRIMARY rows with p_holm < .05", 0)
    put("SigPrimaryTotal", len(pa) + len(ph), f"{S8A} + {S8H}", "PRIMARY rows (automatic seed 42 + human)", 0)
    put("SigPrimaryRobustSeedTests", int(((a["role"] == "PRIMARY") & (a["seed"] == 1337)).sum()), S8A,
        "PRIMARY automatic tests repeated at seed 1337 (same set at 2024)", 0)
    assert int(((a["role"] == "PRIMARY") & (a["seed"] == 1337)).sum()) == int(((a["role"] == "PRIMARY") & (a["seed"] == 2024)).sum())
    # direction check: no significant PRIMARY test points against its registered direction
    wrong = 0
    for df in (pa, ph):
        for _, r in df[df["p_holm"] < 0.05].iterrows():
            v = r["verdict"]
            wrong += not (v.startswith("supported") or v.startswith("significant"))
    put("SigPrimaryWrongDirection", wrong, f"{S8A} + {S8H}", "significant PRIMARY tests whose verdict is not 'supported'/'significant'", 0)
    put("SigAFiveRole", "descriptive", S8A, "A5_detector role (demoted in plan v3)")
    fac42 = pa[pa["test_id"].str.startswith("A6_fac")]
    put("SigASixFacSigSeedA", int((fac42["p_holm"] < 0.05).sum()), S8A, "A6_fac tests significant at seed 42", 0)
    num42 = pa[pa["test_id"].str.startswith("A6_num_unpaired")]
    put("SigASixNumSigSeedA", int((num42["p_holm"] < 0.05).sum()), S8A, "A6_num_unpaired tests significant at seed 42", 0)
    num_all = a[(a["role"] == "PRIMARY") & a["test_id"].str.startswith("A6_num_unpaired")]
    put("SigASixNumSigAnySeed", int((num_all["p_holm"] < 0.05).sum()), S8A, "A6_num_unpaired significant at any seed", 0)
    put("SigASixNumTestsAllSeeds", len(num_all), S8A, "A6_num_unpaired PRIMARY tests over three seeds", 0)
    hw = [abs(r["diff_ci_high"] - r["diff_ci_low"]) * 100 / 2 for _, r in num_all.iterrows()]
    put("SigASixNumHalfWidthMin", min(hw), S8A, "min CI half-width (pp) of A6 numeric, all seeds", 0)
    put("SigASixNumHalfWidthMax", max(hw), S8A, "max CI half-width (pp) of A6 numeric, all seeds", 0)
    b5 = ph[ph["test_id"].str.startswith("B5")]
    put("SigBFiveTests", len(b5), S8H, "B5 PRIMARY tests", 0)
    put("SigBFiveSig", int((b5["p_holm"] < 0.05).sum()), S8H, "B5 PRIMARY tests with p_holm < .05", 0)
    # A7 ranges (registered (a), seed 42)
    a7 = pa[pa["test_id"].str.startswith("A7")]
    put("RangeSigASevenDiffMin", a7["diff"].min() * 100, S8A, "min A7 diff at seed 42 (pp)", 1)
    put("RangeSigASevenDiffMax", a7["diff"].max() * 100, S8A, "max A7 diff at seed 42 (pp)", 1)
    a1 = pa[pa["test_id"].str.startswith("A1")]
    put("RangeSigAOneDiffMin", a1["diff"].min() * 100, S8A, "min A1 diff at seed 42 (pp)", 1)
    put("RangeSigAOneDiffMax", a1["diff"].max() * 100, S8A, "max A1 diff at seed 42 (pp)", 1)
    a3a = pa[pa["test_id"].str.startswith("A3a")]
    put("RangeSigAThreeADiffMin", a3a["diff"].min() * 100, S8A, "min A3a diff (pp)", 1)
    put("RangeSigAThreeADiffMax", a3a["diff"].max() * 100, S8A, "max A3a diff (pp)", 1)
    a3b = pa[pa["test_id"].str.startswith("A3b")]
    put("RangeSigAThreeBDiffMin", abs(a3b["diff"]).min() * 100, S8A, "min |A3b diff| (pp)", 1)
    put("RangeSigAThreeBDiffMax", abs(a3b["diff"]).max() * 100, S8A, "max |A3b diff| (pp)", 1)
    a4 = pa[pa["test_id"].str.startswith("A4")]
    put("SigAFourWorstLo", signed(a4["diff_ci_low"].min() * 100, 1), S8A, "lowest A4 CI bound (pp)")
    put("SigAFourLossRuledOutMin", abs(a4["diff_ci_low"]).min() * 100, S8A, "smallest |A4 CI lower bound| (pp)", 1)
    put("SigAFourLossRuledOutMax", abs(a4["diff_ci_low"]).max() * 100, S8A, "largest |A4 CI lower bound| (pp)", 1)
    put("SigAFourSig", int((a4["p_holm"] < 0.05).sum()), S8A, "A4 tests significant", 0)

    section("stage 8 seed robustness")
    sr = load_csv("reports/stats/seed_robustness_A.csv")
    s = "reports/stats/seed_robustness_A.csv"
    for _, r in sr.iterrows():
        put("Robust" + test_stem(r["test_id"])[3:], "yes" if bool(r["robust_all_3_seeds"]) else "no", s,
            f"{r['test_id']}.robust_all_3_seeds")
    put("RobustTests", len(sr), s, "tests with seed-robustness check", 0)
    put("RobustTestsYes", int(sr["robust_all_3_seeds"].astype(bool).sum()), s, "tests significant after Holm on all 3 seeds", 0)
    fac = sr[sr["test_id"].str.startswith("A6_fac")]
    put("RobustASixFacYes", int(fac["robust_all_3_seeds"].astype(bool).sum()), s, "A6 factual tests robust on 3 seeds", 0)

    section("stage 8 descriptive: bootstrap CIs of Stage 6 headlines")
    bc = load_csv("reports/stats/bootstrap_cis.csv")
    s = "reports/stats/bootstrap_cis.csv"
    MET = {"em_0p1": ("EmPtOneAll", 100, 1), "em_1p0": ("EmOne", 100, 1), "mae": ("Mae", 1, 3),
           "net_extract_rate": ("NetExtract", 100, 1), "abstain_accuracy": ("Abstain", 100, 1),
           "clarify_exact": ("ClarifyExact", 100, 1), "invented_value_rate": ("Invented", 100, 1),
           "dice_word": ("Dice", 1, 3), "chrf_pp": ("Chrf", 1, 1), "bertscore_f1": ("BertF", 1, 3)}
    bb = bc[bc["kind"] == "bootstrap"]
    for key, M, _ in MODELS:
        for arm, V in ((f"{key}_zeroshot", "ZS"), (f"{key}_seed42", "FTSeedA"), (f"{key}_seed1337", "FTSeedB"),
                       (f"{key}_seed2024", "FTSeedC"), (f"{key}_seed42_clarify", "CL")):
            for _, r in bb[bb["arm"] == arm].iterrows():
                stem, k, nd = MET[r["metric"]]
                put(f"Boot{M}{V}{stem}", r["estimate"] * k, s, f"{arm}.{r['metric']}.estimate x{k}", nd)
                put(f"Boot{M}{V}{stem}Lo", r["ci_low"] * k, s, f"{arm}.{r['metric']}.ci_low x{k}", nd)
                put(f"Boot{M}{V}{stem}Hi", r["ci_high"] * k, s, f"{arm}.{r['metric']}.ci_high x{k}", nd)
                if r["metric"] == "em_0p1":
                    put(f"Boot{M}{V}{stem}N", int(r["n"]), s, f"{arm}.em_0p1.n", 0)

    section("stage 8 descriptive: exact-match denominators (three tiers)")
    em = load_csv("reports/stats/exact_match_denominators.csv").set_index("arm")
    s = "reports/stats/exact_match_denominators.csv"
    sm = load_csv(S6).set_index("arm")
    strict = collections.defaultdict(dict)
    for key, M, _ in MODELS:
        for arm, V in ((f"{key}_zeroshot", "ZS"), (f"{key}_seed42", "FTSeedA"), (f"{key}_seed1337", "FTSeedB"),
                       (f"{key}_seed2024", "FTSeedC"), (f"{key}_seed42_clarify", "CL")):
            r = em.loc[arm]
            assert abs(100 * r["exact_0p1_single_rate"] - sm.loc[arm, "exact_numeric_0p1"]) < 0.05, arm
            put(f"{M}{V}CorrectAllN", int(r["correct_0p1_count"]), s, f"{arm}.correct_0p1_count", 0)
            put(f"{M}{V}CorrectAllPct", 100 * r["correct_0p1_rate"], s, f"{arm}.correct_0p1_rate x100 (/146)", 1, "%")
            put(f"{M}{V}ExactPtOneN", int(r["exact_0p1_single_count"]), s, f"{arm}.exact_0p1_single_count (/116)", 0)
            put(f"{M}{V}ExactStrictN", int(r["exact_lt0p005_single_count"]), s, f"{arm}.exact_lt0p005_single_count (/116)", 0)
            put(f"{M}{V}ExactStrictPct", 100 * r["exact_lt0p005_single_rate"], s, f"{arm}.exact_lt0p005_single_rate x100", 1, "%")
            put(f"{M}{V}ExtractedSingle", int(r["single_rows_with_extracted_depth"]), s, f"{arm}.single_rows_with_extracted_depth", 0)
            put(f"{M}{V}MultiAllCorrect", int(r["multi_all_correct_count"]), s, f"{arm}.multi_all_correct_count (/30)", 0)
            strict[V][M] = 100 * r["exact_lt0p005_single_rate"]
        ft = [strict[V][M] for V in ("FTSeedA", "FTSeedB", "FTSeedC")]
        mu, sd = ms(ft)
        put(f"{M}FTExactStrict", mu, s, "mean over seeds of exact_lt0p005_single_rate x100", 1, "%")
        put(f"{M}FTExactStrictSD", sd, s, "stdev over seeds of exact_lt0p005_single_rate x100", 1, "%")
        put(f"{M}ZSExactStrict", strict["ZS"][M], s, f"{key}_zeroshot exact_lt0p005 x100", 1, "%")
        put(f"{M}CLExactStrict", strict["CL"][M], s, f"{key}_seed42_clarify exact_lt0p005 x100", 1, "%")
    put("NumericAll", int(em.iloc[0]["correct_0p1_denominator"]), s, "correct_0p1_denominator", 0)
    put("MultiDenominator", int(em.iloc[0]["multi_denominator"]), s, "multi_denominator", 0)
    put("MultiAllCorrectMaxAnyArm", int(em["multi_all_correct_count"].max()), s, "max multi_all_correct_count over arms", 0)
    for V, lab in (("ZS", "ZS"), ("CL", "CL")):
        put(f"Range{lab}ExactStrictMin", min(strict[V].values()), s, f"min across models ({V}) exact_lt0p005", 1, "%")
        put(f"Range{lab}ExactStrictMax", max(strict[V].values()), s, f"max across models ({V}) exact_lt0p005", 1, "%")
    ftm = [statistics.mean([strict[V][M] for V in ("FTSeedA", "FTSeedB", "FTSeedC")]) for _, M, _ in MODELS]
    put("RangeFTExactStrictMin", min(ftm), s, "min across models of FT mean exact_lt0p005", 1, "%")
    put("RangeFTExactStrictMax", max(ftm), s, "max across models of FT mean exact_lt0p005", 1, "%")

    section("stage 8 descriptive: A7 confident error")
    a7 = load_csv("reports/stats/a7_descriptive.csv")
    s = "reports/stats/a7_descriptive.csv"
    rng_ = collections.defaultdict(list)
    for key, M, _ in MODELS:
        for var, V in (("zeroshot", "ZS"), ("ft_seed42", "FTSeedA")):
            r = a7[(a7["model"] == key) & (a7["variant"] == var)].iloc[0]
            for col, stem in (("confident_error_rate_a", "ConfErrA"), ("confident_error_rate_b_strict", "ConfErrB"),
                              ("truncation_rate", "TruncRate"), ("invented_value_rate", "InventedRate")):
                put(f"{M}{V}{stem}", 100 * r[col], s, f"{key}/{var}.{col} x100", 1, "%")
                lo = {"confident_error_rate_a": "confident_error_rate_a_lo", "confident_error_rate_b_strict": "confident_error_rate_b_strict_lo",
                      "truncation_rate": "truncation_rate_lo", "invented_value_rate": "invented_value_rate_lo"}[col]
                put(f"{M}{V}{stem}Lo", 100 * r[lo], s, f"{key}/{var}.{lo} x100", 1, "%")
                put(f"{M}{V}{stem}Hi", 100 * r[lo[:-3] + '_hi'], s, f"{key}/{var}.{lo[:-3]}_hi x100", 1, "%")
                rng_[(V, stem)].append(100 * r[col])
            put(f"{M}{V}InventedNAseven", int(r["n_invented"]), s, f"{key}/{var}.n_invented", 0)
            put(f"{M}{V}ExplicitAmongInv", 100 * r["explicit_assumption_rate_among_invented"], s,
                f"{key}/{var}.explicit_assumption_rate_among_invented x100", 1, "%")
            put(f"{M}{V}ExplicitAmongInvLo", 100 * r["ea_lo"], s, f"{key}/{var}.ea_lo x100", 1, "%")
            put(f"{M}{V}ExplicitAmongInvHi", 100 * r["ea_hi"], s, f"{key}/{var}.ea_hi x100", 1, "%")
            rng_[(V, "ExplicitAmongInv")].append(100 * r["explicit_assumption_rate_among_invented"])
            put(f"{M}{V}MedianTokAseven", r["median_gen_tokens"], s, f"{key}/{var}.median_gen_tokens", 0, "tokens")
    for (V, stem), vals in rng_.items():
        put(f"Range{V}{stem}Min", min(vals), s, f"min across models {V} {stem}", 1, "%")
        put(f"Range{V}{stem}Max", max(vals), s, f"max across models {V} {stem}", 1, "%")
    put("RangeZSExplicitAmongInvMinInt", min(rng_[("ZS", "ExplicitAmongInv")]), s, "min, whole per cent", 0, "%")
    put("RangeZSExplicitAmongInvMaxInt", max(rng_[("ZS", "ExplicitAmongInv")]), s, "max, whole per cent", 0, "%")
    s2 ="reports/stats/significance.md"
    t = src(s2).read_text(encoding="utf-8")
    m = grab(t, r"Rows at 505–510 tokens treated as not truncated: \{'llama31_8b': (\d+), 'mistral_nemo': (\d+), 'phi35_mini': (\d+), 'phi4_mini': (\d+)\}", "near-cap")
    for i, (_, M, _) in enumerate(MODELS):
        put(f"{M}ZSNearCapRows", int(m.group(i + 1)), s2, "A7 detail: rows at 505-510 tokens treated as not truncated", 0)

    section("stage 8 descriptive: A1b absolute error")
    ab = load_csv("reports/stats/a1b_descriptive.csv")
    s = "reports/stats/a1b_descriptive.csv"
    zs_m, ft_m = [], []
    for key, M, _ in MODELS:
        r = ab[ab["model"] == key].iloc[0]
        put(f"{M}AOneBBoth", int(r["n_both_extracted"]), s, f"{key}.n_both_extracted", 0)
        put(f"{M}AOneBMedZS", r["median_abs_err_zs"], s, f"{key}.median_abs_err_zs", 2, "mm")
        put(f"{M}AOneBMedFT", r["median_abs_err_ft"], s, f"{key}.median_abs_err_ft", 3, "mm")
        put(f"{M}AOneBDiff", signed(r["median_paired_diff_ft_minus_zs"], 2), s, f"{key}.median_paired_diff_ft_minus_zs")
        put(f"{M}AOneBDiffLo", signed(r["ci_low"], 2), s, f"{key}.ci_low")
        put(f"{M}AOneBDiffHi", signed(r["ci_high"], 2), s, f"{key}.ci_high")
        zs_m.append(r["median_abs_err_zs"]); ft_m.append(r["median_abs_err_ft"])
    put("RangeAOneBMedZSMin", min(zs_m), s, "min over models", 2, "mm")
    put("RangeAOneBMedZSMax", max(zs_m), s, "max over models", 2, "mm")
    put("RangeAOneBMedFTMin", min(ft_m), s, "min over models", 3, "mm")
    put("RangeAOneBMedFTMax", max(ft_m), s, "max over models", 3, "mm")

    section("stage 8 descriptive: clarify by language (clarify-trained arms)")
    ac = load_csv("reports/stats/a6_clarify_descriptive.csv")
    s = "reports/stats/a6_clarify_descriptive.csv"
    for key, M, _ in MODELS:
        for lang, Lg in (("bn", "Bn"), ("en", "En")):
            for metric, stem in (("clarify_exact", "ClarifyExact"), ("invented_value", "Invented")):
                r = ac[(ac["arm"] == f"{key}_seed42_clarify") & (ac["lang"] == lang) & (ac["metric"] == metric)].iloc[0]
                put(f"{M}CL{stem}{Lg}Rate", 100 * r["rate"], s, f"{key}_seed42_clarify/{lang}/{metric}.rate x100", 1, "%")
                put(f"{M}CL{stem}{Lg}RateLo", 100 * r["ci_low"], s, f"{key}_seed42_clarify/{lang}/{metric}.ci_low x100", 1, "%")
                put(f"{M}CL{stem}{Lg}RateHi", 100 * r["ci_high"], s, f"{key}_seed42_clarify/{lang}/{metric}.ci_high x100", 1, "%")
    put("ClarifyLangN", int(ac.iloc[0]["n"]), s, "n per language", 0)

    section("stage 8 descriptive: metric validation (B7)")
    dv = load_csv("reports/stats/dice_validation.csv")
    s = "reports/stats/dice_validation.csv"
    MV = {"dice_word": "DiceWord", "rouge_l": "RougeL", "chrf_pp": "Chrf", "bertscore_f1": "BertF",
          "dice_char3": "DiceCharThree", "dice_char5": "DiceCharFive"}
    SV = {"point_biserial_r_vs_farmer_yes": "Rpb", "roc_auc_predicting_farmer_yes": "Auc", "spearman_rho_vs_mean_rating": "Rho"}
    for _, r in dv.iterrows():
        stem = f"Val{MV[r['metric']]}{SV[r['statistic']]}"
        put(stem, r["estimate"], s, f"{r['metric']}.{r['statistic']}.estimate", 3)
        put(stem + "Lo", r["ci_low"], s, f"{r['metric']}.{r['statistic']}.ci_low", 3)
        put(stem + "Hi", r["ci_high"], s, f"{r['metric']}.{r['statistic']}.ci_high", 3)
    put("ValN", int(dv.iloc[0]["n"]), s, "n judgments", 0)

    section("stage 8 descriptive: human acceptance by distance from gold (B8)")
    b8 = load_csv("reports/stats/b8_error_bins.csv")
    s = "reports/stats/b8_error_bins.csv"
    BN = {"exact": "Exact", "<=0.1": "PtOne", "<=1.0": "One", ">1.0": "Over"}
    for _, r in b8.iterrows():
        B = "BinB" + BN[r["bin_abs_err_mm"]]
        put(B + "N", int(r["n"]), s, f"{r['bin_abs_err_mm']}.n", 0)
        put(B + "Mean", r["mean_rating"], s, f"{r['bin_abs_err_mm']}.mean_rating", 2)
        put(B + "MeanLo", r["mean_rating_lo"], s, f"{r['bin_abs_err_mm']}.mean_rating_lo", 2)
        put(B + "MeanHi", r["mean_rating_hi"], s, f"{r['bin_abs_err_mm']}.mean_rating_hi", 2)
        put(B + "Yes", 100 * r["farmer_yes_rate"], s, f"{r['bin_abs_err_mm']}.farmer_yes_rate x100", 0, "%")
        put(B + "YesLo", 100 * r["farmer_yes_lo"], s, f"{r['bin_abs_err_mm']}.farmer_yes_lo x100", 0, "%")
        put(B + "YesHi", 100 * r["farmer_yes_hi"], s, f"{r['bin_abs_err_mm']}.farmer_yes_hi x100", 0, "%")
    put("BinBTotal", int(b8["n"].sum()), s, "sum(n)", 0)

    section("stage 8 descriptive: response language (exploratory)")
    rl = load_csv("reports/stats/response_language.csv")
    s = "reports/stats/response_language.csv"
    SUB = {"all_bn_questions": "All", "numeric_runtime": "Numeric", "factual": "Factual", "clarify": "Clarify", "abstain": "Abstain"}
    tuned = []
    for key, M, _ in MODELS:
        for arm, V in ((f"{key}_zeroshot", "ZS"), (f"{key}_seed42", "FTSeedA"), (f"{key}_seed42_clarify", "CL")):
            for _, r in rl[rl["arm"] == arm].iterrows():
                Sb = SUB[r["subset"]]
                put(f"Lang{M}{V}{Sb}", 100 * r["share_answered_in_bengali"], s, f"{arm}/{r['subset']}.share x100", 1, "%")
                put(f"Lang{M}{V}{Sb}Lo", 100 * r["ci_low"], s, f"{arm}/{r['subset']}.ci_low x100", 1, "%")
                put(f"Lang{M}{V}{Sb}Hi", 100 * r["ci_high"], s, f"{arm}/{r['subset']}.ci_high x100", 1, "%")
                if Sb == "All":
                    put(f"Lang{M}{V}AllN", int(r["n_bn_questions"]), s, f"{arm}/all.n_bn_questions", 0)
                    if V != "ZS":
                        tuned.append(100 * r["share_answered_in_bengali"])
    put("RangeLangTunedMin", min(tuned), s, "min share answered in Bengali over FT and CL arms (all Bengali questions)", 1, "%")

    section("stage 8 descriptive: human arm summary (primary 182 and all 200)")
    hs = load_csv("reports/stats/human_arm_summary.csv")
    s = "reports/stats/human_arm_summary.csv"
    HM = {"mean_rating": ("Rating", 1, 2), "farmer_yes": ("Yes", 100, 1), "acceptable": ("Good", 100, 1)}
    QQ = {"numeric_runtime": "Numeric", "factual": "Factual", "clarify": "Clarify", "abstain": "Abstain"}
    for _, r in hs.iterrows():
        A = dict(ARMS_H)[r["arm"]]
        smp = "Test" if r["sample"] == "test" else "All"
        meas, _, q = r["measure"].partition("/")
        stem, k, nd = HM[meas]
        name = f"Hum{smp}{A[1:]}{QQ.get(q, '')}{stem}"
        put(name, r["estimate"] * k, s, f"{r['sample']}/{r['arm']}/{r['measure']}.estimate x{k}", nd)
        put(name + "Lo", r["ci_low"] * k, s, f"{r['sample']}/{r['arm']}/{r['measure']}.ci_low x{k}", nd)
        put(name + "Hi", r["ci_high"] * k, s, f"{r['sample']}/{r['arm']}/{r['measure']}.ci_high x{k}", nd)
        if not q:
            if f"Hum{smp}{A[1:]}N" not in MACROS:
                put(f"Hum{smp}{A[1:]}N", int(r["n"]), s, f"{r['sample']}/{r['arm']}.n", 0)
        else:
            if f"Hum{smp}{A[1:]}{QQ[q]}N" not in MACROS:
                put(f"Hum{smp}{A[1:]}{QQ[q]}N", int(r["n"]), s, f"{r['sample']}/{r['arm']}/{q}.n", 0)

    section("stage 8 descriptive: agreement with CIs")
    ag = load_json("reports/stats/agreement.json")
    s = "reports/stats/agreement.json"
    AG = {"fleiss_kappa_nominal_1to5": "KappaNominal", "kripp_alpha_ordinal": "AlphaOrdinal",
          "kripp_alpha_interval": "AlphaInterval", "fleiss_kappa_bad_mid_good": "KappaBands",
          "fleiss_kappa_farmer": "KappaFarmer", "kripp_alpha_farmer_nominal": "AlphaFarmer"}
    for k, stem in AG.items():
        c = ag["coefficients"][k]
        put(f"Agr{stem}", c["estimate"], s, f"coefficients.{k}.estimate", 3)
        put(f"Agr{stem}Lo", c["ci_low"], s, f"coefficients.{k}.ci_low", 3)
        put(f"Agr{stem}Hi", c["ci_high"], s, f"coefficients.{k}.ci_high", 3)
    assert MACROS["AgrKappaFarmer"]["value"] == MACROS["AgreeFarmerKappa"]["value"]
    assert MACROS["AgrAlphaOrdinal"]["value"] == MACROS["AgreeRatingOrdAlpha"]["value"]

    section("stage 8 descriptive: Bengali hedge check (exploratory)")
    hc = load_json("reports/stats/bn_hedge_check.json")
    s = "reports/stats/bn_hedge_check.json"
    put("HedgeRegisteredMarkers", len(hc["registered_bn_markers"]), s, "len(registered_bn_markers)", 0)
    put("HedgeUnlistedCandidates", len(hc["unlisted_candidates"]), s, "len(unlisted_candidates)", 0)
    put("HedgeRowsPerArm", hc["n_bn_clarify_rows"], s, "n_bn_clarify_rows", 0)
    ft_hedges = 0
    for key, M, _ in MODELS:
        for arm, V in ((f"{key}_zeroshot", "ZS"), (f"{key}_seed42", "FTSeedA")):
            d = hc["arms"][arm]
            put(f"Hedge{M}{V}BnScript", d["n_outputs_in_bengali_script"], s, f"arms.{arm}.n_outputs_in_bengali_script", 0)
            put(f"Hedge{M}{V}Registered", d["registered_bn_marker_hits"], s, f"arms.{arm}.registered_bn_marker_hits", 0)
            put(f"Hedge{M}{V}Unlisted", d["unlisted_hedge_hits"], s, f"arms.{arm}.unlisted_hedge_hits", 0)
            put(f"Hedge{M}{V}StrictReg", 100 * d["confident_error_b_strict_registered_rate_bn"], s,
                f"arms.{arm}.confident_error_b_strict_registered_rate_bn x100", 0, "%")
            put(f"Hedge{M}{V}StrictExt", 100 * d["confident_error_b_strict_extended_rate_bn_EXPLORATORY"], s,
                f"arms.{arm}.confident_error_b_strict_extended_rate_bn_EXPLORATORY x100", 0, "%")
            if V != "ZS":
                ft_hedges += d["any_hedge_registered_or_unlisted"]
                put(f"Hedge{M}{V}HandRead", len(d.get("hand_read_sample", [])), s, f"arms.{arm}.len(hand_read_sample)", 0)
    put("HedgeFTAny", ft_hedges, s, "sum over FT arms of any_hedge_registered_or_unlisted", 0)
    put("HedgeFTOutputs", 4 * hc["n_bn_clarify_rows"], s, "4 FT arms x n_bn_clarify_rows", 0)

    section("stage 8 environment")
    s = "reports/stats/env_freeze.txt"
    env = dict(l.split("==", 1) for l in src(s).read_text(encoding="utf-8").split() if "==" in l)
    for pkg, name in (("torch", "VerTorch"), ("transformers", "VerTransformers"), ("unsloth", "VerUnsloth"),
                      ("trl", "VerTrl"), ("peft", "VerPeft"), ("bitsandbytes", "VerBitsandbytes"), ("numpy", "VerNumpy"),
                      ("scipy", "VerScipy"), ("statsmodels", "VerStatsmodels"), ("scikit-learn", "VerSklearn"),
                      ("pandas", "VerPandas"), ("krippendorff", "VerKrippendorff"), ("sacrebleu", "VerSacrebleu"),
                      ("bert-score", "VerBertScore"), ("sentence-transformers", "VerSentenceTransformers"),
                      ("faiss-cpu", "VerFaiss"), ("matplotlib", "VerMatplotlib"), ("tokenizers", "VerTokenizers")):
        put(name, env[pkg].split("+")[0], s, f"{pkg} (local version suffix dropped)")
    put("VerCuda", re.search(r"\+cu(\d+)", env["torch"]).group(1)[:2] + "." + re.search(r"\+cu(\d+)", env["torch"]).group(1)[2:], s,
        "CUDA build of torch (+cuXYZ -> X.YZ)")
    put("VerPython", grab(t, r"Python (\d+\.\d+\.\d+)", "python").group(1), "reports/stats/significance.md", "environment line")
    put("StatsReportDate", grab(t, r"\*Generated (\d{4}-\d{2}-\d{2})", "gen").group(1), s2, "report header date")


# ===========================================================================
# 13b. Calculator constants and crop-coefficient table (inputs, not training data)
# ===========================================================================
def calculator_numbers() -> None:
    section("irrigation calculator constants")
    s = "scripts/irrigation_calc.py"
    code = src(s).read_text(encoding="utf-8")
    m = re.findall(r"if mm <= (\d+):\s*\n\s*return round\(mm \* ([0-9.]+), 2\)", code)
    last = grab(code, r"return round\(mm \* ([0-9.]+), 2\)\s*\n\s*\n\s*\ndef rain_tier", "tier 3").group(1)
    (b1, f1), (b2, f2) = m[0], m[1]
    put("RainTierOneMax", int(b1), s, "effective_rain: first branch upper bound (mm)", 0, "mm")
    put("RainTierTwoMax", int(b2), s, "effective_rain: second branch upper bound (mm)", 0, "mm")
    put("RainFactorOne", float(f1), s, "effective_rain: factor for rain <= first bound", 2)
    put("RainFactorTwo", float(f2), s, "effective_rain: factor for rain <= second bound", 2)
    put("RainFactorThree", float(last), s, "effective_rain: factor above second bound", 2)
    area = dict(re.findall(r'"(\w+)": ([0-9.]+),', grab(code, r"AREA = \{(.*?)\n\}", "AREA", re.S).group(1)))
    for unit, name in (("acre", "AreaAcre"), ("bigha", "AreaBigha"), ("katha", "AreaKatha"), ("decimal", "AreaDecimal"),
                       ("hectare", "AreaHectare")):
        put(name, float(area[unit]), s, f"AREA['{unit}'] (m2)", 4 if unit == "decimal" else (2 if unit != "hectare" else 0), "m2")
    put("AreaUnits", len({k for k in area if k != "ha"}), s, "distinct area units in AREA (ha alias excluded)", 0)
    ib = dict((k, (float(a), float(b))) for k, a, b in re.findall(r'"(\w+)": \(([0-9.]+), ([0-9.]+)\)', grab(code, r"INPUT_BOUNDS = \{(.*?)\n\}", "bounds", re.S).group(1)))
    put("BoundEtMax", ib["et0_mm"][1], s, "INPUT_BOUNDS.et0_mm upper", 0, "mm")
    put("BoundRainMax", ib["rainfall_mm"][1], s, "INPUT_BOUNDS.rainfall_mm upper", 0, "mm")
    put("KcStages", len(re.findall(r'^\s+"[\w-]+": "kc_\w+",', grab(code, r"STAGE_TO_KC = \{(.*?)\}", "stages", re.S).group(1), re.M)),
        s, "len(STAGE_TO_KC)", 0)

    section("project history: v2 held-out evaluation (the only surviving v2 result)")
    v2 = ROOT.parent / "reports" / "v2_numeric_eval" / "v2_numeric_eval_summary.txt"
    t2 = v2.read_text(encoding="utf-8", errors="replace")
    s2 = rel(v2)
    put("VTwoEvalRows", int(grab(t2, r"numeric_runtime rows: (\d+)", "v2 rows").group(1)), s2, "numeric_runtime rows", 0)
    put("VTwoMacroFone", float(grab(t2, r"macro-F1 across 6 slots: ([0-9.]+)", "v2 f1").group(1)), s2, "macro-F1 across 6 slots", 3)
    put("VTwoCropAcc", float(grab(t2, r"\ncrop\s+([0-9.]+)%", "v2 crop").group(1)), s2, "crop slot accuracy", 1, "%")
    put("VTwoMaeMm", float(grab(t2, r"MAE_mm  \(full, n=\d+\): ([0-9.]+)", "v2 mae").group(1)), s2, "MAE_mm (full)", 3, "mm")
    put("VTwoZeroGoldPct", float(grab(t2, r"floored to 0mm: \d+/\d+ \(([0-9.]+)%\)", "v2 zero").group(1)), s2,
        "fraction of gold rows floored to 0 mm", 1, "%")

    section("crop-coefficient table (input file)")
    kc_path = ROOT.parent / "kc_values_hybrid.csv"
    kc = pd.read_csv(kc_path)
    k = rel(kc_path)
    put("KcRows", len(kc), k, "rows", 0)
    put("KcRowsFao", int(kc["kc_source"].str.startswith("FAO-56").sum()), k, "kc_source starts with 'FAO-56'", 0)
    put("KcRowsIslam", int(kc["kc_source"].str.startswith("Islam").sum()), k, "kc_source starts with 'Islam'", 0)
    put("KcCitationDoorenbos", int(kc["citation"].str.contains("Doorenbos").sum()), k, "citation names Doorenbos & Pruitt 1977", 0)
    put("KcFaoRowsCitingDoorenbos", int((kc["kc_source"].str.startswith("FAO-56") & kc["citation"].str.contains("Doorenbos")).sum()),
        k, "rows labelled FAO-56 whose citation is Doorenbos & Pruitt 1977", 0)


# ===========================================================================
# 14. Project timeline (dates recorded by the pipeline itself)
# ===========================================================================
def timeline_numbers() -> None:
    import datetime as dt
    import glob
    import os
    section("project timeline (from pipeline timestamps)")

    def day(iso: str) -> str:
        return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%-d %B %Y" if os.name != "nt" else "%#d %B %Y")

    def mday(path: str) -> str:
        return dt.datetime.fromtimestamp(os.path.getmtime(ROOT / path)).strftime("%#d %B %Y" if os.name == "nt" else "%-d %B %Y")
    for f, name, fld in (("reports/gate0_seed_corpus.json", "DateStageZero", "generated_at"),
                         ("reports/gate1_split.json", "DateStageOne", "generated_at"),
                         ("reports/gate2_augment.json", "DateStageTwo", "generated_at"),
                         ("reports/leakage/leakage_report.json", "DateStageThree", "generated_at"),
                         ("reports/gate4_hardened.json", "DateStageFour", "generated_at"),
                         ("reports/s6b/merge_report.json", "DateStageSixB", "timestamp")):
        put(name, day(load_json(f)[fld]), f, fld)
    ts = sorted(load_json(p)["timestamp"] for p in sorted(glob.glob(str(ROOT / "reports/s5/*_seed*.json"))))
    put("DateStageFiveFirst", day(ts[0]), "reports/s5/*_seed*.json", "min(timestamp)")
    put("DateStageFiveLast", day(ts[-1]), "reports/s5/*_seed*.json", "max(timestamp)")
    put("DateStageSix", mday("reports/s6/summary_by_arm.csv"), "reports/s6/summary_by_arm.csv", "file modified time")
    put("DateStageSevenSheet", mday("reports/s7/stage7_rating_sheet.xlsx"), "reports/s7/stage7_rating_sheet.xlsx", "file modified time")
    put("DateStageSevenMerged", mday("reports/human_eval/stage7_ratings_merged.csv"), "reports/human_eval/stage7_ratings_merged.csv", "file modified time")
    t = src("reports/stats/significance.md").read_text(encoding="utf-8")
    put("DateStageEight", day(grab(t, r"\*Generated (\d{4}-\d{2}-\d{2})", "gen").group(1)), "reports/stats/significance.md", "report header")


# ===========================================================================
def write_outputs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "% AUTO-GENERATED by scripts/thesis/make_numbers.py. Do not edit by hand.",
        "% Every macro below names its source file and field. Requires \\usepackage{xspace}.",
        "\\providecommand{\\MISSING}[2]{\\textbf{[MISSING: #1; looked in #2]}}",
        "\\providecommand{\\CITENEEDED}[1]{\\textbf{[CITATION NEEDED: #1]}}",
        "\\providecommand{\\PENDINGREVIEW}[1]{\\textbf{[PENDING REVIEW: #1]}}",
        "\\providecommand{\\PENDINGCONSENT}[1]{\\textbf{[PENDING: signed consent forms#1]}}",
        "",
    ]
    cur = None
    for name, d in MACROS.items():
        if d["section"] != cur:
            cur = d["section"]
            lines += ["", f"% ---- {cur} ----"]
        val = d["value"]
        if not d["missing"] and not val.startswith("\\"):
            val = val.replace("%", "\\%").replace("_", "\\_") if not re.fullmatch(r"-?[0-9.,]+", val) else val
        # \xspace keeps the following space in prose ("\TestClarify clarify rows").
        lines.append(f"\\newcommand{{\\{name}}}{{{val}\\xspace}} % {d['source']} :: {d['field']}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    js = {k: {kk: vv for kk, vv in v.items()} for k, v in MACROS.items()}
    (OUT / "numbers.json").write_text(json.dumps(js, ensure_ascii=False, indent=1, default=str) + "\n",
                                      encoding="utf-8", newline="\n")

    md = ["# Fact sheet", "",
          "Generated by `scripts/thesis/make_numbers.py`. One row per LaTeX macro in `numbers.tex`.",
          "Seeds in macro names: SeedA = 42, SeedB = 1337, SeedC = 2024. Variants: ZS zero-shot, FT fine-tuned "
          "(mean of three seeds unless a Seed suffix is present; SD = sample standard deviation), CL clarify-trained "
          "(seed 42 only).", "",
          f"Total macros: {len(MACROS)}. Missing: {sum(d['missing'] for d in MACROS.values())}.", ""]
    cur = None
    for name, d in MACROS.items():
        if d["section"] != cur:
            cur = d["section"]
            md += ["", f"## {cur}", "", "| macro | value | unit | source | field |", "|---|---|---|---|---|"]
        v = d["value"].replace("|", "\\|")
        md.append(f"| `\\{name}` | {v} | {d['unit']} | `{d['source']}` | {d['field'].replace('|', '/')} |")
    (OUT / "facts.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    test_set_gate()          # must run before any prediction is read
    data_numbers()
    leakage_numbers()
    hardening_numbers()
    training_numbers()
    stage6_numbers()
    confusion_numbers()
    clarify_numbers()
    output_form_numbers()
    human_numbers()
    plan_numbers()
    change_numbers()
    tokenizer_numbers()
    verification_numbers()
    stage8_numbers()
    calculator_numbers()
    timeline_numbers()
    write_outputs()
    n_miss = sum(d["missing"] for d in MACROS.values())
    print(f"wrote {len(MACROS)} macros ({n_miss} missing) -> {rel(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
