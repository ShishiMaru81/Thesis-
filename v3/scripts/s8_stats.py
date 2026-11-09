#!/usr/bin/env python3
"""Stage 8 — statistical significance for Stage 6 (automatic) and Stage 7 (human).

Read-only on all upstream artefacts. Writes only under reports/stats/ and logs/.

    python scripts/s8_stats.py --part gate     # entry gate only (step 1)
    python scripts/s8_stats.py --part plan     # write test_plan.json (step 2)
    python scripts/s8_stats.py --part A        # automatic metrics
    python scripts/s8_stats.py --part B        # human evaluation
    python scripts/s8_stats.py --part report   # significance.md
    python scripts/s8_stats.py --part all
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

SEED = 42
N_BOOT = 10_000

TEST_PATH = PROJECT / "data" / "hardened" / "test_hardened.jsonl"
TEST_SHA_PATH = PROJECT / "data" / "hardened" / "test_hardened.sha256"
S6_DIR = PROJECT / "reports" / "s6"
S6_MASTER = S6_DIR / "results_master.csv"
S6_PRED = S6_DIR / "predictions"
S6_CONF = S6_DIR / "confusion_matrices.json"
HE_DIR = PROJECT / "reports" / "human_eval"
RATINGS = HE_DIR / "stage7_ratings_merged.csv"
# Spec lists these under reports/human_eval/. Actual locations resolved in gate.
ARM_KEY_CANDIDATES = [HE_DIR / "stage7_arm_key_DO_NOT_SHARE.csv",
                      PROJECT / "reports" / "s7" / "stage7_arm_key_DO_NOT_SHARE.csv"]
ASSIGN_KEY_CANDIDATES = [HE_DIR / "stage7_assignment_key.csv",
                         PROJECT / "reports" / "s7" / "stage7_assignment_key.csv"]
OUT_DIR = PROJECT / "reports" / "stats"
LOG_PATH = PROJECT / "logs" / "s8_stats.log"

HUMAN_ARMS = ["llama31_8b_zeroshot", "llama31_8b_seed42",
              "llama31_8b_seed42_clarify", "mistral_nemo_seed42"]
MODELS = ["llama31_8b", "mistral_nemo", "phi35_mini", "phi4_mini"]
SEEDS = [42, 1337, 2024]
DETECTOR_ARM = "llama31_8b_seed42_clarify_with_detector"

# Spec metric name -> results_master `metric` value (None = not in master)
METRIC_MAP = {
    "exact_0.1": "em_0p1", "exact_1.0": "em_1p0", "abs_error": "abs_err",
    "net_extracted": "net_extracted", "predicted_class": None,
    "invented_value": "invented_value", "dice_word": "dice_word",
    "dice_char3": "dice_char3", "dice_char5": "dice_char5", "chrF++": "chrf_pp",
    "ROUGE-L": "rouge_l", "BERTScore": "bertscore_f1",
    "exact_clarify": "clarify_exact", "abstain_correct": "abstain_correct",
}

log = logging.getLogger("s8")


# ---------------------------------------------------------------------------
# read guard: never open training / validation corpora
# ---------------------------------------------------------------------------
_FORBIDDEN_PREFIXES = ("train_augmented", "train.", "val.")
_real_open = builtins.open
_real_path_open = Path.open


def _check(path) -> None:
    try:
        name = os.path.basename(os.fspath(path))
    except (TypeError, ValueError):
        return
    if name.startswith(_FORBIDDEN_PREFIXES) and name.endswith(".jsonl"):
        raise RuntimeError(f"FORBIDDEN READ in s8_stats: {name}")


def _g_open(f, *a, **k):
    _check(f)
    return _real_open(f, *a, **k)


def _g_popen(self, *a, **k):
    _check(self)
    return _real_path_open(self, *a, **k)


builtins.open = _g_open
Path.open = _g_popen


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with _real_open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def first_existing(cands: list[Path]) -> Path | None:
    return next((p for p in cands if p.exists()), None)


class GateFail(SystemExit):
    pass


def check(cond: bool, msg: str) -> None:
    if cond:
        log.info("  PASS  %s", msg)
    else:
        log.error("  FAIL  %s", msg)
        raise GateFail(f"ENTRY GATE FAILED: {msg}")


# ---------------------------------------------------------------------------
# ENTRY GATE
# ---------------------------------------------------------------------------
def entry_gate() -> dict:
    g: dict = {}
    log.info("=" * 78)
    log.info("STAGE 8 ENTRY GATE")
    log.info("=" * 78)

    # ---- 1. test SHA
    log.info("[1] test set integrity")
    expected = TEST_SHA_PATH.read_text(encoding="utf-8").split()[0]
    actual = sha256_of(TEST_PATH)
    check(expected == actual, f"test_hardened.jsonl sha256 {actual[:16]}... matches .sha256")
    test_rows = load_jsonl(TEST_PATH)
    pending = [r for r in test_rows if r.get("gold_status") == "pending_human"]
    scorable = [r for r in test_rows if r.get("gold_status") != "pending_human"]
    check(len(test_rows) == 422, f"422 test rows (got {len(test_rows)})")
    check(len(scorable) == 362 and len(pending) == 60,
          f"362 scorable / 60 pending_human (got {len(scorable)}/{len(pending)})")
    g["pending_by_category"] = dict(Counter(r["category"] for r in pending))
    g["pending_by_qtype"] = dict(Counter(r["qtype"] for r in pending))
    g["scorable_by_qtype"] = dict(Counter(r["qtype"] for r in scorable))
    g["scorable_by_qtype_lang"] = {f"{q}/{l}": n for (q, l), n in sorted(
        Counter((r["qtype"], r["lang"]) for r in scorable).items())}
    log.info("      pending_human by category: %s", g["pending_by_category"])
    log.info("      pending_human by qtype:    %s", g["pending_by_qtype"])
    log.info("      scorable by qtype:         %s", g["scorable_by_qtype"])
    log.info("      scorable by qtype/lang:    %s", g["scorable_by_qtype_lang"])
    scorable_ids = {r["row_id"] for r in scorable}
    # bn/en pairing key for family A6. Spec says pair_id; the locked test set has
    # no such column, so probe candidates and report what each yields.
    test_keys = set().union(*[r.keys() for r in test_rows])
    g["test_columns"] = sorted(test_keys)
    pair_key_results = {}
    for k in ["pair_id", "fact_id", "source_id", "scenario_hash"]:
        if k not in test_keys:
            pair_key_results[k] = "column absent"
            continue
        pairs = defaultdict(list)
        for r in scorable:
            if r.get(k):
                pairs[(r["qtype"], r[k])].append(r["lang"])
        pair_key_results[k] = dict(Counter(
            q for (q, _), langs in pairs.items() if sorted(langs) == ["bn", "en"]))
    g["bn_en_pairing_key_options"] = pair_key_results
    for k, v in pair_key_results.items():
        log.info("      A6 pairing via %-14s -> valid bn/en pairs by qtype: %s", k, v)
    if "pair_id" not in test_keys:
        log.warning("      test set has NO pair_id column. Family A6 will pair on fact_id "
                    "(== source_id). Clarify rows have 0 pairs -> A6 clarify not testable; "
                    "numeric has only 17 pairs.")
        g["a6_pairing_key"] = "fact_id"
    else:
        g["a6_pairing_key"] = "pair_id"

    # ---- 2. ratings
    log.info("[2] stage7_ratings_merged.csv")
    check(RATINGS.exists(), f"exists: {RATINGS.relative_to(PROJECT)}")
    r = pd.read_csv(RATINGS, dtype=str, keep_default_na=False)
    check(len(r) == 1200, f"1,200 rows (got {len(r)})")
    rating = pd.to_numeric(r["rating_1_5"], errors="coerce")
    check(rating.notna().all() and rating.between(1, 5).all() and (rating % 1 == 0).all(),
          "every rating_1_5 is an integer in 1..5")
    farmer = r["would_give_farmer"].str.strip().str.lower()
    check(set(farmer.unique()) <= {"yes", "no"}, f"would_give_farmer in {{yes,no}} (got {sorted(farmer.unique())})")
    check((r["audit_flag"].str.strip() == "").all(), "audit_flag empty on all rows")
    bad = ((rating <= 2) & (farmer == "yes")).sum()
    check(bad == 0, f"no row with rating<=2 AND farmer==yes (got {bad})")
    g["annotators"] = dict(Counter(r["annotator"]))
    g["is_overlap"] = dict(Counter(r["is_overlap"]))
    ov = r[r["is_overlap"] == "yes"]
    solo = r[r["is_overlap"] == "no"]
    ov_j = ov.groupby("judgment_id")["annotator"].nunique()
    check(len(ov_j) == 100 and (ov_j == 5).all(),
          f"100 overlap judgments each rated by 5 annotators (got {len(ov_j)}, min raters {ov_j.min() if len(ov_j) else 0})")
    solo_j = solo.groupby("judgment_id")["annotator"].nunique()
    check(len(solo_j) == 700 and (solo_j == 1).all(),
          f"700 solo judgments each rated by exactly 1 annotator (got {len(solo_j)})")
    check(len(set(ov_j.index) | set(solo_j.index)) == 800, "overlap + solo cover 800 distinct judgment_ids")
    g["solo_arm_by_annotator"] = {a: dict(Counter(d["arm"])) for a, d in solo.groupby("annotator")}
    log.info("      annotators: %s   overlap: %s", g["annotators"], g["is_overlap"])
    log.info("      solo rows arm x annotator (severity confound, family B4):")
    for a, d in g["solo_arm_by_annotator"].items():
        log.info("        %s: %s", a, d)

    # ---- 3. arm key
    log.info("[3] arm key")
    key_path = first_existing(ARM_KEY_CANDIDATES)
    check(key_path is not None, "arm key file found")
    if key_path != ARM_KEY_CANDIDATES[0]:
        log.warning("      arm key NOT at spec path %s; using %s",
                    ARM_KEY_CANDIDATES[0].relative_to(PROJECT), key_path.relative_to(PROJECT))
    g["arm_key_path"] = str(key_path.relative_to(PROJECT))
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False)
    check(len(key) == 800 and key["judgment_id"].nunique() == 800, f"800 unique judgment_ids (got {len(key)})")
    arm_counts = Counter(key["arm"])
    check(set(arm_counts) == set(HUMAN_ARMS), f"4 arms == {HUMAN_ARMS} (got {sorted(arm_counts)})")
    check(all(v == 200 for v in arm_counts.values()), f"200 per arm (got {dict(arm_counts)})")
    per_arm_rows = key.groupby("arm")["row_id"].apply(lambda s: sorted(s))
    ref = per_arm_rows.iloc[0]
    check(len(set(ref)) == 200 and all(list(v) == list(ref) for v in per_arm_rows),
          "each of 200 row_ids appears exactly once per arm")
    # 18 authored-BN rows are not test rows; they were generated in Stage 7.
    key_ids = set(key["row_id"])
    in_test = key_ids & scorable_ids
    not_in_test = key_ids - {r["row_id"] for r in test_rows}
    g["human_rows_in_scorable_test"] = len(in_test)
    g["human_rows_not_in_test_set"] = len(not_in_test)
    check(len(in_test) + len(not_in_test) == 200 and not (key_ids & {r["row_id"] for r in pending}),
          f"human rows: {len(in_test)} scorable test rows + {len(not_in_test)} authored-BN (non-test) rows; 0 pending_human")
    # ratings <-> key consistency
    m = r.merge(key[["judgment_id", "arm", "row_id"]], on="judgment_id", suffixes=("", "_key"))
    check(len(m) == 1200 and (m["arm"] == m["arm_key"]).all() and (m["row_id"] == m["row_id_key"]).all(),
          "ratings_merged arm/row_id agree with arm key on all 1,200 rows")
    assign_path = first_existing(ASSIGN_KEY_CANDIDATES)
    if assign_path is None:
        log.warning("      stage7_assignment_key.csv NOT FOUND at any candidate path. "
                    "Assignment (annotator, judgment_id, is_overlap) is embedded in "
                    "stage7_ratings_merged.csv and was verified above; that will be used instead.")
        g["assignment_key_path"] = None
    else:
        g["assignment_key_path"] = str(assign_path.relative_to(PROJECT))

    # ---- 4. human arm -> stage 6 prediction file
    log.info("[4] human arm -> Stage 6 prediction file")
    mapping = {}
    for arm in HUMAN_ARMS:
        p = S6_PRED / f"{arm}.jsonl"
        mapping[arm] = str(p.relative_to(PROJECT)) if p.exists() else None
        log.info("      %-28s -> %s", arm, mapping[arm] or "MISSING")
    check(all(mapping.values()), "every human arm resolves to a Stage 6 prediction file (exact name match)")
    g["human_arm_to_pred"] = mapping

    # ---- 5. results_master schema
    log.info("[5] results_master.csv schema")
    mst = pd.read_csv(S6_MASTER)
    g["master_columns"] = mst.columns.tolist()
    g["master_rows"] = len(mst)
    g["master_arms"] = sorted(mst["arm"].unique())
    g["master_metrics"] = sorted(mst["metric"].unique())
    log.info("      columns: %s", g["master_columns"])
    log.info("      rows: %d   arms: %d   metrics: %d", len(mst), len(g["master_arms"]), len(g["master_metrics"]))
    log.info("      arms: %s", g["master_arms"])
    log.info("      metrics: %s", g["master_metrics"])
    check({"arm", "row_id", "qtype", "lang", "metric", "value"} <= set(mst.columns),
          "master is long-format per row (arm,row_id,qtype,lang,metric,value)")
    per_arm_n = mst.groupby("arm")["row_id"].nunique()
    check((per_arm_n == 362).all(), f"362 distinct row_ids per arm in master (min {per_arm_n.min()}, max {per_arm_n.max()})")
    avail = {}
    for spec, col in METRIC_MAP.items():
        if col is None:
            avail[spec] = "NOT IN MASTER -> recompute per row from predictions via s6_eval.classify_response"
        elif col in g["master_metrics"]:
            n = mst.loc[mst["metric"] == col].groupby("arm").size().min()
            avail[spec] = f"per-row as '{col}' (>= {n} rows/arm)"
        else:
            avail[spec] = f"MISSING ('{col}' not in master)"
        log.info("      %-16s %s", spec, avail[spec])
    g["metric_availability"] = avail
    check(not any(v.startswith("MISSING") for v in avail.values()), "all required metrics present per row or recomputable")
    # confirm scorer functions import
    import s6_eval  # noqa: F401
    check(callable(s6_eval.classify_response), "s6_eval.classify_response importable for predicted-class recompute")
    # verify recompute reproduces confusion_matrices.json for one arm
    conf = json.load(S6_CONF.open(encoding="utf-8"))
    test_by_id = {r["row_id"]: r for r in test_rows}
    arm0 = "llama31_8b_seed42"
    preds0 = [p for p in load_jsonl(S6_PRED / f"{arm0}.jsonl") if p["row_id"] in scorable_ids]
    cm = Counter()
    for p in preds0:
        gq = s6_eval.GOLD_CLASS[test_by_id[p["row_id"]]["qtype"]]
        cm[(gq, s6_eval.classify_response(p["generation"] or ""))] += 1
    stored = conf[arm0]["overall"]
    ok = all(cm[(gq, pc)] == stored[gq][pc] for gq in stored for pc in stored[gq])
    check(ok, f"recomputed predicted-class confusion for {arm0} reproduces confusion_matrices.json")

    # ---- 6. per-row data for all arms
    log.info("[6] Stage 6 per-row coverage")
    expected_arms = ([f"{m}_zeroshot" for m in MODELS]
                     + [f"{m}_seed{s}" for m in MODELS for s in SEEDS]
                     + [f"{m}_seed42_clarify" for m in MODELS])
    missing_master = [a for a in expected_arms if a not in g["master_arms"]]
    missing_pred = [a for a in expected_arms + [DETECTOR_ARM] if not (S6_PRED / f"{a}.jsonl").exists()]
    log.info("      expected in master: %d arms (4 zs + 12 ft + 4 clarify)", len(expected_arms))
    log.info("      detector arm '%s': predictions %s, in master: %s",
             DETECTOR_ARM, "present" if (S6_PRED / f"{DETECTOR_ARM}.jsonl").exists() else "MISSING",
             DETECTOR_ARM in g["master_arms"])
    check(not missing_master, f"all 20 scored arms in master (missing: {missing_master})")
    check(not missing_pred, f"prediction files for all 21 arms incl. detector (missing: {missing_pred})")
    det = load_jsonl(S6_PRED / f"{DETECTOR_ARM}.jsonl")
    check(all("detector_flag" in p and "predicted_qtype" in p for p in det),
          "detector predictions carry per-row detector_flag + predicted_qtype")
    g["arms_in_master"] = len(g["master_arms"])
    g["arms_with_predictions"] = len(expected_arms) + 1
    extra = sorted(set(g["master_arms"]) - set(expected_arms))
    if extra:
        log.info("      extra arms in master not in plan: %s", extra)

    # ---- 7. toolchain
    log.info("[7] toolchain")
    import shutil
    g["R_available"] = shutil.which("Rscript") is not None
    log.info("      Rscript on PATH: %s -> family B4 method: %s", g["R_available"],
             "R ordinal::clmm" if g["R_available"] else "Python statsmodels fallback")
    pk = {}
    for mod in ["scipy", "statsmodels", "numpy", "sklearn", "krippendorff"]:
        try:
            m_ = __import__(mod)
            pk[mod] = getattr(m_, "__version__", "ok")
        except ImportError:
            pk[mod] = None
    g["python_packages"] = pk
    log.info("      python packages: %s", pk)
    check(pk["scipy"] and pk["statsmodels"] and pk["numpy"], "scipy + statsmodels + numpy importable")

    g["gate"] = "PASS"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "entry_gate.json").write_text(json.dumps(g, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("=" * 78)
    log.info("ENTRY GATE PASS -> %s", (OUT_DIR / "entry_gate.json").relative_to(PROJECT))
    log.info("=" * 78)
    return g


# ---------------------------------------------------------------------------
# STEP 2 / 0a + 0b — PRE-PLAN CHECKS
# ---------------------------------------------------------------------------
S7_DIR = PROJECT / "reports" / "s7"
S7_PRED_BN = S7_DIR / "predictions_translated_bn"
S7_SHEET = S7_DIR / "stage7_rating_sheet.csv"
SAMPLE_BN = PROJECT / "data" / "human_eval" / "bilingual_100" / "sample_100_bn.jsonl"
SAMPLE_EN = PROJECT / "data" / "human_eval" / "bilingual_100" / "sample_100_en.jsonl"
GEN_MAX_NEW_TOKENS = 512
TRUNC_TOKENS = GEN_MAX_NEW_TOKENS - 1   # >= 511 generated tokens == hit the cap


def human_frame() -> pd.DataFrame:
    """200 human-sample questions x metadata, with in_test_set flag."""
    key = pd.read_csv(first_existing(ARM_KEY_CANDIDATES), dtype=str, keep_default_na=False)
    test_ids = {r["row_id"] for r in load_jsonl(TEST_PATH)}
    smp = {r["row_id"]: r for r in load_jsonl(SAMPLE_BN) + load_jsonl(SAMPLE_EN)}
    q = key.drop_duplicates("row_id")[["row_id", "pair_id", "lang", "qtype"]].copy()
    q["category"] = q["row_id"].map(lambda i: smp[i]["category"])
    q["translation_status"] = q["row_id"].map(lambda i: smp[i].get("translation_status", ""))
    q["in_test_set"] = q["row_id"].isin(test_ids)
    return q.reset_index(drop=True)


def load_tokenizer(model_key: str):
    from transformers import AutoTokenizer
    import s6_eval
    mid = s6_eval.MODEL_REGISTRY[model_key]
    try:
        return AutoTokenizer.from_pretrained(mid, local_files_only=True)
    except Exception:           # noqa: BLE001
        return AutoTokenizer.from_pretrained(mid)


def gen_token_counts(arm: str, tok, scorable_ids: set[str]) -> pd.Series:
    preds = [p for p in load_jsonl(S6_PRED / f"{arm}.jsonl") if p["row_id"] in scorable_ids]
    n = [len(tok(p["generation"] or "", add_special_tokens=False)["input_ids"]) for p in preds]
    return pd.Series(n, index=[p["row_id"] for p in preds], name=arm)


def precheck() -> dict:
    out: dict = {}
    log.info("=" * 78)
    log.info("STEP 2 / CHECK 0a — the 18 authored-translation questions")
    log.info("=" * 78)
    hq = human_frame()
    check(hq["in_test_set"].sum() == 182 and (~hq["in_test_set"]).sum() == 18,
          f"in_test_set: {hq['in_test_set'].sum()} True / {(~hq['in_test_set']).sum()} False")
    auth = hq[~hq["in_test_set"]]
    out["authored_18"] = {
        "by_qtype": dict(Counter(auth["qtype"])),
        "by_category": dict(Counter(auth["category"])),
        "by_lang": dict(Counter(auth["lang"])),
        "by_translation_status": dict(Counter(auth["translation_status"])),
        "by_qtype_category": {f"{q}/{c}": n for (q, c), n in sorted(Counter(zip(auth["qtype"], auth["category"])).items())},
    }
    for k, v in out["authored_18"].items():
        log.info("      %-22s %s", k, v)
    exp = {"clarify": 15, "numeric_runtime": 3}
    check(out["authored_18"]["by_qtype"] == exp, f"expected 15 clarify + 3 numeric, 0 factual/abstain (got {out['authored_18']['by_qtype']})")
    check(out["authored_18"]["by_lang"] == {"bn": 18}, "all 18 are bn")
    check(out["authored_18"]["by_translation_status"] == {"authored_bn_pair": 18}, "all 18 translation_status == authored_bn_pair")
    # human sample composition for B-family n's
    comp = hq.groupby(["qtype", "in_test_set"]).size().unstack(fill_value=0)
    out["human_sample_by_qtype"] = {q: {"in_test": int(comp.loc[q].get(True, 0)),
                                        "authored": int(comp.loc[q].get(False, 0)),
                                        "all": int(comp.loc[q].sum())} for q in comp.index}
    log.info("      human sample by qtype (test / authored / all): %s", out["human_sample_by_qtype"])

    # provenance of the 18x4 generations
    log.info("  provenance of the 4 arms' answers to the 18:")
    prov = {"script": "scripts/s7_gen_translated_bn.py",
            "inference_function": "s6_eval.generate_arm (imported; same prompt template, greedy, num_beams=1)",
            "predictions_dir": str(S7_PRED_BN.relative_to(PROJECT)),
            "log": "logs/s7_gen_translated_bn.log", "arms": {}}
    def read_log(p: Path) -> str:      # Tee-Object writes UTF-16LE with BOM; logging writes UTF-8
        b = p.read_bytes()
        return b.decode("utf-16") if b[:2] in (b"\xff\xfe", b"\xfe\xff") else b.decode("utf-8", errors="replace")
    s7_log = read_log(PROJECT / "logs" / "s7_gen_translated_bn.log")
    s6_log = read_log(PROJECT / "logs" / "s6_eval.log")
    import re
    all_ok = True
    for arm in HUMAN_ARMS:
        m7 = json.load((S7_PRED_BN / f"{arm}_meta.json").open(encoding="utf-8"))
        m6 = json.load((S6_DIR / f"{arm}_meta.json").open(encoding="utf-8"))
        pat = rf"loading {re.escape(arm)}  <- (.*)"
        a7 = re.search(pat, s7_log); a6 = re.search(pat, s6_log)
        ad7 = a7.group(1).strip() if a7 else None
        ad6 = a6.group(1).strip() if a6 else None
        # LoRA count logged right after the loading line
        def lora_after(txt, m):
            if not m: return None
            mm = re.search(r"adapter check OK: (\d+) LoRA", txt[m.end():m.end() + 20000])
            return int(mm.group(1)) if mm else None
        rec = {"stage6_adapter": ad6, "stage7_adapter": ad7,
               "stage6_n_lora": m6.get("n_lora_modules"), "stage7_n_lora": m7.get("n_lora_modules"),
               "stage6_n_lora_logged": lora_after(s6_log, a6), "stage7_n_lora_logged": lora_after(s7_log, a7),
               "stage6_max_new_tokens": m6.get("max_new_tokens"), "stage7_max_new_tokens": m7.get("max_new_tokens"),
               "stage7_rows": m7.get("rows")}
        rec["match"] = (ad6 == ad7 and rec["stage6_n_lora"] == rec["stage7_n_lora"]
                        and rec["stage6_max_new_tokens"] == rec["stage7_max_new_tokens"] == GEN_MAX_NEW_TOKENS
                        and rec["stage7_rows"] == 18)
        all_ok &= rec["match"]
        prov["arms"][arm] = rec
        log.info("      %-28s adapter=%s  lora=%s  max_new_tokens=%s  match=%s",
                 arm, ad7, rec["stage7_n_lora"], rec["stage7_max_new_tokens"], rec["match"])
    check(all_ok, "18-row generations used the same adapters, LoRA counts and max_new_tokens=512 as Stage 6")
    # what annotators saw == what those files contain
    sheet = pd.read_csv(S7_SHEET, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    key = pd.read_csv(first_existing(ARM_KEY_CANDIDATES), dtype=str, keep_default_na=False)
    sheet = sheet.merge(key[["judgment_id", "arm"]], on="judgment_id")
    mism = 0
    for arm in HUMAN_ARMS:
        g6 = {p["row_id"]: (p["generation"] or "") for p in load_jsonl(S6_PRED / f"{arm}.jsonl")}
        g7 = {p["row_id"]: (p["generation"] or "") for p in load_jsonl(S7_PRED_BN / f"{arm}.jsonl")}
        sub = sheet[sheet["arm"] == arm]
        for _, r in sub.iterrows():
            src = g7 if r["row_id"] in g7 else g6
            if src.get(r["row_id"]) != r["model_answer"]:
                mism += 1
    check(mism == 0, f"rating-sheet model_answer identical to the prediction files for all 800 judgments (mismatches={mism})")
    prov["rating_sheet_matches_predictions"] = mism == 0
    out["provenance_18"] = prov

    # ---- 0b truncation signal
    log.info("=" * 78)
    log.info("STEP 2 / CHECK 0b — truncation signal for Family A7")
    log.info("=" * 78)
    pk = set().union(*[p.keys() for p in load_jsonl(S6_PRED / "llama31_8b_zeroshot.jsonl")])
    has_fr = "finish_reason" in pk
    has_tc = any(k in pk for k in ("n_generated_tokens", "gen_tokens", "num_tokens"))
    log.info("      prediction keys: %s", sorted(pk))
    log.info("      finish_reason stored: %s   token count stored: %s", has_fr, has_tc)
    out["finish_reason_stored"] = has_fr
    out["token_count_stored"] = has_tc
    scorable_ids = {r["row_id"] for r in load_jsonl(TEST_PATH) if r["gold_status"] != "pending_human"}
    test_q = {r["row_id"]: r["qtype"] for r in load_jsonl(TEST_PATH)}
    tok = load_tokenizer("llama31_8b")
    out["tokenizer"] = {"llama31_8b": type(tok).__name__ + " / " + __import__("s6_eval").MODEL_REGISTRY["llama31_8b"]}
    dist = {}
    for arm in ["llama31_8b_zeroshot", "llama31_8b_seed42"]:
        s = gen_token_counts(arm, tok, scorable_ids)
        qt = pd.Series({i: test_q[i] for i in s.index})
        d = {"n": int(len(s)), "min": int(s.min()), "median": float(s.median()), "p90": float(s.quantile(.9)),
             "max": int(s.max()), "n_ge_511": int((s >= TRUNC_TOKENS).sum()),
             "pct_ge_511": round(100 * float((s >= TRUNC_TOKENS).mean()), 1),
             "clarify_rows_ge_511": int(((s >= TRUNC_TOKENS) & (qt == "clarify")).sum()),
             "clarify_rows_n": int((qt == "clarify").sum())}
        dist[arm] = d
        log.info("      %-22s %s", arm, d)
    out["token_count_distribution"] = dist
    check(dist["llama31_8b_zeroshot"]["max"] >= TRUNC_TOKENS or dist["llama31_8b_zeroshot"]["max"] > 0,
          "re-tokenisation works; truncation flag = generated tokens >= 511 (no finish_reason stored)")
    for mk in ["mistral_nemo", "phi35_mini", "phi4_mini"]:
        t = load_tokenizer(mk)
        out["tokenizer"][mk] = type(t).__name__ + " / " + __import__("s6_eval").MODEL_REGISTRY[mk]
    log.info("      all four tokenizers load from local cache: %s", list(out["tokenizer"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "precheck.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("precheck -> %s", (OUT_DIR / "precheck.json").relative_to(PROJECT))
    return out


# ---------------------------------------------------------------------------
# STEP 2 — TEST PLAN (pre-registration; no p-values computed here)
# ---------------------------------------------------------------------------
MODEL_LABEL = {"llama31_8b": "Llama-3.1-8B", "mistral_nemo": "Mistral-NeMo-12B",
               "phi35_mini": "Phi-3.5-mini", "phi4_mini": "Phi-4-mini"}
ROB_SEEDS = [1337, 2024]
PLAN_VERSION = 3
A3A4_SEED_NOTE = "Clarify-trained arms exist at seed 42 only; A3 and A4 have no seed-robustness check."
A5_CAVEAT = ("Oracle-gated diagnostic. The detector reads gold labels, runs only on known-incomplete queries, and "
             "its rules were written with the test categories in view. This is an upper bound on what surface "
             "rules can catch, not a performance estimate.")
EM_DENOMINATOR_NOTE = ("Two exact-match denominators are in use and are named wherever either figure appears: "
                       "Stage 8 correct_0p1 = all 146 numeric rows (multi-crop rows counted correct only if every "
                       "gold value is within 0.1 mm; rows with no extracted depth counted wrong); Stage 6 headline "
                       "exact_0.1 = the 116 single-crop rows. No test was changed; both are tabulated per arm.")
RESPONSE_LANG_NOTE = ("Exploratory descriptive added after Part A (plan v3), motivated by the A7 marker finding that "
                      "zero-shot answers to Bengali questions were written in English. Response language is "
                      "classified by script: bn if >= 50% of alphabetic characters are in the Bengali block "
                      "(U+0980-U+09FF), else en. No test, no Holm.")
CORRECT_0P1_DEF = ("correct_0p1 on the 146 numeric rows = em_0p1 for single-crop rows where a depth was extracted; "
                   "multi_all_correct (all gold values within 0.1 mm) for the 30 multi-crop rows; 0 when no depth was "
                   "extracted. Both use the 0.1 mm tolerance from s6_eval.score_row.")
CHANGELOG = [
    {"version": 1, "date": "2026-09-23", "note": "initial pre-registration (176 tests)"},
    {"version": 2, "date": "2026-09-23", "changes": [
        "A3a/A3b/A4: removed seed-1337/2024 robustness cells (FT seed s vs clarify seed 42 confounds seed with treatment). "
        "Plan-level note added: '" + A3A4_SEED_NOTE + "' Carried into significance.md as a limitation.",
        "A5: effect size changed from odds ratio to risk difference in safe handling (hybrid minus model-only) with "
        "paired-bootstrap 95% CI; McNemar exact on b retained. In-sample caveat attached to A5 and to every detector "
        "mention in significance.md; detector rules and the test categories that informed each go in an appendix.",
        "B8: Jonckheere-Terpstra dropped; Spearman rho is the only B8 test.",
        "B1 kept as a Holm family of 2.",
        "Operationalisation recorded (no change to the tests): " + CORRECT_0P1_DEF,
    ]},
    {"version": 3, "date": "2026-09-23", "changes": [
        "A5: demoted from PRIMARY to DESCRIPTIVE and removed from Holm (validity finding, not results-driven): the "
        "detector runs only on rows whose GOLD qtype is clarify and one rule reads the GOLD category, so it cannot "
        "produce a false positive or a c-cell by construction; McNemar is not a meaningful test. Risk difference + "
        "paired-bootstrap CI reported descriptively with the label: '" + A5_CAVEAT + "' PRIMARY count at seed 42 "
        "falls from 32 to 31.",
        "Exact-match denominators made explicit (no test changed): " + EM_DENOMINATOR_NOTE,
        "New DESCRIPTIVE family LANG (response language), exploratory: " + RESPONSE_LANG_NOTE,
        "Report-only: A7 reports registered primary (a) first, then sensitivity (b) as the definition matching the "
        "thesis; per-model zero-shot baselines reported prominently. A6 factual Llama: registered Wilcoxon verdict "
        "stands; bootstrap CI of the mean difference noted with skew. A6 numeric: CIs too wide to rule out ~15-point "
        "gaps. diff_unit column added to results CSVs so rate differences (pp) and metric-point differences are not "
        "confused.",
    ]},
]


def T(test_id, family, holm_family, role, hypothesis, direction, test, subset, n, effect, **extra):
    d = {"test_id": test_id, "family": family, "holm_family": holm_family, "role": role,
         "hypothesis": hypothesis, "expected_direction": direction, "test": test,
         "data_subset": subset, "expected_n": n, "effect_size": effect}
    d.update(extra)
    return d


def build_plan(gate: dict, pre: dict) -> dict:
    import platform, numpy, scipy, statsmodels, sklearn, krippendorff
    hs = pre["human_sample_by_qtype"]
    n_clar_p, n_clar_s = hs["clarify"]["in_test"], hs["clarify"]["all"]
    n_num_p, n_num_s = hs["numeric_runtime"]["in_test"], hs["numeric_runtime"]["all"]
    n_fac = hs["factual"]["all"]
    tests: list[dict] = []

    # ======================= PART A =======================
    for m in MODELS:
        tests.append(T(f"A1_{m}", "A1", "A1", "PRIMARY",
                       f"Fine-tuning improves exact numeric accuracy (correct_0p1) for {MODEL_LABEL[m]}.",
                       "FT > zero-shot", "McNemar exact on correct_0p1 (paired by row_id)",
                       f"numeric_runtime scorable rows; arms {m}_zeroshot vs {m}_seed42", 146,
                       "odds ratio b/c with 95% CI (exact conditional)",
                       robustness_seeds=[f"{m}_zeroshot vs {m}_seed{s}" for s in ROB_SEEDS]))
        tests.append(T(f"A1b_{m}", "A1b", None, "DESCRIPTIVE",
                       f"Absolute error on single-crop numeric rows where both arms extracted a number, {MODEL_LABEL[m]}.",
                       "FT lower abs_err", "paired bootstrap CI of the median difference (no test)",
                       f"single-crop numeric rows (scenario present, no scenarios list); both arms net_extracted=1; report rows dropped",
                       "<=116", "median paired difference in abs_err (mm), 95% percentile CI"))
    tests.append(T("A2_friedman", "A2", "A2", "PRIMARY",
                   "The four fine-tuned models differ in exact numeric accuracy.", "any difference",
                   "Friedman (4 arms x 146 row blocks) on correct_0p1",
                   "numeric_runtime rows; llama31_8b/mistral_nemo/phi35_mini/phi4_mini seed42", 146, "Kendall's W",
                   robustness_seeds=[f"all four models seed{s}" for s in ROB_SEEDS]))
    tests.append(T("A2_llama_vs_mistral", "A2", "A2", "PRIMARY",
                   "Llama FT and Mistral FT do not differ in exact numeric accuracy.", "not significant (null expected)",
                   "McNemar exact on correct_0p1",
                   "numeric_runtime rows; llama31_8b_seed42 vs mistral_nemo_seed42", 146,
                   "odds ratio b/c 95% CI; ALSO paired-bootstrap 95% CI of accuracy difference (reported prominently)",
                   robustness_seeds=[f"llama31_8b_seed{s} vs mistral_nemo_seed{s}" for s in ROB_SEEDS]))
    tests.append(T("A2_tier", "A2", "A2", "PRIMARY",
                   "The larger tier {Llama, Mistral} outperforms the smaller tier {Phi-3.5, Phi-4} on exact numeric accuracy.",
                   "large tier > small tier", "Wilcoxon signed-rank on per-row tier-mean correctness",
                   "numeric_runtime rows; per-row mean correct_0p1 within each tier, seed42", 146,
                   "rank-biserial r with bootstrap CI",
                   robustness_seeds=[f"tiers at seed{s}" for s in ROB_SEEDS]))
    for m in MODELS:
        tests.append(T(f"A3a_{m}", "A3a", "A3a", "PRIMARY",
                       f"Clarify training teaches {MODEL_LABEL[m]} to ask for the missing parameter (clarify_exact).",
                       "clarify-trained > FT", "McNemar exact on clarify_exact",
                       f"clarify scorable rows; {m}_seed42 vs {m}_seed42_clarify", 68, "odds ratio b/c 95% CI",
                       robustness_seeds=None, robustness_note=A3A4_SEED_NOTE))
        tests.append(T(f"A3b_{m}", "A3b", "A3b", "PRIMARY",
                       f"Clarify training reduces invented values for {MODEL_LABEL[m]}.",
                       "clarify-trained < FT", "McNemar exact on invented_value",
                       f"clarify scorable rows; {m}_seed42 vs {m}_seed42_clarify", 68, "odds ratio b/c 95% CI",
                       robustness_seeds=None, robustness_note=A3A4_SEED_NOTE))
        tests.append(T(f"A4_{m}", "A4", "A4", "PRIMARY",
                       f"Clarify training does not reduce exact numeric accuracy for {MODEL_LABEL[m]} (no-harm).",
                       "not significant; CI of difference reported prominently", "McNemar exact on correct_0p1",
                       f"numeric_runtime rows; {m}_seed42 vs {m}_seed42_clarify", 146,
                       "odds ratio b/c 95% CI; paired-bootstrap 95% CI of accuracy difference",
                       robustness_seeds=None, robustness_note=A3A4_SEED_NOTE,
                       flag="Phi-3.5 slot F1 was reported as dropping in Stage 6E; flag explicitly" if m == "phi35_mini" else None))
    tests.append(T("A5_detector", "A5", None, "DESCRIPTIVE",
                   "Safe handling of clarify rows with vs without the oracle-gated post-hoc detector (Llama clarify-trained).",
                   "n/a (descriptive)", "risk difference with paired-bootstrap 95% CI; no test",
                   "clarify scorable rows; llama31_8b_seed42_clarify vs llama31_8b_seed42_clarify_with_detector; "
                   "safe = predicted_class in {clarify, abstain} OR detector_flag; unsafe = invented and unflagged; "
                   "detector data read from predictions file (not in results_master)", 68,
                   "risk difference in safe handling (hybrid minus model-only), paired-bootstrap 95% CI",
                   note="detector is gated on GOLD qtype and reads GOLD category -> no false positives, c=0 by construction",
                   label=A5_CAVEAT))
    for arm in [f"{m}_zeroshot" for m in MODELS] + [f"{m}_seed42" for m in MODELS] + [f"{m}_seed42_clarify" for m in MODELS]:
        tests.append(T(f"LANG_{arm}", "LANG", None, "DESCRIPTIVE",
                       f"Share of Bengali questions answered in Bengali script, {arm}.", "n/a (exploratory)",
                       "bootstrap 95% CI over rows; no test", "the 181 bn-question rows among the 362 scored (by qtype too)", 181,
                       "proportion with 95% CI", label="exploratory; added after Part A (plan v3)"))
    tests.append(T("EM_denominators", "EM", None, "DESCRIPTIVE", "correct_0p1 over 146 vs exact_0.1 over 116 single-crop rows, per arm.",
                   "n/a", "counts and proportions; no test", "all 20 scored arms", "146 / 116", "counts", label=EM_DENOMINATOR_NOTE))
    for m in MODELS:
        tests.append(T(f"A6_num_unpaired_{m}", "A6", "A6", "PRIMARY",
                       f"Exact numeric accuracy differs between Bengali and English for {MODEL_LABEL[m]} FT.",
                       "no consistent direction expected across models", "Fisher's exact (2x2, unpaired) on correct_0p1",
                       f"numeric_runtime rows; {m}_seed42; 73 bn vs 73 en", 146, "odds ratio with 95% CI",
                       robustness_seeds=[f"{m}_seed{s}" for s in ROB_SEEDS]))
        tests.append(T(f"A6_num_paired_{m}", "A6", None, "SECONDARY",
                       f"Paired bn/en numeric accuracy for {MODEL_LABEL[m]} FT.", "n/a",
                       "McNemar exact on correct_0p1, paired on fact_id",
                       f"numeric_runtime rows with a valid bn/en fact_id pair; {m}_seed42", 17,
                       "odds ratio b/c 95% CI", label="underpowered; not used for inference"))
        tests.append(T(f"A6_fac_{m}", "A6", "A6", "PRIMARY",
                       f"Factual chrF++ is higher in English than Bengali for {MODEL_LABEL[m]} FT.",
                       "English > Bengali", "Wilcoxon signed-rank on chrf_pp, paired on fact_id",
                       f"factual rows with a valid bn/en fact_id pair; {m}_seed42", 57,
                       "rank-biserial r with bootstrap CI; median paired difference",
                       robustness_seeds=[f"{m}_seed{s}" for s in ROB_SEEDS]))
    for arm in [f"{m}_seed42" for m in MODELS] + [f"{m}_seed42_clarify" for m in MODELS]:
        tests.append(T(f"A6_clarify_desc_{arm}", "A6", None, "DESCRIPTIVE",
                       f"bn vs en clarify_exact and invented_value for {arm}.", "n/a",
                       "bootstrap 95% CIs per language (no test; 0 valid bn/en clarify pairs)",
                       f"clarify rows; {arm}; 34 bn vs 34 en", 68, "rates per language with CIs"))
    a7_def = ("confident_error = 1 iff (a) s6_eval.classify_response(generation) == 'answer' AND "
              "(b) not truncated: generated tokens (re-tokenised with the model's own tokenizer; no finish_reason stored) < 511 AND "
              "(c) net_extracted == 1")
    for m in MODELS:
        tests.append(T(f"A7_{m}", "A7", "A7", "PRIMARY",
                       f"Fine-tuning increases confident errors on clarify rows for {MODEL_LABEL[m]} (thesis title claim).",
                       "FT > zero-shot", "McNemar exact on confident_error",
                       f"clarify scorable rows; {m}_zeroshot vs {m}_seed42; {a7_def}", 68, "odds ratio b/c 95% CI",
                       robustness_seeds=[f"{m}_zeroshot vs {m}_seed{s}" for s in ROB_SEEDS]))
        tests.append(T(f"A7_sens_{m}", "A7", None, "SENSITIVITY",
                       f"As A7_{m} but confident_error additionally requires explicit_assumption == 0.",
                       "FT > zero-shot", "McNemar exact on confident_error_strict",
                       f"clarify scorable rows; {m}_zeroshot vs {m}_seed42", 68, "odds ratio b/c 95% CI", label="raw p only"))
        for arm in [f"{m}_zeroshot", f"{m}_seed42"]:
            tests.append(T(f"A7_desc_{arm}", "A7", None, "DESCRIPTIVE",
                           f"Truncation rate, median generated tokens, invented_value rate, explicit_assumption rate among invented answers, {arm}.",
                           "n/a", "bootstrap 95% CIs; list up to 5 explicit_assumption=1 generations", f"clarify rows; {arm}", 68,
                           "rates / median with CIs"))
    for metric in ["em_0p1", "em_1p0", "mae", "net_extract_rate", "abstain_accuracy", "clarify_exact",
                   "invented_value_rate", "dice_word", "chrf_pp", "bertscore_f1"]:
        tests.append(T(f"BOOT_{metric}", "BOOT", None, "DESCRIPTIVE",
                       f"Headline {metric} per arm with bootstrap CI.", "n/a",
                       "percentile bootstrap, 10,000 row resamples, seed 42", f"all 20 scored arms; qtype-appropriate rows",
                       "per arm", "point estimate + 95% CI; FT arms also mean ± std across 3 seeds"))

    # ======================= PART B =======================
    for role, n_all, tag in [("PRIMARY", 182, "test"), ("SENSITIVITY", 200, "all")]:
        tests.append(T(f"B1_friedman_{tag}", "B1", "B1" if role == "PRIMARY" else None, role,
                       "The four arms differ in mean human rating.", "any difference",
                       "Friedman on mean_rating, blocks = row_id", f"human sample {tag} ({n_all} questions) x 4 arms", n_all, "Kendall's W"))
        tests.append(T(f"B1_cochranq_{tag}", "B1", "B1" if role == "PRIMARY" else None, role,
                       "The four arms differ in farmer_yes (majority vote).", "any difference",
                       "Cochran's Q on farmer_yes, blocks = row_id", f"human sample {tag} x 4 arms", n_all, "none (omnibus)"))
    contrasts = {
        "i": ("llama31_8b_zeroshot", "llama31_8b_seed42", ["overall", "numeric_runtime", "factual"], "FT > zero-shot"),
        "ii": ("llama31_8b_seed42", "llama31_8b_seed42_clarify", ["overall", "clarify", "numeric_runtime"], "clarify-trained >= FT (clarify rows: >)"),
        "iii": ("llama31_8b_seed42", "mistral_nemo_seed42", ["overall", "numeric_runtime", "factual"], "not significant (null expected)"),
    }
    n_by = {"overall": (182, 200), "numeric_runtime": (n_num_p, n_num_s), "factual": (n_fac, n_fac),
            "clarify": (n_clar_p, n_clar_s), "abstain": (hs["abstain"]["all"], hs["abstain"]["all"])}
    for c, (a, b, subsets, direction) in contrasts.items():
        for sub in subsets:
            for role, idx, tag in [("PRIMARY", 0, "test"), ("SENSITIVITY", 1, "all"), ("SENSITIVITY", 0, "test_exclA1")]:
                hf = "B2" if role == "PRIMARY" else None
                tests.append(T(f"B2_{c}_{sub}_{tag}", "B2", hf, role,
                               f"Contrast ({c}) {a} vs {b} on mean human rating, {sub}.", direction,
                               "Wilcoxon signed-rank, paired by row_id", f"human sample {tag}; {sub}", n_by[sub][idx],
                               "rank-biserial r with bootstrap CI",
                               **({"label": "A1's ratings excluded"} if tag == "test_exclA1" else {})))
                tests.append(T(f"B3_{c}_{sub}_{tag}", "B3", "B3" if role == "PRIMARY" else None, role,
                               f"Contrast ({c}) {a} vs {b} on farmer_yes, {sub}.", direction,
                               "McNemar exact, paired by row_id; if 0 discordant pairs report 'not testable, identical outcomes'",
                               f"human sample {tag}; {sub}", n_by[sub][idx], "odds ratio b/c 95% CI",
                               **({"label": "A1's ratings excluded"} if tag == "test_exclA1" else {})))
    for tag, n in [("test", 182 * 4), ("all", 800)]:
        role = "SECONDARY"
        tests.append(T(f"B4_mixedlm_{tag}", "B4", None, role,
                       "Arm effects on rating persist after adjusting for annotator severity and item difficulty.",
                       "same direction as B2", "statsmodels MixedLM: rating ~ C(arm, ref=llama31_8b_seed42), groups=annotator, vc_formula={'row':'0 + C(row_id)'}",
                       f"individual ratings, human sample {tag}", "1092 ratings (test) / 1200 (all)" if tag == "test" else 1200,
                       "arm coefficients with 95% CI", limitation="treats 1-5 rating as interval; R ordinal::clmm unavailable (Rscript not on PATH)",
                       supporting_for="M (Llama FT vs Mistral FT): report explicitly whether direction/significance changes vs B2/B3"))
        tests.append(T(f"B4_bayesglm_{tag}", "B4", None, role,
                       "Arm effects on farmer_yes persist after adjusting for annotator and item.", "same direction as B3",
                       "statsmodels BinomialBayesMixedGLM (variational Bayes): farmer_yes ~ C(arm), random effects annotator + row_id",
                       f"individual ratings, human sample {tag}", "1092 / 1200", "posterior mean + 95% credible interval; NO p-values",
                       limitation="variational approximation; credible intervals, not confidence intervals"))
    for arm in HUMAN_ARMS:
        tests.append(T(f"B5_rating_{arm}", "B5", "B5", "PRIMARY",
                       f"Mean human rating differs between Bengali and English outputs of {arm}.", "English >= Bengali expected on factual-heavy overall",
                       "Wilcoxon signed-rank on mean_rating, paired on Stage 7 pair_id", f"all 200 questions -> 100 bn/en pairs; {arm}", 100,
                       "rank-biserial r with bootstrap CI", note="18 of 100 pairs contain an authored translation"))
        tests.append(T(f"B5_farmer_{arm}", "B5", "B5", "PRIMARY",
                       f"farmer_yes differs between Bengali and English outputs of {arm}.", "as above",
                       "McNemar exact on farmer_yes, paired on pair_id", f"100 pairs; {arm}", 100, "odds ratio b/c 95% CI"))
        tests.append(T(f"B5_rating_noauth_{arm}", "B5", None, "SENSITIVITY", f"As B5_rating_{arm}, pairs with no authored translation.", "as above",
                       "Wilcoxon signed-rank", f"82 native pairs; {arm}", 82, "rank-biserial r"))
        tests.append(T(f"B5_farmer_noauth_{arm}", "B5", None, "SENSITIVITY", f"As B5_farmer_{arm}, pairs with no authored translation.", "as above",
                       "McNemar exact", f"82 native pairs; {arm}", 82, "odds ratio"))
    for stat in ["fleiss_kappa_nominal_1to5", "kripp_alpha_ordinal", "kripp_alpha_interval",
                 "fleiss_kappa_bad_mid_good", "fleiss_kappa_farmer", "kripp_alpha_farmer_nominal",
                 "leave_one_annotator_out_alpha_ordinal"]:
        tests.append(T(f"B6_{stat}", "B6", None, "DESCRIPTIVE", f"Inter-annotator agreement: {stat}.", "n/a",
                       "bootstrap 95% CI over items (10,000); alpha via krippendorff package AND independent implementation (must agree to 3 dp)",
                       "100 shared judgments x 5 annotators", 100, "coefficient with CI",
                       audit_check={"fleiss_kappa_nominal_1to5": 0.392, "kripp_alpha_ordinal": 0.721, "fleiss_kappa_farmer": 0.647}.get(stat),
                       stop_rule="if any audit value differs by > 0.005, STOP and report"))
    for metric in ["dice_word", "dice_char3", "dice_char5", "chrf_pp", "rouge_l", "bertscore_f1"]:
        for stat in ["point_biserial_r_vs_farmer_yes", "roc_auc_predicting_farmer_yes", "spearman_rho_vs_mean_rating"]:
            tests.append(T(f"B7_{metric}_{stat}", "B7", None, "DESCRIPTIVE", f"How well {metric} tracks human judgment ({stat}).", "n/a",
                           "bootstrap 95% CI over the 280 (row_id, arm) judgments", "factual rows in test set x 4 arms (seed 42); assert join == 280",
                           280, stat))
    tests.append(T("B8_spearman", "B8", "B8", "PRIMARY", "Human mean rating decreases with absolute numeric error.", "rho < 0",
                   "Spearman rho between abs_err and mean_rating", f"FT-family numeric outputs in test set (3 FT arms x {n_num_p}); rows with net_extracted=1",
                   f"<= {3 * n_num_p}", "Spearman rho with bootstrap CI"))
    tests.append(T("B8_bins", "B8", None, "DESCRIPTIVE", "Mean rating and % farmer_yes per error bin.", "acceptance tracks the ±0.1 mm tier, not ±1 mm",
                   "bootstrap 95% CIs per bin", "same as B8_spearman", f"<= {3 * n_num_p}", "means / proportions with CIs"))

    roles = Counter(t["role"] for t in tests)
    holm = Counter(t["holm_family"] for t in tests if t["role"] == "PRIMARY")
    plan = {
        "stage": "8", "plan_version": PLAN_VERSION, "changelog": CHANGELOG,
        "written_before_any_p_value": True, "seed": SEED, "n_bootstrap": N_BOOT,
        "a3_a4_seed_note": A3A4_SEED_NOTE, "a5_in_sample_caveat": A5_CAVEAT, "correct_0p1_definition": CORRECT_0P1_DEF,
        "em_denominator_note": EM_DENOMINATOR_NOTE, "response_language_note": RESPONSE_LANG_NOTE,
        "scope": ("Scope: 362 of 422 test rows scored. The 60 rows needing human-authored golds "
                  "(40 out_of_domain, 20 conflicting_values) are excluded from this thesis by design. "
                  "The human evaluation sample additionally contains 18 authored Bengali translations of "
                  "English test items, created to form bn/en pairs; results are reported with and without them."),
        "arms": {"scored_in_results_master": 20, "detector_prediction_set": 1,
                 "human_arms": HUMAN_ARMS, "models": MODELS, "seeds": SEEDS},
        "seed_policy": ("primary = seed 42 (matches human eval and all clarify-trained arms). Every FT-involving "
                        "automatic test that has a like-for-like FT arm at seeds 1337 and 2024 (A1, A2, A6, A7) is "
                        "repeated on those seeds. A claim is 'robust' only if significant after Holm on all three "
                        "seeds. " + A3A4_SEED_NOTE),
        "multiple_comparisons": ("Holm-Bonferroni across PRIMARY tests within each holm_family. SECONDARY and "
                                 "SENSITIVITY report raw p only. DESCRIPTIVE report bootstrap 95% CIs, no p."),
        "human_sample_rule": {"PRIMARY": "182 in_test_set questions", "SENSITIVITY": "all 200 questions",
                              "exception": "B5 uses all 200 (pairs need the authored translations); sensitivity = 82 native pairs",
                              "by_qtype": hs},
        "a6_pairing_key": gate["a6_pairing_key"], "a6_valid_pairs": gate["bn_en_pairing_key_options"]["fact_id"],
        "a7_definition": a7_def,
        "a7_explicit_assumption_markers": {
            "en": ["assum", "suppos", "let's say", "let us say", "if we take", "taking a typical", "typical value",
                   "not provided", "not given", "not mentioned"],
            "bn": ["ধরে", "ধরা যাক", "অনুমান", "দেওয়া নেই", "উল্লেখ নেই"],
            "excluded": ["about", "approximately", "প্রায়"]},
        "a7_note": ("Definition motivated by Stage 6 observations (zero-shot hitting the 512-token cap); "
                    "confirmatory of an observed pattern, not a blind prediction."),
        "b4_method": "Python statsmodels fallback (Rscript not on PATH; R not installed system-wide by policy)",
        "verdict_wording": "Non-significant results are reported as 'no significant difference detected' with the CI of the difference; never 'equivalent' or 'no difference'.",
        "report_structure": {
            "table1_hypotheses": [
                {"id": "H1", "claim": "Fine-tuning beats zero-shot", "tests": ["A1", "B2-i", "B3-i"]},
                {"id": "H2", "claim": "Bengali vs English", "tests": ["A6", "B5"]},
                {"id": "H3", "claim": "Human agreement", "tests": ["B6"]},
                {"id": "T", "claim": "Fine-tuning converts invented values into confident errors (thesis title)", "tests": ["A7"]},
                {"id": "C1", "claim": "Clarify training teaches clarification", "tests": ["A3a", "B3-ii"]},
                {"id": "C2", "claim": "Clarify training reduces invented values", "tests": ["A3b", "B3-ii"]},
                {"id": "C3", "claim": "Clarify training does not harm numeric", "tests": ["A4"]},
                {"id": "D", "claim": "Detector adds safety", "tests": ["A5"]},
                {"id": "M", "claim": "Llama FT vs Mistral FT", "tests": ["A2", "B2-iii", "B3-iii", "B4"]}],
            "table1_columns": ["claim", "tests", "supported / not supported / mixed", "key numbers"],
            "other_tables": ["Table 2 automatic results by family", "Table 3 human results by family",
                             "Table 4 seed robustness", "Table 5 primary-vs-sensitivity human sample", "Table 6 dice validation",
                             "Table 7 agreement with CIs", "Section: What changed"]},
        "environment": {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
                        "statsmodels": statsmodels.__version__, "sklearn": sklearn.__version__,
                        "krippendorff": getattr(krippendorff, "__version__", "installed"), "pandas": pd.__version__,
                        "R": "not available", "env_freeze": "reports/stats/env_freeze.txt"},
        "summary": {"n_tests": len(tests), "by_role": dict(roles), "primary_per_holm_family": dict(holm)},
        "tests": tests,
    }
    return plan


def write_plan(gate: dict, pre: dict) -> dict:
    import subprocess
    plan = build_plan(gate, pre)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "test_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (OUT_DIR / "env_freeze.txt").write_text(freeze, encoding="utf-8")
    log.info("=" * 78)
    log.info("TEST PLAN written -> %s   (%d tests)", (OUT_DIR / "test_plan.json").relative_to(PROJECT), plan["summary"]["n_tests"])
    log.info("  by role: %s", plan["summary"]["by_role"])
    log.info("  PRIMARY per holm_family: %s", plan["summary"]["primary_per_holm_family"])
    log.info("  env_freeze -> reports/stats/env_freeze.txt")
    return plan


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["gate", "precheck", "plan", "A", "B", "report", "all"], default="gate")
    args = ap.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    for noisy in ("httpx", "huggingface_hub", "urllib3", "filelock", "fsspec", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    gate = entry_gate()
    if args.part == "gate":
        return 0
    pre = precheck()
    if args.part == "precheck":
        return 0
    plan = write_plan(gate, pre)
    if args.part == "plan":
        return 0
    if args.part in ("A", "all"):
        import s8_part_a
        s8_part_a.run(plan)
        if args.part == "A":
            return 0
    if args.part in ("B", "all"):
        import s8_part_b
        s8_part_b.run(plan)
        if args.part == "B":
            return 0
    if args.part in ("report", "all"):
        import s8_report
        s8_report.run(plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
