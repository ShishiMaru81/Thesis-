#!/usr/bin/env python3
"""
s1_split_first.py — v3 STAGE 1: split BEFORE any augmentation.

This is the leakage fix. v2 generated 6 paraphrases per fact and then split the
already-expanded corpus; v3 splits the 902-row seed corpus produced by Stage 0,
and only train_seed.jsonl is ever handed to Stage 2. Nothing that reaches val or
test can be a generated variant of anything, because val and test are frozen
before the generator runs.

GROUPING
--------
Split at source_id level with sklearn GroupShuffleSplit, never at row level.
source_id is "<calendar image filename>#<row index>", so every row derived from
one extracted calendar row — both languages of it — moves as a unit.

STRATIFICATION
--------------
Ratios 70/10/20, stratified by (qtype, lang, crop).

  lang   is exact by construction, not by sampling: Stage 0 emits a bn row and
         an en row for the same fact under the SAME source_id, so both languages
         always travel together and every split is 50/50. Verified, not assumed
         — see the Gate 1 table.

  qtype  is a hard stratum. Each qtype is allocated 70/10/20 independently.

  crop   is a soft stratum inside each qtype. A per-crop hard stratum cannot
         work here: the numeric_runtime and abstain strata hold exactly 3
         source_ids per crop, and no allocation of 3 items yields 70/10/20 —
         largest-remainder rounding sends every one of them to (2 train,
         0 val, 1 test), which would leave val with zero numeric and zero
         abstain rows and make eval_loss / early stopping meaningless.
         Instead crops are walked in a seeded order and allocation targets are
         carried CUMULATIVELY across the qtype, so the qtype converges on
         70/10/20 while each individual crop still contributes proportionally.

SCENARIO CONSTRAINT
-------------------
No scenario_hash may appear in more than one split. On collision the entire
source_id group is moved to the split already holding the majority of that
hash, and the move is logged. Seed scenarios are sampled per (crop, stage) so
collisions are not expected — the repair path exists so that it is exercised
and audited rather than assumed away.

Run:  python s1_split_first.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}

V3_ROOT = Path(__file__).resolve().parent.parent
SEED_JSONL = V3_ROOT / "data" / "seed" / "seed_corpus.jsonl"
SPLIT_DIR = V3_ROOT / "data" / "splits"
LOG_PATH = V3_ROOT / "logs" / "s1_split_first.log"
GATE_JSON = V3_ROOT / "reports" / "gate1_split.json"
PREFLIGHT_JSON = V3_ROOT / "reports" / "leakage" / "s1_preflight_lexical.json"


# ----------------------------------------------------------------- allocation
def gss_take(gids: list[str], n_take: int, random_state: int) -> tuple[list[str], list[str]]:
    """Pull n_take groups out of gids using sklearn GroupShuffleSplit.

    Every gid is unique, so one group == one sample and GroupShuffleSplit's
    group-aware selection reduces to a seeded choice of n_take groups. Using it
    rather than random.sample keeps the group semantics explicit and makes the
    split reproducible from (gids, n_take, random_state) alone.
    """
    if n_take <= 0:
        return [], list(gids)
    if n_take >= len(gids):
        return list(gids), []
    arr = np.array(gids, dtype=object)
    X = np.arange(len(gids)).reshape(-1, 1)
    gss = GroupShuffleSplit(n_splits=1, test_size=n_take, random_state=random_state)
    rest_idx, take_idx = next(gss.split(X, groups=arr))
    return sorted(arr[take_idx].tolist()), sorted(arr[rest_idx].tolist())


def allocate(qtype_strata: list[tuple[str, list[str]]], rng: random.Random):
    """Cumulative largest-remainder allocation across one qtype's crop strata.

    Walking crops in a seeded order and tracking CUMULATIVE targets means the
    rounding remainder lands on a different crop each time instead of always on
    the same one, so the qtype total converges on 70/10/20 without any single
    crop being systematically over- or under-represented in test.
    """
    order = list(qtype_strata)
    rng.shuffle(order)
    cum_n = 0
    given = {"train": 0, "val": 0, "test": 0}
    plan = []
    for crop, gids in order:
        n = len(gids)
        cum_n += n
        t_train = round(cum_n * RATIOS["train"])
        t_val = round(cum_n * RATIOS["val"])
        t_test = cum_n - t_train - t_val
        want = {"train": t_train - given["train"],
                "val": t_val - given["val"],
                "test": t_test - given["test"]}
        # Rounding can ask for a negative count or overshoot n; repair by
        # clamping and pushing the difference onto train (the largest bucket).
        for k in want:
            want[k] = max(0, want[k])
        drift = n - sum(want.values())
        want["train"] += drift
        if want["train"] < 0:
            for k in ("test", "val"):
                take = min(-want["train"], want[k])
                want[k] -= take
                want["train"] += take
        for k in given:
            given[k] += want[k]
        plan.append((crop, gids, want))
    return plan


# ----------------------------------------------------------------- preflight
def _is_punct(ch: str) -> bool:
    """Unicode punctuation/symbol test.

    DO NOT replace this with re.sub(r"[^\\w\\s]", ...). Python's `re` defines
    \\w as str.isalnum() plus underscore, and Bengali vowel signs and the
    virama are category Mn/Mc — isalnum() is False for every one of them. That
    pattern therefore deletes all matras and hasant, collapsing তিল (Sesame)
    and তুলা (Cotton) to the same string and manufacturing phantom exact
    matches between splits. This function keeps marks and drops only real
    punctuation, including the Bengali danda (।) and double danda (॥), which
    are category Po.
    """
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def norm(text: str) -> str:
    """Preview of Stage 3 detector 1 / Stage 6 bn_normalize.py. Deliberately
    NOT the versioned normalizer — this is a Gate 1 advisory only, and
    scripts/bn_normalize.py (v3.0) becomes the single source of truth at
    Stage 6. Nothing here is written into a results file."""
    t = unicodedata.normalize("NFC", text)
    t = "".join(" " if _is_punct(c) else c for c in t)
    return " ".join(t.lower().split())


def _selfcheck_normalizer() -> None:
    """Regression guard for the \\w bug above. Cheap; runs on every invocation
    so it cannot rot."""
    a, b = "তিল ফসলের সেচ", "তুলা ফসলের সেচ"
    assert norm(a) != norm(b), (
        "normalizer is destroying Bengali combining marks — "
        f"{a!r} and {b!r} both normalize to {norm(a)!r}")
    assert norm("ধান।") == norm("ধান"), "danda (U+0964) not stripped"
    assert norm("Hello, World!") == "hello world"


def char_ngrams(text: str, n: int = 5) -> set[str]:
    t = norm(text).replace(" ", "")
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return 0.0 if u == 0 else len(a & b) / u


def preflight_lexical(train_rows, test_rows, val_rows, threshold=0.60, top_k=50):
    """Run Stage 3's lexical detector NOW, on seed-vs-seed, so the Gate 1 report
    says whether the corpus can pass Stage 3 at all before Stage 2 spends Gemini
    calls expanding it. Cheap here (~630 x ~180 pairs); the real Stage 3 run
    uses MinHash LSH because the augmented corpus is ~100x larger."""
    out = {}
    tr = [(r, char_ngrams(r["question"])) for r in train_rows]
    for name, rows in (("test", test_rows), ("val", val_rows)):
        pairs = []
        held = [(r, char_ngrams(r["question"])) for r in rows]
        for hr, hg in held:
            best, best_tr = 0.0, None
            for trr, trg in tr:
                j = jaccard(hg, trg)
                if j > best:
                    best, best_tr = j, trr
            pairs.append({
                "held_out_split": name,
                "held_out_row_id": hr["row_id"],
                "held_out_question": hr["question"],
                "nearest_train_question": best_tr["question"] if best_tr else None,
                "jaccard_char5": round(best, 4),
                "same_fact_id": bool(best_tr and best_tr["fact_id"] == hr["fact_id"]),
                "same_crop": bool(best_tr and best_tr["crop"] == hr["crop"]),
                "qtype": hr["qtype"],
                "lang": hr["lang"],
            })
        pairs.sort(key=lambda p: -p["jaccard_char5"])
        scores = [p["jaccard_char5"] for p in pairs]
        arr = np.array(scores) if scores else np.array([0.0])
        out[name] = {
            "n_held_out": len(pairs),
            "n_at_or_above_threshold": int((arr >= threshold).sum()),
            "pct_at_or_above_threshold": round(100.0 * float((arr >= threshold).mean()), 2),
            "max": round(float(arr.max()), 4),
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
            "n_identical_after_normalization": int((arr >= 0.999).sum()),
            "top_pairs": pairs[:top_k],
        }
    out["threshold"] = threshold
    return out


# ----------------------------------------------------------------- main
def main(seed_path: Path, split_dir: Path) -> dict:
    _selfcheck_normalizer()
    rows = [json.loads(l) for l in open(seed_path, encoding="utf-8")]
    moves: list[dict] = []

    # --- group rows by source_id ------------------------------------------
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[r["source_id"]].append(r)
    gmeta = {gid: {"qtype": rs[0]["qtype"], "crop": rs[0]["crop"]}
             for gid, rs in groups.items()}
    for gid, rs in groups.items():
        assert len({r["qtype"] for r in rs}) == 1, f"{gid} spans qtypes"
        assert len({r["crop"] for r in rs}) == 1, f"{gid} spans crops"

    # --- allocate, then select with GroupShuffleSplit ----------------------
    by_qtype: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for gid, m in gmeta.items():
        by_qtype[m["qtype"]][m["crop"]].append(gid)

    assign: dict[str, str] = {}
    alloc_log = []
    stratum_index = 0
    for qtype in sorted(by_qtype):
        strata = sorted((crop, sorted(g)) for crop, g in by_qtype[qtype].items())
        plan = allocate(strata, random.Random(SEED))
        for crop, gids, want in plan:
            rs = SEED + stratum_index          # per-stratum state: a single fixed
            stratum_index += 1                 # random_state would make GSS pick
            #                                    the same index positions in every
            #                                    same-sized stratum — with 3 groups
            #                                    per (numeric, crop) that would put
            #                                    the SAME growth stage in test for
            #                                    all 28 crops.
            test_g, rest = gss_take(sorted(gids), want["test"], rs)
            val_g, train_g = gss_take(rest, want["val"], rs + 10_000)
            for g in test_g:
                assign[g] = "test"
            for g in val_g:
                assign[g] = "val"
            for g in train_g:
                assign[g] = "train"
            alloc_log.append({"qtype": qtype, "crop": crop, "n_groups": len(gids),
                              "random_state": rs, **want})

    # --- scenario_hash constraint -----------------------------------------
    # No scenario_hash may straddle splits. Move the whole source_id group to
    # whichever split already holds the majority of that hash's rows.
    hash_rows = collections.defaultdict(list)
    for r in rows:
        if r["scenario_hash"]:
            hash_rows[r["scenario_hash"]].append(r)
    for shash, rs in sorted(hash_rows.items()):
        splits = {assign[r["source_id"]] for r in rs}
        if len(splits) <= 1:
            continue
        counts = collections.Counter(assign[r["source_id"]] for r in rs)
        majority = max(sorted(counts), key=lambda s: counts[s])
        for gid in sorted({r["source_id"] for r in rs}):
            if assign[gid] != majority:
                moves.append({"scenario_hash": shash, "source_id": gid,
                              "from": assign[gid], "to": majority,
                              "hash_split_counts": dict(counts)})
                assign[gid] = majority

    # --- materialise, deterministic order ---------------------------------
    out = {"train": [], "val": [], "test": []}
    for gid, rs in groups.items():
        out[assign[gid]].extend(rs)
    for k in out:
        out[k].sort(key=lambda r: (r["source_id"], r["qtype"], r["lang"], r["fact_id"]))

    split_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": split_dir / "train_seed.jsonl",
             "val": split_dir / "val.jsonl",
             "test": split_dir / "test.jsonl"}
    for k, p in paths.items():
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in out[k]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    test_sha = hashlib.sha256(paths["test"].read_bytes()).hexdigest()
    (split_dir / "test.sha256").write_text(
        f"{test_sha}  test.jsonl\n", encoding="utf-8", newline="\n")

    with open(split_dir / "split_assignment.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "split", "qtype", "crop", "n_rows", "fact_id"])
        for gid in sorted(groups):
            rs = groups[gid]
            w.writerow([gid, assign[gid], rs[0]["qtype"], rs[0]["crop"],
                        len(rs), rs[0]["fact_id"]])

    # --- overlap verification ---------------------------------------------
    sid = {k: {r["source_id"] for r in v} for k, v in out.items()}
    fid = {k: {r["fact_id"] for r in v} for k, v in out.items()}
    shs = {k: {r["scenario_hash"] for r in v if r["scenario_hash"]} for k, v in out.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    overlap = {
        "source_id": {f"{a}|{b}": sorted(sid[a] & sid[b]) for a, b in pairs},
        "fact_id": {f"{a}|{b}": sorted(fid[a] & fid[b]) for a, b in pairs},
        "scenario_hash": {f"{a}|{b}": sorted(shs[a] & shs[b]) for a, b in pairs},
    }
    zero = {kind: all(len(v) == 0 for v in d.values()) for kind, d in overlap.items()}

    # --- Gate 1 table ------------------------------------------------------
    def pct(rs, key, val):
        return round(100.0 * sum(1 for r in rs if r[key] == val) / len(rs), 1) if rs else 0.0

    table = []
    for k in ("train", "val", "test"):
        rs = out[k]
        table.append({
            "split": "train_seed" if k == "train" else k,
            "rows": len(rs),
            "pct_of_total": round(100.0 * len(rs) / len(rows), 1),
            "pct_bn": pct(rs, "lang", "bn"),
            "pct_en": pct(rs, "lang", "en"),
            "pct_numeric": pct(rs, "qtype", "numeric_runtime"),
            "pct_factual": pct(rs, "qtype", "factual"),
            "pct_abstain": pct(rs, "qtype", "abstain"),
            "distinct_crops": len({r["crop"] for r in rs}),
            "distinct_source_ids": len({r["source_id"] for r in rs}),
        })

    pre = preflight_lexical(out["train"], out["test"], out["val"])
    PREFLIGHT_JSON.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_JSON.write_text(json.dumps(pre, ensure_ascii=False, indent=2), encoding="utf-8")

    n_norm = {}
    for a, b in pairs:
        na = {norm(r["question"]) for r in out[a]}
        nb = {norm(r["question"]) for r in out[b]}
        n_norm[f"{a}|{b}"] = len(na & nb)

    report = {
        "stage": "GATE 1",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": SEED,
        "ratios": RATIOS,
        "splitter": "sklearn.model_selection.GroupShuffleSplit",
        "sklearn_version": __import__("sklearn").__version__,
        "grouping_key": "source_id",
        "input": str(seed_path),
        "input_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        "input_rows": len(rows),
        "input_groups": len(groups),
        "table": table,
        "overlap_counts": {kind: {k: len(v) for k, v in d.items()}
                           for kind, d in overlap.items()},
        "overlap_examples": {kind: {k: v[:5] for k, v in d.items() if v}
                             for kind, d in overlap.items()},
        "zero_overlap": zero,
        "normalized_question_collisions_across_splits": n_norm,
        "scenario_hash_moves": moves,
        "scenario_hash_moves_count": len(moves),
        "allocation_log": alloc_log,
        "crops_per_split": {k: sorted({r["crop"] for r in v}) for k, v in out.items()},
        "crops_absent_from_test": sorted(
            {r["crop"] for r in rows} - {r["crop"] for r in out["test"]}),
        "test_sha256": test_sha,
        "files": {k: str(p) for k, p in paths.items()} | {
            "test_sha256_file": str(split_dir / "test.sha256"),
            "split_assignment": str(split_dir / "split_assignment.csv"),
        },
        "preflight_lexical_summary": {
            k: {kk: vv for kk, vv in v.items() if kk != "top_pairs"}
            for k, v in pre.items() if isinstance(v, dict)
        },
    }
    GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    GATE_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        f"s1_split_first.py\nseed={SEED}\nrun_at={report['generated_at']}\n"
        f"splitter=GroupShuffleSplit(sklearn {report['sklearn_version']})\n"
        f"per-stratum random_state = {SEED} + stratum_index (test), +10000 (val)\n"
        f"train={len(out['train'])} val={len(out['val'])} test={len(out['test'])}\n"
        f"test.sha256={test_sha}\nscenario_hash_moves={len(moves)}\n", encoding="utf-8")
    return report, out, pre


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default=str(SEED_JSONL))
    ap.add_argument("--out", dest="outdir", default=str(SPLIT_DIR))
    a = ap.parse_args()
    rep, out, pre = main(Path(a.infile), Path(a.outdir))

    W = [12, 6, 8, 7, 7, 10, 9, 9, 8, 12]
    hdr = ["split", "rows", "%total", "%bn", "%en", "%numeric", "%factual",
           "%abstain", "crops", "source_ids"]
    print("=" * 66)
    print("GATE 1 — SPLIT FIRST (ratios 70/10/20, grouped by source_id)")
    print("=" * 66)
    print("".join(h.ljust(w) for h, w in zip(hdr, W)))
    print("-" * sum(W))
    for t in rep["table"]:
        cells = [t["split"], t["rows"], t["pct_of_total"], t["pct_bn"], t["pct_en"],
                 t["pct_numeric"], t["pct_factual"], t["pct_abstain"],
                 t["distinct_crops"], t["distinct_source_ids"]]
        print("".join(str(c).ljust(w) for c, w in zip(cells, W)))

    print("\nOVERLAP ACROSS SPLITS (all must be 0)")
    for kind in ("source_id", "fact_id", "scenario_hash"):
        c = rep["overlap_counts"][kind]
        flag = "PASS" if rep["zero_overlap"][kind] else "FAIL"
        print(f"  {kind:15s} {c}  -> {flag}")
    print(f"  {'exact question':15s} {rep['normalized_question_collisions_across_splits']}"
          f"  -> {'PASS' if all(v == 0 for v in rep['normalized_question_collisions_across_splits'].values()) else 'FAIL'}")
    print(f"\nscenario_hash groups moved to resolve collisions: {rep['scenario_hash_moves_count']}")
    print(f"crops absent from test: {rep['crops_absent_from_test'] or 'none'}")
    print(f"\nTEST SHA-256: {rep['test_sha256']}")

    print("\n" + "=" * 66)
    print("ADVISORY — Stage 3 lexical detector run early on the seed split")
    print("(threshold 0.60 char-5gram Jaccard, exactly what Stage 3 will use)")
    print("=" * 66)
    for name in ("test", "val"):
        s = pre[name]
        print(f"  train_seed vs {name:5s}  n={s['n_held_out']:4d}  "
              f"max={s['max']:.3f}  mean={s['mean']:.3f}  p90={s['p90']:.3f}  "
              f">=0.60: {s['n_at_or_above_threshold']} ({s['pct_at_or_above_threshold']}%)")
    print(f"\n  full pair tables -> {PREFLIGHT_JSON}")
    print(f"  gate report      -> {GATE_JSON}")
    print("\nSTOP. Gate 1 requires review before Stage 2.")
