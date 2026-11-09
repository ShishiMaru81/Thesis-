#!/usr/bin/env python3
"""
s3_leakage_audit.py — v3 STAGE 3: four-detector train↔held-out leakage audit.

GATE 3. STOP after reporting. Do not proceed to Stage 4 without approval.

Detectors
---------
  1. Exact match on bn_normalize v3.1 text (FAIL if any)
  2. Char-5gram multiset Jaccard >= 0.60 AND (same fact_id OR same
     normalized gold answer) → FAIL. Template-only high Jaccard is PASS.
  3. Embedding cosine >= 0.92 → candidate for human review (not auto-FAIL).
     Human decisions live in reports/leakage/semantic_human_decisions.json.
     A decision of "leakage" flips Detector 3 to FAIL.
  4. scenario_hash overlap vs heldout_hashes.txt (FAIL if any)

Hard Rule 1 note
----------------
Detectors 1–3 open val.jsonl / test.jsonl by design (this IS the audit).
Detector 4 loads only heldout_hashes.txt (hashes exported by s1b) so the
scenario check does not re-derive hashes from held-out content at runtime.

Run:  python s3_leakage_audit.py
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- paths
V3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_ROOT / "scripts"))

from bn_normalize import (  # noqa: E402
    NORMALIZER_VERSION,
    assert_version,
    normalize,
    run_regression_tests,
)

TRAIN_PATH = V3_ROOT / "data" / "augmented" / "train_augmented.jsonl"
VAL_PATH = V3_ROOT / "data" / "splits" / "val.jsonl"
TEST_PATH = V3_ROOT / "data" / "splits" / "test.jsonl"
TEST_SHA_PATH = V3_ROOT / "data" / "splits" / "test.sha256"
HELDOUT_JSON = V3_ROOT / "data" / "splits" / "heldout_scenario_hashes.json"
HELDOUT_TXT = V3_ROOT / "data" / "splits" / "heldout_hashes.txt"

OUT_DIR = V3_ROOT / "reports" / "leakage"
OUT_JSON = OUT_DIR / "leakage_report.json"
OUT_MD = OUT_DIR / "leakage_report.md"
DECISIONS_PATH = OUT_DIR / "semantic_human_decisions.json"
LOG_PATH = V3_ROOT / "logs" / "s3_leakage_audit.log"

EXPECTED_TRAIN_SHA256 = (
    "f4ba6742198d7e1d1e9d30f7fcfa090be6f7831afc7637bd7899be638bc2266d"
)

JACCARD_THRESHOLD = 0.60
SEMANTIC_THRESHOLD = 0.92
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 42


# ----------------------------------------------------------------- IO
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_heldout_hashes_txt() -> set[str]:
    """Prefer heldout_hashes.txt; materialize it from the s1b JSON if missing."""
    if HELDOUT_TXT.exists():
        hashes = {
            ln.strip()
            for ln in HELDOUT_TXT.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        }
        return hashes

    if not HELDOUT_JSON.exists():
        raise FileNotFoundError(
            f"Neither {HELDOUT_TXT} nor {HELDOUT_JSON} found. "
            "Run s1b_export_heldout_hashes.py first."
        )
    payload = json.loads(HELDOUT_JSON.read_text(encoding="utf-8"))
    hashes = set(payload["scenario_hashes"])
    HELDOUT_TXT.write_text(
        "# scenario_hash blocklist — hashes only; regenerated from "
        "heldout_scenario_hashes.json\n"
        + "\n".join(sorted(hashes))
        + "\n",
        encoding="utf-8",
    )
    return hashes


def load_human_decisions() -> dict:
    """Map pair_key -> {decision: legitimate|leakage, notes, decided_at}."""
    if not DECISIONS_PATH.exists():
        template = {
            "_readme": (
                "Add one entry per Detector-3 candidate pair_key. "
                "decision must be 'legitimate' or 'leakage'. "
                "A 'leakage' decision fails GATE 3 Detector 3."
            ),
            "decisions": {},
        }
        DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DECISIONS_PATH.write_text(
            json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {}
    payload = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
    return dict(payload.get("decisions") or {})


def pair_key(held_split: str, held_id: str, train_id: str) -> str:
    return f"{held_split}:{held_id}::{train_id}"


# ----------------------------------------------------------------- normalize
def norm_question(row: dict) -> str:
    """Detector-1 / Detector-2 question key. bn_normalize v3.1 + lower."""
    lang = row.get("lang") or "en"
    return normalize(row.get("question") or "", lang).lower()


def norm_answer(row: dict) -> str:
    lang = row.get("lang") or "en"
    return normalize(row.get("answer") or "", lang).lower()


# ----------------------------------------------------------------- ngrams
def char_ngrams_multiset(text: str, n: int = 5) -> Counter:
    """Character n-grams over whitespace-stripped normalized text (multiset)."""
    t = text.replace(" ", "")
    if len(t) < n:
        return Counter([t] if t else [])
    return Counter(t[i : i + n] for i in range(len(t) - n + 1))


def jaccard_multiset(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return 0.0 if union == 0 else inter / union


def histogram(scores: list[float], bins: list[float] | None = None) -> dict:
    if bins is None:
        bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
    counts = {f"[{bins[i]:.2f},{bins[i+1]:.2f})": 0 for i in range(len(bins) - 1)}
    labels = list(counts.keys())
    for s in scores:
        placed = False
        for i in range(len(bins) - 1):
            if bins[i] <= s < bins[i + 1]:
                counts[labels[i]] += 1
                placed = True
                break
        if not placed and scores:
            counts[labels[-1]] += 1
    arr = np.array(scores, dtype=float) if scores else np.array([0.0])
    return {
        "bins": counts,
        "n": len(scores),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "p99": round(float(np.percentile(arr, 99)), 6),
    }


# ----------------------------------------------------------------- Detector 1
def detector_exact(train: list[dict], held: list[tuple[str, dict]]) -> dict:
    index: dict[str, list[dict]] = collections.defaultdict(list)
    for r in train:
        index[norm_question(r)].append(r)

    pairs = []
    for split, hr in held:
        nq = norm_question(hr)
        if not nq:
            continue
        for tr in index.get(nq, []):
            pairs.append({
                "held_out_split": split,
                "held_out_row_id": hr.get("row_id"),
                "train_row_id": tr.get("row_id"),
                "held_out_question": hr.get("question"),
                "train_question": tr.get("question"),
                "normalized": nq,
                "same_fact_id": hr.get("fact_id") == tr.get("fact_id"),
                "lang": hr.get("lang"),
            })

    return {
        "count": len(pairs),
        "pairs": pairs[:20],
        "pass": len(pairs) == 0,
        "fail_rule": "FAIL if count > 0",
    }


# ----------------------------------------------------------------- Detector 2
def detector_jaccard(
    train: list[dict],
    held: list[tuple[str, dict]],
    threshold: float = JACCARD_THRESHOLD,
    top_k: int = 50,
) -> dict:
    """Brute-force multiset char-5gram Jaccard (3552×270 ≈ 1e6; exact NN).

    MinHash LSH (datasketch) is used as a belt-and-suspenders candidate
    filter; exact Jaccard is always computed for the reported scores so the
    audit cannot miss a pair because of LSH approximation.
    """
    from datasketch import MinHash, MinHashLSH

    train_prep = []
    for r in train:
        nq = norm_question(r)
        grams = char_ngrams_multiset(nq, 5)
        mh = MinHash(num_perm=128, seed=SEED)
        for g in grams:
            mh.update(g.encode("utf-8"))
        train_prep.append({
            "row": r,
            "nq": nq,
            "grams": grams,
            "na": norm_answer(r),
            "mh": mh,
        })

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for i, tp in enumerate(train_prep):
        # MinHashLSH keys must be unique; row_id can collide across origins.
        lsh.insert(f"t{i}", tp["mh"])

    nearest_scores: list[float] = []
    high_pairs: list[dict] = []
    fail_pairs: list[dict] = []

    for split, hr in held:
        nq = norm_question(hr)
        grams = char_ngrams_multiset(nq, 5)
        na = norm_answer(hr)
        mh = MinHash(num_perm=128, seed=SEED)
        for g in grams:
            mh.update(g.encode("utf-8"))
        lsh_hits = set(lsh.query(mh))

        best_j = -1.0
        for i, tp in enumerate(train_prep):
            j = jaccard_multiset(grams, tp["grams"])
            if j > best_j:
                best_j = j
            if j < threshold:
                continue
            tr = tp["row"]
            same_fact = bool(
                hr.get("fact_id") and hr.get("fact_id") == tr.get("fact_id")
            )
            same_gold = bool(na) and na == tp["na"]
            is_fail = same_fact or same_gold
            rec = {
                "held_out_split": split,
                "held_out_row_id": hr.get("row_id"),
                "train_row_id": tr.get("row_id"),
                "held_out_question": hr.get("question"),
                "train_question": tr.get("question"),
                "jaccard_char5": round(j, 6),
                "same_fact_id": bool(same_fact),
                "same_normalized_gold_answer": bool(same_gold),
                "held_fact_id": hr.get("fact_id"),
                "train_fact_id": tr.get("fact_id"),
                "leakage_fail": is_fail,
                "lang": hr.get("lang"),
                "qtype": hr.get("qtype"),
                "lsh_candidate": f"t{i}" in lsh_hits,
            }
            high_pairs.append(rec)
            if is_fail:
                fail_pairs.append(rec)
        nearest_scores.append(max(best_j, 0.0))

    high_pairs.sort(key=lambda p: -p["jaccard_char5"])
    fail_pairs.sort(key=lambda p: -p["jaccard_char5"])

    return {
        "threshold": threshold,
        "fail_rule": (
            "FAIL if Jaccard >= 0.60 AND "
            "(shared fact_id OR normalized gold_answer identical)"
        ),
        "distribution": histogram(nearest_scores),
        "n_pairs_at_or_above_threshold": len(high_pairs),
        "n_fail_pairs": len(fail_pairs),
        "top_50": high_pairs[:top_k],
        "fail_pairs": fail_pairs[:50],
        "pass": len(fail_pairs) == 0,
        "method": "exact_multiset_jaccard_char5 + MinHashLSH_crosscheck",
    }


# ----------------------------------------------------------------- Detector 3
def detector_semantic(
    train: list[dict],
    held: list[tuple[str, dict]],
    decisions: dict,
    threshold: float = SEMANTIC_THRESHOLD,
    top_k: int = 50,
) -> dict:
    """FAISS cosine on paraphrase-multilingual-MiniLM-L12-v2 embeddings."""
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)

    train_texts = [norm_question(r) for r in train]
    held_texts = [norm_question(hr) for _, hr in held]

    train_emb = model.encode(
        train_texts, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    held_emb = model.encode(
        held_texts, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

    dim = train_emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(train_emb)

    # Top-5 train neighbours per held-out row for candidate discovery.
    k = min(5, len(train))
    sims, idxs = index.search(held_emb, k)

    nearest_scores = [float(sims[i, 0]) for i in range(len(held))]
    candidates = []
    human_leakage = []
    pending = []

    for i, (split, hr) in enumerate(held):
        for j in range(k):
            score = float(sims[i, j])
            if score < threshold:
                continue
            ti = int(idxs[i, j])
            tr = train[ti]
            key = pair_key(split, hr.get("row_id"), tr.get("row_id"))
            decision = (decisions.get(key) or {}).get("decision")
            rec = {
                "pair_key": key,
                "held_out_split": split,
                "held_out_row_id": hr.get("row_id"),
                "train_row_id": tr.get("row_id"),
                "held_out_question": hr.get("question"),
                "train_question": tr.get("question"),
                "cosine": round(score, 6),
                "same_fact_id": hr.get("fact_id") == tr.get("fact_id"),
                "lang": hr.get("lang"),
                "qtype": hr.get("qtype"),
                "human_decision": decision,
            }
            candidates.append(rec)
            if decision == "leakage":
                human_leakage.append(rec)
            elif decision != "legitimate":
                pending.append(rec)

    candidates.sort(key=lambda p: -p["cosine"])

    # Also build top-50 by nearest-neighbour cosine (even below threshold)
    # for the report distribution table.
    top_nn = []
    order = np.argsort(-np.array(nearest_scores))[:top_k]
    for i in order:
        split, hr = held[i]
        ti = int(idxs[i, 0])
        tr = train[ti]
        top_nn.append({
            "held_out_split": split,
            "held_out_row_id": hr.get("row_id"),
            "train_row_id": tr.get("row_id"),
            "held_out_question": hr.get("question"),
            "train_question": tr.get("question"),
            "cosine": round(float(sims[i, 0]), 6),
            "same_fact_id": hr.get("fact_id") == tr.get("fact_id"),
            "lang": hr.get("lang"),
            "qtype": hr.get("qtype"),
            "above_threshold": float(sims[i, 0]) >= threshold,
            "human_decision": (
                decisions.get(pair_key(split, hr.get("row_id"), tr.get("row_id")))
                or {}
            ).get("decision"),
        })

    # Auto-PASS unless a human has marked leakage. Pending review is logged
    # but does not fail the gate (spec: not automatic FAIL).
    passed = len(human_leakage) == 0
    return {
        "threshold": threshold,
        "embedding_model": EMBEDDING_MODEL,
        "distribution": histogram(nearest_scores),
        "n_candidates_at_or_above_threshold": len(candidates),
        "n_pending_human_review": len(pending),
        "n_human_marked_leakage": len(human_leakage),
        "n_human_marked_legitimate": sum(
            1 for c in candidates if c.get("human_decision") == "legitimate"
        ),
        "candidates": candidates[:100],
        "top_50": top_nn,
        "pending_review": pending[:50],
        "human_leakage": human_leakage,
        "decisions_file": str(DECISIONS_PATH),
        "pass": passed,
        "note": (
            "Pairs >= 0.92 are candidates for human review, not automatic FAIL. "
            "Write decisions into semantic_human_decisions.json. "
            "decision='leakage' fails this detector; 'legitimate' clears the pair."
        ),
    }


# ----------------------------------------------------------------- Detector 4
def detector_scenario(train: list[dict], heldout_hashes: set[str]) -> dict:
    overlaps = []
    for r in train:
        sh = r.get("scenario_hash")
        if sh and sh in heldout_hashes:
            overlaps.append({
                "train_row_id": r.get("row_id"),
                "scenario_hash": sh,
                "fact_id": r.get("fact_id"),
                "origin": r.get("origin"),
                "qtype": r.get("qtype"),
            })
    return {
        "heldout_hash_count": len(heldout_hashes),
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "pass": len(overlaps) == 0,
        "source": str(HELDOUT_TXT),
    }


# ----------------------------------------------------------------- reports
def write_markdown(report: dict) -> str:
    d1, d2, d3, d4 = (
        report["detector_1_exact"],
        report["detector_2_jaccard"],
        report["detector_3_semantic"],
        report["detector_4_scenario"],
    )
    lines = [
        "# GATE 3 — Leakage Audit Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Normalizer: `{report['normalizer_version']}`",
        f"- Train: `{report['inputs']['train']}` "
        f"(n={report['inputs']['n_train']}, sha256={report['inputs']['train_sha256'][:16]}…)",
        f"- Val: n={report['inputs']['n_val']}  |  Test: n={report['inputs']['n_test']}  "
        f"(test sha256 verified={report['inputs']['test_sha256_ok']})",
        f"- **Overall verdict: {report['verdict']}**",
        "",
        "## Gate table",
        "",
        "| Detector | Threshold | Finding | Verdict |",
        "|----------|-----------|---------|---------|",
        f"| EXACT | n/a | count={d1['count']} | "
        f"{'PASS' if d1['pass'] else 'FAIL'} |",
        f"| JACCARD | {d2['threshold']} | "
        f"pairs≥τ={d2['n_pairs_at_or_above_threshold']}, "
        f"FAIL-rule hits={d2['n_fail_pairs']} | "
        f"{'PASS' if d2['pass'] else 'FAIL'} |",
        f"| SEMANTIC | {d3['threshold']} | "
        f"candidates={d3['n_candidates_at_or_above_threshold']}, "
        f"pending_review={d3['n_pending_human_review']}, "
        f"human_leakage={d3['n_human_marked_leakage']} | "
        f"{'PASS' if d3['pass'] else 'FAIL'} |",
        f"| SCENARIO | n/a | overlap={d4['overlap_count']} "
        f"/ heldout_hashes={d4['heldout_hash_count']} | "
        f"{'PASS' if d4['pass'] else 'FAIL'} |",
        "",
        "### Detector 2 FAIL rule (approved)",
        "",
        f"> {d2['fail_rule']}",
        "",
        "### Detector 3 human-review note",
        "",
        f"> {d3['note']}",
        "",
        f"Decisions file: `{d3['decisions_file']}`",
        "",
    ]

    # Exact pairs
    lines += ["## Detector 1 — exact matches (top 20)", ""]
    if not d1["pairs"]:
        lines.append("_None._")
    else:
        for p in d1["pairs"]:
            lines.append(
                f"- [{p['held_out_split']}] `{p['held_out_row_id']}` ↔ "
                f"`{p['train_row_id']}`  same_fact={p['same_fact_id']}"
            )
            lines.append(f"  - held: {p['held_out_question']}")
            lines.append(f"  - train: {p['train_question']}")
    lines.append("")

    # Jaccard top
    lines += ["## Detector 2 — top 20 Jaccard pairs (≥ threshold)", ""]
    top = [p for p in d2["top_50"] if p["jaccard_char5"] >= d2["threshold"]][:20]
    if not top:
        lines.append("_No pairs at or above threshold._")
    else:
        lines.append(
            "| jaccard | fail? | same_fact | same_gold | held | train |"
        )
        lines.append("|--------|-------|-----------|-----------|------|-------|")
        for p in top:
            hq = (p["held_out_question"] or "").replace("|", "/")[:60]
            tq = (p["train_question"] or "").replace("|", "/")[:60]
            lines.append(
                f"| {p['jaccard_char5']:.4f} | {p['leakage_fail']} | "
                f"{p['same_fact_id']} | {p['same_normalized_gold_answer']} | "
                f"{hq} | {tq} |"
            )
    lines.append("")
    lines.append("### Jaccard nearest-neighbour distribution")
    lines.append("")
    dist = d2["distribution"]
    lines.append(
        f"n={dist['n']}  min={dist['min']}  mean={dist['mean']}  "
        f"median={dist['median']}  p90={dist['p90']}  max={dist['max']}"
    )
    lines.append("")
    for b, c in dist["bins"].items():
        lines.append(f"- {b}: {c}")
    lines.append("")

    # Semantic
    lines += ["## Detector 3 — semantic candidates (≥ 0.92)", ""]
    if not d3["candidates"]:
        lines.append("_No pairs at or above threshold._")
    else:
        for p in d3["candidates"][:20]:
            lines.append(
                f"- cosine={p['cosine']:.4f} decision={p['human_decision']!r} "
                f"key=`{p['pair_key']}`"
            )
            lines.append(f"  - held: {p['held_out_question']}")
            lines.append(f"  - train: {p['train_question']}")
    lines.append("")
    lines.append("### Semantic nearest-neighbour distribution")
    lines.append("")
    dist = d3["distribution"]
    lines.append(
        f"n={dist['n']}  min={dist['min']}  mean={dist['mean']}  "
        f"median={dist['median']}  p90={dist['p90']}  max={dist['max']}"
    )
    lines.append("")

    # Scenario
    lines += ["## Detector 4 — scenario_hash overlaps", ""]
    if not d4["overlaps"]:
        lines.append("_None. Zero overlap._")
    else:
        for o in d4["overlaps"]:
            lines.append(
                f"- train `{o['train_row_id']}` hash=`{o['scenario_hash']}` "
                f"origin={o.get('origin')}"
            )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"**STOP. Overall verdict: {report['verdict']}. "
        "Do not proceed to Stage 4 without approval.**"
    )
    lines.append("")
    return "\n".join(lines)


def print_gate_table(report: dict) -> None:
    d1 = report["detector_1_exact"]
    d2 = report["detector_2_jaccard"]
    d3 = report["detector_3_semantic"]
    d4 = report["detector_4_scenario"]
    print()
    print("=" * 72)
    print("GATE 3 — LEAKAGE AUDIT")
    print("=" * 72)
    print(f"{'Detector':<10} {'Threshold':<12} {'Finding':<42} {'Verdict'}")
    print("-" * 72)
    print(
        f"{'EXACT':<10} {'n/a':<12} "
        f"{('count=' + str(d1['count'])):<42} "
        f"{'PASS' if d1['pass'] else 'FAIL'}"
    )
    print(
        f"{'JACCARD':<10} {str(d2['threshold']):<12} "
        f"{('FAIL-rule hits=' + str(d2['n_fail_pairs']) + ' / ≥τ=' + str(d2['n_pairs_at_or_above_threshold'])):<42} "
        f"{'PASS' if d2['pass'] else 'FAIL'}"
    )
    print(
        f"{'SEMANTIC':<10} {str(d3['threshold']):<12} "
        f"{('cand=' + str(d3['n_candidates_at_or_above_threshold']) + ' pending=' + str(d3['n_pending_human_review'])):<42} "
        f"{'PASS' if d3['pass'] else 'FAIL'}"
    )
    print(
        f"{'SCENARIO':<10} {'n/a':<12} "
        f"{('overlap=' + str(d4['overlap_count'])):<42} "
        f"{'PASS' if d4['pass'] else 'FAIL'}"
    )
    print("-" * 72)
    print(f"OVERALL VERDICT: {report['verdict']}")
    print("=" * 72)
    print("STOP. Do not proceed to Stage 4 without approval.")
    print()


# ----------------------------------------------------------------- main
def preflight() -> None:
    assert_version(NORMALIZER_VERSION)
    fails = run_regression_tests(verbose=False)
    if fails:
        raise RuntimeError(
            f"bn_normalize regression failed ({len(fails)} cases). Abort."
        )

    train_sha = sha256_file(TRAIN_PATH)
    if train_sha != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(
            f"train_augmented.jsonl SHA-256 mismatch.\n"
            f"  expected {EXPECTED_TRAIN_SHA256}\n"
            f"  got      {train_sha}"
        )

    test_sha_file = TEST_SHA_PATH.read_text(encoding="utf-8").strip().split()[0]
    test_sha = sha256_file(TEST_PATH)
    if test_sha != test_sha_file:
        raise RuntimeError(
            f"test.jsonl SHA-256 mismatch vs test.sha256.\n"
            f"  file says {test_sha_file}\n"
            f"  on disk  {test_sha}"
        )


def main(skip_semantic: bool = False) -> dict:
    preflight()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    train = load_jsonl(TRAIN_PATH)
    val = load_jsonl(VAL_PATH)
    test = load_jsonl(TEST_PATH)
    held = [("val", r) for r in val] + [("test", r) for r in test]
    heldout_hashes = ensure_heldout_hashes_txt()
    decisions = load_human_decisions()

    print(f"normalizer={NORMALIZER_VERSION}")
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  "
          f"heldout_hashes={len(heldout_hashes)}")

    print("\n[Detector 1] exact match…")
    d1 = detector_exact(train, held)
    print(f"  count={d1['count']}  pass={d1['pass']}")

    print("\n[Detector 2] char-5gram multiset Jaccard…")
    d2 = detector_jaccard(train, held)
    print(
        f"  ≥{JACCARD_THRESHOLD}: {d2['n_pairs_at_or_above_threshold']} pairs; "
        f"FAIL-rule hits={d2['n_fail_pairs']}  pass={d2['pass']}"
    )

    if skip_semantic:
        print("\n[Detector 3] SKIPPED (--skip-semantic)")
        d3 = {
            "threshold": SEMANTIC_THRESHOLD,
            "embedding_model": EMBEDDING_MODEL,
            "distribution": {},
            "n_candidates_at_or_above_threshold": 0,
            "n_pending_human_review": 0,
            "n_human_marked_leakage": 0,
            "n_human_marked_legitimate": 0,
            "candidates": [],
            "top_50": [],
            "pending_review": [],
            "human_leakage": [],
            "decisions_file": str(DECISIONS_PATH),
            "pass": True,
            "note": "SKIPPED",
            "skipped": True,
        }
    else:
        print("\n[Detector 3] semantic embedding cosine…")
        d3 = detector_semantic(train, held, decisions)
        print(
            f"  candidates≥{SEMANTIC_THRESHOLD}="
            f"{d3['n_candidates_at_or_above_threshold']}  "
            f"pending={d3['n_pending_human_review']}  "
            f"human_leakage={d3['n_human_marked_leakage']}  pass={d3['pass']}"
        )

    print("\n[Detector 4] scenario_hash blocklist…")
    d4 = detector_scenario(train, heldout_hashes)
    print(f"  overlap={d4['overlap_count']}  pass={d4['pass']}")

    verdict = (
        "PASS"
        if d1["pass"] and d2["pass"] and d3["pass"] and d4["pass"]
        else "FAIL"
    )

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "normalizer_version": NORMALIZER_VERSION,
        "seed": SEED,
        "inputs": {
            "train": str(TRAIN_PATH),
            "val": str(VAL_PATH),
            "test": str(TEST_PATH),
            "heldout_hashes": str(HELDOUT_TXT),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "train_sha256": sha256_file(TRAIN_PATH),
            "test_sha256": sha256_file(TEST_PATH),
            "test_sha256_ok": True,
            "expected_train_sha256": EXPECTED_TRAIN_SHA256,
        },
        "detector_1_exact": d1,
        "detector_2_jaccard": d2,
        "detector_3_semantic": d3,
        "detector_4_scenario": d4,
        "verdict": verdict,
    }

    OUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = write_markdown(report)
    OUT_MD.write_text(md, encoding="utf-8")
    LOG_PATH.write_text(
        f"s3_leakage_audit.py\nrun_at={report['generated_at']}\n"
        f"verdict={verdict}\n"
        f"exact={d1['count']} jaccard_fail={d2['n_fail_pairs']} "
        f"semantic_cand={d3['n_candidates_at_or_above_threshold']} "
        f"scenario_overlap={d4['overlap_count']}\n"
        f"out_json={OUT_JSON}\nout_md={OUT_MD}\n",
        encoding="utf-8",
    )

    print_gate_table(report)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="v3 Stage 3 leakage audit")
    ap.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip Detector 3 (embedding) for a fast dry run",
    )
    args = ap.parse_args()
    main(skip_semantic=args.skip_semantic)
