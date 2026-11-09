#!/usr/bin/env python3
"""Build bilingual Stage-7-style sample: same items in bn and en.

Strict pairing:
  - seed rows: shared fact_id
  - adversarial: shared (category, subtype, crop) with exactly 1 bn + 1 en

Quotas requested: 35 numeric / 35 factual / 20 clarify / 10 abstain per language.
Reports shortfall if the locked test set cannot supply that many true pairs.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
TEST = PROJECT / "data" / "hardened" / "test_hardened.jsonl"
OUT_DIR = PROJECT / "data" / "human_eval" / "bilingual_100"
SEED = 42

REQUESTED = {
    "numeric_runtime": 35,
    "factual": 35,
    "clarify": 20,
    "abstain": 10,
}


def load_scorable():
    rows = [
        json.loads(l)
        for l in TEST.open(encoding="utf-8")
        if l.strip()
    ]
    return [
        r
        for r in rows
        if r.get("gold_status") != "pending_human" and str(r.get("answer", "")).strip()
    ]


def find_pairs(scorable: list[dict]) -> dict[str, list[tuple[dict, dict]]]:
    """Return qtype -> list of (en_row, bn_row) true pairs."""
    by_qtype: dict[str, list[tuple[dict, dict]]] = defaultdict(list)

    # 1) fact_id pairs (seed + any others that share fact_id)
    by_fact: dict[str, list[dict]] = defaultdict(list)
    for r in scorable:
        if r.get("fact_id"):
            by_fact[r["fact_id"]].append(r)

    used_ids: set[str] = set()
    for fid, rs in by_fact.items():
        bn = [x for x in rs if x["lang"] == "bn"]
        en = [x for x in rs if x["lang"] == "en"]
        if len(bn) == 1 and len(en) == 1 and bn[0]["qtype"] == en[0]["qtype"]:
            by_qtype[bn[0]["qtype"]].append((en[0], bn[0]))
            used_ids.add(en[0]["row_id"])
            used_ids.add(bn[0]["row_id"])

    # 2) adversarial: unique (category, subtype, crop) with one bn + one en
    rem = [r for r in scorable if r["row_id"] not in used_ids]
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in rem:
        key = (r.get("category"), r.get("subtype"), r.get("crop"), r["qtype"])
        # skip crop=None ambiguous buckets (many unrelated items)
        if r.get("crop") is None:
            continue
        by_key[key].append(r)

    for key, rs in by_key.items():
        bn = [x for x in rs if x["lang"] == "bn"]
        en = [x for x in rs if x["lang"] == "en"]
        if len(bn) == 1 and len(en) == 1:
            qtype = key[3]
            by_qtype[qtype].append((en[0], bn[0]))
            used_ids.add(en[0]["row_id"])
            used_ids.add(bn[0]["row_id"])

    return dict(by_qtype)


def sample_pairs(
    available: dict[str, list[tuple[dict, dict]]],
    requested: dict[str, int],
    seed: int,
) -> tuple[list[tuple[dict, dict, str]], dict]:
    rng = random.Random(seed)
    selected: list[tuple[dict, dict, str]] = []
    report = {}

    for qtype, n_want in requested.items():
        pool = list(available.get(qtype, []))
        rng.shuffle(pool)
        n_take = min(n_want, len(pool))
        taken = pool[:n_take]
        for en, bn in taken:
            selected.append((en, bn, qtype))
        report[qtype] = {
            "requested": n_want,
            "available_pairs": len(pool),
            "taken": n_take,
            "shortfall": max(0, n_want - n_take),
        }
    return selected, report


def main():
    scorable = load_scorable()
    available = find_pairs(scorable)

    print("Available TRUE bilingual pairs in 362 scorable:")
    for q in ["numeric_runtime", "factual", "clarify", "abstain"]:
        print(f"  {q:20s} {len(available.get(q, [])):3d}  (want {REQUESTED[q]})")

    selected, report = sample_pairs(available, REQUESTED, SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bn_path = OUT_DIR / "sample_100_bn.jsonl"
    en_path = OUT_DIR / "sample_100_en.jsonl"
    pairs_path = OUT_DIR / "sample_pairs.jsonl"
    manifest_path = OUT_DIR / "sample_manifest.json"
    csv_path = OUT_DIR / "sample_pairs.csv"

    bn_rows, en_rows, pair_rows = [], [], []
    for i, (en, bn, qtype) in enumerate(selected, start=1):
        pair_id = f"pair_{i:03d}"
        en_out = {**en, "pair_id": pair_id, "pair_lang": "en", "sample_qtype": qtype}
        bn_out = {**bn, "pair_id": pair_id, "pair_lang": "bn", "sample_qtype": qtype}
        en_rows.append(en_out)
        bn_rows.append(bn_out)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "qtype": qtype,
                "fact_id": en.get("fact_id") or bn.get("fact_id"),
                "crop": en.get("crop") or bn.get("crop"),
                "category": en.get("category"),
                "subtype": en.get("subtype"),
                "row_id_en": en["row_id"],
                "row_id_bn": bn["row_id"],
                "item_id_en": en.get("item_id"),
                "item_id_bn": bn.get("item_id"),
                "question_en": en["question"],
                "question_bn": bn["question"],
                "gold_en": en.get("answer"),
                "gold_bn": bn.get("answer"),
            }
        )

    def write_jsonl(path, rows):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(bn_path, bn_rows)
    write_jsonl(en_path, en_rows)
    write_jsonl(pairs_path, pair_rows)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()) if pair_rows else [])
        w.writeheader()
        w.writerows(pair_rows)

    n_pairs = len(selected)
    manifest = {
        "seed": SEED,
        "source": str(TEST),
        "requested_per_lang": REQUESTED,
        "requested_total_rows": 200,
        "achieved_pairs": n_pairs,
        "achieved_rows": n_pairs * 2,
        "achieved_per_lang": n_pairs,
        "pairing_rule": (
            "Same underlying item in both languages: "
            "(1) shared fact_id, or "
            "(2) adversarial unique (category, subtype, crop) with 1 bn + 1 en"
        ),
        "by_qtype": report,
        "qtype_counts_taken": dict(Counter(q for _, _, q in selected)),
        "outputs": {
            "bn": str(bn_path),
            "en": str(en_path),
            "pairs": str(pairs_path),
            "csv": str(csv_path),
        },
        "note": (
            "The locked test set does not contain 35 numeric or 20 clarify "
            "true bilingual pairs. Shortfalls are reported in by_qtype."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SAMPLE RESULT (seed=42) ===")
    for q, info in report.items():
        print(
            f"  {q:20s} taken={info['taken']:2d}/{info['requested']}  "
            f"available={info['available_pairs']}  shortfall={info['shortfall']}"
        )
    print(f"\nPairs written: {n_pairs}  (= {n_pairs} bn + {n_pairs} en rows)")
    print(f"BN: {bn_path}")
    print(f"EN: {en_path}")
    print(f"Pairs CSV: {csv_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
