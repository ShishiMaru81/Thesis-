#!/usr/bin/env python3
"""
s6b_merge_clarify_train.py — merge clarify examples into the training corpus.

Output: data/augmented/train_augmented_clarify.jsonl (3,852 rows)
        data/rendered/train_clarify.jsonl (prompt/completion for s5_train)

Run:  python scripts/s6b_merge_clarify_train.py
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_ORIG = PROJECT_ROOT / "data" / "augmented" / "train_augmented.jsonl"
CLARIFY_PATH = PROJECT_ROOT / "data" / "augmented" / "s6b_clarify_train.jsonl"
OUT_AUG = PROJECT_ROOT / "data" / "augmented" / "train_augmented_clarify.jsonl"
OUT_RENDERED = PROJECT_ROOT / "data" / "rendered" / "train_clarify.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports" / "s6b" / "merge_report.json"

META_KEYS = ["row_id", "qtype", "answer_kind", "lang", "crop", "origin",
             "fact_id", "scenario_hash", "subtype", "missing_slot"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_rows(rows: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        q, a = r.get("question"), r.get("answer")
        if not isinstance(q, str) or not q.strip():
            raise SystemExit(f"row {i}: empty question")
        if not isinstance(a, str) or not a.strip():
            raise SystemExit(f"row {i}: empty answer")
        rec = {
            "prompt": [{"role": "user", "content": q}],
            "completion": [{"role": "assistant", "content": a}],
        }
        for k in META_KEYS:
            if k in r:
                rec[k] = r[k]
        out.append(rec)
    return out


def main() -> int:
    for p in (TRAIN_ORIG, CLARIFY_PATH):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    orig = load_jsonl(TRAIN_ORIG)
    clarify = load_jsonl(CLARIFY_PATH)
    merged = orig + clarify

    ids = [r["row_id"] for r in merged]
    if len(ids) != len(set(ids)):
        dupes = [k for k, v in collections.Counter(ids).items() if v > 1]
        raise SystemExit(f"row_id collision: {dupes[:10]}")

    if len(merged) != 3852:
        raise SystemExit(f"expected 3852 rows, got {len(merged)}")

    by_qtype = dict(collections.Counter(r.get("qtype") for r in merged))
    by_lang = dict(collections.Counter(r.get("lang") for r in merged))

    OUT_AUG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_AUG.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rendered = render_rows(merged)
    OUT_RENDERED.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RENDERED.open("w", encoding="utf-8") as f:
        for r in rendered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sha = sha256_of(OUT_AUG)
    report = {
        "source_rows": len(orig),
        "clarify_rows": len(clarify),
        "total": len(merged),
        "qtype_distribution": by_qtype,
        "language_distribution": by_lang,
        "sha256": sha,
        "rendered_sha256": sha256_of(OUT_RENDERED),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "output_augmented": str(OUT_AUG),
        "output_rendered": str(OUT_RENDERED),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    bn = by_lang.get("bn", 0)
    en = by_lang.get("en", 0)
    pct_clarify = 100.0 * len(clarify) / len(merged)

    print("\n" + "=" * 60)
    print("s6b_merge_clarify_train — REPORT")
    print("=" * 60)
    print(f"Total training rows : {len(merged)} (up from {len(orig)})")
    print(f"Clarify rows        : {len(clarify)} ({pct_clarify:.1f}% of corpus)")
    print(f"Language            : {bn} bn / {en} en "
          f"({'50/50 OK' if bn == en else 'MISMATCH'})")
    print(f"qtype distribution  : {by_qtype}")
    print(f"SHA-256             : {sha}")
    print(f"-> {OUT_AUG}")
    print(f"-> {OUT_RENDERED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
