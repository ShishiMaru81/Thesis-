#!/usr/bin/env python3
"""Stage 7 — generate model answers for the 18 authored-BN sample rows.

These rows are bilingual-sample translations that do not exist in the locked
test set, so no Stage 6 prediction file covers them. This runs the same
inference path as s6_eval (same prompt shape, greedy decoding) for the four
Stage 7 arms and writes reports/s7/predictions_translated_bn/{arm}.jsonl.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

import s6_eval  # noqa: E402

SAMPLE_BN = PROJECT / "data" / "human_eval" / "bilingual_100" / "sample_100_bn.jsonl"
OUT_DIR = PROJECT / "reports" / "s7" / "predictions_translated_bn"

DEFAULT_ARMS = [
    "llama31_8b_zeroshot",
    "llama31_8b_seed42",
    "llama31_8b_seed42_clarify",
    "mistral_nemo_seed42",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for noisy in ("httpx", "huggingface_hub", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    rows = [json.loads(l) for l in SAMPLE_BN.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("translation_status") == "authored_bn_pair"]
    if not rows:
        raise SystemExit("no authored_bn_pair rows found in sample_100_bn.jsonl")
    print(f"translated BN rows to generate: {len(rows)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        out_path = OUT_DIR / f"{arm}.jsonl"
        if out_path.exists():
            existing = [json.loads(l) for l in out_path.open(encoding="utf-8") if l.strip()]
            if {p["row_id"] for p in existing} == {r["row_id"] for r in rows}:
                print(f"[{arm}] already complete -> skip")
                continue
        preds, meta = s6_eval.generate_arm(
            arm, rows, args.batch_size,
            s6_eval.GEN_MAX_NEW_TOKENS, s6_eval.INFER_MAX_SEQ_LEN,
        )
        with out_path.open("w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        (OUT_DIR / f"{arm}_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[{arm}] wrote {len(preds)} -> {out_path.relative_to(PROJECT)}")

    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
