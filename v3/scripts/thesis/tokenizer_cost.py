#!/usr/bin/env python3
"""
tokenizer_cost.py — descriptive tokenizer cost of Bengali relative to English.

For every test row that has a bn/en partner via fact_id (exactly one bn and one
en row per fact_id), tokenise the QUESTION TEXT ONLY (no chat template, no
special tokens) with each model's local tokenizer and compute, per pair,
    ratio = bn tokens / en tokens.
Report the median ratio with a percentile bootstrap 95% CI (10,000 resamples of
pairs, numpy default_rng(42), reset for each model), and pooled tokens per
character for each language (sum of tokens / sum of NFC code points).

Descriptive only; not a Stage 8 test. Reads the locked test set after
verifying its SHA-256, and the tokenizers from the local Hugging Face cache.
Writes thesis/generated/tokenizer_cost.json and tokenizer_cost_pairs.csv.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import unicodedata
import warnings
from pathlib import Path

import numpy as np

from common import GEN, MODELS, TEST_PATH, load_jsonl, load_tokenizer, rel, verify_test_sha, write_json

N_BOOT = 10_000
SEED = 42


def pairs_by_fact_id(rows: list[dict]) -> list[tuple[dict, dict]]:
    g = collections.defaultdict(list)
    for r in rows:
        g[r["fact_id"]].append(r)
    out = []
    for fid in sorted(g):
        rs = g[fid]
        bn = [r for r in rs if r["lang"] == "bn"]
        en = [r for r in rs if r["lang"] == "en"]
        if len(bn) == 1 and len(en) == 1:
            assert bn[0]["qtype"] == en[0]["qtype"], fid
            out.append((bn[0], en[0]))
    return out


def ntok(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def nchar(text: str) -> int:
    return len(unicodedata.normalize("NFC", text))


def boot_median_ci(x: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def tokenizer_fingerprint(tok) -> dict:
    fp = {"class": type(tok).__name__, "name_or_path": tok.name_or_path, "vocab_size_with_added": len(tok)}
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(tok.name_or_path, "tokenizer.json")
        if isinstance(p, str):
            fp["tokenizer_json_sha256"] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception as e:  # noqa: BLE001
        fp["tokenizer_json_sha256"] = f"unavailable: {type(e).__name__}"
    return fp


def main() -> None:
    sha = verify_test_sha()
    rows = load_jsonl(TEST_PATH)
    pairs = pairs_by_fact_id(rows)
    by_qtype = collections.Counter(bn["qtype"] for bn, _ in pairs)
    for q in ("factual", "numeric_runtime", "abstain", "clarify"):
        by_qtype.setdefault(q, 0)

    result = {"test_sha256": sha, "n_test_rows": len(rows), "n_pairs": len(pairs),
              "pairs_by_qtype": dict(sorted(by_qtype.items())),
              "pairing_rule": "fact_id groups with exactly one bn and one en row (same qtype)",
              "text": "question field only; no chat template; add_special_tokens=False",
              "chars": "len(unicodedata.normalize('NFC', text)) (Unicode code points)",
              "ci": f"percentile bootstrap over pairs, {N_BOOT} resamples, numpy default_rng({SEED}) per model",
              "models": {}}
    pair_rows = []
    for key, stem, label in MODELS:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tok = load_tokenizer(key)
        tb = np.array([ntok(tok, bn["question"]) for bn, _ in pairs], dtype=float)
        te = np.array([ntok(tok, en["question"]) for _, en in pairs], dtype=float)
        cb = np.array([nchar(bn["question"]) for bn, _ in pairs], dtype=float)
        ce = np.array([nchar(en["question"]) for _, en in pairs], dtype=float)
        ratio = tb / te
        lo, hi = boot_median_ci(ratio)
        per_q = {}
        for q in sorted(by_qtype):
            m = np.array([bn["qtype"] == q for bn, _ in pairs])
            if m.any():
                per_q[q] = {"n": int(m.sum()), "median_ratio": float(np.median(ratio[m]))}
        result["models"][key] = {
            "label": label, "tokenizer": tokenizer_fingerprint(tok),
            "load_warnings": sorted({str(x.message)[:200] for x in w}),
            "median_ratio_bn_over_en": float(np.median(ratio)), "ci95_low": lo, "ci95_high": hi,
            "mean_ratio_bn_over_en": float(np.mean(ratio)),
            "min_ratio": float(ratio.min()), "max_ratio": float(ratio.max()),
            "tokens_per_char_bn": float(tb.sum() / cb.sum()),
            "tokens_per_char_en": float(te.sum() / ce.sum()),
            "median_tokens_bn": float(np.median(tb)), "median_tokens_en": float(np.median(te)),
            "by_qtype": per_q,
        }
        for (bn, en), a, b, c, d in zip(pairs, tb, te, cb, ce):
            pair_rows.append({"model": key, "fact_id": bn["fact_id"], "qtype": bn["qtype"],
                              "bn_row_id": bn["row_id"], "en_row_id": en["row_id"],
                              "bn_tokens": int(a), "en_tokens": int(b), "bn_chars": int(c), "en_chars": int(d),
                              "ratio": round(float(a / b), 6)})
        print(f"{label:18s} median ratio {np.median(ratio):.3f} [{lo:.3f}, {hi:.3f}]  "
              f"tok/char bn {tb.sum() / cb.sum():.3f} en {te.sum() / ce.sum():.3f}")

    write_json(GEN / "tokenizer_cost.json", result)
    with (GEN / "tokenizer_cost_pairs.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(pair_rows[0]))
        wr.writeheader()
        wr.writerows(pair_rows)
    print(f"pairs: {len(pairs)} {dict(sorted(by_qtype.items()))} -> {rel(GEN / 'tokenizer_cost.json')}")


if __name__ == "__main__":
    main()
