#!/usr/bin/env python3
"""
gen_tokens.py — per-row generated-token counts for every Stage 6 arm.

Method is identical to the pre-registered Stage 8 A7 descriptive
(scripts/s8_stats.py gen_token_counts): re-encode the stored generation with
the model's own tokenizer, add_special_tokens=False. No finish_reason was stored
at generation time, so a row counts as truncated when it re-encodes to
>= GEN_MAX_NEW_TOKENS - 1 tokens (the pre-registered threshold).

Only the 362 scorable rows are counted. Output:
  thesis/generated/gen_tokens.csv        one row per (arm, row_id)
  thesis/generated/gen_tokens_check.json reproduction check against the point
                                          values in reports/stats/a7_descriptive.csv
                                          and reports/stats/precheck.json
Descriptive only.
"""
from __future__ import annotations

import csv
import json
import statistics

import pandas as pd

from common import (GEN, MODELS, PRED_DIR, ROOT, SEEDS, TEST_PATH, guard, load_jsonl, load_tokenizer,
                    verify_test_sha, write_json)
import s6_eval

CAP = s6_eval.GEN_MAX_NEW_TOKENS
THRESH = CAP - 1


def arms_for(key: str) -> list[tuple[str, str, int | None]]:
    out = [(f"{key}_zeroshot", "zeroshot", None)]
    out += [(f"{key}_seed{s}", "finetuned", s) for s in SEEDS]
    out.append((f"{key}_seed42_clarify", "clarify", 42))
    return out


def main() -> None:
    verify_test_sha()
    test = {r["row_id"]: r for r in load_jsonl(TEST_PATH)}
    scorable = {i for i, r in test.items() if r.get("gold_status") != "pending_human"}
    rows = []
    for key, _, _ in MODELS:
        tok = load_tokenizer(key)
        for arm, variant, seed in arms_for(key):
            preds = [p for p in load_jsonl(PRED_DIR / f"{arm}.jsonl") if p["row_id"] in scorable]
            assert len(preds) == len(scorable), arm
            for p in preds:
                g = p.get("generation") or ""
                n = len(tok(g, add_special_tokens=False)["input_ids"])
                t = test[p["row_id"]]
                rows.append({"arm": arm, "model": key, "variant": variant, "seed": seed if seed else "",
                             "row_id": p["row_id"], "qtype": t["qtype"], "category": t["category"],
                             "subtype": t.get("subtype") or "", "lang": t["lang"],
                             "gen_tokens": n, "gen_chars": len(g), "truncated": int(n >= THRESH)})
        print(f"{key}: done")
    rows.sort(key=lambda r: (r["arm"], r["row_id"]))
    with (GEN / "gen_tokens.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- reproduction check against the Stage 8 descriptive point values
    df = pd.DataFrame(rows)
    a7 = pd.read_csv(guard(ROOT / "reports" / "stats" / "a7_descriptive.csv"))
    checks = []
    for _, r in a7.iterrows():
        arm = r["arm"]
        sub = df[(df["arm"] == arm) & (df["qtype"] == "clarify")]
        trunc = sub["truncated"].mean()
        med = float(statistics.median(sub["gen_tokens"]))
        checks.append({"arm": arm, "n_clarify": len(sub), "a7_n": int(r["n_clarify"]),
                       "truncation_rate": trunc, "a7_truncation_rate": float(r["truncation_rate"]),
                       "median_gen_tokens": med, "a7_median_gen_tokens": float(r["median_gen_tokens"]),
                       "match": bool(len(sub) == r["n_clarify"] and abs(trunc - r["truncation_rate"]) < 1e-12
                                     and abs(med - r["median_gen_tokens"]) < 1e-9)})
    pc = json.loads(guard(ROOT / "reports" / "stats" / "precheck.json").read_text(encoding="utf-8"))
    for arm, d in pc["token_count_distribution"].items():
        sub = df[df["arm"] == arm]
        cl = sub[sub["qtype"] == "clarify"]
        got = {"n": len(sub), "median": float(statistics.median(sub["gen_tokens"])), "max": int(sub["gen_tokens"].max()),
               "n_ge_511": int((sub["gen_tokens"] >= 511).sum()), "clarify_rows_ge_511": int((cl["gen_tokens"] >= 511).sum())}
        checks.append({"arm": arm, "precheck": {k: d[k] for k in got}, "recomputed": got,
                       "match": all(got[k] == d[k] for k in got)})
    summary = {"threshold_tokens": THRESH, "cap_tokens": CAP, "n_rows": len(df),
               "all_match": all(c["match"] for c in checks), "checks": checks}
    write_json(GEN / "gen_tokens_check.json", summary)
    print("reproduction check:", "PASS" if summary["all_match"] else "FAIL")
    for c in checks:
        if not c["match"]:
            print("  MISMATCH", c)


if __name__ == "__main__":
    main()
