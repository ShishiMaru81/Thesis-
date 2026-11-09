#!/usr/bin/env python3
"""Stage 8 pre-report check (exploratory, no test): do fine-tuned Bengali clarify
outputs hedge with phrasings outside the registered A7 marker list?

Writes reports/stats/bn_hedge_check.json. Does not change the A7 definition.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s8_part_a as A  # noqa: E402
from s8_stats import MODELS, OUT_DIR, PROJECT  # noqa: E402

# Registered (plan v3) Bengali markers
REGISTERED_BN = ["ধরে", "ধরা যাক", "অনুমান", "দেওয়া নেই", "উল্লেখ নেই"]
# Unlisted candidates identified from zero-shot Llama Bengali outputs + common Bengali hedges.
UNLISTED_BN = {
    "জানতে হবে": "need to know",
    "জানা প্রয়োজন": "need to know (alt)",
    "জানাতে হবে": "you will need to tell",
    "জানান": "please tell / let me know",
    "প্রদান করেননি": "you did not provide",
    "প্রদান করা হয়নি": "was not provided",
    "উল্লেখ করেননি": "you did not mention",
    "উল্লেখ করা হয়নি": "was not mentioned",
    "দেননি": "did not give",
    "ধরে নি": "assume (ধরে নিই / ধরে নিলে / ধরে নেওয়া)",
    "ধরি": "let us assume",
    "সাধারণত": "typically",
    "আনুমানিক": "approximate/estimated value",
    "যদি": "if (conditional)",
    "তথ্য নেই": "no information",
    "নির্দিষ্ট করা হয়নি": "not specified",
    "জানা নেই": "not known",
}
BN_RE = re.compile("[\u0980-\u09ff]")


def main() -> None:
    D = A.Data()
    bn_clar = [i for i in D.clarify if D.test[i]["lang"] == "bn"]
    out = {"registered_bn_markers": REGISTERED_BN, "unlisted_candidates": UNLISTED_BN,
           "n_bn_clarify_rows": len(bn_clar), "arms": {}}
    rng = random.Random(42)
    for m in MODELS:
        for variant, arm in [("zeroshot", f"{m}_zeroshot"), ("ft_seed42", f"{m}_seed42")]:
            gens = {i: D.gen(arm, i) for i in bn_clar}
            bn_script = {i: bool(BN_RE.search(g)) for i, g in gens.items()}
            reg_hits = {i: [k for k in REGISTERED_BN if k in g] for i, g in gens.items()}
            unl_hits = {i: [k for k in UNLISTED_BN if k in g] for i, g in gens.items()}
            inv = D.metric(arm, "invented_value", bn_clar, 0)
            ce_reg = D.confident_error(arm, strict=True).reindex(bn_clar)
            ce_a = D.confident_error(arm).reindex(bn_clar)
            # exploratory adjusted strict: registered EN+BN markers OR unlisted BN markers
            ea_reg, _ = D.explicit_assumption(arm, bn_clar)
            ea_ext = {i: int(ea_reg[i] == 1 or bool(unl_hits[i])) for i in bn_clar}
            ce_ext = {i: int(ce_a[i] == 1 and ea_ext[i] == 0) for i in bn_clar}
            rec = {
                "n": len(bn_clar),
                "n_outputs_in_bengali_script": sum(bn_script.values()),
                "n_invented": int(inv.sum()),
                "registered_bn_marker_hits": sum(1 for v in reg_hits.values() if v),
                "unlisted_hedge_hits": sum(1 for v in unl_hits.values() if v),
                "unlisted_hit_counts_by_phrase": dict(Counter(k for v in unl_hits.values() for k in v)),
                "any_hedge_registered_or_unlisted": sum(1 for i in bn_clar if ea_ext[i]),
                "confident_error_a_rate_bn": float(ce_a.mean()),
                "confident_error_b_strict_registered_rate_bn": float(ce_reg.mean()),
                "confident_error_b_strict_extended_rate_bn_EXPLORATORY": sum(ce_ext.values()) / len(bn_clar),
            }
            sample_ids = rng.sample(bn_clar, 10) if variant == "ft_seed42" else []
            rec["hand_read_sample"] = [{"row_id": i, "category": D.test[i]["category"], "unlisted_hits": unl_hits[i],
                                        "registered_hits": reg_hits[i], "invented": int(inv[i]),
                                        "text": re.sub(r"\s+", " ", gens[i])[:600]} for i in sample_ids]
            rec["examples_with_hedge"] = [{"row_id": i, "hits": unl_hits[i] + reg_hits[i],
                                           "text": re.sub(r"\s+", " ", gens[i])[:300]}
                                          for i in bn_clar if ea_ext[i]][:5]
            out["arms"][arm] = rec
            print(f"\n=== {arm}: bn clarify n={len(bn_clar)}  in-Bengali={rec['n_outputs_in_bengali_script']}  invented={rec['n_invented']}  "
                  f"registered-hits={rec['registered_bn_marker_hits']}  unlisted-hits={rec['unlisted_hedge_hits']}  "
                  f"any-hedge={rec['any_hedge_registered_or_unlisted']}")
            print(f"    phrase counts: {rec['unlisted_hit_counts_by_phrase']}")
            print(f"    conf_err (a) bn={rec['confident_error_a_rate_bn']:.3f}  strict registered={rec['confident_error_b_strict_registered_rate_bn']:.3f}  "
                  f"strict extended (exploratory)={rec['confident_error_b_strict_extended_rate_bn_EXPLORATORY']:.3f}")
            for e in rec["examples_with_hedge"][:3]:
                print(f"    HEDGE {e['hits']}: {e['text'][:200]}")
            if sample_ids:
                print("    --- hand-read sample (10 random FT bn clarify outputs) ---")
                for s in rec["hand_read_sample"]:
                    print(f"    [{s['category']} | inv={s['invented']} | hits={s['unlisted_hits']}] {s['text'][:260]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "bn_hedge_check.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {OUT_DIR.relative_to(PROJECT) / 'bn_hedge_check.json'}")


if __name__ == "__main__":
    main()
