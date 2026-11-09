#!/usr/bin/env python3
"""
preflight_seqlen.py — CHECK 1: how long are the rendered training sequences,
per base model, and what max_seq_len can each model safely use?

Measures the sequence the TRAINER actually builds, not a raw string concat:
trl tokenizes prompt+completion through the model's chat template, so the
BOS/EOS/role headers count against max_seq_len. Those are worth 20-40 tokens
per record and differ per model family, which is exactly the margin that
decides 768 vs 1024. Falls back to plain concat only if a tokenizer ships no
chat template.

Input: data/rendered/train.jsonl (produced by s2b_render_train.py).
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_ROOT / "data" / "rendered" / "train.jsonl"
OUT = PROJECT_ROOT / "reports" / "eval" / "preflight_seqlen.json"

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-4-mini-instruct",
]

print(f"Loading {CORPUS.relative_to(PROJECT_ROOT)} ...")
if not CORPUS.exists():
    raise SystemExit(f"missing {CORPUS} - run scripts/s2b_render_train.py first")
rows = [json.loads(l) for l in CORPUS.open(encoding="utf-8") if l.strip()]
print(f"Loaded {len(rows)} rows\n")

convs = [r["prompt"] + r["completion"] for r in rows]
flat = [r["prompt"][0]["content"] + r["completion"][0]["content"] for r in rows]

results = {}
print("Tokenizing across 4 models (chat-template applied)...")
for name in MODELS:
    print(f"  {name} ...", end=" ", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(name)
    except Exception as e:
        print(f"SKIP ({type(e).__name__}: {str(e)[:120]})")
        results[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        continue

    templated = tok.chat_template is not None
    if templated:
        # Render to text first, then tokenize with add_special_tokens=False.
        # apply_chat_template(tokenize=True) returns a BatchEncoding in
        # transformers 5.x, so len() on it counts dict keys, not tokens; and the
        # template already emits BOS as text, so re-adding specials double-counts.
        lens = [len(tok(tok.apply_chat_template(c, tokenize=False, add_generation_prompt=False),
                        add_special_tokens=False)["input_ids"])
                for c in convs]
    else:
        lens = [len(tok(t)["input_ids"]) for t in flat]

    lens_sorted = sorted(lens)
    p99 = lens_sorted[int(len(lens_sorted) * 0.99)]
    over_768 = sum(1 for x in lens if x > 768)
    over_1024 = sum(1 for x in lens if x > 1024)
    results[name] = {
        "chat_template": templated,
        "max": max(lens),
        "p50": lens_sorted[len(lens_sorted) // 2],
        "p99": p99,
        "over_768": over_768,
        "over_768_pct": round(100 * over_768 / len(lens), 2),
        "over_1024": over_1024,
        "over_1024_pct": round(100 * over_1024 / len(lens), 2),
    }
    print(f"max={max(lens)}, p99={p99}, over_768={over_768}, over_1024={over_1024}")

print("\n" + "=" * 84)
print("SEQUENCE LENGTH AUDIT RESULTS")
print("=" * 84)
print(f"{'Model':<34} | {'Max':>5} | {'P50':>5} | {'P99':>5} | {'%>768':>7} | {'>1024':>5}")
print("-" * 84)
for name in MODELS:
    r = results[name]
    short = name.split("/")[-1][:33]
    if "error" in r:
        print(f"{short:<34} |  --- tokenizer unavailable ---")
        continue
    print(f"{short:<34} | {r['max']:>5} | {r['p50']:>5} | {r['p99']:>5} "
          f"| {r['over_768_pct']:>7} | {r['over_1024']:>5}")

# ---------------------------------------------------------------- decisions
ok = {k: v for k, v in results.items() if "error" not in v}


def decide(stats: dict) -> tuple[str, str]:
    if stats["over_1024"] == 0 and stats["over_768"] == 0:
        return "768", "no row exceeds 768 - zero truncation at either setting"
    if stats["over_1024"] == 0:
        return "1024", f"{stats['over_768']} rows exceed 768 but none exceed 1024"
    if stats["over_1024_pct"] < 1.0:
        return "1024", f"{stats['over_1024_pct']}% exceed 1024 - small truncation risk"
    return "1024_insufficient", f"{stats['over_1024_pct']}% exceed 1024"


print("\n" + "=" * 84)
print("MAX_SEQ_LEN DECISION (per model)")
print("=" * 84)
decisions = {}
for name, stats in ok.items():
    val, why = decide(stats)
    decisions[name] = val
    print(f"  {name.split('/')[-1]:<34} -> {val:<18} ({why})")

mistral_key = "mistralai/Mistral-Nemo-Instruct-2407"
mistral_decision = decisions.get(mistral_key, "unknown")
print(f"\nMistral-Nemo (the VRAM-tightest model) -> max_seq_len={mistral_decision}")

global_max = max((s["max"] for s in ok.values()), default=0)
all_fit_768 = all(s["over_768"] == 0 for s in ok.values()) if ok else False
verdict = "PASS" if all_fit_768 or all(s["over_1024"] == 0 for s in ok.values()) else "REVIEW"

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    json.dump({
        "timestamp": datetime.datetime.now().isoformat(),
        "corpus": str(CORPUS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "rows": len(rows),
        "measurement": "len(apply_chat_template(prompt+completion, tokenize=True))",
        "results": results,
        "decisions": decisions,
        "mistral_decision": mistral_decision,
        "global_max_tokens": global_max,
        "all_models_fit_768": all_fit_768,
        "verdict": verdict,
    }, f, indent=2)

print(f"\nResults saved to {OUT.relative_to(PROJECT_ROOT)}")
print(f"Global max across all models: {global_max} tokens")
print(f"\nCHECK 1 {verdict}")
