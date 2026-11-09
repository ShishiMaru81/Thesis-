#!/usr/bin/env python3
"""
probe_v2_model.py — Stage 6 parameter-echo sanity probe, run BEFORE building the
tool-call parser (Stage 5). Loads models/v2_run/merged_model/ and runs six fixed
prompts with greedy decoding, printing raw output verbatim. No parsing, no
scoring — that's Stage 5/7's job. This script only answers: does the model
extract stated numbers, or substitute memorized ones?

Run:  python scripts/probe_v2_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_DIR = PROJECT_ROOT / "models" / "v2_run" / "merged_model"
MAX_LEN = 1024

PROMPTS = [
    {
        "label": "Prompt 1 — IN-DISTRIBUTION numeric (from train.jsonl id=Jute_numeric_runtime_en_dad04ac8ac9a)",
        "text": "Need the water figure for Jute — initial stage, ET0 3.8 mm, rain 35.4 mm, area 5 decimal.",
        "kind": "numeric",
        "stated": {"crop": "Jute", "stage": "initial", "et0_mm": 3.8, "rainfall_mm": 35.4,
                   "area_value": 5.0, "area_unit": "decimal"},
    },
    {
        "label": "Prompt 2 — OUT-OF-DISTRIBUTION numeric (tuple verified absent from all_v2.jsonl's 900 tuples)",
        "text": "My Boro Rice field is at the mid-season stage. Today ET0 is 3.7 mm, it rained 42.3 mm, "
                "and the field is 0.5 ha. How much irrigation water does it need today?",
        "kind": "numeric",
        "stated": {"crop": "Boro Rice", "stage": "mid-season", "et0_mm": 3.7, "rainfall_mm": 42.3,
                   "area_value": 0.5, "area_unit": "ha"},
    },
    {
        "label": "Prompt 3 — OUT-OF-DISTRIBUTION numeric, Bengali (same tuple as Prompt 2)",
        "text": "আমার বোরো ধান ক্ষেত মধ্য-মৌসুম পর্যায়ে আছে। আজ ET0 ৩.৭ মিমি, বৃষ্টি হয়েছে ৪২.৩ মিমি, "
                "জমির পরিমাণ ০.৫ হেক্টর। আজ কত পানি সেচ দিতে হবে?",
        "kind": "numeric",
        "stated": {"crop": "Boro Rice", "stage": "mid-season", "et0_mm": 3.7, "rainfall_mm": 42.3,
                   "area_value": 0.5, "area_unit": "ha"},
    },
    {
        "label": "Prompt 4 — factual (must NOT emit <irrigation_calc>)",
        "text": "When should Boro rice be irrigated at the boot stage?",
        "kind": "factual",
        "stated": None,
    },
    {
        "label": "Prompt 5 — out-of-domain (must emit abstention sentinel)",
        "text": "What is the current market price of urea fertilizer in Dhaka?",
        "kind": "abstain",
        "stated": None,
    },
    {
        "label": "Prompt 6 — adversarial / incomplete (must NOT invent numbers)",
        "text": "How much water does my rice field need?",
        "kind": "adversarial",
        "stated": None,
    },
]


def main() -> int:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        est = "seconds per prompt (GPU) — six prompts should complete in well under 5 minutes."
    else:
        est = ("~15 s/token on CPU; six prompts at ~150-250 tokens each "
               "could take 30-45 minutes total.")
    print(f"Runtime estimate: {est}\n")

    print(f"Loading merged model from {MERGED_DIR} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MERGED_DIR),
        max_seq_length=MAX_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded.\n")

    results = []
    for p in PROMPTS:
        print("=" * 88)
        print(p["label"])
        print("=" * 88)
        print(f"USER: {p['text']}")
        if p["stated"]:
            print(f"stated params: {json.dumps(p['stated'], ensure_ascii=False)}")

        messages = [{"role": "user", "content": p["text"]}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids=inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs.shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        raw_with_special = tokenizer.decode(new_tokens, skip_special_tokens=False)

        print(f"\nRAW OUTPUT (skip_special_tokens=True):\n{raw!r}")
        print(f"\nRAW OUTPUT (with special tokens):\n{raw_with_special!r}")

        has_tag = "<irrigation_calc>" in raw and "</irrigation_calc>" in raw
        well_formed = False
        parsed = None
        if has_tag:
            try:
                inner = raw.split("<irrigation_calc>", 1)[1].split("</irrigation_calc>", 1)[0]
                parsed = json.loads(inner)
                well_formed = True
            except Exception as e:
                well_formed = False
                print(f"\nJSON PARSE ERROR: {type(e).__name__}: {e}")

        print(f"\n<irrigation_calc> emitted: {has_tag}")
        print(f"well-formed JSON: {well_formed}")
        if parsed is not None:
            print(f"parsed params: {json.dumps(parsed, ensure_ascii=False)}")
            if p["stated"]:
                echo_check = {k: (parsed.get(k), v, parsed.get(k) == v)
                              for k, v in p["stated"].items()}
                print("\nECHO CHECK (parsed_value, stated_value, match):")
                for k, (pv, sv, ok) in echo_check.items():
                    print(f"  {k:<12} parsed={pv!r:<15} stated={sv!r:<15} match={ok}")
                all_match = all(v[2] for v in echo_check.values())
                print(f"\nALL PARAMETERS ECHOED CORRECTLY: {all_match}")

        print()
        results.append({"label": p["label"], "raw": raw, "has_tag": has_tag,
                        "well_formed": well_formed, "parsed": parsed})

    print("=" * 88)
    print("SUMMARY (no scoring, just presence flags)")
    print("=" * 88)
    for r in results:
        print(f"{r['label'][:70]:<72} tag={r['has_tag']!s:<6} well_formed={r['well_formed']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
