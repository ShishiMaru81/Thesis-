#!/usr/bin/env python3
"""
preflight_models.py — prove all four base models load in 4-bit and accept a
LoRA adapter BEFORE the 12-run matrix starts.

Only Llama-3.1-8B is in the HF cache; the other three arrive as tokenizer and
config stubs. Without this stage the first Mistral or Phi run would be the one
discovering a multi-GB download failure, a gated-repo 401, or an unsloth
architecture gap — six hours into an unattended overnight job, with the GPU
then idle until morning. Loading each model once here surfaces all three
failure modes in minutes and warms the cache for the real runs.

Loads, attaches LoRA with the exact Stage 5 config, records peak VRAM, frees,
and moves on. Exits non-zero if any model fails, so the launcher can stop.
"""
from __future__ import annotations

import datetime
import gc
import json
import time
import traceback
from pathlib import Path

import torch
from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "reports" / "s5" / "preflight_models.json"

MODEL_REGISTRY = {
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral_nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "phi35_mini": "microsoft/Phi-3.5-mini-instruct",
    "phi4_mini": "microsoft/Phi-4-mini-instruct",
}
MAX_SEQ_LEN = 768
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none")


def main() -> int:
    results = {}
    print("=" * 78)
    print("STAGE 5 MODEL PREFLIGHT — 4-bit load + LoRA attach")
    print(f"max_seq_len={MAX_SEQ_LEN}  GPU={torch.cuda.get_device_name(0)}")
    print("=" * 78)

    for key, model_id in MODEL_REGISTRY.items():
        print(f"\n--- {key}  ({model_id}) ---")
        t0 = time.time()
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=MAX_SEQ_LEN,
                dtype=None,
                load_in_4bit=True,
            )
            load_gb = torch.cuda.max_memory_reserved() / 1024**3

            model = FastLanguageModel.get_peft_model(
                model, **LORA, use_gradient_checkpointing="unsloth", random_state=42)
            peft_gb = torch.cuda.max_memory_reserved() / 1024**3

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())

            results[key] = {
                "model_id": model_id,
                "status": "OK",
                "vram_gb_after_load": round(load_gb, 2),
                "vram_gb_after_lora": round(peft_gb, 2),
                "trainable_params": trainable,
                "total_params": total,
                "has_chat_template": tokenizer.chat_template is not None,
                "seconds": round(time.time() - t0, 1),
            }
            print(f"    OK  vram_load={load_gb:.2f} GB  vram_lora={peft_gb:.2f} GB  "
                  f"trainable={trainable:,}  ({time.time()-t0:.0f}s)")

            del model, tokenizer
        except Exception as e:
            results[key] = {
                "model_id": model_id,
                "status": "FAIL",
                "error": f"{type(e).__name__}: {e}",
                "seconds": round(time.time() - t0, 1),
            }
            print(f"    FAIL  {type(e).__name__}: {e}")
            traceback.print_exc()

        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print("PREFLIGHT SUMMARY")
    print("=" * 78)
    print(f"{'model':<16}{'status':<8}{'vram_load':>11}{'vram_lora':>11}{'trainable':>14}")
    print("-" * 60)
    for k, r in results.items():
        if r["status"] == "OK":
            print(f"{k:<16}{r['status']:<8}{r['vram_gb_after_load']:>11.2f}"
                  f"{r['vram_gb_after_lora']:>11.2f}{r['trainable_params']:>14,}")
        else:
            print(f"{k:<16}{r['status']:<8}  {r['error'][:60]}")

    n_ok = sum(1 for r in results.values() if r["status"] == "OK")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.datetime.now().isoformat(),
                   "max_seq_len": MAX_SEQ_LEN,
                   "gpu": torch.cuda.get_device_name(0),
                   "models_ok": n_ok, "models_total": len(MODEL_REGISTRY),
                   "results": results}, f, indent=2)

    print(f"\n{n_ok}/{len(MODEL_REGISTRY)} models ready.  ->  {OUT.relative_to(PROJECT_ROOT)}")
    return 0 if n_ok == len(MODEL_REGISTRY) else 1


if __name__ == "__main__":
    raise SystemExit(main())
