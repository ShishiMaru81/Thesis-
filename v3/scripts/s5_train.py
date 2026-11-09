#!/usr/bin/env python3
"""
s5_train.py — Stage 5: QLoRA fine-tune of one (base model, seed) pair on the
v3 rendered corpus.

Trains ONE cell of the Stage 5 matrix per invocation. The matrix itself is
driven from outside (a shell loop over --model and --seed) so that no seed
list is hard-coded here and any cell can be re-run in isolation after a crash:

    for m in llama31_8b mistral_nemo phi35_mini phi4_mini; do
      for s in 42 1337 2024; do
        python scripts/s5_train.py --model $m --seed $s
      done
    done

CONFIG PROVENANCE
-----------------
Hyperparameters are carried over verbatim from scripts/train_v2.py (the v2
run that produced the thesis's prior results) so Stage 5 varies base model and
seed and nothing else. The one deliberate change is max_seq_len: 1024 -> 768,
authorised by reports/eval/preflight_seqlen.json, which measured a global max
of 528 tokens across all four tokenizers. 768 leaves 45% headroom over the
longest real sequence at zero truncation, and buys back VRAM on Mistral-Nemo.

THE MASKING GATE (runs BEFORE training, aborts on failure)
----------------------------------------------------------
trl 0.24.0 honours `completion_only_loss` only for prompt/completion-shaped
datasets; fed a "messages" column it silently computes loss over the whole
sequence. That silent no-op was bug #4 of the v1 run. Config alone cannot
prove the masking is live, so this script pulls 6 real examples (2 per qtype)
through the actual collator, decodes exactly the positions where
labels != -100, and asserts that span equals the gold answer. If prompt tokens
leak into the loss, or the completion is clipped, training never starts.

FIXED BUDGET — WHY EARLY STOPPING IS OFF BY DEFAULT
---------------------------------------------------
val.jsonl was split off by s1_split_first BEFORE s2's augmentation, so it is
100% un-augmented seed rows and 64% factual, while the training corpus is 84%
numeric_runtime. Driving EarlyStoppingCallback and load_best_model_at_end off
eval_loss would let a distribution the model is mostly NOT trained on decide
when each cell stops and which checkpoint it keeps — and decide it differently
per base model, putting a confound inside the very comparison Stage 5 exists
to make. So all 12 cells train a fixed 3 epochs and keep the final checkpoint;
eval_loss is still logged every 25 steps as a monitoring signal rather than a
control signal. Pass --early_stopping to restore the v2 behaviour.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import EarlyStoppingCallback
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RENDERED_DIR = PROJECT_ROOT / "data" / "rendered"
OUT_ROOT = PROJECT_ROOT / "models" / "s5"
REPORT_DIR = PROJECT_ROOT / "reports" / "s5"

MODEL_REGISTRY = {
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral_nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "phi35_mini": "microsoft/Phi-3.5-mini-instruct",
    "phi4_mini": "microsoft/Phi-4-mini-instruct",
}

# Per-model batch settings (effective batch = batch * grad_accum = 16).
MODEL_BATCH = {
    "llama31_8b": {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 8},
    "mistral_nemo": {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 16},
    "phi35_mini": {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 4},
    "phi4_mini": {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 4},
}

# From reports/eval/preflight_seqlen.json (global max 528 across all 4 tokenizers).
MAX_SEQ_LEN = 768

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none")

# Adapter placement is per-architecture, NOT one list for everyone.
#
# Llama, Mistral-Nemo and unsloth's Phi-3.5 repackage all expose separate
# q/k/v and gate/up projections, so the v2 list applies unchanged. Phi-4-mini
# does not: it ships FUSED `qkv_proj` and `gate_up_proj`. Handed the Llama
# list it matches only o_proj and down_proj — and peft does not raise, because
# two targets did match. The result is a silent 8.9M-parameter adapter against
# Llama's 41.9M, with zero adaptation of attention QKV or the MLP gate/up path.
# Left uncorrected that turns "Phi-4-mini is worse" into a statement about
# adapter placement rather than about the model. Verified against the cached
# safetensors index: 32 layers x [o_proj 16*(3072+3072) + down_proj
# 16*(8192+3072)] = 8,912,896, exactly the preflight count.
TARGET_MODULES = {
    "_default": ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
    "phi4_mini": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
}


def targets_for(model_key: str) -> list[str]:
    return TARGET_MODULES.get(model_key, TARGET_MODULES["_default"])


def assert_targets_attached(model, requested: list[str]) -> dict[str, int]:
    """Every requested target must actually own a LoRA layer.

    This is the guard that would have caught the Phi-4-mini fusion up front.
    peft silently tolerates a target name matching nothing as long as at least
    one other name matches, so a partially-attached adapter looks healthy from
    the outside; only the parameter count gives it away. Fail loudly instead.
    """
    found: dict[str, int] = {t: 0 for t in requested}
    for name, _ in model.named_modules():
        if "lora_A" not in name:
            continue
        for t in requested:
            if f".{t}." in name:
                found[t] += 1
    missing = [t for t, n in found.items() if n == 0]
    if missing:
        raise SystemExit(
            f"LoRA target(s) matched no module: {missing}\n"
            f"attached: { {t: n for t, n in found.items() if n} }\n"
            "The base model's projection names differ from the configured "
            "targets - fix TARGET_MODULES before training.")
    return found


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_masking_probe_indices(rows: list[dict], per_qtype: int = 2) -> list[int]:
    """2 rows per qtype, preferring a bn/en mix so the gate covers both scripts."""
    by_qtype: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_qtype[r.get("qtype", "?")].append(i)

    picked = []
    for qt in sorted(by_qtype):
        idxs = by_qtype[qt]
        chosen, langs = [], set()
        for i in idxs:                       # first pass: one per language
            lang = rows[i].get("lang")
            if lang not in langs:
                chosen.append(i)
                langs.add(lang)
            if len(chosen) == per_qtype:
                break
        for i in idxs:                       # top up if a qtype is single-language
            if len(chosen) == per_qtype:
                break
            if i not in chosen:
                chosen.append(i)
        picked.extend(chosen)
    return picked


def masking_gate(trainer, tokenizer, rows: list[dict], indices: list[int]) -> bool:
    """Decode the loss-carrying span of real collated batches; it must equal the answer."""
    print("\n" + "=" * 78)
    print(f"MASKING GATE — {len(indices)} samples through the live collator (BEFORE training)")
    print("=" * 78)

    ds = trainer.train_dataset
    collator = trainer.data_collator
    all_ok = True

    for i in indices:
        gold = rows[i]["completion"][0]["content"]
        qtype, lang = rows[i].get("qtype"), rows[i].get("lang")

        batch = collator([ds[i]])
        labels = batch["labels"][0]
        input_ids = batch["input_ids"][0]

        kept = (labels != -100)
        n_kept, n_total = int(kept.sum()), int(labels.numel())
        if n_kept == 0:
            print(f"\n--- idx={i} qtype={qtype} lang={lang}: FAIL — every token masked")
            all_ok = False
            continue

        decoded = tokenizer.decode(input_ids[kept], skip_special_tokens=True).strip()
        # The template's trailing EOS/turn-end is inside the completion span by
        # design; skip_special_tokens drops it, so compare on stripped text.
        exact = decoded == gold.strip()
        contained = gold.strip() in decoded or decoded in gold.strip()
        ok = exact or contained
        all_ok &= ok

        # A masked prompt is the whole point: loss tokens must be a strict subset.
        prompt_leaked = n_kept >= n_total

        print(f"\n--- idx={i} qtype={qtype} lang={lang} "
              f"loss_tokens={n_kept}/{n_total} exact={exact} leak={prompt_leaked} ---")
        print(f"  gold    : {gold[:150]}")
        print(f"  unmasked: {decoded[:150]}")
        if prompt_leaked:
            print("  FAIL — no prompt tokens were masked")
            all_ok = False
        if not ok:
            print("  FAIL — unmasked span does not match the gold answer")

    print("\n" + "-" * 78)
    print(f"MASKING GATE: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="smoke-run cap; -1 = full fixed 3-epoch schedule")
    ap.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--early_stopping", action="store_true",
                    help="restore v2 behaviour: stop on eval_loss patience and keep the best "
                         "checkpoint. OFF by default — see FIXED BUDGET note in the docstring.")
    ap.add_argument("--train_file", type=Path, default=None,
                    help="rendered train.jsonl (prompt/completion). Default: data/rendered/train.jsonl")
    ap.add_argument("--run_suffix", default="",
                    help="append to output tag, e.g. 'clarify' -> llama31_8b_seed42_clarify")
    ap.add_argument("--out_root", type=Path, default=OUT_ROOT,
                    help="adapter output root (default: models/s5)")
    args = ap.parse_args()

    model_id = MODEL_REGISTRY[args.model]
    smoke = args.max_steps > 0
    tag = f"{args.model}_seed{args.seed}"
    if args.run_suffix:
        tag += f"_{args.run_suffix}"
    if smoke:
        tag += f"_smoke{args.max_steps}"
    out_dir = args.out_root / tag
    ckpt_dir = out_dir / "checkpoints"
    adapter_dir = out_dir / "final_adapter"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t_wall = time.time()
    print("=" * 78)
    print(f"STAGE 5 RUN — model={args.model} ({model_id})  seed={args.seed}")
    print(f"max_seq_len={args.max_seq_len}  max_steps={args.max_steps}  smoke={smoke}")
    print("=" * 78)

    train_path = args.train_file or (RENDERED_DIR / "train.jsonl")
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(RENDERED_DIR / "val.jsonl")
    print(f"\nLoaded train={len(train_rows)} from {train_path.name}  val={len(val_rows)}")

    # trl must see prompt/completion ONLY — stray metadata columns change how it
    # infers the dataset format, and format inference is what drives the masking.
    train_ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]}
                                  for r in train_rows])
    val_ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]}
                                for r in val_rows])

    print(f"\nLoading {model_id} in 4-bit ...")
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    load_vram_gb = torch.cuda.max_memory_reserved() / 1024**3
    print(f"VRAM after 4-bit load: {load_vram_gb:.2f} GB")

    targets = targets_for(args.model)
    model = FastLanguageModel.get_peft_model(
        model, **LORA,
        target_modules=targets,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    attached = assert_targets_attached(model, targets)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nLoRA targets for {args.model}: {targets}")
    print(f"  adapted modules per target: {attached}")
    print(f"  trainable params: {n_trainable:,}")

    batch_cfg = MODEL_BATCH[args.model]
    eff_batch = (batch_cfg["per_device_train_batch_size"]
                 * batch_cfg["gradient_accumulation_steps"])
    steps_per_epoch = -(-len(train_ds) // eff_batch)
    max_possible = args.max_steps if smoke else steps_per_epoch * 3
    eval_steps = 25 if smoke else 25

    sft_args = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=3,
        max_steps=args.max_steps,
        per_device_train_batch_size=batch_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=batch_cfg["gradient_accumulation_steps"],
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=min(50, max(1, max_possible // 10)) if smoke else 50,
        optim="paged_adamw_32bit",
        max_grad_norm=1.0,
        weight_decay=0.001,
        bf16=True,
        max_length=args.max_seq_len,
        packing=False,
        logging_steps=1 if smoke else 10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        # With no best-model selection there is nothing to reload, so checkpoints
        # exist only for crash recovery: one rolling save per epoch.
        save_steps=eval_steps if args.early_stopping else max(eval_steps, steps_per_epoch),
        save_total_limit=3 if args.early_stopping else 1,
        load_best_model_at_end=args.early_stopping,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=3407,
        completion_only_loss=True,   # live-verified by masking_gate() below
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,  # NOT tokenizer= — see train_v2.py:179
    )
    if args.early_stopping and not smoke:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    # ------------------------------------------------------------- GATE
    probe_idx = pick_masking_probe_indices(train_rows, per_qtype=2)
    gate_ok = masking_gate(trainer, tokenizer, train_rows, probe_idx)
    if not gate_ok:
        print("\nABORTING — completion masking is not active. "
              "Training now would compute loss over prompt tokens (v1 bug #4).")
        return 2

    print(f"\nsteps/epoch={steps_per_epoch}  max steps this run={max_possible}")
    print("\n" + "=" * 78)
    print("TRAINING START")
    print("=" * 78)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    train_seconds = time.time() - t_wall
    peak_vram_gb = torch.cuda.max_memory_reserved() / 1024**3

    # ------------------------------------------------------------- report
    hist = trainer.state.log_history
    train_loss_by_step = {e["step"]: e["loss"] for e in hist if "loss" in e and "step" in e}
    eval_loss_by_step = {e["step"]: e["eval_loss"] for e in hist if "eval_loss" in e and "step" in e}

    print("\n" + "=" * 78)
    print("RUN REPORT")
    print("=" * 78)
    print(f"final training loss: {result.training_loss:.4f}")
    print(f"global_step: {trainer.state.global_step}")
    print(f"best checkpoint: {trainer.state.best_model_checkpoint}")

    print(f"\n{'step':>6}{'train_loss':>13}{'eval_loss':>12}")
    print("-" * 31)
    for s in sorted(set(train_loss_by_step) | set(eval_loss_by_step)):
        tl = train_loss_by_step.get(s)
        el = eval_loss_by_step.get(s)
        print(f"{s:>6}{(f'{tl:.4f}' if tl is not None else ''):>13}"
              f"{(f'{el:.4f}' if el is not None else ''):>12}")

    print(f"\npeak VRAM (reserved): {peak_vram_gb:.2f} GB")
    print(f"wall clock: {train_seconds:.1f} s ({train_seconds/60:.1f} min)")

    print(f"\nSaving adapter -> {adapter_dir}")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_sha = sha256_of(adapter_file) if adapter_file.exists() else None
    print(f"adapter sha256: {adapter_sha}")
    if adapter_sha:
        (adapter_dir / "adapter.sha256").write_text(adapter_sha + "\n", encoding="utf-8")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_key": args.model,
        "model_id": model_id,
        "seed": args.seed,
        "smoke_run": smoke,
        "max_steps": args.max_steps,
        "max_seq_len": args.max_seq_len,
        "lora": LORA,
        "lora_target_modules": targets,
        "lora_modules_attached": attached,
        "trainable_params": n_trainable,
        "train_file": str(train_path),
        "run_suffix": args.run_suffix or None,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "batch_config": batch_cfg,
        "masking_gate": "PASS" if gate_ok else "FAIL",
        "masking_probe_indices": probe_idx,
        "final_training_loss": result.training_loss,
        "global_step": trainer.state.global_step,
        "train_loss_by_step": train_loss_by_step,
        "eval_loss_by_step": eval_loss_by_step,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "peak_vram_gb_reserved": round(peak_vram_gb, 3),
        "vram_gb_after_4bit_load": round(load_vram_gb, 3),
        "wall_seconds": round(train_seconds, 1),
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": adapter_sha,
    }
    manifest_path = REPORT_DIR / f"{tag}.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"manifest -> {manifest_path.relative_to(PROJECT_ROOT)}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
