#!/usr/bin/env python3
"""
train_v2.py — QLoRA fine-tune of Llama-3.1-8B-Instruct on data/v2, with the
audit gates from the v1 failure built in rather than checked after the fact.

DEVIATION FROM THE LITERAL SPEC — READ BEFORE CHANGING THIS FILE
------------------------------------------------------------------
The spec called for `completion_only_loss=True` on a dataset shaped like
v1's — one "messages" column holding [user, assistant]. In this environment
(trl 0.24.0) that combination is a SILENT NO-OP: `completion_only_loss` only
takes effect for "prompt"/"completion"-format datasets (trl's own docstring:
"supported only for prompt-completion datasets"). Fed a "messages" column, no
completion_mask is ever created, so the collator never masks anything, and
training would compute loss over the full sequence — reproducing v1's bug
number 4 exactly, while the config claims it's fixed.

The alternative in trl's docs, `assistant_only_loss=True`, needs the
tokenizer's chat template to carry the `{% generation %}` Jinja keyword.
Verified empirically against three loaders (raw AutoTokenizer on both the
Unsloth-repackaged and the official meta-llama repo, and Unsloth's own
FastLanguageModel loader): none of this model's chat templates have it.
Setting assistant_only_loss=True here raises a hard RuntimeError.

The fix: reshape every record from {"messages": [user, assistant]} to
{"prompt": [user], "completion": [assistant]} before handing it to the
trainer. This is a lossless, information-preserving reshape (a 2-turn
conversation *is* one prompt and one completion) and it makes
`completion_only_loss=True` use trl's OTHER masking path — a plain
token-count diff between apply_chat_template(prompt) and
apply_chat_template(prompt+completion) — which needs no chat-template
feature at all. Verified against a real SFTTrainer instance on 6 held-out
samples (2 per question_type, including the Bengali abstain sentinel and a
<irrigation_calc> tool call): the decoded completion-masked span matches the
gold answer exactly in all 6 cases, with the system preamble and user turn
correctly zeroed out.

Everything else follows the spec's mandatory config.
"""
from __future__ import annotations

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
DATA_DIR = PROJECT_ROOT / "data" / "v2"
OUT_DIR = PROJECT_ROOT / "models" / "v2_run"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
ADAPTER_DIR = OUT_DIR / "final_adapter"
MERGED_DIR = OUT_DIR / "merged_model"

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_LEN = 1024
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def to_prompt_completion(rows: list[dict]) -> Dataset:
    """{"messages": [user, assistant]} -> {"prompt": [user], "completion": [assistant]}.
    See module docstring: this is what makes completion_only_loss actually mask
    the prompt in this trl version. Every v2 record is exactly a 2-turn
    conversation, so the reshape loses nothing."""
    out = []
    for r in rows:
        msgs = r["messages"]
        assert len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant", (
            f"record {r.get('id')} is not a 2-turn user/assistant conversation")
        out.append({"prompt": [msgs[0]], "completion": [msgs[1]]})
    return Dataset.from_list(out)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def print_rendered_samples(rows: list[dict], tokenizer, n: int = 3) -> None:
    """Spec 4.4: verify the <irrigation_calc> block survives templating intact.
    Rendered here purely for visual inspection via apply_chat_template(tokenize=False)
    — this is NOT the data structure fed to the trainer (see module docstring)."""
    numeric = [r for r in rows if r["question_type"] == "numeric_runtime"][:n]
    print("\n" + "=" * 72)
    print(f"RENDERED SAMPLES ({len(numeric)}) — verifying <irrigation_calc> survives templating")
    print("=" * 72)
    ok = 0
    for r in numeric:
        text = tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        intact = "<irrigation_calc>" in text and "</irrigation_calc>" in text
        ok += intact
        print(f"\n--- id={r['id']}  tag_intact={intact} ---")
        print(text)
    print(f"\n{ok}/{len(numeric)} samples: <irrigation_calc> block intact after chat-template rendering.")
    assert ok == len(numeric), "GATE FAILURE: tool-call tag did not survive chat templating"


def main() -> int:
    t_wall_start = time.time()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(DATA_DIR / "train.jsonl")
    val_rows = load_jsonl(DATA_DIR / "val.jsonl")
    print(f"Loaded train={len(train_rows)}  val={len(val_rows)}")

    print("\nLoading Llama-3.1-8B-Instruct (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    print_rendered_samples(train_rows, tokenizer, n=3)

    # ---------------------------------------------------------------- 4.3 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    train_dataset = to_prompt_completion(train_rows)
    val_dataset = to_prompt_completion(val_rows)

    # ---------------------------------------------------------------- 4.1 config
    args = SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=3,                  # ceiling only; early stopping cuts it short
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,        # effective batch 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        optim="paged_adamw_32bit",
        max_grad_norm=1.0,
        weight_decay=0.001,                   # set explicitly per spec
        bf16=True,
        max_length=MAX_LEN,
        packing=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,                        # must equal eval_steps for load_best_model_at_end
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        data_seed=3407,                       # set explicitly per spec
        completion_only_loss=True,            # works: dataset is prompt/completion, see docstring
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,           # NOT `tokenizer=` — that kwarg silently
                                              # produced a None tokenizer inside Unsloth's
                                              # patched SFTTrainer.__init__ and crashed
                                              # fix_untrained_tokens(); verified empirically.
    )

    # ---------------------------------------------------------------- 4.2 early stopping
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    steps_per_epoch = -(-len(train_dataset) // (4 * 4))  # ceil
    max_possible_steps = steps_per_epoch * 3
    print(f"\nsteps/epoch (ceil) = {steps_per_epoch}   max possible steps (3 epochs) = {max_possible_steps}")

    print("\n" + "=" * 72)
    print("TRAINING START")
    print("=" * 72)
    result = trainer.train()
    wall_seconds = time.time() - t_wall_start

    # ---------------------------------------------------------------- GATE 4 report
    log_history = trainer.state.log_history
    final_step = trainer.state.global_step
    best_ckpt = trainer.state.best_model_checkpoint
    best_step = int(Path(best_ckpt).name.split("-")[-1]) if best_ckpt else None
    early_stopped = final_step < max_possible_steps

    print("\n" + "=" * 72)
    print("GATE 4 REPORT")
    print("=" * 72)

    print(f"\nfinal training loss (trainer summary): {result.training_loss:.4f}")
    print(f"final global_step: {final_step}   max possible steps: {max_possible_steps}")
    print(f"early stopping fired: {early_stopped}")
    print(f"best checkpoint: {best_ckpt}  (step {best_step})")

    print("\n-- full eval curve --")
    print(f"{'step':>6}{'epoch':>8}{'train_loss':>12}{'eval_loss':>12}")
    print("-" * 38)
    train_losses_by_step, eval_losses_by_step, epoch_by_step = {}, {}, {}
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "epoch" in entry:
            epoch_by_step[step] = entry["epoch"]
        if "loss" in entry:
            train_losses_by_step[step] = entry["loss"]
        if "eval_loss" in entry:
            eval_losses_by_step[step] = entry["eval_loss"]
    all_steps = sorted(set(train_losses_by_step) | set(eval_losses_by_step))
    for s in all_steps:
        tl = train_losses_by_step.get(s)
        el = eval_losses_by_step.get(s)
        ep = epoch_by_step.get(s, "")
        print(f"{s:>6}{ep!s:>8}{(f'{tl:.4f}' if tl is not None else ''):>12}"
              f"{(f'{el:.4f}' if el is not None else ''):>12}")

    def nearest_train_loss(step):
        eligible = [s for s in train_losses_by_step if s <= step]
        return train_losses_by_step[max(eligible)] if eligible else None

    final_train = nearest_train_loss(final_step)
    final_eval = eval_losses_by_step.get(final_step) or (
        eval_losses_by_step[max(eval_losses_by_step)] if eval_losses_by_step else None)
    best_train = nearest_train_loss(best_step) if best_step is not None else None
    best_eval = eval_losses_by_step.get(best_step) if best_step is not None else None

    print(f"\n-- train/eval gap --")
    if final_train and final_eval:
        print(f"at final step {final_step}: train={final_train:.4f} eval={final_eval:.4f} "
              f"ratio(eval/train)={final_eval/final_train:.2f}x")
    if best_train and best_eval:
        print(f"at best step  {best_step}: train={best_train:.4f} eval={best_eval:.4f} "
              f"ratio(eval/train)={best_eval/best_train:.2f}x")

    # ---------------------------------------------------------------- 4.6 save + merge
    print(f"\nSaving LoRA adapter to {ADAPTER_DIR} ...")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    print(f"Merging to {MERGED_DIR} (save_method='merged_16bit') ...")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(MERGED_DIR), tokenizer, save_method="merged_16bit")

    final_adapter_file = ADAPTER_DIR / "adapter_model.safetensors"
    best_ckpt_file = (Path(best_ckpt) / "adapter_model.safetensors") if best_ckpt else None

    print("\n-- checksums --")
    final_sha = sha256_of(final_adapter_file) if final_adapter_file.exists() else None
    print(f"final_adapter/adapter_model.safetensors sha256: {final_sha}")
    best_sha = None
    if best_ckpt_file and best_ckpt_file.exists():
        best_sha = sha256_of(best_ckpt_file)
        print(f"best checkpoint  adapter_model.safetensors sha256: {best_sha}")
        print(f"MATCH (final == best, confirms load_best_model_at_end worked): {final_sha == best_sha}")
    else:
        print(f"best checkpoint adapter file not found at expected path: {best_ckpt_file}")

    print(f"\nwall-clock time: {wall_seconds:.1f} s  ({wall_seconds/60:.1f} min)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
