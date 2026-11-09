"""
COMPLETE LLAMA-3.1-8B FINE-TUNING PIPELINE
Unified script: format QA dataset → smoke test → full training → merge
Run once and it handles everything sequentially.
"""

import json
import random
import torch
from pathlib import Path
from collections import defaultdict
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============================================================================
# PHASE 1: ENVIRONMENT CHECKS
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: ENVIRONMENT VERIFICATION")
print("="*80)

print("\n🔍 Checking CUDA availability...")
if not torch.cuda.is_available():
    print("❌ CUDA not available! Cannot proceed.")
    exit(1)
print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
print(f"✓ PyTorch version: {torch.__version__}")

print("\n🔍 Checking required packages...")
try:
    import unsloth
    import transformers
    import datasets
    import trl
    import peft
    print("✓ All packages imported successfully")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    exit(1)

print("\n🔍 Verifying HF authentication...")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    print("✓ Llama-3.1-8B-Instruct accessible via HF")
except Exception as e:
    print(f"❌ Cannot access model: {e}")
    exit(1)

# ============================================================================
# PHASE 2: FORMAT QA DATASET → CREATE SPLITS
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: QA DATASET FORMATTING & SPLITTING")
print("="*80)

random.seed(42)

input_file = Path("D:/Research Lab/Thesis/Json_Dataset(Calander)/qa_out/all.jsonl")
data_dir = Path("D:/Research Lab/Thesis/data/finetune")
data_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📂 Loading raw QA pairs from {input_file}")
if not input_file.exists():
    print(f"❌ File not found: {input_file}")
    exit(1)

qa_pairs = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            qa_pairs.append(json.loads(line))

print(f"✓ Loaded {len(qa_pairs)} QA pairs")

# Group by group_id for stratified split
groups = defaultdict(list)
for pair in qa_pairs:
    group_id = pair.get("group_id", "unknown")
    groups[group_id].append(pair)

print(f"✓ Found {len(groups)} unique groups")

# Stratified split
group_ids = list(groups.keys())
random.shuffle(group_ids)

train_size = int(0.8 * len(group_ids))
val_size = int(0.1 * len(group_ids))

train_groups = group_ids[:train_size]
val_groups = group_ids[train_size:train_size+val_size]
test_groups = group_ids[train_size+val_size:]

train_pairs = [p for g in train_groups for p in groups[g]]
val_pairs = [p for g in val_groups for p in groups[g]]
test_pairs = [p for g in test_groups for p in groups[g]]

print(f"\n✓ Train: {len(train_pairs)} pairs ({len(train_groups)} groups)")
print(f"✓ Val:   {len(val_pairs)} pairs ({len(val_groups)} groups)")
print(f"✓ Test:  {len(test_pairs)} pairs ({len(test_groups)} groups)")

# Verify no leakage
assert len(set(train_groups) & set(val_groups)) == 0, "Train/Val leakage!"
assert len(set(train_groups) & set(test_groups)) == 0, "Train/Test leakage!"
assert len(set(val_groups) & set(test_groups)) == 0, "Val/Test leakage!"
print("✓ No group leakage detected")

# Verify message format
def verify_messages(pair):
    if "messages" not in pair:
        return False
    msgs = pair["messages"]
    if len(msgs) != 2:
        return False
    if msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
        return False
    return True

bad_pairs = [p for split_pairs in [train_pairs, val_pairs, test_pairs]
             for p in split_pairs if not verify_messages(p)]
if bad_pairs:
    print(f"⚠ WARNING: {len(bad_pairs)} pairs have malformed messages format")
else:
    print("✓ All pairs have valid messages format")

# Write splits
def write_jsonl(pairs, path):
    with open(path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

write_jsonl(train_pairs, data_dir / "train.jsonl")
write_jsonl(val_pairs, data_dir / "val.jsonl")
write_jsonl(test_pairs, data_dir / "test.jsonl")

print(f"\n✓ Splits written to {data_dir}/")
print(f"  - train.jsonl ({len(train_pairs)} lines)")
print(f"  - val.jsonl ({len(val_pairs)} lines)")
print(f"  - test.jsonl ({len(test_pairs)} lines)")

# ============================================================================
# PHASE 3: SMOKE TEST (20 steps on 100 samples)
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: SMOKE TEST (20 steps on 100 samples)")
print("="*80)

def load_jsonl(path, limit=None):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                data.append(json.loads(line))
    return data

def format_dataset(dataset, tokenizer):
    def _render(examples):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["messages"]
        ]
        return {"text": texts}
    return dataset.map(_render, batched=True, remove_columns=dataset.column_names)

print("\n📂 Loading smoke test datasets (100 train, 20 val)...")
smoke_train_data = load_jsonl(data_dir / "train.jsonl", limit=100)
smoke_val_data = load_jsonl(data_dir / "val.jsonl", limit=20)

print(f"   Train: {len(smoke_train_data)} samples")
print(f"   Val:   {len(smoke_val_data)} samples")

smoke_train_dataset = Dataset.from_list(smoke_train_data)
smoke_val_dataset = Dataset.from_list(smoke_val_data)

print("\n🚀 Loading Llama-3.1-8B-Instruct (4-bit)...")
smoke_model, smoke_tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

smoke_model = FastLanguageModel.get_peft_model(
    smoke_model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

smoke_train_dataset = format_dataset(smoke_train_dataset, smoke_tokenizer)
smoke_val_dataset = format_dataset(smoke_val_dataset, smoke_tokenizer)

print("✓ Model ready for smoke test")

smoke_args = SFTConfig(
    output_dir=str(Path("D:/Research Lab/Thesis/models/llama_qlora_smoke_test/checkpoints")),
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    max_steps=20,
    logging_steps=5,
    eval_steps=10,
    save_steps=20,
    eval_strategy="steps",
    bf16=True,
    seed=42,
    max_seq_length=1024,
    dataset_text_field="text",
    packing=False,
)

print("\n⚙️  SMOKE TEST CONFIG: max_steps=20")

smoke_trainer = SFTTrainer(
    model=smoke_model,
    tokenizer=smoke_tokenizer,
    train_dataset=smoke_train_dataset,
    eval_dataset=smoke_val_dataset,
    args=smoke_args,
)

print("\n🎯 Starting smoke test...")
print("   Watch for loss descent and no OOM errors\n")

try:
    smoke_result = smoke_trainer.train()
    print("\n" + "="*80)
    print("✓✓✓ SMOKE TEST PASSED ✓✓✓")
    print("="*80)
    print(f"Final training loss: {smoke_result.training_loss:.4f}")
    print("No OOM errors. Proceeding to full training.")
    print("="*80)
except Exception as e:
    print(f"\n❌ SMOKE TEST FAILED: {e}")
    print("Cannot proceed to full training.")
    exit(1)

# Clean up smoke test model from VRAM
del smoke_model, smoke_tokenizer, smoke_trainer
torch.cuda.empty_cache()

# ============================================================================
# PHASE 4: FULL TRAINING (3 epochs on 2,976 samples)
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: FULL TRAINING (3 epochs, 2,976 samples)")
print("="*80)

print("\n📂 Loading full datasets...")
full_train_data = load_jsonl(data_dir / "train.jsonl")
full_val_data = load_jsonl(data_dir / "val.jsonl")

print(f"   Train: {len(full_train_data)} samples")
print(f"   Val:   {len(full_val_data)} samples")

full_train_dataset = Dataset.from_list(full_train_data)
full_val_dataset = Dataset.from_list(full_val_data)

print("\n🚀 Loading Llama-3.1-8B-Instruct (4-bit) for full training...")
full_model, full_tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

full_model = FastLanguageModel.get_peft_model(
    full_model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

full_train_dataset = format_dataset(full_train_dataset, full_tokenizer)
full_val_dataset = format_dataset(full_val_dataset, full_tokenizer)

print("✓ Model prepared for QLoRA training")

output_dir = Path("D:/Research Lab/Thesis/models/llama_qlora_full_run")
output_dir.mkdir(parents=True, exist_ok=True)

full_args = SFTConfig(
    output_dir=str(output_dir / "checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
    max_grad_norm=1.0,
    logging_steps=20,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    seed=42,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    max_seq_length=1024,
    dataset_text_field="text",
    packing=False,
)

print("\n⚙️  FULL TRAINING CONFIGURATION:")
print(f"   Epochs: {full_args.num_train_epochs}")
print(f"   Batch size: {full_args.per_device_train_batch_size} × grad_accum={full_args.gradient_accumulation_steps} = effective 16")
print(f"   Learning rate: {full_args.learning_rate}")
print(f"   Scheduler: cosine with warmup_steps={full_args.warmup_steps}")
print(f"   Eval/Save every {full_args.eval_steps}/{full_args.save_steps} steps")
print(f"   Output dir: {output_dir}")

full_trainer = SFTTrainer(
    model=full_model,
    tokenizer=full_tokenizer,
    train_dataset=full_train_dataset,
    eval_dataset=full_val_dataset,
    args=full_args,
)

print("\n🎯 Starting full training...")
print("   This will take 2-4 hours. Progress updates every 20 steps.\n")

try:
    full_result = full_trainer.train()
    print("\n✓ Training completed successfully!")
    print(f"   Final training loss: {full_result.training_loss:.4f}")
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    print("   Saving emergency checkpoint...")
    full_trainer.save_model(output_dir / "interrupted_checkpoint")
    exit(1)
except Exception as e:
    print(f"\n❌ Training failed: {type(e).__name__}: {e}")
    raise

# ============================================================================
# PHASE 5: SAVE & MERGE
# ============================================================================
print("\n" + "="*80)
print("PHASE 5: SAVING & MERGING MODEL")
print("="*80)

# Save LoRA adapter only
adapter_path = output_dir / "final_adapter"
print(f"\n💾 Saving LoRA adapter to {adapter_path}...")
full_model.save_pretrained(adapter_path)
full_tokenizer.save_pretrained(adapter_path)
print("✓ LoRA adapter saved")

# Merge and save full model (Unsloth's merged save path, not raw
# merge_and_unload()+save_pretrained() -- the latter hits a
# NotImplementedError in transformers' core_model_loading weight-conversion
# reversal for models loaded from a bnb-4bit checkpoint)
print(f"\n📦 Merging LoRA adapter with base model...")
merged_path = output_dir / "merged_model"
print(f"   Saving to {merged_path}...")
full_model.save_pretrained_merged(str(merged_path), full_tokenizer, save_method="merged_16bit")
print("✓ Merged model saved")

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎉🎉🎉 COMPLETE FINE-TUNING PIPELINE FINISHED 🎉🎉🎉")
print("="*80)

print("\n📊 SUMMARY:")
print(f"   • Dataset: {len(qa_pairs)} pairs → {len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)} train/val/test")
print(f"   • Smoke test: 20 steps on 100 samples ✓")
print(f"   • Full training: 3 epochs on {len(full_train_data)} samples ✓")
print(f"   • Final training loss: {full_result.training_loss:.4f}")

print("\n📁 ARTIFACT LOCATIONS:")
print(f"   LoRA adapter: {adapter_path}/")
print(f"   Merged model: {merged_path}/")

print("\n🚀 NEXT STEPS:")
print("   1. Run sanity checks on the merged model:")
print("      - Test on a known training Q (factual)")
print("      - Test on a runtime-numeric Q (should emit <irrigation_calc> JSON)")
print("      - Test on out-of-domain Q (should emit abstention sentinel)")
print("   2. Build confidence router (abstention detection + log-prob threshold)")
print("   3. Execute three-way evaluation (FT vs untuned Llama vs RAG)")

print("\n" + "="*80)
print("Pipeline execution complete. All models saved and ready for evaluation.")
print("="*80)
