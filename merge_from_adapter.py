"""
Recovery script: reload the already-trained LoRA adapter and merge+save
using Unsloth's save_pretrained_merged (avoids the transformers 5.5.0
core_model_loading.reverse_op NotImplementedError hit by raw save_pretrained
on a merged bnb-4bit model).
"""
from pathlib import Path
from unsloth import FastLanguageModel

adapter_path = Path("D:/Research Lab/Thesis/models/llama_qlora_full_run/final_adapter")
merged_path = Path("D:/Research Lab/Thesis/models/llama_qlora_full_run/merged_model")

print(f"Loading base model + adapter from {adapter_path}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(adapter_path),
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)
print("✓ Loaded")

print(f"Merging and saving to {merged_path} (save_method=merged_16bit)...")
model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
print("✓ Merged model saved")
