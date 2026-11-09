import json
import random
from pathlib import Path
from collections import defaultdict

# Seed for reproducibility
random.seed(42)

# Input file
input_file = Path("D:/Research Lab/Thesis/Json_Dataset(Calander)/qa_out/all.jsonl")
output_dir = Path("D:/Research Lab/Thesis/data/finetune")
output_dir.mkdir(parents=True, exist_ok=True)

# Load all QA pairs
qa_pairs = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            qa_pairs.append(json.loads(line))

print(f"✓ Loaded {len(qa_pairs)} QA pairs")

# Group by group_id to ensure stratified split
groups = defaultdict(list)
for pair in qa_pairs:
    group_id = pair.get("group_id", "unknown")
    groups[group_id].append(pair)

print(f"✓ Found {len(groups)} unique groups")

# Stratified split: keep groups together
group_ids = list(groups.keys())
random.shuffle(group_ids)

train_size = int(0.8 * len(group_ids))
val_size = int(0.1 * len(group_ids))

train_groups = group_ids[:train_size]
val_groups = group_ids[train_size:train_size+val_size]
test_groups = group_ids[train_size+val_size:]

# Collect pairs by split
train_pairs = [p for g in train_groups for p in groups[g]]
val_pairs = [p for g in val_groups for p in groups[g]]
test_pairs = [p for g in test_groups for p in groups[g]]

print(f"✓ Train: {len(train_pairs)} pairs ({len(train_groups)} groups)")
print(f"✓ Val:   {len(val_pairs)} pairs ({len(val_groups)} groups)")
print(f"✓ Test:  {len(test_pairs)} pairs ({len(test_groups)} groups)")

# Sanity check: no leakage
assert len(set(train_groups) & set(val_groups)) == 0, "Train/Val leakage!"
assert len(set(train_groups) & set(test_groups)) == 0, "Train/Test leakage!"
assert len(set(val_groups) & set(test_groups)) == 0, "Val/Test leakage!"
print("✓ No group leakage detected")

# Verify messages format
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

write_jsonl(train_pairs, output_dir / "train.jsonl")
write_jsonl(val_pairs, output_dir / "val.jsonl")
write_jsonl(test_pairs, output_dir / "test.jsonl")

print(f"\n✓ Splits written to {output_dir}/")
print(f"  - train.jsonl ({len(train_pairs)} lines)")
print(f"  - val.jsonl ({len(val_pairs)} lines)")
print(f"  - test.jsonl ({len(test_pairs)} lines)")