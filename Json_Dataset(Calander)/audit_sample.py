#!/usr/bin/env python3
"""
audit_sample.py — Draw a stratified human-verification sample from qa_out/all.jsonl.

Two constraints, in priority order (they conflict when both are pushed hard,
given 28 crops and a small N, so priority 1 is treated as a hard floor and
priority 2 is applied within whatever budget is left):

  1. Every crop gets >= --min-per-crop rows (default 3), "where possible"
     (i.e. capped at however many rows that crop actually has).
  2. Subject to (1), the overall sample's question_type mix approximates the
     full dataset's question_type mix (factual/numeric_runtime/abstain),
     via weighted sampling (weight = that type's share of the full dataset).
     lang (en/bn) is not weighted separately — each type has an equal en/bn
     split in the source data, so it balances out under random draws.

Both the per-crop floor and the top-up draw via two-level weighted sampling
WITHOUT replacement: (1) pick a question_type, weighted by that type's global
share of the full dataset, (2) pick uniformly among the remaining items of
that type. Doing it in two levels (rather than weighting individual items
directly) keeps the type probabilities correct regardless of how many
candidate rows of each type a given crop happens to have — e.g. a crop with
14 factual candidates but only 6 numeric_runtime ones would otherwise get
its factual share inflated just because there are more factual rows to draw
from, which is a sampling artifact, not the intended proportionality.

Run:  python audit_sample.py --in ./qa_out/all.jsonl --out ./audit_sample.jsonl
"""
import json, random, argparse, collections

def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def sample_by_type_then_uniform(rng, population, type_weight, k):
    """Draw k items without replacement: pick a type by weight, then pick
    uniformly among the remaining items of that type."""
    pool = list(population)
    picks = []
    for _ in range(min(k, len(pool))):
        by_type = collections.defaultdict(list)
        for idx, r in enumerate(pool):
            by_type[r["question_type"]].append(idx)
        types = list(by_type)
        weights = [type_weight[t] for t in types]
        total = sum(weights)
        target = rng.random() * total
        upto = 0.0
        chosen_type = types[-1]  # fallback for float rounding at the boundary
        for t, w in zip(types, weights):
            upto += w
            if upto >= target:
                chosen_type = t
                break
        idxs = by_type[chosen_type]
        chosen_idx = idxs[rng.randrange(len(idxs))]
        picks.append(pool.pop(chosen_idx))
    return picks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="./qa_out/all.jsonl")
    ap.add_argument("--out", dest="outfile", default="./audit_sample.jsonl")
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--min-per-crop", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows = load_rows(a.infile)

    type_counts = collections.Counter(r["question_type"] for r in rows)

    by_crop = collections.defaultdict(list)
    for r in rows:
        by_crop[r["crop"]].append(r)

    selected, selected_ids = [], set()
    short_crops = []  # crops that had fewer than min-per-crop rows available

    # --- phase 1: guarantee the per-crop floor, type-weighted -------------
    for crop in sorted(by_crop):
        pool = by_crop[crop]
        k = min(a.min_per_crop, len(pool))
        if len(pool) < a.min_per_crop:
            short_crops.append((crop, len(pool)))
        picks = sample_by_type_then_uniform(rng, pool, type_counts, k)
        for r in picks:
            selected.append(r); selected_ids.add(r["id"])

    # --- phase 2: top up to N, type-weighted, from the whole remaining pool
    need = a.n - len(selected)
    if need > 0:
        remaining = [r for r in rows if r["id"] not in selected_ids]
        picks = sample_by_type_then_uniform(rng, remaining, type_counts, need)
        for r in picks:
            selected.append(r); selected_ids.add(r["id"])
    elif need < 0:
        rng.shuffle(selected)
        selected = selected[:a.n]

    rng.shuffle(selected)

    with open(a.outfile, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- summary ------------------------------------------------------------
    n = len(selected)
    type_dist = collections.Counter(r["question_type"] for r in selected)
    lang_dist = collections.Counter(r["lang"] for r in selected)
    crop_dist = collections.Counter(r["crop"] for r in selected)

    print(f"Sampled {n} rows -> {a.outfile}")
    print("\nquestion_type distribution (sample vs full dataset):")
    for t, full_c in type_counts.most_common():
        s = type_dist.get(t, 0)
        print(f"  {t:16s} sample={s:3d} ({s/n:5.1%})   full={full_c:4d} ({full_c/len(rows):5.1%})")
    print("\nlang distribution:")
    for l, c in lang_dist.most_common():
        print(f"  {l}: {c} ({c/n:.1%})")
    print(f"\ncrops covered: {len(crop_dist)} / {len(by_crop)}")
    print(f"  min per crop: {min(crop_dist.values())}   max per crop: {max(crop_dist.values())}")
    if short_crops:
        print(f"\nWARNING: {len(short_crops)} crop(s) had fewer than {a.min_per_crop} rows available "
              f"in the full dataset, so all of theirs were included instead:")
        for crop, avail in short_crops:
            print(f"  {crop}: only {avail} row(s) available")

if __name__ == "__main__":
    main()
