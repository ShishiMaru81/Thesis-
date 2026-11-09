#!/usr/bin/env python3
"""
split_v2.py — group-aware, row-stratified 80/10/10 split of data/v2/all_v2.jsonl.

WHY ROW-STRATIFIED AND NOT GROUP-STRATIFIED
  Group sizes are uneven across question_types: numeric_runtime groups hold 2
  rows (a BN/EN twin), factual and abstain groups hold 12. Taking 80% of the
  1,126 groups at random would therefore NOT give 80% of the rows of each type.
  v1 drifted exactly this way (numeric was 28.2% of train but 19.4% of test).
  Here groups are allocated WITHIN each question_type stratum against a row
  quota, and every distribution assertion is then verified on ROW counts.

Exits non-zero on any assertion failure. No warnings-that-are-really-errors.

Run:  python scripts/split_v2.py
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V2_DIR = PROJECT_ROOT / "data" / "v2"
ALL_V2 = V2_DIR / "all_v2.jsonl"

SEED = 42
RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
PP_TOLERANCE = 3.0          # percentage points
NUMERIC_CONTAMINATION_MAX = 0.0   # hard zero, per spec

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    tag = "PASS" if condition else "FAIL"
    print(f"  [{tag}] {label}" + (f"  — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{label}: {detail}")
    return condition


def strip_lang_suffix(gid: str) -> str:
    for suf in ("__bn", "__en"):
        if gid.endswith(suf):
            return gid[: -len(suf)]
    return gid


def allocate(groups: dict[str, list], rng: random.Random) -> dict[str, str]:
    """Assign each group to a split so that ROW counts hit the 80/10/10 quota.

    Greedy largest-deficit: walk the shuffled groups and hand each one to
    whichever split is currently furthest below its row quota. With equal group
    sizes this reduces to proportional allocation; with unequal sizes it still
    tracks the row target, which plain group-proportional allocation does not.
    """
    gids = sorted(groups)
    rng.shuffle(gids)
    total_rows = sum(len(groups[g]) for g in gids)
    quota = {s: RATIOS[s] * total_rows for s in RATIOS}
    filled = {s: 0 for s in RATIOS}
    assignment = {}
    for gid in gids:
        # deterministic tie-break via the fixed split order
        split = max(("train", "val", "test"), key=lambda s: quota[s] - filled[s])
        assignment[gid] = split
        filled[split] += len(groups[gid])
    return assignment


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def spread(values) -> float:
    vals = list(values)
    return max(vals) - min(vals) if vals else 0.0


def main() -> int:
    rows = [json.loads(l) for l in ALL_V2.open(encoding="utf-8") if l.strip()]
    print(f"Loaded {len(rows)} records from {ALL_V2}\n")

    # ---------------------------------------------------------------- split
    by_type_groups: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for r in rows:
        by_type_groups[r["question_type"]][r["group_id"]].append(r)

    rng = random.Random(SEED)
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    print("Row-quota allocation within each question_type stratum:")
    for qtype in sorted(by_type_groups):
        groups = by_type_groups[qtype]
        assignment = allocate(groups, rng)
        counts = collections.Counter()
        gcounts = collections.Counter()
        for gid, split in assignment.items():
            splits[split].extend(groups[gid])
            counts[split] += len(groups[gid])
            gcounts[split] += 1
        n = sum(counts.values())
        print(f"  {qtype:<16} rows {counts['train']:5d}/{counts['val']:4d}/{counts['test']:4d}"
              f"  = {pct(counts['train'],n):5.1f}/{pct(counts['val'],n):4.1f}/{pct(counts['test'],n):4.1f}%"
              f"   groups {gcounts['train']}/{gcounts['val']}/{gcounts['test']}")

    for s in splits:
        rng.shuffle(splits[s])

    train, val, test = splits["train"], splits["val"], splits["test"]
    total = len(train) + len(val) + len(test)
    print(f"\nSplit sizes: train={len(train)} ({pct(len(train),total):.1f}%)  "
          f"val={len(val)} ({pct(len(val),total):.1f}%)  "
          f"test={len(test)} ({pct(len(test),total):.1f}%)  total={total}")

    # ---------------------------------------------------------------- gates
    print("\n" + "=" * 72)
    print("ASSERTIONS")
    print("=" * 72)

    gs = {s: set(r["group_id"] for r in v) for s, v in
          (("train", train), ("val", val), ("test", test))}

    print("\n1. Pairwise group_id intersections")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = gs[a] & gs[b]
        check(len(inter) == 0, f"{a} ∩ {b} group_id", f"{len(inter)} shared")

    print("\n2. Language-stripped key intersections")
    ss = {s: set(strip_lang_suffix(g) for g in v) for s, v in gs.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = ss[a] & ss[b]
        check(len(inter) == 0, f"{a} ∩ {b} lang-stripped key",
              f"{len(inter)} shared" + (f" e.g. {sorted(inter)[:3]}" if inter else ""))
    nonnum = {s: set(strip_lang_suffix(r["group_id"]) for r in v
                     if r["question_type"] != "numeric_runtime")
              for s, v in (("train", train), ("val", val), ("test", test))}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = nonnum[a] & nonnum[b]
        check(len(inter) == 0, f"{a} ∩ {b} lang-stripped key (NON-NUMERIC only)",
              f"{len(inter)} shared")

    print("\n3. question_type row distribution across splits (±3.0 pp)")
    qt_pct = {}
    for s, v in (("train", train), ("val", val), ("test", test)):
        c = collections.Counter(r["question_type"] for r in v)
        qt_pct[s] = {k: pct(c[k], len(v)) for k in
                     sorted(set(r["question_type"] for r in rows))}
    for qtype in sorted(qt_pct["train"]):
        vals = [qt_pct[s][qtype] for s in ("train", "val", "test")]
        check(spread(vals) <= PP_TOLERANCE, f"{qtype:<16}",
              f"train {vals[0]:.2f}% / val {vals[1]:.2f}% / test {vals[2]:.2f}%  "
              f"spread {spread(vals):.2f} pp")

    print("\n4. lang row distribution across splits (±3.0 pp)")
    lang_pct = {}
    for s, v in (("train", train), ("val", val), ("test", test)):
        c = collections.Counter(r["lang"] for r in v)
        lang_pct[s] = {k: pct(c[k], len(v)) for k in ("bn", "en")}
    for lg in ("bn", "en"):
        vals = [lang_pct[s][lg] for s in ("train", "val", "test")]
        check(spread(vals) <= PP_TOLERANCE, f"lang {lg}",
              f"train {vals[0]:.2f}% / val {vals[1]:.2f}% / test {vals[2]:.2f}%  "
              f"spread {spread(vals):.2f} pp")

    print("\n5. No duplicate id")
    ids = [r["id"] for r in train + val + test]
    dup = [i for i, c in collections.Counter(ids).items() if c > 1]
    check(len(dup) == 0, "unique ids", f"{len(dup)} duplicated")

    print("\n6. No identical question string across splits")
    qsets = {s: set(r["messages"][0]["content"] for r in v) for s, v in
             (("train", train), ("val", val), ("test", test))}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = qsets[a] & qsets[b]
        check(len(inter) == 0, f"{a} ∩ {b} question text", f"{len(inter)} shared")

    # ------------------------------------------------- contamination table
    print("\n" + "=" * 72)
    print("GOLD-ANSWER CONTAMINATION  (fraction of rows whose gold string appears verbatim in train)")
    print("=" * 72)
    train_golds = set(r["messages"][1]["content"] for r in train)
    qtypes = sorted(set(r["question_type"] for r in rows))
    print(f"\n{'split':<7}{'type':<18}{'rows':>7}{'in-train':>10}{'pct':>9}")
    print("-" * 51)
    contam = {}
    for s, v in (("val", val), ("test", test)):
        for qtype in qtypes + ["ALL"]:
            sub = v if qtype == "ALL" else [r for r in v if r["question_type"] == qtype]
            if not sub:
                continue
            hit = sum(1 for r in sub if r["messages"][1]["content"] in train_golds)
            contam[(s, qtype)] = pct(hit, len(sub))
            print(f"{s:<7}{qtype:<18}{len(sub):>7}{hit:>10}{pct(hit,len(sub)):>8.1f}%")
        print("-" * 51)

    print("\nNumeric contamination gate (HARD: must be exactly 0.0%)")
    for s in ("val", "test"):
        got = contam.get((s, "numeric_runtime"), 0.0)
        check(got <= NUMERIC_CONTAMINATION_MAX, f"{s} numeric_runtime contamination",
              f"{got:.2f}% (group binding {'intact' if got == 0.0 else 'BROKEN'})")

    # ------------------------------------------- non-zero-gold coverage
    print("\n" + "=" * 72)
    print("NON-ZERO-GOLD NUMERIC COVERAGE  (the only rows where MAE/RMSE has resolution)")
    print("=" * 72)
    tagged = [("train", train), ("val", val), ("test", test)]
    nz_all = [r for _, v in tagged for r in v
              if r["question_type"] == "numeric_runtime"
              and r["grounding"]["expected_net_mm"] > 0]
    num_all = [r for _, v in tagged for r in v if r["question_type"] == "numeric_runtime"]
    print(f"\nnon-zero rows: {len(nz_all)} / {len(num_all)} numeric "
          f"({pct(len(nz_all), len(num_all)):.1f}%)")

    print("\n-- non-zero rows per split --")
    print(f"{'split':<8}{'numeric':>9}{'non-zero':>10}{'pct':>8}")
    print("-" * 35)
    for s, v in tagged:
        n = [r for r in v if r["question_type"] == "numeric_runtime"]
        z = [r for r in n if r["grounding"]["expected_net_mm"] > 0]
        print(f"{s:<8}{len(n):>9}{len(z):>10}{pct(len(z), len(n)):>7.1f}%")

    print("\n-- non-zero rows per stage --")
    st = collections.Counter(r["grounding"]["params"]["stage"] for r in nz_all)
    st_tot = collections.Counter(r["grounding"]["params"]["stage"] for r in num_all)
    print(f"{'stage':<14}{'non-zero':>9}{'of':>7}{'pct':>8}")
    print("-" * 38)
    for k in ("initial", "mid-season", "late-season"):
        print(f"{k:<14}{st[k]:>9}{st_tot[k]:>7}{pct(st[k], st_tot[k]):>7.1f}%")

    print("\n-- non-zero rows per crop (all 28; kc_mid shown for reference) --")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import irrigation_calc as ic
    cr = collections.Counter(r["crop"] for r in nz_all)
    cr_tot = collections.Counter(r["crop"] for r in num_all)
    print(f"{'crop':<16}{'non-zero':>9}{'of':>5}{'pct':>8}{'kc_ini':>8}{'kc_mid':>8}{'kc_end':>8}")
    print("-" * 62)
    for crop in sorted(cr_tot, key=lambda c: -cr[c]):
        row = ic.KC_TABLE[ic.normalize_crop_name(crop)]
        print(f"{crop:<16}{cr[crop]:>9}{cr_tot[crop]:>5}{pct(cr[crop], cr_tot[crop]):>7.1f}%"
              f"{row['kc_ini']:>8}{row['kc_mid']:>8}{row['kc_end']:>8}")
    zero_crops = [c for c in cr_tot if cr[c] == 0]
    if zero_crops:
        print(f"\n  CROPS WITH NO NON-ZERO ROWS AT ALL: {sorted(zero_crops)}")

    print("\n-- non-zero rows per (split, stage) --")
    print(f"{'split':<8}{'initial':>9}{'mid-season':>12}{'late-season':>13}")
    print("-" * 42)
    for s, v in tagged:
        z = [r for r in v if r["question_type"] == "numeric_runtime"
             and r["grounding"]["expected_net_mm"] > 0]
        c = collections.Counter(r["grounding"]["params"]["stage"] for r in z)
        print(f"{s:<8}{c['initial']:>9}{c['mid-season']:>12}{c['late-season']:>13}")

    # ---------------------------------------------------------------- write
    if FAILURES:
        print("\n" + "=" * 72)
        print(f"GATE 3 FAILED — {len(FAILURES)} assertion(s) did not pass. Nothing written.")
        for f in FAILURES:
            print("  ✗ " + f)
        print("=" * 72)
        return 1

    for name, v in (("train", train), ("val", val), ("test", test)):
        p = V2_DIR / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in v:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nWrote {p}  ({len(v)} rows)")

    print("\n" + "=" * 72)
    print("GATE 3 PASSED — every assertion green.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
