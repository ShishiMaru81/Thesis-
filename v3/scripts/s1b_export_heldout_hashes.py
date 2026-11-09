#!/usr/bin/env python3
"""
s1b_export_heldout_hashes.py — export the held-out scenario_hash blocklist.

WHY THIS SCRIPT EXISTS
----------------------
Stage 2 has two requirements that appear to contradict each other:

  HARD RULE 1  s2_augment_train.py must assert that val.jsonl and test.jsonl
               are never opened, and raise if their paths appear.
  FAMILY C     synthetic scenarios must "REJECT if scenario_hash exists in
               val or test".

You cannot check membership in a set you are forbidden to read. This script
resolves it: it is the ONE place the held-out files are opened, it extracts
ONLY the scenario_hash strings, and it discards every question, answer and
provenance field. Stage 2 then reads this derived blocklist and never touches
the held-out splits at all, so its open() guard can be absolute.

The blocklist carries the SHA-256 of both source files so any later run can
prove which frozen splits it was derived from.

Run:  python s1b_export_heldout_hashes.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

V3_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = V3_ROOT / "data" / "splits"
OUT_PATH = SPLIT_DIR / "heldout_scenario_hashes.json"
LOG_PATH = V3_ROOT / "logs" / "s1b_export_heldout_hashes.log"


def main() -> dict:
    sources = {}
    hashes: set[str] = set()
    per_file: dict[str, int] = {}

    for name in ("val.jsonl", "test.jsonl"):
        path = SPLIT_DIR / name
        raw = path.read_bytes()
        sources[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        n = 0
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            # Only scenario_hash is retained. Everything else in the row is
            # read into a local and immediately dropped when the loop advances.
            sh = json.loads(line).get("scenario_hash")
            if sh:
                hashes.add(sh)
                n += 1
        per_file[name] = n

    payload = {
        "purpose": "scenario_hash blocklist for Stage 2 family C",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": sources,
        "rows_with_scenario_hash": per_file,
        "distinct_scenario_hashes": len(hashes),
        "scenario_hashes": sorted(hashes),
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        f"s1b_export_heldout_hashes.py\nrun_at={payload['generated_at']}\n"
        f"val.sha256={sources['val.jsonl']['sha256']}\n"
        f"test.sha256={sources['test.jsonl']['sha256']}\n"
        f"distinct_hashes={len(hashes)}\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    p = main()
    print("held-out scenario_hash blocklist")
    print("=" * 60)
    for name, s in p["sources"].items():
        print(f"  {name:12s} sha256={s['sha256'][:32]}...  "
              f"rows_with_hash={p['rows_with_scenario_hash'][name]}")
    print(f"  distinct hashes exported: {p['distinct_scenario_hashes']}")
    print(f"  -> {OUT_PATH}")
