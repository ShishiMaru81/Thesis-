#!/usr/bin/env python3
"""
verify_adapters.py — close the skipped manifest check for the Stage 6 arms.

For each of the 20 Stage 6 arms:
  * zero-shot arms: confirm the evaluation loaded base weights only
    (meta n_lora_modules == 0 and a "<- base weights (zero-shot)" load line);
  * adapter arms: find the adapter directory the evaluation log says it loaded,
    compute the SHA-256 of adapter_model.safetensors, and compare it with every
    training-time record that exists: the manifest (reports/s5/<arm>.json
    adapter_sha256), the training log ("adapter sha256: ..."), and the sidecar
    file adapter.sha256 when present. Also compare the number of live LoRA
    modules at evaluation (reports/s6/<arm>_meta.json) with the manifest.

Reading only. Writes thesis/generated/adapter_verification.json.
"""
from __future__ import annotations

import datetime as dt
import json
import re

from common import GEN, MODELS, ROOT, SEEDS, read_text_any, rel, sha256_file, write_json

LOGS = ROOT / "logs"
EVAL_LOGS = sorted(p for p in LOGS.glob("*.log") if p.name.startswith(("s6", "s6d", "s6e")))
LOAD_RE = re.compile(r"loading (\S+)\s+<-\s+(.+?)\s*$")


def load_lines() -> dict[str, set[str]]:
    seen: dict[str, set[str]] = {}
    for p in EVAL_LOGS:
        for line in read_text_any(p).splitlines():
            m = LOAD_RE.search(line)
            if m:
                seen.setdefault(m.group(1), set()).add(m.group(2).strip())
    return seen


def training_log_sha(arm: str) -> tuple[str | None, str | None]:
    cands = sorted((LOGS / "s5").glob(f"{arm}_2*.log")) + [LOGS / f"train_{arm}.log"]
    for p in cands:
        if p.exists():
            m = re.findall(r"adapter sha256: ([0-9a-f]{64})", read_text_any(p))
            if m:
                return m[-1], rel(p)
    return None, None


def main() -> None:
    loads = load_lines()
    arms = []
    for key, _, _ in MODELS:
        arms.append((f"{key}_zeroshot", None))
        arms += [(f"{key}_seed{s}", "s5") for s in SEEDS]
        arms.append((f"{key}_seed42_clarify", "clarify"))
    records, problems = [], []
    for arm, kind in arms:
        meta_p = ROOT / "reports" / "s6" / f"{arm}_meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        rec = {"arm": arm, "eval_meta": rel(meta_p), "eval_timestamp": meta["timestamp"],
               "eval_manifest_verify": meta["entry_gate"]["manifest_verify"],
               "eval_n_lora_modules": meta["n_lora_modules"],
               "eval_load_lines": sorted(loads.get(arm, []))}
        if kind is None:
            ok = meta["n_lora_modules"] == 0 and any("base weights" in s for s in loads.get(arm, []))
            rec.update({"adapter": None, "status": "PASS (base weights, 0 LoRA modules)" if ok else "FAIL"})
            if not ok:
                problems.append(f"{arm}: zero-shot arm did not load pure base weights")
            records.append(rec)
            continue
        paths = sorted(loads.get(arm, []))
        if len(paths) != 1:
            problems.append(f"{arm}: expected one adapter load path in eval logs, found {paths}")
        m = re.search(r"(models[\\/]\S+?[\\/]final_adapter)", paths[0]) if paths else None
        adir = ROOT / m.group(1).replace("\\", "/") if m else None
        f = adir / "adapter_model.safetensors" if adir else None
        rec["adapter_file"] = rel(f) if f else None
        if not f or not f.exists():
            rec["status"] = "FAIL (adapter file missing)"
            problems.append(f"{arm}: adapter file missing")
            records.append(rec)
            continue
        got = sha256_file(f)
        rec["sha256_on_disk"] = got
        rec["adapter_mtime"] = dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")
        rec["adapter_older_than_eval"] = rec["adapter_mtime"] < meta["timestamp"]
        man_p = ROOT / "reports" / "s5" / f"{arm}.json"
        refs = {}
        if man_p.exists():
            man = json.loads(man_p.read_text(encoding="utf-8"))
            refs["manifest"] = (man.get("adapter_sha256"), rel(man_p))
            rec["manifest_lora_modules"] = sum(man["lora_modules_attached"].values())
            if rec["manifest_lora_modules"] != meta["n_lora_modules"]:
                problems.append(f"{arm}: eval LoRA modules {meta['n_lora_modules']} != manifest {rec['manifest_lora_modules']}")
        lsha, lsrc = training_log_sha(arm)
        if lsha:
            refs["training_log"] = (lsha, lsrc)
        side = adir / "adapter.sha256"
        if side.exists():
            refs["sidecar"] = (side.read_text(encoding="utf-8").split()[0], rel(side))
        rec["records"] = {k: {"sha256": v[0], "source": v[1], "match": v[0] == got} for k, v in refs.items()}
        missing = [k for k in ("manifest", "training_log") if k not in refs]
        rec["missing_records"] = missing
        allmatch = bool(refs) and all(v[0] == got for v in refs.values())
        rec["status"] = "PASS" if allmatch else "FAIL"
        if not allmatch:
            problems.append(f"{arm}: SHA mismatch {rec['records']}")
        for k in missing:
            problems.append(f"{arm}: no {k} record")
        records.append(rec)
    adapter_arms = [r for r in records if r.get("adapter_file")]
    out = {"n_arms": len(records), "n_adapter_arms": len(adapter_arms),
           "n_zero_shot_arms": len(records) - len(adapter_arms),
           "n_adapter_pass": sum(r["status"] == "PASS" for r in adapter_arms),
           "n_eval_manifest_check_skipped": sum(r["eval_manifest_verify"] == "skipped" for r in records),
           "hash": "SHA-256 of final_adapter/adapter_model.safetensors (the same file s5_train.py hashes)",
           "problems": problems, "arms": records}
    write_json(GEN / "adapter_verification.json", out)
    print(f"{out['n_adapter_pass']}/{out['n_adapter_arms']} adapter arms PASS; "
          f"{out['n_zero_shot_arms']} zero-shot arms; problems: {len(problems)}")
    for p in problems:
        print("  -", p)


if __name__ == "__main__":
    main()
