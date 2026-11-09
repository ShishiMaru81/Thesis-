#!/usr/bin/env python3
"""
s2b_render_train.py — render the v3 QA corpus into the prompt/completion chat
shape that trl's `completion_only_loss` actually masks against.

WHY THIS STAGE EXISTS
---------------------
s0-s2 produce rows in a flat QA schema (question / answer / scenario /
gold_trace / provenance ...). trl 0.24.0's SFTTrainer masks the prompt ONLY
for datasets carrying "prompt" and "completion" columns — fed anything else,
`completion_only_loss=True` is a silent no-op and loss is computed over the
whole sequence. This is the exact failure documented at length in the v2
trainer's module docstring (scripts/train_v2.py). So the reshape has to happen
before training, not inside it.

THE MAPPING, AND WHY IT IS THE ONLY DEFENSIBLE ONE
--------------------------------------------------
  prompt     = [{"role": "user",      "content": row["question"]}]
  completion = [{"role": "assistant", "content": row["answer"]}]

Three properties of the v3 corpus fix this mapping; none of it is invented:

1. NO SYSTEM PREAMBLE. Every one of v2's 3600 training records was a bare
   (user, assistant) pair with no system turn. v3 inherits that convention —
   adding a preamble here would be a new, unvalidated variable in a 12-run
   comparison whose point is to isolate the base model.

2. SCENARIO IS ALREADY IN THE QUESTION. For all 2994 numeric_runtime rows the
   ET0 / rainfall / area / stage values are rendered into the question text by
   s0's renderers ("...ET0 5.33 mm, rain 11.53 mm, area 1.95 bigha..."). The
   `scenario` dict is the machine-readable mirror of text the model already
   sees. Injecting it a second time would duplicate the inputs.

3. GOLD_TRACE IS GRADING METADATA, NOT A TARGET. It carries kc_source,
   citation, tier_factor, clamp_source and the rest of the audit trail that
   Stage 6 scores against. It is not text any answer contains, and training on
   it would teach the model to emit its own grading key.

DIVERGENCE FROM v2 — DELIBERATE, VERIFIED
------------------------------------------
v2's numeric_runtime target was a tool call:
    <irrigation_calc>{"crop": "Jute", "et0_mm": 3.8, ...}</irrigation_calc>
v3's is the worked answer with the arithmetic shown and the numbers already
computed by irrigation_calc at corpus-build time. Verified: 0 of 3552 v3
answers contain the <irrigation_calc> tag. This stage does NOT reintroduce it;
the change is v3's design, and this renderer preserves the corpus as authored.

Metadata columns are carried through for provenance and for Stage 6's
stratified scoring. s5_train.py selects prompt/completion only before handing
the dataset to the trainer.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (label, source path, output path)
SPLITS = [
    ("train", PROJECT_ROOT / "data" / "augmented" / "train_augmented.jsonl",
     PROJECT_ROOT / "data" / "rendered" / "train.jsonl"),
    ("val", PROJECT_ROOT / "data" / "splits" / "val.jsonl",
     PROJECT_ROOT / "data" / "rendered" / "val.jsonl"),
]

# Carried through for Stage 6 stratification; dropped before the trainer sees it.
META_KEYS = ["row_id", "qtype", "answer_kind", "lang", "crop", "origin",
             "fact_id", "scenario_hash"]

REPORT_PATH = PROJECT_ROOT / "reports" / "gate2b_render.json"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render(rows: list[dict], label: str) -> list[dict]:
    """Flat QA row -> {prompt: [user], completion: [assistant], **meta}."""
    out = []
    for i, r in enumerate(rows):
        q = r.get("question")
        a = r.get("answer")

        # GATE: a blank on either side would train the model on an empty target
        # or an unconditioned answer. Neither is recoverable downstream.
        if not isinstance(q, str) or not q.strip():
            raise SystemExit(f"[{label}] row {i} (row_id={r.get('row_id')}): empty/non-str question")
        if not isinstance(a, str) or not a.strip():
            raise SystemExit(f"[{label}] row {i} (row_id={r.get('row_id')}): empty/non-str answer")

        rec = {
            "prompt": [{"role": "user", "content": q}],
            "completion": [{"role": "assistant", "content": a}],
        }
        for k in META_KEYS:
            if k in r:
                rec[k] = r[k]
        out.append(rec)
    return out


def main() -> int:
    summary = {}

    for label, src, dst in SPLITS:
        if not src.exists():
            raise SystemExit(f"missing source corpus: {src}")

        rows = load_jsonl(src)
        rendered = render(rows, label)

        # GATE: v3 answers are worked prose, not tool calls. If this ever fires,
        # the corpus changed shape and the docstring's rationale needs revisiting.
        tagged = sum(1 for r in rendered
                     if "<irrigation_calc>" in r["completion"][0]["content"])
        if tagged:
            raise SystemExit(
                f"[{label}] {tagged} answers contain <irrigation_calc>; v3 expects worked prose")

        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8") as f:
            for rec in rendered:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        by_qtype = dict(sorted(collections.Counter(
            r.get("qtype") for r in rendered).items(), key=lambda kv: -kv[1]))
        by_lang = dict(sorted(collections.Counter(
            r.get("lang") for r in rendered).items()))

        summary[label] = {
            "source": str(src.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "output": str(dst.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "rows": len(rendered),
            "by_qtype": by_qtype,
            "by_lang": by_lang,
            "sha256": sha256_of(dst),
        }

        print(f"{label:>5}: {len(rendered):>5} rows -> {dst.relative_to(PROJECT_ROOT)}")
        print(f"        qtype={by_qtype}")
        print(f"        lang ={by_lang}")
        print(f"        sha256={summary[label]['sha256']}")

    # Show one rendered record per qtype so the shape is eyeballable.
    train_rendered = load_jsonl(SPLITS[0][2])
    print("\n" + "=" * 72)
    print("RENDERED SAMPLES — one per qtype")
    print("=" * 72)
    seen = set()
    for rec in train_rendered:
        qt = rec.get("qtype")
        if qt in seen:
            continue
        seen.add(qt)
        print(f"\n--- qtype={qt}  lang={rec.get('lang')}  row_id={rec.get('row_id')} ---")
        print(f"  prompt[0]     role={rec['prompt'][0]['role']}")
        print(f"                {rec['prompt'][0]['content'][:220]}")
        print(f"  completion[0] role={rec['completion'][0]['role']}")
        print(f"                {rec['completion'][0]['content'][:220]}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump({
            "stage": "s2b_render",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "mapping": {
                "prompt": "[{role: user, content: question}]",
                "completion": "[{role: assistant, content: answer}]",
                "system_preamble": None,
                "scenario": "not injected - already rendered into question text by s0",
                "gold_trace": "not injected - grading metadata for Stage 6",
            },
            "splits": summary,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nReport -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("s2b render complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
