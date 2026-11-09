#!/usr/bin/env python3
"""Assemble the dataset-source zip. Reads existing files only; writes the package folder and zip."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

THESIS = Path(r"D:\Thesis code files\Thesis")
V3 = THESIS / "v3"
STAGE = THESIS / "_dataset_source_package"
ZIP_PATH = THESIS / "dataset_source_package.zip"
IMG_DIR = THESIS / "crop_calender_images"
JSON_DIR = THESIS / "Json_Dataset(Calander)"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_stem(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", Path(name).stem.lower())


def copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()

    gen_v3 = [
        "s0_build_seed_corpus.py", "irrigation_calc.py", "bn_normalize.py",
        "s1_split_first.py", "s1b_export_heldout_hashes.py",
        "s2_augment_train.py", "s2b_render_train.py", "s4_harden_test.py",
        "s6b_generate_clarify.py", "s6b_merge_clarify_train.py",
    ]
    for name in gen_v3:
        copy(V3 / "scripts" / name, STAGE / "01_generation_scripts" / "v3" / name)
    for src, name in [
        (JSON_DIR / "generate_qa.py", "generate_qa.py"),
        (THESIS / "scripts" / "regenerate_numeric_qa.py", "regenerate_numeric_qa.py"),
        (THESIS / "scripts" / "split_v2.py", "split_v2.py"),
        (THESIS / "scripts" / "irrigation_calc.py", "irrigation_calc.py"),
        (THESIS / "scripts" / "irrigation_bridge.py", "irrigation_bridge.py"),
    ]:
        copy(src, STAGE / "01_generation_scripts" / "v2" / name)

    for name in ["s3_leakage_audit.py"]:
        copy(V3 / "scripts" / name, STAGE / "02_validation_scripts" / name)
    copy(V3 / "scripts" / "thesis" / "leakage_review_sample.py",
         STAGE / "02_validation_scripts" / "leakage_review_sample.py")
    for name in ["validate_qa.py", "audit_sample.py", "build_audit_sheet.py", "compute_kappa.py"]:
        copy(JSON_DIR / name, STAGE / "02_validation_scripts" / "v2" / name)
    for name in ["leakage_report.md", "leakage_report.json"]:
        copy(V3 / "reports" / "leakage" / name, STAGE / "02_validation_scripts" / "records" / name)
    copy(V3 / "reports" / "leakage" / "review" / "README.md",
         STAGE / "02_validation_scripts" / "records" / "semantic_review_README.md")
    copy(V3 / "reports" / "leakage" / "review" / "semantic_review_sample.csv",
         STAGE / "02_validation_scripts" / "records" / "semantic_review_sample.csv")
    part3 = THESIS / "v4_work" / "results" / "stage8b" / "part3"
    for name in ["leakage_review_R1.xlsx", "leakage_review_R2.xlsx", "leakage_review_note.md"]:
        copy(part3 / name, STAGE / "02_validation_scripts" / "records" / name)

    json_files = sorted(JSON_DIR.glob("*.json"))
    images = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    by_norm: dict[str, list[Path]] = {}
    for img in images:
        by_norm.setdefault(norm_stem(img.name), []).append(img)
        copy(img, STAGE / "03_source_calendars" / "images" / img.name)
    for jf in json_files:
        copy(jf, STAGE / "03_source_calendars" / "json" / jf.name)

    disk_sha = {img: sha256_file(img) for img in images}
    mapping_rows = []
    used_images = set()
    for jf in json_files:
        doc = json.loads(jf.read_text(encoding="utf-8"))
        recorded = doc.get("source_image") or ""
        recorded_sha = doc.get("source_image_sha256") or ""
        exact = IMG_DIR / recorded
        if exact.is_file():
            hit, method = exact, "exact filename"
        else:
            cands = by_norm.get(norm_stem(recorded), [])
            hit, method = (cands[0], "normalized filename") if len(cands) == 1 else (None, "NO MATCH" if not cands else "AMBIGUOUS")
        if hit is not None:
            used_images.add(hit)
        mapping_rows.append({
            "json_file": jf.name,
            "crop_name_en": doc.get("crop_name_en"),
            "crop_name_bn": doc.get("crop_name_bn"),
            "source_image_recorded": recorded,
            "recorded_sha256": recorded_sha,
            "disk_file": hit.name if hit else "",
            "disk_sha256": disk_sha[hit] if hit else "",
            "sha256_match": bool(hit) and disk_sha[hit] == recorded_sha,
            "match_method": method,
            "human_verified": doc.get("human_verified"),
            "in_study_set": True,
        })
    for img in images:
        if img not in used_images:
            mapping_rows.append({
                "json_file": "", "crop_name_en": "", "crop_name_bn": "",
                "source_image_recorded": "", "recorded_sha256": "",
                "disk_file": img.name, "disk_sha256": disk_sha[img],
                "sha256_match": "", "match_method": "image with no study JSON",
                "human_verified": "", "in_study_set": False,
            })
    map_path = STAGE / "03_source_calendars" / "image_to_json_mapping.csv"
    with map_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mapping_rows[0]))
        w.writeheader()
        w.writerows(mapping_rows)

    crop_rows, field_rows = [], []
    feat_keys = set()
    for jf in json_files:
        doc = json.loads(jf.read_text(encoding="utf-8"))
        aud = doc.get("_extraction_audit") or {}
        prov = aud.get("provenance_counts") or {}
        needs = aud.get("fields_needing_human_review") or []
        crop_rows.append({
            "json_file": jf.name,
            "crop_name_en": doc.get("crop_name_en"),
            "human_verified": doc.get("human_verified"),
            "audit_human_verified": aud.get("human_verified"),
            "verification_note": aud.get("verification_note") or "",
            "image_hash_verified": aud.get("image_hash_verified"),
            "fields_total": aud.get("fields_total"),
            "provenance_counts": json.dumps(prov, ensure_ascii=False),
            "n_fields_needing_human_review": len(needs) if isinstance(needs, list) else "",
            "fields_needing_human_review": json.dumps(needs, ensure_ascii=False),
            "extraction_timestamp": doc.get("extraction_timestamp"),
            "source_document": doc.get("source_document"),
        })
        feats = doc.get("irrigation_features") or {}
        for field, val in feats.items():
            if not isinstance(val, dict):
                continue
            feat_keys.update(val)
            field_rows.append({
                "json_file": jf.name,
                "crop_name_en": doc.get("crop_name_en"),
                "field": field,
                "provenance": val.get("provenance"),
                "needs_human_review": val.get("needs_human_review"),
                "exclude_from_qa_generation": val.get("exclude_from_qa_generation"),
                "numeric_backfill_declined": val.get("numeric_backfill_declined"),
                "has_value_en": bool(str(val.get("value_en") or "").strip()),
                "has_value_bn": bool(str(val.get("value_bn") or "").strip()),
            })
    ver = STAGE / "04_source_verification"
    ver.mkdir(parents=True)
    for name, rows in [("verification_by_crop.csv", crop_rows), ("verification_by_field.csv", field_rows)]:
        with (ver / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    copy(JSON_DIR / "audit_sheet.xlsx", ver / "audit_sheet.xlsx")
    copy(JSON_DIR / "audit_sample.jsonl", ver / "audit_sample.jsonl")
    for name in ["digitisation_check_C1.xlsx", "digitisation_check_C2.xlsx", "digitisation_sample_manifest.json"]:
        copy(part3 / name, ver / name)

    docs = STAGE / "05_documentation" / "gate_reports"
    for name in ["gate0_seed_corpus.json", "gate1_split.json", "gate2_augment.json",
                 "gate2b_render.json", "gate4_hardened.json"]:
        copy(V3 / "reports" / name, docs / name)
    for name in ["generate_report.json", "merge_report.json"]:
        copy(V3 / "reports" / "s6b" / name, docs / name)

    seed_keys = set()
    seed_path = V3 / "data" / "seed" / "seed_corpus.jsonl"
    n_seed = 0
    with seed_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seed_keys.update(json.loads(line))
                n_seed += 1

    test_path = V3 / "data" / "splits" / "test.jsonl"
    sha_path = V3 / "data" / "splits" / "test.sha256"
    recorded = sha_path.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(test_path)
    split_dir = STAGE / "06_original_seed_test_split"
    copy(test_path, split_dir / "test.jsonl")
    copy(sha_path, split_dir / "test.sha256")
    (split_dir / "hash_check.txt").write_text(
        f"file: data/splits/test.jsonl\n"
        f"recorded in test.sha256: {recorded}\n"
        f"sha256 of the copied file: {actual}\n"
        f"match: {actual == recorded}\n",
        encoding="utf-8", newline="\n")

    n_sha_match = sum(1 for r in mapping_rows if r["sha256_match"] is True)
    n_named = sum(1 for r in mapping_rows if r["in_study_set"])
    n_extra = sum(1 for r in mapping_rows if not r["in_study_set"])
    n_verified = sum(1 for r in crop_rows if r["human_verified"] is True)
    prov_counts: dict[str, int] = {}
    for r in field_rows:
        prov_counts[str(r["provenance"])] = prov_counts.get(str(r["provenance"]), 0) + 1
    notes = sorted({(r["crop_name_en"], r["verification_note"]) for r in crop_rows})
    note_groups: dict[str, list[str]] = {}
    for crop, note in notes:
        note_groups.setdefault(note, []).append(crop)
    note_lines = []
    for note, crops in note_groups.items():
        if len(crops) > 3:
            who = f"{len(crops)} files (every crop except the ones listed under the other note)"
        else:
            who = ", ".join(crops) + " only"
        note_lines.append(f"- {who}: \"{note}\"")

    unmatched = [r for r in mapping_rows if r["in_study_set"] and r["match_method"] in {"NO MATCH", "AMBIGUOUS"}]
    sha_ok = [r["crop_name_en"] for r in mapping_rows if r["sha256_match"] is True]

    readme = f"""# Dataset source package

This zip holds the materials used to build the irrigation question-answer corpus: the generation and validation scripts, the BARC calendar images and their JSON extractions, the verification record, the gate reports, and the original seed test split.

It does not contain the finished training files, the crop-coefficient table, the thesis, or the journal template.

Numbers below are read from the files in this package or from the gate reports copied into `05_documentation/gate_reports`.

## 1. Dataset-generation scripts

`01_generation_scripts/v3` is the pipeline that produced the corpus used in the thesis. Run order:

| Script | Role |
|---|---|
| `s0_build_seed_corpus.py` | One canonical question per fact and language, from the 28 JSON extractions. Answers are retrieved from a field, computed by `irrigation_calc.py`, or a refusal. Absent fields are skipped. |
| `irrigation_calc.py` | The only irrigation-arithmetic implementation used by Stage 0 and Stage 2. |
| `bn_normalize.py` | Bengali normaliser used while augmenting (version v3.1 in the Gate 2 report). |
| `s1_split_first.py` | Splits the 902-row seed by `source_id` into train, validation and test, then checks question-text and scenario-hash overlap. |
| `s1b_export_heldout_hashes.py` | Writes the held-out scenario-hash blocklist. Augmentation reads that list and does not open the held-out splits. |
| `s2_augment_train.py` | Augments the training split only. Families A (paraphrase) and B (back-translation) are implemented but were not run. Gate 2 records `families_run` = `CD`: 632 seed + 2,610 synthetic numeric + 310 noise = 3,552 rows. |
| `s2b_render_train.py` | Rewrites each row as a user prompt and an assistant completion. |
| `s4_harden_test.py` | Adds the adversarial test items and marks rows with no reference answer `pending_human`. |
| `s6b_generate_clarify.py` | Builds the 300 clarification examples by dropping one required input from training numeric rows. |
| `s6b_merge_clarify_train.py` | Appends those 300 rows to the 3,552, giving 3,852. |

`01_generation_scripts/v2` is the earlier generate-then-split code (`generate_qa.py`, `split_v2.py`, `regenerate_numeric_qa.py`). Stage 0 does not read its outputs. It is included because it is the script that first turned the JSON extractions into questions.

## 2. Data-validation scripts

| Script | What it checks |
|---|---|
| `s1_split_first.py` (in folder 01) | Duplicate `source_id`, `fact_id` and `scenario_hash` across splits, and identical normalised question text across splits. |
| `s3_leakage_audit.py` | Character-overlap, embedding, and scenario-hash leakage between train and the held-out splits. Output: `records/leakage_report.md`. |
| `leakage_review_sample.py` | Draws the 60-pair human review sample. The two reviewer workbooks in `records/` have an empty decision column. |
| `v2/validate_qa.py` | Recomputes numeric answers in the v2 `qa_out` file and compares them with the numbers written in the answer text. |
| `v2/audit_sample.py`, `build_audit_sheet.py`, `compute_kappa.py` | Draw and score the v2 human-audit sample. |

Gate 1 (`05_documentation/gate_reports/gate1_split.json`) records zero overlap of `source_id`, `fact_id`, `scenario_hash` and normalised question text between train, validation and test.

## 3. Source calendar records

- `03_source_calendars/json`: 28 crop JSON files, one study crop each.
- `03_source_calendars/images`: all {len(images)} image files in `crop_calender_images`. {n_named} match a study JSON. {n_extra} do not (duplicate scans and a crop page outside the 28).
- `03_source_calendars/image_to_json_mapping.csv`: JSON file, the `source_image` name stored in that JSON, the file found on disk, both SHA-256 values, and whether they match.

Filename note. Several JSON files store a name that is not the name on disk (`Screenshot_20260521_004739.png` against `Screenshot 2026-05-21 004739.png`, `Lichi11.jpg` against `Lichi1(1).jpg`, spaces against underscores). The mapping uses the file name after removing spaces, underscores, hyphens and brackets. Unmatched study JSON files: {len(unmatched)}.

Hash note. The SHA-256 stored in the JSON matches the file now on disk for {n_sha_match} of {n_named} study crops{(': ' + ', '.join(sha_ok)) if sha_ok else ''}. The other names match after normalisation, but the bytes do not. `digitisation_sample_manifest.json` records the same finding and says Brinjal was spot-checked by eye and treated as a re-saved image. Inside 27 of the JSON files, `_extraction_audit.image_hash_verified` is true. That flag was written when the JSON was created. It does not mean the file in this folder still has that hash. Coconut has no `image_hash_verified` field.

## 4. Source verification evidence

Every study JSON has top-level `human_verified` = false ({n_verified} of {len(crop_rows)} true).

{chr(10).join(note_lines)}

Field provenance counts across the 28 files (`verification_by_field.csv`): {prov_counts}.

`absent` means the calendar does not state that field. Those fields were not filled in. No file in this package records a later manual correction of an extracted fact.

Two review instruments were prepared and not filled:

- `audit_sheet.xlsx`: 90 sampled question-answer rows. `Reviewer1_verdict` and `Reviewer2_verdict` are empty.
- `digitisation_check_C1.xlsx` and `digitisation_check_C2.xlsx`: the same 50 facts, one sheet per checker. The column `Checker: Correct / Wrong / Not found in image` is empty. The fact list is `digitisation_sample_manifest.json`.

So the extracted crop facts were left unverified. There is no sheet of corrections.

## 5. Dataset documentation

- This file.
- `05_documentation/DATA_DICTIONARY.md`.
- `05_documentation/gate_reports`: Gate 0 (902 seed rows), Gate 1 (split), Gate 2 (augmentation), Gate 2b (render), Gate 4 (hardened test), and the Stage 6B clarification reports.

The seed file scanned for the dictionary has {n_seed} rows and was not copied into this zip.

## 6. Original seed test split

`06_original_seed_test_split/test.jsonl` is `v3/data/splits/test.jsonl`.

`test.sha256` records `{recorded}`.

SHA-256 of the file in this zip: `{actual}`.

Match: {actual == recorded}.

This is the 182-row seed test split from Gate 1, not the later 422-row hardened test.
"""
    (STAGE / "README.md").write_text(readme, encoding="utf-8", newline="\n")

    dict_md = f"""# Data dictionary

Keys below are the keys present in the files, not a wished-for schema.

## Crop JSON (`03_source_calendars/json`, 28 files)

Top-level keys seen in every file: `crop_name_bn`, `crop_name_en`, `source_document`, `source_image`, `source_image_sha256`, `extraction_timestamp`, `human_verified`, `irrigation_features`, `computable_link`, `_extraction_audit`.

`Coconut.json` also has `extraction_model`. Most other files also have `source_image_path` and `source_image_bytes`.

`human_verified` is false in all 28 files.

### `irrigation_features`

A dictionary of calendar fields. Keys inside each field object: {sorted(feat_keys)}.

`provenance` values and how many fields carry each one: {prov_counts}.

`explicit` was read off the calendar. `derived` was obtained from other stated fields by a stated rule. `absent` is not on the calendar and must not be imputed. Stage 0 skips `absent`, and also skips a field flagged `exclude_from_qa_generation`, `needs_human_review` or `numeric_backfill_declined`.

### `_extraction_audit`

Per-file summary copied to `04_source_verification/verification_by_crop.csv`: field counts, provenance counts, the list of fields still needing review, `human_verified`, `verification_note`, and `image_hash_verified` where that key exists.

## Seed row (`v3/data/seed/seed_corpus.jsonl`, {n_seed} rows, not included in this zip)

Keys present on at least one row: {sorted(seed_keys)}.

Stage 0 writes one row per fact and language. `qtype` is `factual`, `numeric_runtime` or `abstain`. Numeric answers come from `irrigation_calc.py`. The scenario values are drawn with `random.Random(42)` and identified by `scenario_hash`.

## Splits

| File | Rows | Role |
|---|---|---|
| `data/splits/train_seed.jsonl` | 632 | Seed rows later augmented. Not in this zip. |
| `data/splits/val.jsonl` | 88 | Validation. Not in this zip. |
| `data/splits/test.jsonl` | 182 | Seed test. Included in folder 06. SHA-256 matches `test.sha256`: {actual == recorded}. |

632 + 88 + 182 = 902, the seed count in `gate0_seed_corpus.json`.

## Augmented and hardened files (not in this zip)

Gate 2: `train_augmented.jsonl`, 3,552 rows = 632 seed + 2,610 synthetic + 310 noise. No paraphrase or back-translation rows.

Gate 2b: the same 3,552 rows rendered to `data/rendered/train.jsonl` as `prompt` / `completion`.

Stage 6B: `s6b_clarify_train.jsonl` (300) merged into `train_augmented_clarify.jsonl` (3,852).

Gate 4: `test_hardened.jsonl`, 422 rows = 182 seed + 240 adversarial. 60 of those have `gold_status` = `pending_human` (40 out-of-domain, 20 conflicting values) and an empty answer.
"""
    (STAGE / "05_documentation" / "DATA_DICTIONARY.md").write_text(dict_md, encoding="utf-8", newline="\n")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(STAGE.parent).as_posix())

    print("zip", ZIP_PATH, "bytes", ZIP_PATH.stat().st_size)
    print("json", len(json_files), "images", len(images), "sha_match", n_sha_match, "unmatched", len(unmatched))
    print("test hash match", actual == recorded)
    print("human_verified true", n_verified, "of", len(crop_rows))
    print("field provenance", prov_counts)
    print("notes", len(notes))
    for r in mapping_rows:
        if r["in_study_set"] and not r["sha256_match"]:
            print(f"  hash differs: {r['json_file']} recorded={r['source_image_recorded']} disk={r['disk_file']} via {r['match_method']}")
        if not r["in_study_set"]:
            print(f"  extra image: {r['disk_file']}")


if __name__ == "__main__":
    main()
