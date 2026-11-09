# Fine-Tuning Creates Confident Errors

### Clarification Failure in Bengali–English Agricultural Question Answering

**Undergraduate thesis · Department of Computer Science and Engineering · BRAC University**

This repository contains the code, bilingual datasets, evaluation records, and manuscript for our undergraduate thesis. The study asks whether fine-tuning an agricultural question-answering model improves task accuracy while also encouraging it to answer incomplete questions with invented inputs.

The main experiment compares four instruction-tuned language models before and after QLoRA fine-tuning. It examines Bengali and English irrigation questions, factual crop-calendar questions, out-of-scope requests, and questions requiring clarification. Follow-up experiments add clarification training examples, explicit instructions and few-shot examples, and a larger model baseline.

## Research overview

An irrigation answer depends on the crop, growth stage, reference evapotranspiration (ET0), rainfall, and field area. If a necessary input is missing, a useful assistant should ask for it. Our experiments distinguish **clarification** from **abstention**: asking for a missing parameter and refusing an out-of-scope question are different behaviours.

The main training corpus contains abstention examples but no clarification examples. Standard fine-tuning improves numerical performance and out-of-scope refusal, yet does not teach the models to request missing inputs. A separate intervention adds 300 clarification examples and measures how much this changes the behaviour.

The study is a controlled evaluation using synthetic and templated questions derived from Bangladesh Agricultural Research Council (BARC) crop calendars and a project-specific irrigation calculator. Its results describe this benchmark and experimental setup; they do not establish field performance with farmers.

## Repository map

| Location | Contents |
|---|---|
| [`v3/scripts/`](v3/scripts/) | Main pipeline: seed creation, splitting, augmentation, leakage audit, test hardening, training, evaluation, and statistics |
| [`v3/data/`](v3/data/) | Seed corpus, locked splits, augmented/rendered training data, hardened test set, and bilingual evaluation samples |
| [`v3/reports/`](v3/reports/) | Gate reports, training manifests, predictions, automatic scores, human ratings, and statistical results |
| [`v3/reproduce/`](v3/reproduce/) | Saved Stage 6 rescoring comparisons and reproduction records |
| [`v3/models/`](v3/models/) | Selected adapter configurations and recorded checksums; weight files are excluded |
| [`v3/thesis/`](v3/thesis/) | LaTeX manuscript, bibliography, generated tables, figures, fonts, and saved PDF |
| [`v4_work/results/stage8b/`](v4_work/results/stage8b/) | Follow-up prompted baselines, hardware replication, class-level probes, and provenance review materials |
| [`v4_work/stage9_frontier/`](v4_work/stage9_frontier/) | Larger-model API experiment code, configuration, frozen plan, and amendments |
| [`v4_work/results/stage9_frontier/`](v4_work/results/stage9_frontier/) | Cached API responses, scores, statistics, and report |
| [`Json_Dataset(Calander)/`](Json_Dataset(Calander)/) | 28 structured crop-calendar extractions and source-validation utilities |
| [`crop_calender_images/`](crop_calender_images/) | 32 source image files, including duplicate scans and material outside the 28-crop study set |
| [`kc_values_hybrid.csv`](kc_values_hybrid.csv) | Crop-coefficient lookup table used by the calculator |
| [`source_documentation/`](source_documentation/) | Source verification records, image mapping, and historical data-package documentation |
| [`scripts/`](scripts/) and [`reports/`](reports/) | Earlier v2 tool-call pipeline and numerical evaluation records, retained for research history |
| [`docs/`](docs/) | Publication inventory and verification notes |

The `v3` experiment is the main thesis pipeline. Earlier v2 scripts train models to emit a calculator tool call; **v3 trains and evaluates prose answers**. These are different experiments, and their metrics should not be combined.

## Dataset

The following counts were checked directly against the published JSONL files.

| Artifact | Rows | Purpose |
|---|---:|---|
| [`seed_corpus.jsonl`](v3/data/seed/seed_corpus.jsonl) | 902 | One canonical question per fact and language; 451 Bengali and 451 English rows |
| [`train_seed.jsonl`](v3/data/splits/train_seed.jsonl) | 632 | Training partition before augmentation |
| [`splits/val.jsonl`](v3/data/splits/val.jsonl) | 88 | Unaugmented validation partition |
| [`splits/test.jsonl`](v3/data/splits/test.jsonl) | 182 | Locked seed test partition |
| [`train_augmented.jsonl`](v3/data/augmented/train_augmented.jsonl) | 3,552 | Training-only augmentation |
| [`rendered/train.jsonl`](v3/data/rendered/train.jsonl) | 3,552 | Prompt/completion training representation |
| [`rendered/val.jsonl`](v3/data/rendered/val.jsonl) | 88 | Prompt/completion validation representation |
| [`s6b_clarify_train.jsonl`](v3/data/augmented/s6b_clarify_train.jsonl) | 300 | Clarification intervention: 150 examples per language |
| [`train_augmented_clarify.jsonl`](v3/data/augmented/train_augmented_clarify.jsonl) | 3,852 | Original training corpus plus clarification examples |
| [`test_hardened.jsonl`](v3/data/hardened/test_hardened.jsonl) | 422 | 182 seed questions plus 240 adversarial questions; 211 per language |

The original augmented corpus contains 2,994 `numeric_runtime`, 430 `factual`, and 128 `abstain` rows, with **zero `clarify` rows**. The hardened test contains misspelled crops, missing parameters, unit ambiguity, multiple crops, code switching, out-of-domain questions, and conflicting values.

For the main Stage 6 evaluation, **362 of 422 test rows are scored**: 146 numeric, 114 factual, 68 clarification, and 34 abstention rows. The remaining 60 have `gold_status = pending_human` and are excluded from that scoring. Stage 8B evaluates those rows at the response-class level without authoring new gold answer text.

Splitting happens before augmentation. Source/fact and scenario identifiers support leakage checks. The original [`test.sha256`](v3/data/splits/test.sha256) and [`test_hardened.sha256`](v3/data/hardened/test_hardened.sha256) files are retained. Git attributes preserve file bytes so that checkout does not change locked dataset hashes.

## Models and training

| Pipeline key | Base model |
|---|---|
| `llama31_8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| `mistral_nemo` | `mistralai/Mistral-Nemo-Instruct-2407` |
| `phi35_mini` | `microsoft/Phi-3.5-mini-instruct` |
| `phi4_mini` | `microsoft/Phi-4-mini-instruct` |

The main matrix contains four zero-shot arms and twelve fine-tuned runs: four models across seeds **42, 1337, and 2024**. Clarification-trained arms use seed 42.

Training uses 4-bit QLoRA through Unsloth, rank 16, alpha 32, dropout 0.05, learning rate `2e-4`, effective batch size 16, maximum sequence length 768, and a fixed three-epoch budget. Microbatch sizes vary by model. Adapter targets also vary by architecture, including Phi-4-mini's fused projections. The training script checks that loss is applied to completion tokens before training begins.

The recorded research environment used an NVIDIA RTX 4080 SUPER with 16 GB VRAM. Historical package versions are recorded in [`env_freeze.txt`](v3/reports/stats/env_freeze.txt) and the training reports. This is an environment snapshot, not a portable installation lockfile.

## Main recorded findings

Numerical accuracy below means an answer within **±0.1 mm on 116 single-crop questions**. Fine-tuned values are mean ± sample standard deviation over the three seeds. The separate 30 multi-crop questions are not included in this denominator.

| Model | Zero-shot accuracy | Fine-tuned accuracy | Fine-tuned numerical MAE (mm) |
|---|---:|---:|---:|
| Llama-3.1-8B | 0.9% | 69.3 ± 2.6% | 0.144 |
| Mistral-NeMo-12B | 3.4% | 67.0 ± 3.5% | 0.112 |
| Phi-3.5-mini | 1.7% | 38.8 ± 1.5% | 0.569 |
| Phi-4-mini | 0.0% | 42.8 ± 1.0% | 0.361 |

Source: [`cross_model_summary.md`](v3/reports/s6/cross_model_summary.md) and [`summary_by_arm.csv`](v3/reports/s6/summary_by_arm.csv). The ±0.1 mm metric is a tolerance-based measure, not literal exact numerical equality.

Across the twelve standard fine-tuned runs, exact clarification is 0% on the 68 scorable clarification probes, while the invented-value rate is 100%. Refusal on the 34 seed abstention questions reaches 100%. Adding clarification examples raises exact clarification to roughly 29–38% across the four models and reduces invented values, but does not eliminate the failure. Multi-crop both-correct accuracy remains 0% in the main comparison.

See [`significance.md`](v3/reports/stats/significance.md) for statistical tests and uncertainty, and [`stage7_audit_report.md`](v3/reports/human_eval/stage7_audit_report.md) for the human-rating audit. Accuracy, response completeness, explicit assumptions, and whether an answer is suitable to give a farmer are separate measures.

### Follow-up experiments

- **Stage 8B:** explicit system instructions and few-shot examples, a full 422-row hardware replication, class-level scoring of the 60 previously excluded rows, and source/provenance review materials. See [`s8b_report.md`](v4_work/results/stage8b/s8b_report.md). The Stage 8B driver scripts referenced by that historical report were not found in the supplied folder; its saved outputs and frozen assets are included.
- **Stage 9:** a larger zero-shot baseline using `openai/gpt-oss-120b` through Groq, with and without the Stage 8B prompt. The saved report records 576 responses: 288 questions per arm. Its primary exact clarification rates are 8.8% without the prompt and 57.4% with it on 68 clarification questions. See [`REPORT.md`](v4_work/results/stage9_frontier/REPORT.md) for the full setup, comparisons, token-budget sensitivity, and pending assumption audit. Three earlier Gemini dry-run responses are retained as excluded historical records.

The saved manuscript build predates these follow-ups. Their presence in the repository does not imply that every later result has been integrated into the manuscript.

## Getting started

Clone the repository and run commands from its root unless a section says otherwise:

```powershell
git clone https://github.com/ShishiMaru81/Thesis-.git
cd Thesis-
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

For an initial check using only the Python standard library:

```powershell
python -X utf8 v3/scripts/irrigation_calc.py
python -X utf8 v3/scripts/s6_eval.py --selftest
Get-FileHash v3/data/hardened/test_hardened.jsonl -Algorithm SHA256
Get-Content v3/data/hardened/test_hardened.sha256
```

The calculator prints a deterministic worked example. The evaluator self-test checks its parser against gold answers; it does not run model inference or reproduce the training results.

### Analysis dependencies

The scripts use NumPy, pandas, SciPy, scikit-learn, statsmodels, SacreBLEU, BERTScore, sentence-transformers, datasketch, FAISS, Krippendorff alpha, matplotlib, Pillow, and openpyxl. Stage 9 additionally uses PyYAML and httpx; the Google provider path uses google-genai. Install the packages needed for the stage being run, using the recorded environment snapshot as the version reference.

Full training additionally needs a compatible CUDA/PyTorch installation, Unsloth, Transformers, TRL, PEFT, bitsandbytes, Accelerate, and datasets. The original environment contains platform-specific builds. Configure a compatible GPU environment before installing this stack; the publication checks did not reinstall it or rerun GPU training. Some model downloads also require access to the relevant Hugging Face model and acceptance of its terms.

### Main pipeline

The distributed data and reports allow inspection without regenerating them. The following commands **regenerate and may overwrite** their corresponding artifacts; use a separate checkout for a new experiment.

```powershell
python v3/scripts/s0_build_seed_corpus.py
python v3/scripts/s1_split_first.py
python v3/scripts/s1b_export_heldout_hashes.py
python v3/scripts/s2_augment_train.py --families CD
python v3/scripts/s2b_render_train.py
python v3/scripts/s3_leakage_audit.py
python v3/scripts/s4_harden_test.py
```

The recorded augmentation uses families C and D. Optional paraphrase/back-translation families A and B were not generated in the main experiment. Review each stage's gate report before proceeding.

Train one model/seed and evaluate an arm:

```powershell
python v3/scripts/s5_train.py --model llama31_8b --seed 42
python v3/scripts/verify_s5_manifests.py
python v3/scripts/s6_eval.py --arms llama31_8b_seed42 --batch_size 16
```

Repeat training for the required model/seed cells to reconstruct the full matrix. Original adapter weights, optimizer states, and model caches are not distributed here, so weight-dependent manifest checks and fine-tuned inference require locally regenerated or separately obtained adapters. Keep those checks enabled for new runs.

Clarification generation and training are implemented by `s6b_generate_clarify.py`, `s6b_merge_clarify_train.py`, and the `s6e_*` scripts. Human evaluation uses the `s7_*` scripts and saved rating sheets. Statistical analysis is organised by `s8_stats.py --part ...`; consult its command-line help and frozen analysis plan before regenerating results.

For Stage 9, inspect `v4_work/stage9_frontier/config.yaml`, the frozen plan, and its amendments. API credentials are read from environment variables such as `GROQ_API_KEY`; keep credentials outside Git. The runner's `--dry-run` makes real API requests, and a full run requires `--confirm`. Existing cached responses and reports can be inspected without making requests. Provider availability, rate limits, and prices in the historical configuration may have changed.

## Irrigation calculation

[`v3/scripts/irrigation_calc.py`](v3/scripts/irrigation_calc.py) provides the arithmetic used to generate numerical targets:

```text
ETc = Kc × ET0
effective_rain = rain × 0.85   when rain ≤ 5 mm
                 rain × 0.45   when 5 < rain ≤ 15 mm
                 rain × 0.25   when rain > 15 mm
net_mm = max(ETc − effective_rain, 0)
liters = net_mm × area_m²
```

The implementation applies rounding, input validation, stage aliases, and area conversions. The rainfall tiers and Kc table are benchmark conventions. They should not be read as a universally validated irrigation prescription. Per-crop BARC min/max clamping is absent because the referenced `irrigation_rules.csv` was not available.

## Thesis manuscript

The manuscript entry point is [`v3/thesis/main.tex`](v3/thesis/main.tex). A saved [`main.pdf`](v3/thesis/main.pdf) and a historical [`BUILD_REPORT.md`](v3/thesis/BUILD_REPORT.md) are included.

With XeLaTeX, latexmk, biber, and the required TeX packages installed:

```powershell
cd v3/thesis
latexmk main.tex
```

Use **XeLaTeX** on Overleaf. The font directory retains its font license files. Generated numbers, tables, and figures are included; their generators live in `v3/scripts/thesis/`. Regenerating them can require analysis dependencies, tokenizer access, or model artifacts beyond the lightweight checks above.

Student and supervisor fields in [`thesis_info.tex`](v3/thesis/core/thesis_info.tex) remain author-input placeholders in the supplied manuscript. The thesis team should fill those fields before submission. Authorship belongs to the undergraduate thesis team; this publication does not add an assistant as a collaborator or co-author.

## Limitations and research status

- The corpus is synthetic and templated, with a limited crop and question distribution. There is no field deployment evaluation in this repository.
- The project is **leakage-audited**, with documented caveats; it should not be described as leakage-free. Consult the leakage reports and semantic-review records.
- Source extraction records contain unresolved human-verification flags. Kc provenance and digitisation review materials are included; their presence alone is not evidence that all review has been completed.
- The main 362-row scoring and later class-level probes use different denominators. Statistical tables also sometimes use all 146 numeric rows; always identify the denominator.
- The Stage 9 human assumption audit is pending in the saved report. Registered phrase matching and completed human auditing are different forms of evidence.
- Historical reports contain original machine paths, timestamps, and references to files outside this publication. Those records are retained as evidence, not as portable setup instructions.
- The local publication check verifies data integrity and lightweight parser/calculator behaviour. It does not constitute a fresh GPU reproduction, statistical reanalysis, or manuscript rebuild.

## Included files and reuse

This publication includes the research code, data, predictions, reports, source verification records, and manuscript needed to understand the supplied work. It excludes the Python environment, downloaded model weights, adapter weight binaries, checkpoints, compiled caches, duplicate ZIP archives, backup manuscript trees, third-party paper collections, and chat handoff/prompt documents. See [`PUBLICATION_NOTES.md`](docs/PUBLICATION_NOTES.md) and the checksummed [`source_manifest.csv`](docs/source_manifest.csv).

No new blanket software or dataset license is assigned by this publication. Source calendars, fonts, models, and other third-party materials retain their respective rights and terms. Refer to the bibliography and source/provenance records when citing underlying material, and contact the thesis repository owner about reuse of the team's original work.
