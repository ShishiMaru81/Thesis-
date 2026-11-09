# Build report

Thesis compile of 2026-09-24. Working directory: `D:\Research Lab\Thesis\v3\thesis`.
PDF: `thesis/main.pdf` (95 pages, 691,009 bytes, valid `%PDF` / `%%EOF`).
`latexmk` exit 0 after the three source fixes listed under Compile.

This file is the decision record promised by `scripts/thesis/make_numbers.py` and `main.tex`.

## 1. Toolchain

No TeX Live is on the system `PATH`. The build uses the session TinyTeX tree:

`c:\Users\user4\AppData\Local\Temp\claude\d--Research-Lab\2c6464ac-4b57-4057-84c3-519c159958b1\scratchpad\tex\TinyTeX`

| Tool | Version |
|---|---|
| XeTeX | 3.141592653-2.6-0.999998 (TeX Live 2026) |
| latexmk | 4.88 (9 March 2026) |
| biber | 2.22 |
| makeindex | 2.18 (nomencl) |

System Perl is absent. TinyTeX's `latexmk.exe` bundles its own Perl (`runscript.tlu`).
Recipe: `latexmkrc` (`$pdf_mode = 5`, biber when `.bcf` changes, nomencl via `makeindex -s nomencl.ist`).

To rebuild from this machine:

```
$env:PATH = "<TinyTeX>\bin\windows;" + $env:PATH
cd "D:\Research Lab\Thesis\v3\thesis"
latexmk main.tex
```

On Overleaf: compiler XeLaTeX. The same `latexmkrc` is read there.

## 2. Compile

First full pass after Chapters 3, 5, 6 and Appendices A–H failed (`latexmk` exit 12). Three fixes, then a clean rebuild:

1. `bibliography/references.bib` — `Computer Speech & Language` → `Computer Speech \& Language` (`vanderlee2021human`). Unescaped `&` is a misplaced alignment tab inside `\printbibliography`.
2. `chapters/chapter_5.tex` — Bengali unit গণ্ডা wrapped in `\bn{}`. latexmk treats missing glyphs in TeX Gyre Termes as a hard error.
3. `chapters/chapter_4.tex` — `\input{generated/tables/t07_metrics}` added next to the metrics section. `tab:metrics` was cited but the table was never included.

Second `latexmk` run: exit 0, two XeLaTeX passes, xdvipdfmx → `main.pdf`. Final log has no undefined references, no missing characters, no citation errors.

Known non-fatal issues (not treated as compile failures):

- Overfull boxes on long `\texttt{...}` paths and a few table captions (T8, Appendix G hashes).
- Underfull boxes inside the F15 example minipages (verbatim model text).
- `biblatex` info lines for missing optional `ieee.dbx` / `biblatex-dm.cfg` (stock IEEE style).

## 3. Manuscript state

| File | Role | Approx. body words (comments and macros stripped) |
|---|---|---|
| `chapters/chapter_1.tex` | Introduction | 2,047 |
| `chapters/chapter_2.tex` | Literature review; T15 | 2,977 |
| `chapters/chapter_3.tex` | Requirements / impacts / constraints; `\label{sec:standards}` | 1,855 |
| `chapters/chapter_4.tex` | Method; T1–T7, T11 | 4,513 |
| `chapters/chapter_5.tex` | Results; `\label{sec:threats}`; T8–T10, T12–T14; F5–F12, F14–F18 | 2,045 |
| `chapters/chapter_6.tex` | Conclusion; RQ answers; non-claims; future work | 820 |
| `appendix/appendix_A.tex` … `H.tex` | Worked examples, per-seed, per-category, rubric, consent, detector, hashes, AI declaration | ~1,260 |
| `core/ethics_statement.tex` | Ethics | 360 |

Cross-refs required by earlier chapters all exist: `sec:standards`, `sec:threats`, `app:detector` / `app:F`, `app:rubric` / `app:D`, `app:ai` / `app:H`, plus `app:A`–`app:H` dual labels.

Tables T1–T15 and appendix tables B, C, G are generated. Appendix B percents are stored as percents (Llama ZS: `0.9 & 0.0 & 100.0 & 14.7 & 11.9`); an earlier generator that multiplied by 100 was corrected in `scripts/thesis/appendices.py` before this compile.

Bibliography: 86 entries in `bibliography/references.bib`, 86 verified in `generated/citation_check.csv`. Only cited keys print (`biblatex` + `sorting=none`).

## 4. Deviations from the BRAC CSE template

Base: BracU Thesis Template, Ayesha Abed Library (FINAL YEAR THESIS Template_CSE400_Fall 2024 ONWARDS), `report` class, A4, 150 mm text width, Times-metric body.

| Deviation | Why |
|---|---|
| XeLaTeX, not pdfLaTeX | Bengali conjuncts via `fontspec` / `Script=Bengali`. |
| `inputenc` removed | XeLaTeX reads UTF-8. |
| Body font TeX Gyre Termes by file name; math `texgyretermes-math.otf` | Times metrics without depending on a system Times. |
| Bundled Noto Serif/Sans Bengali + SIL OFL 1.1 (`fonts/`, `fonts/OFL.txt`, `fonts/SOURCES.md`) | Reproducible Bengali shaping; not a system font. |
| `biblatex` + biber, IEEE numbered, `sorting=none` | Template default is usually `natbib` / author–year or a BST file. Numbered IEEE matches the citation style required for this thesis. |
| `cleveref` with “Fig.” / “Table” / “Chapter” | IEEE-style cross-references. |
| Extra packages: `booktabs`, `multirow`, `makecell`, `xcolor`, `tikz`, `tcolorbox`, `enumitem`, `etoolbox`, `placeins`, `xspace`, `nomencl` | Tables, pipeline TikZ, example boxes, abbreviations list. |
| `\parindent=0em`, `\parskip=0.55em plus 0.1em` | Judgment call: BRAC body is indented; this draft is block paragraphs. Change back if the department requires first-line indent. |
| `\emergencystretch=1.5em` | Long model names and paths. |
| `\let\cleardoublepage=\clearpage` | `oneside` already; avoids blank versos. |
| `generated/numbers.tex` for every factual number | No hand-typed counts, rates, or *p*-values. |
| Red placeholder macros (`\AUTHORINPUT`, `\PENDINGCONSENT`, `\PENDINGREVIEW`, `\STAGEEIGHTB`) | Author-only facts and Stage 8B items that have not run. |
| Figures live in `images/` (not `figures/`) | Path used by `figures.py`. |
| F1 is TikZ (`generated/f01_pipeline.tex`); F15 is a LaTeX box (`generated/f15_examples.tex`) | Matplotlib cannot shape Bengali for F15. |

## 5. Gold-oracle “before” value (0.966)

`make_numbers.py` §11: the gold-oracle score *after* the three post-hoc metric changes is `\ChangeOracleAfter`. The score *before* those changes is not in any surviving file. The repository has no `.git`. The only `0.966` hits in the tree are unrelated slot F1s.

**Decision:** no “before” macro is defined. Chapter 4 and Chapter 5 say the pre-change oracle score is not recorded. We do not invent 0.966 or any other recovered value.

## 6. Placeholders still in the PDF

These are intentional. They print as red boxes. Do not invent the missing text.

### `\AUTHORINPUT` / `\AUTHORINPUTPAR`

| Topic | Where |
|---|---|
| Five student names and IDs | `core/thesis_info.tex` |
| Supervisor, co-supervisor, coordinator, head; designations | same |
| Submission month/year, semester | same |
| Acknowledgement | `core/acknowledgement.tex` |
| Confirm annotator identities / backgrounds | ethics, Ch 3, Ch 4, App D |
| Supervisor answer on ethics approval | ethics, Ch 3 |
| Supervisor written agreement on dropping RAG | Ch 1, Ch 3 |
| BARC terms of use | ethics, Ch 3 |
| Department AI-declaration wording | ethics, Ch 3, App H |
| Consent-form text or pointer to signed PDFs | App E |

### `\PENDINGCONSENT`

Signed consent forms: ethics, Ch 3, Ch 4, App E.

### `\PENDINGREVIEW`

Semantic leakage sample (`\LeakReviewSample` pairs): T3 caption, Ch 4, Ch 5, Ch 6.

### `\STAGEEIGHTB` (designed, not run)

Prompted baseline; digitisation check; crop-coefficient provenance; class-level scoring of the 60 excluded rows; external raters.

`\MISSING` and `\CITENEEDED` are defined but unused in the drafted chapters.

## 7. Determinism

- `scripts/thesis/make_numbers.py` → `generated/numbers.tex` / `numbers.json` / `facts.md`. Last regeneration 2026-09-24 18:28. `0` `\MISSING` uses (the name appears only in `\providecommand`).
- Tables T1–T15, B, C, G: `scripts/thesis/tables.py` and `scripts/thesis/appendices.py`.
- Figures: `scripts/thesis/figures.py` plus F1/F15 TeX.
- References: `scripts/thesis/verify_refs.py`; cache in `generated/refs_cache/`.
- Test-set SHA-256 prefixes and adapter hashes: Appendix G / `tg_hashes.tex`.
- Stage 8 software freeze: `reports/stats/env_freeze.txt`. Plan version `\PlanVersion`.
- R is not on the build machine (noted in Appendix G).

Re-running `make_numbers.py` is expected to be byte-stable unless a source report changes. This compile did not re-run it.

## 8. Fonts and Bengali

Noto Serif/Sans Bengali are bundled. `\bn{...}` selects `\bengalifont` with `Script=Bengali`. A visual conjunct check was done on the skeleton compile (18:10). F15 Bengali is set in LaTeX, not matplotlib.

`scripts/thesis/font_test.py` writes comparison images to `generated/font_test/` when run. That folder is empty in this tree; the script was not re-run for this report. Verdict from the earlier visual check: XeLaTeX + Noto shapes conjuncts correctly; do not put raw Bengali in the Times body font.

## 9. Wording rules kept in the draft

- Numbers only via macros from `generated/numbers.tex`.
- Non-significant results: “no significant difference detected” plus the CI. No “equivalent”, no bare “no difference”.
- A5 detector: oracle-gated descriptive upper bound, not a mitigation result.
- *p*-macros already contain `=~...`; prose writes `$p_\mathrm{Holm}$ \Sig…Pholm` with no extra `=`.

## 10. What Step 5 did not do

- No zip archive. The deliverable PDF is `thesis/main.pdf`.
- No second `make_numbers.py` run (sources unchanged since 18:28).
- Author-only red boxes left in place.
- Overfull `\texttt` paths not rewritten.
- TinyTeX lives in a Claude scratchpad temp directory. If that tree is deleted, install TeX Live / TinyTeX (XeLaTeX, biber, latexmk, tex-gyre, tex-gyre-math, koma-script, nomencl, tcolorbox, booktabs, ieee biblatex) or compile on Overleaf.
