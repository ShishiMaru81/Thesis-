#!/usr/bin/env python3
"""
figures.py — every thesis figure, one function per figure, from repo files only.

Output: thesis/images/<id>.pdf (vector) and <id>.png (300 dpi) for the
matplotlib figures; F1 (TikZ) and F15 (tcolorbox with verbatim model text)
are written as LaTeX to thesis/generated/, because Bengali must be shaped by
XeLaTeX, never by matplotlib. thesis/generated/figures_manifest.json lists
figure id, function, source files, rows used and a one-line takeaway.

Style (fixed across the thesis):
  * One colour per model (Okabe–Ito subset chosen with scripts/thesis/validate_palette.py,
    all-pairs CVD dE 11.0, normal-vision dE 18.4 on white):
      Llama vermillion, Mistral bluish green, Phi-3.5 sky blue, Phi-4 blue.
  * Variant encoding: zero-shot = light tint with coloured edge (hollow marker);
    fine-tuned = solid fill; clarify-trained = hatched bar / outlined diamond.
  * Orange (#E69F00, unused by any model) marks "clarify" wherever it is the point.
  * No chart titles inside images (captions carry the message); panel letters only.
  * Text >= 8 pt at printed size; widths sized for \\textwidth (150 mm).
  * Seed spread always visible (seed dots). No Bengali text in any matplotlib figure.
Deterministic: fixed RNG seeds, fixed metadata, no timestamps.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

from common import GEN, MODELS, ROOT, SEEDS, TEST_PATH, guard, load_jsonl, rel, verify_test_sha

FIG = ROOT / "thesis" / "images"
TEXTWIDTH = 150 / 25.4          # inches (BRAC template: width=150mm)

COLOR = {"llama31_8b": "#D55E00", "mistral_nemo": "#009E73", "phi35_mini": "#56B4E9", "phi4_mini": "#0072B2"}
LABEL = {k: lab for k, _, lab in MODELS}
SHORT = {"llama31_8b": "Llama", "mistral_nemo": "Mistral", "phi35_mini": "Phi-3.5", "phi4_mini": "Phi-4"}
ORDER = [k for k, _, _ in MODELS]
ACCENT = "#E69F00"
INK, INK2, MUTED, GRID = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#e3e3e3"
SEED_MARK = {42: "o", 1337: "s", 2024: "^"}

S6 = ROOT / "reports" / "s6" / "summary_by_arm.csv"
MANIFEST: list[dict] = []


# --------------------------------------------------------------------------- style
def setup_style() -> None:
    plt.rcParams.update({
        "font.family": ["DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "hatch.linewidth": 0.7, "lines.linewidth": 1.4, "lines.solid_capstyle": "round",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100,
        "svg.hashsalt": "thesis", "axes.titlelocation": "left", "axes.titleweight": "normal",
        "axes.titlepad": 4,
    })


def tint(hexcol: str, f: float = 0.28) -> tuple:
    c = np.array(matplotlib.colors.to_rgb(hexcol))
    return tuple(c * f + (1 - f))


def save(fig, fid: str, function: str, sources: list[str], rows: str, takeaway: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{fid}.pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": "matplotlib",
                                             "Creator": "scripts/thesis/figures.py"})
    fig.savefig(FIG / f"{fid}.png", dpi=300, metadata={"Software": "scripts/thesis/figures.py"})
    plt.close(fig)
    MANIFEST.append({"id": fid, "function": function, "files": [f"thesis/images/{fid}.pdf", f"thesis/images/{fid}.png"],
                     "sources": sources, "rows_used": rows, "takeaway": takeaway})


def panel(ax, letter: str, text: str = "") -> None:
    ax.set_title(f"({letter}) {text}".strip(), loc="left", fontsize=8.5, color=INK)


def hgrid(ax) -> None:
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def vgrid(ax) -> None:
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def summary() -> pd.DataFrame:
    return pd.read_csv(guard(S6)).set_index("arm")


def variant_legend(ax, loc="upper left", with_cl=True, bars=True, **kw):
    c = "#7a7a7a"
    if bars:
        h = [Patch(facecolor=tint(c), edgecolor=c, linewidth=0.8, label="zero-shot (base)"),
             Patch(facecolor=c, edgecolor=c, label="fine-tuned (mean of 3 seeds)")]
        if with_cl:
            h.append(Patch(facecolor="white", edgecolor=c, hatch="////", linewidth=0.8, label="clarify-trained (seed 42)"))
    else:
        h = [Line2D([], [], marker="o", ls="", mfc=tint(c), mec=c, mew=1, ms=6, label="zero-shot (base)"),
             Line2D([], [], marker="o", ls="", mfc=c, mec="white", mew=0.8, ms=6, label="fine-tuned (mean of 3 seeds)")]
        if with_cl:
            h.append(Line2D([], [], marker="D", ls="", mfc=c, mec=INK, mew=0.8, ms=5, label="clarify-trained (seed 42)"))
    h.append(Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=0.6, ms=3.2, label="individual seed"))
    return ax.legend(handles=h, loc=loc, **kw)


def seed_dots(ax, x, vals, horizontal=False, spread=0.09, edge=INK):
    offs = np.linspace(-spread, spread, len(vals))
    for o, v in zip(offs, vals):
        if horizontal:
            ax.plot(v, x + o, marker="o", ms=3.2, mfc="white", mec=edge, mew=0.7, ls="", zorder=5)
        else:
            ax.plot(x + o, v, marker="o", ms=3.2, mfc="white", mec=edge, mew=0.7, ls="", zorder=5)


# =========================================================================== F1 (TikZ)
F1_TIKZ = r"""% AUTO-GENERATED by scripts/thesis/figures.py (f1_pipeline_tikz). Numbers are macros from numbers.tex.
\begin{tikzpicture}[
  font=\footnotesize, node distance=4mm and 5mm,
  stage/.style={draw=black!55, rounded corners=2pt, align=center, inner sep=3pt, minimum height=13mm,
                text width=22mm, fill=white},
  key/.style={stage, draw=black, line width=0.9pt},
  gate/.style={stage, draw=black, line width=0.9pt, fill=black!6},
  flow/.style={-{Stealth[length=2.2mm]}, draw=black!70, line width=0.6pt},
  side/.style={font=\scriptsize\itshape, text=black!55}]
  \node[stage] (s0) {\textbf{S0 Digitise}\\\SourceFiles calendars\\\SeedRows rows, \SeedFacts facts};
  \node[key, right=of s0] (s1) {\textbf{S1 Split first}\\by source group\\\SplitRatioTrain/\SplitRatioVal/\SplitRatioTest\,\%};
  \node[gate, right=of s1] (s3) {\textbf{S3 Leakage gate}\\four detectors\\train vs held-out};
  \node[key, above=of s3] (s2) {\textbf{S2 Augment}\\training split only\\\AugRowsSeed{} $\rightarrow$ \AugRows};
  \node[stage, below=of s3] (ho) {\textbf{Held-out split}\\val \ValRows, test \TestSeedRows\\SHA-256 locked};
  \node[stage, right=of s2] (s5) {\textbf{S5 Fine-tune}\\QLoRA, four models\\$\times$ three seeds};
  \node[key, right=of ho] (s4) {\textbf{S4 Harden test}\\+\TestRowsAdv{} adversarial\\\TestRows{} rows, \TestScorable{} scored};
  \node[stage, right=of s5] (s6b) {\textbf{S6B--E Clarify}\\+\ClarifyTrainRows{} examples\\seed 42 only};
  \node[stage, right=of s4] (s6) {\textbf{S6--S7 Evaluate}\\automatic metrics\\\HumAnnotators{} raters, \HumJudgments{} ratings};
  \node[gate, below=of s6] (s8) {\textbf{S8 Statistics}\\plan fixed first\\\SigPrimaryTotal{} primary tests};
  \node[side, above=1mm of s2.north west, anchor=south west] {training side};
  \node[side, below=1mm of ho.south west, anchor=north west] {held-out side (never augmented)};
  \draw[flow] (s0) -- (s1);
  \draw[flow] (s1.north) |- (s2.west);
  \draw[flow] (s1.south) |- (ho.west);
  \draw[flow, draw=black] (s2) -- (s3);
  \draw[flow, draw=black] (ho) -- (s3);
  \draw[flow] (s2) -- (s5);
  \draw[flow] (s5) -- (s6b);
  \draw[flow] (ho) -- (s4);
  \draw[flow] (s4) -- (s6);
  \draw[flow] (s5.south) -- ++(0,-3mm) -| ([xshift=-4mm]s6.north);
  \draw[flow] (s6b.south) -- ([xshift=4mm]s6.north);
  \draw[flow] (s6) -- (s8);
\end{tikzpicture}
"""


def f1_pipeline_tikz() -> None:
    (GEN / "f01_pipeline.tex").write_text(F1_TIKZ, encoding="utf-8", newline="\n")
    MANIFEST.append({"id": "f01_pipeline", "function": "f1_pipeline_tikz", "files": ["thesis/generated/f01_pipeline.tex"],
                     "sources": ["thesis/generated/numbers.tex (macros: SourceFiles, SeedRows, SeedFacts, SplitRatio*, "
                                 "AugRowsSeed, AugRows, ValRows, TestSeedRows, TestRowsAdv, TestRows, TestScorable, "
                                 "ClarifyTrainRows, HumAnnotators, HumJudgments, SigPrimaryTotal)"],
                     "rows_used": "gate reports (counts only, via macros)",
                     "takeaway": "The corpus was split by source before any augmentation; only the training side was augmented, and a four-detector leakage gate stood between the two sides before the test set was hardened and locked."})




# =========================================================================== F2
def f2_data_composition() -> None:
    g1 = json.loads((ROOT / "reports/gate1_split.json").read_text(encoding="utf-8"))
    g2 = json.loads((ROOT / "reports/gate2_augment.json").read_text(encoding="utf-8"))
    g4 = json.loads((ROOT / "reports/gate4_hardened.json").read_text(encoding="utf-8"))
    eg = json.loads((ROOT / "reports/stats/entry_gate.json").read_text(encoding="utf-8"))
    tab = {r["split"]: r["rows"] for r in g1["table"]}
    orig = {r["origin"]: r["rows"] for r in g2["table_by_origin"]}
    sc = eg["scorable_by_qtype"]
    n_sc = sum(sc.values())
    excl = sum(eg["pending_by_category"].values())
    adv = g4["composition"]["adversarial_rows"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(TEXTWIDTH, 4.6), gridspec_kw={"height_ratios": [1.2, 1.25], "hspace": 0.55})
    fig.subplots_adjust(left=0.25, right=0.97, top=0.94, bottom=0.09)
    axw = TEXTWIDTH * (0.97 - 0.25)
    CH = 0.061                                      # approx. inches per character at 8 pt

    def seg_label(ax, left, width_pct, y, txt, color, above):
        seg_in = axw * width_pct / 100
        cx_ = left + width_pct / 2
        head, _, tail = txt.rpartition(" ")
        if len(txt) * CH + 0.08 < seg_in:
            ax.text(cx_, y, txt, ha="center", va="center", fontsize=8, color=color)
        elif max(len(head), len(tail)) * CH + 0.08 < seg_in:
            ax.text(cx_, y, f"{head}\n{tail}", ha="center", va="center", fontsize=8, color=color, linespacing=1.1)
        else:
            yy = y + (0.43 if above else -0.43)
            ha = "right" if cx_ > 80 else "left" if cx_ < 20 else "center"
            ax.annotate(txt, xy=(cx_, y + (0.26 if above else -0.26)), xytext=(cx_, yy),
                        ha=ha, va="bottom" if above else "top", fontsize=8, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    # (a) composition of each stage (shares; counts in labels)
    rows = [(f"Seed corpus ({g1['input_rows']:,})", [("train", tab["train_seed"]), ("val", tab["val"]), ("test", tab["test"])]),
            (f"Training corpus ({g2['output_rows']:,})", [("seed", orig["seed"]), ("C synthetic", orig["synthetic"]), ("D noise", orig["noise"])]),
            (f"Test set ({g4['composition']['total_rows']})", [("seed", tab["test"]), ("adversarial, scored", adv - excl), ("excluded", excl)])]
    shades = ["#4d4d4d", "#9a9a9a", "#d0d0d0"]
    for i, (name, segs) in enumerate(rows):
        y = len(rows) - 1 - i
        tot = sum(n for _, n in segs)
        left = 0.0
        for j, (lab, n) in enumerate(segs):
            p = 100 * n / tot
            a1.barh(y, p, left=left, height=0.62, color=shades[j], edgecolor="white", linewidth=1.2)
            seg_label(a1, left, p, y, f"{lab} {n:,}", "white" if j == 0 else INK, above=(j == 1))
            left += p
    a1.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    a1.set_xlim(0, 100); a1.set_ylim(-0.75, len(rows) - 0.3)
    a1.set_xlabel("share of the stage's rows (%)")
    a1.tick_params(axis="y", length=0)
    panel(a1, "a", "Composition of each stage")
    # (b) qtype mix: training share vs scored-test share, per question type
    train_q = g2["rows_by_qtype"]
    ntr = g2["output_rows"]
    qs = [("numeric_runtime", "numeric"), ("factual", "factual"), ("abstain", "abstain"), ("clarify", "clarify")]
    for i, (q, name) in enumerate(qs):
        y = len(qs) - 1 - i
        ptr = 100 * train_q.get(q, 0) / ntr
        pte = 100 * sc[q] / n_sc
        a2.barh(y + 0.19, ptr, height=0.34, color="#c4c4c4", edgecolor="white")
        a2.barh(y - 0.19, pte, height=0.34, color=ACCENT if q == "clarify" else "#4d4d4d", edgecolor="white")
        a2.text(ptr + 1, y + 0.19, f"{ptr:.1f}%", va="center", fontsize=8,
                fontweight="bold" if q == "clarify" else "normal", color=INK)
        a2.text(pte + 1, y - 0.19, f"{pte:.1f}%", va="center", fontsize=8,
                fontweight="bold" if q == "clarify" else "normal", color=INK)
    a2.set_yticks(range(len(qs)), [n for _, n in qs][::-1])
    a2.set_xlim(0, 100); a2.set_ylim(-0.6, len(qs) - 0.4)
    a2.set_xlabel("share of rows (%)")
    a2.tick_params(axis="y", length=0)
    vgrid(a2)
    a2.legend(handles=[Patch(color="#c4c4c4", label=f"training corpus ({ntr:,} rows)"),
                       Patch(color="#4d4d4d", label=f"test set, scored ({n_sc} rows)")], loc="lower right")
    panel(a2, "b", "Question-type mix")
    save(fig, "f02_data_composition", "f2_data_composition",
         ["reports/gate1_split.json", "reports/gate2_augment.json", "reports/gate4_hardened.json", "reports/stats/entry_gate.json"],
         f"{g1['input_rows']} seed rows; {ntr} training rows; {g4['composition']['total_rows']} test rows ({n_sc} scored)",
         f"Clarification is {100 * train_q.get('clarify', 0) / ntr:.1f}% of the training corpus but {100 * sc['clarify'] / n_sc:.1f}% of the scored test set; abstention is {100 * train_q['abstain'] / ntr:.1f}% of training.")


# =========================================================================== F3
def f3_tokenizer_cost() -> None:
    t = json.loads((GEN / "tokenizer_cost.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(GEN / "tokenizer_cost_pairs.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.3), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.19, right=0.98, top=0.88, bottom=0.2, wspace=0.42)
    ys = np.arange(len(ORDER))[::-1]
    rng = np.random.default_rng(42)
    for y, k in zip(ys, ORDER):
        d = t["models"][k]
        r = pairs[pairs.model == k]["ratio"].to_numpy()
        a1.plot(r, y + rng.uniform(-0.18, 0.18, len(r)), "o", ms=2.2, mfc=tint(COLOR[k], 0.45), mec="none", zorder=2)
        a1.plot([d["ci95_low"], d["ci95_high"]], [y, y], color=INK, lw=1.6, zorder=3, solid_capstyle="butt")
        a1.plot(d["median_ratio_bn_over_en"], y, "o", ms=6.5, mfc=COLOR[k], mec="white", mew=0.8, zorder=4)
        a1.text(d["ci95_high"] + 0.12, y + 0.22, f"{d['median_ratio_bn_over_en']:.2f}", fontsize=8, va="bottom", color=INK)
    a1.axvline(1.0, color=MUTED, lw=0.7)
    a1.text(1.05, ys[-1] - 0.55, "parity", fontsize=8, color=MUTED, va="center")
    a1.set_yticks(ys, [LABEL[k] for k in ORDER]); a1.tick_params(axis="y", length=0)
    a1.set_xlabel("bn tokens / en tokens per question pair")
    a1.set_ylim(-0.8, len(ORDER) - 0.4)
    vgrid(a1)
    panel(a1, "a", f"Ratio per pair (n = {t['n_pairs']}), median, 95% CI")
    a2.set_ylim(-0.8, len(ORDER) - 0.4)
    for y, k in zip(ys, ORDER):
        d = t["models"][k]
        a2.plot([d["tokens_per_char_en"], d["tokens_per_char_bn"]], [y, y], color=tint(COLOR[k], 0.6), lw=2, zorder=2)
        a2.plot(d["tokens_per_char_en"], y, "o", ms=6, mfc=tint(COLOR[k]), mec=COLOR[k], mew=1, zorder=3)
        a2.plot(d["tokens_per_char_bn"], y, "o", ms=6, mfc=COLOR[k], mec="white", mew=0.8, zorder=3)
    a2.set_yticks(ys, [SHORT[k] for k in ORDER]); a2.tick_params(axis="y", length=0)
    a2.set_xlabel("tokens per character")
    a2.set_xlim(0, 1.45)
    vgrid(a2)
    a2.legend(handles=[Line2D([], [], marker="o", ls="", mfc=tint(INK2), mec=INK2, ms=6, label="en"),
                       Line2D([], [], marker="o", ls="", mfc=INK2, mec="white", ms=6, label="bn")],
              loc="lower right", handletextpad=0.2)
    panel(a2, "b", "Tokens per character")
    meds = {k: t["models"][k]["median_ratio_bn_over_en"] for k in ORDER}
    save(fig, "f03_tokenizer_cost", "f3_tokenizer_cost",
         ["thesis/generated/tokenizer_cost.json", "thesis/generated/tokenizer_cost_pairs.csv"],
         f"{t['n_pairs']} bn/en question pairs x 4 tokenizers",
         f"A Bengali question costs {min(meds.values()):.2f}–{max(meds.values()):.2f} times as many tokens as its English twin (median), highest for Llama-3.1-8B and Phi-3.5-mini.")


# =========================================================================== F4
def f4_training_dynamics() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 3.9), sharex=False, gridspec_kw={"hspace": 0.55, "wspace": 0.28})
    srcs = []
    for ax, k, letter in zip(axes.flat, ORDER, "abcd"):
        spe = None
        for s in SEEDS:
            f = ROOT / "reports" / "s5" / f"{k}_seed{s}.json"
            srcs.append(rel(f))
            m = json.loads(f.read_text(encoding="utf-8"))
            tr = sorted((int(a), b) for a, b in m["train_loss_by_step"].items())
            ev = sorted((int(a), b) for a, b in m["eval_loss_by_step"].items())
            ax.plot([a for a, _ in tr], [b for _, b in tr], color=tint(COLOR[k], 0.55), lw=1.0, zorder=2)
            ax.plot([a for a, _ in ev], [b for _, b in ev], color=COLOR[k], lw=1.4, marker=SEED_MARK[s], ms=2.6,
                    mfc="white", mec=COLOR[k], mew=0.7, zorder=3)
            spe = m["global_step"] / 3
        for e in (1, 2):
            ax.axvline(spe * e, color="#cfcfcf", lw=0.8, zorder=1)
        ax.set_ylim(0, None)
        ax.set_xlabel("optimiser step"); ax.set_ylabel("loss")
        hgrid(ax)
        panel(ax, letter, LABEL[k])
    h = [Line2D([], [], color=tint(INK2, 0.55), lw=1.0, label="training loss (3 seeds)"),
         Line2D([], [], color=INK2, lw=1.4, marker="o", ms=2.6, mfc="white", mec=INK2, label="validation loss (3 seeds; markers o s ^ = 42, 1337, 2024)")]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.2)
    rises = []
    for k in ORDER:
        for s in SEEDS:
            ev = json.loads((ROOT / "reports" / "s5" / f"{k}_seed{s}.json").read_text(encoding="utf-8"))["eval_loss_by_step"]
            last = ev[max(ev, key=int)]
            rises.append(last - min(ev.values()))
    save(fig, "f04_training_dynamics", "f4_training_dynamics", sorted(set(srcs)),
         "12 training manifests (train loss every 10 steps, validation loss every 25 steps)",
         f"Training loss falls smoothly in all twelve runs; validation loss reaches its minimum before the last step in every run and ends at most {max(rises):.3f} above it. Validation loss is not comparable across models because tokenisation differs.")


# =========================================================================== F5
def f5_knowledge() -> None:
    sm = summary()
    slots = [("slot_kc_F1", "Kc"), ("slot_et0_mm_F1", "ET0"),
             ("slot_rainfall_mm_F1", "Effective rain"), ("slot_area_sqm_F1", "Field area")]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, 2.35), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.36, wspace=0.18)
    ys = np.arange(len(ORDER))[::-1]
    for ax, (col, name), letter in zip(axes, slots, "abcd"):
        for y, k in zip(ys, ORDER):
            zs = sm.loc[f"{k}_zeroshot", col]
            ft = [sm.loc[f"{k}_seed{s}", col] for s in SEEDS]
            mu = float(np.mean(ft))
            ax.plot([zs, mu], [y, y], color=tint(COLOR[k], 0.5), lw=1.6, zorder=1)
            ax.plot(zs, y, "o", ms=6, mfc=tint(COLOR[k]), mec=COLOR[k], mew=1, zorder=3)
            ax.plot(mu, y, "o", ms=6.5, mfc=COLOR[k], mec="white", mew=0.8, zorder=4)
            seed_dots(ax, y, ft, horizontal=True, spread=0.18)
        ax.set_xlim(-0.05, 1.05); ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel("slot F1")
        vgrid(ax)
        panel(ax, letter, name)
    axes[0].set_yticks(ys, [SHORT[k] for k in ORDER]); axes[0].tick_params(axis="y", length=0)
    variant_legend(axes[0], bars=False, with_cl=False, loc="upper center", bbox_to_anchor=(2.3, -0.36), ncol=3)
    kz = [sm.loc[f"{k}_zeroshot", "slot_kc_F1"] for k in ORDER]
    kf = [np.mean([sm.loc[f"{k}_seed{s}", "slot_kc_F1"] for s in SEEDS]) for k in ORDER]
    save(fig, "f05_knowledge", "f5_knowledge", [rel(S6)], "numeric_runtime rows; per-slot scorable rows as in summary_by_arm.csv",
         f"No base model states a correct Kc (F1 {max(kz):.3f}); after fine-tuning Kc F1 is {min(kf):.3f}–{max(kf):.3f} and ET0 / effective rainfall are near ceiling.")


# =========================================================================== F6
def f6_numeric_accuracy() -> None:
    sm = summary()
    em = pd.read_csv(guard(ROOT / "reports/stats/exact_match_denominators.csv")).set_index("arm")
    val = lambda arm, col: 100 * em.loc[arm, "exact_lt0p005_single_rate"] if col == "strict" else sm.loc[arm, col]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.85), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.9, bottom=0.34, wspace=0.08)
    tiers = (("strict", "Exact (< 0.005 mm)", "a"), ("exact_numeric_0p1", "Within ±0.1 mm (headline)", "b"),
             ("exact_numeric_1p0", "Within ±1.0 mm", "c"))
    for ax, (col, name, letter) in zip(axes, tiers):
        for i, k in enumerate(ORDER):
            zs = val(f"{k}_zeroshot", col)
            ft = [val(f"{k}_seed{s}", col) for s in SEEDS]
            mu = float(np.mean(ft))
            cl = val(f"{k}_seed42_clarify", col)
            ax.bar(i - 0.27, zs, width=0.25, color=tint(COLOR[k]), edgecolor=COLOR[k], linewidth=0.8)
            ax.bar(i, mu, width=0.25, color=COLOR[k], edgecolor=COLOR[k])
            ax.bar(i + 0.27, cl, width=0.25, color="white", edgecolor=COLOR[k], hatch="////", linewidth=0.8)
            seed_dots(ax, i, ft, spread=0.06)
            ax.text(i, max(ft) + 2.5, f"{mu:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(ORDER)), [SHORT[k] for k in ORDER]); ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 108)
        hgrid(ax)
        panel(ax, letter, name)
    axes[0].set_ylabel("single-crop rows correct (%)")
    variant_legend(axes[1], with_cl=True, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, columnspacing=1.5,
                   handletextpad=0.4)
    n = int(sm.loc["llama31_8b_seed42", "n_single"])
    z = [sm.loc[f"{k}_zeroshot", 'exact_numeric_0p1'] for k in ORDER]
    f = [np.mean([sm.loc[f"{k}_seed{s}", 'exact_numeric_0p1'] for s in SEEDS]) for k in ORDER]
    st = [np.mean([val(f"{k}_seed{s}", "strict") for s in SEEDS]) for k in ORDER]
    save(fig, "f06_numeric_accuracy", "f6_numeric_accuracy", [rel(S6), "reports/stats/exact_match_denominators.csv"],
         f"{n} single-crop numeric rows per arm; three tolerance tiers",
         f"At ±0.1 mm accuracy rises from {min(z):.1f}–{max(z):.1f}% to {min(f):.1f}–{max(f):.1f}%; exact agreement (< 0.005 mm) is much lower ({min(st):.1f}–{max(st):.1f}%) and ±1.0 mm much higher.")


# =========================================================================== F7
def f7_abstain_vs_clarify() -> None:
    sm = summary()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.85))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.3, wspace=0.18)
    n_ab = int(sm.loc["llama31_8b_seed42", "n_abstain"]); n_cl = int(sm.loc["llama31_8b_seed42", "n_clarify"])
    for ax, col, name, letter in ((a1, "abstain_accuracy", f"Abstains, out-of-scope rows (n = {n_ab})", "a"),
                                  (a2, "exact_clarify", f"Asks for missing parameter (n = {n_cl})", "b")):
        for i, k in enumerate(ORDER):
            zs = sm.loc[f"{k}_zeroshot", col]
            ft = [sm.loc[f"{k}_seed{s}", col] for s in SEEDS]
            mu = float(np.mean(ft))
            cl = sm.loc[f"{k}_seed42_clarify", col]
            ax.bar(i - 0.27, zs, width=0.25, color=tint(COLOR[k]), edgecolor=COLOR[k], linewidth=0.8)
            ax.bar(i, mu, width=0.25, color=COLOR[k], edgecolor=COLOR[k])
            ax.bar(i + 0.27, cl, width=0.25, color="white", edgecolor=COLOR[k], hatch="////", linewidth=0.8)
            seed_dots(ax, i, ft, spread=0.06)
            if col == "exact_clarify":
                ax.text(i + 0.27, cl + 2.5, f"{cl:.1f}", ha="center", va="bottom", fontsize=8)
                ax.text(i, 3.2, "0", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(range(len(ORDER)), [SHORT[k] for k in ORDER]); ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 108)
        hgrid(ax)
        panel(ax, letter, name)
    a1.set_ylabel("rows handled correctly (%)")
    variant_legend(a1, with_cl=True, loc="upper center", bbox_to_anchor=(1.1, -0.12), ncol=2, columnspacing=1.5)
    cls = [sm.loc[f"{k}_seed42_clarify", 'exact_clarify'] for k in ORDER]
    save(fig, "f07_abstain_vs_clarify", "f7_abstain_vs_clarify", [rel(S6)],
         f"{n_ab} abstain rows and {n_cl} clarify rows per arm",
         f"Fine-tuning teaches abstention (100% in all 12 runs) but not clarification (0% in all 12 runs); adding clarify examples lifts clarification to {min(cls):.1f}–{max(cls):.1f}%.")


# =========================================================================== F8
def f8_confusion() -> None:
    cm = json.loads((ROOT / "reports/s6/confusion_matrices.json").read_text(encoding="utf-8"))
    arms = [("llama31_8b_zeroshot", "Zero-shot"), ("llama31_8b_seed42", "Fine-tuned"),
            ("llama31_8b_seed42_clarify", "Clarify-trained")]
    gold = ["answer", "clarify", "abstain"]; pred = ["answer", "clarify", "abstain", "empty"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.45))
    fig.subplots_adjust(left=0.175, right=0.99, top=0.88, bottom=0.27, wspace=0.08)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ink", ["#ffffff", "#3b3b3b"])
    for ax, (arm, name), letter in zip(axes, arms, "abc"):
        o = cm[arm]["overall"]
        M = np.array([[o[g][p] for p in pred] for g in gold], dtype=float)
        R = M / M.sum(axis=1, keepdims=True)
        ax.imshow(R, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(gold)):
            for j in range(len(pred)):
                ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=8,
                        color="white" if R[i, j] > 0.55 else INK)
        ax.add_patch(Rectangle((0 - 0.5, 1 - 0.5), 1, 1, fill=False, ec=ACCENT, lw=2.2, zorder=5))
        ax.set_xticks(range(len(pred)), pred, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(gold)), [f"{g} ({int(M[i].sum())})" for i, g in enumerate(gold)] if letter == "a" else [])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("predicted response class")
        panel(ax, letter, name)
    axes[0].set_ylabel("gold class (rows)")
    o = [cm[a]["overall"]["clarify"] for a, _ in arms]
    save(fig, "f08_confusion", "f8_confusion", ["reports/s6/confusion_matrices.json"],
         "362 scorable rows per arm (Llama-3.1-8B)",
         f"The clarify→answer cell holds {o[0]['answer']}, {o[1]['answer']} and {o[2]['answer']} of 68 clarify rows for zero-shot, fine-tuned and clarify-trained Llama.")


# =========================================================================== F9
def f9_mitigation() -> None:
    sm = summary()
    det = load_jsonl(ROOT / "reports/s6/predictions/llama31_8b_seed42_clarify_with_detector.jsonl")
    cl = [r for r in det if r.get("qtype") == "clarify" and str(r.get("gold", "")).strip()]
    learned = sum(r.get("predicted_qtype") == "clarify" for r in cl)
    caught = sum(r.get("predicted_qtype") != "clarify" and bool(r.get("detector_flag")) for r in cl)
    resid = 100 * (len(cl) - learned - caught) / len(cl)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7), gridspec_kw={"width_ratios": [2.0, 1.25]})
    fig.subplots_adjust(left=0.19, right=0.985, top=0.9, bottom=0.3, wspace=0.45)
    ys = np.arange(len(ORDER))[::-1]
    for y, k in zip(ys, ORDER):
        ft = [sm.loc[f"{k}_seed{s}", "invented_value_rate"] for s in SEEDS]
        c = sm.loc[f"{k}_seed42_clarify", "invented_value_rate"]
        a1.annotate("", xy=(c + 1.2, y), xytext=(np.mean(ft) - 1.5, y),
                    arrowprops=dict(arrowstyle="-|>", color=tint(COLOR[k], 0.7), lw=1.4, mutation_scale=8))
        a1.plot(np.mean(ft), y, "o", ms=6.5, mfc=COLOR[k], mec="white", mew=0.8, zorder=4)
        seed_dots(a1, y, ft, horizontal=True, spread=0.18)
        a1.plot(c, y, "D", ms=5.5, mfc=COLOR[k], mec=INK, mew=0.8, zorder=4)
        a1.text(c - 2.5, y + 0.3, f"{c:.1f}", ha="right", va="bottom", fontsize=8)
    a1.set_yticks(ys, [LABEL[k] for k in ORDER]); a1.tick_params(axis="y", length=0)
    a1.set_xlim(35, 104); a1.set_xlabel("clarify rows answered with an invented value (%)")
    a1.set_ylim(-0.7, len(ORDER) - 0.3)
    vgrid(a1)
    a1.legend(handles=[Line2D([], [], marker="o", ls="", mfc=INK2, mec="white", ms=6, label="fine-tuned (mean of 3 seeds)"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=0.6, ms=3.2, label="individual seed"),
                       Line2D([], [], marker="D", ls="", mfc=INK2, mec=INK, ms=5, label="clarify-trained (seed 42)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8)
    panel(a1, "a", "Clarify training (n = 68 rows per model)")
    c_ll = sm.loc["llama31_8b_seed42_clarify", "invented_value_rate"]
    a2.bar([0], [c_ll], width=0.55, color="white", edgecolor=COLOR["llama31_8b"], hatch="////", linewidth=0.8)
    a2.bar([1], [resid], width=0.55, color="#ededed", edgecolor=INK2, linewidth=0.8)
    a2.text(0, c_ll + 2, f"{c_ll:.1f}", ha="center", va="bottom", fontsize=8)
    a2.text(1, resid + 2, f"{resid:.1f}", ha="center", va="bottom", fontsize=8)
    a2.set_xticks([0, 1], ["clarify-\ntrained", "+ oracle-\ngated rules"]); a2.tick_params(axis="x", length=0)
    a2.set_ylim(0, 100); a2.set_ylabel("invented value (%)")
    a2.annotate("oracle-gated upper\nbound, in-sample", xy=(1, resid + 9), xytext=(0.62, 86), ha="center", va="center",
                fontsize=8, color=INK2, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MUTED, lw=0.6),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    hgrid(a2)
    panel(a2, "b", "Llama-3.1-8B only")
    save(fig, "f09_mitigation", "f9_mitigation",
         [rel(S6), "reports/s6/predictions/llama31_8b_seed42_clarify_with_detector.jsonl"],
         "68 clarify rows per arm",
         f"Clarify training cuts invented values from 100% to {min(sm.loc[f'{k}_seed42_clarify', 'invented_value_rate'] for k in ORDER):.1f}–{max(sm.loc[f'{k}_seed42_clarify', 'invented_value_rate'] for k in ORDER):.1f}%; the detector's {resid:.1f}% is an oracle-gated, in-sample upper bound for Llama only.")


# =========================================================================== F10
def f10_output_form() -> None:
    gt = pd.read_csv(GEN / "gen_tokens.csv")
    cl = gt[gt.qtype == "clarify"]
    cap = 512
    fig, axes = plt.subplots(4, 1, figsize=(TEXTWIDTH, 5.2), sharex=True)
    fig.subplots_adjust(left=0.3, right=0.98, top=0.96, bottom=0.08, hspace=0.42)
    rng = np.random.default_rng(42)
    rows = [("zero-shot", None), ("fine-tuned 42", 42), ("fine-tuned 1337", 1337), ("fine-tuned 2024", 2024)]
    for ax, k, letter in zip(axes, ORDER, "abcd"):
        labels = []
        for yi, (lab, s) in enumerate(rows):
            y = len(rows) - 1 - yi
            arm = f"{k}_zeroshot" if s is None else f"{k}_seed{s}"
            v = cl[cl.arm == arm]["gen_tokens"].to_numpy()
            j = rng.uniform(-0.3, 0.3, len(v))
            if s is None:
                ax.plot(v, y + j, "o", ms=2.8, mfc=tint(COLOR[k], 0.35), mec=COLOR[k], mew=0.5, ls="", zorder=3)
            else:
                ax.plot(v, y + j, "o", ms=2.6, mfc=COLOR[k], mec="white", mew=0.3, ls="", zorder=3)
            labels.append(f"{lab} ({100 * (v >= cap - 1).mean():.1f}% at cap)")
        ax.axvline(cap, color=INK, lw=0.9, zorder=2)
        ax.text(cap, 1.02, "512-token cap", transform=ax.get_xaxis_transform(), ha="right", va="bottom", fontsize=8, color=INK)
        ax.set_yticks(range(len(rows)), labels[::-1])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(-10, 530); ax.set_ylim(-0.6, len(rows) - 0.4)
        vgrid(ax)
        panel(ax, letter, LABEL[k])
    axes[-1].set_xlabel("generated tokens (stored generation re-encoded with the model's tokenizer)")
    z = [100 * (cl[cl.arm == f"{k}_zeroshot"]["gen_tokens"] >= cap - 1).mean() for k in ORDER]
    save(fig, "f10_output_form", "f10_output_form", ["thesis/generated/gen_tokens.csv"],
         "68 clarify rows x (zero-shot + 3 fine-tuned seeds) x 4 models",
         f"On incomplete queries {min(z):.1f}–{max(z):.1f}% of zero-shot responses run to the token cap; no fine-tuned response does, in any of the 12 runs.")


# =========================================================================== F11
def f11_language() -> None:
    a = pd.read_csv(guard(ROOT / "reports/stats/results_automatic.csv"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.75), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.9, bottom=0.3, wspace=0.08)
    ys = np.arange(len(ORDER))[::-1]
    offs = {42: 0.22, 1337: 0.0, 2024: -0.22}
    n_sig = {}
    for ax, fam, name, letter, k_ in ((a1, "A6_num_unpaired", "Numeric, ±0.1 mm (73 vs 73 rows)", "a", 100.0),
                                      (a2, "A6_fac", "Factual chrF++ (57 paired facts)", "b", 1.0)):
        for y, k in zip(ys, ORDER):
            for s in SEEDS:
                r = a[(a.test_id == f"{fam}_{k}") & (a.seed == s)].iloc[0]
                d, lo, hi = r["diff"] * k_, r["diff_ci_low"] * k_, r["diff_ci_high"] * k_
                sig = r["p_holm"] < 0.05
                n_sig[(fam, k, s)] = sig
                ax.plot([lo, hi], [y + offs[s]] * 2, color=COLOR[k], lw=1.2, zorder=2, solid_capstyle="butt")
                ax.plot(d, y + offs[s], marker=SEED_MARK[s], ms=5.5, ls="",
                        mfc=COLOR[k] if sig else "white", mec=COLOR[k], mew=1.0, zorder=3)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlabel("English minus Bengali" + (" (percentage points)" if k_ == 100 else " (chrF++ points)"))
        vgrid(ax)
        panel(ax, letter, name)
    a1.set_yticks(ys, [LABEL[k] for k in ORDER]); a1.tick_params(axis="y", length=0)
    a1.set_ylim(-0.6, len(ORDER) - 0.4)
    h = [Line2D([], [], marker=SEED_MARK[s], ls="", mfc=INK2, mec=INK2, ms=5, label=f"seed {s}") for s in SEEDS]
    h += [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=5, label="significant (Holm)"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5, label="not significant")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.55, 0.0), columnspacing=1.2, handletextpad=0.3)
    fac_sig = [k for k in ORDER if n_sig[("A6_fac", k, 42)]]
    num = a[a.test_id.str.startswith("A6_num_unpaired")]
    spans = int(((num.diff_ci_low <= 0) & (num.diff_ci_high >= 0)).sum())
    num_sig = int((num.p_holm < 0.05).sum())
    save(fig, "f11_language", "f11_language", ["reports/stats/results_automatic.csv"],
         "A6 tests, fine-tuned arms, 4 models x 3 seeds; numeric: 146 rows (73 per language, unpaired Fisher); factual: 57 fact_id pairs (Wilcoxon)",
         f"{num_sig} of {len(num)} numeric gaps are significant and {spans} of {len(num)} numeric CIs span zero; the factual gap favours English and is significant at seed 42 in {len(fac_sig)} of 4 models.")


# =========================================================================== F12
H_ARMS = [("llama31_8b_zeroshot", "Llama zero-shot"), ("llama31_8b_seed42", "Llama fine-tuned"),
          ("llama31_8b_seed42_clarify", "Llama clarify-trained"), ("mistral_nemo_seed42", "Mistral fine-tuned")]
H_Q = [("numeric_runtime", "numeric"), ("factual", "factual"), ("clarify", "clarify"), ("abstain", "abstain")]


def f12_human_by_qtype() -> None:
    hs = pd.read_csv(guard(ROOT / "reports/stats/human_arm_summary.csv"))
    hs = hs[hs["sample"] == "test"]
    M = np.zeros((4, 4)); LO = np.zeros((4, 4)); HI = np.zeros((4, 4)); N = np.zeros((4, 4), dtype=int)
    for i, (a, _) in enumerate(H_ARMS):
        for j, (q, _) in enumerate(H_Q):
            r = hs[(hs.arm == a) & (hs.measure == f"farmer_yes/{q}")].iloc[0]
            M[i, j], LO[i, j], HI[i, j], N[i, j] = 100 * r.estimate, 100 * r.ci_low, 100 * r.ci_high, int(r.n)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.55))
    fig.subplots_adjust(left=0.24, right=0.99, top=0.86, bottom=0.12)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ink", ["#ffffff", "#2f2f2f"])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.0f}%\n[{LO[i, j]:.0f}, {HI[i, j]:.0f}]\nn = {N[i, j]}", ha="center", va="center",
                    fontsize=8, color="white" if M[i, j] > 55 else INK, linespacing=1.15)
    ax.add_patch(Rectangle((2 - 0.5, 1 - 0.5), 1, 1, fill=False, ec=ACCENT, lw=2.2))
    ax.add_patch(Rectangle((2 - 0.5, 3 - 0.5), 1, 1, fill=False, ec=ACCENT, lw=2.2))
    ax.set_xticks(range(4), [q for _, q in H_Q]); ax.set_yticks(range(4), [a for _, a in H_ARMS])
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("share of outputs a majority would give to a farmer [95% CI]")
    save(fig, "f12_human_by_qtype", "f12_human_by_qtype", ["reports/stats/human_arm_summary.csv"],
         f"primary human sample: {int(hs[hs.measure == 'farmer_yes'].n.iloc[0])} test-set questions x 4 arms; farmer decision by majority vote on shared outputs",
         f"Annotators would give none of the standard fine-tuned models' answers to incomplete queries to a farmer (0 of {N[1, 2]} per arm); the clarify-trained arm reaches {M[2, 2]:.0f}% on those questions.")


# =========================================================================== F17 (replaces F13)
def f17_acceptance_by_distance() -> None:
    b8 = pd.read_csv(guard(ROOT / "reports/stats/b8_error_bins.csv"))
    short = {"exact": "exact", "<=0.1": "0.005–0.1", "<=1.0": "0.1–1.0", ">1.0": "> 1.0"}
    labels = {b: f"{short[b]}\nn = {n}" for b, n in zip(b8.bin_abs_err_mm, b8.n)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.27, wspace=0.28)
    x = np.arange(len(b8))
    y = 100 * b8.farmer_yes_rate.to_numpy()
    hi = 100 * b8.farmer_yes_hi.to_numpy()
    a1.bar(x, y, width=0.6, color="#5a5a5a", edgecolor="#5a5a5a")
    a1.errorbar(x, y, yerr=[y - 100 * b8.farmer_yes_lo, hi - y], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
    for i, v in enumerate(y):
        a1.text(i + 0.08, hi[i] + 2, f"{v:.0f}%", ha="left", va="bottom", fontsize=8)
    a1.set_ylim(0, 112); a1.set_ylabel("would give to a farmer (%)")
    a2.errorbar(x, b8.mean_rating, yerr=[b8.mean_rating - b8.mean_rating_lo, b8.mean_rating_hi - b8.mean_rating],
                fmt="o", ms=6, mfc="#5a5a5a", mec="white", mew=0.8, ecolor=INK, elinewidth=0.9, capsize=2.5)
    a2.set_ylim(0.8, 5.2); a2.set_yticks([1, 2, 3, 4, 5]); a2.set_ylabel("mean rating (1–5)")
    for ax, letter, name in ((a1, "a", "Farmer decision"), (a2, "b", "Rating")):
        ax.set_xticks(x, [labels[b] for b in b8.bin_abs_err_mm]); ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(b8) - 0.4)
        hgrid(ax)
        panel(ax, letter, name)
    fig.text(0.5, 0.01, "distance of the stated net depth from the gold depth (mm; exact = below 0.005 mm)",
             ha="center", va="bottom", fontsize=8)
    save(fig, "f17_acceptance_by_distance", "f17_acceptance_by_distance", ["reports/stats/b8_error_bins.csv"],
         f"{int(b8.n.sum())} fine-tuned numeric outputs (three fine-tuned arms x test-set numeric questions; multi-crop outputs excluded); 95% bootstrap CIs",
         f"Annotators accepted {y[0]:.0f}% of exact answers but only {y[1]:.0f}% of answers 0.005–0.1 mm away and {y[2]:.0f}% of answers 0.1–1.0 mm away: with the reference visible, they judged agreement with it.")




# =========================================================================== F14
def f14_annotators() -> None:
    import krippendorff
    r = pd.read_csv(guard(ROOT / "reports/human_eval/stage7_ratings_merged.csv"))
    ov = r[r.is_overlap == "yes"]
    M = ov.pivot_table(index="annotator", columns="judgment_id", values="rating_1_5", aggfunc="first")
    alpha = lambda mm: krippendorff.alpha(reliability_data=mm.values.astype(float), level_of_measurement="ordinal")
    a_all = alpha(M)
    loo = {a: alpha(M.drop(index=a)) for a in M.index}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6), gridspec_kw={"width_ratios": [1.9, 1]})
    fig.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.36, wspace=0.55)
    ramp = ["#f0f0f0", "#c8c8c8", "#9a9a9a", "#666666", "#333333"]
    anns = list(M.index)
    ys = np.arange(len(anns))[::-1]
    for y, a in zip(ys, anns):
        cnt = M.loc[a].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        left = 0
        for rt, c in cnt.items():
            a1.barh(y, c, left=left, color=ramp[rt - 1], edgecolor="white", linewidth=1.0, height=0.6)
            if c >= 7:
                a1.text(left + c / 2, y, str(c), ha="center", va="center", fontsize=8, color="white" if rt >= 4 else INK)
            left += c
    a1.set_yticks(ys, anns); a1.tick_params(axis="y", length=0)
    a1.set_xlim(0, M.shape[1]); a1.set_xlabel(f"ratings on the {M.shape[1]} shared outputs")
    fig.legend(handles=[Patch(facecolor=ramp[i], edgecolor=MUTED, linewidth=0.5, label=f"rating {i + 1}") for i in range(5)],
               ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.01), handlelength=1.2, columnspacing=1.2, fontsize=8)
    panel(a1, "a", "Rating distribution per annotator")
    for y, a in zip(ys, anns):
        a2.plot(loo[a], y, "o", ms=6, mfc=INK2, mec="white", mew=0.8)
        a2.text(loo[a] + 0.005, y + 0.15, f"{loo[a]:.3f}", fontsize=8, va="bottom")
    a2.axvline(a_all, color=INK, lw=0.9)
    a2.text(a_all + 0.002, -0.95, f"all five: {a_all:.3f}", ha="left", va="center", fontsize=8)
    a2.set_yticks(ys, [f"without {a}" for a in anns]); a2.tick_params(axis="y", length=0)
    a2.set_xlim(0.66, 0.80); a2.set_ylim(-1.3, len(anns) - 0.4)
    a2.set_xlabel("ordinal Krippendorff's α")
    vgrid(a2)
    panel(a2, "b", "Leave-one-out α")
    up = [a for a in anns if loo[a] > a_all]
    save(fig, "f14_annotators", "f14_annotators", ["reports/human_eval/stage7_ratings_merged.csv"],
         f"{M.shape[1]} shared outputs x {M.shape[0]} annotators",
         f"Ordinal alpha is {a_all:.3f} with all five annotators; leaving out {', '.join(up)} raises it (to {', '.join(f'{loo[a]:.3f}' for a in up)}), leaving out any other lowers it.")


# =========================================================================== F15
EXAMPLES = [("en", "4dc57fae-04ec-4a65-9ecf-f48c9a6e6bed"), ("bn", "4dd0a4db-6ec6-49a3-9a57-be097f0db9e2")]


def _excerpt_head(text: str, n_par: int) -> str:
    pars = [p for p in text.split("\n") if p.strip()]
    return "\n".join(pars[:n_par])




# =========================================================================== F15 (LaTeX box)
BN_RUN = re.compile(r"[ঀ-৿‌‍]+(?:[  ।॥]+[ঀ-৿‌‍]+)*[।॥]?")
TEX_ESC = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
           "^": r"\^{}", "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}"}


def tex_escape(s: str) -> str:
    return "".join(TEX_ESC.get(c, c) for c in s)


def tex_mixed(s: str) -> str:
    """Escape LaTeX specials and wrap every Bengali run in \\bn{} (XeLaTeX shapes it with Noto Serif Bengali)."""
    out, pos = [], 0
    for m in BN_RUN.finditer(s):
        out.append(tex_escape(s[pos:m.start()]))
        out.append(r"\bn{" + tex_escape(m.group(0)) + "}")
        pos = m.end()
    out.append(tex_escape(s[pos:]))
    return "".join(out)


def tex_par(s: str) -> str:
    return r"\par ".join(tex_mixed(p.strip()) for p in s.split("\n") if p.strip())


def f15_examples_tex() -> None:
    verify_test_sha()
    test = {r["row_id"]: r for r in load_jsonl(TEST_PATH)}
    gt = pd.read_csv(GEN / "gen_tokens.csv")
    preds = {v: {r["row_id"]: r for r in load_jsonl(ROOT / "reports/s6/predictions" / f"phi35_mini_{v}.jsonl")}
             for v in ("zeroshot", "seed42", "seed42_clarify")}
    cols = [("zeroshot", "Zero-shot"), ("seed42", "Fine-tuned, seed 42"), ("seed42_clarify", "Clarify-trained, seed 42")]
    lines = ["% AUTO-GENERATED by scripts/thesis/figures.py (f15_examples_tex). Model text is verbatim from",
             "% reports/s6/predictions/phi35_mini_*.jsonl; [...] marks an omitted span. Do not edit by hand.",
             r"\tcbset{exouter/.style={enhanced, breakable, colback=white, colframe=black!45, boxrule=0.5pt, arc=1.5pt,",
             r"  left=3pt, right=3pt, top=2pt, bottom=2pt, fonttitle=\footnotesize\bfseries, coltitle=black, colbacktitle=black!7},",
             r"  exinner/.style={colback=white, colframe=black!35, boxrule=0.4pt, arc=1pt, left=2pt, right=2pt, top=1pt,",
             r"  bottom=1pt, fontupper=\scriptsize, fonttitle=\scriptsize\bfseries, coltitle=black, colbacktitle=black!4}}"]
    for lang, rid in EXAMPLES:
        q = test[rid]["question"]
        lines.append(r"\begin{tcolorbox}[exouter, title={" + ("English" if lang == "en" else "Bengali")
                     + r" question, Phi-3.5-mini, test row \texttt{" + rid + "}}]")
        lines.append(r"{\small\textbf{Question.} " + tex_par(q) + r"}")
        lines.append(r"\par\smallskip")
        lines.append(r"\begin{tcbraster}[raster columns=3, raster equal height, raster column skip=1.5mm]")
        for v, title in cols:
            g = preds[v][rid]["generation"]
            ntok = int(gt[(gt.arm == f"phi35_mini_{v}") & (gt.row_id == rid)]["gen_tokens"].iloc[0])
            if v == "zeroshot":
                body = (_excerpt_head(g, 1) + "\n[…]\n" + g[-150:].lstrip()) if lang == "en" else (_excerpt_head(g, 2) + "\n[…]")
            else:
                body = g.strip()
            cap = " (at the cap)" if ntok >= 511 else ""
            extra = r", colback=orange!6" if v == "seed42_clarify" else (r", colframe=black!80, boxrule=0.7pt" if v == "seed42" else "")
            lines.append(r"\begin{tcolorbox}[exinner" + extra + ", title={" + title + r"\hfill " + f"{ntok} tokens{cap}" + "}]")
            lines.append(tex_par(body))
            lines.append(r"\end{tcolorbox}")
        lines.append(r"\end{tcbraster}")
        lines.append(r"\end{tcolorbox}")
    (GEN / "f15_examples.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    MANIFEST.append({"id": "f15_examples", "function": "f15_examples_tex", "files": ["thesis/generated/f15_examples.tex"],
                     "sources": ["data/hardened/test_hardened.jsonl", "reports/s6/predictions/phi35_mini_zeroshot.jsonl",
                                 "reports/s6/predictions/phi35_mini_seed42.jsonl", "reports/s6/predictions/phi35_mini_seed42_clarify.jsonl",
                                 "thesis/generated/gen_tokens.csv"],
                     "rows_used": f"2 test rows ({EXAMPLES[0][1]}, {EXAMPLES[1][1]}) x 3 Phi-3.5-mini variants; text verbatim, [...] marks omitted spans; Bengali set by XeLaTeX",
                     "takeaway": "The same unknown-unit question three ways: the base model rambles to the token cap, the fine-tuned model silently converts the unit and returns a volume, the clarify-trained model asks for the field size."})


# =========================================================================== F16
F16_FAMILIES = [("A1", "A1 numeric: fine-tuned − zero-shot"), ("A2", "A2 numeric: model contrasts"),
                ("A3a", "A3a clarify exact: CL − FT"), ("A3b", "A3b invented value: CL − FT"),
                ("A4", "A4 numeric: CL − FT"), ("A6_num", "A6 numeric: English − Bengali"),
                ("A7", "A7 confident error: FT − zero-shot")]


def f16_forest() -> None:
    a = pd.read_csv(guard(ROOT / "reports/stats/results_automatic.csv"))
    p = a[(a.role == "PRIMARY") & (a.seed == 42) & a["diff"].notna()].copy()
    rows = []
    for fam, flab in F16_FAMILIES:
        sub = p[p.test_id.str.startswith(fam + "_")] if fam != "A6_num" else p[p.test_id.str.startswith("A6_num_unpaired")]
        for _, r in sub.iterrows():
            k = next((kk for kk in ORDER if r.test_id.endswith(kk)), None)
            lab = SHORT[k] if k else {"A2_llama_vs_mistral": "Mistral − Llama", "A2_tier": "large − small tier"}[r.test_id]
            rows.append((fam, flab, lab, k, 100 * r["diff"], 100 * r.diff_ci_low, 100 * r.diff_ci_high, r.p_holm < 0.05))
    fac = p[p.test_id.str.startswith("A6_fac")]
    n = len(rows) + len(F16_FAMILIES)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 6.6), gridspec_kw={"width_ratios": [3.2, 1.25]})
    fig.subplots_adjust(left=0.36, right=0.97, top=0.965, bottom=0.13, wspace=0.22)
    y = n
    ticks, tlabels, fam_y = [], [], []
    last_fam = None
    for fam, flab, lab, k, d, lo, hi, sig in rows:
        if fam != last_fam:
            y -= 1
            fam_y.append((y + 0.15, flab))
            last_fam = fam
        y -= 1
        c = COLOR[k] if k else INK2
        a1.plot([lo, hi], [y, y], color=c, lw=1.3, solid_capstyle="butt", zorder=2)
        a1.plot(d, y, "o", ms=5.5, mfc=c if sig else "white", mec=c, mew=1.0, zorder=3)
        ticks.append(y); tlabels.append(lab)
    from matplotlib.transforms import blended_transform_factory
    for yy, flab in fam_y:
        a1.text(0.01, yy, flab, transform=blended_transform_factory(fig.transFigure, a1.transData), ha="left",
                va="center", fontsize=8, fontweight="bold")
    a1.set_yticks(ticks, tlabels); a1.tick_params(axis="y", length=0)
    a1.axvline(0, color=INK, lw=0.8)
    a1.set_xlabel("difference, percentage points (95% CI)")
    a1.set_ylim(y - 0.8, n - 0.2)
    vgrid(a1)
    panel(a1, "a", "Rates (seed 42)")
    ys = np.arange(len(fac))[::-1]
    for yy, (_, r) in zip(ys, fac.iterrows()):
        k = next(kk for kk in ORDER if r.test_id.endswith(kk))
        sig = r.p_holm < 0.05
        a2.plot([r.diff_ci_low, r.diff_ci_high], [yy, yy], color=COLOR[k], lw=1.3, solid_capstyle="butt")
        a2.plot(r["diff"], yy, "o", ms=5.5, mfc=COLOR[k] if sig else "white", mec=COLOR[k], mew=1.0)
    a2.set_yticks(ys, [SHORT[next(kk for kk in ORDER if t.endswith(kk))] for t in fac.test_id]); a2.tick_params(axis="y", length=0)
    a2.axvline(0, color=INK, lw=0.8)
    a2.set_ylim(-0.8, len(fac) - 0.2)
    a2.set_xlabel("chrF++ points")
    vgrid(a2)
    panel(a2, "b", "A6 factual, en − bn")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=5.5, label="significant after Holm (p < .05)"),
                        Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5.5, label="not significant")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8)
    nsig = sum(r[-1] for r in rows) + int((fac.p_holm < 0.05).sum())
    ns = [t for t, ph in zip(p.test_id, p.p_holm) if ph >= 0.05]
    save(fig, "f16_forest", "f16_forest", ["reports/stats/results_automatic.csv"],
         f"{len(rows) + len(fac)} PRIMARY automatic tests with an effect difference at seed 42 (the Friedman omnibus has no difference and is omitted)",
         f"{nsig} of {len(rows) + len(fac)} plotted PRIMARY effects are significant after Holm; not significant: {', '.join(ns)}.")


# =========================================================================== F18
def f18_response_language() -> None:
    rl = pd.read_csv(guard(ROOT / "reports/stats/response_language.csv"))
    rl = rl[rl.subset == "all_bn_questions"].set_index("arm")
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.2))
    fig.subplots_adjust(left=0.2, right=0.985, top=0.93, bottom=0.34)
    ys = np.arange(len(ORDER))[::-1]
    for y, k in zip(ys, ORDER):
        for arm, off, kind in ((f"{k}_zeroshot", 0.22, "zs"), (f"{k}_seed42", 0.0, "ft"), (f"{k}_seed42_clarify", -0.22, "cl")):
            r = rl.loc[arm]
            v, lo, hi = 100 * r.share_answered_in_bengali, 100 * r.ci_low, 100 * r.ci_high
            ax.plot([lo, hi], [y + off] * 2, color=COLOR[k], lw=1.2, solid_capstyle="butt", zorder=2)
            if kind == "zs":
                ax.plot(v, y + off, "o", ms=6, mfc=tint(COLOR[k]), mec=COLOR[k], mew=1.0, zorder=3)
            elif kind == "ft":
                ax.plot(v, y + off, "o", ms=6, mfc=COLOR[k], mec="white", mew=0.8, zorder=3)
            else:
                ax.plot(v, y + off, "D", ms=5, mfc=COLOR[k], mec=INK, mew=0.8, zorder=3)
            if kind == "zs" and v < 95:
                ax.text(v, y + off + 0.12, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_yticks(ys, [LABEL[k] for k in ORDER]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 104); ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.set_xlabel("Bengali questions answered in Bengali script (%), 95% bootstrap CI")
    vgrid(ax)
    h = [Line2D([], [], marker="o", ls="", mfc=tint("#7a7a7a"), mec="#7a7a7a", ms=6, label="zero-shot"),
         Line2D([], [], marker="o", ls="", mfc="#7a7a7a", mec="white", ms=6, label="fine-tuned (seed 42)"),
         Line2D([], [], marker="D", ls="", mfc="#7a7a7a", mec=INK, ms=5, label="clarify-trained (seed 42)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.4, -0.3), ncol=3)
    tuned = [100 * rl.loc[a].share_answered_in_bengali for a in rl.index if "seed42" in a]
    save(fig, "f18_response_language", "f18_response_language", ["reports/stats/response_language.csv"],
         f"{int(rl.iloc[0].n_bn_questions)} Bengali test questions per arm (exploratory, added after Part A)",
         f"Zero-shot Mistral-NeMo answered only {100 * rl.loc['mistral_nemo_zeroshot'].share_answered_in_bengali:.1f}% of Bengali questions in Bengali; every fine-tuned and clarify-trained arm answered at least {min(tuned):.1f}% in Bengali.")


# ===========================================================================
def main() -> None:
    verify_test_sha()
    setup_style()
    for f in (f1_pipeline_tikz, f2_data_composition, f3_tokenizer_cost, f4_training_dynamics, f5_knowledge,
              f6_numeric_accuracy, f7_abstain_vs_clarify, f8_confusion, f9_mitigation, f10_output_form,
              f11_language, f12_human_by_qtype, f14_annotators, f15_examples_tex, f16_forest,
              f17_acceptance_by_distance, f18_response_language):
        f()
        print("ok", f.__name__)
    (GEN / "figures_manifest.json").write_text(json.dumps(MANIFEST, ensure_ascii=False, indent=1) + "\n",
                                                encoding="utf-8", newline="\n")
    print(f"{len(MANIFEST)} figures -> {rel(FIG)}")


if __name__ == "__main__":
    main()
