#!/usr/bin/env python3
"""
validate_palette.py — Python port of the dataviz skill's validate_palette.js
(categorical checks 2–5), used because Node is not installed on this machine.

Same thresholds, same Machado–Oliveira–Fernandes (2009) severity-1.0 matrices,
same OKLab conversion and Delta E (Euclidean OKLab x100) as the JavaScript
original. Also searches the Okabe–Ito set for the 4-colour model palette.

Usage:
  python validate_palette.py "#0072B2,#D55E00,..." [--pairs all] [--surface #ffffff]
  python validate_palette.py --search-okabe-ito
"""
from __future__ import annotations

import itertools
import math
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}
OKABE_ITO = {"orange": "#E69F00", "sky blue": "#56B4E9", "bluish green": "#009E73", "yellow": "#F0E442",
             "blue": "#0072B2", "vermillion": "#D55E00", "reddish purple": "#CC79A7"}


def hex2srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = math.copysign(abs(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3), 1)
    m = math.copysign(abs(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3), 1)
    s = math.copysign(abs(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3), 1)
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s]


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [max(0.0, min(1.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface="#ffffff", pairs="adjacent"):
    lo, hi = BAND[mode]
    rep, ok = [], True
    off = [(c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi]
    ok &= not off
    rep.append(("Lightness band", "PASS" if not off else "FAIL", off or f"all inside L {lo}-{hi}"))
    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    rep.append(("Chroma floor", "PASS" if not lowc else "FAIL", lowc or f"all >= {CHROMA_FLOOR}"))
    n = len(palette)
    pl = list(itertools.combinations(range(n), 2)) if pairs == "all" else [(i, i + 1) for i in range(n - 1)]
    worst = min(((delta_e(palette[i], palette[j], k), k, palette[i], palette[j]) for k in ("protan", "deutan") for i, j in pl))
    tri = min(delta_e(palette[i], palette[j], "tritan") for i, j in pl)
    st = "PASS" if worst[0] >= CVD_TARGET else "WARN" if worst[0] >= CVD_FLOOR else "FAIL"
    ok &= st != "FAIL"
    rep.append(("CVD separation", st, f"worst {pairs} {worst[2]}<->{worst[3]} dE {worst[0]:.1f} ({worst[1]}); tritan {tri:.1f}"))
    nworst = min((delta_e(palette[i], palette[j]), palette[i], palette[j]) for i, j in pl)
    st = "PASS" if nworst[0] >= NORMAL_FLOOR else "FAIL"
    ok &= st == "PASS"
    rep.append(("Normal-vision floor", st, f"worst {nworst[1]}<->{nworst[2]} dE {nworst[0]:.1f}"))
    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    rep.append(("Contrast vs surface", "WARN" if low else "PASS", low or f"all >= {CONTRAST_MIN}:1"))
    return rep, ok, worst[0], nworst[0]


def search(surface="#ffffff"):
    names = list(OKABE_ITO)
    best = []
    for combo in itertools.combinations(names, 4):
        pal = [OKABE_ITO[c] for c in combo]
        rep, ok, cvd, nv = validate(pal, surface=surface, pairs="all")
        hard_ok = all(r[1] != "FAIL" for r in rep)
        best.append((hard_ok, cvd, nv, combo, rep))
    best.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return best


if __name__ == "__main__":
    if "--search-okabe-ito" in sys.argv:
        for hard_ok, cvd, nv, combo, rep in search()[:6]:
            print(f"{'OK ' if hard_ok else 'BAD'} min all-pairs CVD dE {cvd:5.1f}  normal {nv:5.1f}  {combo}")
            for r in rep:
                print(f"      [{r[1]:4s}] {r[0]:20s} {r[2]}")
    else:
        pal = [c for c in sys.argv[1].split(",") if c.strip()]
        pairs = "all" if "--pairs" in sys.argv and sys.argv[sys.argv.index("--pairs") + 1] == "all" else "adjacent"
        surface = sys.argv[sys.argv.index("--surface") + 1] if "--surface" in sys.argv else "#ffffff"
        rep, ok, _, _ = validate(pal, surface=surface, pairs=pairs)
        for r in rep:
            print(f"  [{r[1]:4s}] {r[0]:20s} {r[2]}")
        print("ALL CHECKS PASS" if ok else "FAILED")
        sys.exit(0 if ok else 1)
