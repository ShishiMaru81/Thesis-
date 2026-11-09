#!/usr/bin/env python3
"""font_test.py — does matplotlib shape Bengali conjuncts with Noto Sans Bengali?

Renders the same strings with matplotlib and, as a shaped reference, with
Pillow's raqm layout engine (when available). Output images go to
thesis/generated/font_test/ for visual inspection; the verdict is recorded by
hand in BUILD_REPORT.md.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image, ImageDraw, ImageFont, features

ROOT = Path(__file__).resolve().parents[2]
FONT = ROOT / "thesis" / "fonts" / "NotoSansBengali-Regular.ttf"
OUT = ROOT / "thesis" / "generated" / "font_test"
# conjuncts, ra-phala, reph, pre-base vowel signs, split vowels
TEXT = ["দ্রোণ  কানি  গণ্ডা", "সেচ  বৃষ্টি  ক্ষেত", "প্রতি  কার্যকর  মৌসুম"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fm.fontManager.addfont(str(FONT))
    name = fm.FontProperties(fname=str(FONT)).get_name()
    fig, ax = plt.subplots(figsize=(6, 1.8), dpi=200)
    ax.axis("off")
    for i, t in enumerate(TEXT):
        ax.text(0.02, 0.8 - i * 0.33, t, fontfamily=name, fontsize=20, transform=ax.transAxes)
    fig.savefig(OUT / "matplotlib.png")
    plt.close(fig)
    raqm = features.check("raqm")
    img = Image.new("RGB", (1200, 360), "white")
    d = ImageDraw.Draw(img)
    engine = ImageFont.Layout.RAQM if raqm else ImageFont.Layout.BASIC
    f = ImageFont.truetype(str(FONT), 56, layout_engine=engine)
    for i, t in enumerate(TEXT):
        d.text((20, 20 + i * 110), t, font=f, fill="black")
    img.save(OUT / ("pillow_raqm.png" if raqm else "pillow_basic.png"))
    print(f"matplotlib {matplotlib.__version__}; font family '{name}'; Pillow raqm available: {raqm}")


if __name__ == "__main__":
    main()
