"""Render the static chart PNGs README.md embeds.

No matplotlib: native extensions are a liability under this machine's
Application Control policy (see src/preview.py). Pure numpy + Pillow,
supersampled 3x and downsampled with LANCZOS for anti-aliasing, following
the categorical/sequential palette and mark specs from the dataviz skill
(references/palette.md, marks-and-anatomy.md).
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "report" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

FONT_DIR = Path("C:/Windows/Fonts")
SCALE = 3

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
RED = "#e34948"


def _font(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


class Chart:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w * SCALE, h * SCALE), SURFACE)
        self.d = ImageDraw.Draw(self.img)

    def s(self, v):
        return round(v * SCALE)

    def text(self, xy, s, size, fill=INK, bold=False, anchor="la"):
        self.d.text((self.s(xy[0]), self.s(xy[1])), s, font=_font(self.s(size), bold),
                     fill=fill, anchor=anchor)

    def text_w(self, s, size, bold=False):
        bbox = self.d.textbbox((0, 0), s, font=_font(self.s(size), bold))
        return (bbox[2] - bbox[0]) / SCALE

    def hline(self, x0, x1, y, color=GRID, width=1):
        self.d.line([(self.s(x0), self.s(y)), (self.s(x1), self.s(y))],
                     fill=color, width=max(1, self.s(width)))

    def bar_v(self, cx, base_y, top_y, width, color):
        half = width / 2
        r = min(4, half)
        self.d.rounded_rectangle(
            [self.s(cx - half), self.s(min(top_y, base_y)),
             self.s(cx + half), self.s(max(top_y, base_y))],
            radius=self.s(r), fill=color, corners=(True, True, False, False))

    def bar_h(self, cy, base_x, tip_x, height, color):
        half = height / 2
        r = min(4, half)
        self.d.rounded_rectangle(
            [self.s(min(base_x, tip_x)), self.s(cy - half),
             self.s(max(base_x, tip_x)), self.s(cy + half)],
            radius=self.s(r), fill=color, corners=(False, True, True, False))

    def dot(self, x, y, d, color):
        self.d.ellipse([self.s(x), self.s(y), self.s(x + d), self.s(y + d)], fill=color)

    def save(self, name):
        self.img.resize((self.w, self.h), Image.LANCZOS).save(OUT / name)
        print("wrote", OUT / name)


def y_of(value, vmin, vmax, y_bottom, y_top, log=False):
    if log:
        f = (math.log10(value) - math.log10(vmin)) / (math.log10(vmax) - math.log10(vmin))
    else:
        f = (value - vmin) / (vmax - vmin)
    return y_bottom - f * (y_bottom - y_top)


def legend(c, x, y, entries):
    cx = x
    for color, label in entries:
        c.dot(cx, y, 10, color)
        c.text((cx + 16, y - 2), label, 12, fill=INK_SECONDARY)
        cx += 16 + c.text_w(label, 12) + 24


# ---------------------------------------------------------------------------
# 1. Headline comparison -- three panels, shared legend
# ---------------------------------------------------------------------------
def headline_comparison():
    c = Chart(1200, 540)
    c.text((40, 28), "Model Zoo stereonet vs HailoStereo", 20, bold=True)
    legend(c, 700, 32, [(RED, "Model Zoo stereonet"), (BLUE, "HailoStereo")])

    panels = [
        dict(title="KITTI 2015 EPE (px, lower is better)", x0=40,
             groups=["float", "int8*"],
             red=[8.223, 10.4], blue=[1.248, 1.430],
             vmax=12, ticks=[0, 4, 8, 12]),
        dict(title="Compute (GOPS, lower is better)", x0=460,
             groups=[""], red=[112.07], blue=[28.41],
             vmax=120, ticks=[0, 40, 80, 120]),
        dict(title="HEF size (MB, lower is better)", x0=880,
             groups=[""], red=[8.74], blue=[4.12],
             vmax=10, ticks=[0, 5, 10]),
    ]
    y_bottom, y_top = 470, 110
    panel_w = 320

    for p in panels:
        x0 = p["x0"]
        c.text((x0, 80), p["title"], 13, fill=INK_SECONDARY)
        for t in p["ticks"]:
            ty = y_of(t, 0, p["vmax"], y_bottom, y_top)
            c.hline(x0, x0 + panel_w, ty, color=GRID)
            c.text((x0 - 8, ty), str(t), 11, fill=INK_MUTED, anchor="rm")
        c.hline(x0, x0 + panel_w, y_bottom, color=BASELINE)

        n = len(p["groups"])
        slot = panel_w / n
        for i, g in enumerate(p["groups"]):
            gx = x0 + slot * i + slot / 2
            bw = 30
            gap = 10
            rx = gx - bw / 2 - gap / 2
            bx = gx + bw / 2 + gap / 2
            rv, bv = p["red"][i], p["blue"][i]
            ry = y_of(rv, 0, p["vmax"], y_bottom, y_top)
            by = y_of(bv, 0, p["vmax"], y_bottom, y_top)
            c.bar_v(rx, y_bottom, ry, bw, RED)
            c.bar_v(bx, y_bottom, by, bw, BLUE)
            c.text((rx, ry - 8), f"{rv:g}", 12, bold=True, anchor="mb")
            c.text((bx, by - 8), f"{bv:g}", 12, bold=True, anchor="mb")
            if g:
                c.text((gx, y_bottom + 10), g, 12, fill=INK_MUTED, anchor="ma")

    c.text((40, 500), "*int8: Model Zoo measured on-device; HailoStereo emulated (no board here yet)",
           12, fill=INK_MUTED)
    c.save("headline_comparison.png")


# ---------------------------------------------------------------------------
# 2. Accuracy by distance band -- measured vs geometric floor, log scale
# ---------------------------------------------------------------------------
def distance_band_accuracy():
    c = Chart(900, 540)
    c.text((40, 28), "Accuracy sits at the geometric floor from 5-40 m", 19, bold=True)
    c.text((40, 56), "mean depth error vs the error each band's own pixel noise implies (log scale, metres)",
           13, fill=INK_SECONDARY)
    legend(c, 40, 88, [(BLUE, "measured mean |dZ|"), (BLUE_LIGHT, "geometric floor")])

    bands = ["0-5 m", "5-10 m", "10-20 m", "20-40 m", "40-80 m"]
    measured = [0.45, 0.20, 0.58, 2.23, 6.69]
    floor = [0.31, 0.18, 0.53, 2.09, 8.55]

    y_bottom, y_top = 470, 150
    x0, x1 = 90, 860
    vmin, vmax = 0.1, 10
    for t, label in [(0.1, "0.1 m"), (1, "1 m"), (10, "10 m")]:
        ty = y_of(t, vmin, vmax, y_bottom, y_top, log=True)
        c.hline(x0, x1, ty, color=GRID)
        c.text((x0 - 10, ty), label, 12, fill=INK_MUTED, anchor="rm")
    c.hline(x0, x1, y_bottom, color=BASELINE)

    n = len(bands)
    slot = (x1 - x0) / n
    for i, band in enumerate(bands):
        gx = x0 + slot * i + slot / 2
        bw = 26
        gap = 8
        mx = gx - bw / 2 - gap / 2
        fx = gx + bw / 2 + gap / 2
        my = y_of(measured[i], vmin, vmax, y_bottom, y_top, log=True)
        fy = y_of(floor[i], vmin, vmax, y_bottom, y_top, log=True)
        c.bar_v(mx, y_bottom, my, bw, BLUE)
        c.bar_v(fx, y_bottom, fy, bw, BLUE_LIGHT)
        c.text((mx, my - 8), f"{measured[i]:.2f}", 11, bold=True, anchor="mb")
        c.text((fx, fy - 8), f"{floor[i]:.2f}", 11, fill=INK_SECONDARY, anchor="mb")
        c.text((gx, y_bottom + 10), band, 12, fill=INK_MUTED, anchor="ma")

    c.save("distance_band_accuracy.png")


# ---------------------------------------------------------------------------
# 3. Border supervision fix -- single series, log scale
# ---------------------------------------------------------------------------
def border_fix():
    c = Chart(800, 520)
    c.text((40, 28), "13% of the frame carried 84% of the error", 19, bold=True)
    c.text((40, 56), "mean EPE by region, px (log scale) -- before/after weak border supervision",
           13, fill=INK_SECONDARY)

    labels = ["Trained region", "Left border, unsupervised", "Left border, weakly supervised"]
    values = [1.41, 47.75, 2.95]

    y_bottom, y_top = 460, 130
    x0, x1 = 90, 760
    vmin, vmax = 1, 100
    for t, label in [(1, "1 px"), (10, "10 px"), (100, "100 px")]:
        ty = y_of(t, vmin, vmax, y_bottom, y_top, log=True)
        c.hline(x0, x1, ty, color=GRID)
        c.text((x0 - 10, ty), label, 12, fill=INK_MUTED, anchor="rm")
    c.hline(x0, x1, y_bottom, color=BASELINE)

    n = len(labels)
    slot = (x1 - x0) / n
    for i, label in enumerate(labels):
        gx = x0 + slot * i + slot / 2
        by = y_of(values[i], vmin, vmax, y_bottom, y_top, log=True)
        c.bar_v(gx, y_bottom, by, 60, BLUE)
        c.text((gx, by - 8), f"{values[i]:.2f} px", 13, bold=True, anchor="mb")
        c.text((gx, y_bottom + 10), label, 12, fill=INK_MUTED, anchor="ma")

    c.save("border_fix.png")


# ---------------------------------------------------------------------------
# 4. Quantization sensitivity -- horizontal bars
# ---------------------------------------------------------------------------
def quantization_sensitivity():
    c = Chart(800, 420)
    c.text((40, 28), "One layer causes most of the identifiable int8 loss", 18, bold=True)
    c.text((40, 55), "per-layer sensitivity, delta EPE px -- simulated single-layer int8",
           13, fill=INK_SECONDARY)

    labels = ["soft_argmin.index", "features.stem4.0", "refine2.stem.0",
              "all other layers (each <= 0.003 px)"]
    values = [0.0242, 0.0072, 0.0067, 0.003]

    x0, x1 = 300, 740
    vmin, vmax = 0, 0.03
    y_top, y_bottom = 100, 360
    n = len(labels)
    row_h = (y_bottom - y_top) / n

    for t in [0, 0.01, 0.02, 0.03]:
        tx = x0 + (t - vmin) / (vmax - vmin) * (x1 - x0)
        c.d.line([(c.s(tx), c.s(y_top - 10)), (c.s(tx), c.s(y_bottom))], fill=GRID, width=max(1, c.s(1)))
        c.text((tx, y_bottom + 10), f"{t:.2f}", 11, fill=INK_MUTED, anchor="ma")
    c.hline(x0, x0, y_top - 10, color=BASELINE)

    for i, (label, v) in enumerate(zip(labels, values)):
        cy = y_top + row_h * i + row_h / 2
        c.text((x0 - 14, cy), label, 12, fill=INK_SECONDARY, anchor="rm")
        tip = x0 + (v - vmin) / (vmax - vmin) * (x1 - x0)
        c.bar_h(cy, x0, tip, 22, BLUE)
        c.text((tip + 8, cy), f"{v:.4f}", 12, bold=True, anchor="lm")

    c.save("quantization_sensitivity.png")


if __name__ == "__main__":
    headline_comparison()
    distance_band_accuracy()
    border_fix()
    quantization_sensitivity()
