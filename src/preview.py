"""
Render predicted disparity next to ground truth, so the model can be LOOKED at.

    python src/preview.py --ckpt runs/sceneflow_cont/best.pt --root data/driving

EPE is a mean over millions of pixels and it hides shape. A model that has
learned a smooth left-to-right gradient and nothing else scores far better than
chance on a road scene, because road scenes largely ARE a smooth gradient. That
is close to what the model this project replaces was doing: its cost volume
compared every hypothesis at zero disparity, so nothing it produced came from
stereo matching, and the number it reported still looked like a stereo number.

This writes a PNG per frame row: left image, prediction, ground truth, error.
Prediction and ground truth share a colour scale within a row, so a prediction
that is merely SMOOTH rather than correct is visible immediately.

Colour: a single-hue sequential ramp for disparity (light = near, dark = far)
and a separate red ramp for error clipped at 3 px, the D1 threshold -- so red
is exactly the pixels that count as wrong. Rainbow ramps like turbo are
conventional for disparity and are avoided here: they invent visual edges at
hue boundaries that do not exist in the data, which is the one artefact this
image is meant to detect.

Only numpy and Pillow -- no matplotlib. Native extensions are a liability on
this machine (see the ml_dtypes note in the README).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo  # noqa: E402
from data import MEAN, STD  # noqa: E402
from train import build_datasets  # noqa: E402

# single hue, light -> dark: magnitude, not identity
DISP_RAMP = np.array([
    [0.937, 0.957, 0.984], [0.776, 0.855, 0.937], [0.549, 0.702, 0.878],
    [0.322, 0.541, 0.784], [0.161, 0.376, 0.643], [0.043, 0.176, 0.361],
])
# error is a status quantity: absent -> critical
ERR_RAMP = np.array([
    [0.965, 0.961, 0.953], [0.965, 0.855, 0.784], [0.937, 0.616, 0.510],
    [0.851, 0.302, 0.239], [0.545, 0.086, 0.078],
])
INVALID = np.array([0.851, 0.824, 0.780])      # no ground truth here
PAD, HEADER = 8, 18
COLUMNS = ("left image", "prediction", "ground truth", "|error|, red >= 3 px")


def ramp(t: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """Map t in [0,1] onto a colour ramp by linear interpolation."""
    t = np.clip(t, 0.0, 1.0) * (len(colours) - 1)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(colours) - 1)
    f = (t - lo)[..., None]
    return colours[lo] * (1 - f) + colours[hi] * f


def to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)


def render_row(sample, pred, width):
    """One frame as four side-by-side panels, already colour-mapped."""
    left = sample["left"].numpy().transpose(1, 2, 0) * STD + MEAN
    disp = sample["disp"].numpy()[0]
    mask = sample["mask"].numpy()[0] > 0.5

    # a shared scale, taken from the ground truth, is what makes a merely
    # smooth prediction visible next to a correct one
    # percentiles, not min/max: a handful of near-field pixels at the top of
    # the 192 px range would otherwise compress every other value into the
    # first two steps of the ramp and hide exactly the structure being checked
    if mask.any():
        lo, hi = (float(v) for v in np.percentile(disp[mask], [2.0, 98.0]))
    else:
        lo, hi = 0.0, 1.0
    span = max(hi - lo, 1e-6)

    gt = ramp((disp - lo) / span, DISP_RAMP)
    gt[~mask] = INVALID
    pr = ramp((pred - lo) / span, DISP_RAMP)

    err = np.abs(pred - disp)
    em = ramp(err / 3.0, ERR_RAMP)          # 3 px is the D1 threshold
    em[~mask] = INVALID

    panels = [to_u8(left), to_u8(pr), to_u8(gt), to_u8(em)]
    imgs = [Image.fromarray(p) for p in panels]
    h = round(imgs[0].height * width / imgs[0].width)
    return [im.resize((width, h), Image.BILINEAR) for im in imgs], (lo, hi)


def compose(rows, width, captions):
    ncol = len(COLUMNS)
    h = rows[0][0].height
    W = ncol * width + (ncol + 1) * PAD
    H = HEADER + len(rows) * (h + HEADER) + PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for c, name in enumerate(COLUMNS):
        draw.text((PAD + c * (width + PAD), 4), name, fill=(60, 60, 60))
    y = HEADER
    for row, caption in zip(rows, captions):
        for c, im in enumerate(row):
            canvas.paste(im, (PAD + c * (width + PAD), y))
        draw.text((PAD, y + h + 3), caption, fill=(90, 90, 90))
        y += h + HEADER
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", choices=("sceneflow", "kitti"), default="sceneflow")
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="artifacts/preview.png")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--width", type=int, default=380)
    p.add_argument("--crop-h", type=int, default=256)
    p.add_argument("--crop-w", type=int, default=512)
    p.add_argument("--pass-name", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HailoStereo()
    model.load_state_dict(blob.get("model", blob), strict=True)   # EXP-1
    model.to(device).eval()

    ns = argparse.Namespace(dataset=args.dataset, root=args.root,
                            crop_h=args.crop_h, crop_w=args.crop_w,
                            pass_name=args.pass_name, limit=None, val_limit=args.n)
    _, val = build_datasets(ns)

    rows, captions = [], []
    for i in range(min(args.n, len(val))):
        sample = val[i]
        with torch.no_grad():
            pred = model(sample["left"][None].to(device),
                         sample["right"][None].to(device))
        pred = pred[0, 0].cpu().numpy()

        disp = sample["disp"].numpy()[0]
        mask = sample["mask"].numpy()[0] > 0.5
        err = np.abs(pred - disp)[mask]
        epe = float(err.mean()) if err.size else float("nan")
        d1 = (100.0 * float(((err > 3.0) &
                             (err > 0.05 * np.abs(disp[mask]))).mean())
              if err.size else float("nan"))

        panels, (lo, hi) = render_row(sample, pred, args.width)
        rows.append(panels)
        captions.append(f"frame {i}   EPE {epe:.2f} px   D1 {d1:.1f}%   "
                        f"scale {lo:.0f}-{hi:.0f} px (2-98 pct)")
        print(captions[-1])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compose(rows, args.width, captions).save(out)
    print(f"\nwrote {out}  ({out.stat().st_size / 1e3:.0f} KB)")
    print(f"checkpoint: epoch {blob.get('epoch', '?')}, "
          f"val EPE {blob.get('epe', float('nan')):.3f} px")


if __name__ == "__main__":
    main()
