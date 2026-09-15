"""
Score a checkpoint on KITTI 2015 under two protocols, and report both.

    python src/eval_kitti.py --ckpt runs/kitti/best.pt --root data/kitti2015/training

Training and the in-loop metric here mask out two regions: ground truth beyond
the model's 192 px range, and the left 192 columns, whose matches lie outside
the right image entirely (CV-2). Excluding them from the LOSS is correct --
they are not a target the architecture can represent, and supervising them
teaches the network to fit its own zero padding.

Excluding them from the reported METRIC is a different claim. The left border is
one of the hardest parts of the frame, so a masked EPE is optimistic relative to
KITTI's official protocol, which scores every pixel with valid ground truth. A
number quoted against a published baseline has to say which protocol it used,
and it is not knowable from the outside whether Hailo's 8.223 px for the
original was computed one way or the other.

So this reports both, plus the gap between them, and nobody has to guess.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo, MAX_DISP  # noqa: E402
from data import Kitti2015  # noqa: E402


def score(pred, disp, mask):
    """EPE and D1-all over the given mask. D1: error > 3 px AND > 5%."""
    err = np.abs(pred - disp)[mask]
    if err.size == 0:
        return float("nan"), float("nan"), 0
    d1 = (err > 3.0) & (err > 0.05 * np.abs(disp[mask]))
    return float(err.mean()), 100.0 * float(d1.mean()), int(err.size)


# The diagnosis (IMPROVEMENT_PLAN.md §1.1) is that error is concentrated at
# large ground-truth disparity, not at the 192 px ceiling. A headline EPE hides
# that: only 1% of KITTI GT is >= 80 px, so a model that badly under-reads there
# still scores ~1.66 px overall. These bins expose it. Signed bias (pred - GT),
# not just magnitude, is what shows the *direction* of the failure -- the shipped
# model reads large disparities as smaller than they are (bias strongly negative).
DISP_BINS = ((0, 40), (40, 80), (80, 100), (100, 120), (120, 160))


def bin_sums(pred, disp, col, lo, hi, border):
    """Summed |resid|, summed signed resid (pred - GT) and pixel count for GT
    disparity in [lo, hi), restricted to the matched region (column >= MAX_DISP)
    or the left border (< MAX_DISP). Sums, not means, so they aggregate across
    frames -- the caller divides by the total count per bin at the end.

    Binning is over every valid-GT pixel (disp > 0, finite), split by column the
    same way training splits loss weight -- so these numbers are directly the
    ones Finding 1 in the handoff quotes. The bins stop at 160 px, below the
    192 px range ceiling, so no in-range pixel is silently excluded by the cap.
    """
    region = (col < MAX_DISP) if border else (col >= MAX_DISP)
    m = (disp >= lo) & (disp < hi) & np.isfinite(disp) & (disp > 0.0) & region
    if not m.any():
        return 0.0, 0.0, 0
    resid = pred[m] - disp[m]
    return float(np.abs(resid).sum()), float(resid.sum()), int(m.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="val", choices=("train", "val"))
    p.add_argument("--bins", action="store_true",
                   help="print the disparity-binned EPE/bias breakdown "
                        "(matched vs border) that Finding 1 is stated in")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HailoStereo()
    model.load_state_dict(blob.get("model", blob), strict=True)   # EXP-1
    model.to(device).eval()
    ds = Kitti2015(args.root, split=args.split)
    print(f"{args.ckpt}: epoch {blob.get('epoch','?')}, "
          f"in-loop val EPE {blob.get('epe', float('nan')):.3f} px")
    print(f"{args.split} split: {len(ds)} pairs\n")

    acc = {"masked": [0.0, 0.0, 0], "official": [0.0, 0.0, 0]}
    # (region, bin) -> [sum|resid|, sum signed resid, count]; filled only with --bins
    binacc = {(reg, b): [0.0, 0.0, 0]
              for reg in ("matched", "border") for b in DISP_BINS}
    for i in range(len(ds)):
        s = ds[i]
        with torch.no_grad():
            pred = model(s["left"][None].to(device),
                         s["right"][None].to(device))[0, 0].cpu().numpy()
        disp = s["disp"].numpy()[0]
        masked = s["mask"].numpy()[0] > 0.5           # training protocol
        official = disp > 0.0                          # every valid GT pixel
        for name, m in (("masked", masked), ("official", official)):
            epe, d1, n = score(pred, disp, m)
            if n:
                acc[name][0] += epe * n
                acc[name][1] += d1 * n
                acc[name][2] += n
        if args.bins:
            col = np.broadcast_to(np.arange(disp.shape[1]), disp.shape)
            for reg, border in (("matched", False), ("border", True)):
                for b in DISP_BINS:
                    a, sg, n = bin_sums(pred, disp, col, b[0], b[1], border)
                    binacc[(reg, b)][0] += a
                    binacc[(reg, b)][1] += sg
                    binacc[(reg, b)][2] += n

    print(f"{'protocol':10s} {'pixels':>12s} {'EPE':>9s} {'D1-all':>9s}")
    out = {}
    for name in ("masked", "official"):
        e, d, n = acc[name]
        out[name] = (e / n, d / n)
        print(f"{name:10s} {n:12,d} {e/n:8.3f}p {d/n:8.2f}%")

    de = out["official"][0] - out["masked"][0]
    print(f"\nthe left {MAX_DISP} columns and out-of-range ground truth cost "
          f"{de:+.3f} px EPE\nand {out['official'][1]-out['masked'][1]:+.2f} "
          f"points of D1 when scored rather than excluded.")
    print("\nreference: Hailo reports 8.223 px float / 10.3 px quantized for the "
          "original\non KITTI, protocol unstated -- compare against the "
          "'official' row, which is\nthe conservative reading.")

    if args.bins:
        # The point of this table (IMPROVEMENT_PLAN.md §1.1): the failure lives
        # at large GT disparity and is a systematic UNDER-read (bias << 0), not a
        # ceiling artifact. Reproduce it for kitti_border/best.pt as a
        # correctness check before trusting the number on any new run.
        print(f"\ndisparity-binned breakdown (over every valid-GT pixel)")
        print(f"{'GT disp':>10s} | {'matched (x>=192)':>28s} | "
              f"{'border (x<192)':>28s}")
        print(f"{'px':>10s} | {'pixels':>9s} {'EPE':>8s} {'bias':>8s} | "
              f"{'pixels':>9s} {'EPE':>8s} {'bias':>8s}")
        print("-" * 74)
        for lo, hi in DISP_BINS:
            cells = []
            for reg in ("matched", "border"):
                a, sg, n = binacc[(reg, (lo, hi))]
                if n:
                    cells.append(f"{n:9,d} {a/n:7.2f}p {sg/n:+7.2f}")
                else:
                    cells.append(f"{0:9d} {'--':>8s} {'--':>8s}")
            print(f"{lo:4d}-{hi:<5d} | {cells[0]} | {cells[1]}")
        print("bias = mean(pred - GT); strongly negative = the model reads large "
              "disparity as too small.")


if __name__ == "__main__":
    main()
