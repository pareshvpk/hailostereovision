"""
Verify a stereo dataset's ingest geometrically, before training on it.

    python src/check_ingest.py --dataset kitti --root data/kitti2015/training
    python src/check_ingest.py --dataset sceneflow --root data/driving

The test: for a rectified pair, the match for left pixel x lies at right pixel
x - d. So warping the right image by the ground-truth disparity should make it
line up with the left image, and the residual |warp(R) - L| should collapse
relative to the unwarped |R - L|. If it does not, the disparity maps do not
describe these images and no amount of training will fix it.

It also sweeps a multiplier over the disparity. A reader that divides by the
wrong constant -- KITTI stores 16-bit PNG at 1/256 px, the converted SceneFlow
maps use 1/32 -- still produces a plausible-looking disparity map, still trains,
and still converges to something. The residual is minimised at the CORRECT
scale, so the sweep locates the true one rather than merely accepting the one
the code happens to use. A sweep whose minimum is not at 1.0 means the reader
is wrong by that factor.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import SceneFlow, Kitti2015, Kitti2012, MEAN, STD  # noqa: E402

SCALES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def residual(left, right, disp, mask, scale=1.0):
    """Mean |warp(R) - L| over pixels the warp can actually reach."""
    h, w = disp.shape
    xs = np.arange(w)[None, :] - disp * scale
    ok = mask & (xs >= 0) & (xs <= w - 1)
    if not ok.any():
        return float("nan"), 0.0
    xi = np.clip(np.round(xs), 0, w - 1).astype(np.int64)
    warped = right[np.arange(h)[:, None], xi]
    return float(np.abs(warped - left)[ok].mean()), float(ok.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("sceneflow", "kitti", "kitti2012"),
                   required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--n", type=int, default=12)
    args = p.parse_args()

    if args.dataset == "kitti":
        ds = Kitti2015(args.root, split="val")
    elif args.dataset == "kitti2012":
        # training=False -> deterministic full-frame crop, the geometry check
        # wants whole frames, not random training crops
        ds = Kitti2012(args.root, training=False)
    else:
        ds = SceneFlow(args.root, training=False)
    n = min(args.n, len(ds))
    print(f"{args.dataset}: {len(ds)} pairs, checking {n}")

    base, warped, cover, dstats, valid = [], [], [], [], []
    per_scale = {s: [] for s in SCALES}
    for i in range(n):
        s = ds[i]
        # undo the normalisation: this compares image intensities, not features
        l = (s["left"].numpy().transpose(1, 2, 0) * STD + MEAN).mean(2)
        r = (s["right"].numpy().transpose(1, 2, 0) * STD + MEAN).mean(2)
        d = s["disp"].numpy()[0]
        m = s["mask"].numpy()[0] > 0.5

        b, _ = residual(l, r, d, m, 0.0)
        base.append(b)
        for sc in SCALES:
            v, cv = residual(l, r, d, m, sc)
            per_scale[sc].append(v)
            if sc == 1.0:
                warped.append(v)
                cover.append(cv)
        valid.append(float(m.mean()))
        if m.any():
            dstats.append((float(d[m].min()), float(d[m].max()),
                           float(np.percentile(d[m], 99))))

    print(f"\nvalidity mask     {100*np.mean(valid):.1f}% of pixels")
    lo = min(x[0] for x in dstats); hi = max(x[1] for x in dstats)
    print(f"disparity range   {lo:.2f} .. {hi:.2f} px "
          f"(99th pct {np.mean([x[2] for x in dstats]):.1f})")
    print(f"warp coverage     {100*np.mean(cover):.1f}% of valid pixels\n")

    print(f"unwarped |R-L|    {np.mean(base):.4f}")
    print(f"warped   |R-L|    {np.mean(warped):.4f}   "
          f"({np.mean(base)/np.mean(warped):.2f}x better)\n")

    print("scale sweep (residual should be minimised at 1.0):")
    means = {sc: float(np.mean(v)) for sc, v in per_scale.items()}
    best = min(means, key=means.get)
    for sc in SCALES:
        mark = "  <-- minimum" if sc == best else ""
        print(f"   x{sc:<6} {means[sc]:.4f}{mark}")

    ok = best == 1.0 and np.mean(base) / np.mean(warped) > 1.3
    print("\n" + ("PASS -- disparity aligns the pair, and at the scale the "
                  "reader uses" if ok else
                  f"FAIL -- residual minimised at x{best}, not x1.0; the "
                  f"disparity reader is off by that factor"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
