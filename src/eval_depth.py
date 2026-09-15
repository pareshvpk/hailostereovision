"""
Score a checkpoint in METRES, banded by range.

    python src/eval_depth.py --ckpt runs/kitti_border/best.pt --root data/kitti2015/training

The network predicts disparity in pixels, not distance. Distance is a
reciprocal of it, Z = f x B / d, and that reciprocal is the whole reason a
single headline EPE says almost nothing about usable range:

    dZ = Z^2 / (f x B) . dd

With KITTI's f x B = 386.5 px.m, one pixel of disparity error is 6 cm at 5 m
and 6.5 m at 50 m -- a factor of 100 across the frame, from the same pixel.
A model quoted at "1.464 px EPE" is therefore excellent at short range and
progressively useless at long range, and the pixel number hides which is which.

So this converts both the prediction and the ground truth to metres with the
same constants and reports the error where it is actually read: per range band.

Alongside the measured error each band carries its GEOMETRIC FLOOR --
Zbar^2/(f.B) x EPE_band, the metre error that band's own pixel error implies to
first order. Measured error near the floor means the distance error is stereo
geometry doing what stereo geometry does, and no amount of training removes it;
measured error well above the floor means the model is losing something extra in
that band. Those are different problems with different fixes, and the table
separates them.

Calibration: f and B are ASSUMPTIONS here, not measurements. KITTI's calib
files are not in this checkout -- data_scene_flow.zip ships no calib/ entry --
so the nominal 715.7 px / 0.54 m are used, the same pair src/demo_server.py
defaults to. What that does and does not invalidate is printed with the results.

Only numpy, torch and Pillow, matching the rest of src/.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo  # noqa: E402
from data import Kitti2015  # noqa: E402

# KITTI 2015 nominal rectified intrinsics -- the same values demo_server.py
# offers in its toolbar. Not read from calib files; see the module docstring.
KITTI_FOCAL = 715.7          # px
KITTI_BASELINE = 0.54        # m

# Range bands, in metres. Cut on GROUND-TRUTH depth so a frame's pixels land in
# the band they really belong to rather than the band the model thinks.
BANDS = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 80.0))

# The prediction is clamped to [0, MAX_DISP] and the sky region is untrained, so
# disparity can arrive at or near zero. Floor it rather than divide by it.
EPS_DISP = 1e-3

DELTA = 1.25                 # the standard depth-accuracy threshold


def depth_from_disp(disp: np.ndarray, fb: float) -> np.ndarray:
    """Z = f x B / d, with d floored so the reciprocal cannot blow up."""
    return fb / np.maximum(disp, EPS_DISP)


class Band:
    """Accumulates one band's pixels across frames.

    Absolute errors are kept rather than summed so the median is exact; at
    ~4M pixels over 40 frames this is tens of MB, which is cheaper than the
    approximation a streaming quantile would cost.
    """

    def __init__(self):
        self.abs_err = []        # |dZ|, metres
        self.rel_err = []        # |dZ| / Z_gt
        self.disp_err = []       # |dd|, pixels
        self.disp_bias = []      # signed dd, pixels -- direction matters
        self.z_bias = []         # signed dZ, metres
        self.z_gt = []           # Z_gt, metres
        self.delta_ok = 0
        self.n = 0

    def add(self, z_pred, z_gt, disp_err):
        """disp_err is SIGNED (pred - gt); magnitudes are taken here."""
        if z_gt.size == 0:
            return
        d = np.abs(z_pred - z_gt)
        self.abs_err.append(d.astype(np.float32))
        self.rel_err.append((d / z_gt).astype(np.float32))
        self.disp_err.append(np.abs(disp_err).astype(np.float32))
        self.disp_bias.append(disp_err.astype(np.float32))
        self.z_bias.append((z_pred - z_gt).astype(np.float32))
        self.z_gt.append(z_gt.astype(np.float32))
        ratio = np.maximum(z_pred / z_gt, z_gt / np.maximum(z_pred, EPS_DISP))
        self.delta_ok += int((ratio < DELTA).sum())
        self.n += int(z_gt.size)

    def summary(self, fb: float) -> dict | None:
        if self.n == 0:
            return None
        abs_err = np.concatenate(self.abs_err)
        rel_err = np.concatenate(self.rel_err)
        disp_err = np.concatenate(self.disp_err)
        disp_bias = np.concatenate(self.disp_bias)
        z_bias = np.concatenate(self.z_bias)
        z_gt = np.concatenate(self.z_gt)
        z_mean = float(z_gt.mean())
        epe = float(disp_err.mean())
        return {
            "n": self.n,
            "mae": float(abs_err.mean()),
            "rmse": float(np.sqrt((abs_err.astype(np.float64) ** 2).mean())),
            "median": float(np.median(abs_err)),
            "absrel": 100.0 * float(rel_err.mean()),
            "delta": 100.0 * self.delta_ok / self.n,
            "epe": epe,
            "dbias": float(disp_bias.mean()),
            "zbias": float(z_bias.mean()),
            # first-order propagation of the band's own pixel error, evaluated
            # at the band's mean depth: dZ = Z^2/(f.B) . dd
            "floor": z_mean * z_mean / fb * epe,
            "z_mean": z_mean,
        }


def print_table(title: str, bands: dict, overall, fb: float) -> None:
    print(f"\n{title}")
    print(f"{'range':>12s} {'pixels':>11s} {'mean |dZ|':>10s} {'RMSE':>8s} "
          f"{'median':>8s} {'AbsRel':>8s} {'d<1.25':>8s} {'EPE':>8s} {'floor':>9s} "
          f"{'bias dd':>9s} {'bias dZ':>9s}")
    rows = [(f"{lo:.0f}-{hi:.0f} m", bands[(lo, hi)].summary(fb)) for lo, hi in BANDS]
    rows.append(("all <= 80 m", overall.summary(fb)))
    for label, b in rows:
        if b is None:
            print(f"{label:>12s} {0:11,d}   (no ground truth in this band)")
            continue
        print(f"{label:>12s} {b['n']:11,d} {b['mae']:9.2f}m {b['rmse']:7.2f}m "
              f"{b['median']:7.2f}m {b['absrel']:7.2f}% {b['delta']:7.2f}% "
              f"{b['epe']:7.3f}p {b['floor']:8.2f}m {b['dbias']:+8.2f}p "
              f"{b['zbias']:+8.2f}m")


def selftest() -> bool:
    """Geometry only -- no data, no weights. Catches a flipped reciprocal or a
    focal/baseline swap, which produce plausible-looking tables."""
    fb = KITTI_FOCAL * KITTI_BASELINE
    ok = True

    z = depth_from_disp(np.full((4, 4), 10.0, dtype=np.float32), fb)
    want = KITTI_FOCAL * KITTI_BASELINE / 10.0        # 38.6478 m
    if not np.allclose(z, want, atol=1e-3):
        print(f"  FAIL  10 px -> {z.flat[0]:.4f} m, expected {want:.4f} m")
        ok = False
    else:
        print(f"  ok    10 px -> {z.flat[0]:.3f} m")

    # halving disparity must double distance
    if not np.isclose(depth_from_disp(np.float32(5.0), fb),
                      2.0 * depth_from_disp(np.float32(10.0), fb), atol=1e-4):
        print("  FAIL  halving disparity did not double distance")
        ok = False
    else:
        print("  ok    halving disparity doubles distance")

    # the first-order floor must agree with a finite difference
    for d in (5.0, 20.0, 80.0):
        z0 = fb / d
        analytic = z0 * z0 / fb * 1.0                  # 1 px of error
        finite = abs(fb / (d - 0.5) - fb / (d + 0.5))
        if abs(analytic - finite) > 0.02 * max(analytic, 1e-6):
            print(f"  FAIL  floor at {d:.0f} px: {analytic:.4f} vs {finite:.4f} m")
            ok = False
        else:
            print(f"  ok    1 px at {d:5.1f} px disparity ({z0:6.2f} m) "
                  f"-> {analytic:7.3f} m")

    # a zero prediction must not produce inf
    if not np.isfinite(depth_from_disp(np.float32(0.0), fb)):
        print("  FAIL  zero disparity produced a non-finite depth")
        ok = False
    else:
        print("  ok    zero disparity is finite")

    print(f"\n{'selftest passed' if ok else 'SELFTEST FAILED'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--root")
    p.add_argument("--split", default="val", choices=("train", "val"))
    p.add_argument("--focal", type=float, default=KITTI_FOCAL,
                   help="rectified focal length in pixels")
    p.add_argument("--baseline", type=float, default=KITTI_BASELINE,
                   help="stereo baseline in metres")
    p.add_argument("--max-depth", type=float, default=80.0,
                   help="ground truth beyond this is dropped (KITTI convention)")
    p.add_argument("--out", default=None,
                   help="optional PNG: left | predicted m | ground truth m | error")
    p.add_argument("--n-preview", type=int, default=4)
    p.add_argument("--selftest", action="store_true",
                   help="run the geometry checks and exit")
    args = p.parse_args()

    print("geometry selftest")
    if not selftest():
        return 1
    if args.selftest:
        return 0
    if not args.ckpt or not args.root:
        p.error("--ckpt and --root are required unless --selftest is given")

    fb = args.focal * args.baseline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HailoStereo()
    model.load_state_dict(blob.get("model", blob), strict=True)   # EXP-1
    model.to(device).eval()
    ds = Kitti2015(args.root, split=args.split)

    print(f"\n{args.ckpt}: epoch {blob.get('epoch','?')}, "
          f"in-loop val EPE {blob.get('epe', float('nan')):.3f} px")
    print(f"{args.split} split: {len(ds)} pairs on {device}")
    print(f"f x B = {args.focal:.1f} px x {args.baseline:.3f} m "
          f"= {fb:.2f} px.m   ->  {args.max_depth:.0f} m is "
          f"{fb / args.max_depth:.2f} px of disparity")

    protocols = ("masked", "official")
    bands = {k: {b: Band() for b in BANDS} for k in protocols}
    overall = {k: Band() for k in protocols}
    # the unbanded, undepth-limited disparity EPE -- the cross-check against
    # eval_kitti.py. Restricting to <= max_depth drops the far tail, so this
    # has to be accumulated separately or the two numbers cannot be compared.
    crosscheck = {k: [0.0, 0] for k in protocols}
    frames = []

    for i in range(len(ds)):
        s = ds[i]
        with torch.no_grad():
            pred = model(s["left"][None].to(device),
                         s["right"][None].to(device))[0, 0].cpu().numpy()
        disp = s["disp"].numpy()[0]
        valid = {"masked": s["mask"].numpy()[0] > 0.5,   # matchable region
                 "official": disp > 0.0}                 # every valid GT pixel

        z_pred_full = depth_from_disp(pred, fb)
        z_gt_full = depth_from_disp(disp, fb)
        derr_full = pred - disp          # SIGNED; Band takes |.| itself

        for k in protocols:
            v = valid[k]
            if v.any():
                crosscheck[k][0] += float(np.abs(derr_full[v]).sum())
                crosscheck[k][1] += int(v.sum())
            # in-range for the depth tables
            v = v & (disp > fb / args.max_depth)
            z_pred, z_gt, derr = z_pred_full[v], z_gt_full[v], derr_full[v]
            overall[k].add(z_pred, z_gt, derr)
            for lo, hi in BANDS:
                sel = (z_gt >= lo) & (z_gt < hi)
                bands[k][(lo, hi)].add(z_pred[sel], z_gt[sel], derr[sel])

        if args.out and len(frames) < args.n_preview:
            frames.append((s, z_pred_full, z_gt_full, valid["official"]))

    for k in protocols:
        title = ("official protocol -- every pixel with valid ground truth"
                 if k == "official" else
                 "masked protocol -- matchable region only (left 192 columns "
                 "and out-of-range GT excluded)")
        print_table(title, bands[k], overall[k], fb)

    print("\ncross-check against src/eval_kitti.py (disparity, no depth limit)")
    for k in protocols:
        tot, n = crosscheck[k]
        print(f"  {k:9s} EPE {tot / n:.3f} px over {n:,d} pixels")
    print("  these must match eval_kitti.py exactly -- same model, same data, "
          "same masks.\n  If they do not, the depth path corrupted something "
          "and the tables above are void.")

    print(f"""
reading the table
  mean |dZ| / RMSE / median   distance error in metres
  AbsRel                      mean |dZ| / Z_gt -- error as a fraction of range
  d<1.25                      share of pixels within 25% of true distance
  EPE                         the band's disparity error, in pixels
  floor                       Zbar^2/(f.B) x EPE -- the metre error that pixel
                              error implies at the band's mean depth, to first
                              order. Measured error at the floor is geometry,
                              not a model defect.
  bias dd                     signed disparity error. Negative = the model
                              under-reads disparity, i.e. reports things
                              FARTHER than they are.
  bias dZ                     signed distance error, metres. Negative = reports
                              things NEARER than they are.

  The floor is symmetric in dd; the model is not. Where |bias dd| approaches
  EPE the band has a systematic direction, and because Z = f.B/d is convex,
  over-reading disparity costs fewer metres than under-reading it by the same
  pixels. That is why a band can measure BELOW its own floor -- the floor is
  an unbiased-error reference, not a bound. Read the two together.

on the calibration
  f and B are nominal, not read from KITTI calib files (none are in this
  checkout). Prediction and ground truth pass through the SAME constants, so
  Z_pred/Z_gt = d_gt/d_pred and the ratio metrics are calibration-free:
  AbsRel and d<1.25 hold exactly even if f x B is wrong.
  mean |dZ|, RMSE, median and floor scale LINEARLY with f x B, and the band
  edges move with it. Re-run with --focal/--baseline for another rig.

known defect
  Above the horizon the KITTI finetune drifts -- LiDAR ground truth never
  reaches the sky, so that region was never supervised, and the model reports
  it at high disparity (near). Neither protocol scores it, because neither has
  ground truth there. Nothing above the horizon in the preview is trustworthy.
""")

    if args.out:
        render_preview(frames, args.out, fb, args.max_depth)
    return 0


# ---------------------------------------------------------------------------
# optional preview
# ---------------------------------------------------------------------------

def render_preview(frames, out, fb: float, max_depth: float) -> None:
    """left | predicted distance | ground-truth distance | relative error.

    Reuses preview.py's ramps: a single hue, light = near. The error panel is
    RELATIVE, clipped at 10%, because a fixed metre clip saturates the far
    field and shows nothing but a red horizon.
    """
    from PIL import Image, ImageDraw
    from preview import ramp, to_u8, DISP_RAMP, ERR_RAMP, INVALID, PAD, HEADER
    from data import MEAN, STD

    cols = ("left image", "predicted distance", "ground truth",
            "relative error, red >= 10%")
    width = 380
    rows, captions = [], []

    for s, z_pred, z_gt, valid in frames:
        left = s["left"].numpy().transpose(1, 2, 0) * STD + MEAN
        v = valid & (z_gt <= max_depth)
        if not v.any():
            continue
        lo, hi = (float(x) for x in np.percentile(z_gt[v], [2.0, 98.0]))
        span = max(hi - lo, 1e-6)

        pr = ramp((z_pred - lo) / span, DISP_RAMP)
        gt = ramp((z_gt - lo) / span, DISP_RAMP)
        gt[~v] = INVALID
        rel = np.abs(z_pred - z_gt) / np.maximum(z_gt, 1e-6)
        em = ramp(rel / 0.10, ERR_RAMP)
        em[~v] = INVALID

        panels = [Image.fromarray(to_u8(x)) for x in (left, pr, gt, em)]
        h = round(panels[0].height * width / panels[0].width)
        rows.append([p.resize((width, h), Image.BILINEAR) for p in panels])
        captions.append(f"median |dZ| {np.median(np.abs(z_pred - z_gt)[v]):.2f} m   "
                        f"AbsRel {100 * rel[v].mean():.1f}%   "
                        f"scale {lo:.1f}-{hi:.1f} m (2-98 pct)")

    if not rows:
        print("no frames to render")
        return

    h = rows[0][0].height
    W = len(cols) * width + (len(cols) + 1) * PAD
    H = HEADER + len(rows) * (h + HEADER) + PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for c, name in enumerate(cols):
        draw.text((PAD + c * (width + PAD), 4), name, fill=(60, 60, 60))
    y = HEADER
    for row, caption in zip(rows, captions):
        for c, im in enumerate(row):
            canvas.paste(im, (PAD + c * (width + PAD), y))
        draw.text((PAD, y + h + 3), caption, fill=(90, 90, 90))
        y += h + HEADER

    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"wrote {out}  ({out.stat().st_size / 1e3:.0f} KB)")
    for c in captions:
        print(f"  {c}")


if __name__ == "__main__":
    sys.exit(main())
