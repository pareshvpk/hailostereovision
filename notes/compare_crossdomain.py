"""
Cross-domain accuracy: both models on SceneFlow Driving's held-out scene.

    ~/.venvs/stereo/bin/python notes/compare_crossdomain.py --n 40

Why this exists. notes/compare_models.py scores both models on KITTI and the
Model Zoo model wins -- but ref/stereonet.yaml declares `training_data: kitti
stereo 2015`, there are only 200 labelled KITTI frames, and our 40-pair val
split is drawn from them. The original was almost certainly trained on the
frames it is being scored on, which is why it reads 1.300 px here against the
8.223 px Hailo itself publishes on a held-out split.

`35mm_focallength/scene_forwards/slow` is HailoStereo's scene-disjoint hold-out
(SCENEFLOW_HOLDOUT in src/train.py) and the Model Zoo model never saw SceneFlow
at all. So neither model trained on these frames.

This is NOT a clean experiment either, and in the opposite direction: HailoStereo
pretrained on other Driving scenes, so it has a domain advantage here exactly as
the original has one on KITTI. Read the two tests together as a bracket, not as
a winner.

Geometry. Both ONNX graphs are fixed at 368x1232 and Driving frames are 540x960,
so each frame is bottom-cropped to 368 rows and zero-padded on the RIGHT to 1232
columns. Padding right is disparity-preserving: a pixel at x matches at x-d, so
nothing in [0,960) matches into the pad. Scoring is confined to the original 960
columns. The edge effect at x~960 is identical for both models.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
from dfc_flow import normalize                      # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from data import SCENEFLOW_PNG_SCALE                # noqa: E402

H, W = 368, 1232
SCENE = "35mm_focallength/scene_forwards/slow"
ORIGINAL = ROOT / "artifacts" / "stereonet_pretrained" / "stereonet.onnx"
OURS = ROOT / "artifacts" / "hailo_stereo.onnx"


def load_pairs(n):
    froot = ROOT / "data" / "driving" / "frames" / SCENE
    droot = ROOT / "data" / "driving" / "disparity" / SCENE
    lefts = sorted((froot / "left").glob("*"))
    step = max(1, len(lefts) // n)
    picked = lefts[::step][:n]
    out = []
    for lp in picked:
        rp = froot / "right" / lp.name
        dp = (droot / "left" / lp.name).with_suffix(".png")
        if not (rp.exists() and dp.exists()):
            continue
        left = np.array(Image.open(lp).convert("RGB"), dtype=np.uint8)
        right = np.array(Image.open(rp).convert("RGB"), dtype=np.uint8)
        disp = np.array(Image.open(dp), dtype=np.float32) / SCENEFLOW_PNG_SCALE
        h, w = disp.shape
        # bottom-crop the rows, keep every column, then pad right to W
        left, right, disp = left[h - H:], right[h - H:], disp[h - H:]
        def fit(a):
            p = np.zeros((H, W) + a.shape[2:], a.dtype)
            p[:, :w] = a
            return p
        out.append((lp.name, fit(left), fit(right), fit(disp), w))
    return out


def score_valid(pred, disp, valid_w):
    """EPE/D1 over the real columns only. Mirrors dfc_flow.score's two protocols."""
    pred, disp = pred[:, :valid_w], disp[:, :valid_w]
    out = {}
    official = np.isfinite(disp) & (disp > 0.0)
    masked = official & (disp < 192.0)
    masked[:, :192] = False
    for name, m in (("masked", masked), ("official", official)):
        err = np.abs(pred - disp)[m]
        if err.size == 0:
            out[name] = (float("nan"), float("nan"), 0)
            continue
        d1 = (err > 3.0) & (err > 0.05 * np.abs(disp[m]))
        out[name] = (float(err.mean()), 100.0 * float(d1.mean()), int(err.size))
    return out


def run(path, pairs, tag):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    acc = {"masked": [0.0, 0.0, 0], "official": [0.0, 0.0, 0]}
    print(f"\n[{tag}] {path.name}")
    for i, (name, l, r, disp, vw) in enumerate(pairs):
        p = lambda a: normalize(a[None]).transpose(0, 3, 1, 2).astype(np.float32)
        pred = np.squeeze(sess.run(None, {ins[0]: p(l), ins[1]: p(r)})[0])
        s = score_valid(pred, disp, vw)
        for k in acc:
            e, d, n = s[k]
            if n:
                acc[k][0] += e * n; acc[k][1] += d * n; acc[k][2] += n
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(pairs)}  running masked EPE "
                  f"{acc['masked'][0]/max(acc['masked'][2],1):.3f} px")
    return {k: (v[0] / v[2], v[1] / v[2], v[2]) for k, v in acc.items() if v[2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    pairs = load_pairs(args.n)
    gt = np.concatenate([d[np.isfinite(d) & (d > 0)] for _, _, _, d, _ in pairs])
    print(f"SceneFlow Driving hold-out '{SCENE}': {len(pairs)} frames")
    print(f"ground-truth disparity: median {np.median(gt):.1f} px, "
          f"p90 {np.percentile(gt,90):.1f} px, "
          f"{100*np.mean(gt>192):.1f}% beyond the 192 px range")

    orig = run(ORIGINAL, pairs, "Model Zoo stereonet")
    ours = run(OURS, pairs, "HailoStereo")

    print("\n" + "=" * 70)
    print(f"{'metric':26s} {'Model Zoo':>12s} {'HailoStereo':>13s} {'ratio':>9s}")
    for label, key, idx in (("EPE masked (px)", "masked", 0),
                            ("EPE official (px)", "official", 0),
                            ("D1-all masked (%)", "masked", 1),
                            ("D1-all official (%)", "official", 1)):
        a, b = orig[key][idx], ours[key][idx]
        print(f"{label:26s} {a:12.3f} {b:13.3f} {a/b:8.2f}x")
    print(f"\nreference: HailoStereo scored 3.610 px on its full SceneFlow "
          f"hold-out during training\n(src/train.py val, whole-scene split).")


if __name__ == "__main__":
    main()
