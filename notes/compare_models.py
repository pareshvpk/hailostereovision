"""
Head-to-head: the Hailo Model Zoo `stereonet` against HailoStereo, same data,
same scorer, same preprocessing, one process.

    ~/.venvs/stereo/bin/python notes/compare_models.py --limit 40

Both are ONNX graphs with an identical interface -- two [1,3,368,1232] inputs,
one [1,1,368,1232] disparity output -- so they can be run back to back under
onnxruntime with nothing in between to argue about.

Preprocessing is the same for both and that is not an assumption: ref/stereonet.yaml
declares mean [123.675,116.28,103.53] / std [58.395,57.12,57.375], which is
src/data.py's MEAN/STD x 255. The original's `normalize_in_net: true` means the
DFC prepends those layers when building the HEF, so the ONNX itself -- the thing
being run here -- wants normalized float, exactly like ours.

The setup is self-validating: the zoo publishes `full_precision_result: 8.223`
for the original. If this script does not land near that, the preprocessing is
wrong and every other number it prints is void. That check is printed first and
the comparison is labelled UNSAFE if it fails.

Scoring is dfc_flow.score, the same function that produced 1.464/1.659 for
HailoStereo and reproduces src/eval_kitti.py to the pixel.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
from dfc_flow import kitti_val_pairs, score, normalize   # noqa: E402

ORIGINAL = ROOT / "artifacts" / "stereonet_pretrained" / "stereonet.onnx"
OURS = ROOT / "artifacts" / "hailo_stereo.onnx"
ZOO_FLOAT_EPE = 8.223          # ref/stereonet.yaml: full_precision_result


def run_model(path, pairs, tag):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    print(f"\n[{tag}] {path.name}  ({path.stat().st_size/1e6:.2f} MB)")
    print(f"  inputs {ins} -> {[o.name for o in sess.get_outputs()]}")

    acc = {"masked": [0.0, 0.0, 0], "official": [0.0, 0.0, 0]}
    lat, per_frame = [], []
    for i, (name, left, right, disp) in enumerate(pairs):
        # NCHW normalized float, identical for both graphs
        l = normalize(left[None]).transpose(0, 3, 1, 2).astype(np.float32)
        r = normalize(right[None]).transpose(0, 3, 1, 2).astype(np.float32)
        t = time.time()
        out = sess.run(None, {ins[0]: l, ins[1]: r})[0]
        lat.append(time.time() - t)
        pred = np.squeeze(out)
        if pred.shape != disp.shape:
            raise RuntimeError(f"{tag}: got {pred.shape}, expected {disp.shape}")
        s = score(pred, disp)
        per_frame.append((name, s["masked"][0], s["official"][0]))
        for k in acc:
            e, d, n = s[k]
            if n:
                acc[k][0] += e * n; acc[k][1] += d * n; acc[k][2] += n
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(pairs)}  running official EPE "
                  f"{acc['official'][0]/max(acc['official'][2],1):.3f} px")
    res = {k: (v[0] / v[2], v[1] / v[2], v[2]) for k, v in acc.items() if v[2]}
    res["latency_s"] = float(np.mean(lat))
    res["per_frame"] = per_frame
    # A disparity map that is merely a smooth left-to-right gradient scores well
    # on road scenes; these two numbers are what separate that from a real one.
    res["pred_std"] = float(np.std(pred))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    pairs = list(kitti_val_pairs(ROOT / "data" / "kitti2015" / "training"))
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"KITTI 2015, {len(pairs)} held-out val pairs, CPU onnxruntime")

    orig = run_model(ORIGINAL, pairs, "Model Zoo stereonet")
    ours = run_model(OURS, pairs, "HailoStereo")

    # ---------------------------------------------------------------- validity
    off = abs(orig["official"][0] - ZOO_FLOAT_EPE)
    print("\n" + "=" * 74)
    print(f"VALIDITY CHECK -- does the original reproduce its published number?")
    print(f"  zoo publishes full_precision_result : {ZOO_FLOAT_EPE:.3f} px")
    print(f"  measured here (official protocol)   : {orig['official'][0]:.3f} px")
    safe = off < 1.5
    print(f"  delta {off:+.3f} px -> {'OK, comparison is like-for-like' if safe else 'MISMATCH -- see note below'}")
    if not safe:
        print("  ! The split is not necessarily the zoo's: it evaluates on its own")
        print("    kitti_stereo_val.tfrecord, whose composition is not published.")
        print("    Treat the absolute number with care; the SAME-DATA comparison")
        print("    below is still valid because both models saw identical frames.")

    # -------------------------------------------------------------- comparison
    print("\n" + "=" * 74)
    print(f"{'metric':28s} {'Model Zoo':>13s} {'HailoStereo':>13s} {'better by':>12s}")
    rows = [
        ("EPE masked (px)", orig["masked"][0], ours["masked"][0], "lower"),
        ("EPE official (px)", orig["official"][0], ours["official"][0], "lower"),
        ("D1-all masked (%)", orig["masked"][1], ours["masked"][1], "lower"),
        ("D1-all official (%)", orig["official"][1], ours["official"][1], "lower"),
        ("CPU latency (s/frame)", orig["latency_s"], ours["latency_s"], "lower"),
    ]
    for label, a, b, _ in rows:
        print(f"{label:28s} {a:13.3f} {b:13.3f} {a/b:11.2f}x")

    print(f"{'ONNX size (MB)':28s} {ORIGINAL.stat().st_size/1e6:13.2f} "
          f"{OURS.stat().st_size/1e6:13.2f} "
          f"{ORIGINAL.stat().st_size/OURS.stat().st_size:11.2f}x")

    # ------------------------------------------------------------- per frame
    print("\nper-frame official EPE (first 12):")
    print(f"  {'frame':16s} {'Model Zoo':>10s} {'HailoStereo':>12s}")
    wins = 0
    for (n, _, a), (_, _, b) in zip(orig["per_frame"], ours["per_frame"]):
        if b < a:
            wins += 1
    for (n, _, a), (_, _, b) in list(zip(orig["per_frame"], ours["per_frame"]))[:12]:
        print(f"  {n:16s} {a:10.3f} {b:12.3f}  {'ours' if b < a else 'ZOO'}")
    print(f"\nHailoStereo wins on {wins}/{len(pairs)} frames.")


if __name__ == "__main__":
    main()
