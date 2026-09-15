"""
Build a calibration set for the Hailo Dataflow Compiler.

    python src/make_calib.py --root data/driving --n 64

The compiler quantizes against real data, and what it sees must be what the
chip sees. The parser is configured with `normalize_in_net: true` (matching the
original's YAML), which means the chip is handed **raw uint8 RGB** and does the
ImageNet normalization on-device. So the calibration arrays here are uint8 NHWC
and deliberately NOT normalized -- feeding normalized floats to a network whose
first layer is a normalization would calibrate every scale in the graph against
a distribution the hardware never receives.

Two inputs, so two arrays. They are written separately and as one npz, because
the DFC accepts either depending on how the calib set is wired up.

Aspect ratio: the exported graph is 368x1232, which is KITTI's shape almost
exactly (native 1242x375), so KITTI frames are near-native here -- scaled to
width 1232 and cropped to 368 rows from the bottom, barely resampled. Driving
frames are 540x960, a different aspect; the same path scales and bottom-crops
them, keeping the road surface and horizon rather than squashing the geometry.

**Calibrate on the domain you deploy in.** The deployed model is KITTI-finetuned,
so `--dataset kitti` is the right choice; a Driving calibration set for a
KITTI model is exactly the domain gap that costs int8 accuracy.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import SceneFlow, Kitti2015  # noqa: E402

HEIGHT, WIDTH = 368, 1232


def load_u8(path: str) -> np.ndarray:
    """Raw uint8 RGB at the graph's input shape, aspect preserved."""
    img = Image.open(path).convert("RGB")
    scale = WIDTH / img.width
    img = img.resize((WIDTH, max(HEIGHT, round(img.height * scale))),
                     Image.BILINEAR)
    top = img.height - HEIGHT          # anchor at the bottom, keep the road
    return np.asarray(img.crop((0, top, WIDTH, img.height)), dtype=np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("sceneflow", "kitti"), default="sceneflow")
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="artifacts/calib")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--pass-name", default=None)
    p.add_argument("--stride", type=int, default=37,
                   help="sample every Nth pair; a prime stride avoids drawing "
                        "every frame from one contiguous shot")
    args = p.parse_args()

    if args.dataset == "kitti":
        # the finetune's own training split -- never the 40 held-out pairs the
        # quantization penalty is reported on
        ds = Kitti2015(args.root, split="train")
        stride = max(1, len(ds.samples) // args.n)
    else:
        ds = SceneFlow(args.root, training=False, pass_name=args.pass_name)
        stride = args.stride
    picks = ds.samples[::stride][:args.n]
    if len(picks) < args.n:
        print(f"warning: only {len(picks)} pairs available at stride "
              f"{args.stride}; asked for {args.n}")

    left = np.stack([load_u8(s[0]) for s in picks])
    right = np.stack([load_u8(s[1]) for s in picks])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_name(out.name + "_left.npy"), left)
    np.save(out.with_name(out.name + "_right.npy"), right)
    # compressed: the npz duplicates the two .npy files, and uncompressed it
    # doubled the calibration set's footprint for no benefit
    np.savez_compressed(out.with_suffix(".npz"), left=left, right=right)

    print(f"{len(picks)} pairs from {args.root}")
    print(f"  left  {left.shape} {left.dtype}  range [{left.min()}, {left.max()}]")
    print(f"  right {right.shape} {right.dtype}")
    print(f"  mean per channel (uint8): {left.reshape(-1, 3).mean(0).round(1)}")
    print(f"wrote {out.with_name(out.name + '_left.npy')}, "
          f"{out.with_name(out.name + '_right.npy')}, {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
