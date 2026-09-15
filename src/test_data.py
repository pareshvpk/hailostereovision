"""
Tests for the dataset layer, run against a synthetic directory tree.

These exist so a defect in the loader surfaces now rather than after a
multi-hour download finishes. They cover the converted Driving layout that
fetch_driving.py writes, the original SceneFlow layout, and the validity mask
that carries the CV-2 fix.

    python src/test_data.py
"""

from __future__ import annotations

import io
import pathlib
import shutil
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import (  # noqa: E402
    SceneFlow,
    read_pfm,
    read_sceneflow_disp,
    SCENEFLOW_PNG_SCALE,
    MEAN,
    STD,
)
from model import MAX_DISP  # noqa: E402

H, W = 96, 320
SCENE = "35mm_focallength/scene_forwards/fast"


def write_pfm(path: pathlib.Path, disp: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"Pf\n")
        fh.write(f"{disp.shape[1]} {disp.shape[0]}\n".encode())
        fh.write(b"-1.0\n")
        fh.write(np.flipud(disp).astype("<f4").tobytes())


def build_tree(root: pathlib.Path, n=4, converted=True):
    """Create a miniature SceneFlow tree in either layout."""
    rng = np.random.default_rng(0)
    frames = root / ("frames" if converted else "frames_cleanpass")
    disp_dir = root / "disparity"
    truth = {}

    for i in range(n):
        name = f"{i:04d}"
        # a disparity ramp plus a slab, so the mask has something to bite on
        disp = np.tile(np.linspace(2, 210, W, dtype=np.float32), (H, 1))
        disp[H // 2:, :] = 40.0
        truth[name] = disp

        for side in ("left", "right"):
            img = (rng.random((H, W, 3)) * 255).astype(np.uint8)
            ip = frames / SCENE / side / f"{name}.{'webp' if converted else 'png'}"
            ip.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(ip)

            dp = disp_dir / SCENE / side / name
            if converted:
                q = np.clip(disp * SCENEFLOW_PNG_SCALE, 0, 65535).astype(np.uint16)
                dp.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(q).save(dp.with_suffix(".png"), format="PNG")
            else:
                write_pfm(dp.with_suffix(".pfm"), disp)
    return truth


def test_converted_layout():
    print("[converted layout: frames/*.webp + disparity/*.png]")
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        truth = build_tree(root, n=4, converted=True)
        # In eval mode `crop` is deliberately ignored: stereo is evaluated on
        # the whole frame, rounded down to a multiple of 16 for the network.
        ds = SceneFlow(root, crop=(64, 256), training=False)
        assert len(ds) == 4, f"found {len(ds)} pairs, expected 4"
        s = ds[0]
        for k in ("left", "right", "disp", "mask"):
            assert k in s, f"sample missing '{k}'"
        assert s["left"].shape == (3, H, W), s["left"].shape
        assert s["disp"].shape == (1, H, W), s["disp"].shape
        assert torch.isfinite(s["left"]).all(), "normalized image has non-finite values"

        # disparity must survive the uint16 PNG round trip
        err = s["disp"][0].numpy() - truth["0000"]
        assert np.abs(err).max() <= 1.0 / SCENEFLOW_PNG_SCALE, \
            f"disparity error {np.abs(err).max():.4f} px exceeds the quantum"
        print(f"  ok  {len(ds)} pairs, full frame {H}x{W}, disparity exact to "
              f"{1 / SCENEFLOW_PNG_SCALE:.5f} px")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_training_crop():
    """In training mode the requested crop is applied, and it must be a
    multiple of 16 for the refinement ladder's fixed 2x upsamples."""
    print("[training crop]")
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        build_tree(root, n=2, converted=True)
        ds = SceneFlow(root, crop=(64, 256), training=True)
        seen = set()
        for _ in range(6):
            s = ds[0]
            assert s["left"].shape == (3, 64, 256), s["left"].shape
            assert s["disp"].shape == (1, 64, 256), s["disp"].shape
            seen.add(float(s["disp"].sum()))
        assert len(seen) > 1, "random crop returned the same window every time"
        print(f"  ok  crops to 64x256, {len(seen)} distinct windows in 6 draws")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_original_layout():
    print("[original layout: frames_cleanpass/*.png + disparity/*.pfm]")
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        build_tree(root, n=3, converted=False)
        ds = SceneFlow(root, crop=(64, 256), training=False)
        assert len(ds) == 3, f"found {len(ds)} pairs, expected 3"
        _ = ds[0]
        print(f"  ok  {len(ds)} pairs read from PFM")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_mask_excludes_unmatchable_border():
    """CV-2: the left MAX_DISP columns reference right-image coordinates that
    do not exist, and must never contribute to the loss."""
    print("[validity mask]")
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        build_tree(root, n=1, converted=True)
        # full width so the border rule is actually exercised
        ds = SceneFlow(root, crop=(96, 320), training=False)
        s = ds[0]
        mask = s["mask"][0].numpy()
        disp = s["disp"][0].numpy()

        assert mask[:, :MAX_DISP].sum() == 0, \
            f"{int(mask[:, :MAX_DISP].sum())} pixels inside the unmatchable " \
            f"left {MAX_DISP}-column border are marked valid (CV-2)"
        assert mask.sum() > 0, "mask excluded everything"
        assert not ((disp >= MAX_DISP) * mask).any(), \
            "pixels beyond the model's disparity range are marked valid"
        assert not ((disp <= 0) * mask).any(), "non-positive disparity marked valid"
        print(f"  ok  border excluded, {100 * mask.mean():.1f}% of pixels valid")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_normalization_roundtrip():
    print("[normalization]")
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        build_tree(root, n=1, converted=True)
        s = SceneFlow(root, crop=(64, 256), training=False)[0]
        arr = s["left"].numpy()
        recovered = arr.transpose(1, 2, 0) * STD + MEAN
        assert recovered.min() >= -0.02 and recovered.max() <= 1.02, \
            f"de-normalized range [{recovered.min():.3f}, {recovered.max():.3f}] " \
            f"is not a valid image"
        print(f"  ok  de-normalizes to [{recovered.min():.3f}, {recovered.max():.3f}]")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pfm_reader_accepts_file_object():
    """fetch_driving.py reads PFMs straight out of a streaming tar."""
    print("[pfm reader]")
    disp = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    buf = io.BytesIO()
    buf.write(b"Pf\n8 6\n-1.0\n")
    buf.write(np.flipud(disp).astype("<f4").tobytes())
    buf.seek(0)
    got = read_pfm(buf)
    assert np.array_equal(got, disp), "PFM read from a file object is wrong"
    print("  ok  reads from an open file object without touching disk")


def main():
    tests = [
        test_pfm_reader_accepts_file_object,
        test_converted_layout,
        test_training_crop,
        test_original_layout,
        test_mask_excludes_unmatchable_border,
        test_normalization_roundtrip,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {type(exc).__name__}: {exc}")
    print("-" * 60)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
