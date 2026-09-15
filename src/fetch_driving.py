"""
Fetch the SceneFlow *Driving* subset and convert it to a compact on-disk form.

Driving is 4,400 rendered road-scene stereo pairs -- the closest SceneFlow
subset to KITTI's domain, which is what this model is ultimately finetuned on.

Why a conversion step
---------------------
SceneFlow stores disparity as float32 PFM: 540*960*4 = 2.07 MB per frame, so
Driving's disparity is 8.9 GB compressed and ~18 GB extracted. Stored instead
as 16-bit PNG scaled by 32, precision is 1/32 = 0.03 px -- far finer than the
1-3 px this model will ever resolve -- and the same data occupies roughly
1.3 GB. Combined with the WebP image variant the whole subset lands near 3.5 GB.

The download is resumable (curl -C -), because 10 GB at a few hundred KB/s is
long enough that a dropped connection is likely. Archives are deleted once
converted.

    python src/fetch_driving.py --root data/driving
    python src/fetch_driving.py --root data/driving --stage convert   # resume after a crash
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import read_pfm  # noqa: E402

BASE = ("https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16"
        "/Release_april16/data/Driving")
IMAGES = f"{BASE}/raw_data/driving__frames_cleanpass_webp.tar"
DISPARITY = f"{BASE}/derived_data/driving__disparity.tar.bz2"

DISP_SCALE = 32.0          # 1/32 px precision, matching the uint16 range
MAX_STORED_DISP = 65535 / DISP_SCALE


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: pathlib.Path, max_bytes: int | None = None,
             max_stalls: int = 8) -> pathlib.Path:
    """Resumable download. Returns immediately if the file is already complete.

    `max_bytes` deliberately truncates the transfer. A bz2 tar is a sequential
    stream, so a prefix still decodes every member that ends before the cut --
    which turns a 16-hour download into however long you are willing to wait,
    at proportionally fewer training pairs.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(url)
    target = min(expected, max_bytes) if (expected and max_bytes) else (max_bytes or expected)

    have = dest.stat().st_size if dest.exists() else 0
    if target and have >= target:
        print(f"  already have {human(have)} of {dest.name}"
              f"{' (capped)' if max_bytes else ''}")
        return dest

    if have:
        print(f"  resuming {dest.name} at {human(have)} of {human(target or 0)}")
    else:
        print(f"  downloading {dest.name} ({human(target or 0)}"
              f"{' of ' + human(expected) + ', capped' if max_bytes and expected else ''})")

    # curl's own --retry does not cover a connection reset partway through a
    # transfer, so one dropped connection kills a multi-hour download. Wrap it
    # in a loop that resumes from whatever is already on disk, and only give up
    # after several consecutive attempts make no progress at all.
    t0, started_at = time.time(), have
    stalled = 0
    while stalled < max_stalls:
        have = dest.stat().st_size if dest.exists() else 0
        if target and have >= target:
            break

        if max_bytes:
            # an explicit byte range cannot be combined with -C, so request the
            # remainder and append it ourselves
            cmd = ["curl", "-L", "--retry", "5", "--retry-delay", "5",
                   "--retry-connrefused", "-r", f"{have}-{target - 1}", url]
            with open(dest, "ab") as fh:
                rc = subprocess.run(cmd, stdout=fh).returncode
        else:
            cmd = ["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "5",
                   "--retry-connrefused", "-o", str(dest), url]
            rc = subprocess.run(cmd).returncode

        now = dest.stat().st_size if dest.exists() else 0
        if target and now >= target:
            break
        if now > have:
            stalled = 0          # made progress; a reset mid-transfer is normal
            print(f"  ...resuming after interruption at {human(now)} "
                  f"({100 * now / target:.1f}%)", flush=True)
        else:
            stalled += 1
            print(f"  no progress (curl exit {rc}), attempt {stalled}/{max_stalls}",
                  flush=True)
            time.sleep(min(60, 5 * 2 ** stalled))

    got = dest.stat().st_size
    rate = (got - started_at) / max(time.time() - t0, 1e-6)
    print(f"  got {human(got)} at {human(rate)}/s")
    if target and got < target:
        raise RuntimeError(
            f"{dest.name}: stopped at {human(got)} of {human(target)} after "
            f"{max_stalls} attempts with no progress. Re-run to resume.")
    return dest


def remote_size(url: str) -> int | None:
    out = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    return None


def extract_images(archive: pathlib.Path, root: pathlib.Path) -> int:
    """WebP frames are already small; extract them as they are."""
    out = root / "frames"
    n = 0
    with tarfile.open(archive, "r|") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".webp"):
                continue
            # strip the archive's own top-level directory
            rel = pathlib.Path(*pathlib.Path(member.name).parts[1:])
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(dest, "wb") as fh:
                shutil.copyfileobj(src, fh)
            n += 1
            if n % 500 == 0:
                print(f"    {n} frames", flush=True)
    return n


def extract_disparity(archive: pathlib.Path, root: pathlib.Path) -> int:
    """Stream the bz2 tar, converting each PFM to a 16-bit PNG as it appears.

    Streaming matters here: the archive expands to ~18 GB of PFM, and none of
    it is ever written to disk.
    """
    out = root / "disparity"
    n, clipped, truncated = 0, 0, False

    def convert(member, src) -> None:
        nonlocal clipped
        disp = read_pfm(src)
        clipped += int(np.count_nonzero(disp > MAX_STORED_DISP))
        quant = np.clip(disp * DISP_SCALE, 0, 65535)
        quant[~np.isfinite(disp)] = 0
        rel = pathlib.Path(*pathlib.Path(member.name).parts[1:]).with_suffix(".png")
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        # no mode= -- passing it to reinterpret dtype is deprecated in Pillow 12
        # and removed in 13; a uint16 array already infers I;16
        Image.fromarray(quant.astype(np.uint16)).save(dest, format="PNG", optimize=True)

    with tarfile.open(archive, "r|bz2") as tar:
        try:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".pfm"):
                    continue
                src = tar.extractfile(member)
                if src is None:
                    continue
                convert(member, src)
                n += 1
                if n % 500 == 0:
                    print(f"    {n} maps", flush=True)
        except (tarfile.TarError, EOFError, OSError) as exc:
            # expected when --max-gb truncated the archive: every member that
            # ended before the cut has already been written
            truncated = True
            print(f"    archive ends early ({type(exc).__name__}); "
                  f"keeping the {n} complete maps decoded so far")

    if clipped:
        print(f"    note: {clipped} pixels exceeded {MAX_STORED_DISP:.0f} px and were "
              f"clipped (they are outside the model's {192} px range regardless)")
    if truncated:
        print(f"    partial dataset: {n} disparity maps")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/driving")
    ap.add_argument("--stage", choices=("all", "download", "convert"), default="all")
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--max-disp-gb", type=float, default=None,
                    help="cap the disparity download (GB). The archive is a "
                         "sequential stream, so a prefix yields proportionally "
                         "fewer complete frames instead of failing.")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    cache = root / "_archives"
    root.mkdir(parents=True, exist_ok=True)

    img_tar = cache / "driving__frames_cleanpass_webp.tar"
    disp_tar = cache / "driving__disparity.tar.bz2"

    if args.stage in ("all", "download"):
        print("[1/2] images")
        download(IMAGES, img_tar)
        print("[2/2] disparity")
        cap = int(args.max_disp_gb * 1024 ** 3) if args.max_disp_gb else None
        download(DISPARITY, disp_tar, max_bytes=cap)

    if args.stage in ("all", "convert"):
        print("\nextracting images...")
        n_img = extract_images(img_tar, root)
        print(f"  {n_img} frames -> {root / 'frames'}")

        print("converting disparity (PFM -> uint16 PNG)...")
        n_disp = extract_disparity(disp_tar, root)
        print(f"  {n_disp} maps -> {root / 'disparity'}")

        if not args.keep_archives:
            shutil.rmtree(cache, ignore_errors=True)
            print("  removed archives")

        size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
        print(f"\ndataset ready at {root}  ({human(size)})")
        print(f"train with:  python src/train.py --dataset sceneflow --root {root}")


if __name__ == "__main__":
    main()
