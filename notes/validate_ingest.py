"""Validate the Driving ingest before committing to a multi-hour download.

Checks two things that would otherwise only fail after ~10 GB had transferred:

  1. the PFM -> uint16 PNG round trip preserves disparity to 1/32 px
  2. the archive member paths match what fetch_driving.py strips

Only a few MB are fetched, using range requests.
"""

import io
import pathlib
import subprocess
import sys
import tarfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from data import read_pfm, read_sceneflow_disp, SCENEFLOW_PNG_SCALE  # noqa: E402
from fetch_driving import IMAGES, DISPARITY, DISP_SCALE  # noqa: E402

SCRATCH = pathlib.Path(__file__).resolve().parent / "_scratch"
SCRATCH.mkdir(exist_ok=True)


def make_pfm(disp: np.ndarray) -> bytes:
    """Write a little-endian grayscale PFM, bottom-row-first, as SceneFlow does."""
    buf = io.BytesIO()
    buf.write(b"Pf\n")
    buf.write(f"{disp.shape[1]} {disp.shape[0]}\n".encode())
    buf.write(b"-1.0\n")
    buf.write(np.flipud(disp).astype("<f4").tobytes())
    return buf.getvalue()


def test_roundtrip():
    print("[1] PFM -> uint16 PNG -> read back")
    rng = np.random.default_rng(0)
    disp = rng.uniform(0, 192, size=(64, 96)).astype(np.float32)
    disp[0, :3] = [0.0, 191.9, 0.03125]        # edges of the range

    parsed = read_pfm(io.BytesIO(make_pfm(disp)))
    assert parsed.shape == disp.shape, f"shape {parsed.shape} != {disp.shape}"
    assert np.allclose(parsed, disp), "PFM reader did not round-trip"
    print(f"    PFM reader ok (flip and endianness correct)")

    quant = np.clip(parsed * DISP_SCALE, 0, 65535).astype(np.uint16)
    png = SCRATCH / "disp.png"
    Image.fromarray(quant, mode="I;16").save(png, optimize=True)
    back = read_sceneflow_disp(png)

    err = np.abs(back - disp).max()
    tol = 1.0 / SCENEFLOW_PNG_SCALE
    assert err <= tol, f"max error {err:.5f} px exceeds the {tol:.5f} px quantum"
    raw = disp.astype(np.float32).nbytes
    print(f"    max error {err:.5f} px (quantum {tol:.5f})")
    print(f"    {raw / 1024:.1f} KB float32 -> {png.stat().st_size / 1024:.1f} KB PNG "
          f"({raw / png.stat().st_size:.1f}x smaller)")


def head(url: str, nbytes: int) -> bytes:
    out = subprocess.run(
        ["curl", "-sL", "-r", f"0-{nbytes - 1}", "--max-time", "180", url],
        capture_output=True)
    return out.stdout


def test_archive_layout():
    print("\n[2] archive member paths")

    print("    images (.tar)...")
    names = []
    with tarfile.open(fileobj=io.BytesIO(head(IMAGES, 3 << 20)), mode="r|") as tar:
        try:
            for m in tar:
                if m.isfile():
                    names.append(m.name)
                if len(names) >= 6:
                    break
        except tarfile.TarError:
            pass
    show(names, "webp")

    print("    disparity (.tar.bz2)...")
    dnames = []
    with tarfile.open(fileobj=io.BytesIO(head(DISPARITY, 8 << 20)), mode="r|bz2") as tar:
        try:
            for m in tar:
                if m.isfile():
                    dnames.append(m.name)
                if len(dnames) >= 6:
                    break
        except (tarfile.TarError, EOFError, OSError):
            pass
    show(dnames, "pfm")

    assert names, "could not read any image member names"
    assert dnames, "could not read any disparity member names"

    # what fetch_driving.py does: drop the first path component
    for n in names[:1] + dnames[:1]:
        parts = pathlib.Path(n).parts
        print(f"    {n}")
        print(f"      strip[0] -> {pathlib.Path(*parts[1:])}")
        assert parts[-2] in ("left", "right"), \
            f"expected a left/ or right/ parent in {n}"


def show(names, kind):
    if not names:
        print(f"      (no {kind} members read)")
    for n in names[:3]:
        print(f"      {n}")


if __name__ == "__main__":
    test_roundtrip()
    test_archive_layout()
    print("\ningest validated")
