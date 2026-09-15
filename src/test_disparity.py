"""
Regression tests for the stereo matching path.

The Hailo Model Zoo `stereonet` shipped for years with a cost volume whose 12
disparity hypotheses were bit-identical -- it performed no disparity search at
all. A shape-only smoke test passes happily on that model. These tests do not.

Run directly:   python src/test_disparity.py
Or via pytest:  pytest src/test_disparity.py -v
"""

from __future__ import annotations

import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import (  # noqa: E402
    HailoStereo,
    CostVolume,
    shift_right_features,
    _shift_all,
    MATCH_SCALE,
    NUM_DISP,
)


def test_export_shift_matches_reference():
    """The export-friendly pad-once/slice-many form must be numerically
    identical to the reference shift. These are two implementations of the same
    equation and they are allowed to disagree by nothing at all."""
    x = torch.randn(1, 8, 5, 40)
    fast = _shift_all(x, NUM_DISP)
    assert len(fast) == NUM_DISP
    for d in range(NUM_DISP):
        ref = shift_right_features(x, d)
        assert torch.equal(fast[d], ref), \
            f"d={d}: static-graph shift disagrees with the reference shift"
    print(f"  ok  static shift matches reference for all {NUM_DISP} hypotheses")


# ---------------------------------------------------------------------------
# 1. the shift itself
# ---------------------------------------------------------------------------

def test_shift_actually_shifts():
    """out[..., x] must equal in[..., x-d], with the first d columns zeroed.

    This is the direct assertion the original model fails: its shift returned
    the input unchanged for every d.
    """
    w = NUM_DISP * 2 + 16          # wider than the largest hypothesis
    # each column holds its own index, so a shift is trivially readable
    x = torch.arange(w, dtype=torch.float32).view(1, 1, 1, w).repeat(1, 4, 3, 1)

    for d in range(NUM_DISP):
        out = shift_right_features(x, d)
        assert out.shape == x.shape, f"d={d}: shift changed the shape"
        if d > 0:
            assert torch.all(out[..., :d] == 0), \
                f"d={d}: columns before the shift boundary must be zero"
        moved = out[..., d:]
        expect = x[..., : w - d]
        assert torch.equal(moved, expect), \
            f"d={d}: shifted content is wrong -- this is the CV-1 defect"
    print(f"  ok  shift is a real shift for all d in 0..{NUM_DISP - 1}")


def test_shift_is_not_identity():
    """Explicitly reject the original's failure mode."""
    x = torch.randn(1, 8, 5, 32)
    identical = [d for d in range(1, NUM_DISP)
                 if torch.equal(shift_right_features(x, d), x)]
    assert not identical, \
        f"shift is a no-op at d={identical} -- the CV-1 defect has returned"
    print("  ok  no disparity level is a no-op")


# ---------------------------------------------------------------------------
# 2. the cost volume built from it
# ---------------------------------------------------------------------------

def test_cost_volume_slices_are_distinct():
    """The exact check that fails on the shipped stereonet, where every
    hypothesis reduced to `left - right` at zero disparity."""
    cv = CostVolume(ch=32, num_disp=NUM_DISP, groups=8).eval()
    left, right = torch.randn(1, 32, 24, 64), torch.randn(1, 32, 24, 64)

    with torch.no_grad():
        vol = cv(left, right)

    g = cv.groups
    assert vol.shape[1] == NUM_DISP * g
    base = vol[:, :g]
    collisions = []
    for d in range(1, NUM_DISP):
        sl = vol[:, d * g:(d + 1) * g]
        if torch.equal(sl, base):
            collisions.append(d)
    assert not collisions, \
        f"cost-volume levels {collisions} are identical to level 0 (CV-1)"
    print(f"  ok  all {NUM_DISP} cost-volume levels are distinct")


# ---------------------------------------------------------------------------
# 3. does matching actually find the right disparity?
# ---------------------------------------------------------------------------

def test_matching_finds_known_disparity():
    """End-to-end geometric check that needs no trained weights.

    Build a right image that is the left image shifted by exactly `k*8` pixels.
    Because the tower is stride-8, its 1/8 features are then shifted by exactly
    `k` cells, so the feature difference at hypothesis d=k is exactly zero --
    whatever the (random) weights happen to be. The argmin of the raw matching
    energy must therefore land on k.

    A model with a broken shift produces a flat energy profile and fails here.
    """
    torch.manual_seed(0)
    model = HailoStereo().eval()

    h, w, pad = 96, 512, MATCH_SCALE * NUM_DISP
    scene = torch.rand(1, 3, h, w + 2 * pad)

    failures = []
    for k in (0, 1, 3, 7, 12, 20):
        # A point at left column x appears at right column x - k*8, so the right
        # image is the scene sampled k*8 pixels further along: right[x] = left[x + k*8].
        shift_px = k * MATCH_SCALE
        left = scene[..., pad:pad + w]
        right = scene[..., pad + shift_px:pad + shift_px + w]

        with torch.no_grad():
            _, _, fl = model.features(left)
            _, _, fr = model.features(right)
            # crop off the border cells, where convolution padding breaks the
            # exact-shift relationship
            margin = NUM_DISP + 4
            energy = [
                (fl[..., margin:] - shift_right_features(fr, d)[..., margin:])
                .pow(2).mean().item()
                for d in range(NUM_DISP)
            ]

        best = min(range(NUM_DISP), key=energy.__getitem__)
        if best != k:
            failures.append((k, best, energy[k], energy[best]))
        else:
            runner_up = sorted(energy)[1]
            print(f"  ok  true disparity {shift_px:3d}px -> argmin bin {best:2d}"
                  f"   energy {energy[k]:.2e} vs next {runner_up:.2e}")

    assert not failures, (
        "matching did not recover the known disparity: "
        + "; ".join(f"true bin {k} -> got {g}" for k, g, _, _ in failures)
    )


def test_disparity_output_is_sane():
    """Shape, sign and finiteness of the deployed (eval-mode) output."""
    model = HailoStereo().eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 96, 320), torch.rand(1, 3, 96, 320))
    assert out.shape == (1, 1, 96, 320), f"unexpected output shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "output contains NaN or Inf"
    assert (out >= 0).all(), "disparity must be non-negative after the final ReLU"
    print(f"  ok  output {tuple(out.shape)}, range "
          f"[{out.min():.2f}, {out.max():.2f}] px")


# ---------------------------------------------------------------------------

def main():
    tests = [
        ("shift is a real shift", test_shift_actually_shifts),
        ("shift is never identity", test_shift_is_not_identity),
        ("static shift matches reference", test_export_shift_matches_reference),
        ("cost volume levels distinct", test_cost_volume_slices_are_distinct),
        ("matching recovers known disparity", test_matching_finds_known_disparity),
        ("output is well formed", test_disparity_output_is_sane),
    ]
    failed = 0
    for title, fn in tests:
        print(f"\n[{title}]")
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {exc}")
    print("\n" + "-" * 60)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
