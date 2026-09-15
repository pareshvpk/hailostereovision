"""
End-to-end learning check on synthetic stereo, with no dataset required.

Generates random-texture stereo pairs from a known disparity field, trains for a
few hundred steps, and asserts the error collapses. This exercises the whole
pipeline -- cost volume, aggregation, soft-argmin, refinement ladder, masked
loss, AMP -- and it is the strongest possible refutation of the CV-1 defect:
a network whose hypotheses are all identical cannot fit a disparity field at
all, no matter how long you train it.

    python src/test_learns.py --steps 400
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo, MAX_DISP  # noqa: E402
from train import masked_loss  # noqa: E402

H, W = 64, 512
MAX_SYNTH_DISP = 96


def make_batch(batch, rng, device):
    """Random-texture pairs warped by a piecewise-constant disparity field.

    right[x] = left[x + d(x,y)] exactly, so the pair is geometrically
    consistent and the ground truth is exact by construction.
    """
    lefts, rights, disps = [], [], []
    span = W + MAX_SYNTH_DISP
    for _ in range(batch):
        # low-frequency texture upsampled to full size, so the images have
        # structure at the scale the 1/8 matcher actually sees
        coarse = rng.random((3, H // 8 + 1, span // 8 + 1)).astype(np.float32)
        scene = np.repeat(np.repeat(coarse, 8, axis=1), 8, axis=2)[:, :H, :span]
        scene = scene + 0.15 * rng.random((3, H, span)).astype(np.float32)
        scene = np.clip(scene, 0.0, 1.0)

        # a few horizontal slabs of constant disparity
        disp = np.zeros((H, W), dtype=np.float32)
        edges = sorted(rng.choice(np.arange(8, H - 8), size=3, replace=False).tolist())
        bounds = [0] + edges + [H]
        for a, b in zip(bounds[:-1], bounds[1:]):
            disp[a:b, :] = rng.integers(0, MAX_SYNTH_DISP)

        xs = np.arange(W)[None, :] + disp.astype(np.int64)
        xs = np.clip(xs, 0, span - 1)
        rows = np.arange(H)[:, None]
        left = scene[:, :, :W]
        right = scene[:, rows, xs]

        lefts.append(left)
        rights.append(right)
        disps.append(disp)

    to = lambda a: torch.from_numpy(np.stack(a)).to(device)  # noqa: E731
    left = to(lefts)
    right = to(rights)
    disp = to(disps)[:, None]
    mask = (disp > 0).float()
    mask[..., :MAX_DISP] = 0.0        # CV-2: no match exists in the left border
    return left, right, disp, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    model = HailoStereo().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.2)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    # fixed held-out batch, never trained on
    eval_rng = np.random.default_rng(999)
    ev = make_batch(4, eval_rng, device)

    def measure():
        model.eval()
        with torch.no_grad():
            pred = model(ev[0], ev[1])
            err = (pred - ev[2]).abs() * ev[3]
            return err.sum().item() / ev[3].sum().clamp(min=1).item()

    start_epe = measure()
    print(f"device {device} | {args.steps} steps | batch {args.batch}")
    print(f"EPE before training: {start_epe:.2f} px\n")

    t0 = time.time()
    model.train()
    for step in range(args.steps):
        left, right, disp, mask = make_batch(args.batch, rng, device)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            loss = masked_loss(model(left, right), disp, mask)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % 50 == 0 or step == args.steps - 1:
            epe = measure()
            model.train()
            print(f"  step {step:4d}  loss {loss.item():7.3f}   held-out EPE {epe:6.2f} px")

    final_epe = measure()
    print(f"\nEPE {start_epe:.2f} -> {final_epe:.2f} px in {time.time() - t0:.0f}s")

    # The relative check is the real one, and it holds at any step count: a
    # model with CV-1 cannot reduce the error at all, because every disparity
    # hypothesis carries the same evidence.
    assert final_epe < start_epe * 0.35, (
        f"model failed to learn disparity: EPE only moved {start_epe:.2f} -> "
        f"{final_epe:.2f}. A cost volume with identical hypotheses (CV-1) fails "
        f"exactly this way.")
    # The absolute bar only means something once the run is long enough to have
    # converged; at 400 steps this task still sits around 7 px.
    if args.steps >= 1500:
        assert final_epe < 5.0, (
            f"EPE {final_epe:.2f} px after {args.steps} steps is too high for "
            f"this task -- expected under 5 px")
    print(f"PASS -- the matching path learns disparity "
          f"({100 * (1 - final_epe / start_epe):.0f}% error reduction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
