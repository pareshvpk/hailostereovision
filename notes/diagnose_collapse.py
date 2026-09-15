"""Diagnose the bistable collapse seen in the synthetic learning test.

Hypothesis: the soft-argmin's softmax saturates or flattens, so the disparity
decode stops carrying gradient and the model settles on a constant prediction.
Tracks temperature, logit spread, softmax entropy and the spread of the decoded
1/8 disparity across several seeds.

Entropy near log(24)=3.18 means a flat softmax -- every hypothesis equally
likely, decoded disparity pinned at the mean index. Entropy near 0 means a
saturated softmax with no gradient.
"""

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from model import HailoStereo  # noqa: E402
from train import masked_loss  # noqa: E402
from test_learns import make_batch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
UNIFORM = float(np.log(24))


def run(seed, steps=400, lr=2e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = HailoStereo().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    ev = make_batch(4, np.random.default_rng(999), DEV)

    print(f"\n=== seed {seed} (lr {lr}, {steps} steps) ===")
    print(f"{'step':>5s} {'temp':>6s} {'logit_std':>10s} {'entropy':>9s} "
          f"{'d8_std':>7s} {'EPE':>7s}")
    for step in range(steps):
        left, right, disp, mask = make_batch(8, rng, DEV)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            loss = masked_loss(m(left, right), disp, mask)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % 100 == 0 or step == steps - 1:
            m.eval()
            with torch.no_grad():
                _, _, fl = m.features(ev[0])
                _, _, fr = m.features(ev[1])
                cost = m.aggregation(m.cost_volume(fl, fr))
                t = float(m.soft_argmin.temperature())
                logits = -cost * t
                p = torch.softmax(logits, 1)
                ent = (-(p * p.clamp_min(1e-9).log()).sum(1)).mean().item()
                d8 = m.soft_argmin(cost)
                pred = m(ev[0], ev[1])
                epe = (((pred - ev[2]).abs() * ev[3]).sum() / ev[3].sum()).item()
            print(f"{step:5d} {t:6.3f} {logits.std().item():10.3f} "
                  f"{ent:6.3f}/{UNIFORM:.2f} {d8.std().item():7.3f} {epe:7.2f}")
            m.train()


if __name__ == "__main__":
    for s in (0, 1, 2):
        run(s)
