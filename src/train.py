"""
Training for HailoStereo.

    # pretrain
    python src/train.py --dataset sceneflow --root D:/data/sceneflow --epochs 20
    # finetune
    python src/train.py --dataset kitti --root D:/data/kitti2015/training \
        --init runs/sceneflow/best.pt --epochs 300 --lr 1e-4

Two things here are corrections of specific defects in the model this replaces:

CV-2  Every loss and metric is masked. The left MAX_DISP columns of a stereo
      pair have no possible match, and ground truth outside the model's
      disparity range is not a target the architecture can represent.

EXP-1 Checkpoints record the full config and the exact metric they achieved,
      and `--init` loads STRICTLY. The original's export used strict=False,
      which silently accepts a checkpoint that does not fit the model and
      leaves the mismatched tensors at random initialization.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo, MAX_DISP  # noqa: E402
from data import SceneFlow, Kitti2015, Kitti2012, AUGMENTATIONS  # noqa: E402

# deep supervision: coarse predictions get less weight than the final one
SCALE_WEIGHTS = (0.5, 0.7, 1.0, 1.0)

# Whole Driving subsets held out of training. SceneFlow Driving is a camera
# flying a continuous trajectory through one rendered scene, so an index-based
# split puts frame N in training and frame N+1 in validation -- the same
# geometry, textures and lighting one time step later. Validating on that
# measures memorisation, not generalisation, and reports a number several times
# better than the model earns. The hold-out is therefore whole subsets.
SCENEFLOW_HOLDOUT = ("35mm_focallength/scene_forwards/slow",)

# Disparity-aware sampling weight for KITTI frames that carry any GT >= FAR_DISP
# (see data.FAR_DISP). 55 of 160 KITTI 2015 train frames qualify; at 3x they are
# drawn about as often as the rest combined, directly attacking the 1% share of
# large-disparity supervision that the diagnosis (IMPROVEMENT_PLAN.md §1.2) found.
FAR_OVERSAMPLE = 3.0


def _scene_of(sample) -> str:
    """.../<focallength>/<scene>/<speed>/left/<frame> -> the three-part key."""
    return "/".join(sample[0].parts[-5:-2])


# ---------------------------------------------------------------------------

def masked_loss(preds, disp, weight):
    """Smooth-L1 over weighted pixels, summed across the refinement ladder.

    `weight` is the per-pixel loss weight from the loader, NOT the metric mask:
    1.0 where the pair can actually be matched, BORDER_WEIGHT on the left edge
    where ground truth exists but no match does, 0 elsewhere. Metrics keep
    using the strict mask, so numbers stay comparable across runs.

    The per-pixel loss is computed first and weighted after. Scaling the inputs
    instead -- smooth_l1(pred*w, disp*w) -- would shrink the residual itself and
    slide it into the quadratic region of the Huber curve, so a fractional
    weight would change the shape of the loss and not just its magnitude.

    Every prediction is already in full-resolution pixel units, so one target
    and one weight serve all four scales.
    """
    total = disp.new_zeros(())
    denom = weight.sum().clamp(min=1.0)
    for scale_w, pred in zip(SCALE_WEIGHTS, preds):
        if pred.shape[-2:] != disp.shape[-2:]:
            pred = F.interpolate(pred, size=disp.shape[-2:],
                                 mode="bilinear", align_corners=False)
        err = F.smooth_l1_loss(pred, disp, reduction="none") * weight
        total = total + scale_w * err.sum() / denom
    return total


@torch.no_grad()
def evaluate(model, loader, device):
    """EPE and D1 (fraction of pixels off by >3px and >5%), the two metrics
    KITTI reports. Masked exactly as training is."""
    model.eval()
    epe_sum = d1_sum = px = 0.0
    for batch in loader:
        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)
        disp = batch["disp"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        pred = model(left, right)
        err = (pred - disp).abs() * mask
        n = mask.sum().item()
        if n == 0:
            continue
        epe_sum += err.sum().item()
        bad = ((err > 3.0) & (err > 0.05 * disp.abs())) * mask
        d1_sum += bad.sum().item()
        px += n
    return (epe_sum / max(px, 1.0)), (100.0 * d1_sum / max(px, 1.0))


# ---------------------------------------------------------------------------

def build_datasets(args):
    if args.dataset == "sceneflow":
        train = SceneFlow(args.root, crop=(args.crop_h, args.crop_w),
                          training=True, pass_name=args.pass_name, limit=args.limit,
                          aug=args.aug)
        val = SceneFlow(args.root, crop=(args.crop_h, args.crop_w),
                        training=False, pass_name=args.pass_name,
                        limit=args.val_limit)

        spec = getattr(args, "holdout", None)
        holdout = (tuple(h.strip() for h in spec.split(",") if h.strip())
                   if spec else SCENEFLOW_HOLDOUT)
        present = {_scene_of(s) for s in train.samples}
        unknown = [h for h in holdout if h not in present]
        if unknown:
            raise SystemExit(
                f"--holdout names {unknown}, which is not in {args.root}. "
                f"Available: {sorted(present)}")

        held = [s for s in train.samples if _scene_of(s) in holdout]
        kept = [s for s in train.samples if _scene_of(s) not in holdout]
        if not kept:
            raise SystemExit("--holdout would leave nothing to train on")

        # stride, not head: a contiguous slice of a fly-through is one moment of
        # one trajectory, so it would report on a fraction of the scene
        limit = args.val_limit or len(held)
        val.samples = held[::max(1, len(held) // limit)][:limit]
        train.samples = kept
        print(f"held out {sorted(holdout)}: {len(held)} frames "
              f"({len(val.samples)} sampled for validation), "
              f"{len(kept)} left to train on")
        return train, val
    far_bias = getattr(args, "disp_aware_sampling", False)
    train = Kitti2015(args.root, crop=(args.crop_h, args.crop_w), split="train",
                      aug=args.aug, far_crop_bias=far_bias)
    val = Kitti2015(args.root, crop=(args.crop_h, args.crop_w), split="val")

    # KITTI 2012. Same rig and encoding as 2015 (data.py::Kitti2012), 194 more
    # labelled real-world pairs -- it attacks the diagnosed defect directly by
    # more than doubling the finetune set (160 -> 354). Validation stays the
    # KITTI 2015 40-frame hold-out so every run's official EPE stays comparable.
    # `train` becomes a ConcatDataset; the val loader and metric are untouched.
    kitti_parts = [train]
    if args.dataset == "kitti_mixed":
        k2012 = Kitti2012(args.kitti2012_root, crop=(args.crop_h, args.crop_w),
                          training=True, aug=args.aug, far_crop_bias=far_bias)
        kitti_parts.append(k2012)
        print(f"kitti_mixed: {len(train)} KITTI 2015 + {len(k2012)} KITTI 2012 "
              f"= {len(train) + len(k2012)} training frames "
              f"(val stays the KITTI 2015 {len(val)})")

    # Replay. A pure KITTI finetune catastrophically forgets: measured 3.30 ->
    # 28.36 px on the held-out SceneFlow scene, for 1.9 px on KITTI. Mixing the
    # pretraining distribution back in at a fixed ratio is the standard remedy.
    # Off unless --replay-root is given, so existing runs are unchanged.
    if getattr(args, "replay_root", None):
        replay = SceneFlow(args.replay_root, crop=(args.crop_h, args.crop_w),
                           training=True, pass_name=args.pass_name, aug=args.aug)
        # The hold-out scene stays held out: it is what the fix is measured on,
        # and training on it would make that measurement meaningless.
        spec = getattr(args, "holdout", None)
        holdout = (tuple(h.strip() for h in spec.split(",") if h.strip())
                   if spec else SCENEFLOW_HOLDOUT)
        kept = [s for s in replay.samples if _scene_of(s) not in holdout]
        if not kept:
            raise SystemExit("--replay-root leaves no frames after the hold-out")
        # Ratio is against the whole KITTI base (2015, plus 2012 under
        # kitti_mixed), so a pure-kitti run reproduces the old count exactly.
        base = sum(len(p) for p in kitti_parts)
        n = int(round(args.replay_ratio * base))
        # stride rather than head: a contiguous slice of a fly-through is one
        # moment of one trajectory, the same reason build_datasets strides val
        replay.samples = kept[::max(1, len(kept) // max(n, 1))][:n]
        print(f"replay: {len(replay.samples)} SceneFlow frames mixed with "
              f"{base} KITTI (ratio {args.replay_ratio})")
        kitti_parts.append(replay)

    train = kitti_parts[0] if len(kitti_parts) == 1 else ConcatDataset(kitti_parts)
    return train, val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("sceneflow", "kitti", "kitti_mixed"),
                   required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--kitti2012-root", default="data/kitti2012/training",
                   help="KITTI 2012 training/ dir (colored_0, colored_1, "
                        "disp_occ); used only when --dataset kitti_mixed")
    p.add_argument("--out", default=None)
    p.add_argument("--init", default=None, help="checkpoint to start from")
    p.add_argument("--resume", default=None,
                   help="checkpoint to resume from -- restores the optimizer, "
                        "the LR schedule and the epoch counter, not just weights")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--crop-h", type=int, default=256)
    p.add_argument("--crop-w", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pass-name", default=None,
                   help="SceneFlow image directory; auto-detected when omitted "
                        "(frames/ from fetch_driving.py, or frames_cleanpass/)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=200)
    p.add_argument("--holdout", default=None,
                   help="comma-separated SceneFlow subsets to reserve for "
                        "validation, e.g. 35mm_focallength/scene_forwards/slow; "
                        f"default {','.join(SCENEFLOW_HOLDOUT)}")
    p.add_argument("--replay-root", default=None,
                   help="SceneFlow root to mix into a KITTI finetune. A pure "
                        "KITTI finetune forgets: 3.30 -> 28.36 px on the "
                        "held-out SceneFlow scene. Off when omitted.")
    p.add_argument("--replay-ratio", type=float, default=1.0,
                   help="SceneFlow frames per KITTI frame when --replay-root "
                        "is given (1.0 = equal mix)")
    p.add_argument("--aug", choices=AUGMENTATIONS, default="basic",
                   help="'strong' adds per-channel jitter and asymmetric "
                        "right-view occlusion on top of crop + colour jitter")
    p.add_argument("--ema", type=float, default=0.0,
                   help="weight EMA decay, e.g. 0.999; 0 disables. Validation, "
                        "best.pt and the exported 'model' weights are the EMA")
    p.add_argument("--disp-aware-sampling", action="store_true",
                   help="oversample KITTI frames carrying GT >= 80 px (3x) and "
                        "bias their crops toward those rows -- attacks the 1%% "
                        "large-disparity supervision share (KITTI branches only)")
    p.add_argument("--epoch-pause", type=float, default=0.0,
                   help="seconds to sleep after each epoch, to duty-cycle the "
                        "GPU and hold laptop temps down (thermal mitigation; "
                        "does not affect the model, only wall time)")
    p.add_argument("--amp", action="store_true", default=True)
    args = p.parse_args()
    if not 0.0 <= args.ema < 1.0:
        p.error("--ema must be in [0, 1)")

    if args.crop_h % 16 or args.crop_w % 16:
        p.error("crop dimensions must be divisible by 16")
    if args.init and args.resume:
        p.error("--init and --resume are mutually exclusive")

    out = pathlib.Path(args.out or f"runs/{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set, val_set = build_datasets(args)

    # Disparity-aware oversampling. A WeightedRandomSampler draws far-GT KITTI
    # frames FAR_OVERSAMPLE x more often; SceneFlow (replay) frames stay at 1x
    # since 42% of their GT is already large. shuffle and the sampler are
    # mutually exclusive, so shuffle is dropped when the sampler is used.
    sampler = None
    if args.disp_aware_sampling:
        parts = (train_set.datasets if isinstance(train_set, ConcatDataset)
                 else [train_set])
        weights, n_far = [], 0
        for part in parts:
            if isinstance(part, (Kitti2015, Kitti2012)):
                flags = part.far_frame_flags()
                n_far += sum(flags)
                weights += [FAR_OVERSAMPLE if f else 1.0 for f in flags]
            else:                                   # SceneFlow replay: uniform
                weights += [1.0] * len(part)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
        print(f"disparity-aware sampling: {n_far} KITTI frames with GT >= 80 px "
              f"weighted {FAR_OVERSAMPLE}x, crops biased toward those rows")

    train_loader = DataLoader(train_set, batch_size=args.batch,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=max(1, args.workers // 2), pin_memory=True)

    model = HailoStereo().to(device)
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        # STRICT -- see EXP-1 in the module docstring
        model.load_state_dict(state, strict=True)
        print(f"initialised strictly from {args.init}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_loader),
        pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")

    # Weight EMA. On 160 KITTI frames the per-epoch val EPE swings ~0.1 px, so
    # "best epoch" is partly a lucky draw. An exponential average of the weights
    # (BatchNorm statistics included) is a smoother, usually better model and
    # costs nothing on device -- it is the same graph with different numbers.
    ema = None
    if args.ema:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema),
                            use_buffers=True)

    start_epoch, best = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # with EMA on, "model" holds the average; the optimizer's weights are raw
        model.load_state_dict(ckpt.get("raw_model", ckpt["model"]), strict=True)  # EXP-1
        if ema is not None:
            if ckpt.get("ema") is None:
                p.error(f"{args.resume} was trained without --ema; resume it "
                        f"without --ema, or start a fresh run with --init")
            ema.load_state_dict(ckpt["ema"])
        # An interrupted run resumes only if the optimizer came with it. Loading
        # weights alone and calling it a resume restarts Adam's moments from
        # zero and the LR from the top of the schedule -- a different run
        # wearing the same name.
        missing = [k for k in ("opt", "sched", "epoch") if k not in ckpt]
        if missing:
            p.error(f"{args.resume} has no {', '.join(missing)}; it predates "
                    f"resume support. Use --init to start a fresh schedule "
                    f"from its weights instead.")
        # OneCycleLR bakes the total step count in, so a resume only means
        # anything if this run reproduces the original schedule. Mismatched
        # --epochs or --batch would otherwise die mid-run with a step-count
        # error several epochs from now.
        want = sched.state_dict().get("total_steps")
        have = ckpt["sched"].get("total_steps")
        if have != want:
            p.error(f"--epochs {args.epochs} at batch {args.batch} gives "
                    f"{want} scheduler steps, but {args.resume} was built for "
                    f"{have}. Resuming needs the original run's --epochs and "
                    f"--batch; to start a fresh schedule from these weights "
                    f"use --init instead.")
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        if ckpt.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best = ckpt.get("best", ckpt["epe"])
        if start_epoch >= args.epochs:
            p.error(f"{args.resume} is already at epoch {ckpt['epoch']}; "
                    f"--epochs {args.epochs} leaves nothing to run")
        print(f"resumed from {args.resume} at epoch {start_epoch} "
              f"(best EPE {best:.3f} px)")

    print(f"device {device} | train {len(train_set)} | val {len(val_set)} | "
          f"crop {args.crop_h}x{args.crop_w} | batch {args.batch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(train_loader):
            left = batch["left"].to(device, non_blocking=True)
            right = batch["right"].to(device, non_blocking=True)
            disp = batch["disp"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = masked_loss(model(left, right), disp, weight)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)

            running += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch:3d} step {step:5d}/{len(train_loader)} "
                      f"loss {running / (step + 1):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)

        scored = model if ema is None else ema.module
        epe, d1 = evaluate(scored, val_loader, device)
        print(f"epoch {epoch:3d} | loss {running / len(train_loader):.4f} | "
              f"val EPE {epe:.3f} px | D1 {d1:.2f}% | {time.time() - t0:.0f}s"
              f"{' (ema)' if ema is not None else ''}",
              flush=True)

        # "model" is always the weights that were scored, so eval_kitti.py,
        # export_onnx.py and dfc_flow.py pick up the EMA with no changes
        payload = {"model": scored.state_dict(), "epoch": epoch,
                   "raw_model": model.state_dict() if ema is not None else None,
                   "ema": ema.state_dict() if ema is not None else None,
                   "epe": epe, "d1": d1, "args": vars(args),
                   "max_disp": MAX_DISP,
                   "opt": opt.state_dict(), "sched": sched.state_dict(),
                   "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                   "best": min(best, epe)}
        torch.save(payload, out / "last.pt")
        if epe < best:
            best = epe
            torch.save(payload, out / "best.pt")
            (out / "best.json").write_text(json.dumps(
                {"epoch": epoch, "epe": epe, "d1": d1}, indent=2))
            print(f"  new best EPE {epe:.3f} px -> {out / 'best.pt'}")

        # Thermal duty-cycle: let the GPU idle between epochs so a laptop's
        # sustained temperature settles below the throttle point. Off by default.
        if args.epoch_pause > 0:
            time.sleep(args.epoch_pause)

    print(f"\ndone. best val EPE {best:.3f} px")
    print("reference: Hailo stereonet reports 8.223 float / 10.3 quantized on KITTI")


if __name__ == "__main__":
    main()
