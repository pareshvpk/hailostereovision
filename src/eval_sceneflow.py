"""
Score a checkpoint on the held-out SceneFlow scene -- the forgetting metric.

    python src/eval_sceneflow.py --ckpt runs/sceneflow_border/best.pt --root data/driving

A pure KITTI finetune catastrophically forgets the large-disparity skill it
learned in pretraining (measured 3.30 -> 28.36 px on this scene), and replay is
the tool tuned against that regression (IMPROVEMENT_PLAN.md §1.2). To tune it we
need the number, reported the same way every run: the EPE on the whole held-out
SceneFlow subset that training itself validates on.

The validation set is built by calling train.build_datasets with dataset=
sceneflow, so it is BY CONSTRUCTION the same hold-out (35mm_focallength/
scene_forwards/slow, stride-sampled to --val-limit=200) and the same masked
metric (train.evaluate) that produced the in-loop `epe` in the pretrain
checkpoint. sceneflow_border/best.pt reproduces its recorded 3.610 px here,
which is the correctness check for this script.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import types

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo  # noqa: E402
from train import build_datasets, evaluate, SCENEFLOW_HOLDOUT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--root", required=True,
                   help="SceneFlow root that holds the hold-out scene, "
                        "e.g. data/driving (what the pretrain trained on)")
    p.add_argument("--holdout", default=None,
                   help=f"comma-separated subsets to score; default "
                        f"{','.join(SCENEFLOW_HOLDOUT)}")
    p.add_argument("--val-limit", type=int, default=200,
                   help="stride-sample the hold-out to this many frames "
                        "(200 matches the pretrain's validation)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # A minimal namespace with exactly the fields build_datasets reads for the
    # sceneflow branch. crop/aug only touch the TRAIN set it also builds; the
    # val set is full-frame (training=False), so they do not affect the score.
    ns = types.SimpleNamespace(
        dataset="sceneflow", root=args.root,
        crop_h=256, crop_w=512, pass_name=None, limit=None,
        val_limit=args.val_limit, holdout=args.holdout, aug="basic")
    _, val_set = build_datasets(ns)

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=max(1, args.workers // 2), pin_memory=True)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HailoStereo()
    model.load_state_dict(blob.get("model", blob), strict=True)   # EXP-1
    model.to(device).eval()

    epe, d1 = evaluate(model, val_loader, device)
    recorded = blob.get("epe", float("nan"))
    print(f"{args.ckpt}: epoch {blob.get('epoch', '?')}, "
          f"recorded in-loop EPE {recorded:.3f} px")
    print(f"held-out SceneFlow ({len(val_set)} frames): "
          f"EPE {epe:.3f} px | D1 {d1:.2f}%")


if __name__ == "__main__":
    main()
