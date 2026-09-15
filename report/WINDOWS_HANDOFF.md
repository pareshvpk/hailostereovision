# Handoff prompt — HailoStereo accuracy work on Windows

> Paste everything below the line into a new Claude Code session started in
> `C:\Microchip\hailo-stereo`. It is written for Claude, not for a human reader.

---

You are continuing work on **HailoStereo**, a stereo-disparity network built to
replace the Hailo Model Zoo `stereonet` on a **Hailo-15H**. The project lives at
`C:\Microchip\hailo-stereo` (the same NTFS drive is mounted on this laptop's
Ubuntu as `/media/paresh/1426359D263580B2/Microchip/hailo-stereo`). This session
runs on **Windows** and does the **training** phases. The final deployment
phase (HEF compile) is Linux-only and will be done later in an Ubuntu session.

## Read first, in this order

1. `report/IMPROVEMENT_PLAN.md` — the plan you are executing. Authoritative.
2. `report/STATUS.md` — project state as of 2026-09-11. **§2.3 and open item 3
   are wrong** about the cause of the 0–5 m error (see "Findings" below); do not
   repeat that claim.
3. `src/train.py`, `src/data.py`, `src/model.py`, `src/eval_kitti.py`.

Code style: long explanatory docstrings/comments that cite *measured* numbers
and defect IDs (CV-2, EXP-1, ...). Match it. New behaviour goes behind opt-in
flags so old runs stay reproducible.

## Where things stand (2026-09-14)

**Shipped checkpoint:** `runs/kitti_border/best.pt` — KITTI 2015 val (40 pairs)
**1.464 px masked / 1.659 px official EPE**, D1 9.92% official. Emulated int8
1.848 px. HEF: `artifacts/hailo_stereo_hailo15h.hef`. **Never overwrite either.**

Other runs: `kitti` 1.407 masked (older leaky pretrain), `kitti_replay` 1.735
masked / 1.876 official (SceneFlow mixed in at ratio 1.0), `sceneflow_border`
3.610 px on the SceneFlow hold-out (the pretrain every KITTI run starts from).

### Already changed today (on Linux, tested there)

- **Bug fix, `src/data.py`:** `_StereoBase` built `random.Random(seed)` in
  `__init__`, so every DataLoader worker got an identical copy → with
  `--workers 4` only 4 distinct augmentation draws per 16 samples. Now an `rng`
  property reseeds per worker from `get_worker_info().seed` (verified 16/16).
  Every earlier run trained with this bug. Always on.
- **`--ema DECAY`** in `train.py` (e.g. `0.999`): `AveragedModel` with
  `get_ema_multi_avg_fn`, `use_buffers=True`. When on, validation and `best.pt`
  use the EMA; checkpoint `"model"` = EMA weights (so `eval_kitti.py`,
  `export_onnx.py`, `dfc_flow.py` need no change), `"raw_model"` and `"ema"`
  are stored for `--resume`. Resume with EMA is implemented but **not yet
  exercised** — test it once (see step 1).
- **`--aug strong`**: adds per-channel gain jitter to both views and, with p=0.5,
  a 50–100 px mean-colour rectangle painted on the **right** view (asymmetric
  occlusion). Default `basic` = old behaviour.
- `src/test_disparity.py` and `src/test_data.py`: 6/6 each after the change.

### Findings that drive the plan

1. **Error is at large disparity, not the 192 px ceiling.** All 0–5 m val pixels
   have GT 77–158 px. Shipped model, val, by GT disparity: 0–80 px EPE 1.2–2.2,
   bias ≈ 0; 80–100 px EPE 10.7 (bias −10); 100–120 px 28–30 (−28/−30);
   120–160 px 35–57 (all negative). Same in the matched region and the left
   192-column border. It **under-reads** large disparity.
2. **Cause: supervision.** Only **1.0%** of KITTI 2015 train GT pixels are
   ≥ 80 px, vs **42%** in SceneFlow Driving. The shipped model under-reads even
   on its own training frames (80–100 bias −5.4, 120–160 −27.4).
3. Pretrain `sceneflow_border` also fails at 100–120 px on KITTI (bias −40.9),
   so it is part domain, not only forgetting. `kitti_replay` fixes the far end
   (120–160 px EPE 34.7 → 8.25) while costing ~0.2 px elsewhere.
4. **Low confidence above 80 px:** only 17/40 val frames have such GT;
   `000161_10` holds 40% of those pixels, three frames 73%. Decide on overall
   official EPE; treat range bins as secondary, needing multi-px differences.
5. Larger crops fit easily (peak VRAM, batch 4, AMP): 256×512 0.40 GiB,
   320×768 0.75, 368×1024 1.16. GPU is an RTX 4060 laptop, 8 GB.
6. KITTI 2012 `data_stereo_flow.zip` (2,008,641,404 bytes) is live at
   `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip`.
   Licence CC BY-NC-SA (non-commercial), same as 2015.

## Your job: Phases 0 → 3 of IMPROVEMENT_PLAN.md

Run phases in order; each phase's winner is the next phase's baseline. After
each run, append a row to a results table in `report/IMPROVEMENT_PLAN.md`
(§5 "Results log", create it): run name, what changed, masked EPE, official
EPE, official D1, EPE+bias for 80–100 / 100–120 / 120–160 px, SceneFlow
hold-out EPE, epoch of best, wall time.

### Step 1 — environment check (5 min)

```bat
cd C:\Microchip\hailo-stereo
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv\Scripts\python src\test_disparity.py
.venv\Scripts\python src\test_data.py
```
- `get_ema_multi_avg_fn` needs **torch ≥ 2.2**. If older, either upgrade the
  venv (CUDA build) or replace it in `train.py` with a manual EMA over
  `state_dict()` (params lerp, integer buffers copied) — keep checkpoint keys
  identical.
- Smoke test including resume: train 3 epochs with `--ema 0.999 --aug strong
  --out runs\_smoke`, interrupt after epoch 1 (or run `--epochs 3` and resume
  from a copied epoch-1 `last.pt` with the same `--epochs 3`), confirm it
  resumes and scores. Delete `runs\_smoke` afterwards.

### Step 2 — Phase 0: tooling (no long GPU jobs)

- **0.1** Extend `src/eval_kitti.py` with a disparity-binned table: bins
  0–40 / 40–80 / 80–100 / 100–120 / 120–160 px, per bin pixel count, EPE and
  signed bias (pred − GT), split into matched (x ≥ 192) and border (x < 192),
  over every valid-GT pixel. Val loader: `Kitti2015(root, split="val")`, model
  in eval mode, full bottom-right-cropped frames. Must reproduce the numbers in
  Finding 1 for `runs/kitti_border/best.pt` — that is your correctness check.
- **0.2** Add SceneFlow hold-out scoring (forgetting metric): a `src/eval_sceneflow.py`
  (or flag) that builds validation exactly as `train.py::build_datasets` does
  for `--dataset sceneflow` (hold-out `35mm_focallength/scene_forwards/slow`,
  stride-sampled to 200) and reuses `train.evaluate`. `sceneflow_border/best.pt`
  must give ≈ 3.610 px.

### Step 3 — Phase 1: finetune ablation (4 runs, ~2.5–3 h on Windows)

All: `--dataset kitti --root data/kitti2015/training --init runs/sceneflow_border/best.pt --epochs 300 --lr 1e-4`.
Log to `runs\<name>_train.log` (`> runs\<name>_train.log 2>&1`).

| run | extra flags |
|---|---|
| `kitti_rngfix` | *(none — RNG fix only)* |
| `kitti_ema_aug` | `--ema 0.999 --aug strong` |
| `kitti_crop768` | `--ema 0.999 --aug strong --crop-h 320 --crop-w 768` |
| `kitti_replay50` | previous + `--replay-root data/driving --replay-ratio 0.5` |

Score each: `eval_kitti.py` on `best.pt` **and** `last.pt`, plus SceneFlow
hold-out. Winner = lowest official EPE; within 0.05 px, prefer smaller ≥ 80 px
bias. Differences < 0.1 px official are noise (per-frame EPE spans 0.93–3.51).

### Step 4 — Phase 2: target the defect (~4–5 h)

- **2.1** Download KITTI 2012 into `data\kitti2012\` (unzip `training\`
  only: `colored_0\`, `colored_1\`, `disp_occ\`, files `*_10.png`, 194 pairs).
  Add `Kitti2012` to `data.py` (same contract, `read_kitti_disp` works — same
  16-bit /256 encoding), and a `--dataset kitti_mixed` in `train.py`:
  train = all 194 KITTI 2012 + 160 KITTI 2015 train; **val stays the KITTI 2015
  40** so numbers compare. Verify with `src/check_ingest.py`-style sanity
  (warping residual) on a few 2012 pairs. Run the Phase 1 winner recipe on it.
- **2.2** Disparity-aware sampling, opt-in flag: oversample training frames with
  any GT ≥ 80 px (55 of 160 in KITTI 2015 train; recount with 2012) — e.g. a
  `WeightedRandomSampler` giving those frames 3× weight — and bias their random
  crop's y toward rows containing that GT. Run on the best recipe so far.
- **2.3** Only if replay helped in Phase 1: sweep `--replay-ratio 0.25 / 1.0`.
  Skippable to save ~1.5 h.

### Step 5 — Phase 3: redo the pretrain (~2 h)

Only worth it if Phases 1–2 still leave ≥ 80 px bias clearly negative.
```bat
.venv\Scripts\python src\train.py --dataset sceneflow --root data/driving ^
  --epochs 30 --batch 16 --lr 1e-3 --workers 8 --ema 0.999 --aug strong ^
  --out runs/sceneflow_fixed > runs\sceneflow_fixed_train.log 2>&1
```
Compare hold-out EPE with 3.610 px, then rerun the best finetune recipe from
`runs/sceneflow_fixed/best.pt`.

**Phase 4 (architecture) is out of scope for this session** unless the user asks;
it needs Linux to check Hailo op support first.

## Guard-rails

- **Power:** laptop, ~100 W draw vs 52 Wh battery; battery reading dropped
  58% → 6% in ~15 min of light GPU use on 2026-09-14. Before any run, check
  it is plugged in and charged (`WMIC PATH Win32_Battery Get EstimatedChargeRemaining,BatteryStatus`).
  Power plan: never sleep. If a run dies, `--resume runs/<name>/last.pt` with
  the **same** `--epochs` and `--batch`.
- **One GPU job at a time.** Short eval scripts between runs are fine.
- New runs always get a new `--out`. Never touch `runs/kitti_border/`,
  `runs/sceneflow_border/`, or `artifacts/*.hef`.
- `best.pt` is selected on the same 40 frames it is scored on — always report
  `last.pt` (EMA) next to it; if they differ by > 0.05 px, prefer `last.pt`.
- Windows DataLoader workers use spawn: slower epochs (~7 s vs ~4 s on Linux) are
  expected, not a bug. If workers crash, try `--workers 2`.
- Ignore the `lr_scheduler.step() before optimizer.step()` and cuDNN
  "Plan failed" warnings — both benign.
- Run long jobs in the background and check logs; do not poll in tight loops.

## Handing back to Linux (Phase 5)

When done, write `report/HANDOFF_TO_LINUX.md` containing: the winning run
directory and exact command line, its official/masked EPE and D1, the binned
table, SceneFlow hold-out EPE, and anything that changed in `src/`. Then tell
the user to **fully shut down Windows** (not hibernate / Fast Startup — it locks
the NTFS drive) and boot Ubuntu. The Linux session will run
`src/export_onnx.py`, `./deploy/hailo-py deploy/dfc_flow.py all`, compare
emulated int8 against 1.848 px, run `eval_depth.py`, and update STATUS.md
(including correcting §2.3).
