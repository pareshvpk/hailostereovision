# Handoff to Linux — Phase 5 (export, compile, emulate, report)

> Windows training (Phases 0–3) is done. This file hands the winning model to
> an Ubuntu session for the Linux-only deployment phase. Written for Claude.

## ✅ COMPLETE — 2026-09-16

All five Phase 5 steps are done. Results, measured on Linux:

| step | result |
|---|---|
| ONNX export | all audits PASS, ONNX↔torch parity **5.57e-04 px** |
| float (reproduced on Linux) | masked **1.156 px**, official **1.248 px**, D1 6.87% — matches Windows exactly |
| int8 emulated | masked **1.333 px**, official **1.430 px**, D1 8.00% (shipped was 1.848) |
| HEF | **4.12 MB, 7 contexts**, compiled in 5 m 52 s |
| depth, 0–5 m | mean \|dZ\| **1.08 → 0.45 m**, EPE 18.66 → 7.29 px |
| STATUS.md | updated; §2.3 + open item 3 corrected; open item 6 fixed |

**Two deviations from the plan below, both deliberate:**

1. **The HEF is not a 0.95-utilization build.** That configuration no longer
   completes here — three attempts of 41–88 min all stalled at context 3/5,
   with 34 of 46 allocator failures being `shmifo in capacity exceeded
   (available: 20, required: 39)`. Built instead with `--compiler-effort 0`
   (automatic utilization): 7 contexts / 4.12 MB rather than 5 / 4.69 MB.
   Smaller, but more context switches, and the FPS effect is unmeasured.
2. **`dfc_flow.py` hardcodes the ONNX and HEF paths**, so "export under a new
   name" was not possible as written. The shipped artefacts were copied to
   `artifacts/shipped/` and `deploy/build/shipped/` (md5-verified) instead.

Still open, both needing a Hailo-15H board: on-device validation (item 1) and
FPS/latency (item 2 — `hailo profiler` crashes on this graph in DFC 5.4.0,
now confirmed on the 7-context build too).

The NTFS drive is at `/media/paresh/1426359D263580B2/Microchip/hailo-stereo`
on Ubuntu (same files as `C:\Microchip\hailo-stereo` on Windows).

## The winning model

**`runs/kitti_mixed_fixed768/best.pt`** — the Phase 3 finetune. Export THIS.

KITTI 2015 val (40 pairs), full-frame, float32:

| protocol | EPE | D1-all |
|---|---|---|
| masked (training mask) | **1.156 px** | 6.38% |
| official (every valid GT pixel) | **1.248 px** | 6.87% |

best ≈ last (official 1.248 vs 1.250) — export `best.pt`.

Disparity-binned (matched x≥192 / border x<192), from `eval_kitti.py --bins`:

| GT disp | matched EPE / bias | border EPE / bias |
|---|---|---|
| 0–40 px | 1.02 / +0.03 | 1.58 / −0.05 |
| 40–80 px | 1.18 / +0.17 | 1.49 / −0.21 |
| 80–100 px | 5.24 / −3.08 | 3.81 / −0.95 |
| 100–120 px | 9.43 / −8.38 | 16.76 / −13.45 |
| 120–160 px | 21.22 / −21.22 | 6.03 / −4.15 |

SceneFlow hold-out (forgetting metric): **13.05 px** (old pretrain scored 3.610;
this finetune trades some SceneFlow for the big KITTI gain).

**Reference to beat / compare against:**
- Shipped `kitti_border` float: official **1.659 px**, D1 9.92%, int8 emulated **1.848 px** (+13.0%).
- New model float official **1.248 px** — a 25% headline improvement. int8 target: ≤ ~1.41 px if degradation stays ~+13%.

## How the winner was produced (exact commands)

Two stages, both on Windows GPU (RTX 4060), cooler mode `--epoch-pause` was a
thermal mitigation only and does not affect the model:

```
# 3.1 pretrain (RNG-fixed, batch 8, lr 5e-4 — batch/lr reduced from the
#     reference 16/1e-3 purely for laptop thermals; hold-out 3.604 px)
python src/train.py --dataset sceneflow --root data/driving \
  --epochs 30 --batch 8 --lr 5e-4 --workers 8 --ema 0.999 --aug strong \
  --out runs/sceneflow_fixed

# 3.2 finetune (the winner)
python src/train.py --dataset kitti_mixed --root data/kitti2015/training \
  --kitti2012-root data/kitti2012/training --init runs/sceneflow_fixed/best.pt \
  --epochs 300 --batch 4 --lr 1e-4 --ema 0.999 --aug strong \
  --crop-h 320 --crop-w 768 --out runs/kitti_mixed_fixed768
```

## What changed in `src/` this session (all additive, behind opt-in flags)

- **`data.py`**: added `Kitti2012` loader (`colored_0`/`colored_1`/`disp_occ`,
  same 16-bit /256 encoding as 2015). Added disparity-aware sampling machinery
  (`FAR_DISP=80`, `_StereoBase.far_frame_flags`, `_read_disp` per subclass,
  `far_crop_bias`/`_crop_y`). None of it changes default behaviour.
- **`train.py`**: `--dataset kitti_mixed` + `--kitti2012-root`; `--disp-aware-sampling`
  (WeightedRandomSampler, `FAR_OVERSAMPLE=3`); `--epoch-pause SECONDS` (thermal
  duty-cycle, not part of the OneCycle total_steps so resume is unaffected).
- **`eval_kitti.py`**: `--bins` flag (the binned table above). Reproduces the
  original Finding-1 numbers on `kitti_border/best.pt` exactly.
- **`eval_sceneflow.py`** (new): SceneFlow hold-out scorer; reuses
  `train.build_datasets` + `train.evaluate`. Reproduces 3.610 on `sceneflow_border`.
- **`check_ingest.py`**: `--dataset kitti2012` option (warping-residual sanity;
  KITTI 2012 PASSED, residual min at x1.0).

`export_onnx.py`, `dfc_flow.py`, `eval_depth.py` were NOT changed — they read
`ckpt["model"]`, which is the EMA/scored weights, so they pick up the new model
with no edits.

## Phase 5 steps (Linux)

1. `python src/export_onnx.py --ckpt runs/kitti_mixed_fixed768/best.pt ...`
   — all audits PASS, ONNX↔torch parity < 1e-3 px. (Confirm the exact CLI in
   `export_onnx.py`; it must not overwrite `artifacts/hailo_stereo.onnx` if you
   want to keep the shipped one — use a new name until Phase 5 passes.)
2. `./deploy/hailo-py deploy/dfc_flow.py all` — HEF builds at 0.95 utilization
   (the shmifo limit still caps at 24 cost-volume slices; the graph is unchanged
   so this should behave exactly as the shipped build).
3. Emulated int8 EPE — compare vs shipped 1.848 px; int8 degradation vs +13.0%.
4. `python src/eval_depth.py` — 0–5 m band vs the old 1.08 m mean |dZ| (this is
   the metric the large-disparity fix should move most).
5. Update `STATUS.md`: new numbers, and **correct §2.3 and open item 3** — the
   0–5 m error was NOT the 192 px ceiling; it was missing large-disparity
   supervision (fixed by KITTI 2012 + the worker-RNG pretrain fix). See
   IMPROVEMENT_PLAN.md §1 and §5 for the full evidence.

## Guard-rails

- **Never overwrite** `runs/kitti_border/`, `runs/sceneflow_border/`, or
  `artifacts/hailo_stereo_hailo15h.hef` until Phase 5 passes. New outputs get
  new names.
- Full results and per-phase verdicts: `report/IMPROVEMENT_PLAN.md` §5.

## Before switching: shut Windows down FULLY

Do **not** hibernate or use Fast Startup — both leave the NTFS partition locked
so Ubuntu can only mount it read-only, and the run dirs/checkpoints won't be
writable. Do a full shutdown (or `shutdown /s /t 0`), then boot Ubuntu.
