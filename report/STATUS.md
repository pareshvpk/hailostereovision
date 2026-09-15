# HailoStereo — Status Report

**Date:** 2026-09-11
**Project:** `hailo-stereo` — a ground-up replacement for the Hailo Model Zoo
`stereonet` entry, targeting **Hailo-15H**.
**Status:** Trained, exported, quantized, **compiled to HEF**. Blocked only on
on-device validation (no Hailo board on this machine).

---

## 1. Executive summary

The Model Zoo `stereonet` was reverse-engineered first
(`report/teardown.html`). Headline finding: its cost volume contains **twelve
bit-identical disparity hypotheses** — it performs no stereo matching at all,
and its 8.22 px KITTI EPE comes from having learned a left-to-right gradient.

This project replaces it. Every design decision traces to a defect ID from that
teardown. The replacement matches geometrically (verified by test, not by
metric), is **3.9× cheaper in compute**, and is **5.6× more accurate** on
KITTI 2015 under the official protocol.

As of 2026-09-10 the model **compiles cleanly to a Hailo-15H HEF** with DFC
5.4.0 — the same compiler release that produced the Model Zoo HEF, so the
comparison measures architecture rather than a toolchain generation gap.

| | Model Zoo `stereonet` | HailoStereo | delta |
|---|---|---|---|
| KITTI 2015 EPE, float | 8.223 px | **1.659 px** | **5.0× better** |
| KITTI 2015 EPE, int8 | 10.4 px (on device) | **1.848 px** (emulated) | **5.6× better** |
| Degradation to int8 | +25% | **+13.0%** | half the loss |
| Compute | 112.07 GOPS | **28.41 GOPS** | 3.9× lower |
| ONNX size | 23.69 MB | **3.38 MB** | 7.0× smaller |
| HEF size | 8.74 MB | **4.69 MB** | 1.9× smaller |
| Disparity hypotheses | 12 (all identical) | **24, all distinct** | — |
| 3D convolutions | 5 | **0** | — |
| Softmax width | 5,441,536 elements | **170,016** | 32× narrower |
| Largest constant | 21.76 MB (index ramp) | **0.442 MB** | 49× smaller |
| Parameters | 423,586 | 794,955 | 1.9× more |
| **Matching actually works** | **no** | **yes** (verified geometrically) | — |

Compute is 3.9× lower with *twice* the disparity hypotheses, and the softmax is
32× narrower despite having 2× the channels.

---

## 2. Accuracy

### 2.1 KITTI 2015, both protocols

Scored on a 40-pair scene-disjoint validation split.

| protocol | EPE | D1-all |
|---|---|---|
| masked (matchable region only) | 1.464 px | 8.70% |
| **official (every pixel with valid GT)** | **1.659 px** | **9.92%** |

The headline figure is the **official** one. The masked number is what training
optimises; quoting it against Hailo's 8.223 px would compare different things.
`src/eval_kitti.py` reports both.

### 2.2 The border problem, and its fix

Training masks two regions out of the loss: GT beyond the 192 px range, and the
left 192 columns whose matches lie outside the right image (defect CV-2).
Excluding them from the *loss* is correct; excluding them from the *reported
metric* is a different claim.

| region | pixels | share | mean EPE | share of total error |
|---|---|---|---|---|
| trained region | 3,353,290 | 86.8% | 1.41 px | 16.3% |
| left 192 columns | 507,724 | 13.2% | **47.75 px** | **83.7%** |
| GT beyond 192 px | 0 | 0% | — | — |

13% of the frame carried 84% of the error. The fix was **weak supervision**
rather than none — the left columns re-enter the loss at reduced weight, so the
network extrapolates from monocular cues and continuity instead of emitting
noise:

| | trained region | left 192 columns |
|---|---|---|
| masked loss | 1.41 px | 47.75 px |
| **weak border supervision** | 1.46 px | **2.95 px** |

**16× reduction in border error for 3.5% in the matched region.** CV-2 was right
about the loss and over-applied in its remedy.

Two things this also settles: the empty "GT beyond 192 px" row confirms
24 × 8 px covers KITTI completely; and single-frame EPE on the 40-image split
ranges 0.93–3.51 px, so checkpoint differences under ~0.1 px are not meaningful.

### 2.3 Distance accuracy, banded by range (new — 2026-09-11)

`src/eval_depth.py` converts prediction and ground truth to metres through the
same constants and reports error where it is actually read. A single headline
EPE hides this completely: with KITTI's f·B = 386.5 px·m, one pixel of
disparity error is **6 cm at 5 m and 6.5 m at 50 m** — a factor of 100 across
the same frame.

Measured 2026-09-11 on `runs/kitti_border/best.pt`, official protocol:

| range | pixels | mean \|dZ\| | RMSE | median | AbsRel | d<1.25 | EPE | geom. floor | bias dZ |
|---|---|---|---|---|---|---|---|---|---|
| 0–5 m | 71,399 | 1.08 m | 1.94 m | 0.59 m | 28.43% | 68.22% | 18.664 px | 0.79 m | +1.06 m |
| 5–10 m | 1,586,946 | **0.26 m** | 1.21 m | 0.11 m | **3.37%** | **98.74%** | 1.412 px | 0.22 m | +0.11 m |
| 10–20 m | 1,463,066 | **0.70 m** | 2.01 m | 0.29 m | **4.84%** | **97.01%** | 1.278 px | 0.63 m | +0.15 m |
| 20–40 m | 550,850 | 2.50 m | 4.49 m | 1.25 m | 8.86% | 91.27% | 1.263 px | 2.43 m | +0.38 m |
| 40–80 m | 187,791 | 7.99 m | 11.00 m | 5.71 m | 14.48% | 76.40% | 1.413 px | 10.43 m | −5.11 m |
| **all ≤ 80 m** | 3,860,052 | **1.14 m** | 3.31 m | 0.23 m | **5.72%** | **95.37%** | 1.659 px | 0.96 m | −0.08 m |

**How to read it.** "Geom. floor" is Z̄²/(f·B) × EPE — the metre error that
band's own pixel error implies to first order. Measured error *at* the floor is
stereo geometry doing what stereo geometry does; no amount of training removes
it. Measured error *above* the floor means the model is losing something extra.

Findings:

- **5–40 m is at the geometric floor** (0.26 vs 0.22, 0.70 vs 0.63, 2.50 vs
  2.43). The model is not the limiting factor in the band that matters most for
  driving; the sensor geometry is. 95%+ of pixels land within 25% of true
  distance out to 40 m.
- **0–5 m is the one genuine model defect.** EPE 18.66 px against ~1.3 px
  everywhere else, with a bias of −18.29 px — the model systematically
  **under-reads disparity at very close range**, reporting near objects as
  farther than they are. Only 71k pixels (1.8%), because KITTI LiDAR rarely
  returns that close, but it is the safety-relevant band. Cause: 24 hypotheses
  × 8 px tops out at 192 px disparity, and under 5 m true disparity exceeds
  that — the model cannot represent it.
- **40–80 m measures *below* its floor** (7.99 vs 10.43). Not an error: the
  floor is an unbiased-error reference, not a bound. Z = f·B/d is convex, so
  over-reading disparity costs fewer metres than under-reading it by the same
  pixels, and this band over-reads (+0.98 px).
- **Calibration caveat.** f = 715.7 px and B = 0.54 m are *nominal*, not read
  from KITTI calib files (none ship in `data_scene_flow.zip`). Prediction and
  GT pass through identical constants, so **AbsRel and d<1.25 are
  calibration-free and hold exactly**. Metre columns and band edges scale
  linearly with f·B.
- **Known defect, unscored by either protocol:** above the horizon the KITTI
  finetune drifts and reports sky at high disparity (near). LiDAR GT never
  reaches there, so it was never supervised and is never measured. Nothing
  above the horizon in a preview is trustworthy.

### 2.4 The split was leaking, and was fixed by retraining

Every run before `sceneflow_holdout` validated on a 2% tail cut of an
index-sorted list. SceneFlow Driving is a camera on a continuous trajectory
through one rendered scene, so that put frames 1–712 of a fly-through in
training and frames 713–800 of *the same fly-through* in validation.

The hold-out is now whole Driving subsets, verified scene-disjoint. Because the
earlier checkpoints had trained on the held-out scene, the fix required a
**retrain from scratch** — re-scoring would have leaked through the weights
instead of through the split.

| run | split | epochs | val EPE | val D1 |
|---|---|---|---|---|
| first pass, 2,156 pairs | leaky | 12 | 38.92 → 4.69 px | 94.9% → 32.6% |
| full set, interrupted | leaky | 20 of 30 | 3.008 px | 16.92% |
| continuation at lr 3e-4 | leaky | 14 | 2.596 px | 14.88% |
| scene-disjoint hold-out | **clean** | 30 | 3.687 px | 22.30% |
| + border supervision | **clean** | 30 | **3.610 px** | **21.96%** |

**The leak was worth about 42%**: 2.596 px on frames adjacent to training data
against 3.687 px on a scene never seen. The last four epochs read 3.697 /
3.703 / 3.687 / 3.692 — the OneCycle anneal finished, so that is convergence,
not a cut-off run.

---

## 3. Deployment — compiled and measured

All figures produced on this machine on 2026-09-10 with **DFC 5.4.0**, via
`./deploy/hailo-py deploy/dfc_flow.py`. Raw output in
`deploy/build/emulator_results.json`.

### 3.1 Emulator accuracy

| context | EPE masked | EPE official | D1 official |
|---|---|---|---|
| torch float (reference) | 1.464 px | 1.659 px | 9.92% |
| `SDK_NATIVE` | **1.464 px** | **1.659 px** | **9.92%** |
| `SDK_FP_OPTIMIZED` | **1.464 px** | **1.659 px** | **9.92%** |
| `SDK_QUANTIZED` | 1.654 px | 1.848 px | 11.92% |

**Both float contexts reproduce PyTorch exactly.** That is the result that most
easily could have been silently wrong — it proves the ONNX translation, the
on-chip `normalization` layers and the NHWC input layout are all correct.
`SDK_NATIVE` runs the parsed graph on normalized float; `SDK_FP_OPTIMIZED` runs
the post-model-script graph on raw uint8. They agree to three decimals, so
normalization moved on-chip without changing the arithmetic.

**int8 costs +0.190 px masked — +13.0%**, against the **+24.6%** that
`src/quantize_sim.py` predicted.

The simulation was not badly built; it modelled a mechanism that no longer
happens. It assumed uniform int8 with percentile calibration, recovered by
equalization and bias correction. DFC 5.4.0 at `optimization_level=2` logs
`Bias Correction skipped` / `Adaround skipped` and runs **Quantization-Aware
Fine-Tuning** instead — gradient distillation against the float model,
converging to a distill loss of 0.0385. Different mechanism, roughly half the
damage.

For comparison, the original degrades **+2.08 px** (8.223 → 10.4); this model
degrades **+0.19 px** (1.464 → 1.654). Both the relative and the absolute
figure are better, against a baseline already 5.6× stronger.

### 3.2 Quantization sensitivity (simulation, still the usable ranking)

`--sensitivity` quantizes each layer alone and ranks them. One layer dominates:

```
   +0.0242 px   soft_argmin.index      <- the [0..23] index ramp
   +0.0072 px   features.stem4.0
   +0.0067 px   refine2.stem.0
   ...                                 (everything else <= 0.003 px)
```

`soft_argmin.index` is 3.4× the next worst and ranked first on all three
checkpoints measured. Its weights are literally the constants `[0..23]`, so
int8 rounding perturbs the *disparity indices themselves* — the one place in
the graph where a weight error is a direct metric error rather than a feature
perturbation. That is the 16-bit promotion carried in the `.alls`.

But **the per-layer deltas sum to ~0.05 px against the 0.36 px the whole graph
loses**, so ~85% of the damage is cumulative rather than attributable. Promoting
the top few is worth doing and does not close the gap. (In the event, QAT
closed most of it instead.)

### 3.3 Compilation

`artifacts/hailo_stereo_hailo15h.hef` — **4.69 MB, 5 contexts, 13 m 15 s**,
sha256 `72895ed3…062189f4`. `deploy/build/hailo_stereo_hailo15h.WORKING.hef` is
a byte-identical backup.

The default 60% utilization **does not compile**:

```
Resources presolve failed: lcus=(180/80)
context hailo_stereo_context_6 shmifo in capacity exceeded
(available: 20, required: 37)
```

180 LCUs against the 15H's 80 forces a multi-context split, and at 60% the
splitter produced nine contexts with the cost volume straddling a boundary.
shmifos are the inter-context streams: 24 shifted slices feeding one concat
means 24 edges crossing a hard limit of 20.
`resources_param(max_utilization=0.95)` packs five denser contexts and keeps the
cost volume intact.

**This is the teardown's own prediction arriving on hardware.** The cost volume
is ~0.0% of the MAC budget and pure memory traffic; the README predicted "MAC
counts do not predict its cost, which is worth remembering when the graph
reaches a dataflow NPU." It is not what costs compute — it is what breaks the
allocator.

### 3.4 Calibration and input convention

- Calibration is **64 KITTI pairs**, uint8 NHWC 368×1232
  (`artifacts/calib_kitti_{left,right}.npy`), recalibrated from SceneFlow to
  the deployment domain on 2026-09-09.
- The ONNX takes ImageNet-normalized NCHW float; the model script prepends
  `normalization` layers, so from `optimize` onward the network takes **raw
  uint8 NHWC 0..255**. Feeding normalized data to the quantized context
  produces a garbage EPE that looks exactly like a quantization failure —
  `dfc_flow.py emulate` selects per context rather than trusting a flag.
- Layer names are not knowable before parsing, and the calibration set has two
  inputs (`left`, `right`) which the CLI's single `--calib-set-path` cannot
  express. Hence `dfc_flow.py` over the four CLI commands; it resolves names
  from the parsed graph into `build/hailo_stereo.resolved.alls`.

---

## 4. Verification — what is actually proven

| check | command | result |
|---|---|---|
| Geometry | `src/test_disparity.py` | **6/6 pass** — argmin of matching energy lands in the correct bin for known shifts. This is the test the original model fails. |
| Ingest | `src/check_ingest.py` | **PASS** — residual minimised at ×1.0, warping 3.61× better than not warping. Catches a reader dividing by the wrong constant. |
| Synthetic learning | `src/test_learns.py --steps 2000` | EPE **28.10 → 2.88 px** (90% reduction) |
| ONNX structural audit | `src/export_onnx.py` | **all PASS** — no 3D convs, rank ≤ 4, no shape arithmetic, no constant > 1 MB, no baked index ramp, softmax confined to matching resolution. **Fails closed.** |
| ONNX/PyTorch parity | same | **8.4e-04 px** on Linux |
| Depth cross-check | `src/eval_depth.py` | reproduces `eval_kitti.py` to the pixel (3,353,290 masked / 3,861,014 official) |
| DFC harness | `dfc_flow.py` | its KITTI loader/scorer reproduces `eval_kitti.py` exactly, so deviations once the DFC runs are the compiler's |

The audit is followed by a parity check for a specific reason:
`fuse_temperature_()` rewrites the cost head's BatchNorm scale by `−t`, and a
sign error there yields a graph that passes every structural check while
returning the **argMAX** of the cost volume — the worst match instead of the
best. Structure cannot detect that; parity can.

### One architectural finding worth keeping

The cost head (`Aggregation.out`) ends in a BatchNorm, and that BatchNorm is
load-bearing. Without it the convolution scales its logits freely — measured
spreads hit std 34 within 100 steps, driving the 24-way softmax to one-hot
(entropy 0.007 of a possible 3.18). A saturated softmax passes no gradient, so
the matching path stops learning and locks onto whichever hypothesis it picked
first. Across three seeds: **EPE 29.19 / 8.55 / 7.41** — one outright failure,
no consistency.

With the BatchNorm, entropy holds at 2.56–2.84 and the same three seeds give
**7.47 / 7.22 / 7.82**. It folds into the convolution at export, so it costs
nothing on device. Reproduced by `notes/diagnose_collapse.py`.

This is the same class of defect as NUM-1 in the original — an unscaled softmax
— arrived at from the other direction.

---

## 5. Open items

| # | item | severity | notes |
|---|---|---|---|
| 1 | **No on-device validation** | **high** | No Hailo PCIe/M.2 card and no `hailort` on this machine. 1.848 px is *emulated*; Hailo's 10.4 px is *on-device*. The emulator is bit-accurate by design, but that equivalence is unconfirmed here. Needs the HEF copied to a 15H board. |
| 2 | **FPS / latency unmeasured** | **high** | `hailo profiler` crashes on this HEF — see below. The 16.7 FPS target is unmeasured and not measurable with this toolchain. |
| 3 | Close-range (0–5 m) accuracy | medium | EPE 18.66 px, bias −18.29 px. 192 px disparity ceiling cannot represent sub-5 m depth. Needs more hypotheses or a coarser base step if that band matters. |
| 4 | Above-horizon drift | medium | Unsupervised (no LiDAR GT in sky) and unscored by either protocol. Model reports sky as near. Cosmetic for metrics, not for a consumer of the depth map. |
| 5 | KITTI registration | low | Data came from the public S3 bucket the authors serve — same bytes, but the licence acknowledgment is outstanding. KITTI 2015 is **CC BY-NC-SA, non-commercial only**. |
| 6 | `.alls` rationale is stale | low | Its justification for `optimization_level=2` ("runs equalization and bias correction") is wrong on DFC 5.4.0, which runs QAT instead. The *decision* was right; the *reason* recorded is not. |

### On item 2 — the profiler bug

```
conv layer hailo_stereo/conv78_sd0
(translated from /refine4/blocks/blocks.2/conv2/Conv)
with element-wise addition requires the output_shape of conv and of the
add to be equal
add_output_shape='[-1, 92, 308, 32]', conv_output_shape=[-1, 92, 40, 32]
```

To reach 95% utilization the allocator spatially defuses wide layers into
`*_sd<N>` shards (conv77 became 12). The estimator then compares a shard's
output shape against the *unsplit* element-wise add of its residual block and
raises. **This is a reporting bug in DFC 5.4.0, not a defect in the compiled
model** — allocation, kernel compilation and the HEF all succeed.

Three workarounds were tried; none works:

- Profiling `hailo_stereo_opt.har` (pre-allocation) succeeds but reports
  `Mapped graph data is missing` for anything allocation-dependent → no FPS.
- `allocator_param(width_splitter_defuse=disabled)` is accepted but the shards
  remain — `_sd` is *spatial* defuse, a different mechanism.
- `max_utilization=0.8`, on the theory that less packing means less splitting,
  does the opposite: **146 shards instead of 97**, conv78 split 12 ways, a
  40 m 32 s compile instead of 13 m 15 s, a 5.80 MB HEF instead of 4.69 MB —
  and the same crash.

That last result closes the search: utilization cannot be lowered to avoid the
split (0.60 fails to allocate at all, 0.80 splits *more*) and cannot
meaningfully be raised past 0.95. Every allocation this graph admits on a 15H
spatially defuses a residual conv with a fused element-wise add, and the DFC
5.4.0 estimator cannot describe one. It needs on-device timing with HailoRT, or
a DFC release that fixes the estimator.

---

## 6. Recommended next steps

1. **Get a Hailo-15H board.** Items 1 and 2 are both blocked on it and nothing
   else in the project is. Copy `artifacts/hailo_stereo_hailo15h.hef`, run
   HailoRT, and the two remaining headline claims (int8 EPE, FPS) become
   measured rather than emulated.
2. **Decide whether 0–5 m matters.** If the application reads depth inside 5 m,
   the 192 px disparity ceiling is the binding constraint and the fix is
   architectural (more hypotheses, or a non-uniform disparity ladder), not more
   training.
3. **Fix the `.alls` comment** to describe QAT rather than
   equalization/bias-correction, so the next reader is not misled about why
   level 2 was chosen.
4. **Complete the KITTI registration** before any non-research use, and note
   the CC BY-NC-SA constraint in whatever ships.
5. *(optional)* Supervise or mask the above-horizon region so the depth map is
   safe to consume whole, rather than safe only where KITTI happens to measure.

---

## 7. Environment

| | |
|---|---|
| Host | Ubuntu 22.04, x86_64, Python 3.10, RTX 4060 (8 GB), 23 GB RAM |
| Training | torch 2.3.0+cu121 (system), numpy pinned < 2.0 (torch 2.3 is built against the 1.x ABI) |
| Compiler | Hailo Dataflow Compiler **5.4.0** (2026-08-16), `--hw-arch hailo15h` |
| Datasets | SceneFlow Driving (4,400 pairs, 3.1 GB) + KITTI 2015 (~2 GB) |
| Gotcha | A ROS Humble `PYTHONPATH` sorts ahead of any venv's `site-packages` and breaks the DFC's pinned numpy/protobuf/tensorflow. Use the `deploy/hailo-py` wrapper, not a manual `activate`. |

Training draws ~100 W against a ~52 Wh battery and a ~65 W adapter, so the
machine net-discharges at ~1%/min *while plugged in*. A full pretrain +
finetune cycle is 150–200 Wh. `--resume` exists partly for this.

---

## 8. Reproducing every number here

Fastest first; each prints its own pass/fail.

```bash
PY=~/.venvs/stereo/bin/python

$PY src/test_disparity.py                                   # 6/6 geometry tests
$PY src/check_ingest.py --dataset kitti --root data/kitti2015/training
$PY src/eval_kitti.py  --ckpt runs/kitti_border/best.pt --root data/kitti2015/training
$PY src/eval_depth.py  --ckpt runs/kitti_border/best.pt --root data/kitti2015/training
$PY src/preview.py     --ckpt runs/kitti_border/best.pt --dataset kitti \
                       --root data/kitti2015/training --n 4 --out artifacts/preview_kitti.png
$PY src/export_onnx.py --ckpt runs/kitti_border/best.pt   # 6 audits + parity
$PY src/quantize_sim.py --ckpt runs/kitti_border/best.pt --dataset kitti \
                        --root data/kitti2015/training --val-limit 40 --calib 24

./deploy/hailo-py deploy/dfc_flow.py all                    # parse→optimize→emulate→compile
```

Interactive inspection: `$PY src/demo_server.py --ckpt runs/kitti_border/best.pt`
then <http://localhost:8000>. Thirteen panels, stage by stage. The four
`P(disparity = 0 / 64 / 128 / 184 px)` panels are the point — in the model this
replaces, all twelve hypotheses were bit-identical, so those panels would have
been copies of each other. The defect would have been obvious in one glance,
and it shipped anyway because nobody could see inside the cost volume.

---

## 9. File map

```
src/model.py           the architecture
src/data.py            SceneFlow + KITTI 2015 loaders, masked validity
src/train.py           deep supervision, masked loss, strict --init
src/eval_kitti.py      accuracy under both protocols
src/eval_depth.py      accuracy in metres, banded by range   [new 2026-09-11]
src/export_onnx.py     ONNX export + defect audit + parity check
src/quantize_sim.py    int8 simulation and per-layer sensitivity
src/preview.py         predictions beside ground truth, shared colour scale
src/demo_server.py     stage-by-stage inspector (stdlib only)
src/make_calib.py      calibration set for the Hailo compiler
deploy/dfc_flow.py     parse → optimize → emulate → compile → profile
deploy/hailo-py        PYTHONPATH-sanitising wrapper
report/teardown.html   reverse-engineering report on the original
notes/                 the analysis scripts behind the teardown
artifacts/hailo_stereo.onnx                3.38 MB, opset 14, 347 nodes
artifacts/hailo_stereo_hailo15h.hef        4.69 MB, 5 contexts  <- the deliverable
artifacts/stereonet_hailo15h_v5.4.0.hef    8.74 MB, the model being replaced
runs/kitti_border/best.pt                  the shipped checkpoint
```
