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
metric), is **3.9× cheaper in compute**, and is **6.6× more accurate** on
KITTI 2015 under the official protocol.

As of 2026-09-16 the model **compiles cleanly to a Hailo-15H HEF** with DFC
5.4.0 — the same compiler release that produced the Model Zoo HEF, so the
comparison measures architecture rather than a toolchain generation gap.

**Allocator change, 2026-09-16.** The shipped HEF was built with an explicit
`resources_param(max_utilization=0.95)` and came out at 4.69 MB in 5 contexts.
That configuration no longer completes on this machine: three attempts ran
41–88 min without finishing, grinding at context 3/5 because 34 of 46 allocator
failures were `shmifo in capacity exceeded (available: 20, required: 39)` — the
24-slice cost volume is expensive to place. Building instead with
`performance_param(compiler_optimization_level=0)` (automatic utilization)
succeeds in **5 min 52 s** and yields **4.12 MB in 7 contexts**. Smaller, but
**not** a like-for-like artefact: more contexts means more context switching,
and its FPS impact is unmeasured (see open item 2).

| | Model Zoo `stereonet` | HailoStereo | delta |
|---|---|---|---|
| KITTI 2015 EPE, float | 8.223 px | **1.248 px** | **6.6× better** |
| KITTI 2015 EPE, int8 | 10.4 px (on device) | **1.430 px** (emulated) | **7.3× better** |
| Degradation to int8 | +25% | **+14.5%** | 1.7× less loss |
| Compute | 112.07 GOPS | **28.41 GOPS** | 3.9× lower |
| ONNX size | 23.69 MB | **3.38 MB** | 7.0× smaller |
| HEF size | 8.74 MB | **4.12 MB** | 2.1× smaller |
| Disparity hypotheses | 12 (all identical) | **24, all distinct** | — |
| 3D convolutions | 5 | **0** | — |
| Softmax width | 5,441,536 elements | **170,016** | 32× narrower |
| Largest constant | 21.76 MB (index ramp) | **0.442 MB** | 49× smaller |
| Parameters | 423,586 | 796,531 | 1.9× more |
| **Matching actually works** | **no** | **yes** (verified geometrically) | — |

Compute is 3.9× lower with *twice* the disparity hypotheses, and the softmax is
32× narrower despite having 2× the channels.

---

## 2. Accuracy

### 2.1 KITTI 2015, both protocols

Scored on a 40-pair scene-disjoint validation split.

| protocol | EPE | D1-all |
|---|---|---|
| masked (matchable region only) | 1.156 px | 6.38% |
| **official (every pixel with valid GT)** | **1.248 px** | **6.87%** |

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

Measured 2026-09-16 on `runs/kitti_mixed_fixed768/best.pt`, official protocol:

| range | pixels | mean \|dZ\| | RMSE | median | AbsRel | d<1.25 | EPE | geom. floor | bias dZ |
|---|---|---|---|---|---|---|---|---|---|
| 0–5 m | 71,399 | 0.45 m | 1.52 m | 0.17 m | 11.30% | 92.73% | 7.288 px | 0.31 m | +0.35 m |
| 5–10 m | 1,586,946 | **0.20 m** | 0.82 m | 0.10 m | **2.61%** | **99.10%** | 1.203 px | 0.18 m | +0.02 m |
| 10–20 m | 1,463,066 | **0.58 m** | 1.78 m | 0.24 m | **4.05%** | **97.56%** | 1.074 px | 0.53 m | +0.16 m |
| 20–40 m | 550,850 | 2.23 m | 4.29 m | 1.03 m | 7.85% | 93.14% | 1.088 px | 2.09 m | +0.62 m |
| 40–80 m | 187,791 | 6.69 m | 9.61 m | 4.50 m | 12.23% | 84.05% | 1.159 px | 8.55 m | −3.04 m |
| **all ≤ 80 m** | 3,860,052 | **0.96 m** | 2.94 m | 0.19 m | **4.53%** | **96.82%** | 1.248 px | 0.72 m | +0.02 m |

Against the superseded `kitti_border` model (measured 2026-09-11, same pixels,
same masks), the 0–5 m band improved on every metric: mean \|dZ\| 1.08 → 0.45 m,
median 0.59 → 0.17 m, AbsRel 28.43% → 11.30%, d<1.25 68.22% → 92.73%, EPE
18.664 → 7.288 px. Overall ≤ 80 m: mean \|dZ\| 1.14 → 0.96 m, AbsRel 5.72% →
4.53%, d<1.25 95.37% → 96.82%.

**How to read it.** "Geom. floor" is Z̄²/(f·B) × EPE — the metre error that
band's own pixel error implies to first order. Measured error *at* the floor is
stereo geometry doing what stereo geometry does; no amount of training removes
it. Measured error *above* the floor means the model is losing something extra.

Findings:

- **5–40 m is at the geometric floor** (0.20 vs 0.18, 0.58 vs 0.53, 2.23 vs
  2.09). The model is not the limiting factor in the band that matters most for
  driving; the sensor geometry is. 93%+ of pixels land within 25% of true
  distance out to 40 m.
- **0–5 m is improved but still the weakest band.** EPE 7.29 px against ~1.1 px
  everywhere else, with a bias of −5.01 px — the model still **under-reads
  disparity at very close range**, reporting near objects as farther than they
  are, but far less than the superseded `kitti_border` model (18.66 px, bias
  −18.29 px). Measured 0.45 m against a 0.31 m floor, so residual model error
  remains on top of geometry. Only 71k pixels (1.8%), because KITTI LiDAR
  rarely returns that close, but it is the safety-relevant band.
  **Cause (corrected 2026-09-16):** *not* the 192 px disparity ceiling. Every
  0–5 m pixel in this split has GT disparity 77–158 px — comfortably inside the
  24 × 8 = 192 px range, so the ceiling was never binding. The real cause was
  missing large-disparity supervision: KITTI 2015 has only 1.0% of GT pixels
  ≥ 80 px, and the model never learned the range. Fixed by adding KITTI 2012
  (354 training pairs) and a per-worker RNG fix in the SceneFlow pretrain.
  See `report/IMPROVEMENT_PLAN.md` §1.1–1.2 for the evidence.
- **40–80 m measures *below* its floor** (6.69 vs 8.55). Not an error: the
  floor is an unbiased-error reference, not a bound. Z = f·B/d is convex, so
  over-reading disparity costs fewer metres than under-reading it by the same
  pixels, and this band over-reads (+0.62 px).
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
| torch float (reference) | 1.156 px | 1.248 px | 6.87% |
| `SDK_NATIVE` | **1.156 px** | **1.248 px** | **6.87%** |
| `SDK_FP_OPTIMIZED` | **1.156 px** | **1.248 px** | **6.87%** |
| `SDK_QUANTIZED` | 1.333 px | 1.430 px | 8.00% |

**Both float contexts reproduce PyTorch exactly.** That is the result that most
easily could have been silently wrong — it proves the ONNX translation, the
on-chip `normalization` layers and the NHWC input layout are all correct.
`SDK_NATIVE` runs the parsed graph on normalized float; `SDK_FP_OPTIMIZED` runs
the post-model-script graph on raw uint8. They agree to three decimals, so
normalization moved on-chip without changing the arithmetic.

**int8 costs +0.177 px masked — +15.3%** (official: +0.181 px, +14.5%), against
the **+24.6%** that `src/quantize_sim.py` predicted.

The simulation was not badly built; it modelled a mechanism that no longer
happens. It assumed uniform int8 with percentile calibration, recovered by
equalization and bias correction. DFC 5.4.0 at `optimization_level=2` logs
`Bias Correction skipped` / `Adaround skipped` and runs **Quantization-Aware
Fine-Tuning** instead — gradient distillation against the float model,
converging to a distill loss of 0.0268. Different mechanism, roughly half the
damage.

For comparison, the original degrades **+2.08 px** (8.223 → 10.4); this model
degrades **+0.18 px** (1.156 → 1.333). Both the relative and the absolute
figure are better, against a baseline already 6.6× stronger. Note the *relative*
int8 cost rose against the superseded model (+15.3% vs +13.0% masked) while the
*absolute* cost fell (+0.177 vs +0.190 px): quantization did not get worse, the
float baseline it is measured against got better.

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

`artifacts/hailo_stereo_hailo15h.hef` — **4.12 MB, 7 contexts, 5 m 52 s**,
sha256 `6a8e8f8b…0cee3396`, built 2026-09-16 from `runs/kitti_mixed_fixed768`
with `--compiler-effort 0`.

The superseded 0.95-utilization build — **4.69 MB, 5 contexts, 13 m 15 s**,
sha256 `72895ed3…062189f4` — is preserved at
`artifacts/shipped/hailo_stereo_hailo15h.hef`, with a byte-identical copy at
`deploy/build/hailo_stereo_hailo15h.WORKING.hef` (both re-verified 2026-09-16).

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
`resources_param(max_utilization=0.95)` packed five denser contexts and kept the
cost volume intact — that is how the superseded HEF was built, and it is why the
setting was chosen. **As of 2026-09-16 that configuration no longer completes**
on this graph: three runs of 41–88 min all stalled at context 3/5, with 34 of 46
allocator failures being `shmifo in capacity exceeded (available: 20,
required: 39)`. The current HEF is built with
`performance_param(compiler_optimization_level=0)`, which hands utilization to
the compiler and lands on seven contexts instead. See §1 "Allocator change".

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
| 1 | **No on-device validation** | **high** | No Hailo PCIe/M.2 card and no `hailort` on this machine. 1.430 px is *emulated*; Hailo's 10.4 px is *on-device*. The emulator is bit-accurate by design, but that equivalence is unconfirmed here. Needs the HEF copied to a 15H board. |
| 2 | **FPS / latency unmeasured** | **high** | `hailo profiler` crashes on this HEF — see below. Retried 2026-09-16 on the new 7-context `--compiler-effort 0` build: **same crash**, so automatic utilization still spatially defuses shards. The 16.7 FPS target is unmeasured and not measurable with this toolchain. |
| 3 | Close-range (0–5 m) accuracy | low | EPE 7.29 px, bias −5.01 px (was 18.66 / −18.29 on `kitti_border`). **The 192 px ceiling explanation was wrong** — every 0–5 m pixel has GT disparity 77–158 px, inside the range; the cause was missing large-disparity supervision (§2.3). Largely addressed by KITTI 2012 + the pretrain RNG fix. Residual error is small and concentrated in a few frames. |
| 4 | Above-horizon drift | medium | Unsupervised (no LiDAR GT in sky) and unscored by either protocol. Model reports sky as near. Cosmetic for metrics, not for a consumer of the depth map. |
| 5 | KITTI registration | low | Data came from the public S3 bucket the authors serve — same bytes, but the licence acknowledgment is outstanding. KITTI 2015 is **CC BY-NC-SA, non-commercial only**. |
| 6 | ~~`.alls` rationale is stale~~ | **fixed 2026-09-16** | Was: justified `optimization_level=2` as "runs equalization and bias correction", which DFC 5.4.0 skips in favour of QAT. Corrected in `deploy/hailo_stereo.alls`, along with two further stale claims found in the same file — the "UNVERIFIED SYNTAX / DFC not installed" header (it parses and loads here), and the claim that the calibration set comes from SceneFlow Driving (it is built from KITTI, `artifacts/calib_kitti_*.npy`). |

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

Four workarounds have been tried; none works:

- Profiling `hailo_stereo_opt.har` (pre-allocation) succeeds but reports
  `Mapped graph data is missing` for anything allocation-dependent → no FPS.
- `allocator_param(width_splitter_defuse=disabled)` is accepted but the shards
  remain — `_sd` is *spatial* defuse, a different mechanism.
- `max_utilization=0.8`, on the theory that less packing means less splitting,
  does the opposite: **146 shards instead of 97**, conv78 split 12 ways, a
  40 m 32 s compile instead of 13 m 15 s, a 5.80 MB HEF instead of 4.69 MB —
  and the same crash.
- `performance_param(compiler_optimization_level=0)` (2026-09-16), which hands
  utilization to the compiler and yields a different allocation entirely —
  7 contexts, 4.12 MB, compiled in 5 m 52 s. **Same crash**, on the same
  `conv78_sd0` shape mismatch. Spatial defusing is not a consequence of the
  0.95 utilization setting; this graph gets defused however it is allocated.

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
2. **Decide whether 0–5 m matters.** The band is now 7.29 px EPE / 11.30%
   AbsRel / 92.73% within 25% (was 18.66 px / 28.43% / 68.22%). The earlier
   diagnosis here was wrong: the 192 px ceiling was never the binding
   constraint (§2.3). Further gains would come from more large-disparity
   supervision — more KITTI 2012-like data, or disparity-aware sampling — not
   from architecture.
3. ~~**Fix the `.alls` comment**~~ — **done 2026-09-16.** It now describes QAT
   rather than equalization/bias-correction, records the measured +15.3% int8
   cost against the simulation's +24.6%, and corrects two further stale claims
   in the same file (the "UNVERIFIED SYNTAX" header and the SceneFlow
   calibration-set attribution).
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
| Datasets | SceneFlow Driving (4,400 pairs, 3.1 GB) + KITTI 2015 (~2 GB) + KITTI 2012 (194 pairs, 2.01 GB) — the shipped model finetunes on KITTI 2015 + 2012 combined (354 train pairs, `--dataset kitti_mixed`) |
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
$PY src/eval_kitti.py  --ckpt runs/kitti_mixed_fixed768/best.pt --root data/kitti2015/training --bins
$PY src/eval_depth.py  --ckpt runs/kitti_mixed_fixed768/best.pt --root data/kitti2015/training
$PY src/preview.py     --ckpt runs/kitti_mixed_fixed768/best.pt --dataset kitti \
                       --root data/kitti2015/training --n 4 --out artifacts/preview_kitti.png
$PY src/export_onnx.py --ckpt runs/kitti_mixed_fixed768/best.pt   # 6 audits + parity
$PY src/quantize_sim.py --ckpt runs/kitti_mixed_fixed768/best.pt --dataset kitti \
                        --root data/kitti2015/training --val-limit 40 --calib 24

./deploy/hailo-py deploy/dfc_flow.py parse                  # -> build/hailo_stereo.har
./deploy/hailo-py deploy/dfc_flow.py optimize               # QFT, ~6 min
./deploy/hailo-py deploy/dfc_flow.py emulate                # fp_optimized + quantized
./deploy/hailo-py deploy/dfc_flow.py emulate --contexts native
./deploy/hailo-py deploy/dfc_flow.py compile --compiler-effort 0   # 5m52s -> 4.12 MB
# NOT `dfc_flow.py all`, and NOT bare `compile`: both use --max-util 0.95, which
# no longer completes on this graph (hours, stuck at context 3/5 on shmifo
# overflow). See §1 "Allocator change" and open item 2.
```

Interactive inspection: `$PY src/demo_server.py --ckpt runs/kitti_mixed_fixed768/best.pt`
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
src/eval_sceneflow.py  SceneFlow hold-out EPE (forgetting metric) [new 2026-09-14]
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
artifacts/hailo_stereo_hailo15h.hef        4.12 MB, 7 contexts  <- the deliverable
artifacts/shipped/                         pre-2026-09-16 HEF + ONNX (kitti_border)
artifacts/stereonet_hailo15h_v5.4.0.hef    8.74 MB, the model being replaced
runs/kitti_mixed_fixed768/best.pt          the shipped checkpoint  <- KITTI 2015 + 2012 finetune
runs/sceneflow_fixed/best.pt               its SceneFlow pretrain (per-worker RNG fixed)
runs/kitti_border/best.pt                  superseded 2026-09-16 (official 1.659 px)
data/kitti2012/training                    KITTI 2012, 194 pairs   [new 2026-09-14]
```
