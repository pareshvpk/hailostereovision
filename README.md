# HailoStereo

A ground-up replacement for the Hailo Model Zoo `stereonet` entry, targeting
**Hailo-15H**.

The original was reverse-engineered first; the teardown report is at
[report/teardown.html](report/teardown.html). Its headline finding: the shipped
model's cost volume has **twelve bit-identical disparity hypotheses** and
performs no stereo matching at all. Every design decision here traces back to a
defect ID from that report.

## Where it stands

| | Model Zoo `stereonet` | HailoStereo |
|---|---|---|
| Compute | 112.07 GOPS | **28.41 GOPS** |
| Refinement share | 90.6% | **63.9%** |
| ONNX size | 23.69 MB | **3.38 MB** |
| Largest constant | 21.76 MB (index ramp) | **0.442 MB** |
| Softmax width | 5,441,536 elements | **170,016** |
| Disparity hypotheses | 12 (all identical) | **24, all distinct** |
| 3D convolutions | 5 | **0** |
| Graph nodes | 168 (+ 5D tensors) | 345 (rank ≤ 4 throughout) |
| Parameters | 423,586 | 794,955 |
| Matching works | **no** | yes — verified geometrically |
| **KITTI 2015 EPE (float)** | **8.223 px** | **1.659 px** |
| **KITTI 2015 EPE (int8)** | **10.4 px** (on device) | **1.848 px** (emulated) |
| Loss to int8 | +25% (measured, on device) | **+13.0% (measured, emulated)** |
| HEF size | 8.74 MB | **4.69 MB** |

Compute is 3.9× lower with twice the disparity hypotheses, and the softmax is
32× narrower despite having 2× the channels.

The KITTI figure is scored under the **official protocol** — every pixel with
valid ground truth, including the left border. See
[Two protocols](#two-protocols-and-why-it-matters); the masked number that
training optimises is 1.464 px, and quoting that against Hailo's 8.223 px would
be comparing different things.

On int8: as of 2026-09-10 this is measured, not simulated. The model was
compiled for Hailo-15H with DFC 5.4.0 and scored on the quantized software
model: **1.654 px masked / 1.848 px official**, a **+13.0%** loss rather than
the +24.6% `src/quantize_sim.py` predicted. The simulation assumed
equalization and bias correction; DFC 5.4.0 skips both at
`optimization_level=2` and runs quantization-aware fine-tuning instead, which
recovers about twice as much. See [deploy/README.md](deploy/README.md).

The original degrades by +2.08 px (8.223 → 10.4); this model by +0.19 px
(1.464 → 1.654). Both the relative and absolute figures are now better, against
a baseline already 5.6× better.

Comparison is like-for-like: DFC 5.4.0 is the same release that produced the
Model Zoo HEF in `artifacts/`. One caveat — 1.848 px is emulated and 10.4 px is
Hailo's on-device number. The emulator is bit-accurate by design, but there is
no 15H board here to confirm it.

## Layout

```
src/model.py           the architecture
src/data.py            SceneFlow + KITTI 2015 loaders, masked validity
src/train.py           training with deep supervision and masked loss
src/export_onnx.py     ONNX export + defect audit + ONNX/torch parity
src/quantize_sim.py    int8 simulation and per-layer sensitivity
src/preview.py         render predictions next to ground truth
src/demo_server.py     interactive stage-by-stage inspector (localhost)
src/make_calib.py      calibration set for the Hailo compiler
src/test_disparity.py  geometry tests (no training needed)
src/test_learns.py     synthetic end-to-end learning test
report/teardown.html   reverse-engineering report on the original
deploy/                .alls model script + the DFC compile sequence
ref/                   original Hailo configs and upstream source
artifacts/             downloaded originals + exported hailo_stereo.onnx
notes/                 the analysis scripts used for the teardown
```

## Running on Linux

`.venv/` is a Windows environment (`Scripts/`, `python.exe`) and does not run
here. The commands below are written against it; on Linux substitute the
equivalent interpreter. One is already set up:

```bash
~/.venvs/stereo/bin/python src/eval_kitti.py --ckpt runs/kitti_border/best.pt --root data/kitti2015/training
```

It reuses the system torch 2.3.0+cu121 (CUDA works) and adds onnx/onnxruntime,
with numpy pinned below 2.0 because torch 2.3 is built against the 1.x ABI.
Verified on 2026-09-10: the eval reproduces 1.464 px masked / 1.659 px official
unchanged from the Windows run.

The Dataflow Compiler needs its own clean environment — see
[deploy/README.md](deploy/README.md).

## Verify before training anything

```bash
.venv/Scripts/python.exe src/test_disparity.py
```

Six geometry tests, no dataset and no trained weights required. The key one
feeds a pair shifted by a known number of pixels and asserts the argmin of the
matching energy lands in the correct bin — the test the original model fails.

```bash
.venv/Scripts/python.exe src/export_onnx.py --ckpt runs/sceneflow_cont/best.pt
```

Exports to ONNX and audits the graph: no 3D convolutions, no tensors above rank
4, no shape arithmetic, no constant above 1 MB, no baked index ramp, and the
softmax confined to matching resolution. **The audit fails closed** — a value it
cannot resolve is a failure, not a pass.

Without `--ckpt` the weights are random. The audit is structural so it still
means something, but the file is not deployable and the run says so.

The audit is followed by a **parity check**: the exported graph and the source
model are run on the same input under onnxruntime and compared. The audit is
structural — it cannot tell whether the graph computes disparity at all.
`fuse_temperature_()` rewrites the cost head's BatchNorm scale by `-t`, and a
sign error there produces a graph that passes every structural check while
returning the argMAX of the cost volume: the worst match instead of the best.

```bash
.venv/Scripts/python.exe src/test_learns.py --steps 2000
```

Trains on synthetic warped stereo for ~5 minutes and asserts the error
collapses. Last run: **EPE 28.10 → 2.88 px** (90% reduction).

### Looking at the output

```bash
.venv/Scripts/python.exe src/preview.py --ckpt runs/sceneflow_cont/best.pt --root data/driving
```

Writes `artifacts/preview.png`: left image, prediction, ground truth and error,
with prediction and ground truth on a shared colour scale so a prediction that
is merely *smooth* is distinguishable from one that is *correct*. EPE is a mean
over millions of pixels and it hides shape — a model that learned a left-to-right
gradient and nothing else scores well on road scenes, because road scenes largely
are a gradient. That is close to what the original was doing.

Disparity uses a single-hue light-to-dark ramp rather than the conventional
turbo/jet. Rainbow ramps invent visual edges at their hue boundaries, which is
the exact artefact this image exists to detect.

### One finding worth keeping

The cost head (`Aggregation.out`) ends in a BatchNorm, and that BatchNorm is
load-bearing. Without it the convolution is free to scale its logits however it
likes; measured spreads hit std 34 within 100 steps, which drives the 24-way
softmax to one-hot — entropy 0.007 out of a possible 3.18. A saturated softmax
passes no gradient, so the entire matching path stops learning and the model
locks onto whichever hypothesis it happened to pick first. Across three seeds
that produced EPE 29.19 / 8.55 / 7.41 — one outright failure and no consistency.

With the BatchNorm, entropy holds at 2.56–2.84 and the same three seeds give
7.47 / 7.22 / 7.82. It folds into the convolution at export, so it costs nothing
on device. `notes/diagnose_collapse.py` reproduces the measurement.

This is the same class of defect as NUM-1 in the original model — an unscaled
softmax — arrived at from the other direction.

## Looking inside it: the inspector

```bash
.venv/Scripts/python.exe src/demo_server.py --ckpt runs/kitti_border/best.pt
```

Then open <http://localhost:8000>. Upload a rectified stereo pair, or press
**Load KITTI sample**. Standard library only — no Flask, no Gradio, nothing to
install.

It shows thirteen panels covering every stage:

| panel | what it tells you |
|---|---|
| Left / right input | what the network actually received, after the 368×1232 fit |
| Matching features, left & right (1/8) | the 32-channel feature the matcher compares, PCA-projected to RGB. Corresponding points should share a colour, offset horizontally by their disparity |
| **P(disparity = 0 / 64 / 128 / 184 px)** | what the cost volume believes at four hypotheses |
| Match confidence | softmax entropy, inverted. Bright is a confident match; textureless road and sky are uncertain |
| Disparity 1/8, refined 1/4, 1/2 | the refinement ladder, rung by rung |
| Final disparity | full resolution |

**The four probability panels are the point.** In the model this replaces, all
twelve disparity hypotheses were bit-identical, so these panels would have been
copies of each other — the defect would have been obvious in one glance, and it
shipped anyway because nobody could see inside the cost volume. Here they differ
sharply: near surfaces light up in the high-disparity slices, far ones in the
low.

### On the latency numbers

Every stage is timed with the CUDA queue drained first, and an **untimed warm-up
pass runs before the measured one**. Both matter. Without the synchronise, all
but the last stage report near zero, because kernel launches return before the
work happens. Without the warm-up, the first stage is charged for the GPU's
climb from idle clocks: an early version timed the two *identical* feature
towers at 22.6 ms and 7.2 ms, and reported a 235 ms total for what is really
20–48 ms.

Expect roughly **20–48 ms** end to end on the RTX 4060 at 368×1232 — it keeps
improving across consecutive runs as the GPU boosts, so treat it as a range.

One result worth noting: the **cost volume is the most expensive stage** in some
runs despite being ~0.0% of the MAC budget. It is 24 shifted slices and a
concat — pure memory traffic, no arithmetic. MAC counts do not predict its cost,
which is worth remembering when reading the compute table above and when the
graph reaches a dataflow NPU.

These are GPU timings and say nothing about Hailo latency. The device number
comes from the HEF profile, once it compiles.

## Reproducing the numbers

Every figure in this README can be re-derived on this machine. In order, fastest
first; each command prints its own pass/fail.

```bash
.venv/Scripts/python.exe src/test_disparity.py
```
Six geometry tests, no dataset and no weights. Asserts the argmin of the
matching energy lands in the right bin for known shifts — the test the original
model fails. Expect **6/6 passed**.

```bash
.venv/Scripts/python.exe src/check_ingest.py --dataset kitti --root data/kitti2015/training
```
Warps the right image by the ground-truth disparity and sweeps a scale
multiplier. Expect **PASS**, residual minimised at **x1.0**, and warping
**3.61x** better than not warping. This is what catches a disparity reader that
divides by the wrong constant.

```bash
.venv/Scripts/python.exe src/eval_kitti.py --ckpt runs/kitti_border/best.pt --root data/kitti2015/training
```
The headline accuracy, under both protocols. Expect **1.464 px masked /
1.659 px official**, D1 8.70% / 9.92%.

```bash
.venv/Scripts/python.exe src/preview.py --ckpt runs/kitti_border/best.pt --dataset kitti --root data/kitti2015/training --n 4 --out artifacts/preview_kitti.png
```
Look at it. Prediction and ground truth share a colour scale, so a prediction
that is merely smooth is distinguishable from one that is correct.

```bash
.venv/Scripts/python.exe src/export_onnx.py --ckpt runs/kitti_border/best.pt
```
Six structural audits plus an ONNX-vs-PyTorch parity check. Expect all
**[PASS]**, parity around **1.4e-04 px**.

```bash
.venv/Scripts/python.exe src/quantize_sim.py --ckpt runs/kitti_border/best.pt --dataset kitti --root data/kitti2015/training --val-limit 40 --calib 24
```
The int8 estimate. Expect **+24.2%** for full int8. Add `--sensitivity` for the
per-layer ranking (slow: 52 evaluations).

The whole sequence is a few minutes and about 100 W — see the note on power in
[Training](#training).

## Getting the data

```bash
.venv/Scripts/python.exe src/fetch_driving.py --root data/driving
```

Downloads the SceneFlow **Driving** subset — 4,400 rendered road-scene pairs,
the closest SceneFlow domain to KITTI. Resumable; re-run it after any
interruption. Add `--max-disp-gb N` to cap the disparity transfer: the archive
is a sequential bz2 stream, so a prefix yields proportionally fewer complete
frames rather than failing.

Disparity is converted from float32 PFM to 16-bit PNG at 1/32 px during
extraction — measured 10.9× smaller on real Driving maps (2025 → 186 KB per
frame), taking the disparity tree from 18.2 GB to 1.67 GB. Total on disk ≈ 3.1 GB.

Note the download is the slow part, not the disk. The Freiburg host is erratic:
measured 200 KB/s one afternoon and 2 MB/s the same evening, so the 10.3 GB is
anywhere from 1.5 to 16 hours. The resume loop exists for exactly this.

KITTI 2015 (~2 GB) needs a registration on the KITTI site and must be fetched
manually; point `--root` at its `training/` directory.

## Training

```bash
.venv/Scripts/python.exe src/train.py --dataset sceneflow --root data/driving --epochs 30 --batch 16 --workers 8
```

**Power:** this machine draws roughly **100 W** under training load, against a
~52 Wh battery and an adapter that supplies about 65 W. It therefore
net-discharges at roughly 1% per minute *while plugged in*, and a full
pretrain-plus-finetune cycle is 150–200 Wh — three to four times what the
battery holds. Long runs need mains power, and even then the charge falls. That
is normal for this class of laptop, not a fault. `--resume` exists partly for
this.

Batch 16 with 8 workers is what this machine wants: throughput saturates near
96 pairs/s at 1.6 GiB of the 4060's 8 GB (batch 4 gives 38 pairs/s, batch 32
buys 2% over batch 16), and image decode runs 27 pairs/s per worker, so eight
workers keep the GPU fed. About 60 s per epoch on the 4,400-pair Driving set.

```bash
.venv/Scripts/python.exe src/train.py --dataset kitti --root <kitti2015>/training --init runs/sceneflow/best.pt --epochs 300 --lr 1e-4
```

`--init` loads **strictly**. The original's export used `strict=False`, which
silently accepts a checkpoint that does not fit the model and leaves the
mismatched tensors randomly initialized (defect EXP-1).

### Resuming an interrupted run

```bash
.venv/Scripts/python.exe src/train.py --dataset sceneflow --root data/driving --out runs/sceneflow_full --resume runs/sceneflow_full/last.pt --epochs 30 --batch 16
```

Checkpoints carry the optimizer, the LR schedule, the AMP scaler and the best
metric so far, and `--resume` restores all of them. `--init` is a different
operation: it takes the weights and starts a new schedule.

The distinction matters because OneCycleLR spends its last epochs annealing, and
a "resume" that silently restarts the schedule puts the LR back at the top of the
cycle with Adam's moments zeroed — a different run wearing the same name. Two
things are therefore refused rather than worked around: a checkpoint with no
optimizer state, and an `--epochs`/`--batch` combination that does not reproduce
the original schedule's step count.

## Environment

Python 3.11, torch 2.14+cu126, onnx **1.16.2** (1.17+ fails to import on this
machine — a Windows Application Control policy blocks the `ml_dtypes` native
extension).

## Where training stands

Driving is downloaded and converted: **4,400 pairs**, 1.5 GB of frames and
1.5 GB of disparity. Ingest was checked geometrically — warping the right view
by the stored disparity cuts `|R-L|` from 0.128 to 0.050 (2.6×), so the maps
really do align the pair. 68% of pixels survive the validity mask in both the
15mm and 35mm subsets; 15% lie beyond the 192 px range and 20% are the
unmatchable left border (CV-2).

### The split was leaking, and the numbers below it are not comparable

Every run before `sceneflow_holdout` validated on a 2% tail cut of an
index-sorted sample list. SceneFlow Driving is a camera flying a continuous
trajectory through one rendered scene, so that put frames 1–712 of
`35mm_focallength/scene_forwards/slow` in training and frames 713–800 of the
*same fly-through* in validation — the same geometry, textures and lighting, one
time step later. Those runs were scored against near-duplicates of their own
training data.

The hold-out is now whole Driving subsets (`SCENEFLOW_HOLDOUT` in
`src/train.py`, overridable with `--holdout`), verified scene-disjoint. Because
the earlier checkpoints trained on the held-out scene, the fix needed a retrain
from scratch — re-scoring them would have leaked through the weights instead of
through the split.

| run | split | epochs | val EPE | val D1 |
|---|---|---|---|---|
| first pass, 2,156 pairs | leaky | 12 | 38.92 → 4.69 px | 94.9% → 32.6% |
| full set, interrupted | leaky | 20 of 30 | 3.008 px | 16.92% |
| continuation at lr 3e-4 | leaky | 14 | 2.596 px | 14.88% |
| scene-disjoint hold-out | **clean** | 30 | 3.687 px | 22.30% |
| + border supervision | **clean** | 30 | **3.610 px** | **21.96%** |

Then finetuned on KITTI 2015 (160 train / 40 val, scene-disjoint):

| model | masked EPE | official EPE | D1-all (official) |
|---|---|---|---|
| masked loss (CV-2 as written) | 1.407 px | 7.501 px | 17.97% |
| **+ weak border supervision** | 1.464 px | **1.659 px** | **9.92%** |

Read the first three as "training converged and the optimisation works", not as
accuracy. Only the last row is an accuracy estimate.

The leak was worth about **42%**: 2.596 px on frames adjacent to training data,
3.687 px on a scene the model has never seen. `runs/sceneflow_holdout/best.pt`
is epoch 28; the last four epochs read 3.697 / 3.703 / 3.687 / 3.692, so the
OneCycle anneal is finished and this is convergence, not another cut-off run.

What the leaky runs do still establish, because none of it depends on the split:
the geometry tests pass (6/6), the exported graph matches PyTorch to 5.26e-04 px,
and `artifacts/preview.png` shows object silhouettes resolving at the correct
depth — the cost volume genuinely matches, which is the property the original
lacked.

For perspective: Hailo reports 8.223 px float for the original on KITTI. That is
a different dataset and not a like-for-like comparison, and KITTI finetuning has
not been done.

## Two protocols, and why it matters

Training masks two regions out of the loss: ground truth beyond the model's
192 px range, and the left 192 columns, whose matches lie outside the right
image (CV-2). Excluding them from the *loss* is right. Excluding them from the
*reported metric* is a different claim, and for a while this project was making
it without saying so.

`src/eval_kitti.py` scores both ways. On the first KITTI model the gap was
enormous:

| region | pixels | share | mean EPE | share of total error |
|---|---|---|---|---|
| trained region | 3,353,290 | 86.8% | 1.41 px | 16.3% |
| left 192 columns | 507,724 | 13.2% | **47.75 px** | **83.7%** |
| GT beyond 192 px | 0 | 0% | — | — |

Thirteen percent of the frame carried 84% of the error. The region was never
supervised, so the cost volume there sees only zero padding and the output is
arbitrary — a masked metric simply did not look at it.

**The fix was to supervise it weakly** rather than not at all: the left columns
re-enter the loss at reduced weight, so the network learns to extrapolate from
monocular cues and continuity with the adjacent matched region, instead of
emitting noise. It cannot match there — there is no evidence to match against —
but it can produce something sane.

| | trained region | left 192 columns |
|---|---|---|
| masked loss | 1.41 px | 47.75 px |
| weak border supervision | 1.46 px | **2.95 px** |

A 16× reduction in border error for 3.5% in the matched region. CV-2 was
correct about the loss and over-applied in its remedy.

Two things this also settles: the empty "GT beyond 192 px" row confirms 24×8 px
covers KITTI completely, and the 40-image val split is small enough that
single-frame EPE ranges from 0.93 to 3.51 px, so differences under ~0.1 px
between checkpoints are not meaningful.

## Quantization

```bash
.venv/Scripts/python.exe src/quantize_sim.py --ckpt runs/sceneflow_cont/best.pt --root data/driving --sensitivity
```

The original lost 25% of its accuracy to int8 and shipped an `.alls` with no
mitigation. This measures the same exposure here, on 40 held-out frames of the
converged model, calibrated on 16 training frames it never evaluates on:

Measured on the final KITTI model, 40 held-out frames, calibrated on 24
training frames:

| | EPE | D1 | vs float |
|---|---|---|---|
| float32 | 1.464 px | 8.70% | — |
| weights int8, per channel | 1.500 px | 8.89% | +2.5% |
| activations int8 | 1.990 px | 11.98% | +35.9% |
| full int8 | 1.818 px | 10.97% | **+24.2%** |

These are reproducible run to run: calibration subsamples activations with a
seeded generator. Unseeded, the full-int8 figure moved 1-2 points between
otherwise identical runs, which is enough to make a quoted number
unverifiable.

The same measurement on the SceneFlow model gave +5.2%. The difference is the
baseline, not the robustness: absolute degradation is +0.36 px here against
+0.19 px there, while the float baseline is 2.5× smaller. Quantization noise is
roughly a fixed floor, so it costs proportionally more as the model improves.

**Activations-only scores worse than full int8**, which should not happen if the
two effects were independent. This was re-measured on all 200 labelled KITTI
frames and it strengthens rather than washes out — so it is systematic, not
sampling noise:

| | EPE | vs float |
|---|---|---|
| float32 | 0.981 px | — |
| weights int8 only | 0.929 px | -5.4% |
| activations int8 only | 1.554 px | +58.4% |
| full int8 | 1.342 px | +36.7% |

(That baseline is optimistic -- 160 of those 200 frames are training frames --
which is why float reads 0.981 px there and 1.464 px on held-out data. Use the
40-frame held-out figures above for anything quotable; this table is only for
comparing configurations against each other on identical data.)

Weight quantization reliably *reduces* the damage done by activation
quantization. The plausible mechanism is that activation quantization carries a
systematic bias -- percentile clipping plus rounding -- that weight rounding
partly cancels, but that has not been demonstrated and is a guess.

It changes nothing for deployment: the shipped configuration is full int8, and
the intermediate rows are diagnostic. The cancellation depends on the
calibration set and should not be relied on.

Weights are nearly free; the cost is in the activations. The soft-argmin
temperature settled at 1.676 and the softmax output calibrates to exactly
[0.000, 1.000] — the full unit interval, 1/255 per step, which is what NUM-1
threw away by running an unscaled softmax over 5.4M elements.

`--sensitivity` quantizes each layer alone and ranks them. One layer dominates:

```
   +0.0242 px   soft_argmin.index      <- the [0..23] index ramp
   +0.0072 px   features.stem4.0
   +0.0067 px   refine2.stem.0
   ...                                 (everything else <= 0.003 px)
```

`soft_argmin.index` is 3.4x the next worst, and ranked first on all three
checkpoints measured. Its weights are the constants
`[0..23]`, and int8 rounding perturbs the disparity indices themselves — the
one place in the graph where a weight error is a direct metric error rather
than a feature perturbation. **That is the 16-bit promotion to write into the
`.alls` first.** It ranked first on the leaky checkpoint too, by a wider margin
(5.7x), so the finding is robust even though the margin is not; treat the order
below first place as noise, since the sweep runs on 12 frames.

**But the per-layer deltas sum to about 0.05 px against the 0.36 px the whole
graph loses, so roughly 85% of the damage is cumulative rather than
attributable to any layer.** Promoting the top few to 16-bit is worth doing and
will not close the gap. The lever that might is the compiler's own optimisation
— equalization and bias correction, which this simulation does not model — so
the real int8 number could land well below +24.6%.

Caveat: this is uniform int8 with percentile calibration, not the Hailo
emulator. The DFC calibrates differently, can promote layers on its own and
applies equalization passes this does not model. The **ranking** is the usable
output; the absolute penalty is an approximate upper bound.

## Not yet done

- **Registering for KITTI.** The data was pulled from the public S3 bucket the
  dataset authors serve; the registration on the KITTI site has not been done.
  Same bytes either way, but the licence acknowledgment is outstanding, and
  KITTI 2015 is CC BY-NC-SA — non-commercial only.
- **Compilation to HEF.** The Hailo Dataflow Compiler is Linux-only and neither
  WSL nor Docker is installed here. Everything that can be prepared off-device
  is in [deploy/](deploy/README.md): the trained ONNX, a 64-pair uint8
  calibration set from `src/make_calib.py`, and an `.alls` carrying the 16-bit
  promotion for the index-ramp layer. The `.alls` syntax is **unverified** —
  it has never been through the parser — and one layer name in it is a
  placeholder that can only be resolved after parsing.
- **Confirming the target device.** The model YAML says `hailo15h, hailo10h`,
  but the Model Zoo publishes a Hailo-8 page and HEF. This decides the `.alls`.
- **Recalibrating on KITTI.** The calibration set is currently SceneFlow
  Driving, because KITTI is not downloaded. Calibration wants the deployment
  domain, and a synthetic-to-real gap there shows up as quantization error.

Disk: 55 GB free on the only drive, with Driving (3.0 GB) and its archives
(9.6 GB, deletable) already on it. Full FlyingThings3D finalpass is ~110 GB
and will not fit. Driving + Monkaa (~30 GB) or a capped FlyingThings3D subset
(`--limit`) will.
