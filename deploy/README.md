# Compiling HailoStereo to a HEF

The handoff is over: this is now a Linux machine (Ubuntu 22.04, x86_64, Python
3.10, RTX 4060, 23 GB RAM), which is a supported Dataflow Compiler host. The
remaining blocker is the DFC itself, which is a login-gated wheel and cannot be
scripted around.

## Run it

```bash
# 1. Download the wheel (see below for where it actually lives)
bash deploy/install_dfc.sh ~/Downloads/hailo_dataflow_compiler-*.whl

# 2. Everything else
./deploy/hailo-py deploy/dfc_flow.py all
```

### Where the Dataflow Compiler actually is

Not under **Vision Processors -> Hailo-15H**, which is the intuitive place to
look and the wrong one. That product offers the *Vision Processor Software
Package* — the SoC-side BSP/Yocto bundle that runs **on** a 15H board.

The DFC is host-side and **device-agnostic**. There is no per-device build; the
target is a `--hw-arch` flag on one wheel. Identify it by the package, not by
the navigation path:

    Package Name: Hailo Dataflow Compiler - Python package (whl)
    File:         hailo_dataflow_compiler-5.4.0-py3-none-linux_x86_64.whl
    Devices:      Hailo-10H, Hailo-15H, Hailo-15L
    Version:      5.4.0 (2026-08-16)

**Use 5.4.0 specifically if you have the choice.** It is the same release that
built `../artifacts/stereonet_hailo15h_v5.4.0.hef`, the Model Zoo entry this
model replaces. Compiling both with one toolchain version means the 8.22 px /
10.4 px / 16.7 FPS comparison measures the architecture rather than a compiler
generation gap.

Note the device list has no Hailo-8: the 5.x line dropped it. That is only a
problem for the stale `../artifacts/stereonet.hef` (v2.19.0, Hailo-8), which is
a historical snapshot and not the competitor. `install_dfc.sh` probes every
architecture and fails unless `hailo15h` is present, so a wrong wheel is caught
at install rather than three steps into a compile.

The Vision Processor Software Package is worth having *later* — it carries the
15H-side HailoRT needed to actually execute the HEF on a board. It is not
needed for anything in this directory.

The wheel is tagged `py3-none`, so pip installs it against **any** Python 3 and
a version mismatch surfaces as an import error later rather than a refusal at
install time. `install_dfc.sh` builds its venv from this machine's `python3`
(3.10.12); if the Installation guide names a different interpreter, create the
venv from that one instead.

`dfc_flow.py` runs parse -> optimize -> emulate -> compile -> profile, and each
step writes into `deploy/build/` so a failure can be resumed rather than
restarted. `--limit 5` on `emulate` gives a fast smoke test.

## Why a script instead of the four CLI commands

The CLI sequence this file used to recommend cannot express two things.

**The calibration set has two inputs.** `hailo optimize --calib-set-path` takes
one path; this graph has `left` and `right`. The Python API accepts a dict
keyed by input layer name, which also keeps the pairs *paired* — a stereo
network calibrated on mismatched left/right frames is being shown disparities
that do not exist.

**The layer names are not knowable in advance.** The parser renames everything,
and the model script refers to layers by name. `dfc_flow.py parse` resolves
them from the parsed graph and generates `build/hailo_stereo.resolved.alls`.

`hailo_stereo.alls` stays as the record of *why* the three decisions were made.
It is no longer the file that gets executed, and it was wrong in a way the old
version of this README did not catch: it flagged `soft_argmin_index` as the one
placeholder, but `input_layer=left` in the two normalization commands is also a
name that does not exist after parsing — and in most DFC versions is also the
wrong syntax. All three are resolved now.

## The PYTHONPATH trap

This machine exports a ROS Humble `PYTHONPATH` into every shell, and those
directories sort **ahead** of any venv's `site-packages`. The DFC pins
numpy/protobuf/tensorflow tightly. `deploy/hailo-py` clears it; use that
wrapper rather than activating the venv by hand.

## What is ready

| | |
|---|---|
| `../artifacts/hailo_stereo.onnx` | 3.38 MB, opset 14, 347 nodes. Re-verified on Linux: `onnx.checker` passes, no 3D convs, rank ≤ 4 throughout, and ONNX-vs-PyTorch parity against `runs/kitti_border/best.pt` at **8.4e-04 px**. |
| `../artifacts/calib_kitti_{left,right}.npy` | 64 KITTI pairs, uint8 NHWC 368×1232. **This supersedes the SceneFlow `calib_*.npy`** — the recalibration this README used to ask for was done on 2026-09-09. `dfc_flow.py` uses the KITTI set. |
| `hailo_stereo.alls` | Rationale for the three compile decisions. Executed form is generated; see above. |
| `dfc_flow.py` | The driver. Its KITTI loader and scorer are validated — running the torch model through them reproduces `eval_kitti.py` to the pixel (3,353,290 masked / 3,861,014 official; EPE 1.463/1.659). Any deviation once the DFC runs is the compiler's, not the harness's. |

## Target architecture

**`hailo15h`** (confirmed against the live Model Zoo 2026-09-10). The current
zoo (compiled release **v5.4.0**) ships stereonet HEFs for hailo8 / hailo8l /
hailo10h / **hailo15h**; the 15H build is 8.74 MB, HW EPE **10.4 px**, 16.7 FPS
at batch 1, saved here as `../artifacts/stereonet_hailo15h_v5.4.0.hef` for a
like-for-like profile.

The `ref/HAILO8_stereo.rst` + Hailo-8 HEF in this repo are an **old v2.19.0
snapshot** — back then Hailo-8 was the only compiled arch. All archs report the
same 8.22 px float EPE, so the no-disparity-search defect ships on 15H too and
the teardown applies unchanged.

## The input convention, which is easy to get backwards

The ONNX takes ImageNet-normalized **NCHW float**. The model script prepends
`normalization` layers, so from `optimize` onward the network takes **raw uint8
NHWC 0..255** — the convention the calibration set is stored in. The emulator
therefore needs different data per context:

| context | input |
|---|---|
| `SDK_NATIVE` | normalized NHWC float (parsed graph, no normalization layer yet) |
| `SDK_FP_OPTIMIZED`, `SDK_QUANTIZED` | raw uint8 NHWC |

Feeding normalized data to the quantized context produces a garbage EPE that
looks exactly like a quantization failure. `dfc_flow.py emulate` picks per
context rather than trusting a flag.

The constants match: `.alls` uses `[123.675, 116.28, 103.53] / [58.395, 57.12,
57.375]`, which is `src/data.py`'s `MEAN`/`STD` × 255.

## Measured

Everything below was produced on this machine on 2026-09-10 with DFC 5.4.0,
via `./deploy/hailo-py deploy/dfc_flow.py ...`. Raw numbers in
`build/emulator_results.json`.

### Accuracy: the software model

40-pair KITTI val split, the same scenes `src/eval_kitti.py` scores.

| context | EPE masked | EPE official | D1 official |
|---|---|---|---|
| torch float (reference) | 1.464 px | 1.659 px | 9.92% |
| `SDK_NATIVE` | **1.464 px** | **1.659 px** | **9.92%** |
| `SDK_FP_OPTIMIZED` | **1.464 px** | **1.659 px** | **9.92%** |
| `SDK_QUANTIZED` | 1.654 px | 1.848 px | 11.92% |

**Both float contexts reproduce PyTorch exactly.** That is the result that
matters most, because it is the one that could have silently been wrong: it
proves the ONNX translation, the on-chip `normalization` layers, and the NHWC
input layout are all correct. `SDK_NATIVE` runs the parsed graph on normalized
input; `SDK_FP_OPTIMIZED` runs the post-model-script graph on raw uint8. They
agree to three decimals, so the normalization moved on-chip without changing
the arithmetic.

**int8 costs +0.190 px masked, or +13.0%** — against the +24.6% that
`src/quantize_sim.py` predicted.

The simulation was not badly built; it modelled something that no longer
happens. It assumed uniform int8 with percentile calibration recovered by
equalization and bias correction. DFC 5.4.0 at `optimization_level=2` logs
`Bias Correction skipped` and `Adaround skipped`, and runs
**Quantization-Aware Fine-Tuning** instead — gradient distillation against the
float model, converging here to a distill loss of 0.0385. Different mechanism,
roughly half the damage. `hailo_stereo.alls`'s justification for level 2
("runs equalization and bias correction") is therefore wrong on this DFC
version, though the decision it argued for was right.

### Against the model this replaces

| | Model Zoo stereonet | HailoStereo |
|---|---|---|
| KITTI EPE, float | 8.22 px | **1.659 px** |
| KITTI EPE, int8 | 10.4 px (on device) | **1.848 px** (emulated) |
| HEF size | 8.74 MB | **4.69 MB** |

**5.6x better** on the official protocol. Both were compiled by DFC **5.4.0**
— the same release that produced `../artifacts/stereonet_hailo15h_v5.4.0.hef`
— so this is a like-for-like toolchain comparison, not a compiler-generation
artefact.

One caveat kept explicit: 1.848 px is the *emulated* int8 figure and 10.4 px is
Hailo's *on-device* figure. The emulator is bit-accurate by design, but that
equivalence is unverified here because there is no 15H board on this machine.

### Compilation

`artifacts/hailo_stereo_hailo15h.hef`, 4.69 MB, **5 contexts**, 13m 15s.

The default 60% utilization does not compile:

    Resources presolve failed: lcus=(180/80)
    context hailo_stereo_context_6 shmifo in capacity exceeded
    (available: 20, required: 37)

180 LCUs against the 15H's 80 forces a multi-context split, and at 60% the
splitter produced nine contexts with the cost volume straddling a boundary.
shmifos are the inter-context streams: 24 shifted slices feeding one concat
means 24 edges crossing, against a hard limit of 20.
`resources_param(max_utilization=0.95)` packs five denser contexts and keeps
the cost volume intact.

This is the teardown's own observation arriving on hardware. The cost volume is
~0.0% of the MAC budget and pure memory traffic; README.md predicted "MAC
counts do not predict its cost, which is worth remembering when the graph
reaches a dataflow NPU." It is not what costs compute — it is what breaks the
allocator.

### Not measured: FPS and latency

**Unresolved.** `hailo profiler` cannot report on this HEF:

    conv layer hailo_stereo/conv78_sd0
    (translated from /refine4/blocks/blocks.2/conv2/Conv)
    with element-wise addition requires the output_shape of conv and of the
    add to be equal
    add_output_shape='[-1, 92, 308, 32]', conv_output_shape=[-1, 92, 40, 32]

To reach 95% utilization the allocator spatially defuses wide layers into
`*_sd<N>` shards (conv77 became 12). The estimator then compares a shard's
output shape against the *unsplit* element-wise add of its residual block and
raises. This is a reporting bug in DFC 5.4.0, not a defect in the compiled
model — allocation, kernel compilation and the HEF all succeed.

Three workarounds were tried; none works:

- profiling `hailo_stereo_opt.har` (pre-allocation) succeeds but reports
  `Mapped graph data is missing` for anything allocation-dependent, so no FPS
- `allocator_param(width_splitter_defuse=disabled)` is accepted but the shards
  remain: `_sd` is *spatial* defuse, a different mechanism
- `max_utilization=0.8`, on the theory that less packing means less splitting,
  does the opposite: **146 shards instead of 97**, conv78 split 12 ways, a
  40m32s compile instead of 13m15s, and a 5.80 MB HEF instead of 4.69 MB.
  Same crash.

That last result closes the search. Utilization cannot be lowered to avoid the
split (0.60 fails to allocate at all, 0.80 splits more) and cannot be raised
past 0.95 meaningfully. Every allocation this graph admits on a 15H spatially
defuses a residual conv that has a fused element-wise add, and the DFC 5.4.0
estimator cannot describe one.

So the 16.7 FPS target is unmeasured, and not measurable with this toolchain.
It needs on-device timing with HailoRT, or a DFC release that fixes the
estimator. **This does not affect the HEF**, which is valid and complete.

### Which HEF is shipped

`artifacts/hailo_stereo_hailo15h.hef` is the **max_utilization=0.95** build:
4.69 MB, 5 contexts, 13m15s, sha256 `72895ed3...062189f4`. The 0.8 experiment
was discarded — it was worse on size, compile time and shard count, with no
compensating benefit. `deploy/build/hailo_stereo_hailo15h.WORKING.hef` is a
byte-identical backup.

`deploy/build/hailo_stereo_compiled.har` was deleted rather than left stale:
it belonged to the discarded 0.8 build. Re-run `compile` to regenerate one
matching the shipped HEF.

## No device on this machine

There is no Hailo PCIe/M.2 card and no `hailort` here, so `dfc_flow.py`
stops at the HEF and the profiler's static estimate. Real on-device latency and
the final HW EPE need the HEF copied to a 15H board.
