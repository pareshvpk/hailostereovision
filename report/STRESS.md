# HailoStereo — Emulator Soak and Stress Test

**Date:** 2026-09-11
**Target:** Hailo-15H, **emulated**. There is no 15H board on this machine, so
"running the model" means the Dataflow Compiler's `SDK_QUANTIZED` context over
`deploy/build/hailo_stereo_opt.har` — the same int8 software model that produced
the 1.848 px figure in `deploy/README.md`, driven by DFC 5.4.0.
**Duration:** 30 minutes continuous, ~130 inferences at ~13 s each.
**Harness:** `notes/stress_emulator.py`
**Raw output:** `deploy/build/stress_log.jsonl` (one JSON line per inference),
`deploy/build/stress_summary.json`, `deploy/build/stress_run.log`

---

## Why this is not just another accuracy run

The 40-pair `emulate` run proves the model is *accurate on data that looks like
its training set*. It says nothing about what the graph does when handed
something degenerate, whether it stays numerically sane across hundreds of
consecutive inferences, or whether it is deterministic.

That matters because **a dataflow NPU has no exception handler**. A NaN on device
is a corrupt frame, not a traceback; a value outside the `[0, 192]` clamp is a
distance reading no downstream consumer will question. So every inference here is
checked for NaN/Inf, clamp violations, shape drift, bitwise reproducibility,
latency creep and RSS growth — independent of EPE.

---

## Verdict

| check | result |
|---|---|
| Crashes / exceptions | **0** across every inference |
| NaN / Inf outputs | **0** |
| Clamp violations (outside `[0, 192]` px) | **0** |
| Output shape drift | **0** — `(368, 1232) float32` throughout |
| Determinism | **bitwise identical**, max abs diff `0.000e+00` px |
| Accuracy vs float on real data | +6.9% int8 cost, as expected |
| Geometry recovery vs float | tracks float to **0.87 px worst case** |
| Photometric robustness | survives inversion, channel loss, 30% impulse noise |
| **Failure-to-abstain** | **systematic — see §5, the one real finding** |

The compiled int8 graph is numerically sound and faithful to the float model.
The defect this run surfaced is in the **model's behaviour on unmatchable
input**, and it is inherited from the float model, not introduced by
quantization or compilation.

---

## 1. Smoke — real KITTI, scored

Eight real val pairs through the quantized emulator, scored against ground truth.

| | masked | official |
|---|---|---|
| torch float, same 8 pairs | 2.235 px | 2.669 px |
| **int8 emulator, same 8 pairs** | **2.389 px** | **2.822 px** |
| int8 cost on this subset | **+0.154 px (+6.9%)** | +0.153 px |

The absolute numbers are well above the published 1.654 / 1.848 px, and that is
**not a regression** — these first eight scenes are genuinely harder than the
split average (2.235 px in float, against 1.464 px over all 40; `000161_10`
alone reads 6.85 px official). The float baseline was re-measured on exactly the
same eight frames to establish this rather than assume it.

This is the trap `dfc_flow.py` documents in a comment: an earlier version of that
script reported "+98.7% int8 cost" on 5 pairs when the true cost was +5.5%.
**Quotable int8 cost remains the 40-pair +13.0%**; the +6.9% here is a
subset figure and is not comparable.

## 2. Determinism

The same pair inferred three times in the same context:

```
3 repeats, bitwise identical: True, max |diff| 0.000e+00 px
```

Not "within tolerance" — **bitwise**. This matters because the emulator's output
is the thing a board is expected to reproduce; a non-deterministic reference
would make any on-device comparison meaningless.

## 3. Geometry — known shift in, known disparity out

A synthetic right view is built from a real left image at constant disparity d
(`right[:, j] = left[:, j+d]`, matching the `I_L(x) = I_R(x-d)` convention
`src/check_ingest.py` warps with). The prediction should recover d. This is the
test the original Model Zoo model fails — run here on the **quantized** graph
rather than in torch.

| true shift | float median | int8 median | Δ | int8 within 2 px |
|---|---|---|---|---|
| 0 px | 57.83 | 57.44 | −0.39 | 0.0% |
| 8 px | 11.49 | 12.09 | +0.60 | 40.9% |
| **16 px** | 16.59 | **15.72** | −0.87 | **88.2%** |
| **32 px** | 32.70 | **32.65** | −0.05 | **96.9%** |
| **64 px** | 65.87 | **65.30** | −0.57 | **80.2%** |
| 96 px | 93.71 | 94.33 | +0.62 | 53.5% |
| 128 px | 112.47 | 112.47 | 0.00 | 1.5% |

**Two separate readings here, and they must not be conflated.**

**(a) The compiled graph is faithful.** Max divergence from float is 0.87 px,
mean 0.44 px, across a 0–128 px sweep — including the two hypotheses where the
model is *wrong*. int8 reproduces the float model's correct answers and its
mistakes alike. Quantization is not distorting the matcher.

**(b) The model's usable envelope is roughly 16–64 px** and degrades at both
ends. The d=0 row is the interesting one and is treated in §5.

*Caveat, stated rather than buried:* a constant-shift warp of a real photograph
is geometrically impossible — a fronto-parallel plane at fixed depth carrying
perspective texture that contradicts it — so the monocular refinement path is
actively fighting the matching path. These numbers **bound** the envelope; they
are not a clean matcher-only measurement.

One oddity, flagged but not claimed: int8 scores *better* than float on
within-2px at 64 px (80.2% vs 56.1%) and 96 px (53.5% vs 40.8%). Consistent with
quantization noise smoothing a bimodal output, but that is a guess on one image.

## 4. Adversarial — 20 degenerate inputs

None crashed. None produced a non-finite. None violated the clamp.

### Photometric robustness is excellent

Each of these plants a known d = 24 px and then abuses the images:

| case | median | within 2 px |
|---|---|---|
| vertically flipped | 24.19 | 93.4% |
| 30% salt-and-pepper noise | 23.58 | 93.9% |
| contrast ÷ 8 | 24.19 | 90.4% |
| fully inverted (255 − x) | 24.19 | 85.6% |
| red channel only | 24.19 | 80.4% |
| saturated +200 | 24.79 | 69.5% |
| contrast × 0.04 | 26.61 | 36.5% |

Inversion, two-thirds channel loss and 30% impulse noise barely move the
estimate. The feature tower is matching structure, not brightness — which is
what a stereo feature extractor is supposed to do, and direct evidence that the
matching path is real.

Only the extreme low-contrast case (×0.04, a 10-level dynamic range) degrades
meaningfully, and even then the median is within 2.6 px.

### The failure mode, and it is uniform

| case | true disparity | model output |
|---|---|---|
| all-black | undefined | mean **105.5** px |
| all-white | undefined | mean **103.7** px |
| mid-gray | undefined | mean **101.5** px |
| vertical ramp | 0 | median **100.37** px |
| horizontal ramp | 0 | median **91.30** px |
| identical noise | 0 | median **81.63** px |
| identical real image | 0 | median **57.44** px |

See §5.

## 5. The finding: the model cannot say "far" or "unknown"

Collect the degenerate cases from §4 with the d=0 row from §3 and one pattern
appears in all of them. **Whenever there is nothing to match, the model reports
something close.**

On a featureless frame it emits ~100 px — about **3.9 m** at KITTI's
f·B = 386.5 px·m — with a standard deviation of ~13 px. That is not a flat
default value or a degenerate constant; it is *structured-looking garbage*, the
kind of output that survives a plausibility check.

This unifies three observations that were previously filed separately:

1. the **above-horizon drift** documented in `src/eval_depth.py` ("the model
   reports sky at high disparity"),
2. the **d = 0 geometry failure** in §3, and
3. the **textureless behaviour** in §4.

They are one defect, not three. Zero disparity means infinite distance, and the
model has no way to express it. The root cause is visible in the training setup:
KITTI LiDAR ground truth never returns from the sky, so the region was never
supervised, and genuine zero-disparity content does not occur in the training
distribution at all. The network was never given a reason to learn "far".

**Severity.** The error direction is the dangerous one. A stereo front-end that
reports empty or untextured space as an obstacle at 3.9 m will cause spurious
braking, not missed detection. Textureless surfaces — an unmarked wall, a
uniformly lit road, fog, an overexposed sky — are common, not exotic.

**This is inherited from the float model.** Verified directly: the float model
returns 57.83 px on identical-image input where the quantized graph returns
57.44 px. Quantization and compilation are not the cause and fixing the `.alls`
will not help.

### The compounding problem: the HEF exposes no confidence

```
ONNX inputs : [('left',  [1, 3, 368, 1232]),
               ('right', [1, 3, 368, 1232])]
ONNX outputs: [('disparity', [1, 1, 368, 1232])]
```

**One output.** The match-confidence map — softmax entropy over the 24
hypotheses — exists only in `src/demo_server.py`, computed in torch for the
inspector panel. It was never exported, so it is not in the ONNX and not in the
HEF.

The consequence on device: a consumer receives "3.9 m" and has no way to
distinguish a confident match from a featureless frame. The information needed
to reject these readings **is computed inside the graph and then discarded**.

The fix is cheap. The softmax probability volume already exists as an
intermediate tensor; entropy is a reduce over it, and the export would add one
output of 368×1232 at negligible compute. `demo_server.py` already demonstrates
that the entropy map cleanly separates matched from unmatched regions — it is
the "Match confidence" panel, described there as "textureless road and sky are
uncertain, edges are sharp."

## 6. Soak — 30 minutes continuous

| | |
|---|---|
| Total inferences | **121** (121 completed, **0 raised**) |
| Latency | mean **14.70 s**, min 12.49, max 27.35, p95 22.40 |
| Latency drift | first 10 **21.37 s** → last 10 **13.63 s** (**−36.2%**) |
| RSS | 3,654 → 4,500 MB (+845 MB over 121 inferences) |
| Non-finite outputs | **0 NaN, 0 Inf** |
| Clamp violations | **0** outside [0, 192] px |
| Scored inferences | 53, mean EPE 2.518 px |

**Latency improves rather than degrades.** The −36.2% is TensorFlow/XLA warm-up
on the first inferences (the very first cost 21.9 s in an earlier probe), after
which it settles at a flat 13.3–13.9 s for the rest of the run. No thermal or
allocator degradation over half an hour.

**RSS plateaus.** The +845 MB is almost entirely front-loaded; across the soak
the last ten readings oscillate between 4,435 and 4,500 MB, and RSS **fell**
from 4,475 to 4,435 MB at one point. Memory is being reclaimed, so this is cache
fill, not a leak.

**Sustained determinism — the strongest result in the run.** Six scenes were
each inferred **seven times** spread across the full 30 minutes, interleaved with
adversarial inputs:

| scene | repeats | EPE spread |
|---|---|---|
| 000160_10 | 7 | **0.00e+00** |
| 000161_10 | 7 | **0.00e+00** |
| 000162_10 | 7 | **0.00e+00** |
| 000163_10 | 7 | **0.00e+00** |
| 000164_10 | 7 | **0.00e+00** |
| 000165_10 | 7 | **0.00e+00** |

Forty-two scored inferences, zero variance. The graph does not drift, does not
accumulate state between inferences, and is not perturbed by the degenerate
inputs cycled between them. §2's three-repeat check proved determinism; this
proves it *holds under sustained mixed load*.

## 7. Failures logged

Three, all the same finding, all reproduced in float:

```
[geometry] shift=0:   median  57.44 != 0
[geometry] shift=8:   median  12.09 != 8
[geometry] shift=128: median 112.47 != 128
```

These are **model envelope**, not graph defects — §3(a) shows int8 tracks float
to 0.87 px worst case on exactly these cases. No crash, no numerical fault, and
no failure attributable to quantization or compilation was found in 121
inferences.

## 8. What this does and does not establish

**Establishes:** the compiled int8 graph is numerically sound, bitwise
deterministic under sustained load, free of NaN/Inf and clamp violations across
121 inferences including 20 adversarial inputs, faithful to the float model to
under 1 px, and robust to severe photometric abuse.

**Does not establish:** anything about real hardware. This is the DFC emulator,
which is bit-accurate *by design* — but that equivalence is unverified here
because there is no 15H board on this machine. Latency figures above are
emulator wall-clock on an RTX 4060 and say **nothing** about device throughput;
the 16.7 FPS target remains unmeasured, and is unmeasurable with DFC 5.4.0
because `hailo profiler` crashes on this HEF (see `deploy/README.md`).

## 9. Recommendations

1. **Export match confidence as a second HEF output.** Highest value per unit of
   work in this report. The entropy is already computed inside the graph and
   thrown away; without it, nothing downstream can reject a fabricated distance.
2. **Supervise the unmatchable regions toward "far" rather than leaving them
   unconstrained.** The weak-border supervision already in `src/train.py` is
   exactly the right pattern (it cut border error 16×, README §"Two protocols");
   apply the same idea above the horizon and to low-texture regions.
3. **Consider clamping or flagging output where entropy is near-uniform**, once
   (1) exists — a cheap guard that turns a fabricated 3.9 m into an explicit
   no-reading.
4. **Re-run this harness on a 15H board** when one is available. The script takes
   `--context`; the emulator numbers here become the reference the device must
   reproduce, and §2/§6's bitwise determinism is what makes that comparison
   meaningful.
5. Do not act on the §3 envelope numbers without a cleaner experiment — the
   constant-shift warp confounds the matching and monocular paths. A proper
   measurement needs synthetic stereo with consistent geometry
   (`src/test_learns.py` already builds such pairs).

## 10. Reproducing

```bash
./deploy/hailo-py notes/stress_emulator.py --minutes 30
```

Exit code is non-zero if any check fails. Phases are bounded by the overall
deadline and skipped if the budget cannot reach them — the summary names which
phases actually ran, so "no failures" can never be read as "that check passed"
when the check never executed. Every inference appends one JSON line to
`deploy/build/stress_log.jsonl`, so a run that dies halfway is still readable.

`--context fp_optimized` runs the same battery against the float graph, which is
how the float/int8 comparisons in §3 were obtained.
