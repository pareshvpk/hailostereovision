"""
Drive the Hailo Dataflow Compiler end to end for HailoStereo, targeting hailo15h.

    python deploy/dfc_flow.py all

Subcommands run in order and each writes its output into deploy/build/, so a
failed step can be re-run without repeating the ones before it:

    parse      artifacts/hailo_stereo.onnx  ->  build/hailo_stereo.har
               and resolve the layer names the model script needs
    optimize   + resolved .alls + KITTI calib set -> build/hailo_stereo_opt.har
    emulate    score the software model on the KITTI val split
    compile    -> artifacts/hailo_stereo_hailo15h.hef
    profile    latency / FPS report

WHY A SCRIPT AND NOT THE CLI
The four-command CLI sequence in README.md cannot express two things this graph
needs. First, the network has TWO inputs, and `hailo optimize --calib-set-path`
takes a single path; the Python API accepts a dict keyed by input layer name,
which is the only clean way to feed a stereo pair. Second, the model script
refers to layers by name, and the parser renames every layer -- those names are
not knowable until after parsing, so resolving them has to happen between the
parse and optimize steps rather than by hand.

ON THE INPUT CONVENTION
The ONNX takes ImageNet-normalized NCHW float. The model script prepends
`normalization` layers, so from `optimize` onward the network takes RAW uint8
NHWC 0..255 instead -- the same convention the calibration set is stored in.
That means the emulator is fed different data depending on the context:

    SDK_NATIVE                    normalized NHWC float  (parsed graph only)
    SDK_FP_OPTIMIZED / QUANTIZED  raw uint8 NHWC         (normalization on-chip)

Getting this backwards produces a garbage EPE that looks like a quantization
failure, so `emulate` picks the right one per context rather than trusting a
flag.

DFC API SIGNATURES MOVE BETWEEN VERSIONS. Where a call is version-sensitive it
is wrapped with a message naming the alternative, rather than failing with a
bare TypeError.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "deploy" / "build"
ONNX = ROOT / "artifacts" / "hailo_stereo.onnx"
ALLS_SRC = ROOT / "deploy" / "hailo_stereo.alls"
ALLS_OUT = BUILD / "hailo_stereo.resolved.alls"
HAR_PARSED = BUILD / "hailo_stereo.har"
HAR_OPT = BUILD / "hailo_stereo_opt.har"
HEF = ROOT / "artifacts" / "hailo_stereo_hailo15h.hef"

# Float reference for the CURRENTLY shipped checkpoint, as a 40-pair figure:
# (EPE masked, EPE official, D1 official). Measured by src/eval_kitti.py.
# UPDATE THIS whenever the shipped checkpoint changes -- it is only a display
# and sanity-check baseline, but a stale value here reads as a real defect.
# runs/kitti_mixed_fixed768/best.pt, 2026-09-16.
TORCH_FLOAT_REF = (1.156, 1.248, 6.87)
NAMES = BUILD / "resolved_names.json"

HW_ARCH = "hailo15h"
NET_NAME = "hailo_stereo"
H, W = 368, 1232

# The ONNX node the model script promotes to 16-bit. Its output IS the
# disparity, so int8 rounding of its weights [0..23] perturbs the prediction
# directly -- see the rationale in hailo_stereo.alls.
INDEX_ONNX_NODE = "/soft_argmin/index/Conv"

# ImageNet statistics, 0..255 scale. Must equal src/data.py MEAN/STD x 255.
NORM_MEAN = [123.675, 116.28, 103.53]
NORM_STD = [58.395, 57.12, 57.375]


# ---------------------------------------------------------------------------
# data -- mirrors src/data.py Kitti2015(split="val") exactly
# ---------------------------------------------------------------------------

def kitti_val_pairs(root: pathlib.Path, limit=None):
    """The same 40 held-out scenes src/eval_kitti.py scores, as RAW uint8.

    src/data.py returns normalized tensors; the Hailo graph wants raw uint8, so
    the crop is reproduced here rather than reusing the loader. The crop must
    match to the pixel or the EPE is not comparable to the 1.659 px baseline:
    multiple of 16, anchored bottom-right.
    """
    from PIL import Image

    lefts = sorted((root / "image_2").glob("*_10.png"))
    samples = []
    for lp in lefts:
        rp, dp = root / "image_3" / lp.name, root / "disp_occ_0" / lp.name
        if rp.exists() and dp.exists():
            samples.append((lp, rp, dp))
    samples = samples[-40:]                       # n_val=40, deterministic
    if limit:
        samples = samples[:limit]

    for lp, rp, dp in samples:
        left = np.array(Image.open(lp).convert("RGB"), dtype=np.uint8)
        right = np.array(Image.open(rp).convert("RGB"), dtype=np.uint8)
        disp = np.array(Image.open(dp), dtype=np.float32) / 256.0
        h, w = disp.shape
        nh, nw = (h // 16) * 16, (w // 16) * 16
        sl = (slice(h - nh, None), slice(w - nw, None))
        yield lp.name, left[sl], right[sl], disp[sl]


def normalize(raw_u8: np.ndarray) -> np.ndarray:
    """Raw uint8 NHWC -> normalized float NHWC, matching src/data.py."""
    return (raw_u8.astype(np.float32) - np.array(NORM_MEAN, np.float32)) / \
        np.array(NORM_STD, np.float32)


def score(pred, disp):
    """EPE and D1-all under both protocols. Mirrors src/eval_kitti.py.

    masked   -- what training optimises: valid GT, in range, left border excluded
    official -- every pixel with valid ground truth
    """
    out = {}
    official = disp > 0.0
    masked = official & (disp < 192.0) & np.isfinite(disp)
    masked[:, :192] = False
    for name, m in (("masked", masked), ("official", official)):
        err = np.abs(pred - disp)[m]
        if err.size == 0:
            out[name] = (float("nan"), float("nan"), 0)
            continue
        d1 = (err > 3.0) & (err > 0.05 * np.abs(disp[m]))
        out[name] = (float(err.mean()), 100.0 * float(d1.mean()), int(err.size))
    return out


# ---------------------------------------------------------------------------
# layer-name resolution
# ---------------------------------------------------------------------------

def _hn_dict(runner):
    hn = runner.get_hn()
    return json.loads(hn) if isinstance(hn, str) else hn


def resolve_names(runner) -> dict:
    """Map ONNX-side identities onto the names the parser assigned.

    Three names are needed and none of them is knowable before parsing:
    the two input layers (the normalization commands attach to them) and the
    1x1 convolution that turns the softmax into a disparity.

    deploy/README.md flags only the third as a placeholder. The two input names
    are placeholders too -- `input_layer=left` in the handwritten .alls is both
    the wrong name and, in most DFC versions, the wrong syntax.
    """
    hn = _hn_dict(runner)
    layers = hn["layers"]

    inputs = [(n, d) for n, d in layers.items() if d.get("type") == "input_layer"]
    if len(inputs) != 2:
        raise RuntimeError(
            f"expected 2 input layers, parser produced {len(inputs)}: "
            f"{[n for n, _ in inputs]}")

    # Match by the ONNX tensor each input came from; fall back to declared
    # order, which follows start_node_names.
    def side_of(name, d):
        orig = " ".join(d.get("original_names") or []) + " " + name
        has_l, has_r = "left" in orig.lower(), "right" in orig.lower()
        return "left" if has_l and not has_r else "right" if has_r and not has_l else None

    sides = {}
    for n, d in inputs:
        s = side_of(n, d)
        if s and s not in sides:
            sides[s] = n
    if set(sides) != {"left", "right"}:
        ordered = [n for n, _ in inputs]
        print(f"  ! could not tell the inputs apart by name; assuming parse "
              f"order left={ordered[0]} right={ordered[1]}")
        sides = {"left": ordered[0], "right": ordered[1]}

    # The index conv: prefer the ONNX node name the parser recorded.
    index = None
    for n, d in layers.items():
        if INDEX_ONNX_NODE in (d.get("original_names") or []):
            index = n
            break
    if index is None:
        # Fall back on shape: 1x1, 24 in, 1 out is unique in this graph.
        cands = []
        for n, d in layers.items():
            if d.get("type") != "conv":
                continue
            ks = (d.get("params") or {}).get("kernel_shape")
            if ks and list(ks[:2]) == [1, 1] and int(ks[2]) == 24 and int(ks[3]) == 1:
                cands.append(n)
        if len(cands) == 1:
            index = cands[0]
            print(f"  ! {INDEX_ONNX_NODE} not in original_names; matched by "
                  f"shape (1x1, 24->1) instead")
        else:
            raise RuntimeError(
                f"could not identify the soft-argmin index conv: "
                f"{len(cands)} candidates {cands}. Inspect "
                f"{HAR_PARSED} and set it by hand in {ALLS_OUT}.")

    return {"left": sides["left"], "right": sides["right"], "index": index}


def write_alls(names: dict, args) -> None:
    """Emit the model script with real layer names.

    Generated rather than string-patched: the handwritten hailo_stereo.alls is
    the record of WHY these three decisions were made and stays authoritative
    for that. This file is the executable form of the same decisions.
    """
    lvl = args.opt_level
    ft = ""
    if lvl >= 2:
        # QFT OOMs on an 8 GB laptop 4060 at the default batch size: this graph
        # trains on 368x1232 stereo PAIRS, so one sample is two full-resolution
        # images plus a 24-hypothesis cost volume. batch_size=1 is not a
        # tuning preference, it is what fits.
        #
        # dataset_size is clamped to the calibration set (64 pairs). The
        # default is far larger and QFT would silently reuse frames to reach
        # it -- fine, but it makes the epoch count mean something other than
        # what it says.
        ft = (f"\n# 4. QFT batch size. The default exhausts 8 GB on 368x1232 "
              f"stereo pairs.\npost_quantization_optimization(finetune, "
              f"policy=enabled, batch_size={args.finetune_batch}, "
              f"dataset_size={args.finetune_dataset})\n")

    ALLS_OUT.write_text(f"""\
# GENERATED by deploy/dfc_flow.py -- do not edit.
# Rationale for all three decisions: deploy/hailo_stereo.alls
# Layer names resolved from the parsed graph on {time.strftime('%Y-%m-%d %H:%M')}.

# 1. Normalization on-chip. The chip takes raw uint8; both branches of a
#    siamese tower must be scaled identically.
normalization_left = normalization({NORM_MEAN}, {NORM_STD}, {names['left']})
normalization_right = normalization({NORM_MEAN}, {NORM_STD}, {names['right']})

# 2. The index ramp stays at 16-bit. Its output IS the disparity, so int8
#    rounding of its [0..23] weights perturbs the prediction directly.
#    Cost: 24 MACs/pixel, 0.0% of the 14.21 GMAC budget.
#    ONNX node {INDEX_ONNX_NODE} -> {names['index']}
quantization_param({{{names['index']}}}, precision_mode=a16_w16)

# 3. Optimization level. NOTE: level 2 does more than the hand-written
#    rationale assumed -- on DFC 5.4.0 it adds quantization-aware fine-tuning
#    (QFT), a gradient-based distillation pass on top of the equalization and
#    bias correction. QFT is what makes this step expensive.
model_optimization_flavor(optimization_level={lvl}, compression_level=0)
{ft}""")
    print(f"  wrote {ALLS_OUT.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def step_parse(args):
    from hailo_sdk_client import ClientRunner

    print(f"[parse] {ONNX.relative_to(ROOT)} -> {HW_ARCH}")
    runner = ClientRunner(hw_arch=HW_ARCH)
    runner.translate_onnx_model(
        str(ONNX), NET_NAME,
        start_node_names=["left", "right"],
        end_node_names=["disparity"],
        net_input_shapes={"left": [1, 3, H, W], "right": [1, 3, H, W]},
    )
    BUILD.mkdir(parents=True, exist_ok=True)
    runner.save_har(str(HAR_PARSED))
    print(f"  wrote {HAR_PARSED.relative_to(ROOT)}")

    names = resolve_names(runner)
    NAMES.write_text(json.dumps(names, indent=2))
    for k, v in names.items():
        print(f"  {k:6s} -> {v}")
    write_alls(names, args)

    layers = _hn_dict(runner)["layers"]
    print(f"  {len(layers)} layers parsed")
    return runner


def _calib():
    """Calibration set as a dict keyed by input layer name.

    This is the two-input problem deploy/README.md leaves open: the CLI's
    --calib-set-path takes one path and this graph has two inputs. The Python
    API takes a dict, so both arrays go in together and stay paired -- which
    matters, because a stereo network calibrated on mismatched left/right
    frames sees disparities that do not exist.
    """
    names = json.loads(NAMES.read_text())
    left = np.load(ROOT / "artifacts" / "calib_kitti_left.npy")
    right = np.load(ROOT / "artifacts" / "calib_kitti_right.npy")
    assert left.shape == right.shape == (64, H, W, 3), (left.shape, right.shape)
    assert left.dtype == right.dtype == np.uint8, (left.dtype, right.dtype)
    print(f"  calib: {left.shape[0]} KITTI pairs, raw uint8 NHWC")
    return {names["left"]: left, names["right"]: right}


def step_optimize(args):
    from hailo_sdk_client import ClientRunner

    print(f"[optimize] {HAR_PARSED.relative_to(ROOT)}")
    runner = ClientRunner(har=str(HAR_PARSED), hw_arch=HW_ARCH)
    runner.load_model_script(str(ALLS_OUT))
    runner.optimize(_calib())
    runner.save_har(str(HAR_OPT))
    print(f"  wrote {HAR_OPT.relative_to(ROOT)}")
    return runner


def step_emulate(args):
    """Score the software model. This is the number the whole exercise is for."""
    from hailo_sdk_client import ClientRunner, InferenceContext

    # The InferenceContext enum has gained and lost members across DFC
    # releases -- SDK_FP_OPTIMIZED in particular is not present in every
    # version. Resolve by name so a missing one is a skipped context with a
    # clear message, not an AttributeError before any work starts.
    spec = {
        "native": ("SDK_NATIVE", HAR_PARSED, False),
        "fp_optimized": ("SDK_FP_OPTIMIZED", HAR_OPT, True),
        "quantized": ("SDK_QUANTIZED", HAR_OPT, True),
    }
    contexts = {}
    for tag, (attr, har, raw) in spec.items():
        enum = getattr(InferenceContext, attr, None)
        if enum is None:
            print(f"  ! this DFC has no InferenceContext.{attr}; "
                  f"'{tag}' unavailable. Have: "
                  f"{[x.name for x in InferenceContext]}")
            continue
        contexts[tag] = (enum, har, raw)

    wanted = [c for c in (args.contexts or ["fp_optimized", "quantized"])
              if c in contexts]
    if not wanted:
        raise RuntimeError("no usable inference contexts on this DFC version")
    root = ROOT / "data" / "kitti2015" / "training"
    pairs = list(kitti_val_pairs(root, limit=args.limit))
    print(f"[emulate] {len(pairs)} KITTI val pairs, contexts: {', '.join(wanted)}")

    results = {}
    for tag in wanted:
        ctx_enum, har, raw_input = contexts[tag]
        if not har.exists():
            print(f"  {tag}: SKIP -- {har.relative_to(ROOT)} missing")
            continue
        runner = ClientRunner(har=str(har), hw_arch=HW_ARCH)
        names = json.loads(NAMES.read_text())
        acc = {"masked": [0.0, 0.0, 0], "official": [0.0, 0.0, 0]}
        t0 = time.time()
        with runner.infer_context(ctx_enum) as ctx:
            for i, (name, left, right, disp) in enumerate(pairs):
                l = left[None].astype(np.uint8) if raw_input else normalize(left[None])
                r = right[None].astype(np.uint8) if raw_input else normalize(right[None])
                out = runner.infer(ctx, {names["left"]: l, names["right"]: r})
                pred = np.squeeze(out[0] if isinstance(out, (list, tuple)) else out)
                if pred.shape != disp.shape:
                    raise RuntimeError(
                        f"emulator returned {pred.shape}, expected {disp.shape}")
                s = score(pred, disp)
                for k in acc:
                    e, d, n = s[k]
                    if n:
                        acc[k][0] += e * n
                        acc[k][1] += d * n
                        acc[k][2] += n
                if (i + 1) % 10 == 0:
                    print(f"    {i+1}/{len(pairs)}  running masked EPE "
                          f"{acc['masked'][0]/max(acc['masked'][2],1):.3f} px")
        results[tag] = {k: (v[0] / v[2], v[1] / v[2]) for k, v in acc.items() if v[2]}
        print(f"  {tag}: {time.time()-t0:.0f}s")

    full = args.limit is None
    print(f"\n{'context':14s} {'EPE masked':>12s} {'EPE official':>14s} {'D1 official':>13s}")
    # TORCH_FLOAT_REF is a 40-pair figure. Printing it next to a
    # --limit run invites exactly the wrong conclusion: an earlier version of
    # this script reported "+98.7% int8 cost" on 5 pairs when the true cost was
    # +5.5%, because the first 5 KITTI val scenes are harder than the split
    # average (2.757 px vs 1.463 px in float). Only show it when comparable.
    if full:
        print(f"{'torch float':14s} {TORCH_FLOAT_REF[0]:11.3f}p "
              f"{TORCH_FLOAT_REF[1]:13.3f}p {TORCH_FLOAT_REF[2]:12.2f}%"
              f"   <- 40-pair reference")
    for tag, r in results.items():
        print(f"{tag:14s} {r['masked'][0]:11.3f}p {r['official'][0]:13.3f}p "
              f"{r['official'][1]:12.2f}%")
    if not full:
        print(f"\n(torch float reference omitted: it is a 40-pair figure and "
              f"this run scored {args.limit}.)")

    # Quantization cost is measured against the float path THIS RUN produced,
    # never against a stored constant -- that keeps it correct at any --limit
    # and isolates int8 damage from any float-side discrepancy.
    if "quantized" in results and "fp_optimized" in results:
        q, f = results["quantized"]["masked"][0], results["fp_optimized"]["masked"][0]
        print(f"\nint8 cost vs the emulator's own float: {q - f:+.3f} px masked "
              f"({100.0 * (q - f) / f:+.1f}%).")
        print(f"src/quantize_sim.py predicted +24.6% (uniform int8, percentile "
              f"calibration, no QFT modelled).")
        if full and abs(f - TORCH_FLOAT_REF[0]) > 0.01:
            print(f"! WARNING: the float context reads {f:.3f} px, not "
                  f"{TORCH_FLOAT_REF[0]:.3f}. Either TORCH_FLOAT_REF is stale "
                  f"(did the checkpoint change?) or suspect the normalization "
                  f"layers or the NHWC input layout, not quantization.")
    elif "quantized" in results:
        print("\n(no fp_optimized run: int8 cost needs both contexts)")

    if "quantized" in results:
        q = results["quantized"]["official"][0]
        print(f"\nModel Zoo v5.4.0 hailo15h: 8.22 px float / 10.4 px on device.")
        print(f"this model, int8 emulated, official protocol: {q:.3f} px")
        print(f"PASS -- {10.4/q:.1f}x better than the on-device competitor."
              if q < 10.4 else "FAIL -- no better than the model this replaces.")
    (BUILD / "emulator_results.json").write_text(json.dumps(results, indent=2))


ALLS_COMPILE = BUILD / "hailo_stereo.compile.alls"


def write_compile_alls(args) -> None:
    """Allocator settings, kept separate from the optimization model script.

    These only affect compilation, so putting them here means retuning the
    allocator costs a 15-minute compile rather than also redoing the six-minute
    optimize.

    WHY UTILIZATION IS RAISED. At the 60% default the splitter cut the graph
    into nine contexts and failed:

        context hailo_stereo_context_6 shmifo in capacity exceeded
        (available: 20, required: 37)

    shmifos are the inter-context streams. The cost volume is 24 shifted slices
    feeding one concat -- if the slices are produced in one context and consumed
    in the next, all 24 must cross as separate streams and blow the 20 limit.
    Denser contexts mean fewer boundaries and a better chance the whole cost
    volume lands in a single context. Note this is the same structure the
    teardown flagged as pure memory traffic with a negligible MAC count: it is
    cheap to compute and expensive to place, and here it is what breaks the
    allocator.

    Single-context is not an option -- presolve reports lcus=(180/80), so the
    graph needs at least three contexts on a 15H no matter what.
    """
    # MUTUALLY EXCLUSIVE. performance_param drives the "performance flow",
    # which insists on choosing utilization itself:
    #     Compilation failed: Performance Flow requires automatic resource
    #     utilization
    # So an explicit max_utilization and a compiler_optimization_level cannot
    # both be set. max_utilization is the one that addresses the shmifo
    # overflow, so it wins by default and compiler effort is opt-in.
    lines = ["# GENERATED by deploy/dfc_flow.py -- do not edit.",
             "# Allocator settings only; see write_compile_alls() for the reasoning."]
    if args.compiler_effort:
        lines.append(f"performance_param(compiler_optimization_level={args.compiler_effort})")
        note = f"compiler_optimization_level={args.compiler_effort}, utilization automatic"
    else:
        lines.append(f"resources_param(max_utilization={args.max_util})")
        note = f"max_utilization={args.max_util}"

    # Width splitting is how the allocator fits wide layers into clusters, and
    # raising utilization makes it split aggressively -- conv77 came back as 12
    # shards. The resulting `*_sd<N>` layers are legal and the HEF is fine, but
    # they break `hailo profiler` on the compiled HAR:
    #
    #   conv layer hailo_stereo/conv78_sd0 with element-wise addition requires
    #   the output_shape of conv and of the add to be equal
    #   add_output_shape='[-1, 92, 308, 32]', conv_output_shape=[-1, 92, 40, 32]
    #
    # The estimator compares a shard's shape against the unsplit ew_add and
    # throws. That is a DFC reporting bug, not a defect in the compiled model.
    # Disabling the width splitter avoids the shards so a performance estimate
    # can be produced -- at the risk that allocation no longer fits.
    if args.no_width_split:
        lines.append("allocator_param(width_splitter_defuse=disabled)")
        note += ", width_splitter_defuse=disabled"
    ALLS_COMPILE.write_text("\n".join(lines) + "\n")
    print(f"  wrote {ALLS_COMPILE.relative_to(ROOT)} ({note})")


def step_compile(args):
    from hailo_sdk_client import ClientRunner

    print(f"[compile] {HAR_OPT.relative_to(ROOT)} -> {HW_ARCH}")
    runner = ClientRunner(har=str(HAR_OPT), hw_arch=HW_ARCH)
    write_compile_alls(args)
    runner.load_model_script(str(ALLS_COMPILE))
    t0 = time.time()
    hef = runner.compile()
    HEF.write_bytes(hef)
    print(f"  wrote {HEF.relative_to(ROOT)} "
          f"({HEF.stat().st_size/1e6:.2f} MB) in {time.time()-t0:.0f}s")
    runner.save_har(str(BUILD / "hailo_stereo_compiled.har"))


def step_profile(args):
    har = BUILD / "hailo_stereo_compiled.har"
    har = har if har.exists() else HAR_OPT
    out = BUILD / "profile.html"
    print(f"[profile] {har.relative_to(ROOT)}")
    # The `hailo` console script lives in the DFC venv's bin/, which is NOT on
    # PATH when this runs via deploy/hailo-py (that wrapper invokes the venv
    # interpreter directly rather than activating the venv). Resolve it next to
    # the running interpreter, falling back to PATH for an activated shell.
    exe = pathlib.Path(sys.executable).parent / "hailo"
    cmd = [str(exe) if exe.exists() else "hailo",
           "profiler", str(har), "--out-path", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        print("  profiler failed -- try: hailo profiler "
              f"{har.relative_to(ROOT)}")
        return
    print(f"  wrote {out.relative_to(ROOT)}")
    for line in r.stdout.splitlines():
        if any(k in line.lower() for k in ("fps", "latency", "utilization")):
            print(f"  {line.strip()}")
    print("  compare: Model Zoo v5.4.0 hailo15h is 16.7 FPS at batch 1")


STEPS = {
    "parse": step_parse, "optimize": step_optimize, "emulate": step_emulate,
    "compile": step_compile, "profile": step_profile,
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=list(STEPS) + ["all"])
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N val pairs (smoke test)")
    p.add_argument("--contexts", nargs="*",
                   choices=["native", "fp_optimized", "quantized"],
                   help="emulate: which precisions to score")
    p.add_argument("--opt-level", type=int, default=2, choices=[0, 1, 2],
                   help="parse: optimization_level. 2 adds quantization-aware "
                        "fine-tuning (memory hungry); 1 stops after "
                        "equalization and bias correction")
    p.add_argument("--finetune-batch", type=int, default=1,
                   help="parse: QFT batch size. 1 is what fits in 8 GB here")
    p.add_argument("--finetune-dataset", type=int, default=64,
                   help="parse: QFT dataset size; the calib set has 64 pairs")
    p.add_argument("--max-util", type=float, default=0.95,
                   help="compile: resources_param max_utilization. The 60%% "
                        "compiler default splits into 9 contexts and overflows "
                        "the 20-shmifo limit on the cost volume. WARNING: as of "
                        "2026-09-16 the 0.95 setting no longer completes on this "
                        "graph -- 3 runs of 41-88 min all stalled at context 3/5, "
                        "34 of 46 allocator failures being shmifo overflow. Use "
                        "--compiler-effort 0 instead (5m52s, 7 contexts, 4.12 MB)")
    p.add_argument("--no-width-split", action="store_true",
                   help="compile: disable width_splitter_defuse. Avoids the "
                        "*_sd<N> shards that crash `hailo profiler` on the "
                        "compiled HAR; may make allocation infeasible")
    p.add_argument("--compiler-effort", default=None,
                   help="compile: performance_param compiler_optimization_level. "
                        "Mutually exclusive with --max-util; setting this "
                        "hands utilization back to the compiler")
    args = p.parse_args()

    try:
        import hailo_sdk_client  # noqa: F401
    except ImportError:
        sys.exit("hailo_sdk_client not importable -- the Dataflow Compiler is "
                 "not installed in this interpreter.\n"
                 "See deploy/README.md for the install sequence.")

    BUILD.mkdir(parents=True, exist_ok=True)
    order = ["parse", "optimize", "emulate", "compile", "profile"] \
        if args.step == "all" else [args.step]
    for name in order:
        STEPS[name](args)
        print()


if __name__ == "__main__":
    main()
