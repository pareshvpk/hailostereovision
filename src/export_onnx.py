"""
Export HailoStereo to ONNX and audit the exported graph against the defects
found in the Model Zoo stereonet teardown.

The audit is the point. An export that produces a valid ONNX file but
reintroduces a 5D tensor, a Conv3d, or a materialised index ramp is a failed
export, and this script says so rather than leaving it for the compiler.

    python src/export_onnx.py                          # structure audit only
    python src/export_onnx.py --ckpt runs/sceneflow/best.pt

Without --ckpt the weights are random. The graph audit is still valid -- it is
a structural check and does not care what the weights are -- but the resulting
file is a mock, not something to hand to the compiler, and it is labelled as
one in the output. The checkpoint loads STRICTLY (EXP-1).
"""

from __future__ import annotations

import argparse
import collections
import copy
import pathlib
import sys

import numpy as np
import onnx
import torch
from onnx import numpy_helper, shape_inference

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo, count_ops  # noqa: E402

HEIGHT, WIDTH = 368, 1232
OUT = pathlib.Path(__file__).resolve().parent.parent / "artifacts" / "hailo_stereo.onnx"

# thresholds -- each one is a defect from the teardown, made unfailable
MAX_INITIALIZER_MB = 1.0     # MEM-1: stereonet shipped a single 21.76 MB constant
MAX_TENSOR_RANK = 4          # HW-1: stereonet used 5D tensors around its Conv3d
BANNED_OPS = {"Conv3d", "ConvTranspose3d"}
# ops that compute shapes at runtime; the Hailo parser needs a static graph
DYNAMIC_SHAPE_OPS = {"Shape", "ConstantOfShape", "NonZero", "Range",
                     "Expand", "Tile", "Where", "ScatterND"}


def export(ckpt: str | None = None, out: pathlib.Path = OUT) -> pathlib.Path:
    model = HailoStereo()
    if ckpt:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        # STRICT -- the original's export used strict=False and silently shipped
        # randomly initialised tensors wherever the checkpoint did not fit (EXP-1)
        model.load_state_dict(blob.get("model", blob), strict=True)
        print(f"loaded {ckpt}: epoch {blob.get('epoch', '?')}, "
              f"val EPE {blob.get('epe', float('nan')):.3f} px")
    else:
        print("NO CHECKPOINT -- exporting random weights. The audit below"
              " is structural and still valid; the file is not deployable.")
    # fuse after loading: this folds the TRAINED temperature into the trained
    # cost head, which is not the same as folding the initial one
    model = model.eval()
    reference_state = copy.deepcopy(model.state_dict())
    model.fuse_temperature_()
    left = torch.zeros(1, 3, HEIGHT, WIDTH)
    right = torch.zeros(1, 3, HEIGHT, WIDTH)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (left, right), str(out),
        input_names=["left", "right"], output_names=["disparity"],
        opset_version=14, do_constant_folding=True, dynamo=False,
    )
    strip_identity(out)
    return out, reference_state


def strip_identity(path: pathlib.Path) -> int:
    """Remove Identity nodes by rewiring their consumers.

    The exporter leaves dozens of them. They are semantically free, but they
    inflate the graph the Hailo parser has to walk and obscure the real node
    count when auditing. Graph outputs are left alone so names stay stable.
    """
    model = onnx.load(str(path))
    graph = model.graph
    protected = {o.name for o in graph.output}
    remap: dict[str, str] = {}

    keep = []
    for node in graph.node:
        if node.op_type == "Identity" and node.output[0] not in protected:
            remap[node.output[0]] = node.input[0]
        else:
            keep.append(node)

    def resolve(name: str) -> str:
        seen = set()
        while name in remap and name not in seen:
            seen.add(name)
            name = remap[name]
        return name

    for node in keep:
        for i, inp in enumerate(node.input):
            node.input[i] = resolve(inp)

    removed = len(graph.node) - len(keep)
    del graph.node[:]
    graph.node.extend(keep)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return removed


def audit(path: pathlib.Path) -> int:
    model = shape_inference.infer_shapes(onnx.load(str(path)))
    graph = model.graph
    problems: list[str] = []

    size_mb = path.stat().st_size / 1e6
    print("=" * 68)
    print("EXPORTED GRAPH")
    print("=" * 68)
    print(f"file            {path.name}   {size_mb:.2f} MB")
    print(f"opset           {graph_opset(model)}")
    print(f"nodes           {len(graph.node)}")
    for vi in list(graph.input) + list(graph.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        print(f"  {vi.name:12s} {dims}")

    print("\n-- operators --")
    hist = collections.Counter(n.op_type for n in graph.node)
    for op, n in hist.most_common():
        print(f"   {op:20s} {n}")

    # ---- HW-1: no 3D convolution, no 5D tensors -------------------------
    print("\n" + "=" * 68)
    print("AUDIT")
    print("=" * 68)

    conv3d = [n.name for n in graph.node
              if n.op_type == "Conv" and len(_kernel_shape(n)) == 3]
    banned = [n.op_type for n in graph.node if n.op_type in BANNED_OPS]
    if conv3d or banned:
        problems.append(f"HW-1: 3D convolution present ({len(conv3d) + len(banned)})")
    print(f"[{_mark(not conv3d and not banned)}] HW-1  no 3D convolutions"
          f"                    (found {len(conv3d) + len(banned)})")

    ranks = []
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        ranks.append((len(vi.type.tensor_type.shape.dim), vi.name))
    over = [(r, n) for r, n in ranks if r > MAX_TENSOR_RANK]
    if over:
        problems.append(f"HW-1: {len(over)} tensors above rank {MAX_TENSOR_RANK}")
    print(f"[{_mark(not over)}] HW-1  every tensor is rank <= {MAX_TENSOR_RANK}"
          f"              (max rank {max(r for r, _ in ranks)})")

    # ---- PARSE: no shape-dynamic ops ------------------------------------
    # The Hailo parser wants a static graph. Exporting F.pad per hypothesis
    # emitted Shape/ConstantOfShape/Cast chains; pad-once/slice-many does not.
    dynamic = collections.Counter(
        n.op_type for n in graph.node if n.op_type in DYNAMIC_SHAPE_OPS)
    if dynamic:
        problems.append("PARSE: shape-dynamic ops present -- "
                        + ", ".join(f"{k}x{v}" for k, v in dynamic.items()))
    print(f"[{_mark(not dynamic)}] PARSE static graph, no shape arithmetic"
          f"        ({sum(dynamic.values())} found"
          f"{'' if not dynamic else ': ' + ', '.join(dynamic)})")

    # ---- MEM-1: no materialised constant grids --------------------------
    inits = [(numpy_helper.to_array(t), t.name) for t in graph.initializer]
    inits.sort(key=lambda x: -x[0].nbytes)
    biggest, big_name = (inits[0] if inits else (np.zeros(0), "-"))
    big_mb = biggest.nbytes / 1e6
    ok_mem = big_mb <= MAX_INITIALIZER_MB
    if not ok_mem:
        problems.append(f"MEM-1: initializer {big_name} is {big_mb:.2f} MB")
    print(f"[{_mark(ok_mem)}] MEM-1 largest constant <= {MAX_INITIALIZER_MB} MB"
          f"             ({big_mb:.3f} MB, {big_name})")

    ramps = [n for a, n in inits
             if a.ndim == 4 and a.size > 10_000
             and all(len(np.unique(a[0, c])) == 1 for c in range(min(a.shape[1], 32)))]
    if ramps:
        problems.append(f"MEM-1: constant index ramp still present ({ramps[0]})")
    print(f"[{_mark(not ramps)}] MEM-1 no broadcast index ramp baked in"
          f"          (found {len(ramps)})")

    # ---- NUM-1: softmax must run at matching resolution -----------------
    sm = [n for n in graph.node if n.op_type == "Softmax"]
    vi_map = {v.name: v for v in
              list(graph.value_info) + list(graph.output) + list(graph.input)}
    sm_elems, unresolved = [], []
    for n in sm:
        v = vi_map.get(n.output[0])
        dims = ([x.dim_value for x in v.type.tensor_type.shape.dim]
                if v is not None else [])
        if dims and all(dims):
            sm_elems.append((int(np.prod(dims)), tuple(dims)))
        else:
            unresolved.append(n.output[0])

    budget = (HEIGHT // 8) * (WIDTH // 8) * 24 * 2
    if not sm:
        ok_num, detail = False, "no Softmax node found at all"
        problems.append("NUM-1: graph has no softmax -- disparity is not being decoded")
    elif unresolved:
        # an audit that cannot read the value must fail, not pass silently
        ok_num, detail = False, f"shape unresolved for {unresolved}"
        problems.append(f"NUM-1: could not resolve softmax shape ({unresolved})")
    else:
        worst, shape = max(sm_elems)
        ok_num = worst <= budget
        detail = f"{worst:,} elements, {shape}"
        if not ok_num:
            problems.append(f"NUM-1: softmax over {worst:,} elements "
                            f"(budget {budget:,})")
    print(f"[{_mark(ok_num)}] NUM-1 softmax at matching resolution"
          f"           ({detail})")

    # ---- parameters ------------------------------------------------------
    total_params = sum(a.size for a, _ in inits)
    print(f"\nparameters      {total_params:,}  ({total_params / 1e6:.3f} M)")

    print("\n" + "=" * 68)
    if problems:
        print("FAILED")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PASSED -- graph is clear of every defect the audit covers")
    return 0


def compute_budget():
    print("\n" + "=" * 68)
    print("COMPUTE BUDGET")
    print("=" * 68)
    model = HailoStereo()
    totals, total = count_ops(model, HEIGHT, WIDTH)
    order = sorted(totals.items(), key=lambda kv: -kv[1])
    print(f"{'stage':18s} {'GMAC':>9s} {'GOPS':>9s} {'share':>8s}")
    for name, macs in order:
        print(f"{name:18s} {macs / 1e9:9.2f} {2 * macs / 1e9:9.2f} "
              f"{100 * macs / total:7.1f}%")
    print(f"{'TOTAL':18s} {total / 1e9:9.2f} {2 * total / 1e9:9.2f} {100.0:7.1f}%")
    print(f"\nreference: stereonet was 56.03 GMAC / 112.07 GOPS, "
          f"90.6% of it in refinement")
    refine = sum(v for k, v in totals.items() if k.startswith("refine"))
    print(f"this model: {100 * refine / total:.1f}% in refinement")
    return total


def _kernel_shape(node):
    for a in node.attribute:
        if a.name == "kernel_shape":
            return list(a.ints)
    return []


def graph_opset(model):
    return ", ".join(f"{o.domain or 'ai.onnx'} {o.version}" for o in model.opset_import)


MAX_PARITY_PX = 1e-2


def parity(path: pathlib.Path, reference_state: dict) -> int:
    """Run the exported graph and the source model on the same input and
    compare.

    The audit above is structural: it proves the graph has no 3D convolution
    and no baked ramp, not that it computes disparity. `fuse_temperature_`
    rewrites the cost head's BatchNorm scale by -t, and a sign error there
    produces a graph that passes every structural check and silently returns
    the argMAX of the cost volume -- the worst possible match instead of the
    best. That failure is invisible without this.
    """
    print()
    print("=" * 68)
    print("PARITY  (exported graph vs source model)")
    print("=" * 68)
    try:
        import onnxruntime as ort
    except ImportError as exc:
        print(f"[FAIL] onnxruntime unavailable ({exc}); parity NOT checked")
        return 1

    ref = HailoStereo()
    ref.load_state_dict(reference_state, strict=True)
    ref.eval()

    torch.manual_seed(0)
    left = torch.randn(1, 3, HEIGHT, WIDTH)
    right = torch.randn(1, 3, HEIGHT, WIDTH)
    with torch.no_grad():
        expected = ref(left, right).numpy()

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(["disparity"],
                   {"left": left.numpy(), "right": right.numpy()})[0]

    if got.shape != expected.shape:
        print(f"[FAIL] shape {got.shape} != {expected.shape}")
        return 1
    diff = np.abs(got - expected)
    worst, mean = float(diff.max()), float(diff.mean())
    ok = worst <= MAX_PARITY_PX
    print(f"[{_mark(ok)}] max |onnx - torch| = {worst:.2e} px "
          f"(mean {mean:.2e}, tolerance {MAX_PARITY_PX:.0e})")
    print(f"       disparity range: torch [{expected.min():.2f}, "
          f"{expected.max():.2f}] px, onnx [{got.min():.2f}, {got.max():.2f}] px")
    return 0 if ok else 1


def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="trained checkpoint; without it the export is a "
                         "structural mock with random weights")
    ap.add_argument("--out", default=str(OUT))
    cli = ap.parse_args()

    path, reference_state = export(cli.ckpt, pathlib.Path(cli.out))
    code = audit(path)
    code |= parity(path, reference_state)
    compute_budget()
    raise SystemExit(code)
