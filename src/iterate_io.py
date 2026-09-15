"""Run the exported HailoStereo ONNX graph for N rounds and check the values of
every input and output each round.

Two things are verified at once:
  * coverage  -- each round feeds a different real KITTI-domain pair (cycled
                 through artifacts/calib_kitti_*.npy), so the value ranges seen
                 are the ones the deployed model actually produces.
  * stability -- a fixed "canary" pair is run every single round and its output
                 is compared bit-for-bit against round 0. onnxruntime on CPU is
                 deterministic, so any drift here is a real problem.

Inputs are raw 0-255 float NCHW: the graph normalizes in-network (YAML
normalize_in_net: true), matching src/export_onnx.py and deploy/README.md.
"""
import argparse
import time
import numpy as np
import onnxruntime as ort

DISP_MAX = 192  # 24 hypotheses * 8 px stride -- output should live in [0, 192]


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    finite = np.isfinite(a)
    n = a.size
    return {
        "shape": tuple(a.shape),
        "min": float(a[finite].min()) if finite.any() else float("nan"),
        "max": float(a[finite].max()) if finite.any() else float("nan"),
        "mean": float(a[finite].mean()) if finite.any() else float("nan"),
        "std": float(a[finite].std()) if finite.any() else float("nan"),
        "nan": int((~np.isfinite(a)).sum()),
        "n": n,
    }


def to_nchw(pair_hwc):
    # uint8 NHWC -> float32 NCHW, kept on the raw 0-255 scale
    return pair_hwc.transpose(2, 0, 1)[None].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="artifacts/hailo_stereo.onnx")
    ap.add_argument("--left", default="artifacts/calib_kitti_left.npy")
    ap.add_argument("--right", default="artifacts/calib_kitti_right.npy")
    ap.add_argument("--rounds", type=int, default=50)
    args = ap.parse_args()

    L = np.load(args.left, mmap_mode="r")
    R = np.load(args.right, mmap_mode="r")
    npairs = min(len(L), len(R))
    print(f"loaded {npairs} pairs  left={L.shape}{L.dtype}  right={R.shape}{R.dtype}")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    innames = [i.name for i in sess.get_inputs()]
    outname = sess.get_outputs()[0].name
    print(f"onnx inputs={innames}  output={outname}")

    # canary: pair 0, run every round to prove determinism
    can_l, can_r = to_nchw(np.asarray(L[0])), to_nchw(np.asarray(R[0]))
    canary_ref = None

    print()
    hdr = (f"{'rnd':>3} {'pair':>4} | {'in L min/max/mean':>24} "
           f"{'in R mean':>9} | {'out min':>8} {'out max':>8} {'out mean':>8} "
           f"{'out std':>7} | {'nan':>3} {'>192px':>7} {'<0':>6} {'=0':>6} "
           f"{'canary':>8} {'ms':>6}")
    print(hdr)
    print("-" * len(hdr))

    agg = {"omin": [], "omax": [], "omean": [], "ostd": [],
           "oob": [], "neg": [], "zero": [], "nan": [], "ms": []}
    canary_ok = True
    inputs_sane = True

    for rnd in range(args.rounds):
        idx = rnd % npairs
        l = to_nchw(np.asarray(L[idx]))
        r = to_nchw(np.asarray(R[idx]))

        # input sanity: raw pixel data must be finite and in [0, 255]
        for name, arr in (("left", l), ("right", r)):
            s = stats(arr)
            if s["nan"] or s["min"] < 0 or s["max"] > 255:
                inputs_sane = False
                print(f"  !! INPUT {name} out of spec round {rnd}: "
                      f"min={s['min']} max={s['max']} nan={s['nan']}")

        t0 = time.perf_counter()
        out = sess.run([outname], {innames[0]: l, innames[1]: r})[0]
        dt = (time.perf_counter() - t0) * 1e3

        # canary every round on the same fixed input
        can = sess.run([outname], {innames[0]: can_l, innames[1]: can_r})[0]
        if canary_ref is None:
            canary_ref = can.copy()
            can_delta = 0.0
        else:
            can_delta = float(np.abs(can - canary_ref).max())
            if can_delta != 0.0:
                canary_ok = False

        li, ri, so = stats(l), stats(r), stats(out)
        oob = float((out > DISP_MAX).mean() * 100)
        neg = float((out < 0).mean() * 100)
        zero = float((out == 0).mean() * 100)

        for k, v in (("omin", so["min"]), ("omax", so["max"]),
                     ("omean", so["mean"]), ("ostd", so["std"]),
                     ("oob", oob), ("neg", neg), ("zero", zero),
                     ("nan", so["nan"]), ("ms", dt)):
            agg[k].append(v)

        print(f"{rnd:3d} {idx:4d} | "
              f"{li['min']:6.1f}/{li['max']:6.1f}/{li['mean']:6.1f} "
              f"{ri['mean']:9.1f} | "
              f"{so['min']:8.3f} {so['max']:8.3f} {so['mean']:8.3f} "
              f"{so['std']:7.3f} | "
              f"{so['nan']:3d} {oob:6.2f}% {neg:5.2f}% {zero:5.2f}% "
              f"{can_delta:8.1e} {dt:6.1f}")

    def rng(k):
        a = np.array(agg[k])
        return a.min(), a.max(), a.mean()

    print()
    print("=" * 68)
    print(f"SUMMARY over {args.rounds} rounds ({npairs} distinct pairs cycled)")
    print("=" * 68)
    for label, k in (("output min", "omin"), ("output max", "omax"),
                     ("output mean", "omean"), ("output std", "ostd"),
                     (">192px %", "oob"), ("negative %", "neg"),
                     ("exactly 0 %", "zero"), ("nan count", "nan"),
                     ("latency ms", "ms")):
        lo, hi, mu = rng(k)
        print(f"  {label:14s}  min={lo:10.4f}  max={hi:10.4f}  mean={mu:10.4f}")

    print()
    print(f"  inputs in [0,255], finite ........ {'PASS' if inputs_sane else 'FAIL'}")
    print(f"  no NaN/Inf in any output ......... "
          f"{'PASS' if sum(agg['nan']) == 0 else 'FAIL'}")
    print(f"  canary bit-identical all rounds .. {'PASS' if canary_ok else 'FAIL'}")
    print(f"  all outputs within [0,192] px .... "
          f"{'PASS' if max(agg['oob']) == 0 and min(agg['omin']) >= 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
