"""
Soak and stress the compiled graph on the Hailo-15H EMULATOR, in place of a board.

    ./deploy/hailo-py notes/stress_emulator.py --minutes 30

There is no 15H card on this machine, so "run the model" means the DFC's
SDK_QUANTIZED context over deploy/build/hailo_stereo_opt.har -- the same
int8 software model that produced the 1.848 px figure in deploy/README.md.

What this is for. The 40-pair emulate run proves the model is ACCURATE on data
that looks like its training set. It says nothing about what the graph does when
handed something degenerate, whether it stays numerically sane over hundreds of
consecutive inferences, or whether it is deterministic. A dataflow NPU has no
exception handler: a NaN on device is a corrupt frame, not a traceback. So each
inference here is checked for the things that would be invisible on hardware --
NaN/Inf, out-of-clamp values, shape drift, bitwise non-determinism, latency and
RSS creep -- not only for EPE.

Phases, in order, each bounded by the overall deadline:

  smoke        real KITTI pairs, scored. Must reproduce the 1.848 px baseline.
  determinism  one pair inferred repeatedly. Must be BITWISE identical.
  geometry     synthetic right built by shifting left by a known d. The
               prediction must recover d. This is the test the original model
               fails, run here on the quantized graph rather than in torch.
  adversarial  degenerate inputs: black, white, noise, saturated, textureless,
               swapped, decorrelated. None may crash or produce a non-finite.
  soak         the remaining time cycling everything, watching for drift.

Every inference appends one JSON line to build/stress_log.jsonl, so a run that
dies halfway is still readable.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "deploy" / "build"
HAR_OPT = BUILD / "hailo_stereo_opt.har"
NAMES = BUILD / "resolved_names.json"
HW_ARCH = "hailo15h"

MAX_DISP = 192.0          # src/model.py clamps the output to [0, MAX_DISP]
H, W = 368, 1232          # the crop the graph was compiled for

sys.path.insert(0, str(ROOT / "deploy"))
from dfc_flow import kitti_val_pairs, score, normalize   # noqa: E402


# ---------------------------------------------------------------------------
# process introspection
# ---------------------------------------------------------------------------

def rss_mb() -> float:
    """Resident set size in MB, read from /proc rather than psutil (not pinned)."""
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return pages * 4096 / 1e6


# ---------------------------------------------------------------------------
# input synthesis
# ---------------------------------------------------------------------------

def shift_pair(left: np.ndarray, d: int):
    """Synthesize a right view at constant disparity d.

    The convention is I_L(x) = I_R(x - d) -- src/check_ingest.py warps with
    xs = x - disp -- so right[:, j] = left[:, j + d]. The last d columns have
    no source and are left black; they are excluded from the assertion.
    """
    right = np.zeros_like(left)
    if d == 0:
        return left.copy()
    right[:, : W - d] = left[:, d:]
    return right


def synth_inputs(rng: np.random.Generator, real_left: np.ndarray):
    """The adversarial battery.

    Each entry is (name, left, right, expectation). `expectation` is None where
    there is no correct answer and the only requirement is "do not crash and do
    not emit a non-finite".
    """
    z = lambda v: np.full((H, W, 3), v, np.uint8)           # noqa: E731
    noise = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    noise2 = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    ramp = np.tile(np.linspace(0, 255, W, dtype=np.uint8)[None, :, None], (H, 1, 3))
    vramp = np.tile(np.linspace(0, 255, H, dtype=np.uint8)[:, None, None], (1, W, 3))
    lowc = (real_left.astype(np.float32) * 0.04 + 120).astype(np.uint8)
    sp = real_left.copy()
    m = rng.random((H, W)) < 0.30
    sp[m] = rng.integers(0, 256, (int(m.sum()), 3), dtype=np.uint8)

    return [
        # degenerate constants -- no texture anywhere, matching is undefined
        ("black",            z(0),        z(0),        None),
        ("white",            z(255),      z(255),      None),
        ("mid-gray",         z(128),      z(128),      None),
        ("black-vs-white",   z(0),        z(255),      None),
        # pure noise -- every hypothesis is equally bad
        ("noise-identical",  noise,       noise,       0),
        ("noise-decorrelated", noise,     noise2,      None),
        # structured but matchless
        ("h-ramp",           ramp,        ramp,        0),
        ("v-ramp",           vramp,       vramp,       0),
        # real content, degenerate pairing
        ("real-identical",   real_left,   real_left,   0),
        ("real-swapped",     shift_pair(real_left, 32), real_left, None),
        ("real-low-contrast", lowc,       shift_pair(lowc, 24), 24),
        ("real-salt-pepper", sp,          shift_pair(sp, 24),   24),
        ("real-vs-noise",    real_left,   noise,       None),
        ("real-vs-black",    real_left,   z(0),        None),
        ("black-vs-real",    z(0),        real_left,   None),
        # saturation extremes on real content
        ("real-clipped-hi",  np.minimum(real_left.astype(np.int16) + 200, 255).astype(np.uint8),
                             np.minimum(shift_pair(real_left, 24).astype(np.int16) + 200, 255).astype(np.uint8), 24),
        ("real-clipped-lo",  (real_left // 8), shift_pair(real_left // 8, 24), 24),
        # channel pathologies
        ("red-only",         real_left * np.array([1, 0, 0], np.uint8),
                             shift_pair(real_left, 24) * np.array([1, 0, 0], np.uint8), 24),
        ("inverted",         255 - real_left, 255 - shift_pair(real_left, 24), 24),
        ("vflipped",         real_left[::-1].copy(), shift_pair(real_left, 24)[::-1].copy(), 24),
    ]


# ---------------------------------------------------------------------------
# one inference + every check that does not need ground truth
# ---------------------------------------------------------------------------

class Checker:
    def __init__(self, logf):
        self.logf = logf
        self.rows = []
        self.failures = []
        self.n = 0

    def record(self, phase, case, pred, dt, extra=None, err=None):
        self.n += 1
        row = {"i": self.n, "phase": phase, "case": case, "t": round(time.time(), 1),
               "dt": round(dt, 2), "rss_mb": round(rss_mb(), 1)}
        if err is not None:
            row["error"] = err
            self.failures.append((phase, case, err))
        else:
            finite = np.isfinite(pred)
            nan = int(np.isnan(pred).sum())
            inf = int((~finite & ~np.isnan(pred)).sum())
            below = int((pred < -1e-6).sum())
            above = int((pred > MAX_DISP + 1e-3).sum())
            row.update({
                "shape": list(pred.shape), "dtype": str(pred.dtype),
                "min": float(pred.min()), "max": float(pred.max()),
                "mean": float(pred.mean()), "std": float(pred.std()),
                "nan": nan, "inf": inf, "below0": below, "above_max": above,
            })
            if list(pred.shape) != [H, W]:
                self.failures.append((phase, case, f"shape {pred.shape}"))
            if nan or inf:
                self.failures.append((phase, case, f"{nan} NaN, {inf} Inf"))
            if below or above:
                self.failures.append(
                    (phase, case, f"{below} px < 0, {above} px > {MAX_DISP} "
                                  f"(clamp violated)"))
        if extra:
            row.update(extra)
        self.rows.append(row)
        self.logf.write(json.dumps(row) + "\n")
        self.logf.flush()
        return row


def infer(runner, ctx, names, left, right):
    l = left[None].astype(np.uint8)
    r = right[None].astype(np.uint8)
    out = runner.infer(ctx, {names["left"]: l, names["right"]: r})
    return np.squeeze(out[0] if isinstance(out, (list, tuple)) else out)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--context", default="quantized",
                    choices=["quantized", "fp_optimized", "bit_exact"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from hailo_sdk_client import ClientRunner, InferenceContext

    ctx_attr = {"quantized": "SDK_QUANTIZED", "fp_optimized": "SDK_FP_OPTIMIZED",
                "bit_exact": "SDK_BIT_EXACT"}[args.context]
    ctx_enum = getattr(InferenceContext, ctx_attr)

    deadline = time.time() + args.minutes * 60.0
    rng = np.random.default_rng(args.seed)
    names = json.loads(NAMES.read_text())
    pairs = list(kitti_val_pairs(ROOT / "data" / "kitti2015" / "training"))

    logpath = BUILD / "stress_log.jsonl"
    logf = logpath.open("w")
    chk = Checker(logf)

    print(f"[stress] context={ctx_attr} budget={args.minutes:.0f} min "
          f"har={HAR_OPT.name}")
    print(f"[stress] {len(pairs)} KITTI val pairs available, log -> "
          f"{logpath.relative_to(ROOT)}")
    rss0 = rss_mb()
    print(f"[stress] RSS at start {rss0:.0f} MB\n")

    runner = ClientRunner(har=str(HAR_OPT), hw_arch=HW_ARCH)
    summary = {"context": ctx_attr, "phases": {}}

    def left_budget():
        return deadline - time.time()

    with runner.infer_context(ctx_enum) as ctx:

        def run(phase, case, left, right, disp=None, expect=None):
            """One guarded inference. Returns (row, pred) or (row, None)."""
            t = time.time()
            try:
                pred = infer(runner, ctx, names, left, right)
            except Exception as e:                      # noqa: BLE001
                row = chk.record(phase, case, None, time.time() - t,
                                 err=f"{type(e).__name__}: {e}")
                print(f"  [{phase}] {case:22s} EXCEPTION {type(e).__name__}: {e}")
                traceback.print_exc()
                return row, None
            extra = {}
            if disp is not None:
                s = score(pred, disp)
                extra = {"epe_masked": s["masked"][0], "epe_official": s["official"][0],
                         "d1_official": s["official"][1]}
            if expect is not None:
                # assert on the region the synthetic shift can actually support:
                # past the left border (no match exists there) and clear of the
                # unsourced right edge.
                reg = pred[:, 200:W - 64]
                extra["shift_expect"] = expect
                extra["shift_median"] = float(np.median(reg))
                extra["shift_p10"] = float(np.percentile(reg, 10))
                extra["shift_p90"] = float(np.percentile(reg, 90))
                extra["shift_frac_within_2px"] = float(
                    np.mean(np.abs(reg - expect) <= 2.0))
            row = chk.record(phase, case, pred, time.time() - t, extra)
            return row, pred

        # ------------------------------------------------------------------
        # 1. SMOKE -- real data, must reproduce the published baseline
        # ------------------------------------------------------------------
        print("=== phase 1: smoke (real KITTI, scored) ===")
        acc = {"masked": [0.0, 0.0, 0], "official": [0.0, 0.0, 0]}
        smoke_n = 0
        for name, left, right, disp in pairs[:8]:
            if left_budget() < 120:
                break
            row, pred = run("smoke", name, left, right, disp=disp)
            if pred is None:
                continue
            s = score(pred, disp)
            for k in acc:
                e, d, n = s[k]
                if n:
                    acc[k][0] += e * n; acc[k][1] += d * n; acc[k][2] += n
            smoke_n += 1
            print(f"  [smoke] {name:22s} {row['dt']:5.1f}s  "
                  f"EPE {s['masked'][0]:6.3f} masked / {s['official'][0]:6.3f} official  "
                  f"range [{row['min']:.1f},{row['max']:.1f}]")
        if acc["masked"][2]:
            sm = acc["masked"][0] / acc["masked"][2]
            so = acc["official"][0] / acc["official"][2]
            summary["phases"]["smoke"] = {"n": smoke_n, "epe_masked": sm,
                                          "epe_official": so}
            print(f"  -> {smoke_n} pairs: {sm:.3f} px masked / {so:.3f} px official")
            print(f"     40-pair reference for this context: 1.654 / 1.848 px")

        # ------------------------------------------------------------------
        # 2. DETERMINISM -- the same input must give the same bits
        # ------------------------------------------------------------------
        print("\n=== phase 2: determinism ===")
        det = []
        name, left, right, disp = pairs[0]
        for k in range(3):
            if left_budget() < 90:
                break
            _, pred = run("determinism", f"{name}#{k}", left, right, disp=disp)
            if pred is not None:
                det.append(pred)
        if len(det) >= 2:
            diffs = [float(np.abs(det[0] - d).max()) for d in det[1:]]
            exact = all(np.array_equal(det[0], d) for d in det[1:])
            summary["phases"]["determinism"] = {"repeats": len(det),
                                                "bitwise_identical": exact,
                                                "max_abs_diff": max(diffs)}
            print(f"  {len(det)} repeats, bitwise identical: {exact}, "
                  f"max |diff| {max(diffs):.3e} px")
            if not exact:
                chk.failures.append(("determinism", name,
                                     f"not bitwise identical, max {max(diffs):.3e}"))

        # ------------------------------------------------------------------
        # 3. GEOMETRY -- known shift in, known disparity out
        # ------------------------------------------------------------------
        print("\n=== phase 3: geometry (synthetic known-disparity shifts) ===")
        base_left = pairs[3][1]
        geo = []
        for d in (0, 8, 16, 32, 64, 96, 128):
            if left_budget() < 90:
                break
            row, pred = run("geometry", f"shift={d}px", base_left,
                            shift_pair(base_left, d), expect=d)
            if pred is None:
                continue
            geo.append((d, row["shift_median"], row["shift_frac_within_2px"]))
            ok = abs(row["shift_median"] - d) <= 3.0
            print(f"  shift {d:3d} px -> median {row['shift_median']:6.2f} px  "
                  f"p10/p90 {row['shift_p10']:6.2f}/{row['shift_p90']:6.2f}  "
                  f"within2px {100*row['shift_frac_within_2px']:5.1f}%  "
                  f"{'ok' if ok else 'OFF'}")
            if not ok:
                chk.failures.append(("geometry", f"shift={d}",
                                     f"median {row['shift_median']:.2f} != {d}"))
        summary["phases"]["geometry"] = [
            {"d": d, "median": m, "within2px": f} for d, m, f in geo]

        # ------------------------------------------------------------------
        # 4. ADVERSARIAL -- degenerate inputs, no crash, no non-finite
        # ------------------------------------------------------------------
        print("\n=== phase 4: adversarial ===")
        battery = synth_inputs(rng, pairs[3][1])
        adv = []
        for cname, l, r, expect in battery:
            if left_budget() < 90:
                print(f"  (budget exhausted, {len(battery)-len(adv)} cases skipped)")
                break
            row, pred = run("adversarial", cname, l, r, expect=expect)
            if pred is None:
                adv.append(cname)
                continue
            adv.append(cname)
            tail = ""
            if expect is not None:
                tail = (f"  expect {expect:3d} -> median {row['shift_median']:6.2f}"
                        f"  within2px {100*row['shift_frac_within_2px']:5.1f}%")
            print(f"  {cname:22s} {row['dt']:5.1f}s  "
                  f"[{row['min']:6.1f},{row['max']:6.1f}] mean {row['mean']:6.1f} "
                  f"std {row['std']:5.1f}  nan {row['nan']} inf {row['inf']}{tail}")
        summary["phases"]["adversarial"] = {"cases": len(adv)}

        # ------------------------------------------------------------------
        # 5. SOAK -- burn the remaining budget, watch for drift
        # ------------------------------------------------------------------
        print(f"\n=== phase 5: soak ({left_budget()/60:.1f} min remaining) ===")
        cycle = [("real", p) for p in pairs[:6]] + \
                [("synth", c) for c in battery[:6]]
        soak_epe, k = [], 0
        while left_budget() > 20:
            kind, item = cycle[k % len(cycle)]
            k += 1
            if kind == "real":
                name, l, r, disp = item
                row, pred = run("soak", name, l, r, disp=disp)
                if pred is not None:
                    soak_epe.append(row["epe_masked"])
            else:
                cname, l, r, expect = item
                row, pred = run("soak", cname, l, r, expect=expect)
            if k % 5 == 0:
                el = (time.time() - (deadline - args.minutes * 60)) / 60
                print(f"  [soak] {k:3d} inferences, {el:4.1f} min elapsed, "
                      f"RSS {row['rss_mb']:6.0f} MB, last {row['dt']:.1f}s")
        summary["phases"]["soak"] = {"inferences": k,
                                     "epe_samples": len(soak_epe)}

    # ----------------------------------------------------------------------
    # report
    # ----------------------------------------------------------------------
    rows = [r for r in chk.rows if "error" not in r]
    dts = np.array([r["dt"] for r in rows])
    rsss = np.array([r["rss_mb"] for r in rows])
    print("\n" + "=" * 72)
    print(f"total inferences      {chk.n}   ({len(rows)} completed, "
          f"{chk.n - len(rows)} raised)")
    if len(dts):
        print(f"latency               mean {dts.mean():.2f}s  min {dts.min():.2f}  "
              f"max {dts.max():.2f}  p95 {np.percentile(dts,95):.2f}")
        first, last = dts[:10].mean(), dts[-10:].mean()
        print(f"latency drift         first10 {first:.2f}s -> last10 {last:.2f}s "
              f"({100*(last-first)/first:+.1f}%)")
        print(f"RSS                   {rsss[0]:.0f} -> {rsss[-1]:.0f} MB "
              f"({rsss[-1]-rsss[0]:+.0f} MB over {chk.n} inferences)")
    nan_tot = sum(r.get("nan", 0) for r in rows)
    inf_tot = sum(r.get("inf", 0) for r in rows)
    clamp = sum(r.get("below0", 0) + r.get("above_max", 0) for r in rows)
    print(f"non-finite outputs    {nan_tot} NaN, {inf_tot} Inf")
    print(f"clamp violations      {clamp} px outside [0, {MAX_DISP:.0f}]")
    ep = [r["epe_masked"] for r in rows if "epe_masked" in r]
    if ep:
        print(f"EPE over all scored   mean {np.mean(ep):.3f} px  "
              f"({len(ep)} scored inferences)")
    print(f"\nFAILURES: {len(chk.failures)}")
    for ph, case, msg in chk.failures:
        print(f"  [{ph}] {case}: {msg}")
    if not chk.failures:
        # Name the phases that actually ran. A budget too small to reach a
        # phase skips it silently, and "no failures" must not be read as
        # "that check passed" when the check never executed.
        ran = sorted({r["phase"] for r in chk.rows})
        skipped = [p for p in ("smoke", "determinism", "geometry",
                               "adversarial", "soak") if p not in ran]
        print(f"  none, over phases that ran: {', '.join(ran)}")
        if skipped:
            print(f"  NOT CHECKED (budget too small to reach): "
                  f"{', '.join(skipped)}")

    summary.update({
        "total_inferences": chk.n, "completed": len(rows),
        "exceptions": chk.n - len(rows),
        "latency_mean_s": float(dts.mean()) if len(dts) else None,
        "latency_max_s": float(dts.max()) if len(dts) else None,
        "rss_start_mb": float(rsss[0]) if len(rsss) else None,
        "rss_end_mb": float(rsss[-1]) if len(rsss) else None,
        "nan_total": nan_tot, "inf_total": inf_tot, "clamp_violations": clamp,
        "failures": [{"phase": p, "case": c, "msg": m} for p, c, m in chk.failures],
    })
    (BUILD / "stress_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {(BUILD/'stress_summary.json').relative_to(ROOT)} and "
          f"{logpath.relative_to(ROOT)}")
    logf.close()
    return 1 if chk.failures else 0


if __name__ == "__main__":
    sys.exit(main())
