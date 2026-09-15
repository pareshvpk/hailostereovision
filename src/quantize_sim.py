"""
Int8 quantization simulation for HailoStereo.

    python src/quantize_sim.py --ckpt runs/sceneflow_cont/best.pt --root data/driving
    python src/quantize_sim.py --ckpt ... --root ... --sensitivity

The model this replaces lost 25% of its accuracy to int8 and shipped an `.alls`
with no mitigation for it (defect NUM-1 in the teardown). The soft-argmin
temperature in this model exists specifically so that calibration has a sane
logit range to work with. This script measures whether that worked, before
anyone has access to a Linux box with the Hailo Dataflow Compiler on it.

WHAT THIS IS NOT
----------------
This is not the Hailo emulator and its numbers are not the numbers the DFC will
produce. It models uniform int8 with min/max-percentile calibration. The DFC
does its own calibration, can promote layers to 16-bit, and applies equalization
passes this does not. Treat the RANKING it produces as the useful output -- which
layers are fragile -- and the absolute EPE as an approximate upper bound on the
damage.

WHAT IT MODELS
--------------
One quantization point per convolution, which is what the deployed graph has:
BatchNorm folds into the preceding convolution at export, so a conv+BN pair is
one fused layer with one output scale. Quantization is applied to

  * every convolution's weights   -- signed int8, symmetric, per output channel
  * every fused layer's output    -- unsigned/signed int8, asymmetric, per tensor
  * the softmax output            -- the site of NUM-1; probabilities land in
                                     [0,1] and get 1/255 resolution, which is
                                     what makes a soft-argmin fragile
  * the two input images          -- the chip quantizes its inputs too

LeakyReLU is monotonic and range-preserving, so quantizing at the fused layer's
pre-activation output is a close proxy for quantizing after it, and it keeps the
count of quantization points equal to the count of layers on device.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo  # noqa: E402
from train import build_datasets, evaluate  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


# ---------------------------------------------------------------------------
# fake quantization
# ---------------------------------------------------------------------------

def quantize_weight(w: torch.Tensor, bits: int = 8, per_channel: bool = True):
    """Symmetric signed fake-quant. Per output channel by default, which is what
    the Hailo compiler uses for convolution weights."""
    qmax = 2 ** (bits - 1) - 1
    if per_channel:
        flat = w.reshape(w.shape[0], -1)
        scale = flat.abs().amax(dim=1).clamp(min=1e-12) / qmax
        scale = scale.reshape(-1, *([1] * (w.dim() - 1)))
    else:
        scale = w.abs().amax().clamp(min=1e-12) / qmax
    return (w / scale).round().clamp(-qmax - 1, qmax) * scale


class ActQuant:
    """Asymmetric per-tensor activation fake-quant with a calibrated range.

    Calibration takes a high percentile rather than the raw maximum: a single
    outlier activation in one calibration image would otherwise stretch the
    scale over the whole run and cost real resolution everywhere else.
    """

    def __init__(self, bits: int = 8, percentile: float = 99.99):
        self.bits = bits
        self.percentile = percentile
        self.lo = None
        self.hi = None
        self.calibrating = False
        self.enabled = False

    def observe(self, x: torch.Tensor) -> None:
        v = x.detach().flatten().float()
        if v.numel() > 100_000:      # torch.quantile has a size ceiling
            # Seeded, so the reported penalty is reproducible. Unseeded
            # subsampling moved the full-int8 figure by 1-2 points between
            # otherwise identical runs, which is enough to make a quoted
            # number unverifiable.
            g = torch.Generator().manual_seed(0)
            idx = torch.randint(v.numel(), (100_000,), generator=g)
            v = v[idx.to(v.device)]
        q = self.percentile / 100.0
        lo = torch.quantile(v, 1.0 - q).item()
        hi = torch.quantile(v, q).item()
        self.lo = lo if self.lo is None else min(self.lo, lo)
        self.hi = hi if self.hi is None else max(self.hi, hi)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.lo is None:
            return x
        levels = 2 ** self.bits - 1
        scale = max((self.hi - self.lo) / levels, 1e-12)
        zero = round(-self.lo / scale)
        q = (x / scale).round() + zero
        return (q.clamp(0, levels) - zero) * scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self.observe(x)
        if self.enabled:
            return self.apply(x)
        return x


# ---------------------------------------------------------------------------
# instrumenting the model
# ---------------------------------------------------------------------------

class QuantModel:
    """Attaches fake-quant to a HailoStereo instance in place.

    Layers are discovered structurally: a Conv2d followed immediately by a
    BatchNorm2d under the same parent is one fused device layer, and the pair
    gets a single output quantization point on the BatchNorm. Every other
    convolution gets one on itself.
    """

    def __init__(self, model: nn.Module, w_bits=8, a_bits=8,
                 per_channel=True, percentile=99.99):
        self.model = model
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.per_channel = per_channel
        self.layers = {}         # name -> {"conv", "site", "act", "fp32"}
        self.handles = []
        self._build(percentile)

    # ------------------------------------------------------------------
    def _build(self, percentile):
        # pair each conv with the batchnorm that folds into it
        paired = {}
        for parent in self.model.modules():
            kids = list(parent.named_children())
            for (na, a), (_, b) in zip(kids, kids[1:]):
                if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                    paired[id(a)] = b

        for name, conv in self.model.named_modules():
            if not isinstance(conv, nn.Conv2d):
                continue
            site = paired.get(id(conv), conv)
            act = ActQuant(self.a_bits, percentile)
            self.layers[name] = {"conv": conv, "site": site, "act": act,
                                 "fp32": conv.weight.detach().clone()}
            self.handles.append(
                site.register_forward_hook(self._act_hook(act)))

        # NUM-1: the softmax output, quantized on the way into the index conv
        self.softmax_act = ActQuant(self.a_bits, percentile)
        self.handles.append(
            self.model.soft_argmin.index.register_forward_pre_hook(
                self._pre_hook(self.softmax_act)))

        # the chip quantizes its inputs too
        self.input_act = ActQuant(self.a_bits, percentile)
        self.handles.append(
            self.model.register_forward_pre_hook(self._input_hook()))

    @staticmethod
    def _act_hook(act):
        def fn(mod, inp, out):
            return act(out)
        return fn

    @staticmethod
    def _pre_hook(act):
        def fn(mod, inp):
            return (act(inp[0]),)
        return fn

    def _input_hook(self):
        def fn(mod, inp):
            return tuple(self.input_act(t) for t in inp)
        return fn

    # ------------------------------------------------------------------
    def all_acts(self):
        return ([d["act"] for d in self.layers.values()]
                + [self.softmax_act, self.input_act])

    def set_calibrating(self, flag: bool):
        for a in self.all_acts():
            a.calibrating = flag

    def set_enabled(self, names=None, acts=True, weights=True):
        """Enable quantization for `names` (all layers when None).

        Weight quantization is applied destructively to the live parameter and
        restored from the fp32 copy each time, so exactly the requested set is
        ever quantized -- no residue from a previous configuration.
        """
        chosen = set(self.layers) if names is None else set(names)
        for name, d in self.layers.items():
            on = name in chosen
            d["act"].enabled = acts and on
            d["conv"].weight.data.copy_(
                quantize_weight(d["fp32"], self.w_bits, self.per_channel)
                if (weights and on) else d["fp32"])
        # the softmax and input points belong to the whole-graph configuration,
        # not to any single layer, so a per-layer sweep leaves them in float
        whole = names is None
        self.softmax_act.enabled = acts and whole
        self.input_act.enabled = acts and whole

    def restore(self):
        self.set_enabled(names=set(), acts=False, weights=False)

    def remove(self):
        self.restore()
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def build_loaders(args):
    ns = argparse.Namespace(dataset=args.dataset, root=args.root,
                            crop_h=args.crop_h, crop_w=args.crop_w,
                            pass_name=args.pass_name, limit=None,
                            val_limit=args.val_limit)
    train_set, val_set = build_datasets(ns)
    val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2,
                     pin_memory=True)
    # calibration draws from TRAIN, never from the set we report on, and sees
    # full frames with no colour jitter -- calibrating on augmented crops would
    # fit the scales to a distribution the deployed model never sees
    train_set.samples = train_set.samples[:args.calib]
    train_set.training = False
    calib = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=2)
    return calib, val


def calibrate(qm, loader, device):
    qm.set_calibrating(True)
    qm.set_enabled(names=set(), acts=False, weights=False)
    with torch.no_grad():
        for batch in loader:
            qm.model(batch["left"].to(device), batch["right"].to(device))
    qm.set_calibrating(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", choices=("sceneflow", "kitti"), default="sceneflow")
    p.add_argument("--root", required=True)
    p.add_argument("--crop-h", type=int, default=256)
    p.add_argument("--crop-w", type=int, default=512)
    p.add_argument("--pass-name", default=None)
    p.add_argument("--val-limit", type=int, default=200)
    p.add_argument("--calib", type=int, default=64,
                   help="calibration images, drawn from the training split")
    p.add_argument("--percentile", type=float, default=99.99)
    p.add_argument("--sensitivity", action="store_true",
                   help="also sweep every layer alone, ranking them by damage")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HailoStereo()
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)   # EXP-1
    model.to(device).eval()
    print(f"{args.ckpt}: epoch {ckpt.get('epoch', '?')}, "
          f"float EPE recorded {ckpt.get('epe', float('nan')):.3f} px")

    calib_loader, val_loader = build_loaders(args)

    base_epe, base_d1 = evaluate(model, val_loader, device)
    print(f"\nfloat32           EPE {base_epe:6.3f} px   D1 {base_d1:5.2f}%")

    qm = QuantModel(model, percentile=args.percentile)
    print(f"instrumented {len(qm.layers)} convolutions "
          f"+ softmax + input")
    calibrate(qm, calib_loader, device)

    rows = []
    for label, kw in (("weights int8 (per channel)", dict(acts=False, weights=True)),
                      ("activations int8", dict(acts=True, weights=False)),
                      ("full int8", dict(acts=True, weights=True))):
        qm.set_enabled(None, **kw)
        epe, d1 = evaluate(model, val_loader, device)
        rows.append((label, epe, d1))
        print(f"{label:18s}EPE {epe:6.3f} px   D1 {d1:5.2f}%   "
              f"({100 * (epe - base_epe) / base_epe:+.1f}% EPE)")
    qm.restore()

    t = float(model.soft_argmin.temperature().detach())
    sm = qm.softmax_act
    print(f"\nsoft-argmin temperature {t:.3f}; calibrated softmax range "
          f"[{sm.lo:.4f}, {sm.hi:.4f}] -> {(sm.hi - sm.lo) / 255:.5f} per step")

    if args.sensitivity:
        print("\nper-layer sensitivity (each layer quantized alone):")
        scores = []
        for name in qm.layers:
            qm.set_enabled([name])
            epe, _ = evaluate(model, val_loader, device)
            scores.append((epe - base_epe, name))
        qm.restore()
        scores.sort(reverse=True)
        for delta, name in scores[:15]:
            print(f"  {delta:+8.4f} px   {name}")
        print("  ...")
        for delta, name in scores[-3:]:
            print(f"  {delta:+8.4f} px   {name}")
        print("\nThe layers at the top are the candidates for 16-bit in the "
              "\n.alls quantization script when the compiler is available.")

    qm.remove()
    print("\nreference: the original stereonet reports 8.223 float / 10.3 "
          "quantized on KITTI -- a 25% loss to int8.")


if __name__ == "__main__":
    main()
