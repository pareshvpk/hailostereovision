"""
HailoStereo -- a stereo disparity network designed for the Hailo-15H dataflow NPU.

This is a ground-up replacement for the Hailo Model Zoo `stereonet` entry. Every
structural decision below traces to a defect found in the teardown of that model;
the defect ID is cited at each site.

Operator budget
---------------
The op set is restricted to what is provably compilable by the Hailo Dataflow
Compiler, established empirically from the layer table of the shipped
`stereonet.hef`: conv, ew_add, ew_sub, concat, slice, resize, softmax,
reduce_sum, normalization. In particular there are NO 3D convolutions and no
5D tensors anywhere in this graph (HW-1).

Geometry
--------
Matching runs at 1/8 resolution with D=24 hypotheses, giving a maximum disparity
of 24*8 = 192 px at the 368x1232 native input -- the same range the original
targeted, at twice its disparity resolution (ARCH-1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_DISP = 192          # full-resolution pixels
MATCH_SCALE = 8         # matching happens at 1/8
NUM_DISP = MAX_DISP // MATCH_SCALE   # 24 hypotheses


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

def conv_bn_act(cin, cout, k=3, stride=1, dilation=1, slope=0.1):
    pad = dilation * (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(slope, inplace=True),
    )


class ResBlock(nn.Module):
    """Pre-folded residual block. BatchNorm folds into the convolution at export,
    so the deployed graph is conv -> leaky -> conv -> ew_add -> leaky."""

    def __init__(self, ch, dilation=1, slope=0.1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.LeakyReLU(slope, inplace=True)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + x)


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------

class FeatureNet(nn.Module):
    """Siamese tower. Returns guidance features at 1/2 and 1/4 for the
    refinement ladder, and the matching feature at 1/8.

    Channel counts rise as resolution falls, which is the opposite of the
    original's flat 32 channels everywhere (HW-1).
    """

    def __init__(self, match_ch=32):
        super().__init__()
        self.stem2 = conv_bn_act(3, 16, 3, stride=2)     # 1/2
        self.stem4 = conv_bn_act(16, 24, 3, stride=2)    # 1/4
        self.stem8 = conv_bn_act(24, 32, 3, stride=2)    # 1/8
        self.res = nn.Sequential(*[ResBlock(32) for _ in range(6)])
        self.head = nn.Conv2d(32, match_ch, 3, 1, 1, bias=False)

    def forward(self, x):
        f2 = self.stem2(x)
        f4 = self.stem4(f2)
        f8 = self.res(self.stem8(f4))
        return f2, f4, self.head(f8)


# --------------------------------------------------------------------------
# cost volume  -- this is the site of defect CV-1
# --------------------------------------------------------------------------

def shift_right_features(right: torch.Tensor, d: int) -> torch.Tensor:
    """Align the right image's features to the left image at disparity `d`.

    For a rectified pair, the match for left pixel x lies at right pixel x-d.
    So the shifted tensor must satisfy  out[..., x] = right[..., x - d],  with
    the first `d` columns undefined (they reference negative coordinates).

    THIS IS THE FUNCTION THE ORIGINAL GOT WRONG (CV-1). Upstream padded on the
    right and then sliced [:W], which returns the input untouched:

        WRONG:  torch.cat((x, zeros_d), 3)[..., :W]     ->  x, unshifted
        RIGHT:  torch.cat((zeros_d, x), 3)[..., :W]     ->  x shifted by d

    F.pad's pad spec is (left, right) on the last dimension, so (d, 0) pads the
    LEFT edge -- which is what makes the subsequent [:W] slice a real shift.

    Used by the tests and as the reference definition. CostVolume.forward uses
    the pad-once/slice-many form below, which is equivalent but exports to a
    static graph; see `_shift_all`.
    """
    if d == 0:
        return right
    w = right.shape[-1]
    return F.pad(right, (d, 0))[..., :w]


def _shift_all(right: torch.Tensor, num_disp: int):
    """All `num_disp` shifts of `right`, as a static ONNX subgraph.

    Equivalent to calling shift_right_features for each d, but pads once and
    takes constant-offset slices out of the padded tensor. F.pad called per
    hypothesis exports through torch's dynamic padding path, which emits
    Shape / ConstantOfShape / Cast chains that the Hailo parser will not take.
    Padding once and slicing gives one Concat and `num_disp` constant Slices.
    """
    b, c, h, w = (int(s) for s in right.shape)
    lead = num_disp - 1
    zeros = right.new_zeros((b, c, h, lead))
    padded = torch.cat([zeros, right], dim=3)          # width w + num_disp - 1
    # hypothesis d starts `d` columns earlier in the padded tensor
    return [padded[..., lead - d: lead - d + w] for d in range(num_disp)]


class CostVolume(nn.Module):
    """Group-wise cost volume, laid out in the channel dimension.

    The original stacked hypotheses on a fifth axis and filtered them with
    nn.Conv3d, which the compiler had to decompose (HW-1). Here the D
    hypotheses live in channels, so aggregation is ordinary 2D convolution.

    Cost is a learned reduction of the feature difference. Subtraction is used
    rather than correlation because ew_sub is proven to compile in this
    pipeline; set cost="corr" to try the (usually stronger) product form once
    ew_mult is confirmed on the target.
    """

    def __init__(self, ch=32, num_disp=NUM_DISP, groups=8, cost="sub"):
        super().__init__()
        assert ch % groups == 0, "channels must divide evenly into groups"
        assert cost in ("sub", "corr")
        self.num_disp = num_disp
        self.groups = groups
        self.cost = cost
        # One shared grouped 1x1 that squeezes the 32-channel cost signature
        # down to `groups` numbers per hypothesis.
        self.reduce = nn.Conv2d(ch, groups, 1, groups=groups, bias=False)

    @property
    def out_channels(self) -> int:
        return self.num_disp * self.groups

    def forward(self, left, right):
        shifted = _shift_all(right, self.num_disp)
        slices = [
            self.reduce(left - rs if self.cost == "sub" else left * rs)
            for rs in shifted
        ]
        return torch.cat(slices, dim=1)          # [B, D*G, H/8, W/8]


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

class Aggregation(nn.Module):
    """2D hourglass over the channel-packed cost volume. Ends in one cost value
    per disparity hypothesis."""

    def __init__(self, cin, num_disp=NUM_DISP, ch=64):
        super().__init__()
        self.stem = conv_bn_act(cin, ch, 3)
        self.res_a = ResBlock(ch)
        self.res_b = ResBlock(ch)
        self.down = conv_bn_act(ch, ch, 3, stride=2)     # 1/16
        self.res_c = ResBlock(ch)
        self.res_d = ResBlock(ch)
        self.merge = conv_bn_act(ch, ch, 3)
        self.res_e = ResBlock(ch)
        # BatchNorm on the cost head is load-bearing, not routine. Without it
        # this convolution is free to scale its logits arbitrarily; measured
        # spreads reached std 34 within 100 steps, which drives the 24-way
        # softmax to one-hot (entropy 0.007 of a possible 3.18). A saturated
        # softmax passes no gradient, so the whole matching path stops learning
        # and the model locks onto whichever hypothesis it picked first.
        # BN holds the logits near unit scale and folds into the convolution at
        # export, so it costs nothing on device.
        self.out = nn.Sequential(
            nn.Conv2d(ch, num_disp, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_disp),
        )

    def forward(self, x):
        x = self.res_b(self.res_a(self.stem(x)))
        y = self.res_d(self.res_c(self.down(x)))
        # scale_factor, not size=x.shape[-2:] -- reading .shape emits an ONNX
        # Shape node, which makes the graph shape-dynamic and blocks the Hailo
        # parser. Exact here because HailoStereo asserts H,W divisible by 16.
        y = F.interpolate(y, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.res_e(x + self.merge(y))
        return self.out(x)


# --------------------------------------------------------------------------
# soft-argmin  -- defects NUM-1 and MEM-1
# --------------------------------------------------------------------------

class SoftArgmin(nn.Module):
    """Decode a cost volume to disparity, at MATCHING resolution.

    Two departures from the original:

    NUM-1  The original bilinearly upsampled all 12 cost channels to full
           resolution and ran the softmax over 5.4M elements with no scaling.
           Here the softmax runs at 1/8 over 170K elements and carries a
           learned temperature, so int8 calibration has a range to work with.

    MEM-1  The original multiplied by a materialised [1,12,368,1232] index ramp
           -- 21.8 MB, 92% of its file. Multiplying by an index ramp and summing
           over channels is exactly a 1x1 convolution with weights [0..D-1], so
           that is what this is. The ramp costs D floats.
    """

    def __init__(self, num_disp=NUM_DISP):
        super().__init__()
        self.num_disp = num_disp
        self.log_temp = nn.Parameter(torch.zeros(1))
        idx = nn.Conv2d(num_disp, 1, 1, bias=False)
        idx.weight.data = torch.arange(num_disp, dtype=torch.float32).view(1, num_disp, 1, 1)
        idx.weight.requires_grad_(False)
        self.index = idx

    def temperature(self) -> torch.Tensor:
        # Clamped so quantization never sees a degenerate logit range. The
        # upper bound is 2.0, not 5.0: with BatchNorm holding the aggregation's
        # logits near unit scale, anything above ~2 pushes the softmax back
        # toward the saturated, gradient-free regime.
        return self.log_temp.exp().clamp(0.2, 2.0)

    def forward(self, cost):
        prob = F.softmax(-cost * self.temperature(), dim=1)
        return self.index(prob)                   # [B,1,H/8,W/8], units of 1/8-res px


# --------------------------------------------------------------------------
# refinement ladder  -- defects HW-1 and ARCH-1
# --------------------------------------------------------------------------

class RefineStage(nn.Module):
    """One rung of the ladder: upsample the disparity, double it to keep pixel
    units consistent with the new scale, and predict a residual conditioned on
    guidance features from the left image.

    Channel width falls as resolution rises. The original ran six 32-channel
    residual blocks at the full 368x1232, which the compiler had to defuse into
    22 spatial slices per layer (HW-1); nothing here is wide enough to need that.
    """

    def __init__(self, guide_ch, hidden, dilations=(1, 2, 4)):
        super().__init__()
        self.stem = conv_bn_act(1 + guide_ch, hidden, 3)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dilation=d) for d in dilations])
        self.out = nn.Conv2d(hidden, 1, 3, 1, 1, bias=True)

    def forward(self, disp, guide):
        # scale_factor keeps the graph static (see Aggregation.forward); the
        # *2.0 converts disparity from the coarser scale's pixel units to this one.
        up = F.interpolate(disp, scale_factor=2.0, mode="bilinear",
                           align_corners=False) * 2.0
        x = self.stem(torch.cat([up, guide], dim=1))
        return up + self.out(self.blocks(x))


# --------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------

class HailoStereo(nn.Module):
    """Left/right RGB in, full-resolution disparity in pixels out.

    Returns a list of predictions from coarse to fine during training (for deep
    supervision); a single full-resolution tensor in eval, which is what gets
    exported.
    """

    def __init__(self, num_disp=NUM_DISP, groups=8, match_ch=32, cost="sub"):
        super().__init__()
        self.num_disp = num_disp
        self.features = FeatureNet(match_ch=match_ch)
        self.cost_volume = CostVolume(match_ch, num_disp, groups, cost=cost)
        self.aggregation = Aggregation(self.cost_volume.out_channels, num_disp)
        self.soft_argmin = SoftArgmin(num_disp)
        self.refine4 = RefineStage(guide_ch=24, hidden=32, dilations=(1, 2, 4))
        self.refine2 = RefineStage(guide_ch=16, hidden=24, dilations=(1, 2))
        self.refine1 = RefineStage(guide_ch=3, hidden=16, dilations=(1, 1))

    def forward(self, left, right):
        h, w = int(left.shape[-2]), int(left.shape[-1])
        if h % 16 or w % 16:
            raise ValueError(
                f"input must be divisible by 16, got {h}x{w}. The refinement "
                "ladder and the aggregation hourglass upsample by a fixed "
                "factor of 2, which is only exact when every scale divides."
            )
        _, _, fr8 = self.features(right)
        fl2, fl4, fl8 = self.features(left)

        cost = self.aggregation(self.cost_volume(fl8, fr8))
        d8 = self.soft_argmin(cost)          # 1/8-res pixels
        d4 = self.refine4(d8, fl4)           # 1/4-res pixels
        d2 = self.refine2(d4, fl2)           # 1/2-res pixels
        d1 = self.refine1(d2, left)          # full-res pixels

        # Clamp to the representable disparity range. The floor was always here
        # (an F.relu); the ceiling caps the refinement head, which is otherwise
        # unbounded and emits up to ~2.5x MAX_DISP in GT-free regions (sky,
        # occlusion). Measured cost to official KITTI EPE: 0.0000 px, since KITTI
        # GT never exceeds MAX_DISP. Exports to a single Clip op, free on device.
        d1 = torch.clamp(d1, 0.0, float(MAX_DISP))
        if self.training:
            # rescaled to full-resolution pixels for a single loss scale
            return [d8 * 8.0, d4 * 4.0, d2 * 2.0, d1]
        return d1

    # ----------------------------------------------------------------
    def fuse_temperature_(self):
        """Fold the soft-argmin temperature and its sign into the aggregation's
        output convolution, so the exported graph is conv -> softmax -> conv
        with no stray scalar multiply for the compiler to place."""
        t = float(self.soft_argmin.temperature())
        # The cost head ends in BatchNorm, so scale that; it in turn folds into
        # the convolution when the graph is exported in eval mode.
        bn = self.aggregation.out[-1]
        bn.weight.data.mul_(-t)
        bn.bias.data.mul_(-t)
        self.soft_argmin.log_temp.data.zero_()
        self.soft_argmin.forward = _fused_softargmin.__get__(self.soft_argmin)
        return self


def _fused_softargmin(self, cost):
    """Post-fusion soft-argmin: the negation and temperature now live in the
    weights of the convolution that produced `cost`."""
    return self.index(F.softmax(cost, dim=1))


# --------------------------------------------------------------------------

def count_ops(model, height=368, width=1232, device="cpu"):
    """MAC count by stage, so the compute budget can be checked against the
    original's 90.6%-in-refinement split before any training happens."""
    totals, handles = {}, []

    def hook(name):
        def fn(mod, inp, out):
            # weight.numel() is already Cout * Cin/groups * k * k
            macs = out.shape[-1] * out.shape[-2] * mod.weight.numel()
            group = name.split(".")[0]
            totals[group] = totals.get(group, 0) + macs
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(hook(name)))

    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, height, width, device=device)
        model(x, x)
    for h in handles:
        h.remove()

    # the siamese tower runs twice; the hook already counted both passes
    total = sum(totals.values())
    return totals, total
