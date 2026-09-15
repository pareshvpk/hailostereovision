"""
Datasets for stereo disparity training: SceneFlow (pretrain) and KITTI 2015
(finetune).

Both yield the same contract:

    left   float32 [3,H,W]  normalized
    right  float32 [3,H,W]  normalized
    disp   float32 [1,H,W]  disparity in full-resolution pixels
    mask   float32 [1,H,W]  1 where disp is valid AND matchable -- metrics
    weight float32 [1,H,W]  loss weight; 1 matchable, 0.2 on the left border

The `mask` is not cosmetic. It carries the CV-2 fix: the first MAX_DISP columns
of any stereo pair reference right-image coordinates that do not exist, so the
cost volume there is padding, not evidence. Training against those columns
teaches the network to fit its own zero padding. They are excluded here, and
the same mask is used at evaluation.
"""

from __future__ import annotations

import pathlib
import random
import re
import struct

import numpy as np
import torch
from torch.utils.data import Dataset

from model import MAX_DISP

# ImageNet statistics, matching the normalization Hailo's parser applies on-chip
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# file readers
# ---------------------------------------------------------------------------

def read_pfm(src) -> np.ndarray:
    """SceneFlow ships disparity as PFM. Returns a float32 [H,W] array.

    Accepts a path or an already-open binary file object, so PFMs can be read
    straight out of a streaming tar without ever hitting disk.
    """
    fh = open(src, "rb") if isinstance(src, (str, pathlib.Path)) else src
    close = fh is not src
    try:
        header = fh.readline().rstrip()
        if header not in (b"PF", b"Pf"):
            raise ValueError(f"{src}: not a PFM file")
        colour = header == b"PF"

        line = fh.readline()
        while line.startswith(b"#"):
            line = fh.readline()
        m = re.match(rb"^(\d+)\s+(\d+)\s*$", line)
        if not m:
            raise ValueError(f"{src}: malformed PFM dimensions")
        width, height = int(m.group(1)), int(m.group(2))

        scale = float(fh.readline().rstrip())
        endian = "<" if scale < 0 else ">"

        count = width * height * (3 if colour else 1)
        data = np.frombuffer(fh.read(count * 4), dtype=endian + "f4")
        data = data.reshape((height, width, 3) if colour else (height, width))
        # PFM rows run bottom-to-top
        return np.flipud(data).astype(np.float32).copy()
    finally:
        if close:
            fh.close()


def read_kitti_disp(path) -> np.ndarray:
    """KITTI 2015 disparity: 16-bit PNG, value/256 in pixels, 0 means invalid."""
    from PIL import Image
    raw = np.array(Image.open(path), dtype=np.float32)
    return raw / 256.0


# fetch_driving.py stores SceneFlow disparity as uint16 PNG at 1/32 px, which
# is ~14x smaller than the float32 PFM and far finer than this model resolves.
SCENEFLOW_PNG_SCALE = 32.0


def read_sceneflow_disp(path) -> np.ndarray:
    """Read SceneFlow disparity in either the original PFM or the converted
    uint16 PNG form, chosen by extension."""
    path = pathlib.Path(path)
    if path.suffix.lower() == ".pfm":
        return read_pfm(path)
    from PIL import Image
    return np.array(Image.open(path), dtype=np.float32) / SCENEFLOW_PNG_SCALE


def read_image(path) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# shared sample handling
# ---------------------------------------------------------------------------

def _normalize(img: np.ndarray) -> np.ndarray:
    return ((img - MEAN) / STD).transpose(2, 0, 1)


def _colour_jitter(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Mild per-view photometric jitter. Applied asymmetrically to the two
    views so the network cannot rely on exact intensity equality -- which is
    what makes a subtraction cost robust on real, unbalanced camera pairs."""
    gain = rng.uniform(0.8, 1.2)
    bias = rng.uniform(-0.06, 0.06)
    gamma = rng.uniform(0.85, 1.15)
    out = np.clip(img * gain + bias, 0.0, 1.0) ** gamma
    return out.astype(np.float32)


# The left MAX_DISP columns carry no stereo evidence -- their matches lie
# outside the right image -- but they are still 15% of every output frame, and
# leaving them out of the loss entirely leaves them untrained. Measured on the
# KITTI finetune, the untrained border emitted a mean error of 47.8 px and
# accounted for 84% of all error in 13% of the pixels. Supervising them at
# reduced weight lets the network learn to extrapolate from monocular cues and
# from continuity with the matchable region, without letting a region that
# cannot be matched dominate the gradient.
BORDER_WEIGHT = 0.2


def _valid_mask(disp: np.ndarray) -> np.ndarray:
    """Where the ground truth exists, is in range, and has a real match (CV-2).

    This is the MATCHING mask: it gates every reported metric, so numbers stay
    comparable across runs. Training uses `_loss_weight` instead.
    """
    mask = (disp > 0.0) & (disp < float(MAX_DISP)) & np.isfinite(disp)
    mask[:, :MAX_DISP] = False
    return mask.astype(np.float32)


def _loss_weight(disp: np.ndarray) -> np.ndarray:
    """Per-pixel loss weight: 1.0 where matchable, BORDER_WEIGHT on the left
    edge where the ground truth is valid but no match exists, 0 elsewhere."""
    in_range = (disp > 0.0) & (disp < float(MAX_DISP)) & np.isfinite(disp)
    weight = in_range.astype(np.float32)
    weight[:, :MAX_DISP] *= BORDER_WEIGHT
    return weight


def _occlude(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Asymmetric occlusion: paint one random rectangle of the RIGHT view with
    its mean colour. Real pairs contain regions visible to only one camera; a
    subtraction cost that has never seen one learns to trust every match. The
    left view and the ground truth are untouched, so the network must fill the
    hole from context -- the same skill the left border and occlusions need."""
    h, w, _ = img.shape
    bh, bw = rng.randint(50, 100), rng.randint(50, 100)
    y, x = rng.randint(0, h - bh), rng.randint(0, w - bw)
    out = img.copy()
    out[y:y + bh, x:x + bw] = img.reshape(-1, 3).mean(axis=0)
    return out


def _channel_jitter(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Per-channel gain, so the two views can also disagree in white balance."""
    gains = np.array([rng.uniform(0.9, 1.1) for _ in range(3)], dtype=np.float32)
    return np.clip(img * gains, 0.0, 1.0)


AUGMENTATIONS = ("basic", "strong")


# The diagnosed defect (IMPROVEMENT_PLAN.md §1.2) is that only ~1% of KITTI 2015
# GT is >= this many pixels, so the network never learns to read large disparity.
# Disparity-aware sampling (train.py --disp-aware-sampling) oversamples the frames
# that carry any such GT and biases their crops toward the rows that hold it.
FAR_DISP = 80.0


class _StereoBase(Dataset):
    def __init__(self, samples, crop=(256, 512), training=True, seed=0,
                 aug="basic", far_crop_bias=False, far_crop_prob=0.5):
        if aug not in AUGMENTATIONS:
            raise ValueError(f"aug must be one of {AUGMENTATIONS}, got {aug!r}")
        self.samples = samples
        self.crop = crop
        self.training = training
        self.aug = aug
        self.seed = seed
        # When set, training crops of a frame that contains any GT >= FAR_DISP
        # are, with probability far_crop_prob, positioned to include one of the
        # rows that holds that GT -- so the oversampling is not wasted on a crop
        # that happens to miss the large-disparity object entirely.
        self.far_crop_bias = far_crop_bias
        self.far_crop_prob = far_crop_prob
        self._rng = random.Random(seed)
        self._rng_owner = None
        if not samples:
            raise RuntimeError(
                "no stereo pairs found -- check the dataset root path")

    def _read_disp(self, dp):
        """Read one disparity map, in this dataset's encoding. Overridden per
        subclass; used by far_frame_flags without loading the images too."""
        raise NotImplementedError

    def far_frame_flags(self, thresh=FAR_DISP):
        """Per-sample bool: does this frame carry any valid, in-range GT
        disparity >= thresh? One disparity read per frame (no images), used to
        build the WeightedRandomSampler for disparity-aware oversampling."""
        flags = []
        for _, _, dp in self.samples:
            d = self._read_disp(dp)
            valid = (d > 0.0) & (d < float(MAX_DISP)) & np.isfinite(d)
            flags.append(bool((valid & (d >= thresh)).any()))
        return flags

    @property
    def rng(self) -> random.Random:
        """A generator private to the current DataLoader worker.

        A Random built in __init__ is pickled into every worker as an identical
        copy, so with --workers 4 all four workers drew the SAME crop offsets and
        jitter values in lockstep -- a quarter of the augmentation diversity the
        loader appeared to provide. torch gives each worker a distinct seed
        (and a fresh one per epoch unless workers persist); reseed from it.
        """
        info = torch.utils.data.get_worker_info()
        owner = None if info is None else (info.id, info.seed)
        if owner != self._rng_owner:
            self._rng = random.Random(self.seed if info is None else info.seed)
            self._rng_owner = owner
        return self._rng

    def __len__(self):
        return len(self.samples)

    def _crop_y(self, disp, h, ch):
        """Vertical crop origin. Uniform by default; with far_crop_bias set and
        the frame carrying large-disparity GT, with probability far_crop_prob
        the window is placed to contain one such row (see FAR_DISP)."""
        if self.far_crop_bias and self.rng.random() < self.far_crop_prob:
            valid = (disp > 0.0) & (disp < float(MAX_DISP)) & np.isfinite(disp)
            far_rows = np.where((valid & (disp >= FAR_DISP)).any(axis=1))[0]
            if far_rows.size:
                r = int(far_rows[self.rng.randrange(far_rows.size)])
                lo = max(0, r - ch + 1)
                hi = min(h - ch, r)
                if lo <= hi:
                    return self.rng.randint(lo, hi)
        return self.rng.randint(0, h - ch)

    def _finish(self, left, right, disp):
        h, w = disp.shape
        ch, cw = self.crop

        if self.training:
            if h < ch or w < cw:
                raise ValueError(f"image {h}x{w} smaller than crop {ch}x{cw}")
            y = self._crop_y(disp, h, ch)
            x = self.rng.randint(0, w - cw)
            left = left[y:y + ch, x:x + cw]
            right = right[y:y + ch, x:x + cw]
            disp = disp[y:y + ch, x:x + cw]
            left = _colour_jitter(left, self.rng)
            right = _colour_jitter(right, self.rng)
            if self.aug == "strong":
                left = _channel_jitter(left, self.rng)
                right = _channel_jitter(right, self.rng)
                if self.rng.random() < 0.5:
                    right = _occlude(right, self.rng)
        else:
            # crop to a multiple of 16, anchored bottom-right the way KITTI
            # evaluation crops, so the horizon stays in frame
            nh, nw = (h // 16) * 16, (w // 16) * 16
            left, right, disp = left[h - nh:, w - nw:], right[h - nh:, w - nw:], disp[h - nh:, w - nw:]

        return {
            "left": torch.from_numpy(_normalize(left)),
            "right": torch.from_numpy(_normalize(right)),
            "disp": torch.from_numpy(disp)[None],
            "mask": torch.from_numpy(_valid_mask(disp))[None],
            "weight": torch.from_numpy(_loss_weight(disp))[None],
        }


# ---------------------------------------------------------------------------
# SceneFlow
# ---------------------------------------------------------------------------

class SceneFlow(_StereoBase):
    """Any SceneFlow subset (FlyingThings3D, Driving, Monkaa).

    Expects the standard layout, with `frames_cleanpass` (or finalpass) and
    `disparity` trees sharing a directory structure:

        <root>/frames_cleanpass/.../left/0006.png
        <root>/disparity/.../left/0006.pfm

    Also accepts the compact layout written by `fetch_driving.py`:

        <root>/frames/.../left/0006.webp
        <root>/disparity/.../left/0006.png     (uint16, 1/32 px)

    The image pass directory and both file formats are auto-detected, so the
    same class serves the original download and the converted one.
    """

    def __init__(self, root, crop=(256, 512), training=True, pass_name=None,
                 limit=None, seed=0, aug="basic", far_crop_bias=False,
                 far_crop_prob=0.5):
        root = pathlib.Path(root)
        image_root = self._find_image_root(root, pass_name)
        disp_root = root / "disparity"

        samples = []
        for lp in sorted(image_root.rglob("left/*")):
            if lp.suffix.lower() not in (".png", ".webp", ".jpg"):
                continue
            # .../<scene>/left/0006.x  ->  .../<scene>/right/0006.x
            rp = lp.parent.parent / "right" / lp.name
            if not rp.exists():
                continue
            stem = disp_root / lp.relative_to(image_root)
            # converted PNG first, original PFM as a fallback
            dp = next((c for c in (stem.with_suffix(".png"), stem.with_suffix(".pfm"))
                       if c.exists()), None)
            if dp is not None:
                samples.append((lp, rp, dp))

        if not samples:
            raise RuntimeError(
                f"no stereo pairs under {root}. Expected images in "
                f"{image_root} and disparity in {disp_root}.")
        if limit:
            samples = samples[:limit]
        super().__init__(samples, crop, training, seed, aug,
                         far_crop_bias=far_crop_bias, far_crop_prob=far_crop_prob)

    @staticmethod
    def _find_image_root(root: pathlib.Path, pass_name: str | None) -> pathlib.Path:
        if pass_name:
            return root / pass_name
        for candidate in ("frames", "frames_cleanpass", "frames_cleanpass_webp",
                          "frames_finalpass"):
            if (root / candidate).is_dir():
                return root / candidate
        raise RuntimeError(
            f"no image directory found under {root} -- looked for frames/, "
            f"frames_cleanpass/, frames_cleanpass_webp/, frames_finalpass/")

    def _read_disp(self, dp):
        return read_sceneflow_disp(dp)

    def __getitem__(self, i):
        lp, rp, dp = self.samples[i]
        return self._finish(read_image(lp), read_image(rp), read_sceneflow_disp(dp))


# ---------------------------------------------------------------------------
# KITTI 2015
# ---------------------------------------------------------------------------

class Kitti2015(_StereoBase):
    """KITTI 2015 training split (200 pairs with `disp_occ_0` ground truth).

    `split` selects a deterministic train/val partition; KITTI publishes no
    validation labels, so the last 40 scenes are held out.
    """

    def __init__(self, root, crop=(256, 512), split="train", n_val=40, seed=0,
                 aug="basic", far_crop_bias=False, far_crop_prob=0.5):
        root = pathlib.Path(root)
        lefts = sorted((root / "image_2").glob("*_10.png"))
        samples = []
        for lp in lefts:
            rp = root / "image_3" / lp.name
            dp = root / "disp_occ_0" / lp.name
            if rp.exists() and dp.exists():
                samples.append((lp, rp, dp))
        if not samples:
            raise RuntimeError(
                f"no KITTI pairs under {root} -- expected image_2/, image_3/, disp_occ_0/")
        samples = samples[:-n_val] if split == "train" else samples[-n_val:]
        super().__init__(samples, crop, training=(split == "train"), seed=seed,
                         aug=aug, far_crop_bias=far_crop_bias,
                         far_crop_prob=far_crop_prob)

    def _read_disp(self, dp):
        return read_kitti_disp(dp)

    def __getitem__(self, i):
        lp, rp, dp = self.samples[i]
        return self._finish(read_image(lp), read_image(rp), read_kitti_disp(dp))


# ---------------------------------------------------------------------------
# KITTI 2012
# ---------------------------------------------------------------------------

class Kitti2012(_StereoBase):
    """KITTI 2012 training split (194 pairs with `disp_occ` ground truth).

    Same acquisition rig and encoding as KITTI 2015 -- 16-bit PNG, value/256 in
    full-resolution pixels, 0 invalid -- so `read_kitti_disp` and the whole
    _StereoBase contract carry over unchanged. Only the directory names differ:
    `colored_0`/`colored_1` for the rectified colour pair (there are grayscale
    `image_0`/`image_1` trees too; the colour ones match how KITTI 2015 is read)
    and `disp_occ` for the all-pixels ground truth.

    It exists to attack the diagnosed defect (IMPROVEMENT_PLAN.md §1.2): KITTI
    2015 has only 1% of GT >= 80 px, and 194 more labelled real-world pairs more
    than doubles the finetune set (160 -> 354). All frames are used for training;
    validation stays the KITTI 2015 40-frame hold-out so numbers stay comparable.
    Only the `*_10.png` reference frame of each pair carries disparity.
    """

    def __init__(self, root, crop=(256, 512), training=True, seed=0, aug="basic",
                 far_crop_bias=False, far_crop_prob=0.5):
        root = pathlib.Path(root)
        lefts = sorted((root / "colored_0").glob("*_10.png"))
        samples = []
        for lp in lefts:
            rp = root / "colored_1" / lp.name
            dp = root / "disp_occ" / lp.name
            if rp.exists() and dp.exists():
                samples.append((lp, rp, dp))
        if not samples:
            raise RuntimeError(
                f"no KITTI 2012 pairs under {root} -- expected "
                f"colored_0/, colored_1/, disp_occ/ with *_10.png")
        super().__init__(samples, crop, training=training, seed=seed, aug=aug,
                         far_crop_bias=far_crop_bias, far_crop_prob=far_crop_prob)

    def _read_disp(self, dp):
        return read_kitti_disp(dp)

    def __getitem__(self, i):
        lp, rp, dp = self.samples[i]
        return self._finish(read_image(lp), read_image(rp), read_kitti_disp(dp))
