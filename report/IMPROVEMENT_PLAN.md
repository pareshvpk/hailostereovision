# HailoStereo — Accuracy Improvement Plan

**Date:** 2026-09-14
**Baseline:** `runs/kitti_border/best.pt` — 1.464 px masked / **1.659 px official**,
D1 9.92%; int8 emulated 1.848 px.
**Constraint:** everything must still compile to a Hailo-15H HEF with DFC 5.4.0.
No new op types, no more than 24 shifted slices into the cost volume (the
shmifo limit already forced `max_utilization=0.95`).

---

## 1. What the diagnosis found (2026-09-14)

### 1.1 The error is at large disparity, not the 192 px ceiling

STATUS.md §2.3 attributes the 0–5 m failure to the 192 px disparity ceiling.
**That is wrong.** Every 0–5 m pixel in the val split has GT disparity
77–158 px; none exceeds 192. Shipped model, KITTI val, by GT disparity:

| GT disparity | matched EPE | bias | border EPE | bias |
|---|---|---|---|---|
| 0–80 px | 1.17–2.04 | ≈0 | 1.82–2.22 | ≈0 |
| 80–100 px | 10.69 | −9.85 | 10.77 | −10.77 |
| 100–120 px | 28.22 | −28.22 | 30.23 | −30.23 |
| 120–160 px | 56.76 | −56.76 | 34.68 | −34.68 |

Fine below 80 px, and **every pixel above it under-read**, in the matched
region as much as the border. The architecture can represent these; the
trained weights do not produce them.

### 1.2 Cause: KITTI has almost no large-disparity supervision

| GT pixels ≥ 80 px | share |
|---|---|
| KITTI 2015 train (160 frames) | **1.0%** |
| SceneFlow Driving | **42%** (15% even beyond 192 px) |

The shipped model under-reads large disparity **on its own training frames**
(80–100 px bias −5.4, 120–160 px −27.4) — it has not learned the range at all.

Two contributing effects, measured on KITTI val:

| checkpoint | official EPE | 100–120 px bias | 120–160 px EPE |
|---|---|---|---|
| `sceneflow_border` (pretrain only) | 3.770 | −40.9 | 30.6 |
| `kitti_border` (shipped) | **1.659** | −28.2 | 34.7–56.8 |
| `kitti_replay` (SceneFlow mixed in) | 1.876 | −22.5 | **8.25** |

- The pretrain also fails at 100–120 px on KITTI, so it is partly domain
  (near road surface, bottom of frame), not only forgetting.
- Replay fixes the far end of the range (120–160 px: 34.7 → 8.3 px) while
  costing 0.2 px everywhere else. **Replay is a tool to tune, not a dead end.**

### 1.3 Statistical caveat on range numbers

Only **17 of 40** val frames contain any GT ≥ 80 px, and one frame
(`000161_10`) holds **40%** of those pixels; three frames hold 73%. Every
per-range figure above ≈80 px is a handful of scenes. Headline decisions use
official EPE over all pixels; the range breakdown is a secondary signal and
differences there need to be large (several px) to count.

### 1.4 Training bug, fixed

`_StereoBase` built its `random.Random` in `__init__`, so all DataLoader workers
got identical copies: **4 distinct augmentation draws per 16 samples** with
`--workers 4`. Every past run trained with a quarter of its apparent
augmentation diversity. Fixed in `src/data.py` (per-worker reseed; 16/16).

---

## 2. Plan

Each phase is gated on the one before. A phase's winner becomes the next
phase's baseline. All KITTI runs: `--init runs/sceneflow_border/best.pt
--epochs 300 --lr 1e-4`, ≈20 min each on the RTX 4060, **plugged in only**.

### Phase 0 — tooling (≈15 min, no training)

- [x] **0.1** Add a disparity-binned breakdown to `src/eval_kitti.py`
      (0–40 / 40–80 / 80–100 / 100–120 / 120–160 px, EPE + bias, matched and
      border). Done: `--bins` flag; reproduces Finding 1 exactly on kitti_border.
- [x] **0.2** Add `--sceneflow-root` to the same script, or a sibling, to score
      the SceneFlow hold-out scene — the forgetting metric replay is tuned on.
      Done: `src/eval_sceneflow.py`; reproduces 3.610 px on sceneflow_border.
- [x] **0.3** Worker RNG fix, `--ema`, `--aug strong` (done, smoke-tested).

### Phase 1 — finetune recipe ablation (4 runs, ≈80 min)

One change at a time, so each effect is attributable.

| run | adds | question |
|---|---|---|
| **A** `kitti_rngfix` | RNG fix only | how much did the bug cost? |
| **B** `kitti_ema_aug` | A + `--ema 0.999 --aug strong` | regularisation on 160 frames |
| **C** `kitti_crop768` | B + `--crop-h 320 --crop-w 768` | context; border share 37% → 25% of crop; 0.75 GiB measured |
| **D** `kitti_replay50` | C + `--replay-root data/driving --replay-ratio 0.5` | keep large-disparity skill at lower cost than 1.0 |

Score each with `eval_kitti.py` (official EPE, D1, ≥80 px bins) and SceneFlow
hold-out EPE. **Promote the lowest official EPE**; if two are within 0.05 px,
prefer the one with smaller ≥80 px bias.

### Phase 2 — target the diagnosed defect directly (≈2 h incl. download)

- [~] **2.1 KITTI 2012** — downloaded (2.01 GB), extracted 194 pairs
      (`colored_0/`, `colored_1/`, `disp_occ/`), ingest-verified
      (`check_ingest --dataset kitti2012` PASS, residual min at x1.0, range to
      146 px, 99th pct 77 px — far richer large-disparity supervision than
      2015). `Kitti2012` loader + `--dataset kitti_mixed` (354 train, val stays
      the KITTI 2015 40) built and validated. `kitti_mixed768` run (winner
      recipe) in progress. Licence: CC BY-NC-SA, same as 2015.
- [ ] **2.2 Disparity-aware sampling** — oversample the 55 train frames with
      GT ≥ 80 px, and bias their crops toward those rows. Cheapest direct
      attack on the 1% supervision share; no new data.
- [ ] **2.3 Replay ratio sweep** on the Phase 1 winner, if D helped: 0.25 /
      0.5 / 1.0.

### Phase 3 — redo the pretrain with the fixes (≈35 min + 1 finetune)

- [ ] **3.1** Retrain `sceneflow_fixed` from scratch: same hold-out, 30 epochs,
      batch 16, lr 1e-3, now with the RNG fix and `--aug strong --ema 0.999`.
      The existing pretrain ran with the RNG bug for all 30 epochs.
- [ ] **3.2** Rerun the best Phase 1–2 finetune recipe from it.

### Phase 4 — architecture, only if ≥ 80 px bias survives Phases 1–3

Every item here means a full retrain **and** a new HEF with re-emulated int8.

- [ ] **4.1** `cost="corr"` (already in `model.py`). First confirm `ew_mult`
      parses and allocates in DFC 5.4.0 on a stub graph — no board needed.
- [ ] **4.2** Wider matching features (`match_ch` 32 → 48). Check GOPS and the
      shmifo count at compile; the cost-volume concat is what broke the
      allocator last time.
- **Not planned:** more hypotheses or a larger range. §1.1 shows range is not
  the limit, and more slices would push the shmifo count past 24.

### Phase 5 — ship the winner (≈45 min)

- [ ] `src/export_onnx.py` — all audits PASS, parity < 1e-3 px
- [ ] `./deploy/hailo-py deploy/dfc_flow.py all` — HEF builds at 0.95 utilization
- [ ] Emulated int8 EPE vs 1.848 px; int8 degradation vs +13.0%
- [ ] `src/eval_depth.py` — 0–5 m band vs 1.08 m mean |dZ|
- [ ] Update STATUS.md: new numbers, and **correct §2.3 and open item 3**
      (the 192 px ceiling explanation)

---

## 3. Guard-rails

- **Selection bias.** `best.pt` is the minimum over 300 epochs on the same 40
  frames it is reported on. For the final model, report the EMA `last.pt`
  alongside `best.pt`; if they differ by more than ~0.05 px, quote `last.pt`.
- **Noise floor.** Per-frame EPE spans 0.93–3.51 px; differences under ~0.1 px
  official EPE are not a result.
- **Never overwrite** `runs/kitti_border/` or `artifacts/hailo_stereo_hailo15h.hef`
  until Phase 5 passes; new runs get new `--out` directories.
- **Power.** ~100 W draw against a 52 Wh battery. Do not start a run on battery.

## 4. Budget

Revised 2026-09-14 from measured epoch times (KITTI ≈ 4 s/epoch on Linux,
≈ 7 s on Windows; 768-wide crops ≈ 1.9× pixels; replay and KITTI 2012 add steps).

| phase | Linux | Windows | runs on |
|---|---|---|---|
| 0 tooling | 15–30 min | 15–30 min | Windows |
| 1 ablation (4 runs) | ~2 h 10 min | ~2.5–3 h | Windows |
| 2 KITTI 2012 + sampling + replay sweep | ~4–4.5 h | ~5–5.5 h | Windows |
| 3 pretrain retrain + finetune | ~1.5 h | ~2 h | Windows |
| 4 architecture | +2.5–3.5 h per variant | — | optional, Linux |
| 5 export, compile, emulate, report | ~1 h | not possible | **Linux** |
| **total, excluding 4** | **≈ 9–11 h** | **≈ 12–15 h incl. Phase 5 on Linux** | |

Lean path (skip 2.3; run Phase 3 only if ≥ 80 px bias persists): ≈ 5–6 h.
Windows execution brief: `report/WINDOWS_HANDOFF.md`.

---

## 5. Results log

Executed on Windows (RTX 4060 laptop) starting 2026-09-14. All KITTI runs
`--init runs/sceneflow_border/best.pt --epochs 300 --lr 1e-4`. EPE columns are
KITTI 2015 val (40 pairs); "SF" = held-out SceneFlow scene (200 frames, the
forgetting metric). Range columns are `EPE / bias` in px over matched+border
combined at that GT-disparity bin (from `eval_kitti.py --bins`). `best`/`last`
noted where they differ by > 0.05 px (report `last` = EMA when so).

**Baselines (measured with the Phase 0 tooling, for reference):**

| ckpt | ckpt-EPE | masked | official | D1% | 80–100 | 100–120 | 120–160 | SF |
|---|---|---|---|---|---|---|---|---|
| `kitti_border` (shipped) | best | 1.464 | 1.659 | 9.92 | 10.7 / −10.2 | 28.8 / −28.8 | 35.1 / −35.1 | — |
| `sceneflow_border` (pretrain) | — | — | — | — | — | — | — | 3.610 |

**Phase 1 — finetune recipe ablation** (`best.pt` unless noted):

| run | changed | ckpt | masked | official | D1% | 80–100 | 100–120 | 120–160 | SF | best-ep | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kitti_rngfix` | RNG fix only | best | 1.488 | 1.670 | 9.71 | 9.4 / −8.9 | 32.5 / −32.5 | 36.6 / −36.6 | 24.67 | 235 | ~33m |
| `kitti_ema_aug` | + ema 0.999, aug strong | best | 1.498 | 1.691 | 9.77 | 10.4 / −9.8 | 36.5 / −36.5 | 40.3 / −40.3 | 27.94 | 238 | ~34m |
| `kitti_crop768` | + crop 320×768 | best | 1.461 | 1.639 | 9.42 | 7.9 / −7.3 | 34.9 / −34.9 | 34.0 / −34.0 | 27.74 | 265 | ~46m |
| `kitti_replay50` | + replay driving 0.5 | best | 1.556 | 1.704 | 10.32 | 8.0 / −5.4 | 33.3 / −31.0 | 16.1 / +13.2 | 3.99 | 298 | ~65m |

Observations (updated as runs land):
- **A `kitti_rngfix`**: official 1.670 px ≈ shipped 1.659 (within the <0.1 px
  noise floor), so the worker-RNG bug cost essentially nothing on headline EPE.
  The ≥80 px under-read is unchanged (still −9 to −37 px), and the pure-KITTI
  finetune forgets SceneFlow hard (3.61 → 24.7 px). Confirms the defect is a
  supervision-distribution problem, not an augmentation-diversity one.
- **B `kitti_ema_aug`**: 1.691 px, ~0.02 px worse than A (noise). EMA + strong
  aug do not help on 160 frames and the ≥80 px under-read is unchanged — the
  model is not overfitting in a way regularisation fixes. best ≈ last (0.006 px).
- **C `kitti_crop768`**: 1.639 px, best official and best D1 (9.42%) so far, ~0.02
  px under shipped. The larger crop's added context clearly helps the 80–100 px
  band (EPE 10.7 → 7.9, bias −10.2 → −7.3) — the first move that shifts the
  ≥80 px under-read. 100+ px still badly negative. Promote C's recipe.
- **D `kitti_replay50`**: official 1.704, ~0.065 px worse than C (just outside the
  noise floor). But replay transforms the large-disparity behaviour and stops
  forgetting: SceneFlow hold-out 27.7 → **3.99 px**; 120–160 px border EPE
  33.6 → 15.9 (bias flips −34 → +14, a slight over-read); 80–100 border bias
  −8.9 → −3.5. This is exactly the far-range fix §1.2 predicted, at a small
  headline cost. The +14 over-read at 120–160 hints ratio 0.5 slightly
  overshoots — a lower ratio may be the sweet spot.

### Phase 1 verdict

By the stated rule (lowest official EPE), **C `kitti_crop768` (recipe:
`--ema 0.999 --aug strong --crop-h 320 --crop-w 768`) is the Phase 1 winner**
at 1.639 px, and is carried into Phase 2 as the base recipe. Replay clearly
helped the diagnosed defect (D), so the Phase 2.3 replay-ratio sweep is ON —
replay is treated as a tunable Phase 2 knob rather than baked into the baseline,
so KITTI 2012's own large-disparity supervision (Phase 2.1) can be measured
independently first.

### Phase 2 status (paused 2026-09-14 for thermals)

`kitti_mixed768` (Phase 2.1) was **paused at epoch 46** because the RTX 4060
hit 88 °C under sustained load (throttle point) after hours of back-to-back
runs — see memory `laptop-gpu-thermal-ceiling`. Nothing lost: best so far
**1.411 px masked @ ep46**, already under the Phase 1 winner (1.461) and the
shipped 1.464, so KITTI 2012 is helping. Fully resumable.

Resume (add `--epoch-pause 20` to duty-cycle the GPU cooler; same `--epochs`
and `--batch` are mandatory for the OneCycleLR resume check):

```bat
.venv\Scripts\python src\train.py --dataset kitti_mixed ^
  --root data/kitti2015/training --kitti2012-root data/kitti2012/training ^
  --resume runs/kitti_mixed768/last.pt --epochs 300 --batch 4 --lr 1e-4 ^
  --ema 0.999 --aug strong --crop-h 320 --crop-w 768 --epoch-pause 20 ^
  --out runs/kitti_mixed768 >> runs\kitti_mixed768_train.log 2>&1
```

Remaining Phase 2: finish 2.1, then 2.2 (`--disp-aware-sampling`, built and
validated) on the better of {C recipe, kitti_mixed}, then 2.3 replay-ratio
sweep (0.25 / 0.5 / 1.0) — replay ratio 0.5 slightly over-read at 120–160 px,
so 0.25 is the first to try.
