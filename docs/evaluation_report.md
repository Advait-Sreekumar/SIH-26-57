# Evaluation Report — AI4Shipwrecks Sonar Hazard Detection System

**Date:** 2026-09-17  
**Scope:** All measured performance numbers for this submission. No fabricated or estimated values.

---

## 1. Primary Segmentation Performance

Model: U-Net (ResNet34 encoder, no scSE), checkpoint `best_model_v1_iou0.71.pth`.

| Split | IoU | Dice | Notes |
|-------|-----|------|-------|
| Validation (12 train sites, in-distribution) | 0.713 | 0.833 | Best epoch 55/60 |
| **Test (13 held-out sites, never seen during training)** | **0.427** | **0.598** | **Primary metric** |

**Test IoU is the headline number.** Validation IoU reflects in-distribution generalisation only.

Dataset: AI4Shipwrecks (NOAA / Thunder Bay National Marine Sanctuary), 286 images, 28 sites, CC-BY-4.0. Site-based split — 12 train, 13 test, zero overlap at site level. Split integrity verified in `docs/phase0_gap_analysis.md` Finding 8 (POSITIVE).

SOTA context: published sonar ATR benchmarks report Dice/IoU 0.55–0.77. Test IoU 0.427 is below this range. Contributing factors: small dataset (286 images), single survey region, hard 13-site held-out test, no sonar-specific augmentation in training.

---

## 2. Inference Latency Benchmark

Hardware: Intel Core i7-12700H, NVIDIA RTX 4050 Laptop GPU, CUDA 12.1, PyTorch 2.5.1+cu121.  
Images: 10 test images (2476×1728 px), seed=42. Tile counts per image: 8–32.

Timings cover: preprocess + tile inference + sigmoid + mask + connected components.  
Excludes: file I/O, geotagging, report write.

| Backend | Mean | Std | Min | Max | Model size |
|---------|------|-----|-----|-----|------------|
| PyTorch FP32 / CPU | 4.29 s | 1.75 s | 1.53 s | 6.23 s | — |
| PyTorch FP32 / GPU (RTX 4050) | 0.37 s | 0.16 s | 0.13 s | 0.54 s | — |
| ONNX FP32 / CPU | 3.08 s | 1.52 s | 0.89 s | 5.27 s | 97.7 MB |
| ONNX INT8 / CPU | 5.26 s | 1.84 s | 2.09 s | 7.06 s | 24.6 MB |

**Key findings:**
- ONNX INT8 is *slower* than FP32 on this CPU — dynamic quantisation overhead outweighs the smaller model at this tile count and image size.
- ONNX INT8 max pixel diff vs. FP32 = 0.6497 (WARN) — numerical accuracy tradeoff on top of the latency penalty. **Do not use INT8 for inference.**
- Wide latency spread (8–32 tiles per image) drives the large min/max range across backends.
- ONNX GPU (CUDA EP) not benchmarked: `onnxruntime-gpu` not in `requirements.txt`.
- ONNX FP32 numerical match vs. PyTorch: max diff 2.57e-05 (PASS).

Real-time and edge-ready are **not claimed** — those labels require a measured number on the target deployment hardware. Full data: `pipeline/benchmark_results.json`.

---

## 3. ONNX Export Validation

| File | Format | Opset | Max diff vs PyTorch FP32 | Status |
|------|--------|-------|--------------------------|--------|
| `runs/wreck_unet.onnx` | FP32 | 16 | 2.57e-05 | PASS |
| `runs/wreck_unet_int8.onnx` | INT8 dynamic QUInt8 | 16 | 0.6497 | WARN |

FP32 is the validated inference reference. INT8 retained for size reference only.

---

## 4. Cross-Domain Generalisation Check

Framing: **cross-domain generalization check (non-mine anomalous-object subset)** — not a mine-detection evaluation.

Eval dataset: Santos et al. 2024, AUV SSS, 1,170 images (license: highly likely CC-BY-4.0, not independently confirmed at repo level). Metric: IoU-of-boxes (bbox from mask connected components vs. YOLO GT bbox). **Not comparable to pixel-level test IoU 0.427.**

| Class | GT objects | Mean IoU-of-boxes | Det@0.1 | Det@0.5 |
|-------|-----------|-------------------|---------|---------|
| MILCO (man-made anomalous objects) | 437 | 0.007 | 3.0% | **0.0%** |
| NOMBO (natural seafloor features) | 231 | 0.037 | 10.8% | **2.2%** |

False positive rate on 866 background frames: **28.5%** (247/866 frames with at least one prediction; 662 total false-positive boxes).

**Finding:** Shipwreck-trained features do not transfer to the AUV SSS domain. Contributing factors:
- Frequency mismatch: 132 kHz towfish vs. 900–1800 kHz AUV (7–14× difference)
- Structurally dissimilar targets: large irregular debris fields vs. compact cylindrical objects
- Different shadow-to-target geometry (towfish grazing angle vs. AUV altitude)
- Water type: Great Lakes freshwater vs. coastal marine acoustic propagation

This result is expected and informative. It characterises the domain specificity of the learned features. Full data: `pipeline/crossdomain_results.json`.

---

## 5. Confidence Score — Calibration Status

Formula:
```
conf = 100 * (0.6 * model_score + 0.4 * (0.7 * geo + 0.3 * shadow))

model_score  = 0.5 * mean_sigmoid_prob + 0.5 * peak_sigmoid_prob
geo          = 0.40 * aspect_score + 0.25 * solidity_score + 0.35 * edge_sharpness
shadow       = target-vs-surrounding-ring median contrast (clipped 0–1)
```

- Weights are hand-tuned, not learned from data.
- No Platt scaling or isotonic regression applied — **score is not a calibrated probability**.
- `likely_rock_or_shadow` threshold (35.0) set by inspection, not empirical threshold optimisation.
- No ECE (Expected Calibration Error) measured — this is a known gap.

The score is useful as a rough within-survey ranking signal. It is labelled "Sonar-heuristic confidence score (0–100, uncalibrated)" throughout.

---

## 6. Robustness Under Sonar Perturbations

`pipeline/synthaug.py` implements four perturbation types:
- `speckle_noise(img, look)` — multiplicative Rayleigh speckle
- `synthesize_shadow(img, n_shadows, strength, direction)` — synthetic acoustic shadow bands
- `radial_distortion(img, k1, k2)` — barrel/pincushion distortion
- `heave_pitch_roll(img, max_shift, max_shear)` — platform motion simulation

Composition: `apply_sonar_synth(img, p_speckle=0.8, p_shadow=0.5, p_radial=0.3, p_heave=0.5)` — stochastic.

**Measured before/after IoU or detection-rate deltas: not available.** The augmentations are implemented and wired as a "Stress test" toggle in `app.py`, but a reproducible harness measuring clean vs. perturbed performance on the held-out test set has not been run. This is a documented gap — the perturbations are present for demonstration, not for a validated robustness claim. Writing the harness and producing measured numbers is listed as future work.

---

## 7. Test Suite Coverage

| File | Tests | Status | Scope |
|------|-------|--------|-------|
| `pipeline/test_review_store.py` | 7 | **All pass** | Review persistence, run-id isolation, action validation, FK enforcement, note normalisation, overwrite semantics, truncation |
| `pipeline/test_xtf_pipeline.py` | 0 pytest-collectable | Demo script | Has hardcoded `D:\SIH\` paths; no `def test_*` functions; not collected by pytest |

Coverage gaps: no unit tests for `preprocess.py` (Lee filter, CLAHE, TileBlender), `confidence.py` (heuristic formula, threshold behaviour), or `geotag.py` (pixel→lat/lon transform). These are documented as future work.

---

## 8. What Is Not Claimed

- No detection capability for pipes, cylinders, or ghost nets — investigated, closed with documented blockers, documented as future work
- No external sonar benchmark validation — SCTD is aerial UAV photography, invalid for sonar evaluation
- No calibrated probability output — score is an uncalibrated heuristic
- No real-time performance — measured latency is 0.37 s/image GPU (not a real-time target)
- No edge-ready claim — no embedded/microcontroller benchmark exists
- No military/defense deployment readiness
- No cross-domain generalisation — directly measured and confirmed: model does not transfer across SSS domains
- No mine-detection capability — the Santos 2024 dataset was used only to measure cross-domain failure, framed throughout as "cross-domain generalization check (non-mine anomalous-object subset)"
- No measured robustness numbers — synthaug.py implemented, harness not yet run
- No in-domain per-frame FP rate — pixel-level IoU does not measure detection-level false alarms on non-wreck frames

---

## 9. Cross-References

| Data | File |
|------|------|
| Inference benchmark | `pipeline/benchmark_results.json` |
| Cross-domain check | `pipeline/crossdomain_results.json` |
| Dataset / license audit | `docs/24574879_gonogo.md`, `docs/subpipe_gonogo.md` |
| Phase 0 audit findings | `docs/phase0_gap_analysis.md` |
| Architecture as-built | `docs/phase1_architecture.md` |
| Class expansion feasibility | `docs/phase1_5_class_expansion_feasibility.md` |
| Judge Q&A | `docs/judge_qa.md` |
