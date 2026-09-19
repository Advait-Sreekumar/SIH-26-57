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

Two benchmarks measure different pipeline scopes:

**A. Pure GPU forward pass** (source: `pipeline/benchmark_results.json`):  
Preprocessing done before the timer (`preprocessed=True`); timed window covers tiled
forward pass + sigmoid + mask + connected components only. N=10 images (2476×1728 px),
seed=42, tile counts 8–32. Excludes: file I/O, geotagging, report write.

| Backend | Mean | Std | Min | Max | Model size |
|---------|------|-----|-----|-----|------------|
| PyTorch FP32 / CPU | 4.29 s | 1.75 s | 1.53 s | 6.23 s | — |
| **PyTorch FP32 / GPU (RTX 4050)** | **0.37 s** | 0.16 s | 0.13 s | 0.54 s | — |
| ONNX FP32 / CPU | 3.08 s | 1.52 s | 0.89 s | 5.27 s | 97.7 MB |
| ONNX INT8 / CPU | 5.26 s | 1.84 s | 2.09 s | 7.06 s | 24.6 MB |

**B. Full `SonarDetector()` call including preprocessing** (source: `pipeline/perf_profile_results.json`):  
Timer covers `preprocess()` (CPU histogram equalisation + normalisation) + tiled
forward pass + sigmoid + `extract_detections()`. N=1 image (24 tiles), 3 warm runs.

| Measurement | Value |
|-------------|-------|
| Warm inference mean | **686 ms** |

The ~316 ms gap (686 ms − 370 ms) is preprocessing overhead on a ~4 MP image.
Both figures are correct — they measure different scopes.

**Key findings:**
- ONNX INT8 is *slower* than FP32 on this CPU — dynamic quantisation overhead outweighs the smaller model at this tile count and image size.
- ONNX INT8 max pixel diff vs. FP32 = 0.6497 (WARN) — numerical accuracy tradeoff on top of the latency penalty. **Do not use INT8 for inference.**
- Wide latency spread (8–32 tiles per image) drives the large min/max range across backends.
- ONNX GPU (CUDA EP) not benchmarked: `onnxruntime-gpu` not in `requirements.txt`.
- ONNX FP32 numerical match vs. PyTorch: max diff 2.57e-05 (PASS).

Real-time and edge-ready are **not claimed** — those labels require a measured number on the target deployment hardware.

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
- Frequency mismatch: 132 kHz (AI4Shipwrecks Iver3 AUV / EdgeTech 2205) vs. 900–1800 kHz AUV (7–14× difference)
- Structurally dissimilar targets: large irregular debris fields vs. compact cylindrical objects
- Different shadow-to-target geometry (AI4Shipwrecks ~2-5 m AUV altitude vs. Santos 2024 AUV survey geometry)
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

**Harness:** `pipeline/robustness_harness.py` — 120 test images, seed=42. Each image run clean then once with a single perturbation at default strength. Metric: mean pixel IoU over images where at least one prediction or ground-truth mask was present (n=102–119 depending on perturbation).

Baseline (clean, no perturbation): mean IoU = 0.0581, mean detections per image = 3.4.

**Measured per-perturbation deltas** (full data: `pipeline/robustness_results.json`):

| Perturbation | Perturbed IoU | Delta IoU | Mean detections (perturbed) | Notes |
|---|---|---|---|---|
| `speckle_noise` | 0.0128 | **-0.0453 (-78%)** | 4.7 | Largest IoU drop; multiplicative speckle degrades segmentation boundaries |
| `synthesize_shadow` | 0.0579 | -0.0002 (negligible) | 3.0 | Shadow bands have minimal effect — model is already shadow-aware |
| `radial_distortion` | 0.0525 | -0.0056 (-10%) | 4.5 | Mild geometric warp causes modest boundary errors |
| `heave_pitch_roll` | 0.0327 | **-0.0254 (-44%)** | 10.1 | Large false-positive spike (3.4 → 10.1 detections) from platform motion shear |

**Key findings:**
- Speckle noise is the most damaging perturbation to segmentation quality (-78% relative IoU).
- Heave/pitch/roll causes a 3× false-positive spike (10.1 vs 3.4 clean baseline) — motion shear creates bright edge artefacts the model incorrectly classifies as targets.
- Shadow synthesis and radial distortion have limited effect on IoU, suggesting the model has some inherent robustness to these.
- No robustness threshold is certified — these are characterisation numbers, not pass/fail guarantees.

---

## 7. Test Suite Coverage

| File | Tests | Status | Scope |
|------|-------|--------|-------|
| `pipeline/test_review_store.py` | 7 | **All pass** | Review persistence, run-id isolation, action validation, FK enforcement, note normalisation, overwrite semantics, truncation |
| `pipeline/test_pipeline.py` | 33 | **All pass** | Lee filter, CLAHE, tile/blend, preprocess, geometric confidence, shadow score, heuristic formula, pixel→lat/lon, geotag |
| `pipeline/test_xtf_pipeline.py` | 0 pytest-collectable | Demo script | Has hardcoded `D:\SIH\` paths; no `def test_*` functions; not collected by pytest |

Tests in `test_pipeline.py` found two real production bugs, now fixed:
- `confidence.py` `_component_stats`: `xs.min()` on an empty mask raised `ValueError`; now returns neutral values `(1.0, 1.0, 0.0)`.
- `geotag.py` `pixel_to_local_m`: along-track displacement was contributing to easting instead of northing; heading=0 now correctly maps higher y_px to higher latitude.

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
- No robustness claim — perturbation deltas measured (Section 6) but no robustness threshold is certified
- In-domain per-frame FP rate measured: 52.2% (24/46 GT-negative frames); 22.4% excl. Monohansett_01 outlier. Source: `pipeline/fp_rate_results.json`

---

## 9. Cross-References

| Data | File |
|------|------|
| Inference benchmark | `pipeline/benchmark_results.json` |
| Cross-domain check | `pipeline/crossdomain_results.json` |
| Robustness harness | `pipeline/robustness_results.json` |
| Dataset / license audit | `docs/24574879_gonogo.md`, `docs/subpipe_gonogo.md` |
| Phase 0 audit findings | `docs/phase0_gap_analysis.md` |
| Architecture as-built | `docs/phase1_architecture.md` |
| Class expansion feasibility | `docs/phase1_5_class_expansion_feasibility.md` |
| Judge Q&A | `docs/judge_qa.md` |
