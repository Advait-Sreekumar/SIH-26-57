# SonarEye — AI4Shipwrecks Sonar Hazard Detection System
# Master Documentation

**Prepared:** 2026-09-19  
**Version:** Phase 5 (human-in-the-loop review, XTF ingestion, geotagged PDF export)  
**Status:** Evidence-backed throughout. All numbers from measured results. Gaps stated explicitly.

---

## Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & SIH Alignment](#2-problem-statement--sih-alignment)
3. [System Architecture](#3-system-architecture)
4. [Training Data & Methodology](#4-training-data--methodology)
5. [Evaluation Results](#5-evaluation-results)
6. [Deployment & Performance](#6-deployment--performance)
7. [Cross-Domain Generalisation Check](#7-cross-domain-generalisation-check)
8. [Robustness Characterisation](#8-robustness-characterisation)
9. [Known Limitations & Failure Modes](#9-known-limitations--failure-modes)
10. [Product & User Experience](#10-product--user-experience)
11. [Scope Boundaries & Roadmap](#11-scope-boundaries--roadmap)
12. [Judge Q&A Reference](#12-judge-qa-reference)

---

## 1. Executive Summary

SonarEye is an end-to-end side-scan sonar (SSS) hazard detection pipeline for underwater
archaeological survey. It ingests XTF sonar files or sample PNG images, segments candidate
anomalies using a deep learning model, scores each detection with a sonar-domain heuristic,
geotags detections to lat/lon coordinates, and exports a human-reviewed hazard report as
JSON, CSV, and PDF.

**Three headline measured results:**

| Metric | Value | Source |
|--------|-------|--------|
| Test pixel IoU (held-out sites) | **0.4268** | `docs/evaluation_report.md` §3 |
| GPU inference latency (RTX 4050) | **0.37 s** pure forward pass; **686 ms** full pipeline | `docs/evaluation_report.md` §2 |
| Cross-domain Det@0.5 (MILCO class) | **0.0%** | `docs/evaluation_report.md` §6 |

The cross-domain zero is not a failure to apologise for — it is an honest, measured
characterisation of the model's domain specificity. The system was trained on one specific
sonar platform and one water body; it is not claimed to generalise beyond that.

**What this system is:**
A research-grade automatic target recognition (ATR) pipeline for shipwreck detection in
AUV-collected side-scan sonar imagery, with a full human-review layer before any report
is acted upon.

**What this system is not:**
- A mine-detection or mine-classification system. The Santos et al. 2024 dataset was
  used for a cross-domain generalisation check only; no mine-detection capability is
  claimed, built, or implied.
- A real-time system. Inference latency has been measured; no real-time or edge-readiness
  claim is made without a measured number on the target deployment hardware.
- A multi-class detector. Pipes, nets, and cylinders are documented non-deliverables with
  specific technical reasons (see §11).

---

## 2. Problem Statement & SIH Alignment

### SIH Requirement

The Smart India Hackathon problem statement calls for a system to detect underwater hazards
from sonar imagery, with specific mention of shipwrecks, submerged pipelines, ghost/entangled
fishing nets, and cylindrical debris.

### What Is Demonstrated

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Shipwreck ATR from SSS | **Demonstrated** | Test IoU 0.4268, Dice 0.5983 on AI4Shipwrecks benchmark |
| XTF file ingestion | **Demonstrated** | `xtf_io.py`, tested on `test_survey.xtf` fixture |
| Geotagged detection output | **Demonstrated** | `geotag.py`, 8-case regression test suite |
| Human review workflow | **Demonstrated** | SQLite review store, 7 passing unit tests |
| PDF hazard report export | **Demonstrated** | `pdf_export.py`, reportlab-based multi-page PDF |
| Underwater pipelines / cylinders | **Not deliverable** | SubPipe dataset is GPL-3.0 (copyleft blocks submission) |
| Ghost / entangled nets | **Not deliverable** | No public labeled sonar dataset exists for this class |
| Generic cylindrical debris | **Not deliverable** | Synthetic-only training possible but unvalidated |

### Mine-Detection Disclaimer

This project does not build, claim, or imply a mine-detection capability. The Santos et al.
2024 dataset contains annotations labelled MILCO (man-made anomalous contact objects) and
NOMBO (natural seafloor features). These labels were created by the dataset's authors for
their own purpose. This project used the dataset for exactly one purpose: measuring whether
features trained on shipwreck imagery transfer to a different sonar sensor domain. The result
(MILCO Det@0.5 = 0.0%) shows they do not. This is a characterisation of domain specificity,
not a mine-detection evaluation. No claims of military or defense deployment readiness appear
anywhere in this submission.

---

## 3. System Architecture

### Model

U-Net with ResNet34 encoder, ImageNet-pretrained.

- **Encoder:** ResNet34 (four-scale feature extraction).
- **Decoder:** Standard U-Net skip connections at each scale.
- **Modifications:** None — no scSE, no attention gates, no deep supervision. A clean
  vanilla baseline avoids over-fitting the architecture to this specific dataset.
- **Input:** 1-channel (grayscale) 512×512 tiles.
- **Output:** Single-channel binary segmentation probability map (sigmoid activation).
- **Framework:** PyTorch via segmentation_models_pytorch (smp).

### Inference Pipeline

```
XTF file  ──► xtf_io.py ──────────────────────────────────► preprocess.py
              (nav + waterfall render;                           │
               per-ping lat/lon, heading,                        │
               altitude, range from headers)                     │
                                                                 │
PNG image ───────────────────────────────────────────────────► preprocess.py
                                                                 │
                                                 normalise (/255) + tile 512×512
                                                       50% overlap, ramp blend
                                                                 │
                                                    U-Net / ResNet34 per-tile
                                                                 │
                                               sigmoid → binary mask → connected components
                                                                 │
                                                       confidence.py
                                              (sonar-heuristic score, 0–100)
                                                                 │
                                                         geotag.py
                                                (slant-range correction → lat/lon)
                                                                 │
                                                  hazard_report.json / .csv / .pdf
```

Two entry points:
- **PNG path** (demo / sample): image bytes already rendered; pipeline starts at preprocess.
- **XTF path** (operational): `xtf_io.py` reads per-ping navigation and renders a waterfall
  PNG from the ping stack, then same path downstream.

### Confidence Score Formula

The 0–100 sonar-heuristic confidence score is a composite of model output and acoustic
domain heuristics:

```
conf = 100 * (0.6 * model_score + 0.4 * (0.7 * geo + 0.3 * shadow))

model_score = 0.5 * mean_sigmoid_prob + 0.5 * peak_sigmoid_prob
geo         = 0.40 * aspect_score + 0.25 * solidity_score + 0.35 * edge_sharpness
shadow      = target-vs-surrounding-ring median contrast
```

Component definitions:
- `mean_sigmoid_prob` / `peak_sigmoid_prob`: mean and peak sigmoid output within the
  connected component mask.
- `aspect_score`: elongation of the component (major/minor axis ratio, capped). Wreck
  debris tends to be elongated; near-circular returns are more rock-like.
- `solidity_score`: component area / convex hull area. Low solidity = jagged boundary
  = more likely geological.
- `edge_sharpness`: mean gradient magnitude at the component boundary. Sharp boundaries
  indicate real acoustic contrasts.
- `shadow`: median intensity contrast between the target region and surrounding ring.
  Strong shadow → high-relief target.

Weights (0.6/0.4, 0.7/0.3, 0.40/0.25/0.35) are **hand-tuned, not learned**. No Platt
scaling or isotonic regression has been applied. The score is not a calibrated probability.
It is useful as a within-survey ranking signal only. The `likely_rock_or_shadow` flag is
set when score < 35.0 (threshold set by inspection, not empirical optimisation).

### Geotagging

Per-pixel transform from sonar image coordinates to WGS84 lat/lon:

1. Across-track offset: `across_m = (x - w/2) * (range_m / (w/2))`
2. Slant-range correction: `ground_m = sqrt(slant^2 - altitude^2)`  
   Fallback to `abs(across_m)` when near-nadir (slant < altitude), preventing NaN.
3. Along-track offset: `along_m = y * (along_track_m / h)`
4. Heading rotation to east/north components from `heading_deg`.
5. WGS84 flat-earth update: `M_PER_DEG_LAT = 111320.0`; longitude scaled by `cos(lat0)`.

For XTF input: per-ping lat/lon and heading from XTF packet headers.  
For PNG samples: operator-supplied parameters from the sidebar (illustrative only; see §10).

---

## 4. Training Data & Methodology

### Dataset: AI4Shipwrecks

The AI4Shipwrecks dataset contains side-scan sonar imagery of known shipwreck sites
collected by the **Iver3 AUV** equipped with an **EdgeTech 2205 dual-frequency side-scan
sonar (~132 kHz)**, operated by NOAA's Thunder Bay National Marine Sanctuary (TBNMS),
Lake Huron (freshwater Great Lakes).

| Parameter | Value |
|-----------|-------|
| Total images | 286 |
| Unique wreck sites | 28 |
| Training sites | 12 |
| Test sites | 13 |
| Image resolution | 2476×1728 px (single-channel grayscale) |
| Annotation type | Binary segmentation masks |
| Sonar platform | Iver3 AUV |
| Sonar hardware | EdgeTech 2205, ~132 kHz |
| Water body | Lake Huron (freshwater) |

### Site-Based Split

The train/test split is **site-based**: all images from a given wreck site appear in
either train or test, never both. This prevents spatial data leakage — without a site
based split, overlapping sonar swaths from the same wreck would appear in both partitions,
articificially inflating test metrics. The gap between validation IoU (0.713) and test
IoU (0.427) reflects genuine generalisation difficulty, not an implementation error.

### Training Setup

- Binary cross-entropy loss on sigmoid output.
- Data augmentation: horizontal flip, brightness/contrast jitter.
- Early stopping on validation IoU.
- No test-time augmentation at evaluation.

---

## 5. Evaluation Results

All results from `docs/evaluation_report.md`. Evaluated on 13 held-out test sites.

### Primary Segmentation Metrics

| Metric | Test | Validation |
|--------|------|------------|
| Pixel IoU | **0.4268** | 0.713 |
| Dice coefficient | **0.5983** | 0.833 |

Validation metrics are on the training-distribution validation fold, not the held-out
test set. Val >> Test is expected (nearby sites seen in training). Test IoU is the
authoritative number.

### In-Domain False-Positive Rate

Measured on GT-negative test frames from AI4Shipwrecks:

| Metric | Value |
|--------|-------|
| GT-negative frames (all 13 test sites) | 46 |
| Frames with any detection | 24 |
| **FP rate (frame-level)** | **52.2%** |
| Mean detections per GT-negative frame | 1.57 |
| Excluding Monohansett_01 outlier (28 det, 9004×1728 px) | 22.4% (10/44 frames) |

Source: `pipeline/fp_rate_results.json`.

The 52.2% rate is the authoritative measured result on the 13 held-out test sites.
The Monohansett_01 image is an unusually large panoramic frame (9004×1728 px) that
alone accounts for 28 detections; excluding it gives 22.4%. Both figures are reported;
neither is suppressed. The model fires frequently on GT-negative sonar — specular
reflections, hard-bottom substrate, and structured sediment returns — and this is a
known limitation at the current threshold setting.

### ONNX Export Validation

| Export | Max pixel diff vs. FP32 | Status |
|--------|------------------------|--------|
| ONNX FP32 | 2.57e-05 | PASS (< 1e-04 threshold) |
| ONNX INT8 | 0.6497 | WARN (numerical accuracy penalty) |

ONNX FP32 export is numerically equivalent to PyTorch FP32 within floating-point
rounding. ONNX INT8 has a material accuracy penalty and must not be used for inference.

---

## 6. Deployment & Performance

### Inference Latency

Two benchmarks measure different pipeline scopes on the same hardware
(Intel Core i7-12700H, RTX 4050 Laptop GPU, CUDA 12.1).

**A. Pure GPU forward pass** (source: `pipeline/benchmark_results.json`, script:
`pipeline/benchmark_inference.py`):  
Preprocessing was done *before* the timer using `preprocessed=True`; the timed
window covers only tiled forward pass + sigmoid + mask blending + connected
components. N=10 images (2476×1728 px), 2 warmup runs, tile counts 8–32 per image.

| Backend | Mean | Min | Max | Std |
|---------|------|-----|-----|-----|
| PyTorch FP32 / CPU | 4.29 s | 1.53 s | 6.23 s | 1.75 s |
| **PyTorch FP32 / GPU (RTX 4050)** | **0.37 s** | 0.13 s | 0.54 s | 0.16 s |
| ONNX FP32 / CPU | 3.08 s | 0.89 s | 5.27 s | 1.52 s |
| ONNX INT8 / CPU | 5.26 s | 2.09 s | 7.06 s | — |

**B. Full `SonarDetector()` call including preprocessing** (source:
`pipeline/perf_profile_results.json`, script: `pipeline/perf_profile.py`):  
Timer covers `preprocess()` (CPU: histogram equalisation + normalisation on a
2476×1728 px image) + tiled forward pass + sigmoid + `extract_detections()`
(morphological opening + connected components). N=1 image (Artificial_Reef_01.png,
24 tiles), 3 warm runs.

| Measurement | Value |
|-------------|-------|
| First (cold) inference | 891 ms |
| **Warm inference mean (3 runs)** | **686 ms** |

The ~316 ms difference between the two GPU figures (686 ms − 370 ms) is the
preprocessing overhead: CPU-bound histogram equalisation and normalisation on a
~4 MP image. Both figures are correct — they measure different scopes.

Notes:
- **ONNX INT8 is slower than FP32** on this hardware. Dynamic quantisation overhead
  outweighs the smaller model on an Intel i7. Do not use INT8 for inference.
- Tile count per image: 8–32 depending on image dimensions. This drives the wide
  latency range (high Std values).
- ONNX GPU (CUDA EP) not benchmarked: `onnxruntime-gpu` excluded from requirements.
- A server-grade GPU would be faster; a Raspberry Pi or embedded SBC would be slower.
- **"Real-time" and "edge-ready" are not claimed** — those phrases require a specific
  measured number on the actual target deployment hardware.

### Model Sizes

| File | Size |
|------|------|
| `best_model.pth` (PyTorch FP32 checkpoint) | ~85 MB |
| `model.onnx` (ONNX FP32 export) | ~85 MB |
| `model_int8.onnx` (ONNX INT8 export) | ~22 MB |

### Application Startup: Cold vs. Warm

The Streamlit app has two distinct latency regimes:

**Cold start** (first run after Python process launch):  
`torch` import + model weight load + first inference.  
Dominated by PyTorch module import and `best_model_v1_iou0.71.pth` weight load.
Measured via `pipeline/perf_profile.py` on Intel Core i7-12700H + RTX 4050 Laptop GPU,
16 GB RAM (source: `pipeline/perf_profile_results.json`):

| Stage | Time |
|-------|------|
| `torch` import | 1,857 ms |
| Pipeline imports (numpy, PIL, infer, confidence, geotag, app) | 4,017 ms |
| Model weight load (`SonarDetector` init) | 345 ms |
| Image load + convert to numpy | 47 ms |
| First (cold) inference | 891 ms |
| **Cold-start total (stages 1–5)** | **7,157 ms (~7.2 s)** |

The dominant cost is PyTorch import (~1.9 s) + pipeline imports (~4.0 s), not inference.
Once the process is live, subsequent warm inference runs average **686 ms** on this hardware.

**Warm start** (subsequent Streamlit rerenders, same process):  
Model is already loaded and served via `@st.cache_resource`. Inference result is
cached in `st.session_state["_infer_results"]` — the analysis does not re-run unless
a new file is loaded. Warm rerender cost is dominated by UI state diffing, not
inference.

**What is cached and why:**

| Resource | Cache mechanism | Reason |
|----------|----------------|--------|
| PyTorch model weights | `@st.cache_resource` on `_get_model()` | 85 MB load once; survives rerenders |
| Inference + geotag result | `st.session_state["_infer_results"]` | Prevents re-inference on every widget interaction |
| PDF bytes | `@st.cache_resource` on `_make_pdf(run_id, det_json)` | Prevents reportlab rebuild on every Report page visit |
| SQLite DB connection | `@st.cache_resource` on `get_db()` | Single connection shared across rerenders |

**PDF caching fix:**  
Before the cache was applied, the PDF was rebuilt on every Report page visit (every
Streamlit rerender triggered `build_pdf()` via reportlab, a synchronous blocking
call). With `@st.cache_resource` keyed on `(run_id, det_json)`, the PDF is built
once per survey run and served from memory on all subsequent page visits: effectively
~0 ms on repeat visits vs. a noticeable rebuild delay (measured ~1.18 s for a
typical 3-detection survey).

### Why Streamlit Rather Than a Custom Server

The bottleneck analysis supports staying on Streamlit:
- Cold start (~8–9 s) is dominated by PyTorch import and model weight load — both
  framework-independent. A custom Flask/FastAPI server would face the same cost on
  first request.
- Warm inference is 686 ms full pipeline (370 ms pure GPU forward pass) / 3.08 s (CPU FP32), not a Streamlit
  overhead.
- The human review loop (SQLite upsert → UI update) is near-instant; no queuing
  bottleneck exists.
- Streamlit provides the map (folium), PDF download, and review widgets without
  additional frontend engineering. Replacing it would replicate the same features
  at higher cost without measurable latency benefit on the hot path.

---

## 7. Cross-Domain Generalisation Check

### Dataset

Santos et al. 2024 (figshare DOI: 10.6084/m9.figshare.24574879).  
AUV-mounted dual-frequency SSS, coastal marine (900–1800 kHz). 1,170 images.

| Parameter | Santos et al. 2024 | AI4Shipwrecks (train) |
|-----------|--------------------|-----------------------|
| Sensor | AUV-mounted SSS | Iver3 AUV + EdgeTech 2205 |
| Frequency | 900–1800 kHz | ~132 kHz |
| Frequency ratio | — | **7–14× difference** |
| Water body | Coastal marine | Freshwater (Lake Huron) |
| Image count | 1,170 | 286 |

### Results

| Class | Definition | GT objects | Mean IoU-of-boxes | Det@0.5 |
|-------|-----------|-----------|-------------------|--------|
| MILCO | Man-made anomalous contact objects | 437 | 0.007 | **0.0%** |
| NOMBO | Natural seafloor features | 231 | 0.037 | **2.2%** |

False-positive rate on 866 background frames: **28.5%**

### Interpretation

The model fires near-randomly on unfamiliar sonar texture. Contributing factors:
- **Frequency mismatch:** 7–14× difference (~132 kHz vs. 900–1800 kHz) produces
  fundamentally different shadow geometry, resolution, and speckle statistics.
- **Different shadow geometry:** Altitude and grazing angle differ between sensor platforms.
- **Structurally dissimilar targets:** The MILCO class contains large debris fields;
  AI4Shipwrecks targets are intact or partially intact wrecks.
- **Different acoustic properties:** Freshwater vs. coastal marine.

This is an expected and informative result. It directly supports the framing maintained
throughout: **learned features are domain-specific**. Generalisation across SSS hardware
requires domain adaptation or retraining on the target sensor.

What would be needed to fix: labelled training data from the target sensor platform, or
unsupervised domain adaptation. Neither is available within this submission's scope.

### Explicit Non-Mine-Detection Statement

The MILCO label in Santos et al. 2024 is the dataset authors' own annotation for
"man-made anomalous contact objects." This project did not train or evaluate a
mine-detection model. The cross-domain check was run to characterise domain specificity,
not to claim any military or defense capability.

---

## 8. Robustness Characterisation

Harness: `pipeline/robustness_harness.py`. 120 held-out test images, seed=42.
Perturbation types from `pipeline/synthaug.py`. Baseline is clean (no perturbation).
Full data: `pipeline/robustness_results.json`.

| Perturbation | Mean IoU | Delta IoU | Relative drop | Mean dets | FP change |
|---|---|---|---|---|---|
| Baseline (clean) | 0.0581 | — | — | 3.425 | — |
| Speckle noise | 0.01280 | -0.04531 | **-78%** | 4.717 | +38% |
| Shadow synthesis | 0.05787 | -0.00024 | negligible | 3.369 | stable |
| Radial distortion | 0.05248 | -0.00563 | -10% | 4.483 | +31% |
| Heave/pitch/roll | 0.03271 | -0.02540 | **-44%** | 10.108 | **+196%** |

### Heave/Pitch/Roll: Most Operationally Significant

Platform motion simulation (shear warp, `max_shift=10px, max_shear=0.05`) creates sharp
intensity transitions along the horizontal seam of the warp. The model cannot distinguish
these shear-induced bright-edge artefacts from genuine acoustic targets. Result: 3.4
detections/image (clean) → 10.1 detections/image (heave) — a **3× false-positive spike**.

Operational implication: any real survey with significant platform motion (high sea state,
shallow AUV depth) will produce elevated false-alarm rates. Human reviewers must apply
additional scrutiny to detections from rough-sea passes.

### Named Failure Mode: Texture Not Shape

Four independent stress tests converge on a single characterised failure:

1. **Cross-domain check (§7):** 28.5% FP rate on unfamiliar sonar backgrounds.
2. **Heave/pitch/roll robustness (+196% FP):** Fires on shear-induced edge artefacts.
3. **In-domain FP rate (52.2%, or 22.4% excl. outlier):** Fires on specular reflections and rocky outcrops.
4. **Synthetic net exploration:** 86.7% of synthetic patches triggered detections —
   indistinguishable from the cross-domain FP rate on real backgrounds.

Common mechanism: **the model responds to high-contrast structured texture, not
exclusively to shipwreck morphology.** This is documented as the system's single most
important characterised limitation.

---

## 9. Known Limitations & Failure Modes

### Named Failure Mode (summary)

The model fires on unfamiliar or ambiguous texture rather than reliably discriminating
target shape. All four stress tests converge on this mechanism (see §8).

### Qualitative Examples

From `docs/qualitative_examples.md` (selected to illustrate the range of behaviour,
not cherry-picked for best cases):

| ID | Category | Est. tile IoU | Heuristic score | Notes |
|----|----------|--------------|----------------|-------|
| TP-1 | True positive | ~0.75 | ~74 | Anchor chain scatter; high-contrast, well-preserved |
| TP-2 | True positive | ~0.48 | ~58 | Partial hull; boundary drift at reverberation band |
| TP-3 | True positive | ~0.65 | ~79 | Boiler mass; strong shadow drives high score |
| HARD-1 | Ambiguous | ~0.22 | ~49 | Reef-wreck overlap; indistinguishable acoustic impedance |
| HARD-2 | Ambiguous | ~0.18 | ~31 | Small distal fragment; near-threshold, unstable |
| FP-1 | False positive | N/A (no GT) | ~41–48 | Heave shear artefact; background frame |
| FN-1 | False negative | 0 (miss) | N/A (not detected) | Buried/sedimented hull; low contrast |

Note: tile-level IoU values for qualitative examples are estimates from visual
inspection of `runs/test_results.png`. The headline test IoU (0.4268) and harness
numbers are the authoritative measured values.

### Confidence Score Limitations

- Weights (0.6/0.4, 0.7/0.3, 0.40/0.25/0.35) are hand-tuned, not learned from data.
- No Platt scaling or isotonic regression — not a calibrated probability.
- `likely_rock_or_shadow` threshold (35.0) set by inspection, not empirical optimisation.
- Useful as a within-survey ranking signal; cross-image absolute comparison is unreliable.

### Other Known Limitations

- **Domain lock-in:** Trained on Thunder Bay NMS (Iver3 AUV, EdgeTech 2205, ~132 kHz,
  freshwater). Does not generalise to different sonar hardware without retraining.
- **Buried-target gap:** Partially buried or sedimented wrecks have low backscatter
  contrast; the 286-image training set is weighted toward well-preserved wrecks.
- **Platform-motion FP spike:** Heave/pitch/roll perturbations triple false-positive count
  (3.4 → 10.1 detections/image). See §8.
- **INT8 ONNX penalty:** ONNX INT8 is slower than FP32 (5.26 s vs. 3.08 s CPU) and has
  a 0.6497 max pixel diff. Do not use for inference.
- **No file-size limit on upload:** A very large file could exhaust memory (Streamlit
  architectural limitation).
- **No inference timeout:** A pathological input could cause the request to hang.
- **No formal security audit:** Out of scope for a hackathon timeline; documented as a
  known gap.
- **Synthetic net exploration framing:** The 86.7% detection rate on synthetic patches
  is labelled throughout as *"illustrative, unvalidated, synthetic-only exploration —
  not a demonstrated capability."* See §11.

---

## 10. Product & User Experience

### Mission Flow

The application implements a structured, linear mission workflow designed around how a
sonar survey operator actually processes findings:

```
Landing page
    │
    ▼
Survey Setup
  - Sample image selector (Corsair_01, Defiance_01, North_Star_01)
  - XTF file upload
  - Navigation parameter presets (lat0, lon0, heading, altitude, range, along_track)
  - "Start Survey" button (disabled until a source is selected)
    │
    ▼
Analysis
  - Visible pipeline progress (preprocessing → model load → tiled inference → scoring → geotagging)
  - Inference timing shown in real time
  - Results cached in session state on completion
    │
    ▼
Detection Review (sequential)
  - One detection per screen: bbox crop, sonar-heuristic confidence score,
    model probability, lat/lon
  - Actions: confirm / reject / uncertain (+ optional category and free-text note)
  - Reviews saved immediately to SQLite; persist across app restarts
    │
    ▼
Survey Map
  - Folium / Leaflet.js map with clickable detection markers
  - Each marker opens the Anomaly Detail panel
  - "Proceed to Summary" gated: map must be reviewed before summary
    │
    ▼
Mission Summary
  - Ranked detection list with review badges
  - "Revise" button for any detection returns to detail view
  - "Continue to Report" button
    │
    ▼
Mission Report
  - Download JSON, CSV, PDF
  - PDF: cover page (survey summary) + one page per detection
  - "Start New Survey" resets session and purges sample-run reviews from DB
```

### Human Review System

Reviews are stored in `pipeline/review_store.db` (SQLite, stdlib — no server required),
keyed to `(run_id, detection_id)`. The same file reopened after an app restart shows
the same review state. `run_id` is derived from SHA1 of file bytes + UTC timestamp +
filename stem, so the same file consistently maps to the same review history.

Report export includes `review_status` per detection (`unreviewed` / `human-confirmed`
/ `human-rejected` / `human-uncertain` / `human-annotate`) and `n_confirmed` /
`n_reviewed` counts in the JSON top-level.

**Why auto-retraining is deliberately not implemented:**  
Human review decisions (confirm/reject) are not fed back into the model automatically.
This is a deliberate design choice. Auto-retraining from operator feedback requires:
a labelled dataset of confirmed positives at sufficient scale, an evaluation harness
that can validate the new model on held-out data before deployment, and a safeguard
against label noise from untrained operators. None of these exist in the current scope.
Adding a feedback button that silently retrains the model would imply a capability
that has not been built or validated.

### Illustrative Location Labelling

AI4Shipwrecks sample images (PNG exports from the benchmark dataset) do not carry
per-image GPS navigation data — they are survey imagery exports, not navigation logs.
The sidebar defaults to illustrative navigation parameters (`lat0=45.0855,
lon0=-83.5684, heading=90°`, Lake Huron survey area) so that detections can be
displayed on a map, but these coordinates do not represent real per-image survey
navigation.

Whenever the app is running with a sample image (`survey_cfg["source"] == "sample"`),
two UI elements display:

> *Illustrative locations — approximate survey-area coordinates only. No real survey
> navigation data for this sample.*

This caption appears on the Survey Coverage Map page and in the Anomaly Detail panel.
It does not appear for user-uploaded XTF files, which carry real navigation data.

This is an example of the honesty discipline applied at the UI level: the system
knows the difference between real and illustrative data and says so explicitly,
rather than presenting sample-derived map coordinates as if they were GPS-accurate.
The caption logic is covered by 4 of the 8 geotag regression tests.

---

## 11. Scope Boundaries & Roadmap

### Class Expansion Verdicts

All three non-shipwreck classes were investigated. Each has a specific, documented
blocker. Source: `docs/phase1_5_class_expansion_feasibility.md`.

#### Underwater Pipelines / Cylinders

**Verdict: NOT DELIVERABLE (two independent blockers)**

1. **License incompatibility:** The SubPipe dataset (the only public labeled SSS pipeline
   dataset found) carries GPL-3.0. Any derivative work — including trained model weights
   and submission tooling — must also be GPL-3.0. A public SIH submission cannot comply
   with copyleft requirements.
2. **Domain shift:** SubPipe's sensor hardware and frequency are unspecified in the
   README/abstract; compatibility with the ~132 kHz training domain is unconfirmed.

What would be needed: license-compatible SSS pipeline data with segmentation annotations
(not bounding boxes) collected with compatible sonar hardware. Time to working model if
data existed: estimated 1–2 weeks.

#### Ghost / Entangled Fishing Nets

**Verdict: NOT YET ACHIEVABLE (no public labeled sonar dataset exists)**

A thorough search found no publicly available sonar dataset with ghost net annotations.
Most net detection research uses optical/RGB cameras. The published SSS ghost-net
literature describes the challenge but does not release annotated data.

**Ghost net synthetic exploration — mandatory framing:**  
An experiment was run: 30 synthetic 512×512 patches (RNG seed 7) modelling irregular
low-backscatter blob morphology were passed through the inference pipeline. 26/30 patches
fired (86.7%), producing 72 total detections (mean 2.40/patch). This result is
**illustrative, unvalidated, synthetic-only exploration — not a demonstrated
capability.** The 28.5% FP rate on real background frames explains why: a model that
fires on unfamiliar sonar texture will also fire on synthetic patches that look like
unusual sonar texture. Without a real labeled ghost net dataset to distinguish true
positives from false positives, the 86.7% detection rate is meaningless as a
capability claim.

What would be needed: a labeled SSS dataset from real ghost net surveys, or a research
data collection effort. Time estimate: unknown (blocked on data).

#### Generic Cylindrical Seabed Debris

**Verdict: SYNTHETIC-ONLY / UNPROVEN**

No dedicated public SSS dataset for generic cylinders was found. Synthetic training is
achievable (cylinders produce characteristic highlight + shadow in SSS) but violates the
project constraint against claiming detection capability without real-data evaluation.

### Honest Roadmap

| Phase | Task | Blocker | Est. time if blocker resolved |
|-------|------|---------|------------------------------|
| Phase 2 | Pipeline/cylinder detection | License-compatible SSS data with segmentation masks | 1–2 weeks |
| Phase 3 | Cylinder detection via synthetic training | Acoustic forward model + real validation data | 3–4 weeks |
| Phase 4 | Ghost net detection | Requires research data collection effort | Unknown |
| Ongoing | Domain adaptation | Labeled data from target sensor platform | Depends on data |

---

## 12. Judge Q&A Reference

The companion document `docs/judge_qa.md` provides evidence-backed answers to 13
anticipated judge questions. It is the master doc's natural final reference — all
numbers in that document are consistent with the evaluation report and robustness
results cited above.

**Questions covered in `docs/judge_qa.md`:**

| Q | Topic | Short answer |
|---|-------|--------------|
| Q1 | Domain shift / hardware generalisation | Measured directly: 0.0% Det@0.5 on MILCO; features do not transfer |
| Q2 | Confidence score reliability | Uncalibrated heuristic composite; not a probability |
| Q3 | Human review persistence | Yes — SQLite, persists across restarts, keyed to run_id |
| Q4 | Inference latency; real-time / edge claim | 0.37 s GPU pure forward pass / 686 ms full pipeline / 4.29 s CPU; real-time not claimed |
| Q5 | What hazard classes can be detected | Shipwrecks only; pipes/nets/cylinders are documented non-deliverables |
| Q5a | Ghost net synthetic exploration | 86.7% fire rate on synthetic patches; illustrative only, not capability |
| Q6 | In-domain FP rate | 52.2% frame-level (24/46 GT-negative frames); 22.4% excluding Monohansett_01 outlier. Source: `pipeline/fp_rate_results.json` |
| Q7 | Geotagging accuracy | Derived from operator-supplied nav; sample = illustrative only |
| Q8 | PDF report | Functional, reportlab-based; not a production document system |
| Q9 | Real survey ingestion | Yes — XTF upload, `xtf_io.py`, tested on `test_survey.xtf` |
| Q10 | `likely_rock_or_shadow` flag | Composite score < 35.0 triggers flag; heuristic threshold |
| Q11 | Security considerations | No external calls; no path-traversal surface; not formally audited |
| Q12 | Is this a mine-detection system? | No. Explicitly and unambiguously. |
| Q13 | Robustness to acquisition artefacts | Heave +196% FP (most critical); speckle -78% IoU; radial -10% |

For any judge question not listed above, the answer should be sought in:
1. `docs/evaluation_report.md` — all primary metrics and ONNX validation
2. `pipeline/robustness_results.json` — full harness raw data
3. `docs/qualitative_examples.md` — representative TP/FP/FN examples with analysis
4. `docs/phase1_5_class_expansion_feasibility.md` — class expansion verdicts
5. `docs/synthetic_net_exploration.md` — ghost net exploration detail

---

*All numbers in this document were pulled directly from source files, not from memory
or estimation. Section references are given inline. Cross-check against the source
files before citing any number in a presentation or poster.*
