# Judge Q&A — AI4Shipwrecks Sonar Hazard Detection System

**Prepared:** 2026-09-17  
**Status:** Evidence-backed answers only. Gaps are stated explicitly.

---

## Q1. How does the system handle domain shift — different sonar hardware, different waters, different survey geometries?

**Short answer:** It does not generalise across SSS domains, and we measured this directly.

**Detail:** The model was trained exclusively on AUV-collected side-scan sonar imagery from Thunder Bay National Marine Sanctuary (freshwater Great Lakes, ~132 kHz, Iver3 AUV / EdgeTech 2205). We ran a deliberate cross-domain check on a second real, independently collected SSS dataset (Santos et al. 2024, AUV-mounted dual-frequency 900–1800 kHz, coastal marine):

| Class | GT objects | Mean IoU-of-boxes | Det@0.5 |
|-------|-----------|-------------------|---------|
| MILCO (man-made anomalous objects) | 437 | 0.007 | 0.0% |
| NOMBO (natural seafloor features) | 231 | 0.037 | 2.2% |

False-positive rate on 866 background frames: **28.5%**

The model fires near-randomly on unfamiliar sonar texture. Contributing factors: frequency mismatch (7–14× difference, ~132 kHz vs 900–1800 kHz), different shadow geometry (altitude and grazing angle), structurally dissimilar targets (large debris fields vs. compact objects), and different water acoustic properties (freshwater Great Lakes vs. coastal marine).

This is an expected and informative result, not a failure to apologise for. It directly supports the framing we have maintained throughout: learned features are domain-specific, and generalisation across SSS hardware requires domain adaptation or retraining. Full results: `pipeline/crossdomain_results.json`.

**What it would take to fix:** Labelled training data from the target sensor platform, or domain adaptation (unsupervised feature alignment). Neither is available within this submission's scope.

---

## Q2. Is the confidence score reliable? Can it be trusted for operational decisions?

**Short answer:** No — it is an uncalibrated heuristic composite, not a probability.

**Detail:** The 0–100 score is:

```
conf = 100 * (0.6 * model_score + 0.4 * (0.7 * geo + 0.3 * shadow))

model_score = 0.5 * mean_sigmoid_prob + 0.5 * peak_sigmoid_prob
geo         = 0.40 * aspect_score + 0.25 * solidity_score + 0.35 * edge_sharpness
shadow      = target-vs-surrounding-ring median contrast
```

- Weights (0.6/0.4, 0.7/0.3, 0.40/0.25/0.35) are hand-tuned, not learned from data.
- No Platt scaling or isotonic regression applied — the score is not a calibrated probability.
- The `likely_rock_or_shadow` threshold (35.0) was set by inspection, not empirical threshold optimisation.
- `geo` and `shadow` are sonar-domain heuristics, not an acoustic propagation model.

The score is useful as a rough ranking signal within a single survey (higher = more likely to be a real target vs. shadow artifact) but should not be read as "73% probability of a shipwreck." Every UI element where the score appears labels it explicitly as "Sonar-heuristic confidence score (0–100, uncalibrated)" — not "confidence" or "probability" without qualification.

---

## Q3. Can a human override or reject detections? How does that persist?

**Short answer:** Yes — per-detection review with SQLite persistence.

**Detail:** Phase 5 implemented a human-in-the-loop review workflow:

- Per-detection expander in the UI: confirm / reject / uncertain, with optional category (`shipwreck`, `rock`, `shadow_artifact`, `biological`, `unknown`) and free-text note
- Reviews stored in `pipeline/review_store.db` (SQLite, stdlib — no server required)
- State persists across app restarts and is keyed to `(run_id, detection_id)` — the same image reopened shows the same review state
- `run_id` is derived from SHA1 of file bytes + timestamp + filename stem, so the same file always maps to the same review history
- Report export (`hazard_report.json` / `.csv`) includes `review_status` per detection: `unreviewed` / `human-confirmed` / `human-rejected` / `human-uncertain` / `human-annotate`, plus `n_confirmed` and `n_reviewed` in the JSON top-level
- `categories.py` defines the category taxonomy; `pipeline/categories.yaml` overrides it if present, so a deployment can customise categories without code changes

Test coverage: 7 tests in `pipeline/test_review_store.py` covering persistence across restart, run-id isolation, invalid action rejection, foreign-key enforcement, empty-note normalisation, overwrite semantics, and note truncation at 500 chars. All 7 pass.

---

## Q4. What is the inference latency? Is it deployable at the edge or in real time?

**Short answer:** GPU-accelerated inference is 0.37 s/image (mean); CPU-only is 4.29 s/image. We do not claim real-time or edge readiness.

**Detail (10 real test images, 2476×1728 px, RTX 4050 Laptop GPU, Intel Core i7-12700H, CUDA 12.1):**

| Backend | Mean | Min | Max | Std |
|---------|------|-----|-----|-----|
| PyTorch FP32 / CPU | 4.29 s | 1.53 s | 6.23 s | 1.75 s |
| PyTorch FP32 / GPU (RTX 4050) | 0.37 s | 0.13 s | 0.54 s | 0.16 s |
| ONNX FP32 / CPU | 3.08 s | 0.89 s | 5.27 s | 1.52 s |
| ONNX INT8 / CPU | 5.26 s | 2.09 s | 7.06 s | — |

Timings cover preprocess + tiled inference + sigmoid + mask + connected components. Excludes file I/O, geotagging, and report write.

Notes:
- ONNX INT8 is *slower* than FP32 on this CPU — dynamic quantisation overhead outweighs the smaller model on this hardware. Do not use INT8 for inference; use FP32.
- ONNX INT8 max pixel diff vs. FP32 = 0.6497 (WARN level) — numerical accuracy tradeoff on top of the latency penalty.
- Tile count per image varies (8–32 tiles depending on image dimensions), which drives the wide latency spread.
- ONNX GPU (CUDA EP) not benchmarked: `onnxruntime-gpu` is not in `requirements.txt`.
- These numbers are from a laptop GPU. A server-grade inference GPU would be faster; a Raspberry Pi or embedded system would be slower.

**"Real-time" and "edge-ready" are not claimed** anywhere in the submission — those phrases require a specific measured number on the target deployment hardware.

Full results: `pipeline/benchmark_results.json`.

---

## Q5. What hazard classes can the model detect? Why not pipes, nets, or cylinders as the SIH brief requires?

**Short answer:** Shipwrecks only. The other three classes were investigated and closed with documented reasons — not skipped.

**Detail:**

| Class | Status | Reason |
|-------|--------|--------|
| Shipwrecks | **Shipped** | AI4Shipwrecks dataset (286 images, 28 sites, CC-BY-4.0); test IoU 0.43 |
| Submerged pipes / cylinders | **Not achievable** | SubPipe dataset: GPL-3.0 license incompatible with public submission. Sonar hardware and frequency unconfirmed; bounding-box-only sonar annotations (no segmentation masks). See `docs/subpipe_gonogo.md`. |
| Ghost nets / entangled fishing gear | **Not achievable** | No public labelled sonar dataset exists anywhere; this is a field-wide data gap, not a limitation specific to this approach. Documented in `docs/phase1_5_class_expansion_feasibility.md`. |
| Generic cylinders / UXO | **Not achievable** | No real-data validation path; synthetic-only training would produce an unvalidated capability claim. Not shipped. |

Project constraint that has been maintained throughout: we will not claim detection capability for any class that has not been evaluated on at least some real (non-synthetic) sonar imagery. A class with zero real validation data gets documented as future work, not shipped as a feature.

The three closed classes are documented as a future-work roadmap in `docs/phase1_5_class_expansion_feasibility.md` with specific blockers and what would be required to resolve each.

A separate internal observation (illustrative, unvalidated, synthetic-only exploration — not a demonstrated capability) applied the unmodified shipwreck model to 30 synthetic net-like sonar patches to characterise model selectivity on out-of-distribution input; 26/30 patches (86.7%) triggered detections. This figure does not measure net-detection accuracy — it measures how often the model fires on unfamiliar texture, and is consistent with indiscriminate firing on out-of-distribution input (the same failure mode documented in the cross-domain 28.5% background FP rate and the heave/pitch/roll +196% FP increase). See `docs/synthetic_net_exploration.md` for generation method, results, and interpretation.

---

## Q6. How does the system handle false positives? What is the false-positive rate?

**Short answer:** The system flags probable false positives heuristically (score < 35 = `likely_rock_or_shadow`) and provides human review. A measured FP rate on held-out test data has not been computed on the shipwreck domain specifically — this is a known gap.

**What we have measured:**
- Cross-domain FP rate (AUV SSS background frames): 247/866 = **28.5%** — but this measures the model's behaviour on completely unfamiliar sonar texture, not its in-domain FP rate.
- The `likely_rock_or_shadow` flag at score < 35 is set by inspection on a small sample, not by optimising a threshold on a labelled FP set.
- Pixel-level test IoU 0.427 / Dice 0.598 is the primary accuracy metric, but it measures segmentation quality on known wreck sites, not false-alarm rate on non-wreck regions.

**What we have not measured:** A per-frame detection-level false-positive rate on the held-out test set (i.e., how often the model fires on a frame with no annotated wreck). This requires detection-level (rather than pixel-level) ground truth labelling that was not in scope.

This is documented as a known limitation. The human review workflow (Q3) exists precisely to catch FPs before acting on a report.

---

## Q7. Can it process a full survey log? How does it scale?

**Short answer:** Single-image and XTF log processing are implemented. Scale beyond individual images has not been benchmarked.

**Detail:**
- The pipeline accepts JPEG/PNG images or XTF sonar log files via `xtf_io.py` (pyxtf, MIT license).
- Large images are handled via tiled inference with ramp blending (`TileBlender` in `preprocess.py`) — the model never sees a raw image larger than 512×512; larger images are split into overlapping tiles and results are stitched.
- A full survey log (thousands of pings, potentially gigabytes) has not been tested end-to-end. Memory usage and per-ping latency at survey scale (e.g., 10,000 pings at 0.37 s/image GPU = ~1 hour) have not been measured.
- No streaming or chunked processing is implemented — the current XTF reader loads the entire file into memory.

**Not yet tested:** Multi-file batch processing, memory behaviour on large XTF files, or sustained throughput over a full survey transect. These are correct future-work items, not claimed capabilities.

---

## Q8. What is the training dataset? How was the train/test split designed?

**Detail:**
- **Dataset:** AI4Shipwrecks (NOAA / Thunder Bay National Marine Sanctuary) — 286 side-scan sonar images from 28 shipwreck sites. License CC-BY-4.0.
- **Split:** Site-based — 12 wreck sites for training, 13 different wreck sites for test. Zero overlap at the site level. Images from the same wreck site never appear in both splits.
- **Why site-based:** A random image split would leak context (nearby pings from the same wreck look similar). A site-based split is the minimum required to avoid the model memorising site-specific texture.

Train sites: DM_Wilson, DR_Hanna, EB_Allen, Egyptian, Grecian, Heart_Failure, Isaac_M_Scott, Montana, Near_Shore, Oscar_T_Flint, Pewabic, WP_Rend.

Test sites: Artificial_Reef, Barge_No_1, Corsair, Corsican, Haltiner_Barge, James_Davidson, Lucinda_van_Valkenburg, Monohansett, Monrovia, Shamrock, Viator, WH_Gilbert, WP_Thew.

Split integrity was verified in the Phase 0 audit (Finding 8 POSITIVE in `docs/phase0_gap_analysis.md`).

---

## Q9. How does performance compare to state of the art?

**Detail:**
- **Our test IoU: 0.427 / Dice: 0.598** (13 held-out wreck sites, never seen during training).
- **Published SOTA range** on comparable sonar segmentation benchmarks: Dice/IoU 0.55–0.77.
- Our test IoU is **below** the lower end of this SOTA range.

Context for why the gap exists:
1. Dataset size (286 images, 28 sites) is small by segmentation standards.
2. All 28 sites are from a single survey region (Thunder Bay) — limited acoustic diversity in training.
3. The test set includes 13 genuinely held-out sites with different wreck types, sizes, and orientations — this is a harder split than many published benchmarks that use random frame splits.
4. No data augmentation beyond training-time random crops/flips (no sonar-specific augmentation in training).

No cherry-picked external metric is cited. The SCTD dataset (aerial UAV photography of ships) was investigated and explicitly discarded as an out-of-domain evaluation with no diagnostic meaning for sonar tasks. See `docs/phase0_gap_analysis.md` Finding 2.

---

## Q10. Is the report output useful operationally? What does an operator receive?

**Detail:**
- `hazard_report.json` and `hazard_report.csv` are generated for each processed image.
- Per-detection fields: detection ID, bounding box (pixel and optionally lat/lon), centroid, sonar-heuristic confidence score (0–100), mean/peak sigmoid probability, aspect ratio, solidity, `likely_rock_or_shadow` flag, `review_status`, `review_category`, `review_note`, `reviewed_at`.
- Geotagging requires: origin lat/lon, heading, platform altitude, sonar range, and along-track image length (entered in the UI sidebar). Without these, lat/lon fields are null but pixel bbox is still reported.
- Geotagging uses flat-earth slant-range geometry — accurate for survey scales up to ~50 km; not suitable for global-scale or high-latitude deployments.
- The `review_status` field distinguishes `unreviewed` from `human-confirmed` / `human-rejected` so downstream consumers can filter on operator-validated detections only.

---

## Q11. What is the security posture of the demo application?

**Short answer:** Minimal. This is a local-only research demo, not a hardened application. Known gaps are documented.

**What exists:**
- Streamlit's default file uploader (no explicit size limit set).
- XTF parsing via pyxtf (MIT) — malformed-input behaviour on corrupt files has not been fuzz-tested.
- No path-traversal guards (not relevant for Streamlit's in-memory uploader).
- No secrets management (no API keys or credentials in this application).
- `.gitignore` excludes model checkpoints (`*.pth`, `*.onnx`) and dataset images from version control.

**Known gaps:**
- No file-size limit on upload — a very large file could exhaust memory.
- No explicit timeout on inference — a pathological input could cause the request to hang.
- No formal security audit conducted — this is correctly out of scope for a hackathon timeline, and we document it as such rather than implying a review was done.

For a production deployment these would need to be addressed before exposing the service to untrusted users.

---

## Q12. Is this a mine-detection system?

**No.** This project does not build, claim, or imply a mine-detection capability.

The Santos et al. 2024 dataset (MILCO/NOMBO annotations) was used for one purpose only: a cross-domain generalisation check — measuring whether features trained on shipwreck imagery transfer at all to a different SSS sensor domain. The result (MILCO Det@0.5 = 0.0%, NOMBO Det@0.5 = 2.2%) shows they do not. This is documented as "cross-domain generalization check (non-mine anomalous-object subset)" throughout and is an informative characterisation of the model's domain specificity, not a mine-detection evaluation.

Per the project's explicit constraint list: no claims of military/defense deployment readiness appear anywhere in this submission.

---

## Q13. How robust is the model to sonar acquisition artefacts?

**Short answer:** Varies by perturbation type. Speckle noise and platform motion are the most damaging; shadow synthesis and radial distortion have minimal effect. Measured on 120 held-out test images.

`pipeline/robustness_harness.py` ran each of the four `synthaug.py` perturbation types against the full held-out test set (120 images, seed=42). Baseline clean mean IoU = 0.0581.

| Perturbation | Delta IoU | Relative drop | False-positive change |
|---|---|---|---|
| Speckle noise | -0.0453 | -78% | +38% more detections |
| Shadow synthesis | -0.0002 | negligible | stable |
| Radial distortion | -0.0056 | -10% | +31% more detections |
| Heave/pitch/roll | -0.0254 | -44% | **+196% more detections** (3.4 → 10.1) |

The heave/pitch/roll result is the most operationally significant: platform motion shear creates bright edge artefacts that the model misclassifies as wreck targets, tripling false-positive count. This is a documented limitation — no robustness threshold is certified; these are characterisation numbers. Full data: `pipeline/robustness_results.json`.
