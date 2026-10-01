# AI4Shipwrecks — Sonar Hazard Detection System

Smart India Hackathon (SIH) project: automatic detection of man-made underwater hazards
in side-scan sonar (SSS) imagery, with physics-informed confidence scoring and
geotagged hazard reports.

## Honest Performance Summary

| Split | IoU | Dice | Notes |
|-------|-----|------|-------|
| Validation (site-based, within training domain) | **0.713** | **0.833** | Best epoch 55/60; site-based split, no leakage |
| **Test (held-out sites, never seen during training)** | **0.427** | **0.598** | Primary evaluation metric |

**The test IoU (0.427) is the correct headline number.** The validation IoU (0.713) reflects
in-distribution generalisation within the same survey region. The gap between val and test
is expected given the small dataset (286 images, 28 shipwreck sites from Thunder Bay NMS)
and domain shift across survey campaigns.

SOTA range on comparable sonar segmentation benchmarks: Dice/IoU 0.55–0.77.
Test IoU 0.427 is currently below this range.

## Current Detection Scope

**The model detects shipwrecks only.** It is trained exclusively on the AI4Shipwrecks
dataset (Thunder Bay National Marine Sanctuary shipwreck surveys).

Detection of pipes, cylinders, and ghost/entangled nets is **not achievable in the
current version** — investigated and closed for this submission:
- **Pipes/cylinders (SubPipe):** dataset found, but GPL-3.0 license blocks use in a
  public submission; sonar hardware/frequency unconfirmed; bounding-box-only SSS
  annotations (no segmentation masks). See `docs/subpipe_gonogo.md`.
- **Ghost nets:** no public labeled sonar dataset exists anywhere; this is a
  field-wide data gap, not a limitation specific to this approach
- **Generic cylinders:** no real-data validation path; synthetic-only training would
  be unvalidated capability

All three are documented as future work in `docs/phase1_5_class_expansion_feasibility.md`.

## Sonar-Heuristic Confidence Score — Not a Probability

The 0–100 "confidence" score shown in the UI is a **heuristic composite**, not a
calibrated probability:

```
conf = 100 * (0.6 * model_score + 0.4 * (0.7 * geo + 0.3 * shadow))

model_score = 0.5 * mean_sigmoid_prob + 0.5 * peak_sigmoid_prob
```

where `geo` encodes aspect ratio (target ~3.5), convex-hull solidity, and boundary
edge sharpness; `shadow` measures target-vs-surrounding-ring median contrast.
These are sonar-domain heuristics, not an acoustic propagation model.
Values below 35 are flagged as possible rock/shadow false positives.

## External Validation

No independent real-world sonar benchmark with a matching object taxonomy is
currently available. The site-based test split (13 held-out wreck sites, never
seen during training) is the primary held-out evaluation.

Note: The SCTD dataset used in early experiments is an aerial/UAV photography
dataset (ships and aircraft from above), **not sonar**. SCTD metrics have no
sonar-validation meaning and are not cited.

## Repository Layout

```
AI4Shipwrecks/AI4Shipwrecks/   # Training code + checkpoints (Codebase A — production)
pipeline/                       # Inference + Streamlit UI
  app.py                        # Run: streamlit run pipeline/app.py
  infer.py                      # SonarDetector (tile inference + ramp blending)
  confidence.py                 # Heuristic confidence scoring
  preprocess.py                 # Lee filter, CLAHE, TileBlender
  geotag.py                     # Pixel-to-lat/lon slant-range geometry
  xtf_io.py                     # XTF sonar log parser
docs/
  phase0_gap_analysis.md        # Full audit findings
```

## Running the Demo

```bash
pip install -r requirements.txt
streamlit run pipeline/app.py
```

> **Streamlit Community Cloud — Python version:**
> This app must be deployed with **Python 3.11**.
> In your app's Streamlit Cloud dashboard go to **Advanced settings → Python version** and select **3.11**.
> A `.python-version` file containing `3.11` is committed to the repository as a hint, but
> Streamlit Cloud's UI setting takes precedence.
> Using the default Python 3.14 build will fail because several compiled dependencies
> (e.g. `onnxruntime < 1.24`, older `opencv-python-headless`) lack `cp314` wheels.

```bash
export SONAR_CKPT=/path/to/best_model_v1_iou0.71.pth
export SONAR_TEST_IMAGES=/path/to/test/images
```

### Pre-flight checklist

| Check | How to verify |
|-------|---------------|
| Python >= 3.9 | `python --version` |
| Checkpoint present | `ls AI4Shipwrecks/AI4Shipwrecks/runs/best_model_v1_iou0.71.pth` |
| Requirements installed | `pip install -r requirements.txt` |
| Test images available | Set `SONAR_TEST_IMAGES` or place images in `AI4Shipwrecks/AI4Shipwrecks/test/` |

The checkpoint is not committed (`.gitignore` excludes `*.pth`). Restore it from your local
training run or set `SONAR_CKPT` to point to an absolute path.

### Expected output

After uploading a sonar image:

1. **Overlay panel** — input image with detection bounding boxes and mask outlines.
2. **Detection table** — detection ID, bounding box, mean sigmoid probability,
   sonar-heuristic confidence score (0-100, uncalibrated), GPS coordinates.
3. **Review panel** — per-detection confirm / reject / uncertain actions that persist
   to `pipeline/review_store.db` across restarts.
4. **Download links** — `hazard_report.json` and `hazard_report.csv` with
   `review_status` (`unreviewed` | `human-confirmed` | `human-rejected`) per detection.

Typical detection count on a clean 512x512 sonar tile: 3-5 detections (baseline mean 3.4
det/image; robustness harness baseline, `docs/evaluation_report.md` Section 6).

Latency: ~3 s/image ONNX FP32 CPU; ~0.37 s/image PyTorch GPU. Do not use INT8 -- it is
slower on this CPU and has a numerical accuracy penalty (see evaluation report Section 2).

### Failure modes

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| `FileNotFoundError: best_model_v1_iou0.71.pth` | Checkpoint not present | Restore from training or set `SONAR_CKPT` |
| Zero detections on a target-rich image | Threshold too high or wrong format | Verify 8-bit grayscale input; check `SonarDetector.threshold` |
| 10+ detections on a flat background frame | Platform motion artefact (heave shear) | Review manually; see `docs/qualitative_examples.md` FP-1 |
| Score near 35 flips between runs | Near-threshold instability | Genuinely uncertain case; see `docs/qualitative_examples.md` HARD-2 |
| `onnxruntime` import error | Missing dependency | `pip install onnxruntime` |

## Dataset

- **AI4Shipwrecks** (NOAA / Thunder Bay National Marine Sanctuary)
  — 286 side-scan sonar images, 28 shipwreck sites
  — License: see original dataset documentation

## Known Limitations

1. Test IoU 0.43 is below published SOTA for sonar ATR (0.55–0.77 Dice/IoU)
2. Confidence score is uncalibrated — no Platt scaling or isotonic regression applied
3. Detection scope limited to shipwrecks; pipes/nets/cylinders not yet covered
4. Inference latency (10 images, seed=42, CPU=Intel Core i7-12700H, GPU=RTX 4050 Laptop, CUDA 12.1):
   | Backend | mean | min | max |
   |---------|------|-----|-----|
   | PyTorch FP32 / CPU | 4.29s | 1.53s | 6.23s |
   | PyTorch FP32 / GPU (RTX 4050) | 0.37s | 0.13s | 0.54s |
   | ONNX FP32 / CPU | 3.08s | 0.89s | 5.27s |
   | ONNX INT8 / CPU† | 5.26s | 2.09s | 7.06s |

   † INT8 max diff vs FP32 = 0.6497 WARN — use FP32 for inference. INT8 is **slower** on this CPU (dynamic quant overhead outweighs smaller model). ONNX sizes: FP32 97.7 MB, INT8 24.6 MB. Times cover preprocess+tile_inference; exclude file I/O and report write. Full results in `pipeline/benchmark_results.json`.
5. Human-in-the-loop review workflow implemented (Phase 5): per-detection confirm/reject/uncertain/annotate, SQLite persistence, generic category system
6. Cross-domain generalisation: shipwreck model evaluated on Santos et al. 2024 AUV SSS dataset (1,170 images, "cross-domain generalization check (non-mine anomalous-object subset)"). MILCO Det@0.5 0.0%, NOMBO Det@0.5 2.2%, FP rate on background 28.5%. Model does not transfer across SSS sensor domains (frequency mismatch ~132 kHz vs 900-1800 kHz, different survey geometry). See `pipeline/crossdomain_results.json`.
7. **Primary characterised failure mode -- texture firing, not shape discrimination.** Across four independent stress tests the model consistently fires on unfamiliar or ambiguous texture rather than reliably discriminating true target shape. This is the single most important characterised limitation:
   - *Cross-domain background FP rate (Santos 2024):* 28.5% of clean background frames fired despite containing no target-like structure -- model activates on unfamiliar sonar texture alone.
   - *Heave/pitch/roll robustness (synthaug.py):* Platform motion shear raised mean FP count by +196% (3.4 to 10.1 det/frame) -- bright edge artefacts from an unfamiliar acquisition geometry trigger the same indiscriminate firing.
   - *In-domain GT-negative FP rate (AI4Shipwrecks test set):* 24/46 frames with no ground-truth annotation fired at least one detection (52.2% upper-bound frame-level FP rate; 22.4% excluding one disproportionately large outlier). GT-negative frames are wreck-site boundary frames, not clean non-wreck seafloor, so this is an upper bound. See `pipeline/fp_rate_results.json`.
   - *Synthetic net-patch exploration (illustrative, unvalidated, synthetic-only -- not a demonstrated capability):* 26/30 synthetic net-like patches triggered detections (86.7%). Overlay analysis shows broad blobs spanning most of patch width, not strand-geometry tracing -- consistent with texture response, not shape recognition. See `docs/synthetic_net_exploration.md`.

   Three independent characterisations (cross-domain, motion-artefact, net-patch) converge on the same conclusion, and the in-domain boundary-frame result is directionally consistent. Any deployment should assume the model will fire on unfamiliar high-contrast texture; the human review workflow (Q3, `pipeline/review_store.py`) is the primary mitigation. See `docs/judge_qa.md` Q6 and Q13 for full numbers.
