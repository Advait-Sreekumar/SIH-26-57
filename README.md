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
  public submission; additionally, AUV-mounted vs towfish SSS collection geometry
  creates the same domain mismatch that invalidated SCTD (see `docs/subpipe_gonogo.md`)
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

Checkpoint is auto-resolved from the repository layout. Override with:
```bash
export SONAR_CKPT=/path/to/best_model_v1_iou0.71.pth
export SONAR_TEST_IMAGES=/path/to/test/images
```

## Dataset

- **AI4Shipwrecks** (NOAA / Thunder Bay National Marine Sanctuary)
  — 286 side-scan sonar images, 28 shipwreck sites
  — License: see original dataset documentation

## Known Limitations

1. Test IoU 0.43 is below published SOTA for sonar ATR (0.55–0.77 Dice/IoU)
2. Confidence score is uncalibrated — no Platt scaling or isotonic regression applied
3. Detection scope limited to shipwrecks; pipes/nets/cylinders not yet covered
4. ONNX FP32 export verified 2026-09-16: max diff 2.57e-05 PASS. INT8 dynamic-quant diff 0.6497 WARN (use FP32 for inference; see `AI4Shipwrecks/AI4Shipwrecks/onnx_err.txt`).
5. No human-in-the-loop review workflow yet (planned Phase 5)
