# Phase 1 Architecture — As-Built

**Date:** 2026-09-16  
**Scope:** Production codebase (Codebase A). Reflects actual code, not the SIH brief.

---

## Model

| Parameter | Value |
|-----------|-------|
| Architecture | U-Net |
| Encoder | ResNet34 (ImageNet pretrained weights used during training) |
| Decoder attention | None (checkpoint has no scSE keys despite `decoder_attention_type='scse'` in training script) |
| Input channels | 1 (greyscale sonar) |
| Output | 1-channel logit map (sigmoid threshold for mask) |
| Input resolution | 512 x 512 (tiles with ramp blending for larger images) |
| Framework | segmentation-models-pytorch 0.5.0, PyTorch 2.5.1 |

**Checkpoint:** `AI4Shipwrecks/AI4Shipwrecks/runs/best_model_v1_iou0.71.pth`  
Keys stored: `encoder`, `model_state_dict`, training epoch metadata.

---

## Dataset

- **AI4Shipwrecks** (Thunder Bay National Marine Sanctuary, NOAA)  
  286 side-scan sonar images, 28 shipwreck sites  
  Site-based split: 12 train sites / 13 test sites, zero overlap  
- No data from other surveys was used for training or validation.

---

## Performance

| Split | IoU | Dice |
|-------|-----|------|
| Validation (in-distribution) | 0.713 | 0.833 |
| **Test (held-out sites)** | **0.427** | **0.598** |

Test IoU is the headline metric. Val IoU reflects in-distribution generalisation only.

---

## Inference Pipeline

```
sonar image / XTF
     |
     v
preprocess.py  -- Lee speckle filter, CLAHE, pad to 512
     |
     v
infer.py (SonarDetector)
  - torch.load checkpoint
  - tile inference with ramp blending for images > 512px
  - sigmoid --> binary mask at threshold (default 0.5)
  - connected-component extraction (min area 80px)
     |
     v
confidence.py (score_detections)
  - heuristic score: 100*(0.6*model_score + 0.4*(0.7*geo + 0.3*shadow))
  - model_score = mean sigmoid prob in detection region
  - geo = aspect-ratio + solidity + edge-sharpness composite
  - shadow = target-vs-background contrast
  - likely_rock_or_shadow flag if score < 35
     |
     v
geotag.py (PixelGeoMapper)
  - slant-range geometry: flat-earth pixel --> lat/lon
  - inputs: origin lat/lon, heading, altitude, sonar range, along-track length
     |
     v
save_report --> hazard_report.json + hazard_report.csv
```

**XTF path:** `xtf_io.py` parses pyxtf waterfall, extracts nav data, applies optional slant-range correction before the pipeline above.

---

## ONNX Export

| File | Format | Opset | Max diff vs PyTorch |
|------|--------|-------|---------------------|
| `runs/wreck_unet.onnx` | FP32 | 16 | 2.57e-05 PASS |
| `runs/wreck_unet_int8.onnx` | INT8 dynamic QUInt8 | 16 | 0.6497 WARN |

FP32 is the validated reference. INT8 is retained for throughput benchmarking only.

Export script: `AI4Shipwrecks/AI4Shipwrecks/export_onnx.py`

---

## Sonar-Heuristic Confidence Score — Not a Probability

```
conf = 100 * (0.6 * model_score + 0.4 * (0.7 * geo + 0.3 * shadow))

model_score  = 0.5 * mean_sigmoid_prob + 0.5 * peak_sigmoid_prob
geo          = 0.40 * aspect_score + 0.25 * solidity_score + 0.35 * edge_sharpness
shadow       = target-vs-surrounding-ring median contrast (clipped 0–1)
```

`geo` and `shadow` are sonar-domain heuristics (not an acoustic propagation model).
No Platt scaling or isotonic regression applied — score is not a calibrated probability.
The `likely_rock_or_shadow` threshold (35.0) was set by inspection, not threshold optimisation.

---

## What Is Not Claimed

- No detection capability for pipes, cylinders, or ghost nets (no validated training data)
- No external sonar benchmark validation (SCTD is aerial photography, not sonar)
- No calibrated probability output
- No real-time performance claim (inference time not yet measured on target hardware)
- No military/defense deployment readiness
