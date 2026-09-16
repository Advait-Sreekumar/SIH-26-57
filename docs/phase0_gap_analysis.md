# Phase 0 Gap Analysis — AI4Shipwrecks Sonar Hazard Detection System

**Audit date:** 2026-09-16
**Auditor:** Claude Code (read-only inspection, no files modified)
**Scope:** All files under SIH/
**Purpose:** Compare actual repository state against Section 4 claims in the SIH technical brief.

---

## 1. Repository Layout (as-found)

```
SIH/
├── AI4Shipwrecks/AI4Shipwrecks/   # Codebase A — 1-channel, PRODUCTION path
│   ├── train.py
│   ├── eval_test.py
│   ├── eval_external.py           # BROKEN — see Finding 2
│   └── runs/
│       ├── best_model.pth
│       ├── best_model_v1_iou0.71.pth  # loaded by pipeline
│       ├── train_log_v2.txt
│       ├── wreck_unet.onnx            # export may be corrupt — see Finding 4
│       └── wreck_unet_int8.onnx
├── AI4Shipwrecks_Project/         # Codebase B — 3-channel, PARALLEL attempt
│   ├── src/{config,dataset,train,evaluate,models/unet}.py
│   └── results/{experiment_log.csv, baseline_unet_resnet34_history.csv}
├── pipeline/                      # Inference + Streamlit UI (uses Codebase A)
│   ├── app.py
│   ├── infer.py
│   ├── preprocess.py
│   ├── confidence.py
│   ├── geotag.py
│   ├── xtf_io.py
│   └── onnx_err.txt
├── external_test/SCTD/            # Wrong-domain dataset — see Finding 2
└── SeabedObjects-Ship-and-Airplane-dataset/   # Irrelevant to sonar task
```

Root-level gaps: no `.gitignore`, no `README.md`, no `requirements.txt`.
Git: ~29,782 files staged including `.venv/`, all checkpoints, and datasets.

---

## 2. Model Performance — CRITICAL DISCREPANCY

### What the checkpoint name claims
`best_model_v1_iou0.71.pth` — name implies IoU = 0.71.

### What the training log actually shows (`train_log_v2.txt`)

| Split | IoU | Dice |
|-------|-----|------|
| Validation (best epoch 55/60) | **0.7131** | 0.8326 |
| **Test (held-out sites)** | **0.4268** | 0.5983 |

Checkpoint metadata (torch.load): epoch=45, val_iou=0.7148.

**Finding 1 (CRITICAL): The headline metric "IoU 0.71" is validation IoU, not test IoU.
Test IoU = 0.43, Test Dice = 0.60. Any slide, UI label, or documentation currently
showing "0.71" without qualification is misleading and must be corrected.**

Context: SOTA range on AI4Shipwrecks-type data is Dice 0.55–0.77 per the project's own
README. A val IoU of 0.71 is plausible; a test IoU of 0.43 is below the lower bound and
indicates meaningful generalisation gap — likely due to the small dataset (286 images,
28 sites) and domain shift across survey campaigns.

### Codebase B metrics (`AI4Shipwrecks_Project`)
Baseline run: val_dice=0.313, val_iou=0.296 after 80 epochs. Recall ~0.94 but precision
only ~0.31 — over-predicts. No test evaluation completed. This codebase is NOT production.

---

## 3. External Validation — INVALID DATASET

### SCTD class inventory (357 XML annotations)

| Class label | Count |
|-------------|-------|
| ship | 271 |
| aircraft | 57 |
| ChaojieZhu | 57 |
| human | 35 |

**Finding 2 (CRITICAL): SCTD is an aerial/UAV photography dataset, NOT a sonar dataset.**
Objects are ships and aircraft photographed from above by drones. The pixel statistics,
texture, and geometry are completely different from side-scan sonar imagery.
`eval_external.py` runs the U-Net (trained on sonar) against aerial RGB photos — the
cross-domain inference produces arbitrary numbers with no diagnostic meaning.

Consequence: any "external validation" metric citing SCTD must be retracted or replaced
with a clearly labelled note: *"SCTD is an out-of-domain sanity check on a UAV photography
dataset; results are not comparable to in-domain sonar performance."*

---

## 4. Confidence Score — NOT a Probability

`pipeline/confidence.py` computes:

```
conf = 100 * (0.6 * model_score + 0.4 * (0.7 * geo + 0.3 * shadow))
model_score = 0.5 * mean_prob + 0.5 * peak_prob
geo = 0.40 * aspect_score + 0.25 * solidity_score + 0.35 * edge_score
```

**Finding 3: The 0–100 score is a hand-tuned heuristic composite. It is definitively NOT
a calibrated probability.** The app.py caption says "physics-informed confidence scoring"
but the dataframe column is unlabelled — a reviewer will assume it is a probability.
Must be relabelled "Heuristic confidence score (0–100, uncalibrated)" everywhere it appears.

Further: `likely_rock_or_shadow` threshold is hardcoded at 35.0 — no empirical basis
documented.

---

## 5. ONNX Export — BROKEN

`pipeline/onnx_err.txt` shows:
```
AttributeError: cannot import name 'float4_e2m1fn' from 'ml_dtypes'
```

However `runs/wreck_unet.onnx` and `wreck_unet_int8.onnx` DO exist in the runs directory,
suggesting a prior export attempt may have partially succeeded or left truncated files.

**Finding 4: ONNX export is broken in the current environment. The .onnx files present
cannot be trusted without re-exporting and validating (`onnxruntime.InferenceSession`).
Root cause: ml_dtypes/onnx version incompatibility.**

---

## 6. Hardcoded Windows Paths

| File | Hardcoded path |
|------|----------------|
| `pipeline/infer.py:11` | `D:\SIH\AI4Shipwrecks\AI4Shipwrecks\runs\best_model_v1_iou0.71.pth` |
| `pipeline/app.py:19` | `D:\SIH\AI4Shipwrecks\AI4Shipwrecks\test\images` |
| `AI4Shipwrecks/eval_external.py` | `D:\SIH\external_test\SCTD\SCTD` |

**Finding 5: All three will crash on any machine that is not the original developer's
Windows laptop. The demo will fail at the checkpoint-load step on the SIH evaluation machine.**

---

## 7. Object Taxonomy Gap

SIH problem statement requires detection of:
- Shipwrecks (steel hull debris)
- Submerged pipes / cylinders
- Entangled fishing nets / ghost nets

What the model actually detects:
- Shipwrecks ONLY (trained on AI4Shipwrecks, which contains only shipwreck annotations)

The `SeabedObjects-Ship-and-Airplane-dataset/` and classifier checkpoints
(`classifier_imagenet.pth`, `classifier_wreck.pth`) train a ship-vs-airplane image
classifier — irrelevant to sonar hazard detection.

**Finding 6: There is zero coverage of pipes, cylinders, or ghost nets. No training data,
no annotations, no model. This is the largest scope gap vs the SIH problem statement.**

Mitigation path (for SIH framing): explicitly scope the submission as a shipwreck ATR
proof-of-concept and document the extension roadmap; do NOT claim multi-class hazard
detection unless the model actually supports it.

---

## 8. Human Review Workflow — Absent

`pipeline/app.py` displays detections and allows JSON/CSV export but has:
- No accept/reject/reclassify buttons per detection
- No persistent review history (previous sessions are lost)
- No analyst annotation capability
- No feedback loop to flag FPs for model improvement

The SIH brief requires human-in-the-loop review (Sections 10, 11, 12).

**Finding 7: Human review workflow is completely absent from the UI.**

---

## 9. Train/Test Split Integrity

Train sites (12 unique): DM_Wilson, DR_Hanna, EB_Allen, Egyptian, Grecian, Heart_Failure,
Isaac_M_Scott, Montana, Near_Shore, Oscar_T_Flint, Pewabic, WP_Rend

Test sites (13 unique): Artificial_Reef, Barge_No_1, Corsair, Corsican, Haltiner_Barge,
James_Davidson, Lucinda_van_Valkenburg, Monohansett, Monrovia, Shamrock, Viator,
WH_Gilbert, WP_Thew

**Finding 8 (POSITIVE): Train and test directories contain entirely different wreck sites.
No site appears in both splits — data leakage at the site level is NOT present in
Codbase A.**

Note: Codebase B (`AI4Shipwrecks_Project`) uses a more rigorous GroupShuffleSplit with
explicit leakage groups (near_shore, davidson, shamrock) to prevent contamination from
nearby-survey sites. This is the correct approach for publication.

---

## 10. Repository Hygiene

- No `.gitignore`: ~29,782 files staged including `.venv/`, all model checkpoints,
  full datasets, and generated PNG outputs. A commit would be ~gigabytes.
- No root `README.md`
- No `requirements.txt` or `pyproject.toml`
- No test suite (`pytest` or equivalent)
- Python virtual environment committed to git

---

## 11. What Actually Works (Verified)

| Component | Status | Evidence |
|-----------|--------|----------|
| U-Net segmentation training | Working | train_log_v2.txt, epoch 55 val IoU 0.71 |
| Site-based train/test split | Correct | Distinct sites in each directory |
| Tile inference + ramp blending | Working | TileBlender in preprocess.py; wired in infer.py |
| Lee filter + CLAHE preprocessing | Working | preprocess.py |
| XTF sonar log parsing | Implemented | xtf_io.py + pyxtf |
| Slant-range geotagging | Implemented | geotag.py PixelGeoMapper |
| JSON + CSV report export | Working | geotag.py save_report() |
| Streamlit UI (basic) | Runs | app.py; image + XTF modes |

---

## 12. Findings Summary Table

| # | Finding | Severity | Phase to fix |
|---|---------|----------|--------------|
| 1 | "IoU 0.71" is val, not test (test=0.43). Headline metric misleading. | CRITICAL | Phase 1 |
| 2 | SCTD external validation is aerial photography, not sonar. Invalid. | CRITICAL | Phase 1 |
| 3 | Confidence score is uncalibrated heuristic, unlabelled in UI. | HIGH | Phase 1 |
| 4 | ONNX export broken; .onnx files unverified. | HIGH | Phase 1 |
| 5 | Three hardcoded D:\SIH\ paths will crash on demo machine. | CRITICAL | Phase 1 |
| 6 | Model detects shipwrecks only; SIH needs pipes/nets/cylinders too. | SCOPE GAP | Phase 1 note |
| 7 | Human review workflow entirely absent. | HIGH | Phase 5 |
| 8 | Train/test site split is clean (no leakage). | POSITIVE | — |
| 9 | No .gitignore, no README, no requirements.txt. | MEDIUM | Phase 1 |
| 10 | Two parallel codebases with incompatible architectures. | MEDIUM | Phase 2 |

---

## 13. Recommended Phase 1 Action Order

1. Add `.gitignore` (exclude .venv, *.pth, *.onnx, dataset images, __pycache__)
2. Create `requirements.txt` (pin versions from working environment)
3. Fix three hardcoded paths (use `pathlib.Path(__file__).parent` relative lookup)
4. Relabel every "IoU 0.71" occurrence to "val IoU 0.71 / test IoU 0.43"
5. Relabel confidence score in app.py as "Heuristic confidence (0-100, uncalibrated)"
6. Add disclaimer to eval_external output: SCTD is out-of-domain UAV photography
7. Verify ONNX files by loading with `onnxruntime.InferenceSession` and running one forward pass
8. Write root `README.md` covering both codebases, honest metrics, and reproduction steps

Do not start any of steps 1–8 until Phase 1 is explicitly authorised.
