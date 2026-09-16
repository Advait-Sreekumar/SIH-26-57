# Phase 1.5 — Class Expansion Feasibility Report

**Date:** 2026-09-16
**Scope:** Pipes/cylinders, ghost/entangled nets — classes required by SIH but absent from current model
**Rule:** A class gets a model only after evaluation on real (non-purely-synthetic) sonar imagery.
If no real validation data exists, the class is documented as "not yet achievable" with a
clear explanation of what would be needed.

---

## Summary Verdicts

| Class | Verdict | Confidence |
|-------|---------|------------|
| Underwater pipelines / cylindrical infrastructure | **Not achievable in available time** (SubPipe: GPL-3.0 blocks use + AUV vs towfish domain mismatch) | High |
| Ghost/entangled fishing nets | **Not yet achievable** (no public labeled sonar dataset exists anywhere) | High |
| Generic cylindrical seabed debris | **Synthetic-only / unproven** (no real-data validation path) | High |

---

## 1. Underwater Pipelines / Cylinders

### Dataset investigated: SubPipe

- **Repository:** https://github.com/remaro-network/SubPipe-dataset
- **Paper:** https://arxiv.org/abs/2401.17907 (Gonzalez-Garcia et al.)
- **Modality:** Side-scan sonar (SSS) — LF and HF channels — plus RGB optical and IMU
- **Annotated SSS frames:** ~6,335 bounding-box annotations across 10,030 SSS images
- **Annotation type on SSS channel:** Bounding boxes only (COCO + YOLO)
  Segmentation masks exist only on the optical/RGB channel, not the sonar channel.

### License — BLOCKS USE

Repository LICENSE: GPL-3.0 (copyleft). Any derivative work including trained model
weights and submission tooling must also be GPL-3.0. The README additionally states
the data is "property of Oceanscan-MST" with no explicit training rights grant.
A public SIH submission cannot comply with GPL-3.0 copyleft requirements without
GPL-licensing the entire submission.

### Domain shift — additional blocking concern

SubPipe was collected by an AUV (Oceanscan-MST LAUV) with a side-mounted SSS.
AI4Shipwrecks uses surface-towed towfish SSS. These differ in:
- Grazing angle (steep AUV-mounted vs shallow towfish)
- Swath width and altitude
- Seafloor illumination geometry and shadow geometry

This is the same class of domain mismatch that invalidated SCTD cross-domain eval.

### Verdict: NOT ACHIEVABLE IN AVAILABLE TIME

Two independent blocking criteria: license incompatibility and domain shift.
See docs/subpipe_gonogo.md for the full go/no-go analysis.

**What would be needed:** License-compatible, towfish-survey SSS pipeline data
with segmentation annotations (not bounding boxes). Time to working model if
data existed: 1–2 weeks.

---

## 2. Ghost Nets / Entangled Fishing Nets

### Dataset search result

A thorough search found **no publicly available sonar dataset** with ghost net /
entangled fishing net annotations.

Key findings:
- Most net detection research uses optical/RGB underwater cameras, not sonar.
- Sonar detection of ghost nets is an active research problem; published papers describe
  the challenge but do not release annotated datasets.
- The MDPI paper "Enhancing Side-Scan Sonar Segmentation with Synthetic Data"
  (https://www.mdpi.com/2077-1312/13/3/616) proposes synthetic generation for SSS but
  targets generic objects, not nets specifically.
- Ghost nets present as irregular low-backscatter features with complex geometry in SSS;
  they are extremely difficult to distinguish from seaweed, sediment texture, or debris.

### Synthetic data feasibility

The project already has `pipeline/synthaug.py` which models:
- Speckle noise
- Intensity striping (sonar speed-of-sound distortions)
- Acoustic shadow simulation
- CLAHE-like contrast enhancement

Extending this to simulate ghost nets would require:
1. A 3D mesh model of a net draped on seabed
2. An acoustic backscatter simulator (e.g., BELLHOP-based forward model) to render SSS
3. Annotation of the synthetic output

This is a 2-4 week research task. Even if done, synthetic-only training would produce
an unproven model that cannot be honestly presented as validated capability.

### Verdict: NOT YET ACHIEVABLE

No real sonar validation data exists. Synthetic-only training is possible but violates
the project constraint against claiming detection capability without real-data evaluation.

**Honest documentation for SIH judges:**
"Ghost net detection in SSS imagery remains an open research problem. No annotated
sonar dataset for ghost nets is currently publicly available. This class is documented
as a future research direction. Physical reasoning suggests ghost nets would appear as
low-backscatter irregular patches with reduced acoustic shadow; discriminating them from
seabed texture would require multi-look analysis and likely multi-class training data
that does not yet exist."

---

## 3. Generic Cylindrical Seabed Debris

### Dataset search result

No dedicated public SSS dataset for generic cylinders (drums, pipes, munitions, canisters)
was found. Some ONNX / torpedo / UXO detection datasets exist in the defense domain but
are not publicly accessible.

### Synthetic feasibility

Cylinders are simpler to simulate acoustically than nets:
- A cylinder produces a characteristic bright highlight + dark acoustic shadow
- Aspect ratio and shadow length encode diameter and burial depth
- Existing sonar simulators can render cylinders realistically

However, the same constraint applies: synthetic training without real validation data
cannot be presented as validated detection capability.

### Verdict: SYNTHETIC-ONLY / UNPROVEN

Achievable as a research experiment, unacceptable as a claimed product feature.

---

## 4. Recommended SIH Framing

Present the scope honestly to judges:

1. **What is demonstrated:** Shipwreck ATR in SSS imagery, with site-based test IoU 0.43
   on the AI4Shipwrecks (Thunder Bay NMS) benchmark. Working end-to-end: XTF parsing,
   preprocessing, tile inference, heuristic confidence scoring, geotagged JSON/CSV reports.

2. **What is scoped out of this version:** Pipe/cylinder/net detection — not because it
   was ignored, but because no annotated SSS validation data exists for these classes.
   Claiming detection without validation data would produce misleading metrics.

3. **Roadmap (honest, time-bounded):**
   - Phase 2: Obtain SubPipe (verify if SSS partition exists); if yes, fine-tune and
     evaluate; report honest IoU. Time estimate: 1 week.
   - Phase 3: Synthetic data generation for cylinders using acoustic forward model;
     train and evaluate on synthetic hold-out; label as unproven until real data obtained.
     Time estimate: 3-4 weeks.
   - Phase 4: Ghost nets — dependent on a research data collection effort.
     Time estimate: unknown (blocked on data).

4. **What judges should ask:** "What would it take to detect ghost nets?" The honest
   answer shows engineering maturity: understanding the data gap, the acoustic physics,
   and a concrete path to addressing it.

---

## 5. One Actionable Item Before Demo

The existing `pipeline/synthaug.py` already applies sonar noise augmentation during
inference (stress test mode in the UI). This demonstrates physics-informed sonar
understanding without overclaiming object detection capability for classes that lack
validation data. This is the correct framing for the demo.
