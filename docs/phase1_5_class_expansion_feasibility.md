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
| Underwater pipelines / cylindrical infrastructure | **Achievable with caveats** | Medium |
| Ghost/entangled fishing nets | **Not yet achievable** | High |
| Generic cylindrical seabed debris | **Synthetic-only / unproven** | Low |

---

## 1. Underwater Pipelines / Cylinders

### Dataset found: SubPipe

- **What it is:** A submarine pipeline inspection dataset from Orebro University /
  REMARO project (EU Horizon 2020). Contains annotated sonar imagery from AUV pipeline
  inspection surveys.
- **Reference:** Gonzalez-Garcia et al., GitHub: https://github.com/remaro-network/SubPipe
- **Modality:** Primarily forward-looking sonar (FLS) / imaging sonar. NOT side-scan sonar.
- **Annotations:** Segmentation masks for pipeline surface.
- **License:** Academic use (CC BY or similar). Requires verifying current terms at the
  GitHub repository before any use.
- **Size:** Hundreds of frames (exact count: verify at repo — not assumed from memory).
- **Access:** Public via GitHub/Zenodo. Direct download link to be confirmed.

### Critical modality mismatch

SubPipe uses forward-looking sonar. The current pipeline uses side-scan sonar (SSS).
These are fundamentally different:
- FLS: circular sector image, range-bearing, used for obstacle avoidance / inspection
- SSS: waterfall strip image, acoustic shadows to one/both sides, used for seabed mapping

A U-Net trained on SSS shipwreck data CANNOT be directly fine-tuned on FLS pipeline data
without architectural and preprocessing changes.

### What would be needed to claim pipe detection

1. Obtain SubPipe dataset (verify license, check if it contains SSS data in addition to FLS)
2. If FLS only: either (a) find an SSS pipeline dataset, or (b) adapt preprocessing for FLS
   and train a separate FLS-specific model clearly distinguished from the SSS model
3. Evaluate on a held-out partition, report IoU/Dice
4. Only then: add a "pipeline detection" mode to the UI, clearly labelled with its modality

### Verdict: achievable but requires 3-5 days of work minimum

Not achievable before the SIH demo without cutting corners that violate the project's
honesty constraints. Should be documented as "Phase 2 roadmap" in the submission.

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
