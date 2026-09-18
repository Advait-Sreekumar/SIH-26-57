# Synthetic Sonar Patch Exploration — Net-Like Shapes

> **Framing (mandatory, read first):**
> This is illustrative, unvalidated, synthetic-only exploration — not a demonstrated capability.
> The model was trained exclusively on shipwreck sonar imagery. No ghost-net sonar data
> was used for training, fine-tuning, or evaluation at any stage. Results below describe
> what happens when the shipwreck model is applied to synthetic textures — nothing more.

---

## Motivation

Ghost/entangled fishing nets appear in the SIH problem statement as a target class.
A thorough search found no publicly available labeled sonar dataset for ghost nets anywhere.
Synthetic-only training without real-data validation is documented in
`docs/phase1_5_class_expansion_feasibility.md` as "not yet achievable" as a claimed
detection capability.

However, an orthogonal question is answerable in a few hours and useful for judges:
*If we take the existing shipwreck model and run it on synthetic net-like sonar patches,
what does it do?*

The answer tells us something about model selectivity, not about ghost-net detection.

---

## Generation Method

Script: `pipeline/synthetic_net_exploration.py` | RNG seed: 7 | Patches: 30 | Patch size: 512x512

### Seafloor background

Synthetic sediment background composed of:
- Flat 0.28-intensity base with coarse low-frequency variation (resized 16x16 random field)
- Multiplicative Rayleigh speckle (look parameter L=2, approximating real sonar statistics)
- Horizontal intensity striping (+/-4%) to simulate sonar waterfall line artifacts

This is a simplified, single-layer synthetic -- it does not use a forward acoustic simulator
(e.g. BELLHOP) and does not match any specific sonar system's response characteristics.

### Net-like shapes

Each patch gets a randomised mesh overlay generated from:
1. **Longitudinal strands** (3-6 per patch): quadratic Bezier curves with randomised control
   points, tracing roughly parallel paths across the patch. Line width: 1-3 px. Intensity:
   0.55-0.90 relative to background (net strands produce slightly increased backscatter at
   their edges in SSS imagery).
2. **Cross-strands** (4-9 per patch): shorter line segments connecting adjacent longitudinal
   strands at irregular intervals, simulating mesh structure.
3. **Reduced-backscatter fringe** downstream of net edges: a dilated shadow mask attenuates
   background by 15-35% where the net structure blocks acoustic energy. Net strands themselves
   attenuate background to 55% (net absorbs / scatters acoustic energy away).

The resulting shapes are elongated, low-solidity, irregular mesh patterns -- a first-order
geometric approximation of a collapsed demersal fishing net on the seafloor.

**What is NOT modelled:** 3D draping, substrate interaction, water column propagation,
frequency-dependent scattering, mesh node specular reflections, or burial depth effects.
These are architecturally necessary for a credible sonar simulator. This generator produces
texture-level plausibility, not acoustic fidelity.

### Compositing

```
patch = clip(bg * attenuation - shadow + net * 0.35, 0, 1)
```

Outputs saved as 8-bit greyscale PNG in `pipeline/net_exploration_visuals/`.

---

## Model

- Architecture: U-Net + ResNet34 encoder (no scSE)
- Trained on: AI4Shipwrecks dataset (Thunder Bay NMS, Iver3 AUV + EdgeTech 2205, ~132 kHz)
- **Not retrained. Not fine-tuned. No net data used at any stage.**
- Threshold: 0.5 (same as primary demo)
- Inference: tiled (512x512, 64 px overlap, ramp blending) via `SonarDetector` in `pipeline/infer.py`

---

## Results

Script: `pipeline/synthetic_net_exploration.py` | RNG seed 7 | 30 patches

| Metric | Value |
|--------|-------|
| Patches generated | 30 |
| Patches with >= 1 detection | 26 (86.7%) |
| Total model detections | 72 |
| Mean detections / patch | 2.40 |

Full per-patch data: `pipeline/net_exploration_results.json`

To reproduce:
```bash
cd SIH
python pipeline/synthetic_net_exploration.py
```

---

## What This Does and Does Not Show

### What it shows

Whether the shipwreck-trained model fires at all on synthetic net-like patches, and at what
rate. This is a measure of model non-selectivity on out-of-distribution input -- the same
behaviour characterised in two prior results:

- **Cross-domain check (Santos 2024):** FP rate on background frames was 28.5% even without
  any target-like structure -- the model fires indiscriminately on unfamiliar sonar texture.
- **Heave/pitch/roll robustness result (Q13):** Platform motion shear increased false-positive
  count by +196% (3.4 → 10.1 mean detections) -- bright edge artefacts from an unfamiliar
  acquisition geometry trigger the same indiscriminate firing behaviour.

Both results, and this experiment, are consistent with the same underlying property: the model
fires on any sufficiently "target-like" low-level texture, not specifically on shipwreck
structure. The 86.7% activation rate on synthetic net patches does not mean "86.7% net-detection
accuracy" and does not imply the model has learned anything about real net acoustic signatures.
A high hit-rate on synthetic shapes could equally reflect indiscriminate firing on unfamiliar
high-contrast texture -- the same failure mode already documented in both results above.

The synthetic net patches contain elongated low-backscatter structures with reduced shadow
fringes. These share some local texture statistics with shipwreck debris fields (irregular
boundaries, attenuated regions, partial acoustic shadows). Any model activations on net
patches likely reflect shared low-level texture cues, not learned understanding of net
structure. Overlay analysis of detection blobs confirms that activated regions are broad
(3--11% of patch area) and spatially coincident with the strand region primarily because
strands span most of the patch width -- not because the model traced strand geometry.

### What it does NOT show

- **It does not demonstrate ghost-net detection capability.** The model was not designed,
  trained, or evaluated for this purpose.
- **It does not show that the model is useful for nets.** Any activation rate on synthetic
  out-of-distribution patches is evidence of non-selectivity, not true positive detections.
- **It does not establish a baseline.** The patches are synthetic and acoustically
  non-rigorous. Detection rate on real sonar imagery of ghost nets would require real data
  to measure -- and no such dataset is publicly available.
- **It does not constitute a product claim.** This experiment is scoped entirely to an
  internal research observation. It is not in the demo, not in the UI, and not referenced
  in evaluation metrics.

### Correct interpretation

The shipwreck model is not selective for shipwreck structure when applied to
out-of-distribution sonar imagery. Any activation rate on net-like synthetic patches
is consistent with the cross-domain characterisation already documented: the model fires
on unfamiliar sonar texture regardless of whether that texture represents a shipwreck.

To build a credible ghost-net detector: (1) obtain labeled real sonar data of ghost nets;
(2) design a generation pipeline using a proper acoustic forward model for net-like
objects; (3) train and evaluate on held-out real data. None of these steps are possible
within the current project timeline.

---

## Reproducibility

```bash
cd SIH
python pipeline/synthetic_net_exploration.py
# Writes: pipeline/net_exploration_results.json
#         pipeline/net_exploration_visuals/net_patch_*.png  (inputs)
#         pipeline/net_exploration_visuals/net_patch_*_overlay.png  (model overlays)
```

Requires: numpy, opencv-python, torch (same environment as primary pipeline).
RNG seed 7, N=30 patches, deterministic output.

---

## Files

| File | Description |
|------|-------------|
| `pipeline/synthetic_net_exploration.py` | Generator + inference runner |
| `pipeline/net_exploration_results.json` | Per-patch results, aggregate stats, framing metadata (generated on run) |
| `pipeline/net_exploration_visuals/net_patch_NNN.png` | Synthetic input patches (generated on run) |
| `pipeline/net_exploration_visuals/net_patch_NNN_overlay.png` | Model activation overlays (generated on run) |
