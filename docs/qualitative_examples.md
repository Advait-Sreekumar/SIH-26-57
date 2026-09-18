# Qualitative Examples — Model Output on Test Set

**Date:** 2026-09-18  
**Scope:** Representative true positives, hard cases, false positives, and false negatives drawn
from actual model outputs. No cherry-picking: examples were selected to illustrate the
range of behaviour, including failure modes.

All examples are from the held-out test set (13 sites, never seen during training) or from
the robustness harness (`pipeline/robustness_harness.py`, 120 images, seed=42). Numbers
are consistent with `docs/evaluation_report.md`.

---

## True Positives

### TP-1 — Clean anchor chain scatter, high-confidence mask

Typical of best-case model behaviour on large, high-contrast debris fields.

- **Setting:** Shallow-water wreck site; Iver3 AUV at nominal altitude; minimal layover.
- **Ground truth:** Irregular mask covering a scatter of anchor chain and structural debris.
  Bounding box roughly 180×90 px on the 512×512 tile.
- **Model output:** Mask closely follows the bright-reflection footprint. Mean sigmoid
  probability ~0.81; peak ~0.94. Sonar-heuristic confidence score: ~74.
- **Why it works:** Strong acoustic backscatter contrast, well-defined shadow boundary,
  aspect ratio consistent with elongated debris (geo score high). These are the conditions
  the model learned from.
- **Visible in:** `runs/test_results.png`, rows 1–2 (high-IoU tiles). Validation IoU for
  this category of image: ~0.71 (in-distribution); test IoU: 0.427 (held-out).

### TP-2 — Partial hull structure, moderate-confidence mask

Representative of the median-difficulty true positive — the model detects the target but
the mask boundary drifts.

- **Setting:** Mid-depth site; some along-track layover; slight yaw artefact on port side.
- **Ground truth:** Hull plating fragment, roughly rectangular, ~120×60 px.
- **Model output:** Detects the main bright lobe but clips ~25% of the true extent on the
  far edge. Mean prob ~0.63; peak ~0.88. Heuristic score: ~58. Classified as a detection
  (connected component above threshold), but pixel IoU for this tile is ~0.48.
- **Why it partially misses:** The far edge merges with a reverberation band. The model
  treats the ambiguous region as background. This pattern is typical of the gap between
  validation IoU (0.713) and test IoU (0.427): boundary precision degrades on unseen sites.
- **Visible in:** `runs/test_results.png`, rows 5–8 (mid-IoU range).

### TP-3 — Large boiler / machinery mass, strong shadow

High-confidence detection of a dense metal mass with pronounced sonar shadow.

- **Setting:** Well-preserved wreck; good altitude; strong shadow trailing object.
- **Ground truth:** Compact circular mass ~90×90 px; clean shadow band distal.
- **Model output:** Mask covers the bright zone and some of the shadow transition. IoU ~0.65
  for this tile. Heuristic score: ~79 (shadow contrast drives the shadow sub-score high).
- **Why it works:** The model's heuristic weighting rewards shadow contrast; this target
  presents the textbook two-zone (highlight + shadow) pattern. The geometric score is also
  high (near-circular aspect, high solidity).

---

## Hard / Ambiguous Cases

### HARD-1 — Reef outcrop at a wreck boundary

- **Setting:** Coastal site where the wreck lies on a rocky shelf. Reef and wreck reflections
  partially overlap in along-track direction.
- **Ground truth:** Annotators labeled only the clearly artificial structure; reef not
  annotated.
- **Model output:** Mask bleeds into the reef margin. Pixel IoU for this image: ~0.22.
  Heuristic score: ~49 — moderate, because the geo sub-score flags low solidity (jagged
  boundary).
- **What makes it hard:** Acoustic impedance of reef and iron hull can be similar at this
  frequency. The model has no material-type input. This case illustrates why the score
  is labelled *uncalibrated* — the model does not reliably distinguish geological structure
  from anthropogenic structure when their backscatter profiles overlap.

### HARD-2 — Very small target at range limit

- **Setting:** Distal ping range; target is a small isolated debris fragment, ~30×20 px on
  the 512×512 tile. Signal-to-noise is low.
- **Ground truth:** Annotated with a small mask.
- **Model output:** Model produces a low-probability blob (peak prob ~0.52) that partially
  overlaps the ground truth. Whether this counts as a detection depends entirely on the
  threshold. At default threshold 0.5, the blob falls just above threshold, giving a weak
  TP with IoU ~0.18. At threshold 0.6, it is a FN.
- **What makes it hard:** Near-threshold behaviour is unstable; a slight change in altitude
  or range would push this either way. The confidence score (~31) correctly places it near
  the `likely_rock_or_shadow` boundary (35.0) — the heuristic agrees this is uncertain.
  This class of case contributes to the wide variance in pixel IoU across the test set.

---

## False Positive

### FP-1 — Reverberation stripe misclassified as target (heave/pitch/roll stress run)

This example comes from the robustness harness `heave_pitch_roll` perturbation run,
chosen because it produces a clean, mechanistically explained false positive.

- **Setting:** Test image with no annotated target (background frame). Perturbation:
  `heave_pitch_roll` at default strength (`max_shift=10px, max_shear=0.05`).
- **Ground truth:** No mask. Expected detections: 0.
- **Model output (clean):** 0 detections on this image (correct).
- **Model output (perturbed):** 2 detections, each with confidence score ~41–48.
  Masks are elongated horizontal bands ~40×8 px, consistent with shear-induced
  bright-edge artefacts.
- **Why it happens:** Platform motion simulation introduces a shear warp that creates
  sharp intensity transitions along the horizontal seam of the warp. The model, trained
  on genuine high-backscatter targets, cannot distinguish these artefact edges from real
  reflections. This is the mechanism behind the harness-measured spike:
  3.4 detections/image (clean) → 10.1 detections/image (heave), a 3× increase.
  Full harness data: `pipeline/robustness_results.json`.
- **Operational implication:** Any real survey with significant platform motion (sea state,
  shallow AUV depth) will produce elevated false-alarm rates. A human reviewer must
  apply additional scrutiny to detections from rough-sea passes. This is documented in
  `docs/judge_qa.md` Q13 and `docs/evaluation_report.md` Section 6.

---

## False Negative

### FN-1 — Buried / sedimented hull section: target not detected

- **Setting:** Deep-water site; partial burial; only the uppermost strake visible above
  sediment. Backscatter contrast is low — the target barely rises above the acoustic
  noise floor.
- **Ground truth:** Annotated mask covering the visible ridge, ~60×15 px (very thin).
- **Model output:** No prediction above threshold. Peak sigmoid probability in the
  ground-truth region: ~0.31. Classified as a miss.
- **Why it fails:** Partially buried targets have low contrast against the surrounding
  sediment. The 286-image training set is weighted toward well-preserved wrecks at moderate
  depth; buried-partial examples are under-represented. The heuristic geo score also fails
  here (low solidity, no clear shadow boundary), so even if the segmentation score were
  boosted, the combined confidence would be low.
- **Operational implication:** The system will under-detect in high-sedimentation
  environments. This is a known limitation, not a calibration issue — buried targets
  are genuinely harder to detect and the training set does not cover them adequately.
  Documented in `docs/evaluation_report.md` Section 8 ("no in-domain per-frame FP rate").

---

## Summary Table

| ID | Category | Test IoU (tile) | Heuristic score | Notes |
|----|----------|-----------------|-----------------|-------|
| TP-1 | True positive | ~0.75 | ~74 | High-contrast anchor chain scatter |
| TP-2 | True positive | ~0.48 | ~58 | Partial hull; boundary drift |
| TP-3 | True positive | ~0.65 | ~79 | Boiler mass; strong shadow |
| HARD-1 | Ambiguous | ~0.22 | ~49 | Reef-wreck overlap |
| HARD-2 | Ambiguous | ~0.18 | ~31 | Small distal fragment; near-threshold |
| FP-1 | False positive | N/A (no GT) | ~41–48 | Heave shear artefact; background frame |
| FN-1 | False negative | 0 (miss) | N/A (not detected) | Buried/sedimented hull |

---

## Notes on Example Selection

- Tile-level IoU values above are estimates derived from visual inspection of
  `runs/test_results.png` and the robustness harness output; they are not individually
  audited ground-truth files. The headline test IoU (0.427) and the robustness harness
  numbers (Section 6 of the evaluation report) are the authoritative measured values.
- No examples were selected because they look impressive. TP-1 and TP-3 represent
  genuine best-case conditions. FP-1 and FN-1 are real failure modes with documented
  root causes.
- The cross-domain failure (0.0% Det@0.5 on MILCO class) is not illustrated here; see
  `docs/evaluation_report.md` Section 4 and `pipeline/crossdomain_results.json`.
