# SonarEye — Pitch Deck Outline

**Status:** Draft for review (Part C, Step 1). Do not build the `.pptx` until this slide content is approved.

**Ground rules honored throughout:**
- Every number is a measured result from a source file in this repo — nothing estimated or fabricated.
- No mine-detection / MILCO / mine-classification language anywhere.
- The synthetic-net work is always labeled *"illustrative, unvalidated, synthetic-only exploration — not a demonstrated capability."*
- No SCTD external-validation claims.
- "Real-time" / "edge-ready" are not claimed — only the measured latency numbers, with their scope stated.
- Confidence is always called the **sonar-heuristic confidence score** (never "physics-informed").

**Diagram assets available (Part B, `docs/diagrams/`):**
- `01_system_architecture.png` — end-to-end pipeline + human review loop
- `02_confidence_score_breakdown.png` — confidence formula tree
- `03_evaluation_charts.png` — Panel A IoU bars, Panel B robustness deltas
- `04_class_expansion_decisions.png` — taxonomy branch/status diagram

---

## Slide 1 — Title / The Problem

**Title:** SonarEye — Finding what doesn't belong on the seafloor

**Bullets:**
- The ecological hook: lost fishing gear ("ghost nets") and marine debris keep catching life long after they're abandoned — but on side-scan sonar they blend into natural clutter (rock, reef, sand ripples).
- The operational bottleneck: sonar survey review is still largely manual — an analyst scrubs through hours of imagery frame by frame.
- The core hard problem: distinguishing a real anomalous object from high-contrast natural texture, at scale, without missing things.

**Visual:** No diagram — title slide. Optional: a single representative sonar frame if one is cleared for use.

---

## Slide 2 — What We Built and Proved

**Title:** An end-to-end sonar hazard-detection pipeline, validated on real data

**Bullets:**
- Full pipeline: raw sonar in → preprocessing → tiled U-Net segmentation → post-processing → geotagging → mission report, with a human review loop.
- Validated on **real shipwreck side-scan sonar** (AI4Shipwrecks: 286 images, 28 sites, site-based split with zero site overlap between train and test).
- Headline result, honestly framed: **Test IoU 0.4268 / Dice 0.5983** on 13 fully held-out sites (in-distribution validation reaches IoU 0.713 / Dice 0.833 — the gap itself is a finding, not hidden).

**Visual:** `01_system_architecture.png` (small, as a teaser) — full-size version is Slide 3.

---

## Slide 3 — System Architecture

**Title:** How it works, end to end

**Bullets:**
- Preprocessing: Lee speckle filter + water-column crop + normalize + CLAHE.
- Inference: U-Net with ResNet34 encoder (ImageNet-pretrained), 1-channel 512×512 tiles at 50% overlap with ramp blending.
- Post-processing: connected components + sonar-heuristic confidence scoring.
- Geotagging: pixel → lat/lon via PixelGeoMapper.
- Output: JSON + CSV + PDF mission report.
- Human review loop persists confirm / reject / uncertain decisions to SQLite.

## Slide 4 — Evidence: Evaluation & Robustness

**Title:** What the numbers say — and one insight that ties them together

**Bullets:**
- Segmentation: Test IoU 0.4268 (held-out) vs Validation IoU 0.713 (in-distribution); SOTA reference band for this task is roughly 0.55–0.77.
- Robustness (IoU delta vs clean baseline, baseline IoU 0.0581): speckle noise −0.0453, heave/pitch/roll −0.0254, radial distortion −0.0056, shadow synth −0.0002.
- **The single convergent finding:** the model *fires on high-contrast structured texture, not exclusively on shipwreck shape.* The perturbations that most degrade it (speckle, geometric jitter) are the ones that alter texture — four independent stress tests point the same way.

**Visual:** `03_evaluation_charts.png` (Panel A IoU bars + Panel B robustness deltas).

---

## Slide 5 — Evidence: Deployment & Performance

**Title:** Measured performance, with scope stated

**Bullets:**
- Two distinct latency scopes, measured (warm): **pure GPU forward pass 0.37 s** vs **full pipeline 686 ms** (preprocessing + inference + post-processing). CPU full pipeline: 4.29 s.
- Cold start: ~7,120 ms total, of which ~5.9 s is one-time imports — a warm-vs-cold distinction that matters for how the tool is deployed.
- Model export: ONNX FP32 97.7 MB, max diff 2.57e-05 (PASS). INT8 24.6 MB, diff 0.6497 (WARN — not used).
- Runs today as a Streamlit app; no claim of "real-time" or "edge-ready" — those require measurement on target hardware, which we have not done.

**Visual:** No Part B diagram — a simple measured latency / model-size table built from these numbers.

---

## Slide 6 — The Taxonomy Gap: Investigated, Not Hidden

**Title:** We looked at all four object classes in the brief — and found real blockers

**Bullets:**
- Shipwrecks: **SHIPPED** — licensed, labeled data existed; test IoU 0.427.
- Pipes + cylinders: **BLOCKED** — the candidate dataset (SubPipe) is GPL-3.0 (blocks a closed-source submission) and is AUV-towfish domain-shifted.
- Ghost nets: **BLOCKED** — no public labeled sonar dataset exists.
- Generic debris: **DEFERRED** — no labeled dataset found after literature search.
- Framing: three of these are *real, distinct, documented blockers.* The system boundary is defined by evidence, not omission.

## Slide 7 — Class-Agnostic Architecture

**Title:** The pipeline doesn't know what it's looking at

**Bullets:**
- The segmentation model outputs "anomalous object vs seafloor" — it is not hardcoded to "shipwreck."
- Adding a second class is a data problem, not an architecture rewrite: the review workflow already stores a generic `category` field (shipwreck, rock, shadow_artifact, biological, unknown).
- Shipwrecks were simply the class that had **real, licensed, labeled data to prove the architecture on.** The design generalizes; the evidence is class-specific by necessity.

**Visual:** Reuse `01_system_architecture.png` (highlight the class-agnostic inference + generic-category review stages), or a small callout box. No new claim.

---

## Slide 8 — Synthetic Net Exploration (Caveated Footnote)

**Title:** A texture probe — explicitly not a capability

**Bullets:**
- We ran an exploratory probe on synthetic net-like texture: 26/30 (86.7%) patches triggered a detection (72 detections total, seed 7).
- This is consistent with the Slide 4 finding — the model fires on *unfamiliar high-contrast texture, not on net shape.*
- **Label, verbatim and unavoidable:** *"illustrative, unvalidated, synthetic-only exploration — not a demonstrated capability."* This is **not** ghost-net detection and is not presented as one.

**Visual:** None, or a small muted inset clearly captioned with the label above. Kept deliberately understated.

---

## Slide 9 — Human-in-the-Loop / Product Workflow

**Title:** AI assists, a human verifies

**Bullets:**
- Mission flow: landing → survey → analysis → review-in-sequence → map → summary → report.
- Each detection is reviewed in sequence: confirm / reject / uncertain, with optional category and note; decisions persist across restarts (SQLite).
- The report distinguishes **AI-flagged / unreviewed** from **human-confirmed** — the tool never presents an unreviewed detection as a verified finding.

**Visual:** `01_system_architecture.png` (human review loop portion), or product/UX screenshots if available.

---

## Slide 10 — What Real Deployment Would Need Next

**Title:** The honest path to a deployable ghost-net tool

**Bullets:**
- Labeled ghost-net sonar data — the one missing ingredient. Plausible route: partner with a conservation / ghost-gear-retrieval organization to co-label real survey imagery, exactly the way AI4Shipwrecks itself was built.
- Embedded-hardware benchmarking before any "edge" or "real-time" claim can be made.
- Validation on Indian-waters sonar to confirm generalization beyond freshwater Great Lakes data.

**Visual:** None required, or a simple 3-step "next" strip. No new performance claims.

---

## Slide 11 — Close

**Title:** What's real today, and what's next

**Bullets:**
- **Real and tested today:** an end-to-end pipeline validated on real held-out shipwreck sonar (Test IoU 0.4268), with measured latency, a human review loop, and geotagged reports.
- **Honestly bounded:** we documented every class we couldn't ship and why; the net work is an unvalidated texture probe, not a capability.
- **Next:** labeled ghost-net data + hardware benchmarking + regional validation turn a proven architecture into a deployable tool.

**Visual:** Optional — recap strip of the four Part B diagrams as thumbnails.

---

## Notes for Step 2 (deck build — pending approval)

- Diagrams are already dark-themed (bg `#1a1a2e`); a matching dark slide master keeps them legible.
- Slides 5 and 10 need small tables/strips that are not yet drawn — these can be built as native pptx tables (no new diagram needed).
- `python-pptx` availability will be checked only after this outline is approved.



