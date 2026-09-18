# SubPipe Dataset — Go/No-Go Decision

**Date:** 2026-09-16  
**Decision: NO-GO**

---

## Criteria Applied

### 1. License — FAILS

Repository LICENSE: GPL-3.0 (copyleft).

GPL-3.0 requires that any derivative work — including trained model weights and any
software incorporating them — also be released under GPL-3.0. A SIH hackathon submission
(public demo, submitted code, slides) would become GPL-3.0-encumbered if trained on
SubPipe data.

Additional concern: the README states the data is "property of Oceanscan-MST" with no
explicit grant of rights for derivative model training. GPL-3.0 covers the repo's code;
the data ownership creates a second, unresolved layer.

This criterion alone is sufficient to block use. We do not train on data whose license
we cannot comply with for a public submission.

### 2. Frame count — Passes (moot given license failure)

- Low Frequency annotated: 3,163 frames
- High Frequency annotated: 3,172 frames
- Total annotated: 6,335 across 10,030 SSS images
- Well above the 150–200 frame floor.

### 3. Domain shift — Significant concerns

- **Annotation type on SSS channel:** Bounding boxes only (COCO + YOLO formats).
  Segmentation masks exist only on the optical/RGB channel — not the sonar channel.
  This means bounding-box-supervised training only, producing coarser output than
  the current shipwreck segmentation model.
- **Sonar hardware and frequency:** Not documented in README or arXiv abstract.
  Cannot confirm compatibility with AI4Shipwrecks SSS characteristics.
- **Collection geometry mismatch:** SubPipe is AUV-mounted SSS. AI4Shipwrecks is also
  AUV-mounted (Iver3 + EdgeTech 2205, ~132 kHz, Thunder Bay). Platform type does not
  distinguish them — but sonar hardware and frequency are different: SubPipe's sensor
  is unspecified in the README/abstract; the frequency and shadow geometry of the two
  AUV surveys would still require re-validation before cross-training. More importantly,
  the GPL-3.0 license is a hard blocker regardless of geometry compatibility.
- **Single object class:** Only pipeline annotations — no diversity that would help
  generalise to seabed hazard detection broadly.

---

## Conclusion

SubPipe is investigated and closed. Pipe/cylinder detection is **not achievable** in
available time with publicly available, license-compatible SSS data.

This joins ghost nets and generic cylinders as a future-work item:
- No license-compatible annotated SSS pipeline dataset exists
- Sonar hardware and frequency of potential datasets would need verification against
  AI4Shipwrecks (Iver3/EdgeTech 2205, ~132 kHz) before cross-training
- Bounding-box-only supervision on the SSS channel would produce lower-fidelity
  output, requiring clear labelling as such

**What would be needed to achieve pipe detection:**
1. Obtain or collect SSS pipeline data with segmentation annotations from a survey
   using compatible sonar hardware (matching AI4Shipwrecks frequency range ~132 kHz)
2. Confirm license compatibility for model training and public submission
3. Train, evaluate on a held-out split, report IoU/Dice
4. Only then: add as a clearly-labelled second detection head

**Time estimate if data were available:** 1–2 weeks.

---

## Source

- Dataset: https://github.com/remaro-network/SubPipe-dataset (GPL-3.0, Oceanscan-MST)
- Paper: https://arxiv.org/abs/2401.17907 (Gonzalez-Garcia et al.)
