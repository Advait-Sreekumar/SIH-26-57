"""In-domain FP rate: run shipwreck model on GT-negative test frames,
report per-frame detection count and aggregate FP rate.

Usage: python pipeline/fp_rate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_PIPELINE = Path(__file__).parent
_REPO = _PIPELINE.parent
sys.path.insert(0, str(_PIPELINE))

from infer import SonarDetector

TEST_DIR = _REPO / "AI4Shipwrecks" / "AI4Shipwrecks" / "test"
OUT_PATH = _PIPELINE / "fp_rate_results.json"


def main() -> None:
    label_dir = TEST_DIR / "labels"
    image_dir = TEST_DIR / "images"

    neg_stems = []
    for p in sorted(label_dir.iterdir()):
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.max() == 0:
            neg_stems.append(p.stem)

    print("GT-negative frames: {}".format(len(neg_stems)))

    detector = SonarDetector()  # uses default CKPT from infer.py

    per_frame = []
    fp_count = 0
    for stem in sorted(neg_stems):
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = image_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print("  SKIP {}: image not found".format(stem))
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("  SKIP {}: cv2 could not read".format(stem))
            continue

        _prob, _mask, dets = detector(img)
        n = len(dets)
        scores = []
        for d in dets:
            s = d.get("score", d.get("model_score", 0))
            scores.append(round(float(s), 3))
        per_frame.append({"stem": stem, "n_detections": n, "scores": scores})
        if n > 0:
            fp_count += 1
        print("  {}: {} detections".format(stem, n))

    total = len(per_frame)
    fp_rate = fp_count / total if total > 0 else float("nan")
    mean_n = sum(r["n_detections"] for r in per_frame) / total if total > 0 else float("nan")

    result = {
        "framing": "in-domain FP rate on GT-negative test frames (no ground-truth annotation)",
        "n_gt_negative_frames": total,
        "n_frames_with_any_detection": fp_count,
        "fp_rate": round(fp_rate, 4),
        "mean_detections_per_neg_frame": round(mean_n, 3),
        "note": (
            "GT-negative means the label mask is all-zero. "
            "These frames may contain non-wreck seafloor, shadow, or rock features. "
            "Any detection on these frames is a false positive at the frame level."
        ),
        "per_frame": per_frame,
    }

    OUT_PATH.write_text(json.dumps(result, indent=2))
    print("\nResults: {}/{} frames fired ({:.1%})".format(fp_count, total, fp_rate))
    print("Mean detections/neg-frame: {:.2f}".format(mean_n))
    print("Written: {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
