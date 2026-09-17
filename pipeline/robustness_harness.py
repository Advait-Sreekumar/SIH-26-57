"""Robustness harness: measure per-perturbation IoU delta on held-out test images.

Outputs pipeline/robustness_results.json with before/after IoU for each of the four
synthaug.py perturbation types applied deterministically (seed fixed).

Usage:
    cd <repo_root>
    python pipeline/robustness_harness.py

Requires: SONAR_TEST_IMAGES env var OR the default test image path to resolve.
Test images must be grayscale PNG/JPEG; paired mask PNGs must exist alongside them
in a `labels/` sibling directory at the same level as `images/`.

If no ground-truth masks are available, pass --no-masks to measure detection count
(n_detections) instead of IoU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Resolve paths relative to this file
_PIPELINE = Path(__file__).parent
_REPO_ROOT = _PIPELINE.parent
sys.path.insert(0, str(_PIPELINE))

from synthaug import heave_pitch_roll, radial_distortion, speckle_noise, synthesize_shadow
from infer import SonarDetector
from preprocess import preprocess

RNG_SEED = 42
DEFAULT_TEST_DIR = _REPO_ROOT / "AI4Shipwrecks" / "AI4Shipwrecks" / "test"
OUT_PATH = _PIPELINE / "robustness_results.json"

# Each perturbation applied at a single fixed parameterisation (deterministic, no stochastic p)
PERTURBATIONS = {
    "speckle_noise": lambda img: speckle_noise(img, look=1.0),
    "synthesize_shadow": lambda img: synthesize_shadow(img, n_shadows=2, strength=0.75,
                                                        direction=0.785),  # 45 deg, fixed
    "radial_distortion": lambda img: radial_distortion(img, k1=-0.15, k2=0.05),
    "heave_pitch_roll": lambda img: _heave_fixed(img),
}


def _heave_fixed(img: np.ndarray) -> np.ndarray:
    """heave_pitch_roll uses np.random internally; seed before calling."""
    np.random.seed(RNG_SEED)
    return heave_pitch_roll(img, max_shift=12, max_shear=0.06)


def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = float((pred_mask & gt_mask).sum())
    union = float((pred_mask | gt_mask).sum())
    return inter / union if union > 0 else float("nan")


def load_test_images(test_dir: Path, use_masks: bool) -> list[tuple]:
    """Return list of (img_array_float32, gt_mask_or_None, stem)."""
    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {images_dir}")
    paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No PNG/JPEG images found in {images_dir}")
    records = []
    for p in paths:
        raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        img = raw.astype(np.float32) / 255.0
        gt = None
        if use_masks:
            mask_path = labels_dir / (p.stem + ".png")
            if mask_path.exists():
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                gt = (m > 0).astype(np.uint8) if m is not None else None
        records.append((img, gt, p.stem))
    return records


def run_harness(test_dir: Path, use_masks: bool = True) -> dict:
    np.random.seed(RNG_SEED)
    detector = SonarDetector()
    records = load_test_images(test_dir, use_masks)
    n = len(records)
    print(f"Loaded {n} test images from {test_dir}")

    results: dict = {
        "harness": "robustness_harness.py",
        "n_images": n,
        "rng_seed": RNG_SEED,
        "test_dir": str(test_dir),
        "metric": "pixel_iou" if use_masks else "n_detections",
        "baseline": {},
        "perturbations": {},
    }

    def _score(img_float: np.ndarray, gt: np.ndarray | None) -> dict:
        proc, _ = preprocess(img_float)
        prob, mask = detector.predict(proc, preprocessed=True)
        dets = detector.extract_detections(prob, mask)
        out: dict = {"n_detections": len(dets)}
        if gt is not None and use_masks:
            # Resize mask to match preprocessed image dimensions
            resized_gt = cv2.resize(gt, (mask.shape[1], mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            out["iou"] = _compute_iou(mask, resized_gt)
        return out

    # Baseline: clean images
    base_ious, base_dets = [], []
    for img, gt, stem in records:
        s = _score(img, gt)
        base_dets.append(s["n_detections"])
        if "iou" in s:
            base_ious.append(s["iou"])

    results["baseline"] = {
        "mean_iou": float(np.nanmean(base_ious)) if base_ious else None,
        "mean_n_detections": float(np.mean(base_dets)),
        "n_valid_iou": len([x for x in base_ious if not np.isnan(x)]),
    }
    print(f"  Baseline  mean_iou={results['baseline']['mean_iou']:.4f}  "
          f"mean_dets={results['baseline']['mean_n_detections']:.2f}")

    # Per-perturbation
    for pert_name, pert_fn in PERTURBATIONS.items():
        np.random.seed(RNG_SEED)
        ious, dets = [], []
        for img, gt, _ in records:
            perturbed = pert_fn(img.copy())
            s = _score(perturbed, gt)
            dets.append(s["n_detections"])
            if "iou" in s:
                ious.append(s["iou"])
        mean_iou = float(np.nanmean(ious)) if ious else None
        mean_dets = float(np.mean(dets))
        baseline_iou = results["baseline"]["mean_iou"]
        delta = (mean_iou - baseline_iou) if (mean_iou is not None and baseline_iou is not None) else None
        results["perturbations"][pert_name] = {
            "mean_iou": mean_iou,
            "mean_n_detections": mean_dets,
            "iou_delta_vs_baseline": round(delta, 4) if delta is not None else None,
            "n_valid_iou": len([x for x in ious if not np.isnan(x)]),
        }
        print(f"  {pert_name:<22s} mean_iou={mean_iou:.4f}  "
              f"delta={delta:+.4f}  mean_dets={mean_dets:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Robustness harness for sonar detector")
    parser.add_argument("--test-dir", default=None,
                        help="Path to test directory (must contain images/ subdirectory)")
    parser.add_argument("--no-masks", action="store_true",
                        help="Skip IoU computation (no ground-truth masks available)")
    parser.add_argument("--out", default=str(OUT_PATH),
                        help="Output JSON path")
    args = parser.parse_args()

    test_dir_str = args.test_dir or os.environ.get("SONAR_TEST_IMAGES")
    test_dir = Path(test_dir_str) if test_dir_str else DEFAULT_TEST_DIR
    if not test_dir.exists():
        print(f"ERROR: test directory not found: {test_dir}", file=sys.stderr)
        print("Set --test-dir or SONAR_TEST_IMAGES env var.", file=sys.stderr)
        sys.exit(1)

    results = run_harness(test_dir, use_masks=not args.no_masks)

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
