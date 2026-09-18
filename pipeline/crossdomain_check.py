"""Cross-domain generalization check (non-mine anomalous-object subset).

Runs the shipwreck-trained U-Net on the 24574879 dataset (Santos et al. 2024,
CC-BY-4.0, highly likely). Compares predicted bounding boxes (derived from mask
connected components) against ground-truth YOLO bounding boxes using IoU-of-boxes.

This is NOT a mine-detection evaluation. It measures whether features learned on
AUV-collected (Iver3 + EdgeTech 2205, ~132 kHz) shipwreck imagery transfer at all
to AUV-collected sonar imagery containing annotated man-made objects in a different
sensor domain.

Results are reported per class (MILCO / NOMBO) and for background (no-annotation)
frames separately. Do NOT compare the bbox-IoU numbers against our pixel-level test
IoU 0.427 -- they are different metrics on different data.

See docs/24574879_gonogo.md for domain mismatch documentation.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

DATASET_ROOT = _HERE.parent / "24574879"
YEAR_DIRS = ["2010", "2015", "2017", "2018", "2021"]
IOU_THRESHOLD = 0.1  # conservative: expected domain shift means 0.5 may be too strict
IOU_THRESHOLD_STRICT = 0.5  # also report standard PASCAL VOC threshold

# Class names per Training/obj.names.txt
CLASS_NAMES = {0: "MILCO", 1: "NOMBO"}


def load_yolo_boxes(txt_path: Path, img_w: int, img_h: int):
    """Parse YOLO annotation file -> list of (class_id, x1, y1, x2, y2) in pixels."""
    if not txt_path.exists():
        return []
    lines = [l.strip() for l in txt_path.read_text().strip().splitlines() if l.strip()]
    boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        boxes.append((cls, max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)))
    return boxes


def boxes_from_mask(mask: np.ndarray, min_area: int = 80):
    """Extract bounding boxes from binary mask via connected components."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    boxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, x + w, y + h))
    return boxes


def iou_boxes(a, b):
    """IoU between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def best_iou(pred_box, gt_boxes):
    """Max IoU between one predicted box and all GT boxes."""
    if not gt_boxes:
        return 0.0
    return max(iou_boxes(pred_box, (gx1, gy1, gx2, gy2)) for _, gx1, gy1, gx2, gy2 in gt_boxes)


def run():
    from infer import SonarDetector

    detector = SonarDetector()
    detector.threshold = 0.5

    # Collect all image paths
    all_pairs = []
    for year in YEAR_DIRS:
        ydir = DATASET_ROOT / year / year
        if not ydir.exists():
            ydir = DATASET_ROOT / year  # fallback if single-level
        for jpg in sorted(ydir.glob("*.jpg")):
            txt = jpg.with_suffix(".txt")
            all_pairs.append((jpg, txt))

    print(f"Found {len(all_pairs)} image/annotation pairs across {YEAR_DIRS}")
    print("Model: shipwreck U-Net (AI4Shipwrecks, Iver3 AUV / EdgeTech 2205, ~132 kHz). Domain: AUV SSS (different sensor).")
    print("Metric: IoU-of-boxes (NOT pixel IoU). Do not compare to test IoU 0.427.")
    print()

    # Per-class accumulators
    # class -> list of max IoU values for each GT object
    gt_ious = {0: [], 1: []}  # MILCO, NOMBO
    # False positive counts: predicted boxes on background frames
    bg_frames_total = 0
    bg_frames_with_prediction = 0
    fp_on_bg = 0  # total predicted boxes on background frames

    annotated_frames = {0: 0, 1: 0}  # frames containing each class
    errors = []

    for idx, (jpg_path, txt_path) in enumerate(all_pairs):
        if idx % 100 == 0:
            print(f"  [{idx}/{len(all_pairs)}] processing...")

        img_raw = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            errors.append(str(jpg_path))
            continue

        img_h, img_w = img_raw.shape
        img = img_raw.astype(np.float32) / 255.0

        # Resize to 512x512 for model (same as XTF path in app.py)
        resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

        # Run model
        try:
            prob, mask, _ = detector(resized, preprocessed=True)
        except Exception as e:
            errors.append(f"{jpg_path}: {e}")
            continue

        # Scale mask back to original image size
        mask_full = cv2.resize(
            (prob > detector.threshold).astype(np.uint8),
            (img_w, img_h),
            interpolation=cv2.INTER_NEAREST,
        )

        pred_boxes = boxes_from_mask(mask_full)
        gt_boxes = load_yolo_boxes(txt_path, img_w, img_h)

        if not gt_boxes:
            # Background frame
            bg_frames_total += 1
            if pred_boxes:
                bg_frames_with_prediction += 1
                fp_on_bg += len(pred_boxes)
        else:
            # For each GT box, find best matching predicted box
            for cls, gx1, gy1, gx2, gy2 in gt_boxes:
                annotated_frames[cls] = annotated_frames.get(cls, 0) + 1
                if pred_boxes:
                    best = max(
                        iou_boxes((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                        for (px1, py1, px2, py2) in pred_boxes
                    )
                else:
                    best = 0.0
                if cls in gt_ious:
                    gt_ious[cls].append(best)

    print(f"\nDone. Errors: {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(" ", e)

    # Compute stats
    results = {
        "framing": "cross-domain generalization check (non-mine anomalous-object subset)",
        "model": "shipwreck U-Net (AI4Shipwrecks, Iver3 AUV + EdgeTech 2205 ~132 kHz, CC-BY-4.0)",
        "eval_dataset": "Santos et al. 2024, 24574879, AUV SSS, license: highly likely CC-BY-4.0 (not independently confirmed at repo level)",
        "metric": "IoU-of-boxes (bbox derived from mask connected components vs YOLO GT bbox)",
        "note_do_not_compare": "These numbers are NOT comparable to pixel-level test IoU 0.427 (different metric, different domain)",
        "iou_threshold_strict": IOU_THRESHOLD_STRICT,
        "iou_threshold_lenient": IOU_THRESHOLD,
        "total_images": len(all_pairs),
        "background_frames": bg_frames_total,
        "bg_frames_with_any_prediction": bg_frames_with_prediction,
        "fp_boxes_on_background_frames": fp_on_bg,
        "false_positive_rate_on_bg_frames": round(bg_frames_with_prediction / max(1, bg_frames_total), 4),
        "per_class": {},
    }

    print("\n--- Cross-domain generalization check (non-mine anomalous-object subset) ---")
    print(f"Total images: {len(all_pairs)} | Background frames: {bg_frames_total}")
    print(f"FP rate on background frames: {bg_frames_with_prediction}/{bg_frames_total} = "
          f"{results['false_positive_rate_on_bg_frames']:.1%}")
    print()
    print(f"{'Class':<8} {'GT objs':>8} {'Mean IoU':>10} {'Det@0.1':>9} {'Det@0.5':>9}")
    print("-" * 48)

    for cls_id, cls_name in CLASS_NAMES.items():
        ious = gt_ious[cls_id]
        n = len(ious)
        if n == 0:
            print(f"{cls_name:<8} {'0':>8} {'N/A':>10} {'N/A':>9} {'N/A':>9}")
            results["per_class"][cls_name] = {"n_gt_objects": 0}
            continue
        mean_iou = float(np.mean(ious))
        det_lenient = sum(1 for v in ious if v >= IOU_THRESHOLD) / n
        det_strict = sum(1 for v in ious if v >= IOU_THRESHOLD_STRICT) / n
        print(f"{cls_name:<8} {n:>8} {mean_iou:>10.3f} {det_lenient:>8.1%} {det_strict:>8.1%}")
        results["per_class"][cls_name] = {
            "n_gt_objects": n,
            "mean_iou_of_boxes": round(mean_iou, 4),
            f"detection_rate_iou_ge_{IOU_THRESHOLD}": round(det_lenient, 4),
            f"detection_rate_iou_ge_{IOU_THRESHOLD_STRICT}": round(det_strict, 4),
            "iou_distribution": {
                "0.0": sum(1 for v in ious if v == 0.0),
                "0.0-0.1": sum(1 for v in ious if 0.0 < v < 0.1),
                "0.1-0.3": sum(1 for v in ious if 0.1 <= v < 0.3),
                "0.3-0.5": sum(1 for v in ious if 0.3 <= v < 0.5),
                ">=0.5": sum(1 for v in ious if v >= 0.5),
            },
        }

    out = _HERE / "crossdomain_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")
    return results


if __name__ == "__main__":
    run()
