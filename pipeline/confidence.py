import cv2
import numpy as np


def _component_stats(gray, comp_mask):
    ys, xs = np.nonzero(comp_mask)
    if len(xs) == 0:
        return 1.0, 1.0, 0.0
    h, w = comp_mask.shape
    area = len(xs)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    major = max(bw, bh)
    minor = max(min(bw, bh), 1)
    aspect = major / minor
    contour, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = 1.0
    solidity = 1.0
    if contour:
        hull = cv2.convexHull(contour[0] if len(contour) == 1 else np.vstack(contour))
        hull_area = max(cv2.contourArea(hull), 1.0)
        solidity = min(area / hull_area, 1.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    boundary = cv2.dilate(comp_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~comp_mask
    inner_grad = float(grad[comp_mask].mean()) if area else 0.0
    bound_grad = float(grad[boundary].mean()) if boundary.any() else 0.0
    edge_sharpness = np.tanh(bound_grad / (inner_grad + 1e-6) - 1.0)
    return aspect, solidity, edge_sharpness


def shadow_score(gray, comp_mask, search=40, dark_frac=0.25):
    h, w = gray.shape
    ys, xs = np.nonzero(comp_mask)
    if len(xs) == 0:
        return 0.0
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    ring = np.zeros((h, w), np.float32)
    rx0, rx1 = max(0, x0 - search), min(w, x1 + search)
    ry0, ry1 = max(0, y0 - search), min(h, y1 + search)
    ring[ry0:ry1, rx0:rx1] = 1.0
    ring[ry0 + bh // 2 : ry1 - bh // 2, rx0 + bw // 2 : rx1 - bw // 2] = 0.0
    ring = ring.astype(bool) & ~comp_mask
    if not ring.any() or not comp_mask.any():
        return 0.0
    target_med = np.median(gray[comp_mask])
    ring_med = np.median(gray[ring])
    contrast = (target_med - ring_med) / (ring_med + 1e-6)
    return float(np.clip(contrast, 0, 1))


def geometric_confidence(gray, comp_mask):
    aspect, solidity, edge = _component_stats(gray, comp_mask)
    aspect_score = float(np.clip(1.0 - abs(aspect - 3.5) / 3.5, 0, 1))
    solidity_score = float(np.clip((solidity - 0.4) / 0.5, 0, 1))
    edge_score = float(np.clip(edge, 0, 1))
    geo = (
        0.40 * aspect_score
        + 0.25 * solidity_score
        + 0.35 * edge_score
    )
    return geo, {"aspect": aspect, "solidity": solidity, "edge": edge_score}


def score_detections(gray, prob, detections, mask, alpha=0.6):
    results = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        comp = np.zeros(gray.shape, bool)
        comp[y1:y2, x1:x2] = mask[y1:y2, x1:x2].astype(bool)
        geo, geo_parts = geometric_confidence(gray, comp)
        sh = shadow_score(gray, comp)
        model_score = 0.5 * d["mean_prob"] + 0.5 * d["peak_prob"]
        conf = 100.0 * (alpha * model_score + (1 - alpha) * (0.7 * geo + 0.3 * sh))
        results.append(
            {
                **d,
                "confidence": round(conf, 1),
                "geo_score": round(geo, 3),
                "shadow_score": round(sh, 3),
                "geo_parts": {k: round(v, 3) for k, v in geo_parts.items()},
                "likely_rock_or_shadow": conf < 35.0,
            }
        )
    results.sort(key=lambda d: -d["confidence"])
    return results
