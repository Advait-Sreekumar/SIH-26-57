"""Illustrative, unvalidated, synthetic-only exploration -- not a demonstrated capability.

Generates synthetic sonar-like patches with net-like shapes and runs the existing
shipwreck-trained model on them to observe whether it produces any detections.

Framing -- READ BEFORE INTERPRETING RESULTS:
  Any model activation on a synthetic shape does NOT demonstrate real ghost-net
  detection capability. The model was never trained on net imagery. Activations may
  reflect responses to texture statistics, not meaningful structural features.
  An absence of activation is equally informative. Both outcomes are reported plainly.

  This is: illustrative, unvalidated, synthetic-only exploration -- not a demonstrated capability.
  Do NOT run during the primary demo. Do NOT present results as a detection capability claim.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

VISUALS_DIR = Path(__file__).parent / "net_exploration_visuals"
RESULTS_PATH = Path(__file__).parent / "net_exploration_results.json"
N_PATCHES = 30
PATCH_SIZE = 512
RNG_SEED = 7


def synthetic_seafloor(rng, h=PATCH_SIZE, w=PATCH_SIZE):
    base = np.full((h, w), 0.28, dtype=np.float32)
    coarse = rng.uniform(0.0, 0.08, size=(h // 32 + 2, w // 32 + 2)).astype(np.float32)
    coarse_up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    base = np.clip(base + coarse_up, 0.05, 0.6)
    speckle = rng.rayleigh(scale=1.0 / np.sqrt(2.0), size=(h, w)).astype(np.float32)
    speckle = np.clip(speckle, 0.3, 2.5)
    img = np.clip(base * speckle, 0.0, 1.0)
    stripe = 1.0 + 0.04 * np.sin(np.linspace(0, 80 * np.pi, h)).astype(np.float32)
    img = np.clip(img * stripe[:, None], 0.0, 1.0)
    return img


def draw_net_shape(rng, h=PATCH_SIZE, w=PATCH_SIZE):
    canvas = np.zeros((h, w), dtype=np.float32)
    n_strands = int(rng.integers(3, 7))
    y_starts = rng.uniform(h * 0.2, h * 0.8, size=n_strands)
    y_ends = y_starts + rng.uniform(-h * 0.25, h * 0.25, size=n_strands)
    y_ends = np.clip(y_ends, h * 0.05, h * 0.95)
    x_left = int(w * rng.uniform(0.05, 0.25))
    x_right = int(w * rng.uniform(0.75, 0.95))
    strand_pts = []
    for i in range(n_strands):
        pts = np.array([
            [x_left, y_starts[i]],
            [w * rng.uniform(0.35, 0.65), (y_starts[i] + y_ends[i]) / 2 + rng.uniform(-40, 40)],
            [x_right, y_ends[i]],
        ], dtype=np.float32)
        t = np.linspace(0, 1, 120)
        curve = (
            (1 - t)[:, None] ** 2 * pts[0]
            + 2 * (1 - t)[:, None] * t[:, None] * pts[1]
            + t[:, None] ** 2 * pts[2]
        ).astype(np.int32)
        strand_pts.append(curve)
        lw = int(rng.integers(1, 4))
        intens = float(rng.uniform(0.55, 0.90))
        for j in range(len(curve) - 1):
            cv2.line(canvas, tuple(curve[j]), tuple(curve[j + 1]), intens, lw)
    n_cross = int(rng.integers(4, 10))
    for _ in range(n_cross):
        idx = int(rng.integers(0, max(1, len(strand_pts) - 1)))
        t_pick = int(rng.integers(0, len(strand_pts[idx])))
        p1 = strand_pts[idx][t_pick]
        idx2 = (idx + 1) % len(strand_pts)
        t_pick2 = int(rng.integers(0, len(strand_pts[idx2])))
        p2 = strand_pts[idx2][t_pick2]
        lw = int(rng.integers(1, 3))
        intens = float(rng.uniform(0.45, 0.80))
        cv2.line(canvas, tuple(p1), tuple(p2), intens, lw)
    shadow = cv2.dilate(canvas, np.ones((5, 15), np.uint8), iterations=1) * rng.uniform(0.15, 0.35)
    shadow = np.where(canvas > 0.01, 0.0, shadow)
    return canvas, shadow


def make_net_patch(rng):
    bg = synthetic_seafloor(rng)
    net, shadow = draw_net_shape(rng)
    attenuation = np.where(net > 0.01, 0.55, 1.0).astype(np.float32)
    img = bg * attenuation
    img = np.clip(img - shadow, 0.0, 1.0)
    img = np.clip(img + net * 0.35, 0.0, 1.0)
    return img


def run():
    from infer import SonarDetector

    print("Illustrative, unvalidated, synthetic-only exploration -- not a demonstrated capability.")
    print(f"Generating {N_PATCHES} synthetic net-like patches and running shipwreck model...")
    print()
    VISUALS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    try:
        detector = SonarDetector()
    except Exception as e:
        print(f"Could not load model: {e}")
        detector = None
    records = []
    total_detections = 0
    patches_with_detections = 0
    for i in range(N_PATCHES):
        patch = make_net_patch(rng)
        vis_path = VISUALS_DIR / f"net_patch_{i:03d}.png"
        cv2.imwrite(str(vis_path), (patch * 255).astype(np.uint8))
        n_dets = 0
        conf_scores = []
        if detector is not None:
            try:
                prob, mask, dets = detector(patch, preprocessed=True)
                n_dets = len(dets)
                conf_scores = [round(float(d.get("mean_prob", 0.0)), 4) for d in dets]
                overlay_bgr = cv2.cvtColor((patch * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                mask_vis = (mask * 120).astype(np.uint8)
                overlay_bgr[:, :, 2] = np.clip(
                    overlay_bgr[:, :, 2].astype(np.uint16) + mask_vis, 0, 255
                ).astype(np.uint8)
                cv2.imwrite(str(VISUALS_DIR / f"net_patch_{i:03d}_overlay.png"), overlay_bgr)
            except Exception as e:
                print(f"  patch {i}: inference error: {e}")
        total_detections += n_dets
        if n_dets > 0:
            patches_with_detections += 1
        records.append({"patch_id": i, "n_detections": n_dets, "model_prob_scores": conf_scores})
        if i % 5 == 0:
            print(f"  [{i+1:2d}/{N_PATCHES}] patch {i:03d}: {n_dets} detection(s)")
    detection_rate = patches_with_detections / N_PATCHES
    mean_dets = total_detections / N_PATCHES
    print()
    print(f"--- Results (illustrative, unvalidated, synthetic-only) ---")
    print(f"Patches generated:            {N_PATCHES}")
    print(f"Patches with >= 1 detection:  {patches_with_detections} ({detection_rate:.1%})")
    print(f"Total detections:             {total_detections}")
    print(f"Mean detections / patch:      {mean_dets:.2f}")
    print()
    print("Reminder: a detection does NOT mean the model recognises ghost nets.")
    print("The model was trained on shipwreck imagery only.")
    results = {
        "framing": "illustrative, unvalidated, synthetic-only exploration -- not a demonstrated capability",
        "model": "shipwreck U-Net (AI4Shipwrecks, Iver3 AUV + EdgeTech 2205 ~132 kHz) -- not retrained, not fine-tuned",
        "n_patches": N_PATCHES,
        "rng_seed": RNG_SEED,
        "patches_with_detections": patches_with_detections,
        "detection_rate": round(detection_rate, 4),
        "total_detections": total_detections,
        "mean_detections_per_patch": round(mean_dets, 4),
        "per_patch": records,
        "interpretation": (
            "Detection rate measures how often the shipwreck model fires on synthetic shapes. "
            "High rate = model not selective (expected on out-of-distribution input). "
            "Low rate = model does not generalise. Neither validates ghost-net detection."
        ),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run()
