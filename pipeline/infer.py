from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from preprocess import TileBlender, preprocess, tile_image

# Resolve checkpoint relative to this file: pipeline/../AI4Shipwrecks/AI4Shipwrecks/runs/
_REPO_ROOT = Path(__file__).parent.parent
CKPT = _REPO_ROOT / "AI4Shipwrecks" / "AI4Shipwrecks" / "runs" / "best_model_v1_iou0.71.pth"
if not CKPT.exists():
    # Allow override via environment variable for CI / non-standard layouts
    import os
    _env = os.environ.get("SONAR_CKPT")
    if _env:
        CKPT = Path(_env)
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at {CKPT}. "
            "Set the SONAR_CKPT environment variable to the correct path."
        )


def has_scse(state_dict):
    return any("attention" in k for k in state_dict.keys())


def load_detector(ckpt_path=CKPT, device="cuda"):
    # weights_only=False: checkpoint is a repo-internal dict {"encoder": str,
    # "model_state_dict": OrderedDict}. torch>=2.9 weights_only=True only allows
    # plain tensors; the string "encoder" key requires full unpickling.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = smp.Unet(
        encoder_name=ckpt.get("encoder", "resnet34"),
        encoder_weights=None,
        in_channels=1,
        classes=1,
        decoder_attention_type="scse" if has_scse(ckpt["model_state_dict"]) else None,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


class SonarDetector:
    def __init__(self, ckpt_path=CKPT, device=None, tile=512, overlap=64, threshold=0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_detector(ckpt_path, self.device)
        self.tile = tile
        self.overlap = overlap
        self.threshold = threshold

    @torch.no_grad()
    def predict_logits(self, img, preprocessed=False):
        if not preprocessed:
            img, _ = preprocess(img)
        h, w = img.shape
        pad_h = max(0, self.tile - h)
        pad_w = max(0, self.tile - w)
        if pad_h or pad_w:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
        blender = TileBlender(img.shape[0], img.shape[1], self.tile, self.overlap)
        tiles = tile_image(img, self.tile, self.overlap)
        for y, x, patch in tiles:
            t = torch.from_numpy(patch)[None, None].to(self.device)
            logits = self.model(t)[0, 0].float().cpu().numpy()
            blender.add(y, x, logits)
        return blender.result()[:h, :w]

    def predict(self, img, preprocessed=False):
        prob = 1 / (1 + np.exp(-self.predict_logits(img, preprocessed)))
        return prob, (prob > self.threshold).astype(np.uint8)

    def extract_detections(self, prob, mask=None, min_area=80):
        mask = (prob > self.threshold).astype(np.uint8) if mask is None else mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        dets = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x, y, bw, bh = stats[i, :4]
            comp = labels == i
            mean_prob = float(prob[comp].mean())
            peak_prob = float(prob[comp].max())
            dets.append(
                {
                    "id": len(dets),
                    "bbox_xyxy": [int(x), int(y), int(x + bw), int(y + bh)],
                    "area_px": area,
                    "centroid_px": [float(centroids[i][0]), float(centroids[i][1])],
                    "mean_prob": mean_prob,
                    "peak_prob": peak_prob,
                }
            )
        return dets

    def __call__(self, img, preprocessed=False):
        prob, mask = self.predict(img, preprocessed)
        return prob, mask, self.extract_detections(prob, mask)


def false_positive_rate(detector, images, gt_masks, threshold=None):
    if threshold:
        detector.threshold = threshold
    fp_total, img_total = 0, 0
    for img, gt in zip(images, gt_masks):
        _, mask, dets = detector(img)
        n_gt = cv2.connectedComponents(gt, 8)[0] - 1 if gt is not None and gt.any() else 0
        fp = 0
        for d in dets:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            if gt is None or not gt[y1:y2, x1:x2].any():
                fp += 1
        fp_total += fp
        img_total += 1
        _ = n_gt
    return fp_total / max(img_total, 1)
