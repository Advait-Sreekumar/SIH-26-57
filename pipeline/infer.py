"""
Sonar anomaly detector — U-Net / ResNet34.

Checkpoint resolution order (resolved lazily, never at import time):
  1. SONAR_CKPT  env var — explicit local path
  2. Repo-relative default:  <repo>/AI4Shipwrecks/AI4Shipwrecks/runs/best_model_v1_iou0.71.pth
  3. Download from CKPT_URL env var (or st.secrets["CKPT_URL"]) to ~/.cache/sonar/
     TODO: set CKPT_URL in Streamlit Cloud → App Settings → Secrets to the
           GitHub-Release asset download URL, e.g.
           https://github.com/Advait-Sreekumar/SIH-26-57/releases/download/model-weights-v1/best_model_v1_iou0.71.pth
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F  # noqa: F401  (re-exported for callers)

from preprocess import TileBlender, preprocess, tile_image

# ---------------------------------------------------------------------------
# Internal constants — not evaluated at import time
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CKPT = (
    _REPO_ROOT / "AI4Shipwrecks" / "AI4Shipwrecks" / "runs" / "best_model_v1_iou0.71.pth"
)
_CKPT_FILENAME = "best_model_v1_iou0.71.pth"
_CACHE_DIR = Path.home() / ".cache" / "sonar"


# ---------------------------------------------------------------------------
# Checkpoint resolution (lazy — call only when you actually need the model)
# ---------------------------------------------------------------------------

def _get_ckpt_url() -> Optional[str]:
    """Read CKPT_URL from env or st.secrets (if Streamlit is active)."""
    url = os.environ.get("CKPT_URL")
    if url:
        return url
    try:
        import streamlit as st  # noqa: PLC0415
        return st.secrets.get("CKPT_URL")
    except Exception:
        return None


def _download_checkpoint(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest*, skipping if already present."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import streamlit as st  # noqa: PLC0415
        _show = lambda msg: st.info(msg)  # noqa: E731
    except Exception:
        _show = print

    _show(f"Downloading model checkpoint from {url.split('?')[0]} …")
    import urllib.request

    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp))
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    _show(f"Checkpoint saved to {dest}")


def resolve_checkpoint() -> Path:
    """
    Return the path to the checkpoint file, downloading it if necessary.

    Raises RuntimeError (not FileNotFoundError) so callers can catch it
    without crashing the entire Streamlit process at import time.
    """
    # 1. Explicit env override
    env_path = os.environ.get("SONAR_CKPT")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise RuntimeError(
            f"SONAR_CKPT is set to '{env_path}' but the file does not exist."
        )

    # 2. Repo-local default
    if _DEFAULT_CKPT.exists():
        return _DEFAULT_CKPT

    # 3. Cached download
    cached = _CACHE_DIR / _CKPT_FILENAME
    if cached.exists():
        return cached

    url = _get_ckpt_url()
    if url:
        _download_checkpoint(url, cached)
        return cached

    # 4. Nothing worked
    raise RuntimeError(
        f"Model checkpoint not found.\n"
        f"  Tried: {_DEFAULT_CKPT}\n"
        f"  Tried: {cached}\n"
        "  Fix: set the CKPT_URL secret/env-var to the hosted download URL, or\n"
        "       set SONAR_CKPT to a local path.\n"
        "  See: pipeline/infer.py docstring for the expected URL format."
    )


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def has_scse(state_dict) -> bool:
    return any("attention" in k for k in state_dict.keys())


def load_detector(ckpt_path: Optional[Path] = None, device: str = "cpu"):
    """
    Load and return the U-Net model.

    *ckpt_path* defaults to resolve_checkpoint().
    weights_only=False: checkpoint is a repo-internal dict
        {"encoder": str, "model_state_dict": OrderedDict}.
    torch >= 2.9 weights_only=True only allows plain tensors; the string
    "encoder" value requires full unpickling — safe for our own file.
    map_location="cpu": Streamlit Cloud has no GPU.
    """
    if ckpt_path is None:
        ckpt_path = resolve_checkpoint()
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = smp.Unet(
        encoder_name=ckpt.get("encoder", "resnet34"),
        encoder_weights=None,
        in_channels=1,
        classes=1,
        decoder_attention_type="scse" if has_scse(ckpt["model_state_dict"]) else None,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class SonarDetector:
    def __init__(
        self,
        ckpt_path: Optional[Path] = None,
        device: Optional[str] = None,
        tile: int = 512,
        overlap: int = 64,
        threshold: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Resolve lazily so import doesn't fail on deployed app
        if ckpt_path is None:
            ckpt_path = resolve_checkpoint()
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


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

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
