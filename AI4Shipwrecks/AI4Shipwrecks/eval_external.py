import random
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage
from tqdm import tqdm

# ============================================================
# IMPORTANT DISCLAIMER: SCTD is an aerial/UAV photography dataset
# (ships and aircraft from above-water drones), NOT a sonar dataset.
# Running this script produces cross-domain metrics with no validity
# for sonar hazard detection. Results must NOT be cited as external
# sonar validation. See docs/phase0_gap_analysis.md Finding #2.
# The primary held-out evaluation is the site-based test split in
# AI4Shipwrecks/test/ (13 wreck sites never seen during training).
# ============================================================

# Resolve SCTD root: prefer SONAR_SCTD env var, then relative repo layout
import os as _os
_REPO_ROOT_EXT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SCTD = _REPO_ROOT_EXT / "external_test" / "SCTD" / "SCTD"
SCTD_ROOT = Path(_os.environ.get("SONAR_SCTD", str(_DEFAULT_SCTD)))
if not SCTD_ROOT.exists():
    print(
        f"[eval_external] WARNING: SCTD not found at {SCTD_ROOT}. "
        "Set SONAR_SCTD env var to override. "
        "NOTE: SCTD is aerial photography, NOT sonar — results have no sonar-validation meaning."
    )
CKPT = Path(__file__).resolve().parent / "runs" / "best_model_v1_iou0.71.pth"
OUT_DIR = Path(__file__).resolve().parent / "runs"
IMG_SIZE = 512
HIT_FRAC = 0.10
SEED = 42


def has_scse(state_dict):
    return any("attention" in k for k in state_dict.keys())


def load_model(device):
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    model = smp.Unet(
        encoder_name=ckpt.get("encoder", "resnet34"),
        encoder_weights=None,
        in_channels=1,
        classes=1,
        decoder_attention_type="scse" if has_scse(ckpt["model_state_dict"]) else None,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def parse_annotations():
    samples = []
    for xml_path in sorted((SCTD_ROOT / "Annotations").glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        boxes = []
        for obj in root.iter("object"):
            name = obj.find("name").text.strip().lower()
            if name != "ship":
                continue
            b = obj.find("bndbox")
            boxes.append(
                (
                    float(b.find("xmin").text),
                    float(b.find("ymin").text),
                    float(b.find("xmax").text),
                    float(b.find("ymax").text),
                )
            )
        if boxes:
            samples.append((SCTD_ROOT / "JPEGImages" / root.find("filename").text, boxes))
    return samples


@torch.no_grad()
def main():
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    samples = parse_annotations()
    print(f"SCTD shipwreck images with boxes: {len(samples)}")

    hits = 0
    total_boxes = 0
    pred_px_total = 0.0
    pred_px_in_boxes = 0.0
    per_image = []
    vis = []

    for img_path, boxes in tqdm(samples):
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        h, w = img.shape
        tensor = F.interpolate(
            torch.from_numpy(img)[None, None],
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).to(device)
        pred = (model(tensor).sigmoid()[0, 0] > 0.5).cpu().float()

        sx, sy = IMG_SIZE / w, IMG_SIZE / h
        boxes512 = [
            (x1 * sx, y1 * sy, x2 * sx, y2 * sy) for x1, y1, x2, y2 in boxes
        ]

        box_mask = torch.zeros(IMG_SIZE, IMG_SIZE)
        for x1, y1, x2, y2 in boxes512:
            box_mask[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1] = 1.0

        n_hit = 0
        for x1, y1, x2, y2 in boxes512:
            area = max((y2 - y1) * (x2 - x1), 1.0)
            inside = pred[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1].sum().item()
            n_hit += inside / area >= HIT_FRAC
        hits += n_hit
        total_boxes += len(boxes512)

        pred_px = pred.sum().item()
        pred_px_total += pred_px
        pred_px_in_boxes += (pred * box_mask).sum().item()

        iou = (pred * box_mask).sum().item() / max(((pred + box_mask) > 0).sum().item(), 1e-7)
        per_image.append((img_path.name, n_hit, len(boxes512), iou))
        if len(vis) < 12 and pred_px > 50:
            vis.append((img_path.name, tensor[0, 0].cpu(), boxes512, pred))

    print(f"\nBox detection rate (@{int(HIT_FRAC * 100)}% of box covered): {hits}/{total_boxes} = {hits / max(total_boxes, 1):.4f}")
    print(f"Prediction inside GT boxes: {pred_px_in_boxes / max(pred_px_total, 1e-7):.4f}")
    mean_iou = np.mean([iou for _, _, _, iou in per_image])
    print(f"Mean mask-vs-box IoU: {mean_iou:.4f}")

    misses = [(n, nh, nb, i) for n, nh, nb, i in per_image if nh < nb]
    print(f"Images with >=1 missed box: {len(misses)}/{len(per_image)}")

    fig, axes = plt.subplots(4, 3, figsize=(13, 16))
    for ax, (name, img, boxes, pred) in zip(axes.flat, vis):
        ax.imshow(img, cmap="gray")
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=1.5))
        ax.contour(pred.numpy(), levels=[0.5], colors="red", linewidths=1.2)
        ax.set_title(name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("SCTD external check: green = GT box, red = model prediction", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sctd_external_check.png", dpi=110)
    print(f"Saved visualization -> {OUT_DIR / 'sctd_external_check.png'}")


if __name__ == "__main__":
    main()
