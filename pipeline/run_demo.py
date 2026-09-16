from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from confidence import score_detections
from geotag import PixelGeoMapper, SonarMeta, save_report
from infer import SonarDetector
from preprocess import preprocess
from synthaug import apply_sonar_synth

DATA = Path(r"D:\SIH\AI4Shipwrecks\AI4Shipwrecks")
OUT = Path(r"D:\SIH\pipeline\demo_out")


def run(image_path, meta=None, stress=False, out_dir=OUT, show=True):
    out_dir.mkdir(exist_ok=True)
    img = np.array(cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE), dtype=np.float32) / 255.0

    if stress:
        img = apply_sonar_synth(img)

    clean, gap = preprocess(img)
    detector = SonarDetector()
    prob, mask, dets = detector(clean, preprocessed=True)
    scored = score_detections(clean, prob, dets, mask)

    if meta is None:
        meta = SonarMeta(lat0=45.0855, lon0=-83.5684, heading_deg=90.0, altitude_m=15.0, range_m=50.0, along_track_m=120.0)
    mapper = PixelGeoMapper(clean.shape[0], clean.shape[1], meta)
    geod = mapper.geotag(scored)
    json_path, csv_path = save_report(geod, out_dir / "hazard_report")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(clean, cmap="gray")
    axes[0].set_title("Preprocessed sonar")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Segmentation mask")
    axes[2].imshow(clean, cmap="gray")
    for d in geod:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        color = "red" if d["likely_rock_or_shadow"] else "lime"
        axes[2].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.5))
        axes[2].text(x1, y1 - 4, f"{d['confidence']:.0f}", color=color, fontsize=8)
    axes[2].set_title("Detections (green=keep, red=suspected rock/shadow)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "overlay.png", dpi=130)

    print(f"Image: {image_path.name} | gap cols: {gap} | detections: {len(geod)}")
    for d in geod[:8]:
        print(
            f"  id {d['id']:2d} | conf {d['confidence']:5.1f} | prob {d['mean_prob']:.3f} "
            f"| geo {d['geo_score']:.2f} | shadow {d['shadow_score']:.2f} "
            f"| lat {d['lat']:.6f} lon {d['lon']:.6f}"
        )
    print(f"Report: {json_path} + {csv_path}")
    return geod


if __name__ == "__main__":
    test_img = DATA / "test" / "images" / "Corsair_01.png"
    print("=== Normal run ===")
    run(test_img)
    print("\n=== Stress test (synthetic speckle/shadow/distortion/heave) ===")
    run(test_img, stress=True)
