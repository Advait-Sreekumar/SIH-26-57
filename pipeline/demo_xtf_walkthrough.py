"""Manual walkthrough demo -- NOT collected by pytest. Not part of the Streamlit demo flow.
Run directly:  python pipeline/demo_xtf_walkthrough.py

This is not test coverage. See pipeline/test_pipeline.py and pipeline/test_review_store.py
for the actual pytest suite.

Two paths require local updates before running:
  XTF     (line 16): path to a real .xtf file on your machine
  out_dir (line 49): output directory; must exist before running
The demo will fail with FileNotFoundError on both if left as-is on any machine other than
the original dev environment.
"""
import cv2
import numpy as np

from confidence import score_detections
from geotag import PixelGeoMapper, save_report
from infer import SonarDetector
from preprocess import preprocess
from xtf_io import nav_to_meta, read_xtf

XTF = r"D:\SIH\external_test\synthetic_track.xtf"
MODEL_SIZE = 512


def detect_resized(det, clean, threshold=None):
    h, w = clean.shape
    resized = cv2.resize(clean, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
    prob_s, mask_s, dets_s = det(resized, preprocessed=True)
    sx, sy = w / MODEL_SIZE, h / MODEL_SIZE
    for d in dets_s:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        d["bbox_xyxy"] = [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]
        d["centroid_px"] = [d["centroid_px"][0] * sx, d["centroid_px"][1] * sy]
    prob = cv2.resize(prob_s, (w, h))
    mask = (prob > det.threshold).astype(np.uint8)
    return prob, mask, dets_s


def main():
    wf, nav, info = read_xtf(XTF)
    meta = nav_to_meta(nav, wf.shape)
    clean, gap = preprocess(wf)
    det = SonarDetector()
    prob, mask, dets = detect_resized(det, clean)
    scored = score_detections(clean, prob, dets, mask)
    geo = PixelGeoMapper(clean.shape[0], clean.shape[1], meta).geotag(scored)

    print(f"detections: {len(geo)}")
    for d in geo:
        print(
            f"  id {d['id']} | conf {d['confidence']} | prob {d['mean_prob']:.3f} "
            f"| lat {d['lat']:.6f} lon {d['lon']:.6f} | bbox {d['bbox_xyxy']}"
        )
    save_report(geo, "demo_out/xtf_walkthrough_report")  # output dir must exist
    print("report saved -> demo_out/xtf_hazard_report.json + .csv")

    target_lat = 45.0000 + 65 * 1e-5
    if geo:
        errs = [abs(d["lat"] - target_lat) for d in geo]
        print(f"closest detection to true target latitude: {min(errs):.6f} deg (~{min(errs) * 111320:.1f} m)")


if __name__ == "__main__":
    main()
