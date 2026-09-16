import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from confidence import score_detections
from geotag import PixelGeoMapper, SonarMeta, save_report
from infer import SonarDetector
from preprocess import preprocess
from synthaug import apply_sonar_synth
from xtf_io import load_input, nav_to_meta, slant_range_correct

st.set_page_config(page_title="Sonar Hazard Detector", layout="wide")
st.title("Sonar Hazard Detection & Reporting")
st.caption(
    "Shipwreck segmentation (U-Net/ResNet34 | val IoU 0.71 / test IoU 0.43) "
    "+ heuristic confidence scoring (uncalibrated 0–100 score) "
    "+ geotagged hazard reports"
)

_REPO_ROOT = Path(__file__).parent.parent
DATA_IMG = _REPO_ROOT / "AI4Shipwrecks" / "AI4Shipwrecks" / "test" / "images"
if not DATA_IMG.exists():
    import os as _os
    _env = _os.environ.get("SONAR_TEST_IMAGES")
    DATA_IMG = Path(_env) if _env else DATA_IMG


@st.cache_resource
def get_detector():
    return SonarDetector()


input_mode = st.sidebar.radio("Input source", ["Image file", "XTF sonar log"])

meta_from_file = None
file_bytes = None
name = None
if input_mode == "Image file":
    uploaded = st.sidebar.file_uploader("Upload sonar image", type=["png", "jpg", "jpeg"])
    sample = st.sidebar.selectbox("...or pick a sample", [""] + sorted(p.name for p in DATA_IMG.glob("*.png")))
    if uploaded:
        file_bytes, name = uploaded.getvalue(), uploaded.name
    elif sample:
        file_bytes, name = (DATA_IMG / sample).read_bytes(), sample
else:
    uploaded = st.sidebar.file_uploader("Upload .xtf sonar log", type=["xtf"])
    if uploaded:
        file_bytes, name = uploaded.getvalue(), uploaded.name

threshold = st.sidebar.slider("Segmentation threshold", 0.3, 0.8, 0.5, 0.05)
min_area = st.sidebar.slider("Min detection area (px)", 20, 500, 80, 10)
stress = st.sidebar.checkbox("Apply synthetic noise (stress test)", disabled=input_mode == "XTF sonar log")

if input_mode == "Image file":
    lat0 = st.sidebar.number_input("Origin lat", value=45.0855, format="%.6f")
    lon0 = st.sidebar.number_input("Origin lon", value=-83.5684, format="%.6f")
    heading = st.sidebar.slider("Vehicle heading (deg)", 0.0, 359.0, 90.0)
    altitude = st.sidebar.slider("Altitude (m)", 5.0, 50.0, 15.0)
    range_m = st.sidebar.slider("Sonar range per side (m)", 10.0, 100.0, 50.0)
    along_track = st.sidebar.slider("Along-track length (m)", 20.0, 500.0, 120.0)
    meta_from_file = None
else:
    st.sidebar.caption("Navigation & geometry auto-loaded from the XTF ping headers.")
    slant_fix = st.sidebar.checkbox("Apply slant-range correction", value=True)

if file_bytes:
    import tempfile

    suffix = Path(name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    if input_mode == "XTF sonar log":
        wf, nav, info = load_input(tmp_path)
        st.sidebar.success(
            f"Parsed: {info.get('sonar_name', b'?').decode()} | {info['n_pings']} pings | "
            f"waterfall {wf.shape[0]}x{wf.shape[1]}"
        )
        meta_from_file = nav_to_meta(nav, wf.shape)
        st.sidebar.write(
            f"Origin: {meta_from_file.lat0:.5f}, {meta_from_file.lon0:.5f} | "
            f"heading {meta_from_file.heading_deg:.0f} deg | alt {meta_from_file.altitude_m:.0f} m"
        )
        if slant_fix:
            wf = slant_range_correct(wf, meta_from_file.altitude_m, meta_from_file.range_m)
        img = wf
    else:
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        if stress:
            img = apply_sonar_synth(img)

    with st.spinner("Preprocessing..."):
        clean, gap = preprocess(img)

    detector = get_detector()
    detector.threshold = threshold
    with st.spinner("Running segmentation..."):
        if input_mode == "XTF sonar log":
            h, w = clean.shape
            resized = cv2.resize(clean, (512, 512), interpolation=cv2.INTER_AREA)
            prob_s, mask_s, dets = detector(resized, preprocessed=True)
            sx, sy = w / 512, h / 512
            for d in dets:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                d["bbox_xyxy"] = [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]
                d["centroid_px"] = [d["centroid_px"][0] * sx, d["centroid_px"][1] * sy]
            prob = cv2.resize(prob_s, (w, h))
            mask = (prob > threshold).astype(np.uint8)
        else:
            prob, mask, dets = detector(clean, preprocessed=True)
        dets = detector.extract_detections(prob, mask, min_area=min_area)
        scored = score_detections(clean, prob, dets, mask)

    if input_mode == "XTF sonar log":
        meta = meta_from_file
    else:
        meta = SonarMeta(
            lat0=lat0, lon0=lon0, heading_deg=heading,
            altitude_m=altitude, range_m=range_m, along_track_m=along_track,
        )
    mapper = PixelGeoMapper(clean.shape[0], clean.shape[1], meta)
    geod = mapper.geotag(scored)
    json_path, csv_path = save_report(geod, Path("hazard_report"))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Preprocessed")
        st.image((clean * 255).astype(np.uint8), use_container_width=True)
    with c2:
        st.subheader("Mask")
        st.image(mask * 255, use_container_width=True)
    with c3:
        st.subheader("Detections")
        overlay = cv2.cvtColor((clean * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for d in geod:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            color = (0, 0, 255) if d["likely_rock_or_shadow"] else (0, 255, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, f"{d['confidence']:.0f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader(f"{len(geod)} detections")
    st.caption(
        "Confidence is a heuristic composite score (0–100, uncalibrated). "
        "It is NOT a probability. Values below 35 are flagged as possible rock/shadow false positives."
    )
    if geod:
        rows = [
            {
                "id": d["id"],
                "heuristic_conf (0-100)": d["confidence"],
                "model_prob": round(d["mean_prob"], 3),
                "geo_score": d["geo_score"],
                "shadow_score": d["shadow_score"],
                "lat": d["lat"],
                "lon": d["lon"],
                "flag": "possible rock/shadow" if d["likely_rock_or_shadow"] else "",
            }
            for d in geod
        ]
        st.dataframe(rows, use_container_width=True)
        st.map([{"lat": d["lat"], "lon": d["lon"]} for d in geod if not d["likely_rock_or_shadow"]])

        cdl, cdr = st.columns(2)
        cdl.download_button("Download JSON report", json_path.read_bytes(), "hazard_report.json", "application/json")
        cdr.download_button("Download CSV report", csv_path.read_bytes(), "hazard_report.csv", "text/csv")
else:
    st.info("Upload a sonar image or select a sample to begin.")
