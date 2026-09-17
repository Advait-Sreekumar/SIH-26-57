import hashlib
import time
import datetime
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from categories import load_categories
from confidence import score_detections
from geotag import PixelGeoMapper, SonarMeta, save_report
from infer import SonarDetector
from preprocess import preprocess
from review_store import (
    delete_review,
    get_reviews_for_run,
    init_db,
    register_run,
    upsert_review,
)
from synthaug import apply_sonar_synth
from xtf_io import load_input, nav_to_meta, slant_range_correct


def make_run_id(file_bytes: bytes, image_name: str) -> str:
    digest = hashlib.sha1(file_bytes).hexdigest()[:10]
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    stem = Path(image_name).stem[:20]
    return f"{stem}_{ts}_{digest}"


@st.cache_resource
def get_db():
    return init_db()

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

with st.expander("About this model — scope, limitations & performance", expanded=False):
    st.markdown("""
**Detection scope:** Shipwrecks only. Trained on the AI4Shipwrecks dataset (Thunder Bay NMS,
28 sites, towfish side-scan sonar). Pipes, cylinders, and ghost nets are **not detected** —
no annotated sonar training data exists for those classes.

**Performance (test set, held-out sites never seen during training):**

| Split | IoU | Dice |
|-------|-----|------|
| Validation (in-distribution) | 0.713 | 0.833 |
| **Test (held-out sites)** | **0.427** | **0.598** |

Test IoU 0.427 is the headline metric. SOTA on comparable sonar benchmarks: 0.55–0.77.
No independent external sonar benchmark exists for this taxonomy.

**Inference latency** (PyTorch FP32, 10 images, CPU=i7-12700H, GPU=RTX 4050, CUDA 12.1):

| Backend | mean | min | max |
|---------|------|-----|-----|
| PyTorch FP32 / CPU | 4.29s | 1.53s | 6.23s |
| PyTorch FP32 / GPU (RTX 4050) | 0.37s | 0.13s | 0.54s |
| ONNX FP32 / CPU | 3.08s | 0.89s | 5.27s |
| ONNX INT8 / CPU† | 5.26s | 2.09s | 7.06s |

† INT8 is **slower** than FP32 on this CPU (dynamic-quant overhead). Max diff vs FP32 = 0.6497 WARN.
Spread tracks tile count (8–32 tiles per image → 1.5–6.2s range). Full results: `pipeline/benchmark_results.json`.
"""
)

_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB hard limit for this local demo

if file_bytes:
    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        st.error(
            f"File is {len(file_bytes) / 1024 / 1024:.1f} MB — exceeds the 200 MB upload limit. "
            "Reduce file size or split the survey log before uploading."
        )
        st.stop()

    import tempfile

    suffix = Path(name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    if input_mode == "XTF sonar log":
        try:
            wf, nav, info = load_input(tmp_path)
        except ValueError as exc:
            st.error(f"Could not parse XTF file: {exc}")
            st.stop()
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
    _t_infer_start = time.perf_counter()
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
    _t_infer_s = time.perf_counter() - _t_infer_start

    if input_mode == "XTF sonar log":
        meta = meta_from_file
    else:
        meta = SonarMeta(
            lat0=lat0, lon0=lon0, heading_deg=heading,
            altitude_m=altitude, range_m=range_m, along_track_m=along_track,
        )
    mapper = PixelGeoMapper(clean.shape[0], clean.shape[1], meta)
    geod = mapper.geotag(scored)
    if "run_id" not in st.session_state or st.session_state.get("last_file") != name:
        st.session_state["run_id"] = make_run_id(file_bytes, name)
        st.session_state["last_file"] = name
    run_id = st.session_state["run_id"]
    conn = get_db()
    register_run(conn, run_id, name, len(geod))
    reviews = get_reviews_for_run(conn, run_id)
    json_path, csv_path = save_report(geod, Path("hazard_report"), run_id=run_id, reviews=reviews)

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
    st.caption(f"Inference time: {_t_infer_s:.2f}s (preprocess + tile inference; PyTorch FP32 / CPU)")
    st.caption(
        "Confidence is a heuristic composite score (0–100, uncalibrated). "
        "It is NOT a probability. Values below 35 are flagged as possible rock/shadow false positives."
    )
    if geod:
        rows = [
            {
                "id": d["id"],
                "review_status": reviews.get(d["id"], {}).get("action", "unreviewed"),
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

        # ---- Review panel ----
        st.divider()
        st.subheader("Review Detections")
        st.caption(
            "Mark each detection as confirmed, rejected, uncertain, or annotated. "
            "Actions are saved immediately and persist across restarts."
        )

        categories = load_categories()
        cat_keys = list(categories.keys())
        cat_labels = [f"{k} — {v}" for k, v in categories.items()]

        def _action_idx(det_id):
            rev = reviews.get(det_id, {})
            action = rev.get("action")
            opts = ["confirm", "reject", "uncertain", "annotate"]
            return opts.index(action) if action in opts else None

        def _cat_idx(det_id):
            rev = reviews.get(det_id, {})
            cat = rev.get("category")
            return cat_keys.index(cat) if cat in cat_keys else 0

        for d in geod:
            det_id = d["id"]
            rev = reviews.get(det_id, {})
            action = rev.get("action")
            badge = f" [{action}]" if action else " [unreviewed]"
            flag_note = " (flagged: possible rock/shadow)" if d["likely_rock_or_shadow"] else ""
            label = f"Detection #{det_id}{badge}{flag_note}  conf={d['confidence']:.0f}"

            with st.expander(label, expanded=(action is None)):
                col_img, col_form = st.columns([1, 2])

                with col_img:
                    # bbox crop from preprocessed image
                    try:
                        x1, y1, x2, y2 = d["bbox_xyxy"]
                        pad = 10
                        crop = clean[
                            max(0, y1 - pad): min(clean.shape[0], y2 + pad),
                            max(0, x1 - pad): min(clean.shape[1], x2 + pad),
                        ]
                        st.image((crop * 255).astype('uint8'), caption="Sonar crop", use_container_width=True)
                    except Exception:
                        st.caption("(crop unavailable)")
                    st.metric("Sonar-heuristic confidence", f"{d['confidence']:.1f}")
                    st.metric("Model prob", f"{d['mean_prob']:.3f}")
                    st.caption(f"Lat {d['lat']:.6f} | Lon {d['lon']:.6f}")

                with col_form:
                    cur_action_idx = _action_idx(det_id)
                    action_opts = ["confirm", "reject", "uncertain", "annotate"]
                    radio_opts = action_opts + ["(no review)"]
                    radio_idx = cur_action_idx if cur_action_idx is not None else len(action_opts)
                    chosen_action = st.radio(
                        "Action",
                        radio_opts,
                        index=radio_idx,
                        key=f"action_{run_id}_{det_id}",
                        horizontal=True,
                    )
                    chosen_cat = st.selectbox(
                        "Category",
                        cat_labels,
                        index=_cat_idx(det_id),
                        key=f"cat_{run_id}_{det_id}",
                    )
                    chosen_note = st.text_area(
                        "Note (optional)",
                        value=rev.get("note") or "",
                        key=f"note_{run_id}_{det_id}",
                        max_chars=500,
                    )
                    b_save, b_clear = st.columns(2)
                    if b_save.button("Save", key=f"save_{run_id}_{det_id}"):
                        if chosen_action != "(no review)":
                            upsert_review(
                                conn, run_id, det_id,
                                chosen_action,
                                cat_keys[cat_labels.index(chosen_cat)],
                                chosen_note or None,
                            )
                            st.rerun()
                    if b_clear.button("Clear", key=f"clear_{run_id}_{det_id}"):
                        delete_review(conn, run_id, det_id)
                        st.rerun()
else:
    st.info("Upload a sonar image or select a sample to begin.")
