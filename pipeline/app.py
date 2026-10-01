import hashlib
import json
import time
import datetime
import io
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit_antd_components as sac

try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False
    folium = None
    st_folium = None

# Checkpoint management for deployment.
# The model checkpoint is a 97 MB .pth file excluded from the repo (.gitignore).
# Before importing infer.py, resolve the checkpoint from one of these sources:
#   1. SONAR_CKPT env var (explicit path) — takes precedence
#   2. SONAR_CKPT_URL env var (hosted URL) — downloads to a local path
#   3. Local file in the repo (AI4Shipwrecks/AI4Shipwrecks/runs/)
#   4. Common local fallback paths
if "SONAR_CKPT" not in os.environ:
    _REPO_ROOT = Path(__file__).parent.parent
    _default_ckpt = _REPO_ROOT / "AI4Shipwrecks" / "AI4Shipwrecks" / "runs" / "best_model_v1_iou0.71.pth"

    if _default_ckpt.exists():
        os.environ["SONAR_CKPT"] = str(_default_ckpt)
    else:
        _hosted_url = os.environ.get("SONAR_CKPT_URL")
        if _hosted_url:
            try:
                import requests

                _ckpt_filename = os.environ.get("SONAR_CKPT_FILENAME", "best_model_v1_iou0.71.pth")
                _local_path = _REPO_ROOT / _ckpt_filename

                if not _local_path.exists():
                    st.write(f"Downloading checkpoint from {_hosted_url.split('?')[0]}...")
                    _response = requests.get(_hosted_url, stream=True, timeout=60)
                    _response.raise_for_status()
                    _total = int(_response.headers.get("content-length", 0))
                    with open(_local_path, "wb") as _f:
                        for _chunk in _response.iter_content(chunk_size=8192):
                            if _chunk:
                                _f.write(_chunk)
                    st.success(f"Checkpoint downloaded to {_local_path}")
                os.environ["SONAR_CKPT"] = str(_local_path)
            except ImportError:
                st.warning("requests library not installed — install it or set SONAR_CKPT manually")
            except Exception as _e:
                st.error(f"Failed to download checkpoint from {_hosted_url}: {_e}")
                st.error("Set the SONAR_CKPT environment variable to a local path instead.")
        else:
            for _alt in (
                _REPO_ROOT / "best_model_v1_iou0.71.pth",
                Path.home() / "models" / "best_model_v1_iou0.71.pth",
            ):
                if _alt.exists():
                    os.environ["SONAR_CKPT"] = str(_alt)
                    break

from categories import load_categories
from confidence import score_detections
from geotag import PixelGeoMapper, SonarMeta, save_report
from geolocate import NavTrackGeoMapper, coordinate_units_plausible
from infer import SonarDetector
import mapviz
from preprocess import preprocess
from review_store import (
    delete_review,
    delete_sample_run,
    get_reviews_for_run,
    init_db,
    purge_old_sample_runs,
    register_run,
    upsert_review,
)
from synthaug import apply_sonar_synth
from xtf_io import load_input, nav_to_meta, slant_range_correct


def make_run_id(file_bytes: bytes, image_name: str) -> str:
    digest = hashlib.sha1(file_bytes).hexdigest()[:10]
    stem = Path(image_name).stem[:20]
    return f"{stem}_{digest}"


@st.cache_resource
def get_db():
    return init_db()


@st.cache_resource
def get_detector():
    return SonarDetector()


st.set_page_config(page_title="SonarEye", layout="wide", page_icon="\U0001f4e1")

# ---- Global CSS: dark navy/teal scientific instrumentation theme ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root {
    --bg:         #0d1117;
    --surface:    #151b23;
    --surface-2:  #1b222c;
    --border:     #232b36;
    --border-soft:#1d242e;
    --primary:    #2ea8a0;   /* teal — the one brand + action accent */
    --primary-hi: #3dc4bb;
    --primary-dim:#1f7e78;
    --teal:       #2ea8a0;
    --text:       #e6edf3;
    --muted:      #8b949e;
    --green:      #3fb950;
    --amber:      #d29922;
    --red:        #f85149;
    --blue:       #58a6ff;   /* reserved: "unreviewed" status only */
    --accent:     #58a6ff;
    --steel:      #7ea8c4;   /* machine/AI flag — cool, distinct from status hues */
    --mono: 'IBM Plex Mono','SF Mono','Consolas',monospace;
    --sans: 'IBM Plex Sans','Segoe UI',system-ui,-apple-system,sans-serif;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}
/* Reclaim Streamlit's oversized default block padding so the app renders
   cleanly at 100% browser zoom on a 1366x768 demo laptop. */
[data-testid="stMainBlockContainer"], .block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 3rem !important;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
.stMarkdown, .stText, p, li, span, label { color: var(--text) !important; font-family: var(--sans) !important; }
p, li { line-height: 1.55; }
/* Type scale — one family, distinct weight/size/colour per level. */
h1, h2, h3, h4 { color: var(--text) !important; font-family: var(--sans) !important; letter-spacing: -0.01em; }
h1 { font-size: 2.0rem !important; font-weight: 700 !important; }
h2 { font-size: 1.5rem !important; font-weight: 600 !important; margin-top: 0.4rem !important; padding-bottom: 5px; }
/* Focal "scan line": one short teal rule under each screen's title. */
[data-testid="stMainBlockContainer"] h2 { display: inline-block; border-bottom: 2px solid var(--primary); }
h3 { font-size: 1.12rem !important; font-weight: 600 !important; color: #cdd7e1 !important; margin-top: 0.5rem !important; }
h4 { font-size: 0.95rem !important; font-weight: 600 !important; color: var(--muted) !important; }
/* Buttons encode one idea: teal = act or "you are here"; quiet = everything else.
   Primary (CTAs + active nav item) is solid teal; secondary is a ghost control. */
.stButton > button {
    border-radius: 5px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    font-family: var(--sans) !important;
    transition: background-color .12s ease, border-color .12s ease, color .12s ease;
}
.stButton > button[kind="primary"] {
    background-color: var(--primary) !important;
    color: #06201e !important;
    border: 1px solid var(--primary) !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: var(--primary-hi) !important;
    border-color: var(--primary-hi) !important;
}
.stButton > button[kind="secondary"] {
    background-color: var(--surface-2) !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
}
.stButton > button[kind="secondary"]:hover {
    color: var(--primary-hi) !important;
    border-color: var(--primary-dim) !important;
}
/* Sidebar buttons read as nav rows: left-aligned, full width. */
[data-testid="stSidebar"] .stButton > button { text-align: left !important; justify-content: flex-start !important; }
/* Visible keyboard focus for accessibility. */
.stButton > button:focus-visible,
[data-baseweb="input"] input:focus-visible,
[data-baseweb="textarea"] textarea:focus-visible {
    outline: 2px solid var(--primary-hi) !important;
    outline-offset: 1px !important;
}
/* Metrics read like instrument readouts: mono value, quiet label, hairline frame.
   (Streamlit renamed the container testid metric-container -> stMetric; match both.) */
[data-testid="metric-container"], [data-testid="stMetric"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
}
[data-testid="metric-container"] label, [data-testid="stMetric"] label { color: var(--muted) !important; font-size: 0.74rem !important; letter-spacing: 0.02em; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.5rem !important; font-family: var(--mono) !important; font-weight: 500 !important; }
.stDataFrame { background-color: var(--surface) !important; border: 1px solid var(--border-soft) !important; border-radius: 6px !important; }
.stExpander { background-color: var(--surface) !important; border: 1px solid var(--border-soft) !important; border-radius: 6px !important; }
hr { border-color: var(--border) !important; }
.status-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
/* color !important beats the global `span{color:var(--text)!important}` rule so the
   confirmed/rejected/uncertain/unreviewed status coding survives the cascade. */
.badge-confirmed  { background: #12291a; color: #56d364 !important; border: 1px solid var(--green); }
.badge-rejected   { background: #2a1517; color: #ff7b72 !important; border: 1px solid var(--red); }
.badge-uncertain  { background: #2a2109; color: #e3b341 !important; border: 1px solid var(--amber); }
.badge-annotate   { background: #2a2109; color: #e3b341 !important; border: 1px solid var(--amber); }
.badge-unreviewed { background: #0d1d2e; color: #79c0ff !important; border: 1px solid var(--blue); }
/* Machine flag: cool steel, deliberately outside the human-decision status hues. */
.badge-ai         { background: #14202b; color: #9cc0d6 !important; border: 1px solid #3a5568; }
.nav-section-header {
    font-size: 0.66rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted-2, #6e7681) !important;
    padding: 16px 4px 6px 4px;
    margin: 0;
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 6px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.stat-card .label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
.stat-card .value { font-size: 1.85rem; font-weight: 600; color: var(--text); line-height: 1.15; font-family: var(--mono); margin-top: 4px; }
.stat-card .sub   { font-size: 0.78rem; color: var(--muted); margin-top: 3px; }
.info-box {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-left: 3px solid var(--primary);
    border-radius: 5px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.85rem;
    line-height: 1.55;
    color: var(--muted) !important;
}
.warn-box {
    background: #241b09;
    border: 1px solid var(--border-soft);
    border-left: 3px solid var(--amber);
    border-radius: 5px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.85rem;
    line-height: 1.55;
    color: #e3b341 !important;
}
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 9px 0;
    font-size: 0.85rem;
    border-bottom: 1px solid var(--border-soft);
}
.pipeline-step .dot-done    { width:9px; height:9px; border-radius:50%; background:var(--green); flex-shrink:0; }
.pipeline-step .dot-running { width:9px; height:9px; border-radius:50%; background:var(--amber); flex-shrink:0; }
.pipeline-step .dot-pending { width:9px; height:9px; border-radius:50%; background:var(--border); flex-shrink:0; }
.pipeline-step .step-label  { color: var(--text); flex-grow:1; }
.pipeline-step .step-count  { color: var(--muted); font-size:0.78rem; font-family:var(--mono); }
/* Landing page — the hero is the focal moment: a single instrument wordmark. */
.landing-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 65vh;
    gap: 1.4rem;
}
.sonareye-hero {
    font-size: 3.8rem;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: 0.02em;
    text-align: center;
    margin: 0;
    font-family: var(--mono);
}
.hero-sub {
    color: #8b949e;
    font-size: 1rem;
    text-align: center;
    margin: 0;
    letter-spacing: 0.01em;
    line-height: 1.5;
}
.hero-accent { color: #2ea8a0; text-shadow: 0 0 22px rgba(46,168,160,0.35); }
/* Hide Streamlit's built-in sidebar collapse/expand arrow button */
[data-testid="collapsedControl"] { display: none !important; }
/* Fix Streamlit file-uploader: the label prop already renders a heading above the
   dropzone, so hide the duplicate inner "Upload" button text that appears inside
   the drag-and-drop zone to avoid the visual "uploadUpload" stacking. */
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { display: none !important; }
[data-testid="stFileUploaderDropzone"] button span { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ---- Paths ----
_REPO_ROOT = Path(__file__).parent.parent
DATA_IMG = _REPO_ROOT / "AI4Shipwrecks" / "AI4Shipwrecks" / "test" / "images"
if not DATA_IMG.exists():
    import os as _os
    _env = _os.environ.get("SONAR_TEST_IMAGES")
    DATA_IMG = Path(_env) if _env else DATA_IMG

_SAMPLE_NAV_FILE = DATA_IMG / "sample_nav.json"
_SAMPLE_NAV: dict = json.loads(_SAMPLE_NAV_FILE.read_text(encoding="utf-8")) if _SAMPLE_NAV_FILE.exists() else {}


# ---- SSS device presets ----
# EdgeTech 2205: AI4Shipwrecks dataset device, 132 kHz, ~50 m/side typical operational
# EdgeTech 4125: edgetech.com — 400 kHz 150 m/side, 900 kHz 75 m/side
# Klein 3000: maritimepropulsion.com — 500 kHz 25-600 m/ch, 100 kHz 100-600 m/ch
_SSS_PRESETS = {
    "Custom / Manual":                    {"range_m": None,  "freq": None},
    "EdgeTech 2205 (132 kHz) — dataset":  {"range_m": 50.0,  "freq": "132 kHz"},
    "EdgeTech 4125 — 400 kHz mode":       {"range_m": 150.0, "freq": "400 kHz"},
    "EdgeTech 4125 — 900 kHz mode":       {"range_m": 75.0,  "freq": "900 kHz"},
    "Klein 3000 — 500 kHz mode":          {"range_m": 75.0,  "freq": "500 kHz"},
    "Klein 3000 — 100 kHz mode":          {"range_m": 300.0, "freq": "100 kHz"},
}


def _badge(status: str) -> str:
    cls = f"badge-{status.lower().replace(' ', '-')}"
    return f'<span class="status-badge {cls}">{status}</span>'


def _ai_badge() -> str:
    return '<span class="status-badge badge-ai">AI: Detected</span>'


def _purge_sample_run_if_needed(run_id, source) -> bool:
    """Hard-delete a sample run from SQLite. Never deletes upload-tagged runs."""
    if not run_id or source != "sample":
        return False
    return delete_sample_run(get_db(), run_id)


def _reset_survey_session() -> None:
    """Explicit start-over: drop the committed survey and in-session review cursor.

    Sample/demo runs are hard-deleted from SQLite (reviews + run row).
    User-uploaded survey reviews are left in place so they persist across refresh.
    """
    prev_id = st.session_state.get("run_id")
    prev_src = (st.session_state.get("survey_cfg") or {}).get("source")
    _purge_sample_run_if_needed(prev_id, prev_src)
    st.session_state["survey_started"] = False
    st.session_state["survey_cfg"] = None
    for k in (
        "run_id",
        "last_file",
        "_infer_results",
        "_infer_file_key",
        "_infer_params_key",
        "review_det_id",
        "selected_det",
        "detail_return",
        "_sr_done",
        "_sr_total",
        "_map_last_tip",
    ):
        st.session_state.pop(k, None)
    st.session_state["review_idx"] = 0
    st.session_state["review_complete"] = False
    st.session_state["cfg_sample"] = ""


def _commit_survey(cfg: dict) -> None:
    st.session_state["survey_cfg"] = cfg
    st.session_state["survey_started"] = True
    # New commit: reset the review cursor; detections belong to this file.
    st.session_state["review_idx"] = 0
    st.session_state["review_complete"] = False
    st.session_state.pop("review_det_id", None)
    st.session_state.pop("selected_det", None)
    if cfg.get("source") == "sample":
        # Wipe any leftover demo rows for this file hash before the new run registers.
        st.session_state["_sample_reset_pending"] = True


def _survey_inputs_ready(file_bytes, input_mode: str, selected_device: str, range_m) -> tuple[bool, list[str]]:
    missing = []
    if not file_bytes:
        missing.append("Load a sonar image or XTF log")
    if input_mode == "Image file":
        if not selected_device:
            missing.append("Select an SSS device preset (or Custom / Manual)")
        if range_m is None:
            missing.append("Set range per side (or pick an SSS preset)")
    return (len(missing) == 0, missing)


def _pdf_latin(text) -> str:
    """Helvetica core fonts are latin-1 only; em-dashes were crashing PDF export."""
    if text is None:
        return "-"
    s = str(text)
    for src, dst in (
        ("\u2014", "-"),
        ("\u2013", "-"),
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2022", "-"),
        ("\u00a0", " "),
    ):
        s = s.replace(src, dst)
    return s.encode("latin-1", "replace").decode("latin-1")


def build_mission_pdf(
    *,
    name: str,
    run_id: str,
    geod: list,
    reviews: dict,
    rc: dict,
    t_infer: float,
    clean,
    plans: dict | None = None,
) -> bytes:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def _ln_cell(w, h, txt, **kwargs):
        pdf.cell(w, h, _pdf_latin(txt), new_x=XPos.LMARGIN, new_y=YPos.NEXT, **kwargs)

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(13, 17, 23)
    _ln_cell(0, 14, "SonarEye - Mission Report", align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    _ln_cell(0, 6, "Side-scan sonar anomaly detection & operator verification platform", align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    _ln_cell(0, 7, "Survey Metadata")
    pdf.set_font("Helvetica", "", 9)
    generated_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    for label, val in [
        ("Survey file",   name),
        ("Run ID",        run_id),
        ("Generated",     generated_at),
        ("Anomalies",     str(len(geod))),
        ("Reviewed",      str(rc["reviewed"])),
        ("Confirmed",     str(rc["confirmed"])),
        ("Rejected",      str(rc["rejected"])),
        ("High-conf >=60", str(rc["high_conf"])),
        ("Inference",     f" {t_infer:.2f}s"),
    ]:
        pdf.cell(50, 6, _pdf_latin(f"{label}:"), border=0)
        _ln_cell(0, 6, val)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    _ln_cell(0, 7, "Detection Summary")
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(230, 237, 243)
    col_w = [10, 22, 28, 28, 38, 38]
    headers = ["ID", "Conf", "Lat", "Lon", "AI Classification", "Human Review"]
    for w, h in zip(col_w, headers):
        pdf.cell(w, 6, _pdf_latin(h), border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 7)
    for d in geod:
        rev = reviews.get(d["id"], {})
        classif = "Natural feature" if d["likely_rock_or_shadow"] else "Artificial anomaly"
        row_vals = [
            str(d["id"]),
            f"{d['confidence']:.0f}",
            f"{d['lat']:.5f}" if d.get("lat") is not None else "-",
            f"{d['lon']:.5f}" if d.get("lon") is not None else "-",
            classif,
            rev.get("action", "unreviewed"),
        ]
        for w, v in zip(col_w, row_vals):
            pdf.cell(w, 5, _pdf_latin(str(v)[:22]), border=1)
        pdf.ln()
    pdf.ln(4)

    # Location sketch (static; no live map object, no tile fetch)
    geo_pts = [d for d in geod if d.get("lat") is not None and d.get("lon") is not None]
    if geo_pts:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            lons = [d["lon"] for d in geo_pts]
            lats = [d["lat"] for d in geo_pts]
            colors = ["#3fb950" if reviews.get(d["id"], {}).get("action") == "confirm"
                      else "#f85149" if reviews.get(d["id"], {}).get("action") == "reject"
                      else "#58a6ff" for d in geo_pts]
            ax.scatter(lons, lats, c=colors, s=40, zorder=3)
            for d in geo_pts:
                ax.annotate(f"#{d['id']}", (d["lon"], d["lat"]), fontsize=7, xytext=(3, 3),
                            textcoords="offset points")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Detection locations")
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(useOffset=False, style="plain")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            pdf.set_font("Helvetica", "B", 11)
            _ln_cell(0, 7, "Map snapshot (detection locations)")
            pdf.image(buf, x=15, w=170)
            pdf.ln(4)
        except Exception:
            pdf.set_font("Helvetica", "I", 8)
            _ln_cell(0, 5, "(Map snapshot could not be rendered; table coordinates above remain authoritative.)")

    if clean is not None:
        pdf.set_font("Helvetica", "B", 11)
        _ln_cell(0, 7, "Sonar Survey Image with Detections")
        overlay_pdf = cv2.cvtColor((clean * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for d in geod:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            color_bgr = (0, 80, 248) if d["likely_rock_or_shadow"] else (63, 185, 80)
            cv2.rectangle(overlay_pdf, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(overlay_pdf, str(d["id"]), (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)
        # Downscale so the page cannot overflow
        max_px = 1400
        h0, w0 = overlay_pdf.shape[:2]
        if max(h0, w0) > max_px:
            scale = max_px / max(h0, w0)
            overlay_pdf = cv2.resize(overlay_pdf, (int(w0 * scale), int(h0 * scale)))
        _, img_enc = cv2.imencode(".png", overlay_pdf)
        img_buf = io.BytesIO(img_enc.tobytes())
        max_w = 170
        img_h = max_w * overlay_pdf.shape[0] / overlay_pdf.shape[1]
        img_h = min(img_h, 90)
        pdf.image(img_buf, x=15, w=max_w, h=img_h)

    # Cleanup route plans (spec PART 19) — only routes the operator explicitly
    # added; every value below is computed by the routing module, not fabricated.
    if plans:
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 30, 30)
        _ln_cell(0, 7, "Recommended Cleanup Routes (decision-support)")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(60, 60, 60)
        for pl in plans.values():
            _cur = "live forecast" if pl.get("current_is_live") else "DEMONSTRATION (synthetic)"
            lines = [
                f"Anomaly #{pl['det_id']} ({pl.get('object_class', 'anomaly')}) at "
                f"{pl['anomaly_lat']:.5f}, {pl['anomaly_lon']:.5f}",
                f"Departure port: {pl['port_name']} ({pl['port_type']}) at "
                f"{pl['port_lat']:.4f}, {pl['port_lon']:.4f}",
                f"Objective: {pl['objective']} | Distance: {pl['distance_km']} km | "
                f"ETA: {pl['eta_h']} h at {pl['vessel_speed_kn']:.1f} kn",
                f"Mean along-track current assist: {pl['mean_assist_ms']:+.3f} m/s",
                f"Current data: {_cur} | source: {pl['current_source']} | "
                f"timestamp: {pl['current_timestamp']}",
                f"{pl['route_class_label']} Generated {pl['generated_at']}.",
            ]
            for ln in lines:
                pdf.multi_cell(0, 4, _pdf_latin(ln))
            pdf.ln(2)

    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 30, 30)
    _ln_cell(0, 6, "System Information")
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4, _pdf_latin(
        "Model: U-Net + ResNet34 encoder (no scSE attention). "
        "Trained on AI4Shipwrecks (Thunder Bay NMSA, 286 images, 28 wreck sites, EdgeTech 2205 ~132 kHz). "
        "Internal test IoU: 0.427 / Dice: 0.598 (13 held-out sites). "
        "Confidence scores are heuristic composites - not calibrated probabilities. "
        "All detections require operator verification."
    ))
    raw = pdf.output()
    return bytes(raw)


# ================================================================
# LANDING PAGE
# ================================================================
if "app_entered" not in st.session_state:
    st.session_state["app_entered"] = False

if not st.session_state["app_entered"]:
    st.markdown("""
    <div class="landing-wrap">
        <p class="sonareye-hero">Sonar<span class="hero-accent">Eye</span></p>
        <p class="hero-sub">Side-scan sonar anomaly detection &amp; operator verification platform</p>
        <p class="hero-sub" style="font-size:0.82rem; margin-top:-0.8rem;">
            U-Net / ResNet34 &nbsp;&bull;&nbsp; AI4Shipwrecks dataset &nbsp;&bull;&nbsp;
            Test IoU 0.427 &nbsp;&bull;&nbsp; Interactive inference
        </p>
    </div>
    """, unsafe_allow_html=True)
    _, col_btn, _ = st.columns([3, 1, 3])
    with col_btn:
        if st.button("Launch SonarEye", use_container_width=True):
            st.session_state["app_entered"] = True
            st.rerun()
    st.stop()


# ================================================================
# SIDEBAR — persistent navigation + survey config
# ================================================================
with st.sidebar:
    st.markdown('<p class="sonareye-hero" style="font-size:1.4rem; margin-bottom:4px;">Sonar<span class="hero-accent">Eye</span></p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e; font-size:0.72rem; margin-top:0; margin-bottom:8px;">Anomaly Detection Platform</p>', unsafe_allow_html=True)
    st.divider()

    # Navigation
    st.markdown('<p class="nav-section-header">Navigation</p>', unsafe_allow_html=True)
    if "nav" not in st.session_state:
        st.session_state["nav"] = "overview"
    if "review_idx" not in st.session_state:
        st.session_state["review_idx"] = 0
    if "review_complete" not in st.session_state:
        st.session_state["review_complete"] = False
    if "survey_started" not in st.session_state:
        st.session_state["survey_started"] = False

    nav_items = [
        ("overview",    "Overview"),
        ("survey",      "Survey"),
        ("analysis",    "Analysis"),
        ("detections",  "Detections"),
        ("detail",      "Anomaly Detail"),
        ("map",         "Map"),
        ("plan",        "Plan Cleanup"),
        ("review",      "Summary"),
        ("report",      "Report"),
        ("system",      "System & Validation"),
    ]
    _cur_nav_sb = st.session_state.get("nav", "overview")
    for key, label in nav_items:
        # Sequential review is part of the detections/queue stage
        active = _cur_nav_sb == key or (key == "detections" and _cur_nav_sb == "seq_review")
        if st.button(label, key=f"nav_{key}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state["nav"] = key
            st.rerun()

    st.divider()

    # Survey configuration — keyed widgets so sidebar extra widgets cannot
    # shift implicit IDs and wipe the sample/file selection (classic Streamlit pitfall).
    st.markdown('<p class="nav-section-header">Survey Configuration</p>', unsafe_allow_html=True)

    meta_from_file = None
    pending_bytes = None
    pending_name = None
    pending_source = "upload"
    file_bytes = None
    name = None
    slant_fix = True
    selected_device = "Custom / Manual"
    range_m = 50.0
    # Demo survey origin: open water in the English Channel ~10 NM SSE of
    # Plymouth (in-sea, not on land). A land origin was the root cause of the
    # "anomaly appears on land" bug — any correct pixel->lat/lon transform
    # anchored to a land point yields land detections (spec PART 1).
    lat0, lon0, heading, altitude, along_track = 50.1000, -4.2000, 90.0, 15.0, 120.0
    threshold, min_area, stress = 0.5, 80, False
    input_mode = "Image file"

    _survey_locked = bool(st.session_state.get("survey_started")) and st.session_state.get("survey_cfg")

    if _survey_locked:
        cfg = st.session_state["survey_cfg"]
        input_mode = cfg["input_mode"]
        file_bytes = cfg["file_bytes"]
        name = cfg["name"]
        pending_source = cfg.get("source", "upload")
        threshold = cfg["threshold"]
        min_area = cfg["min_area"]
        stress = cfg["stress"]
        lat0 = cfg.get("lat0", lat0)
        lon0 = cfg.get("lon0", lon0)
        heading = cfg.get("heading", heading)
        altitude = cfg.get("altitude", altitude)
        selected_device = cfg.get("selected_device", selected_device)
        range_m = cfg.get("range_m", range_m)
        along_track = cfg.get("along_track", along_track)
        slant_fix = cfg.get("slant_fix", True)

        st.success(f"Locked: **{name}**")
        st.caption("File and analysis parameters are locked while this survey is in progress.")
        st.caption(f"Threshold {threshold:.2f} · min area {min_area} px")
        if input_mode == "Image file":
            st.caption(f"{selected_device} · range {range_m:.0f} m/side")
        if st.button("Start New Survey", key="sb_new_survey", use_container_width=True):
            _reset_survey_session()
            st.session_state["nav"] = "survey"
            st.rerun()
    else:
        input_mode = st.radio(
            "Input source",
            ["Image file", "XTF sonar log"],
            label_visibility="collapsed",
            key="cfg_input_mode",
        )
        if input_mode == "Image file":
            uploaded = st.file_uploader(
                "Upload sonar image", type=["png", "jpg", "jpeg"], key="cfg_upload_img"
            )
            _qp_sample = st.query_params.get("sample", "")
            _sample_opts = [""] + sorted(p.name for p in DATA_IMG.glob("*.png"))
            if _qp_sample in _sample_opts and not st.session_state.get("cfg_sample"):
                st.session_state["cfg_sample"] = _qp_sample
            sample = st.selectbox(
                "...or pick a sample",
                _sample_opts,
                key="cfg_sample",
            )
            if uploaded:
                pending_bytes, pending_name = uploaded.getvalue(), uploaded.name
                pending_source = "upload"
            elif sample:
                pending_bytes, pending_name = (DATA_IMG / sample).read_bytes(), sample
                pending_source = "sample"
        else:
            uploaded = st.file_uploader(
                "Upload .xtf sonar log", type=["xtf"], key="cfg_upload_xtf"
            )
            if uploaded:
                pending_bytes, pending_name = uploaded.getvalue(), uploaded.name
                pending_source = "upload"

        threshold = st.slider("Segmentation threshold", 0.3, 0.8, 0.5, 0.05, key="cfg_threshold")
        min_area = st.slider("Min detection area (px)", 20, 500, 80, 10, key="cfg_min_area")
        stress = st.checkbox(
            "Synthetic noise (stress test)",
            disabled=input_mode == "XTF sonar log",
            key="cfg_stress",
        )

        if input_mode == "Image file":
            lat0 = st.number_input("Origin lat", value=50.1000, format="%.6f", key="cfg_lat0")
            lon0 = st.number_input("Origin lon", value=-4.2000, format="%.6f", key="cfg_lon0")
            heading = st.slider("Heading (deg)", 0.0, 359.0, 90.0, key="cfg_heading")
            altitude = st.slider("Altitude (m)", 5.0, 50.0, 15.0, key="cfg_altitude")
            st.markdown("**SSS Device**")
            selected_device = st.selectbox(
                "Device preset",
                list(_SSS_PRESETS.keys()),
                key="cfg_device",
                label_visibility="collapsed",
            )
            preset = _SSS_PRESETS[selected_device]
            if preset["freq"]:
                st.caption(f"Frequency: {preset['freq']}")
            if preset["range_m"] is not None:
                range_m = preset["range_m"]
                st.caption(f"Range/side: {range_m:.0f} m (preset)")
            else:
                range_m = st.slider("Range per side (m)", 10.0, 600.0, 50.0, key="cfg_range_m")
            along_track = st.slider("Along-track length (m)", 20.0, 500.0, 120.0, key="cfg_along_track")
        else:
            st.caption("Navigation & geometry auto-loaded from XTF ping headers.")
            slant_fix = st.checkbox("Apply slant-range correction", value=True, key="cfg_slant_fix")

    # Sequential-review progress is AFTER keyed/locked config so it can never
    # shift file-selector widget identity.
    if _cur_nav_sb == "seq_review" and st.session_state.get("survey_started"):
        # Read fresh from DB so the counter always matches the main-page display
        # (session-state cache is one render behind when navigating).
        _sb_run_id = st.session_state.get("run_id")
        _sb_total = st.session_state.get("_sr_total", 0)
        if _sb_run_id and _sb_total > 0:
            _sb_reviews_live = get_reviews_for_run(get_db(), _sb_run_id)
            _sb_done = len(_sb_reviews_live)
        else:
            _sb_done = st.session_state.get("_sr_done", 0)
        if _sb_total > 0:
            st.progress(
                min(_sb_done / _sb_total, 1.0),
                text=f"Review: {_sb_done}/{_sb_total}",
            )


# ================================================================
# COMPUTATION — runs only for a *committed* survey (Start Survey).
# Widget/pending selection is NOT enough: that was causing analysis
# to bind to a selectbox that Streamlit could reset on nav reruns.
# Source of truth is st.session_state["survey_cfg"].
# ================================================================
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024
geod = []
reviews = {}
run_id = None
json_path = csv_path = None
clean = mask = prob = None
img = None
_t_infer_s = 0.0
_nav_pings = None  # per-ping nav list (XTF only), kept for future track-line overlay
_pipeline_log = []  # list of (label, count_str) for pipeline visualization

if file_bytes and st.session_state.get("survey_started"):
    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        st.error(
            f"File is {len(file_bytes) / 1024 / 1024:.1f} MB — exceeds the 200 MB upload limit."
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
        _nav_pings = nav
        _pipeline_log.append(("Ingest XTF", f"{info['n_pings']} pings, {info['n_channels']} channels"))
        st.sidebar.success(
            f"Parsed: {info.get('sonar_name', b'?').decode()} | {info['n_pings']} pings"
        )
        meta_from_file = nav_to_meta(nav, wf.shape)
        if slant_fix:
            wf = slant_range_correct(wf, meta_from_file.altitude_m, meta_from_file.range_m)
        img = wf
    else:
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        if stress:
            img = apply_sonar_synth(img)
        _pipeline_log.append(("Ingest image", f"{img.shape[1]}x{img.shape[0]} px"))

    clean, gap = preprocess(img)
    _pipeline_log.append(("Speckle reduction (Lee filter) + gain normalisation", ""))
    _pipeline_log.append(("Water-column crop", f"gap={gap}px"))

    # ── Session-state inference cache ──────────────────────────────────────────
    # run_id is stable for the same (file_bytes, name) pair.  Checking it here
    # means st.rerun() from a review save skips the ~4 s CPU inference pipeline
    # entirely — only the review UI section re-renders.
    _file_key = (name, len(file_bytes))
    _file_changed = st.session_state.get("_infer_file_key") != _file_key
    _params_key = (threshold, min_area, stress)
    _params_changed = st.session_state.get("_infer_params_key") != _params_key
    _need_infer = _file_changed or _params_changed or "_infer_results" not in st.session_state

    if _need_infer:
        with st.spinner("Running detection pipeline — segmentation, confidence scoring, geotagging…"):
            detector = get_detector()
            detector.threshold = threshold
            _t_infer_start = time.perf_counter()
            if input_mode == "XTF sonar log":
                h, w = clean.shape
                resized = cv2.resize(clean, (512, 512), interpolation=cv2.INTER_AREA)
                prob_s, mask_s, _ = detector(resized, preprocessed=True)
                sx, sy = w / 512, h / 512
                prob = cv2.resize(prob_s, (w, h))
                mask = (prob > threshold).astype(np.uint8)
                dets_raw = detector.extract_detections(prob, mask, min_area=min_area)
                for d in dets_raw:
                    x1, y1, x2, y2 = d["bbox_xyxy"]
                    d["bbox_xyxy"] = [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]
                    d["centroid_px"] = [d["centroid_px"][0] * sx, d["centroid_px"][1] * sy]
            else:
                prob, mask, _ = detector(clean, preprocessed=True)
                dets_raw = detector.extract_detections(prob, mask, min_area=min_area)

            n_tiles = max(1, (clean.shape[0] // 512) * (clean.shape[1] // 512))
            _pipeline_log.append(("Segmentation inference", f"{n_tiles} tile(s), threshold={threshold}"))
            scored = score_detections(clean, prob, dets_raw, mask)
            _t_infer_s = time.perf_counter() - _t_infer_start
            _pipeline_log.append(("Connected-component extraction", f"{len(dets_raw)} component(s), min_area={min_area}px"))
            _pipeline_log.append(("Confidence scoring", f"{len(scored)} detection(s) scored"))

        if input_mode == "XTF sonar log":
            # Per-ping nav-aware geolocation. Each detection is placed from the
            # navigation fix of the ping (image row) it sits on, plus a
            # perpendicular across-track offset — it follows the real track
            # instead of a single-origin straight-line guess (the old bug that
            # displaced detections ~half the track length onto shore).
            meta = meta_from_file
            _geo_warn = None
            units_ok, units_reason = coordinate_units_plausible(_nav_pings)
            if not units_ok:
                for d in scored:
                    d["lat"] = d["lon"] = None
                    d["geo_valid"] = False
                    d["geo_reason"] = units_reason
                geod = scored
                _geo_warn = ("Anomaly positions could not be reliably determined "
                             f"from the available navigation data ({units_reason}).")
            else:
                nav_mapper = NavTrackGeoMapper(_nav_pings, clean.shape[0], clean.shape[1])
                geod = nav_mapper.geotag(scored)
                _n_bad = sum(1 for d in geod if not d.get("geo_valid", True))
                if _n_bad:
                    _geo_warn = (f"{_n_bad} of {len(geod)} anomaly position(s) failed the "
                                 "off-track sanity check and are hidden from the map.")
            _pipeline_log.append(("Geotagging", f"per-ping nav fix over {len(_nav_pings)} pings"))
        else:
            meta = SonarMeta(
                lat0=lat0, lon0=lon0, heading_deg=heading,
                altitude_m=altitude, range_m=range_m, along_track_m=along_track,
            )
            mapper = PixelGeoMapper(clean.shape[0], clean.shape[1], meta)
            # geotag() sets geo_valid / geo_reason honestly (range + off-origin
            # sanity check). We TRUST that verdict here — no force-set to True.
            geod = mapper.geotag(scored)
            _geo_warn = None
            _n_bad = sum(1 for d in geod if not d.get("geo_valid", True))
            if _n_bad:
                _geo_warn = ("Anomaly position could not be reliably determined "
                             "from the available navigation data "
                             f"({_n_bad} of {len(geod)} detection(s) failed the "
                             "coordinate sanity check and are hidden from the map).")
            # Additional water advisory ONLY — a flag, never a coordinate move
            # (spec PART 4: do not cheat the map). Coastline data is offline and
            # fail-open, so absence of data never invalidates a fix.
            try:
                from routing.coastline import coastline_available, is_land
                if coastline_available():
                    _n_land = sum(
                        1 for d in geod
                        if d.get("geo_valid") and d.get("lat") is not None
                        and is_land(d["lat"], d["lon"])
                    )
                    if _n_land:
                        _adv = (f"Advisory: {_n_land} valid fix(es) fall on land per the "
                                "coastline layer — verify the survey origin/heading. "
                                "Coordinates are shown as computed and NOT relocated.")
                        _geo_warn = (_geo_warn + " " + _adv) if _geo_warn else _adv
            except Exception:
                pass  # coastline is a best-effort advisory; never block on it
            _pipeline_log.append(("Geotagging", f"origin ({meta.lat0:.4f}, {meta.lon0:.4f})"))

        st.session_state["_geo_warn"] = _geo_warn
        # Store in session state so review-save reruns are instant
        st.session_state["_infer_results"] = (prob, mask, geod, meta, _t_infer_s)
        st.session_state["_infer_file_key"] = _file_key
        st.session_state["_infer_params_key"] = _params_key
    else:
        # Cache hit: restore results without re-running inference
        prob, mask, geod, meta, _t_infer_s = st.session_state["_infer_results"]
        n_tiles = max(1, (clean.shape[0] // 512) * (clean.shape[1] // 512))
        _pipeline_log.append(("Segmentation inference", f"{n_tiles} tile(s) [cached]"))
        _pipeline_log.append(("Confidence scoring", f"{len(geod)} detection(s) [cached]"))
        _pipeline_log.append(("Geotagging", f"origin ({meta.lat0:.4f}, {meta.lon0:.4f}) [cached]"))

    _src = (st.session_state.get("survey_cfg") or {}).get("source", "upload")
    _rid = make_run_id(file_bytes, name)
    if _src == "sample":
        # Distinct from an upload of the same bytes so demo reset cannot touch user data.
        _rid = "s_" + _rid
    if st.session_state.pop("_sample_reset_pending", False) and _src == "sample":
        delete_sample_run(get_db(), _rid)
    if st.session_state.get("last_file") != name or "run_id" not in st.session_state:
        st.session_state["run_id"] = _rid
        st.session_state["last_file"] = name
        # Do not reset review_idx here — that used to fire whenever the
        # sidebar sample widget flickered empty-then-back on a nav rerun.
    run_id = st.session_state["run_id"]
    conn = get_db()
    if not st.session_state.get("_purged_old_samples"):
        purge_old_sample_runs(conn, keep_run_id=run_id)
        st.session_state["_purged_old_samples"] = True
    register_run(conn, run_id, name, len(geod), source=_src)
    reviews = get_reviews_for_run(conn, run_id)  # always fresh from DB
    json_path, csv_path = save_report(geod, Path("hazard_report"), run_id=run_id, reviews=reviews)
    _pipeline_log.append(("Report assembly", "JSON + CSV written"))


# ================================================================
# HELPERS (used across multiple sections)
# ================================================================

def _review_counts():
    n_confirmed = sum(1 for r in reviews.values() if r.get("action") == "confirm")
    n_rejected  = sum(1 for r in reviews.values() if r.get("action") == "reject")
    n_uncertain = sum(1 for r in reviews.values() if r.get("action") in ("uncertain", "annotate"))
    n_reviewed  = len(reviews)
    n_unreviewed = len(geod) - n_reviewed
    n_high_conf = sum(1 for d in geod if d["confidence"] >= 60)  # display bucket — arbitrary round threshold, not empirically validated
    return dict(
        confirmed=n_confirmed, rejected=n_rejected, uncertain=n_uncertain,
        reviewed=n_reviewed, unreviewed=n_unreviewed, high_conf=n_high_conf,
    )


def _signal_quality_heuristic(img: np.ndarray) -> tuple[str, float]:
    """Heuristic signal quality from image variance/contrast. Returns (label, score 0-1)."""
    std = float(np.std(img))
    # Empirically: clean sonar tiles have std ~0.10-0.20; very low = flat/featureless
    score = float(np.clip((std - 0.02) / 0.18, 0.0, 1.0))
    if score >= 0.7:
        label = "Good"
    elif score >= 0.4:
        label = "Moderate"
    else:
        label = "Low"
    return label, score


# ================================================================
# MAIN CONTENT — section routing
# ================================================================
cur_nav = st.session_state.get("nav", "overview")

# ── Mission progress stepper ────────────────────────────────────────────────
# Horizontal breadcrumb shown when a file is loaded. Tells the operator
# where they are in the mission flow without cluttering the sidebar.
_MISSION_STAGES = [
    ("survey",     "Survey"),
    ("analysis",   "Analysis"),
    ("detections", "Queue"),
    ("seq_review", "Review"),
    ("map",        "Map"),
    ("review",     "Summary"),
    ("report",     "Report"),
]
_STAGE_INDEX = {
    "survey": 0, "analysis": 1, "detections": 2,
    "seq_review": 3, "map": 4, "detail": 4, "review": 5, "report": 6,
}
if file_bytes and cur_nav in _STAGE_INDEX:
    _csi = _STAGE_INDEX[cur_nav]
    _parts = []
    for _si, (_sid, _slabel) in enumerate(_MISSION_STAGES):
        if _si < _csi:
            _parts.append(
                f"<span style='color:#3fb950;font-size:0.76rem;'>✔ {_slabel}</span>"
            )
        elif _si == _csi:
            _parts.append(
                f"<span style='background:#1d6fa4;color:#e6edf3;"
                f"padding:2px 9px;border-radius:4px;"
                f"font-size:0.76rem;font-weight:700;'>{_slabel}</span>"
            )
        else:
            _parts.append(
                f"<span style='color:#484f58;font-size:0.76rem;'>{_slabel}</span>"
            )
    st.markdown(
        "<div style='margin-bottom:4px;'>" +
        " <span style='color:#484f58;'>›</span> ".join(_parts) +
        "</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------
# OVERVIEW
# ----------------------------------------------------------------
if cur_nav == "overview":
    st.markdown("## Mission Overview")
    if not st.session_state.get("survey_started") or not file_bytes:
        st.markdown('<div class="info-box">No survey in progress. Load a sonar image or XTF log on the <b>Survey</b> page, configure geotag parameters, then click <b>Start Survey</b>.</div>', unsafe_allow_html=True)
        st.markdown("")
        col_start, _ = st.columns([1, 3])
        if col_start.button("Start New Survey", key="ov_start_new", use_container_width=True, type="primary"):
            st.session_state["nav"] = "survey"
            st.rerun()
        st.markdown("")
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, sub in [
            (c1, "Anomalies",     "—", "AI-flagged candidates"),
            (c2, "High-confidence","—", "Score ≥ 60"),
            (c3, "Needs review",  "—", "Unreviewed"),
            (c4, "Confirmed",     "—", "Operator-verified"),
        ]:
            col.markdown(f'<div class="stat-card"><div class="label">{label}</div><div class="value">{val}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)
    else:
        rc = _review_counts()
        st.markdown(f"**Survey:** `{name}`  &nbsp;|&nbsp;  Run ID: `{run_id}`  &nbsp;|&nbsp;  Inference: {_t_infer_s:.2f}s")
        st.markdown("")
        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (c1, "Anomalies",      len(geod),          "AI-flagged candidates"),
            (c2, "High-confidence", rc["high_conf"],    "Score ≥ 60"),
            (c3, "Needs review",   rc["unreviewed"],    "Unreviewed"),
            (c4, "Confirmed",      rc["confirmed"],     "Operator-verified"),
        ]
        for col, label, val, sub in cards:
            col.markdown(f'<div class="stat-card"><div class="label">{label}</div><div class="value">{val}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)
        st.markdown("")
        r1, r2 = st.columns(2)
        r1.metric("Rejected",  rc["rejected"])
        r2.metric("Uncertain / Annotated", rc["uncertain"])
        st.markdown("")
        col_a, col_b, col_c = st.columns([1, 1, 1])
        if col_a.button("Go to Detections", key="ov_to_det", use_container_width=True, type="primary"):
            st.session_state["nav"] = "detections"
            st.rerun()
        if col_b.button("Go to Map", key="ov_to_map", use_container_width=True):
            st.session_state["nav"] = "map"
            st.rerun()
        if col_c.button("Start New Survey", key="ov_restart", use_container_width=True):
            _reset_survey_session()
            st.session_state["nav"] = "survey"
            st.rerun()


# ----------------------------------------------------------------
# SURVEY
# ----------------------------------------------------------------
elif cur_nav == "survey":
    st.markdown("## Survey Configuration")
    _started = bool(st.session_state.get("survey_started") and file_bytes)
    _preview_bytes = file_bytes if _started else pending_bytes
    _preview_name = name if _started else pending_name
    _ready, _missing = _survey_inputs_ready(
        _preview_bytes, input_mode, selected_device, range_m,
    )

    if not _preview_bytes:
        st.markdown(
            '<div class="info-box">Use the sidebar to load a sonar image or XTF log. '
            "Configure geotag parameters and an SSS device preset, then click <b>Start Survey</b>.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("""
**Dataset context:** AI4Shipwrecks — Thunder Bay NMSA, 28 wreck sites, Iver3 AUV, EdgeTech 2205 SSS (~132 kHz). 286 images, CC-BY-4.0.

**Model scope:** Trained on shipwreck targets. The model flags candidates that resemble the acoustic signature of man-made debris at this frequency range. All detections require operator verification.

**Confidence scores** are heuristic composites (0–100) derived from model probability, geometric shape features, and shadow contrast — not calibrated probabilities.
        """)
    else:
        if _started:
            st.success(
                f"Survey in progress: **{_preview_name}**  — {len(geod)} anomaly(ies) in the current run."
            )
        else:
            st.success(f"Ready to start: **{_preview_name}** — file loaded, not yet committed.")

        st.markdown("### Survey Quality Snapshot")
        st.markdown(
            '<div class="info-box"><b>Note:</b> Quality indicators below are heuristic estimates '
            "derived from image statistics, not from calibrated sensor data.</div>",
            unsafe_allow_html=True,
        )

        q1, q2, q3 = st.columns(3)
        q1.metric("Data read", "Complete", help="Image decoded without errors")
        if input_mode == "XTF sonar log":
            q2.metric("Navigation metadata", "From XTF", help="Lat/lon/heading extracted from ping headers")
        else:
            q2.metric("Navigation metadata", "Manual", help="Geotag parameters entered manually")

        _snap_img = None
        if input_mode == "Image file" and _preview_bytes:
            _snap_img = cv2.imdecode(
                np.frombuffer(_preview_bytes, np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if _snap_img is not None:
                _snap_img = _snap_img.astype(np.float32) / 255.0
        if _started and clean is not None:
            sig_label, sig_score = _signal_quality_heuristic(clean)
        elif _snap_img is not None:
            sig_label, sig_score = _signal_quality_heuristic(_snap_img)
        else:
            sig_label, sig_score = "—", 1.0
        q3.metric(
            "Signal quality (heuristic)",
            sig_label,
            help="Variance-based heuristic. Not a calibrated SNR measurement.",
        )
        if sig_score < 0.4:
            st.markdown(
                '<div class="warn-box">Low signal variance detected. Image may be featureless or over-normalised. Detection quality may be reduced.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    if _missing:
        st.markdown(
            "<div class='warn-box'><b>Still needed:</b> " + "; ".join(_missing) + "</div>",
            unsafe_allow_html=True,
        )

    col_go, _ = st.columns([1, 3])
    if _started:
        if col_go.button(
            "Continue to Analysis",
            key="survey_continue",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["nav"] = "analysis"
            st.rerun()
    else:
        if col_go.button(
            "Start Survey",
            key="survey_start",
            type="primary",
            use_container_width=True,
            disabled=not _ready,
        ):
            _commit_survey({
                "input_mode": input_mode,
                "file_bytes": pending_bytes,
                "name": pending_name,
                "threshold": threshold,
                "min_area": min_area,
                "stress": stress,
                "lat0": lat0,
                "lon0": lon0,
                "heading": heading,
                "altitude": altitude,
                "selected_device": selected_device,
                "range_m": range_m,
                "along_track": along_track,
                "slant_fix": slant_fix,
                "source": pending_source,
            })
            st.session_state["nav"] = "analysis"
            st.rerun()


# ----------------------------------------------------------------
# ANALYSIS
# ----------------------------------------------------------------
elif cur_nav == "analysis":
    st.markdown("## Analysis")
    if not file_bytes:
        st.warning("No survey loaded — go to Survey first.")
        if st.button("Go to Survey", key="an_to_survey", type="primary"):
            st.session_state["nav"] = "survey"
            st.rerun()
    else:
        st.caption(f"{name}  |  {len(geod)} anomaly(ies)  |  Inference: {_t_infer_s:.2f}s")

        # Phase 4 — pipeline checklist
        st.markdown("### Processing Pipeline")
        for label, count in _pipeline_log:
            st.markdown(
                f'<div class="pipeline-step"><div class="dot-done"></div><div class="step-label">{label}</div><div class="step-count">{count}</div></div>',
                unsafe_allow_html=True
            )
        st.markdown("")

        st.markdown("### Sonar Imagery")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Preprocessed")
            st.image((clean * 255).astype(np.uint8), use_container_width=True)
        with c2:
            st.caption("Segmentation mask")
            st.image(mask * 255, use_container_width=True)
        with c3:
            st.caption("Detections overlay")
            overlay = cv2.cvtColor((clean * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for d in geod:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                color = (0, 80, 248) if d["likely_rock_or_shadow"] else (63, 185, 80)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, f"#{d['id']}  {d['confidence']:.0f}",
                            (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("")
        col_q, _ = st.columns([1, 3])
        if col_q.button("Go to Detections", key="an_to_det", use_container_width=True, type="primary"):
            st.session_state["nav"] = "detections"
            st.rerun()


# ----------------------------------------------------------------
# DETECTIONS QUEUE (Phase 5)
# ----------------------------------------------------------------
elif cur_nav == "detections":
    st.markdown("## Detection Queue")

    if not file_bytes:
        st.warning("No survey loaded — start a survey first.")
        if st.button("Go to Survey", key="dq_to_survey", type="primary"):
            st.session_state["nav"] = "survey"
            st.rerun()
    elif len(geod) == 0:
        # Phase 5 empty-state fix
        st.markdown('<div class="info-box">No anomalies detected in this survey.</div>', unsafe_allow_html=True)
        st.markdown("")
        e1, e2, e3 = st.columns(3)
        e1.metric("Tiles processed", max(1, (clean.shape[0] // 512) * (clean.shape[1] // 512)))
        e2.metric("Inference time", f"{_t_infer_s:.2f}s")
        e3.metric("Detections", 0)
        st.markdown("")
        st.markdown('<div class="info-box">This may indicate a feature-poor image, a very high segmentation threshold, or a survey area with no anomalous objects above the minimum detection area. Analysis parameters are locked for this survey; use Start New Survey to reconfigure.</div>', unsafe_allow_html=True)
        if st.button("Back to Analysis", key="dq_empty_an", type="primary"):
            st.session_state["nav"] = "analysis"
            st.rerun()
    else:
        rc = _review_counts()

        # ── Primary CTA: Review in Sequence ─────────────────────────────────
        if rc["unreviewed"] > 0:
            _cta_c, _info_c = st.columns([1, 3])
            with _cta_c:
                if st.button(
                    f"Review in Sequence  ({rc['unreviewed']} remaining)",
                    key="start_seq_review",
                    type="primary",
                    use_container_width=True,
                ):
                    # Jump to first unreviewed detection
                    _first_unrev = next(
                        (i for i, d in enumerate(geod)
                         if d["id"] not in reviews),
                        0,
                    )
                    st.session_state["review_idx"] = _first_unrev
                    st.session_state["review_complete"] = False
                    st.session_state["nav"] = "seq_review"
                    st.rerun()
            with _info_c:
                st.markdown(
                    "<span style='color:#8b949e;font-size:0.82rem;'>"
                    "Opens detections one at a time — each decision saves "
                    "and advances automatically to the next unreviewed."
                    "</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.success(
                f"All {len(geod)} detections reviewed — "
                f"{rc['confirmed']} confirmed · {rc['rejected']} rejected · "
                f"{rc['uncertain']} uncertain."
            )
            if st.button("View on Map", key="queue_to_review", type="primary"):
                st.session_state["nav"] = "map"
                st.rerun()

        st.divider()

        # Phase 7 — AI vs Human aggregate counts
        st.markdown(
            f"**AI:** {_ai_badge()} &nbsp; {len(geod)} anomaly(ies) &nbsp;&nbsp; "
            f"High-confidence: **{rc['high_conf']}** &nbsp;&nbsp; "
            f"Needs review: **{rc['unreviewed']}** &nbsp;&nbsp; "
            f"Confirmed: **{rc['confirmed']}** &nbsp;&nbsp; "
            f"Rejected: **{rc['rejected']}**",
            unsafe_allow_html=True
        )
        st.markdown("")

        # Sort controls
        sort_col, _ = st.columns([2, 4])
        sort_by = sort_col.selectbox(
            "Sort by",
            ["Confidence (high first)", "Review status", "Review priority (closest to threshold)"],
            label_visibility="collapsed",
            key="dq_sort",
        )

        def _sort_key(d):
            if sort_by == "Confidence (high first)":
                return -d["confidence"]
            elif sort_by == "Review status":
                order = {"unreviewed": 0, "uncertain": 1, "annotate": 2, "confirm": 3, "reject": 4}
                return order.get(reviews.get(d["id"], {}).get("action", "unreviewed"), 0)
            else:  # review priority
                # Phase 11: closest to likely_rock_or_shadow threshold (35.0) = most ambiguous
                return abs(d["confidence"] - 35.0)

        sorted_geod = sorted(geod, key=_sort_key)

        # Detection list
        for d in sorted_geod:
            det_id = d["id"]
            rev = reviews.get(det_id, {})
            action = rev.get("action", "unreviewed")
            flag_str = " — possible natural feature (rock/shadow)" if d["likely_rock_or_shadow"] else ""

            col_info, col_conf, col_status, col_action = st.columns([3, 1, 1, 1])
            col_info.markdown(
                f"{_ai_badge()} &nbsp; **#{det_id}** &nbsp; "
                f"Artificial anomaly (shipwreck-class model){flag_str}",
                unsafe_allow_html=True
            )
            col_conf.metric("Confidence", f"{d['confidence']:.0f}", label_visibility="collapsed")
            col_status.markdown(_badge(action), unsafe_allow_html=True)
            if col_action.button("Detail", key=f"det_open_{det_id}"):
                st.session_state["selected_det"] = det_id
                st.session_state["nav"] = "detail"
                st.rerun()
            st.markdown('<hr style="margin:4px 0; border-color:#21262d;">', unsafe_allow_html=True)


# ----------------------------------------------------------------
# ANOMALY DETAIL (Phase 6)
# ----------------------------------------------------------------
elif cur_nav == "detail":
    det_id = st.session_state.get("selected_det")
    if not file_bytes or not geod or det_id is None:
        st.warning("No detection selected. Go to Detections and click Detail on a detection.")
        if st.button("Go to Detections", key="det_none_to_q", type="primary"):
            st.session_state["nav"] = "detections"
            st.rerun()
    else:
        d = next((x for x in geod if x["id"] == det_id), None)
        if d is None:
            st.warning(f"Detection #{det_id} not found in current survey.")
        else:
            rev = reviews.get(det_id, {})
            action = rev.get("action", "unreviewed")
            flag_str = "Possible natural feature (rock/shadow/seabed formation)" if d["likely_rock_or_shadow"] else "Artificial anomaly (shipwreck-class model)"

            st.markdown(f"## Anomaly Detail — #{det_id}")
            st.markdown(
                f"{_ai_badge()} &nbsp; Human: {_badge(action)} &nbsp;&nbsp; "
                f"**{flag_str}**",
                unsafe_allow_html=True
            )
            st.markdown('<div class="warn-box">This is an AI-generated candidate and requires operator verification.</div>', unsafe_allow_html=True)

            # Visual evidence chain
            x1, y1, x2, y2 = d["bbox_xyxy"]
            pad = 20
            sy0, sy1 = max(0, y1 - pad), min(clean.shape[0], y2 + pad)
            sx0, sx1 = max(0, x1 - pad), min(clean.shape[1], x2 + pad)

            orig_crop  = img[sy0:sy1, sx0:sx1] if "img" in locals() else clean[sy0:sy1, sx0:sx1]
            clean_crop = clean[sy0:sy1, sx0:sx1]
            mask_crop  = mask[sy0:sy1, sx0:sx1]
            prob_crop  = prob[sy0:sy1, sx0:sx1] if prob is not None else np.zeros_like(clean_crop)

            # Shadow region crop
            comp = np.zeros(clean.shape, bool)
            comp[y1:y2, x1:x2] = mask[y1:y2, x1:x2].astype(bool)
            shadow_vis = np.zeros((*clean.shape, 3), np.uint8)
            shadow_vis[comp] = [240, 180, 0]  # amber = target
            ring = np.zeros(clean.shape, np.uint8)
            search = 40
            rx0, rx1 = max(0, x1 - search), min(clean.shape[1], x2 + search)
            ry0, ry1 = max(0, y1 - search), min(clean.shape[0], y2 + search)
            ring[ry0:ry1, rx0:rx1] = 1
            ring[y1:y2, x1:x2] = 0
            shadow_vis[ring.astype(bool) & ~comp] = [46, 160, 67]  # green = search ring
            shadow_crop = shadow_vis[sy0:sy1, sx0:sx1]

            # Final overlay crop
            overlay_full = cv2.cvtColor((clean * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            color = (0, 80, 248) if d["likely_rock_or_shadow"] else (63, 185, 80)
            cv2.rectangle(overlay_full, (x1, y1), (x2, y2), color, 2)
            overlay_crop = overlay_full[sy0:sy1, sx0:sx1]

            ec1, ec2, ec3, ec4, ec5 = st.columns(5)
            ec1.caption("Original")
            ec1.image((orig_crop * 255).astype(np.uint8), use_container_width=True)
            ec2.caption("Enhanced (preprocessed)")
            ec2.image((clean_crop * 255).astype(np.uint8), use_container_width=True)
            ec3.caption("Segmentation mask")
            ec3.image(mask_crop * 255, use_container_width=True)
            ec4.caption("Shadow search region")
            ec4.image(shadow_crop, use_container_width=True)
            ec5.caption("Final detection")
            ec5.image(cv2.cvtColor(overlay_crop, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("")

            # Confidence breakdown (Phase 6)
            def _strength(v):
                # Display-only bands; not derived from data
                if v >= 0.65: return "Strong"
                elif v >= 0.35: return "Moderate"
                else: return "Weak"

            model_score = 0.5 * d["mean_prob"] + 0.5 * d.get("peak_prob", d["mean_prob"])
            geo_score   = d["geo_score"]
            sh_score    = d["shadow_score"]
            geo_parts   = d.get("geo_parts", {})

            st.markdown(f"### Composite Evidence Score: {d['confidence']:.0f} / 100")
            st.markdown('<div class="info-box">Score is a heuristic composite — not a calibrated probability. Formula: 60% model probability + 40% × (70% geometry + 30% shadow contrast). Weights are hand-tuned.</div>', unsafe_allow_html=True)

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Model probability",  f"{model_score:.3f}", help=f"Strength: {_strength(model_score)}")
            sc2.metric("Geometric score",     f"{geo_score:.3f}",   help=f"Strength: {_strength(geo_score)} — aspect, solidity, edge sharpness")
            sc3.metric("Shadow contrast",     f"{sh_score:.3f}",    help=f"Strength: {_strength(sh_score)} — target vs surrounding ring")

            if geo_parts:
                with st.expander("Geometric sub-scores"):
                    gp1, gp2, gp3 = st.columns(3)
                    gp1.metric("Aspect score",    f"{geo_parts.get('aspect', 0):.3f}")
                    gp2.metric("Solidity score",  f"{geo_parts.get('solidity', 0):.3f}")
                    gp3.metric("Edge sharpness",  f"{geo_parts.get('edge', 0):.3f}")

            if d.get("lat") is not None and d.get("lon") is not None:
                st.markdown(f"**Lat:** {d['lat']:.6f}  |  **Lon:** {d['lon']:.6f}  |  Flagged: {'Yes (likely natural)' if d['likely_rock_or_shadow'] else 'No'}")
            else:
                st.markdown(
                    "**Position:** _could not be reliably determined_  |  "
                    f"Flagged: {'Yes (likely natural)' if d['likely_rock_or_shadow'] else 'No'}"
                )
                st.caption(f"Geolocation sanity check: {d.get('geo_reason') or 'coordinate rejected'}")

            st.markdown("")

            # Review controls (from Phase 5)
            st.markdown("### Verification")
            categories = load_categories()
            cat_keys = list(categories.keys())
            cat_labels = [f"{k} — {v}" for k, v in categories.items()]

            cur_cat = rev.get("category")
            cat_idx = cat_keys.index(cur_cat) if cur_cat in cat_keys else 0
            action_opts = ["confirm", "reject", "uncertain", "annotate"]
            radio_opts = action_opts + ["(no review)"]
            cur_act_idx = action_opts.index(action) if action in action_opts else len(action_opts)

            col_rv, col_form = st.columns([1, 2])
            with col_form:
                chosen_action = st.radio(
                    "Action", radio_opts, index=cur_act_idx,
                    key=f"det_action_{run_id}_{det_id}", horizontal=True
                )
                chosen_cat = st.selectbox(
                    "Category", cat_labels, index=cat_idx,
                    key=f"det_cat_{run_id}_{det_id}"
                )
                chosen_note = st.text_area(
                    "Note (optional)", value=rev.get("note") or "",
                    key=f"det_note_{run_id}_{det_id}", max_chars=500
                )
                b_save, b_clear = st.columns(2)
                if b_save.button("Save", key=f"det_save_{run_id}_{det_id}"):
                    if chosen_action != "(no review)":
                        conn = get_db()
                        upsert_review(
                            conn, run_id, det_id, chosen_action,
                            cat_keys[cat_labels.index(chosen_cat)],
                            chosen_note or None,
                        )
                        st.rerun()
                if b_clear.button("Clear", key=f"det_clear_{run_id}_{det_id}"):
                    conn = get_db()
                    delete_review(conn, run_id, det_id)
                    st.rerun()

            # --- PART 2/24: respond to a CONFIRMED anomaly ------------------
            # The cleanup planner is only offered once an operator has confirmed
            # the anomaly AND it has a reliable position — we never plan a route
            # to an unvalidated / null coordinate.
            if action == "confirm":
                if d.get("lat") is not None and d.get("lon") is not None:
                    st.markdown("### Respond")
                    st.caption("Anomaly confirmed. Plan a current-aware cleanup/inspection "
                               "route from a real maritime port.")
                    if st.button("PLAN CLEANUP", key=f"det_plan_{run_id}_{det_id}",
                                 type="primary"):
                        st.session_state["plan_target"] = {
                            "det_id": det_id,
                            "lat": float(d["lat"]),
                            "lon": float(d["lon"]),
                            "object_class": d.get("object_class", "anomaly"),
                        }
                        st.session_state["nav"] = "plan"
                        st.rerun()
                else:
                    st.info("Cleanup planning is unavailable — this confirmed anomaly has no "
                            "reliable position (geolocation sanity check failed).")

            _ret = st.session_state.get("detail_return", "detections")
            _ret_label = {"review": "Summary", "map": "Map"}.get(_ret, "Detections")
            col_back, _ = st.columns([1, 4])
            if col_back.button(
                f"← Back to {_ret_label}",
                key="det_back",
                use_container_width=True,
            ):
                st.session_state.pop("detail_return", None)
                st.session_state["nav"] = _ret if _ret in ("review", "map", "detections") else "detections"
                st.rerun()


# ----------------------------------------------------------------
# MAP — clickable detection markers (streamlit-folium)
# ----------------------------------------------------------------
elif cur_nav == "map":
    st.markdown("## Survey Coverage Map")
    _gw = st.session_state.get("_geo_warn")
    if _gw:
        st.warning(f"**GEOLOCATION WARNING** — {_gw}")
    if (st.session_state.get("survey_cfg") or {}).get("source") == "sample":
        st.caption("Illustrative locations — approximate survey-area coordinates only. No real survey navigation data for this sample.")
    if not file_bytes or not geod:
        st.warning("No detections to map — start a survey first.")
        if st.button("Go to Survey", key="map_to_survey", type="primary"):
            st.session_state["nav"] = "survey"
            st.rerun()
    else:
        _map_points = mapviz.build_survey_points(geod, reviews)

        st.caption(
            "Click a marker (or a numbered detection below) to focus it. "
            "Basemap: Esri World Imagery. "
            "Green=confirmed  Red=rejected  Amber=uncertain  Blue=unreviewed  Grey=possible natural feature."
        )

        if not _FOLIUM_OK:
            st.error(
                "Map click handling requires streamlit-folium. "
                "Install it with: pip install streamlit-folium folium"
            )
        elif not _map_points:
            st.warning("No geotagged points to plot.")
        else:
            _focus_id = st.session_state.get("_map_focus")
            _focus_pt = next((p for p in _map_points if p["det_id"] == _focus_id), None)
            _bounds = mapviz.bounds_of(_map_points)
            _clat = (_bounds[0][0] + _bounds[1][0]) / 2.0
            _clon = (_bounds[0][1] + _bounds[1][1]) / 2.0
            _fmap = folium.Map(location=[_clat, _clon], tiles=None,
                               min_zoom=3, max_bounds=True)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery",
                name="Esri",
                no_wrap=True,      # stop the basemap repeating sideways at low zoom
                min_zoom=3,
            ).add_to(_fmap)
            # Fit to all detections with padding and a capped zoom, so a single
            # point (or a cluster only metres apart) doesn't slam to street level
            # or leave the map parked at world zoom where points overlap.
            _fmap.fit_bounds(_bounds, padding=(40, 40), max_zoom=17)
            for pt in _map_points:
                if pt["det_id"] == _focus_id:      # highlight ring for focused det
                    folium.CircleMarker(
                        location=[pt["lat"], pt["lon"]], radius=16,
                        color="#ffffff", weight=3, fill=False, opacity=0.95,
                    ).add_to(_fmap)
                folium.CircleMarker(
                    location=[pt["lat"], pt["lon"]],
                    radius=13 if pt["det_id"] == _focus_id else 10,
                    color=pt["color"],
                    fill=True,
                    fill_color=pt["color"],
                    fill_opacity=0.9,
                    weight=2,
                    tooltip=f"det:{pt['det_id']}",
                    popup=folium.Popup(
                        f"#{pt['det_id']} {pt['classification']}<br/>"
                        f"conf {pt['confidence']:.0f} / {pt['review_status']}",
                        max_width=240,
                    ),
                ).add_to(_fmap)
            # Programmatic re-centre: pass center/zoom from session state and key
            # the map on the focus so streamlit-folium re-applies the view (its
            # viewport otherwise persists across reruns). Unfocused -> None so the
            # baked-in fit_bounds is used instead.
            _map_out = st_folium(
                _fmap,
                height=520,
                width=None,
                center=[_focus_pt["lat"], _focus_pt["lon"]] if _focus_pt else None,
                zoom=18 if _focus_pt else None,
                returned_objects=["last_object_clicked", "last_object_clicked_tooltip"],
                key=f"survey_coverage_map_{_focus_id}",
            )
            _tip = None
            if isinstance(_map_out, dict):
                _tip = _map_out.get("last_object_clicked_tooltip")
                if not _tip and _map_out.get("last_object_clicked"):
                    loc = _map_out["last_object_clicked"]
                    lat_c, lon_c = loc.get("lat"), loc.get("lng")
                    if lat_c is not None and lon_c is not None:
                        _tip = min(
                            _map_points,
                            key=lambda p: (p["lat"] - lat_c) ** 2 + (p["lon"] - lon_c) ** 2,
                        )
                        _tip = f"det:{_tip['det_id']}"
            if _tip and _tip != st.session_state.get("_map_last_tip"):
                try:
                    _clicked_id = int(str(_tip).split(":")[-1])
                except (TypeError, ValueError):
                    _clicked_id = None
                if _clicked_id is not None and any(d["id"] == _clicked_id for d in geod):
                    st.session_state["_map_last_tip"] = _tip
                    st.session_state["selected_det"] = _clicked_id
                    st.session_state["detail_return"] = "map"
                    st.session_state["nav"] = "detail"
                    st.rerun()

            # ---- Clickable detection controls (focus the map on one) ----
            st.markdown("**Detections** — click a number to focus it on the map.")
            _valid_ids = {p["det_id"] for p in _map_points}
            _by_id = {d["id"]: d for d in geod}
            _all_ids = mapviz.clickable_detection_ids(geod)
            _ncol = 6
            for _start in range(0, len(_all_ids), _ncol):
                _row = _all_ids[_start:_start + _ncol]
                _cols = st.columns(_ncol)
                for _slot, _did in enumerate(_row):
                    _d = _by_id[_did]
                    _act = reviews.get(_did, {}).get("action", "unreviewed")
                    _dot = (mapviz.NATURAL_DOT if _d.get("likely_rock_or_shadow")
                            else mapviz.STATUS_DOT.get(_act, mapviz.STATUS_DOT["unreviewed"]))
                    _mappable = _did in _valid_ids
                    _label = f"{_dot} #{_did}" + ("" if _mappable else " ⚠")
                    if _cols[_slot].button(
                            _label, key=f"mapfocus_{_did}", use_container_width=True,
                            disabled=not _mappable,
                            type="primary" if _did == _focus_id else "secondary"):
                        st.session_state["_map_focus"] = _did
                        st.rerun()
            _fc1, _fc2 = st.columns([1, 3])
            if _fc1.button("Show all", key="map_show_all", use_container_width=True):
                st.session_state["_map_focus"] = None
                st.rerun()
            if _focus_pt:
                _fc2.caption(
                    f"Focused: #{_focus_pt['det_id']} · {_focus_pt['classification']} · "
                    f"{_focus_pt['review_status']} · "
                    f"{_focus_pt['lat']:.6f}, {_focus_pt['lon']:.6f}"
                )
                if _fc2.button("Open Anomaly Detail", key="map_focus_detail"):
                    st.session_state["selected_det"] = _focus_pt["det_id"]
                    st.session_state["detail_return"] = "map"
                    st.session_state["nav"] = "detail"
                    st.rerun()

        st.markdown("")
        m1, m2, _ = st.columns([1, 1, 2])
        if m1.button("Continue to Summary", key="map_to_sum", type="primary", use_container_width=True):
            st.session_state["nav"] = "review"
            st.rerun()
        if m2.button("Back to Queue", key="map_to_det", use_container_width=True):
            st.session_state["nav"] = "detections"
            st.rerun()


# ----------------------------------------------------------------
# PLAN CLEANUP — current-aware route from a real port (spec PART 2-24)
# ----------------------------------------------------------------
elif cur_nav == "plan":
    st.markdown("## Plan Cleanup Route")
    st.caption("Decision-support route planning for a cleanup/inspection vessel. "
               "Not autonomous navigation, not a certified navigational route.")
    target = st.session_state.get("plan_target")
    if not target:
        st.info("No confirmed anomaly selected. Open a detection, confirm it, then press "
                "**PLAN CLEANUP** to plan a route to it.")
        if st.button("Go to Detections", key="plan_to_det", type="primary"):
            st.session_state["nav"] = "detections"
            st.rerun()
    else:
        a_lat, a_lon = target["lat"], target["lon"]
        st.markdown("### Target Anomaly")
        tcol1, tcol2, tcol3 = st.columns(3)
        tcol1.metric("Detection", f"#{target['det_id']}")
        tcol2.metric("Latitude", f"{a_lat:.6f}")
        tcol3.metric("Longitude", f"{a_lon:.6f}")
        st.caption(f"Class: {target.get('object_class', 'anomaly')} · position from the "
                   "confirmed, sanity-checked geolocation (never a fabricated coordinate).")

        pc1, pc2 = st.columns(2)
        speed_kn = pc1.number_input(
            "Vessel cruising speed (kn)", min_value=2.0, max_value=25.0,
            value=float(st.session_state.get("plan_speed_kn", 8.0)), step=0.5,
            key="plan_speed_input",
            help="Documented planning assumption — a constant through-water cruising "
                 "speed. Change it to re-plan.",
        )
        prefer_live = pc2.checkbox(
            "Use live current forecast (Open-Meteo)", value=True, key="plan_prefer_live",
            help="If off, or if the live lookup fails, a clearly-labelled demonstration "
                 "current field is used instead.",
        )

        # PART 21: cache the plan; recompute only when target or inputs change.
        plan_key = (target["det_id"], round(a_lat, 6), round(a_lon, 6),
                    round(speed_kn, 2), bool(prefer_live))
        plan = None
        if st.session_state.get("_plan_key") == plan_key:
            plan = st.session_state.get("_plan_obj")
        if plan is None:
            from routing.plan import plan_cleanup
            _stages = {
                "current": "Loading current data",
                "ports": "Finding suitable port",
                "ranking": "Ranking departure ports",
                "routing": "Building marine route & optimizing",
                "done": "Finalizing",
            }
            with st.status("Planning cleanup route…", expanded=True) as _status:
                _seen = {}
                def _progress(stage):
                    if stage in _stages and stage not in _seen:
                        _seen[stage] = True
                        st.write(f"• {_stages[stage]}")
                plan = plan_cleanup(a_lat, a_lon, vessel_speed_kn=speed_kn,
                                    prefer_live_current=prefer_live, progress=_progress)
                _status.update(label="Route planning complete", state="complete",
                               expanded=False)
            st.session_state["_plan_key"] = plan_key
            st.session_state["_plan_obj"] = plan
            st.session_state["plan_speed_kn"] = speed_kn
            # Do NOT cache a transient failure (e.g. a port/current API timeout):
            # drop the key so simply revisiting the page retries the live lookup.
            if not plan.feasible:
                st.session_state["_plan_key"] = None

        # Provenance banner (PART 16/17) — always honest about data source.
        prov = plan.current_provenance
        if prov is not None and not prov.is_live:
            st.warning(prov.label)
        elif prov is not None:
            st.info(prov.label)

        # Honest data-source / safety notices (e.g. offline port catalog, land
        # not checked) — shown whether or not a route was found.
        for w in plan.warnings:
            if w and w != (prov.label if prov else None):
                st.caption(w)

        if not plan.feasible:
            st.error(f"Route planning could not complete: {plan.reason}")
        else:
            rec = plan.recommended_route
            port = plan.recommended_port

            # RECOMMENDED DEPARTURE PORT (PART 3/4)
            st.markdown("### Recommended Departure Port")
            st.markdown(f"**{port.name}** — {port.port_type}"
                        + (f" · {port.country}" if port.country else ""))
            st.caption(port.suitability)
            _src = ("bundled offline catalog" if port.osm_id == "offline-catalog"
                    else f"OpenStreetMap {port.osm_id}")
            st.caption(f"Source: {_src} · "
                       f"{port.straight_km:.1f} km straight-line to anomaly (reference only — "
                       "NOT the selection criterion).")
            st.success(plan.recommended_reason)

            # CURRENT CONDITIONS (PART 5) — reported from the sampled field.
            st.markdown("### Current Conditions")
            if rec.legs:
                _near = rec.legs[-1]  # leg closest to the anomaly
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Current speed", f"{_near.current_speed_ms:.2f} m/s")
                cc2.metric("Flowing toward", f"{_near.current_bearing_deg:.0f}° (from N)")
                _assist = plan.recommended_route.mean_assist_ms
                cc3.metric("Mean along-track assist", f"{_assist:+.2f} m/s",
                           help="+ = current helps along the route, − = current opposes. "
                                "Only the component projected onto the route heading counts.")
            if prov is not None:
                st.caption(f"Source: {prov.source} · Resolution: {prov.resolution}"
                           + (f" · Sampled: {prov.timestamp} UTC" if prov.timestamp else ""))

            # ROUTE DETAILS (PART 15) — every value is computed, none fabricated.
            st.markdown("### Route Details")
            _km = rec.total_dist_m / 1000.0
            _hrs = rec.total_time_s / 3600.0
            rd1, rd2, rd3, rd4 = st.columns(4)
            rd1.metric("Total distance", f"{_km:.1f} km")
            rd2.metric("Estimated transit", f"{_hrs:.1f} h")
            rd3.metric("Route segments", f"{len(plan.segments)}")
            rd4.metric("Cruising speed", f"{plan.vessel_speed_kn:.1f} kn")
            st.caption(f"Objective: current-aware minimum travel time · "
                       f"Generated {plan.generated_at} · {plan.route_class_label}")
            st.caption("Assumptions: constant through-water cruising speed; current field "
                       "treated as locally uniform for the live single-point sample; ETA "
                       "excludes on-site cleanup/inspection time. Fuel savings are NOT "
                       "modelled and are not reported.")

            # WHY THIS ROUTE (PART 12) — supported statements only.
            with st.expander("Why this route?", expanded=False):
                st.markdown(
                    f"- **{port.name}** gives the shortest *current-aware* travel time of the "
                    f"candidate ports evaluated (not merely the closest by straight line).\n"
                    f"- The route is optimized over a water-only marine grid, so it follows "
                    f"navigable water rather than a straight line across land.\n"
                    f"- Along-track current assist averages **{rec.mean_assist_ms:+.2f} m/s** "
                    f"over the route under the current field above.\n"
                    f"- This is a planning/visualization route for decision support — it is "
                    f"not a certified navigational route and does not model obstacles beyond "
                    f"the coastline mask."
                )

            # ALTERNATIVES (PART 13) — labelled by objective, none called "best".
            st.markdown("### Alternative Routes")
            _obj_labels = {
                "current_time": "Current-aware (recommended objective)",
                "distance": "Shortest distance",
                "min_resistance": "Lowest current resistance",
            }
            _alt_rows = []
            for obj, sol in plan.routes.items():
                if not sol.feasible:
                    _alt_rows.append({"Route": _obj_labels.get(obj, obj),
                                      "Distance (km)": "—", "Transit (h)": "—",
                                      "Mean assist (m/s)": "—", "Status": sol.reason or "infeasible"})
                    continue
                _alt_rows.append({
                    "Route": _obj_labels.get(obj, obj),
                    "Distance (km)": round(sol.total_dist_m / 1000.0, 1),
                    "Transit (h)": round(sol.total_time_s / 3600.0, 2),
                    "Mean assist (m/s)": round(sol.mean_assist_ms, 3),
                    "Status": "recommended" if obj == plan.primary_objective else "alternative",
                })
            import pandas as _pd
            st.dataframe(_pd.DataFrame(_alt_rows), use_container_width=True, hide_index=True)

            # ROUTE MAP (PART 10/11) — indicative straight-line, port -> all
            # confirmed anomalies. Drawn as straight segments only; the ETA above
            # is the optimizer's current-aware figure for the primary anomaly.
            st.markdown("### Recommended Route Map")
            _conf_pts = [p for p in mapviz.build_survey_points(geod, reviews)
                         if reviews.get(p["det_id"], {}).get("action") == "confirm"]
            # The anomaly this plan targets is always a stop, even if the confirm
            # flag hasn't round-tripped through the review store yet.
            if not any(p["det_id"] == target["det_id"] for p in _conf_pts):
                _conf_pts.append({"det_id": target["det_id"], "lat": a_lat, "lon": a_lon,
                                  "review_status": "confirm", "confidence": 0.0,
                                  "classification": "Artificial anomaly",
                                  "color": mapviz.STATUS_HEX["confirm"]})
            if not _FOLIUM_OK:
                st.error("Route map requires streamlit-folium (pip install streamlit-folium folium).")
            elif not _conf_pts:
                st.warning("No confirmed anomaly with a valid position to route to.")
            else:
                _verts, _ordered = mapviz.build_route((port.lat, port.lon), _conf_pts)
                _rmap = folium.Map(location=[port.lat, port.lon], tiles=None,
                                   min_zoom=3, max_bounds=True)
                folium.TileLayer(
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri World Imagery", name="Esri", no_wrap=True, min_zoom=3,
                ).add_to(_rmap)
                _rmap.fit_bounds(mapviz.bounds_of(_verts), padding=(50, 50), max_zoom=14)
                # Dashed teal = indicative straight-line path (starts AT the port pin).
                folium.PolyLine(_verts, color="#2dd4bf", weight=4, opacity=0.9,
                                dash_array="8,8",
                                tooltip="Indicative straight-line cleanup route").add_to(_rmap)
                folium.Marker(
                    [port.lat, port.lon], tooltip=f"Departure: {port.name}",
                    icon=folium.Icon(color="green", icon="ship", prefix="fa"),
                ).add_to(_rmap)
                for _i, _s in enumerate(_ordered, start=1):
                    folium.Marker(
                        [_s["lat"], _s["lon"]],
                        tooltip=f"Stop {_i}: anomaly #{_s['det_id']}",
                        icon=folium.DivIcon(
                            icon_size=(26, 26), icon_anchor=(13, 13),
                            html=(f"<div style='background:{_s['color']};color:#fff;"
                                  "border:2px solid #fff;border-radius:50%;width:26px;"
                                  "height:26px;line-height:22px;text-align:center;"
                                  f"font-weight:700;font-size:13px;'>{_i}</div>")),
                    ).add_to(_rmap)
                st_folium(_rmap, height=480, width=None,
                          returned_objects=[], key="cleanup_route_map")
                st.caption(
                    "**Indicative straight-line route, not a navigable route** — no chart "
                    "or coastline data backs these segments. Green marker = departure port "
                    f"(**{port.name}**); numbered markers = cleanup stops in visit order "
                    "(nearest-neighbour). The transit estimate above is the current-aware "
                    "figure for the primary anomaly."
                )

            # ADD TO MISSION REPORT (PART 19)
            if st.button("Add this route to the mission report", key="plan_add_report",
                         type="primary"):
                plans = st.session_state.setdefault("mission_plans", {})
                plans[target["det_id"]] = {
                    "det_id": target["det_id"],
                    "anomaly_lat": a_lat, "anomaly_lon": a_lon,
                    "object_class": target.get("object_class", "anomaly"),
                    "port_name": port.name, "port_type": port.port_type,
                    "port_lat": port.lat, "port_lon": port.lon,
                    "distance_km": round(_km, 1), "eta_h": round(_hrs, 2),
                    "vessel_speed_kn": plan.vessel_speed_kn,
                    "objective": "current-aware minimum travel time",
                    "mean_assist_ms": round(rec.mean_assist_ms, 3),
                    "current_source": prov.source if prov else "n/a",
                    "current_timestamp": (prov.timestamp if prov else "") or "n/a",
                    "current_is_live": bool(prov.is_live) if prov else False,
                    "route_class_label": plan.route_class_label,
                    "generated_at": plan.generated_at,
                }
                st.success("Route added to the mission report.")

        st.markdown("")
        if st.button("← Back to Anomaly Detail", key="plan_back"):
            st.session_state["nav"] = "detail"
            st.rerun()


# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
elif cur_nav == "review":
    st.markdown("## Review Summary")
    if not file_bytes:
        st.warning("No survey loaded.")
        if st.button("Go to Survey", key="sum_to_survey", type="primary"):
            st.session_state["nav"] = "survey"
            st.rerun()
    elif len(geod) == 0:
        st.markdown('<div class="info-box">No anomalies detected in this survey — nothing to summarise.</div>', unsafe_allow_html=True)
        if st.button("Go to Analysis", key="sum_empty_an", type="primary"):
            st.session_state["nav"] = "analysis"
            st.rerun()
    else:
        rc = _review_counts()
        st.markdown(f"**Survey:** `{name}`  &nbsp;|&nbsp;  Run ID: `{run_id}`")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Confirmed", rc["confirmed"])
        s2.metric("Rejected", rc["rejected"])
        s3.metric("Uncertain / annotated", rc["uncertain"])
        s4.metric("Unreviewed", rc["unreviewed"])
        st.caption("Click an anomaly to reopen its review panel with the existing decision pre-filled and editable.")
        st.markdown("")

        for d in geod:
            det_id = d["id"]
            rev = reviews.get(det_id, {})
            action = rev.get("action", "unreviewed")
            flag_str = " — possible natural feature" if d["likely_rock_or_shadow"] else ""
            c_id, c_conf, c_st, c_act = st.columns([3, 1, 1, 1])
            c_id.markdown(
                f"{_ai_badge()} &nbsp; **#{det_id}**{flag_str}",
                unsafe_allow_html=True,
            )
            c_conf.markdown(f"conf **{d['confidence']:.0f}**")
            c_st.markdown(_badge(action), unsafe_allow_html=True)
            if c_act.button("Revise", key=f"sum_rev_{det_id}"):
                st.session_state["selected_det"] = det_id
                st.session_state["detail_return"] = "review"
                st.session_state["nav"] = "detail"
                st.rerun()
            st.markdown('<hr style="margin:4px 0; border-color:#21262d;">', unsafe_allow_html=True)

        st.markdown("")
        _first_conf = next(
            (d["id"] for d in geod if reviews.get(d["id"], {}).get("action") == "confirm"),
            geod[0]["id"],
        )
        b1, b2, b3 = st.columns([2, 2, 1])
        with b1:
            if st.button(
                "Investigate Anomalies",
                key="sum_investigate",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["selected_det"] = _first_conf
                st.session_state["detail_return"] = "review"
                st.session_state["nav"] = "detail"
                st.rerun()
        with b2:
            if st.button("Continue to Report", key="sum_to_report", use_container_width=True):
                st.session_state["nav"] = "report"
                st.rerun()
        with b3:
            if st.button("Queue", key="sum_to_queue", use_container_width=True):
                st.session_state["nav"] = "detections"
                st.rerun()


# ----------------------------------------------------------------
# REPORT (Phase 10)
# ----------------------------------------------------------------
elif cur_nav == "report":
    st.markdown("## Mission Report")
    if not file_bytes or json_path is None:
        st.warning("No survey loaded — start a survey first.")
        if st.button("Go to Survey", key="rep_to_survey", type="primary"):
            st.session_state["nav"] = "survey"
            st.rerun()
    else:
        rc = _review_counts()
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| Survey file | {name} |
| Run ID | `{run_id}` |
| Anomalies detected | {len(geod)} |
| Reviewed | {rc['reviewed']} |
| Confirmed | {rc['confirmed']} |
| Rejected | {rc['rejected']} |
| Uncertain / Annotated | {rc['uncertain']} |
| High-confidence (≥60) | {rc['high_conf']} |
| Inference time | {_t_infer_s:.2f}s |
        """)

        try:
            with st.spinner("Generating PDF mission report…"):
                pdf_bytes = build_mission_pdf(
                    name=name,
                    run_id=run_id,
                    geod=geod,
                    reviews=reviews,
                    rc=rc,
                    t_infer=_t_infer_s,
                    clean=clean,
                    plans=st.session_state.get("mission_plans"),
                )
            pdf_ok = True
            pdf_err_msg = None
        except Exception as _pdf_err:
            pdf_ok = False
            pdf_bytes = None
            pdf_err_msg = f"{type(_pdf_err).__name__}: {_pdf_err}"

        cdl, cdr, cdpdf = st.columns(3)
        cdl.download_button("Download JSON", json_path.read_bytes(), "survey_report.json", "application/json")
        cdr.download_button("Download CSV", csv_path.read_bytes(), "survey_report.csv", "text/csv")
        if pdf_ok:
            cdpdf.download_button("Download PDF Report", pdf_bytes, "survey_report.pdf", "application/pdf")
        else:
            cdpdf.error(f"Report generation failed: {pdf_err_msg}")
            st.error(f"Report generation failed: {pdf_err_msg}")
        st.markdown("")
        if st.button("← Back to Summary", key="rep_back_sum", type="primary"):
            st.session_state["nav"] = "review"
            st.rerun()


# ----------------------------------------------------------------
# SYSTEM & VALIDATION (Phase 8)
# ----------------------------------------------------------------
elif cur_nav == "system":
    st.markdown("## System & Validation")
    st.markdown('<div class="info-box">All numbers on this page are from measured evaluations. Sources are cited by file. No estimated or fabricated values.</div>', unsafe_allow_html=True)

    # --- Model architecture ---
    st.markdown("### Model Architecture")
    col_a, col_b = st.columns(2)
    col_a.markdown("""
**Architecture:** U-Net with ResNet34 encoder
**Attention:** None (no scSE, no CBAM)
**Input:** 512×512 greyscale tiles (tiled with 64px overlap, ramp blending)
**Output:** Per-pixel sigmoid probability map
**Post-processing:** Connected-component extraction, heuristic confidence scoring
**Checkpoint:** `best_model_v1_iou0.71.pth` (epoch 55/60)
    """)
    col_b.markdown("""
**Dataset:** AI4Shipwrecks — NOAA / Thunder Bay NMSA
**Images:** 286 (28 wreck sites, site-based split)
**License:** CC-BY-4.0
**Sensor:** Iver3 AUV, EdgeTech 2205 SSS, ~132 kHz
**Split:** 12 train sites, 13 held-out test sites — zero overlap
    """)

    st.divider()

    # --- Validation results ---
    st.markdown("### Validation Results")
    st.markdown("""
| Split | IoU | Dice | Notes |
|-------|-----|------|-------|
| Validation (in-distribution) | 0.713 | 0.833 | 12 training sites, best epoch |
| **Test (primary metric)** | **0.427** | **0.598** | **13 held-out sites, never seen in training** |

**Why they differ:** Validation IoU is measured on the same sites used during training (in-distribution). Test IoU is measured on 13 entirely separate sites — the primary, honest metric. The gap (0.713 → 0.427) reflects genuine generalisation difficulty across unseen survey sites, typical for a 286-image single-domain dataset.

SOTA context: published sonar ATR benchmarks report Dice/IoU 0.55–0.77. Test IoU 0.427 is below this range, consistent with the dataset size and hard site-based hold-out.
    """)

    st.divider()

    # --- Inference latency ---
    st.markdown("### Inference Latency")
    st.markdown("""
Hardware: Intel Core i7-12700H, NVIDIA RTX 4050 Laptop GPU, CUDA 12.1, PyTorch 2.5.1.
Measured on 10 test images (2476×1728 px), seed=42. Timings cover preprocess + tiled inference + post-processing. Excludes file I/O and report write.

| Backend | Mean | Std | Min | Max | Model size |
|---------|------|-----|-----|-----|------------|
| PyTorch FP32 / CPU | 4.29 s | 1.75 s | 1.53 s | 6.23 s | — |
| PyTorch FP32 / GPU (RTX 4050) | **0.37 s** | 0.16 s | 0.13 s | 0.54 s | — |
| ONNX FP32 / CPU | 3.08 s | 1.52 s | 0.89 s | 5.27 s | 97.7 MB |
| ONNX INT8 / CPU | 5.26 s | 1.84 s | 2.09 s | 7.06 s | 24.6 MB |

**ONNX INT8 is slower than FP32** on this CPU — dynamic quantisation overhead outweighs the size benefit at this tile count. ONNX INT8 also has numerical accuracy issues (max pixel diff 0.6497 vs. PyTorch). **Do not use INT8 for inference.** "Real-time" is not claimed.
    """)

    st.divider()

    # --- Robustness ---
    st.markdown("### Robustness Under Sonar Perturbations")
    st.markdown("""
Measured on 120 test images, seed=42. Each image run clean then with a single perturbation.
Baseline (clean): mean IoU = 0.058, mean detections/image = 3.4.

| Perturbation | Perturbed IoU | Δ IoU | Det/image (perturbed) | Notes |
|---|---|---|---|---|
| Speckle noise | 0.013 | **−78%** | 4.7 | Largest IoU drop; boundary degradation |
| Synthetic shadow | 0.058 | negligible | 3.0 | Model is shadow-aware; minimal effect |
| Radial distortion | 0.053 | −10% | 4.5 | Mild geometric warp |
| Heave/pitch/roll | 0.033 | **−44%** | **10.1** | **3× false-positive spike from shear artefacts** |

No robustness threshold is certified — these are characterisation numbers, not pass/fail guarantees.
Full data: `pipeline/robustness_results.json`
    """)

    st.divider()

    # --- Cross-domain ---
    st.markdown("### Cross-Domain Generalisation Check")
    st.markdown('<div class="warn-box"><b>Framing (mandatory):</b> This used a subset of the Santos 2024 AUV SSS dataset as a cross-domain generalization check for non-mine anomalous objects. It is not a mine-detection evaluation. The word "mine" does not appear in any SonarEye output.</div>', unsafe_allow_html=True)
    st.markdown("""
Eval dataset: Santos et al. 2024, 1,170 images, AUV SSS, non-mine anomalous-object subset.
Metric: IoU-of-boxes (bbox from mask components vs. YOLO GT bbox) — not comparable to pixel-level test IoU.

| Class | GT objects | Mean IoU-of-boxes | Det@0.5 |
|-------|-----------|-------------------|--------|
| Man-made anomalous objects (MILCO) | 437 | 0.007 | **0.0%** |
| Natural seafloor features (NOMBO) | 231 | 0.037 | **2.2%** |
| Background frames | 866 | — | **28.5% FP rate** |

**Finding:** Shipwreck-trained features do not transfer to this AUV SSS domain. Frequency mismatch (~132 kHz vs. 900–1800 kHz), structurally dissimilar targets, different shadow geometry, and water-type differences all contribute. This result is expected and informative — it characterises domain specificity, not a defect.
    """)

    st.divider()

    # --- Failure analysis ---
    st.markdown("### Qualitative Failure Analysis")
    st.markdown("Source: `docs/qualitative_examples.md` — held-out test set and robustness harness.")

    cases = [
        ("TP-1", "True positive",  "~0.75", "~74", "Anchor chain scatter — high contrast, well-defined shadow. Best-case conditions."),
        ("TP-2", "True positive",  "~0.48", "~58", "Partial hull — mask boundary drifts at far edge merged with reverberation band. Typical median-difficulty case."),
        ("TP-3", "True positive",  "~0.65", "~79", "Boiler/machinery mass — strong shadow drives shadow sub-score. Near-circular, high solidity."),
        ("HARD-1", "Ambiguous",    "~0.22", "~49", "Reef-wreck overlap — acoustic impedance of reef and hull similar at 132 kHz. Model cannot distinguish."),
        ("HARD-2", "Ambiguous",    "~0.18", "~31", "Small distal fragment (~30×20px) — near-threshold instability. Score correctly places it near likely_rock_or_shadow boundary (35.0)."),
        ("FP-1",  "False positive", "N/A",  "~41–48", "Background frame, heave perturbation — shear creates bright-edge artefacts. Mechanism behind 3× FP spike."),
        ("FN-1",  "False negative", "0 (miss)", "N/A", "Buried/sedimented hull — low contrast, under-represented in 286-image training set."),
    ]
    cols = st.columns([1, 1.5, 1, 1, 4])
    for h, text in zip(["ID", "Category", "Tile IoU", "Score", "Description"], cols):
        h_col = text
        h_col.markdown(f"**{h}**")
    for row in cases:
        c1, c2, c3, c4, c5 = st.columns([1, 1.5, 1, 1, 4])
        c1.markdown(f"`{row[0]}`")
        c2.markdown(row[1])
        c3.markdown(row[2])
        c4.markdown(row[3])
        c5.markdown(row[4])

    st.divider()

    # --- Dominant failure mode ---
    st.markdown("### Convergent Failure-Mode Finding")
    st.markdown("""
Four independent measurements converge on the same finding:

> **The model fires on unfamiliar sonar texture rather than discriminating structural shape.**

Evidence:
1. **Cross-domain FP rate 28.5%** on background frames (Santos 2024) — the model fires on unfamiliar SSS texture with no target present
2. **Heave/pitch/roll +196% FP spike** (3.4 → 10.1 det/image) — shear artefacts that resemble high-contrast targets trigger indiscriminate firing
3. **Synthetic net-patch activation 86.7%** (26/30 patches) — the model fires on synthetic elongated mesh textures that share low-level statistics with shipwreck debris fields (illustrative, unvalidated, synthetic-only exploration — not a demonstrated net-detection capability)
4. **Reef-wreck overlap HARD-1** — the model cannot distinguish geological from anthropogenic backscatter when their acoustic profiles overlap

This failure mode is structurally tied to the training set: 286 images of a single wreck-type domain, no hard-negative mining, no sonar-specific augmentation beyond the four stochastic perturbations in `synthaug.py`.
    """)

    st.divider()

    # --- Scope & limitations ---
    st.markdown("### Scope & Limitations")
    st.markdown("""
| Target class | Status | Reason |
|---|---|---|
| Shipwrecks (structural debris, hull plating, anchors, machinery) | Trained & evaluated | AI4Shipwrecks dataset, 28 sites |
| Pipes / cylinders | Not included | Sidescantoolbox GPL-3.0 licence conflict — cannot redistribute modified training data |
| Ghost / entangled fishing nets | Not included | No public labeled sonar dataset exists anywhere for this class |
| Generic seabed debris | Unproven | Not specifically trained or evaluated |

**Confidence score:** Heuristic composite (model prob × 0.6 + geometry × 0.28 + shadow × 0.12). Weights are hand-tuned — not learned. No calibration applied. Not a probability.

**In-domain false positive rate:** Not measured at detection level — pixel-level IoU does not measure per-frame detection alarms on non-wreck frames. Known gap.
    """)

    st.divider()

    # --- Synthetic net exploration note ---
    st.markdown("### Synthetic Net-Patch Exploration (Research Note)")
    st.markdown('<div class="warn-box"><b>Framing (mandatory):</b> Illustrative, unvalidated, synthetic-only exploration — not a demonstrated capability. The model was trained exclusively on shipwreck sonar imagery. No ghost-net sonar data was used at any stage.</div>', unsafe_allow_html=True)
    st.markdown("""
Script: `pipeline/synthetic_net_exploration.py` | 30 synthetic patches, seed=7

| Metric | Value |
|--------|-------|
| Patches with ≥1 detection | 26 / 30 (86.7%) |
| Total model detections | 72 |
| Mean detections / patch | 2.40 |

The 86.7% activation rate does not indicate ghost-net detection accuracy. It reflects the same indiscriminate texture-firing behaviour documented in the cross-domain check and heave robustness result. Activated regions are broad (3–11% of patch area) and spatially coincident with net strands because the strands span most of the patch — not because the model traced strand geometry.
    """)
    st.markdown("")
    if st.button("← Back to Overview", key="sys_to_ov", type="primary"):
        st.session_state["nav"] = "overview"
        st.rerun()

elif cur_nav == "seq_review":
    # ════════════════════════════════════════════════════════════════════════
    # SEQUENTIAL REVIEW — one-detection-at-a-time guided flow
    # State: review_idx (int) advances on save; completion screen at the end
    # ════════════════════════════════════════════════════════════════════════
    if not file_bytes or not geod:
        st.warning("No survey loaded. Load a survey and go to Detection Queue first.")
        if st.button("Back to Queue", key="sr_empty_back"):
            st.session_state["nav"] = "detections"
            st.rerun()
    else:
        _sr_total = len(geod)
        _sr_rc = _review_counts()
        _sr_n_done = _sr_rc["reviewed"]
        st.session_state["_sr_done"] = _sr_n_done
        st.session_state["_sr_total"] = _sr_total

        # ── Progress bar ──────────────────────────────────────────────────────
        _sr_pct = _sr_n_done / _sr_total if _sr_total > 0 else 1.0
        _pb_col, _stat_col = st.columns([3, 1])
        with _pb_col:
            st.progress(_sr_pct, text=f"**{_sr_n_done} / {_sr_total} reviewed**")
        with _stat_col:
            st.markdown(
                f"<div style='text-align:right;font-size:0.82rem;color:#8b949e;'>"
                f"<span style='color:#3fb950;'>&#10004; {_sr_rc['confirmed']}</span>  "
                f"<span style='color:#f85149;'>&#10006; {_sr_rc['rejected']}</span>  "
                f"<span style='color:#d29922;'>? {_sr_rc['uncertain']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Completion screen ─────────────────────────────────────────────────
        if _sr_n_done >= _sr_total and _sr_total > 0:
            st.session_state["review_complete"] = True
            st.markdown("## Review Complete")
            st.markdown(f"All **{_sr_total}** detections reviewed.")
            _cc1, _cc2, _cc3, _cc4 = st.columns(4)
            _cc1.metric("Confirmed",  _sr_rc["confirmed"])
            _cc2.metric("Rejected",   _sr_rc["rejected"])
            _cc3.metric("Uncertain",  _sr_rc["uncertain"])
            _cc4.metric("Total",      _sr_total)
            st.divider()
            _btn_a, _btn_b, _btn_back = st.columns([2, 2, 1])
            with _btn_a:
                if st.button("View on Map", key="sr_to_review",
                             type="primary", use_container_width=True):
                    st.session_state["nav"] = "map"
                    st.rerun()
            with _btn_b:
                _first_conf = next(
                    (d["id"] for d in geod
                     if reviews.get(d["id"], {}).get("action") == "confirm"),
                    geod[0]["id"],
                )
                if st.button("Investigate Anomalies", key="sr_to_detail",
                             use_container_width=True):
                    st.session_state["selected_det"] = _first_conf
                    st.session_state["detail_return"] = "map"
                    st.session_state["nav"] = "detail"
                    st.rerun()
            with _btn_back:
                if st.button("Back to Queue", key="sr_done_back"):
                    st.session_state["nav"] = "detections"
                    st.rerun()

        else:
            # ── Single-detection review panel ─────────────────────────────────
            # Build ordered sequence: unreviewed first, then reviewed
            _sr_order = (
                [d["id"] for d in geod if d["id"] not in reviews] +
                [d["id"] for d in geod if d["id"] in reviews]
            )
            # Track by det_id so position survives _sr_order recomputation after saves
            _target_id = st.session_state.get("review_det_id")
            if _target_id in _sr_order:
                _ridx = _sr_order.index(_target_id)
            else:
                _ridx = max(0, min(
                    st.session_state.get("review_idx", 0),
                    len(_sr_order) - 1,
                ))
            st.session_state["review_idx"] = _ridx
            _det_id = _sr_order[_ridx]
            st.session_state["review_det_id"] = _det_id
            _det = next((d for d in geod if d["id"] == _det_id), geod[_ridx])

            # Header row: position + prev/next nav
            _is_reviewed = _det_id in reviews
            _prev_action = reviews.get(_det_id, {}).get("action", "")
            _status_badge = {
                "confirm":   "<span style='color:#3fb950;'>&#10004; Previously confirmed</span>",
                "reject":    "<span style='color:#f85149;'>&#10006; Previously rejected</span>",
                "uncertain": "<span style='color:#d29922;'>? Previously uncertain</span>",
                "annotate":  "<span style='color:#d29922;'>&#9998; Previously annotated</span>",
            }.get(_prev_action, "<span style='color:#8b949e;'>Unreviewed</span>")

            _hdr, _nav_btns = st.columns([3, 2])
            with _hdr:
                st.markdown(
                    f"### Detection {_ridx + 1} of {_sr_total}  "
                    f"<span style='font-size:0.88rem;font-weight:400;'>{_status_badge}</span>",
                    unsafe_allow_html=True,
                )
            with _nav_btns:
                _pc, _nc = st.columns(2)
                with _pc:
                    if st.button("&#8592; Prev", key="sr_prev",
                                 disabled=(_ridx == 0),
                                 use_container_width=True):
                        st.session_state["review_idx"] = _ridx - 1
                        st.session_state["review_det_id"] = _sr_order[_ridx - 1]
                        st.rerun()
                with _nc:
                    if st.button("Next &#8594;", key="sr_next",
                                 disabled=(_ridx >= len(_sr_order) - 1),
                                 use_container_width=True):
                        st.session_state["review_idx"] = _ridx + 1
                        st.session_state["review_det_id"] = _sr_order[_ridx + 1]
                        st.rerun()

            # Detection summary metrics
            _conf   = _det.get("confidence", 0.0)
            _lat    = _det.get("lat") or 0.0
            _lon    = _det.get("lon") or 0.0
            _ai_cls = "Possible natural feature" if _det.get("likely_rock_or_shadow") else "Artificial anomaly"
            _mc1, _mc2, _mc3 = st.columns(3)
            _mc1.metric("Composite score", f"{_conf:.1f}")
            _mc2.metric("AI classification", _ai_cls)
            _mc3.metric("Location", f"{_lat:.5f}, {_lon:.5f}")
            if (st.session_state.get("survey_cfg") or {}).get("source") == "sample":
                st.caption("Illustrative location — approximate survey-area coordinates only. No real survey navigation data for this sample.")

            # Sonar crop evidence
            _bbox = _det.get("bbox_xyxy")
            if _bbox and clean is not None:
                try:
                    _x1, _y1, _x2, _y2 = _bbox
                    _pad = 15
                    _cy0 = max(0, _y1 - _pad); _cy1 = min(clean.shape[0], _y2 + _pad)
                    _cx0 = max(0, _x1 - _pad); _cx1 = min(clean.shape[1], _x2 + _pad)
                    _crop = clean[_cy0:_cy1, _cx0:_cx1]
                    if _crop.size > 0:
                        st.image(
                            (np.clip(_crop, 0, 1) * 255).astype(np.uint8),
                            caption="Sonar crop (enhanced)",
                            use_container_width=False,
                            width=300,
                        )
                except Exception:
                    pass

            st.divider()

            # ── Review form ───────────────────────────────────────────────────
            st.markdown("**Record your assessment:**")
            _existing = reviews.get(_det_id, {})
            _act_opts  = ["confirm", "reject", "uncertain", "annotate"]
            _def_act   = _act_opts.index(_existing["action"]) if _existing.get("action") in _act_opts else 2
            _sr_action = st.radio(
                "Decision",
                options=_act_opts,
                format_func=lambda x: {
                    "confirm":   "Confirm — treat as verified anomaly",
                    "reject":    "Reject — likely natural feature or artifact",
                    "uncertain": "Uncertain — needs further investigation",
                    "annotate":  "Annotate — add notes without a verdict",
                }[x],
                index=_def_act,
                horizontal=True,
                key=f"sr_action_{run_id}_{_det_id}",
            )
            _sr_cats = load_categories()
            _sr_cat_keys = list(_sr_cats.keys())
            _existing_cat = _existing.get("category", _sr_cat_keys[0])
            _sr_cat_idx = _sr_cat_keys.index(_existing_cat) if _existing_cat in _sr_cat_keys else 0
            _sr_category = st.selectbox(
                "Category",
                options=_sr_cat_keys,
                format_func=lambda k: f"{k} — {_sr_cats[k]}",
                index=_sr_cat_idx,
                key=f"sr_cat_{run_id}_{_det_id}",
            )
            _sr_note = st.text_area(
                "Notes (optional)",
                value=_existing.get("note") or "",
                max_chars=500,
                placeholder="Observations, uncertainties, follow-up needed…",
                key=f"sr_note_{run_id}_{_det_id}",
            )

            _is_last = (_ridx >= len(_sr_order) - 1)
            _save_label = {
                "confirm":   "Confirm & Next",
                "reject":    "Reject & Next",
                "uncertain": "Mark Uncertain & Next",
                "annotate":  "Annotate & Next",
            }.get(_sr_action, "Save & Next")
            if _is_last:
                _save_label = _save_label.replace(" & Next", " (last)")

            _sv_col, _sk_col, _ex_col = st.columns([2, 1, 1])
            with _sv_col:
                if st.button(_save_label,
                             key=f"sr_save_{run_id}_{_det_id}",
                             type="primary",
                             use_container_width=True):
                    upsert_review(conn, run_id, _det_id, _sr_action, _sr_category, _sr_note)
                    if not _is_last:
                        # Advance to next unreviewed, skipping already-reviewed
                        _updated_reviews = get_reviews_for_run(conn, run_id)
                        _next_i = next(
                            (i for i, did in enumerate(_sr_order)
                             if i > _ridx and did not in _updated_reviews),
                            _ridx + 1 if _ridx + 1 < len(_sr_order) else _ridx,
                        )
                        st.session_state["review_idx"] = _next_i
                        st.session_state["review_det_id"] = _sr_order[_next_i]
                    st.rerun()
            with _sk_col:
                if st.button("Skip", key=f"sr_skip_{run_id}_{_det_id}",
                             disabled=_is_last):
                    st.session_state["review_idx"] = min(_ridx + 1, len(_sr_order) - 1)
                    st.session_state["review_det_id"] = _sr_order[st.session_state["review_idx"]]
                    st.rerun()
            with _ex_col:
                if st.button("Exit Review", key="sr_exit"):
                    st.session_state["nav"] = "detections"
                    st.rerun()


else:
    st.info("Select a section from the sidebar.")
