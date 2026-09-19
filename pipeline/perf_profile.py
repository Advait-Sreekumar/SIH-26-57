"""Performance profiling for all pipeline stages.

Run from repo root or pipeline/:
    python pipeline/perf_profile.py

Hardware: Intel i7-12700H + RTX 4050 Laptop GPU, 16 GB RAM
All times via time.perf_counter() (wall-clock, single run each stage).
Writes results to pipeline/perf_profile_results.json.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

results = []

def rec(stage, t, notes=""):
    results.append({"stage": stage, "ms": round(t * 1000, 1), "notes": notes})
    print(f"  {stage:<52s}  {t*1000:>8.1f} ms  {notes}")

print("")
print("=== SonarEye Performance Profile ===")
print("  Hardware: Intel i7-12700H + RTX 4050 Laptop GPU, 16 GB RAM")
print("")

# ── Stage 1: torch import ─────────────────────────────────────────────────────
t0 = time.perf_counter()
import torch
t1 = time.perf_counter()
rec("torch import", t1 - t0)

# ── Stage 2: other pipeline imports ──────────────────────────────────────────
t0 = time.perf_counter()
import numpy as np
from PIL import Image as PILImage
from infer import SonarDetector
from confidence import score_detections
from geotag import PixelGeoMapper, SonarMeta
from app import build_mission_pdf
t1 = time.perf_counter()
rec("pipeline imports (numpy, PIL, infer, confidence, geotag, pdf_export)", t1 - t0)

# ── Stage 3: model weight load ────────────────────────────────────────────────
t0 = time.perf_counter()
detector = SonarDetector()
t1 = time.perf_counter()
rec("model weight load (SonarDetector init)", t1 - t0)

# ── Find a test image ─────────────────────────────────────────────────────────
samples = sorted(ROOT.glob("AI4Shipwrecks/**/test/images/*.png"))
if not samples:
    samples = sorted((ROOT.parent / "AI4Shipwrecks").glob("**/test/images/*.png"))
if not samples:
    print("  No test image found — skipping inference stages.")
    sys.exit(1)
img_path = samples[0]
print(f"  Test image: {img_path.name} ({img_path.stat().st_size // 1024} KB)")
print("")

# ── Stage 4: image load + preprocess ─────────────────────────────────────────
t0 = time.perf_counter()
import io
img_bytes = img_path.read_bytes()
img_arr = np.array(PILImage.open(io.BytesIO(img_bytes)).convert("L"))
t1 = time.perf_counter()
rec("image load + convert to numpy (disk read + PIL decode)", t1 - t0)
print(f"  Image shape: {img_arr.shape}")

# ── Stage 5: first (cold) inference ──────────────────────────────────────────
t0 = time.perf_counter()
prob, mask, preds = detector(img_arr)
t1 = time.perf_counter()
rec("first inference (cold: preprocess+tile+sigmoid+post)", t1 - t0, f"{len(preds)} detections")

# ── Stage 6: warm inference (3 runs, report mean) ────────────────────────────
warm_times = []
for _ in range(3):
    t0 = time.perf_counter()
    _, _, _ = detector(img_arr)
    t1 = time.perf_counter()
    warm_times.append(t1 - t0)
warm_mean = sum(warm_times) / len(warm_times)
rec("warm inference mean (3 runs)", warm_mean, f"runs: {[round(t*1000,1) for t in warm_times]} ms")

# ── Stage 7: confidence scoring ───────────────────────────────────────────────
t0 = time.perf_counter()
dets = score_detections(img_arr, prob, preds, mask)
t1 = time.perf_counter()
rec("confidence scoring (score_detections)", t1 - t0, f"{len(dets)} detections")

# ── Stage 8: geotagging ───────────────────────────────────────────────────────
t0 = time.perf_counter()
meta = SonarMeta(lat0=45.0855, lon0=-83.5684, heading_deg=90.0,
                 altitude_m=15.0, range_m=50.0, along_track_m=120.0)
mapper = PixelGeoMapper(img_h=img_arr.shape[0], img_w=img_arr.shape[1], meta=meta)
geod = mapper.geotag(dets)
t1 = time.perf_counter()
rec("geotagging (PixelGeoMapper.geotag)", t1 - t0, f"{len(geod)} detections")

# ── Stage 9: PDF first render ─────────────────────────────────────────────────
t0 = time.perf_counter()
try:
    pdf_bytes = build_mission_pdf(
        name=img_path.name,
        run_id="proftest_001",
        geod=geod,
        reviews={},
        rc={"reviewed": 0, "confirmed": 0, "rejected": 0,
             "uncertain": 0, "unreviewed": len(geod), "high_conf": 0},
        t_infer=warm_mean,
        clean=img_arr,
    )
    pdf_ok = True
except Exception as e:
    pdf_bytes = b""
    pdf_ok = False
    print(f"  PDF build failed: {e}")
t1 = time.perf_counter()
rec("PDF first render (build_mission_pdf)", t1 - t0,
    f"{len(pdf_bytes)//1024} KB" if pdf_ok else "FAILED")

# ── Stage 10: PDF second render (simulates cache miss) ───────────────────────
if pdf_ok:
    t0 = time.perf_counter()
    pdf_bytes2 = build_mission_pdf(
        name=img_path.name,
        run_id="proftest_001",
        geod=geod,
        reviews={},
        rc={"reviewed": 0, "confirmed": 0, "rejected": 0,
             "uncertain": 0, "unreviewed": len(geod), "high_conf": 0},
        t_infer=warm_mean,
        clean=img_arr,
    )
    t1 = time.perf_counter()
    rec("PDF second render (no cache; baseline for @st.cache_resource benefit)",
        t1 - t0, f"{len(pdf_bytes2)//1024} KB")

# ── Summary ───────────────────────────────────────────────────────────────────
cold_total = sum(r["ms"] for r in results[:5])  # import + imports + model + img + cold infer
print("")
print(f"  Cold-start total (stages 1–5: torch import through first inference):  "
      f"{cold_total:.1f} ms")
print(f"  Warm inference mean:  {warm_mean*1000:.1f} ms")
print("")

# ── Write results JSON ────────────────────────────────────────────────────────
out = {
    "tool": "perf_profile.py",
    "hardware": "Intel Core i7-12700H, RTX 4050 Laptop GPU, CUDA 12.1, 16 GB RAM",
    "image": img_path.name,
    "image_shape": list(img_arr.shape),
    "cold_start_total_ms": round(cold_total, 1),
    "warm_inference_mean_ms": round(warm_mean * 1000, 1),
    "stages": results,
}
out_path = ROOT / "perf_profile_results.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"  Results written to: {out_path}")
print("")
