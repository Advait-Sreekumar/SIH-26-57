# Geolocation VALIDATION tests (spec PART 1 sanity checks + PART 22).
# Complements test_geotag_regression.py (which locks the raw transform math).
# Here we assert the honesty guarantees: valid fixes land in the sea, wildly
# out-of-geometry fixes are nulled (never silently kept or moved), port and
# starboard detections fall on opposite sides, and headings rotate the swath.
# Run: pytest pipeline/test_geolocation_validation.py -v   (no streamlit import)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

from geotag import PixelGeoMapper, SonarMeta

# In-water English Channel demo origin (~10 NM SSE of Plymouth). The former
# default (45.0855, -83.5684) was ~11 km inland — the "anomaly on land" bug.
ORIGIN_LAT, ORIGIN_LON = 50.1000, -4.2000


def _mapper(heading=90.0, range_m=50.0, along=120.0, alt=15.0):
    meta = SonarMeta(lat0=ORIGIN_LAT, lon0=ORIGIN_LON, heading_deg=heading,
                     altitude_m=alt, range_m=range_m, along_track_m=along)
    return PixelGeoMapper(img_h=1024, img_w=640, meta=meta)


# --- Valid, in-swath detections keep coordinates and are flagged valid -------
def test_multiple_anomalies_all_valid_in_swath():
    m = _mapper()
    dets = [{"id": i, "centroid_px": (cx, cy)}
            for i, (cx, cy) in enumerate([(320, 0), (120, 300), (500, 700), (320, 1023)])]
    out = m.geotag(dets)
    assert all(d["geo_valid"] for d in out), out
    assert all(d["lat"] is not None and d["lon"] is not None for d in out)
    assert all(d["geo_reason"] == "" for d in out)


# --- Root cause guard: the in-water origin actually sits in the sea ----------
def test_valid_fixes_are_in_water():
    coastline = pytest.importorskip("routing.coastline")
    if not coastline.coastline_available():
        pytest.skip("coastline layer unavailable")
    m = _mapper()
    out = m.geotag([{"id": 0, "centroid_px": (320, 300)}])
    d = out[0]
    assert d["geo_valid"]
    assert coastline.is_land(d["lat"], d["lon"]) is False, "demo fix must be in the sea"


# --- Out-of-geometry coordinate is nulled, not kept, not moved ---------------
def test_far_coordinate_is_nulled_with_reason():
    m = _mapper()
    out = m.geotag([{"id": 0, "centroid_px": (10 ** 6, 10 ** 6)}])
    d = out[0]
    assert d["geo_valid"] is False
    assert d["lat"] is None and d["lon"] is None      # never silently kept
    assert "origin" in d["geo_reason"]                 # honest, human-readable


# --- Port vs starboard land on opposite sides of the track -------------------
def test_port_and_starboard_opposite_sides():
    m = _mapper(heading=0.0)  # north-facing: across-track is east/west
    # x < centre = port, x > centre = starboard (centre col = 320)
    port = m.geotag([{"id": 0, "centroid_px": (100, 300)}])[0]
    star = m.geotag([{"id": 1, "centroid_px": (540, 300)}])[0]
    assert port["geo_valid"] and star["geo_valid"]
    # heading north => across-track offset is longitude; opposite signs
    assert (port["lon"] - ORIGIN_LON) * (star["lon"] - ORIGIN_LON) < 0


# --- Different headings rotate the swath (east vs north differ) --------------
def test_heading_rotates_swath():
    east = _mapper(heading=90.0).geotag([{"id": 0, "centroid_px": (500, 600)}])[0]
    north = _mapper(heading=0.0).geotag([{"id": 0, "centroid_px": (500, 600)}])[0]
    assert east["geo_valid"] and north["geo_valid"]
    # same pixel under 90-deg-different headings must not map to the same point
    assert abs(east["lat"] - north["lat"]) > 1e-6 or abs(east["lon"] - north["lon"]) > 1e-6


# --- Wider range/along still validates in-swath detections -------------------
@pytest.mark.parametrize("range_m,along", [(30.0, 60.0), (75.0, 200.0), (150.0, 400.0)])
def test_various_ranges_keep_in_swath_valid(range_m, along):
    m = _mapper(range_m=range_m, along=along)
    out = m.geotag([{"id": 0, "centroid_px": (400, 500)}])
    assert out[0]["geo_valid"], out[0]
