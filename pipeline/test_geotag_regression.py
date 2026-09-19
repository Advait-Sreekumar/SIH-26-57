# Regression tests for the pixel-to-lat/lon geotag transform.
# Run: pytest pipeline/test_geotag_regression.py -v
# Does NOT import streamlit.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

from geotag import M_PER_DEG_LAT, PixelGeoMapper, SonarMeta

TOL_M = 0.5  # half-metre tolerance on all geometry assertions


def metres_per_deg_lon(lat_deg):
    return M_PER_DEG_LAT * np.cos(np.radians(lat_deg))


def latlon_to_metres(lat1, lon1, lat2, lon2, ref_lat):
    dn = (lat2 - lat1) * M_PER_DEG_LAT
    de = (lon2 - lon1) * metres_per_deg_lon(ref_lat)
    return float(np.sqrt(dn**2 + de**2))


# ---------------------------------------------------------------------------
# Case 1: nadir pixel maps exactly to the origin coordinate.
# heading=0 (north), 640x480 image, range=50, along_track=100.
# Pixel (320, 0) is the nadir at the start of the swath: expect (lat0, lon0).
# ---------------------------------------------------------------------------
def test_nadir_pixel_at_origin():
    meta = SonarMeta(lat0=45.0855, lon0=-83.5684, heading_deg=0.0,
                     altitude_m=15.0, range_m=50.0, along_track_m=100.0)
    mapper = PixelGeoMapper(img_h=480, img_w=640, meta=meta)
    lat, lon = mapper.pixel_to_latlon(320, 0)
    dist = latlon_to_metres(lat, lon, 45.0855, -83.5684, 45.0855)
    assert dist < TOL_M, f'nadir pixel should map to origin; got ({lat}, {lon}), dist={dist:.3f} m'


# ---------------------------------------------------------------------------
# Case 2: known-input / known-output regression lock.
# heading=90 (east), 640x480, range=50, along_track=120.
# Pixel (480, 120): across=25 px, slant=25*0.15625=3.906 m ... wait,
# mpa = 50/320 = 0.15625 m/px, so across_m = 160*0.15625 = 25 m
# ground = sqrt(25^2 - 15^2) = sqrt(400) = 20 m
# along = 120 * (120/480) = 30 m
# heading=90: east=[1,0], north=[0,-1]
# e = ground*north[0] + along*east[0] = 0 + 30 = 30 m
# n = ground*north[1] + along*east[1] = -20 + 0 = -20 m
# expected lat = 45.0855 + (-20)/111320 = 45.085320...
# expected lon = -83.5684 + 30 / (111320 * cos(45.0855)) = -83.568018...
# Pre-computed: lat=45.0853203, lon=-83.5680183
# ---------------------------------------------------------------------------
def test_known_pixel_heading_east():
    meta = SonarMeta(lat0=45.0855, lon0=-83.5684, heading_deg=90.0,
                     altitude_m=15.0, range_m=50.0, along_track_m=120.0)
    mapper = PixelGeoMapper(img_h=480, img_w=640, meta=meta)
    lat, lon = mapper.pixel_to_latlon(480, 120)
    expected_lat = 45.0853203
    expected_lon = -83.5680183
    dist = latlon_to_metres(lat, lon, expected_lat, expected_lon, 45.0855)
    assert dist < TOL_M, (
        f'regression mismatch: got ({lat:.7f}, {lon:.7f}), '
        f'expected ({expected_lat}, {expected_lon}), dist={dist:.3f} m'
    )


# ---------------------------------------------------------------------------
# Case 3: sample-survey sidebar defaults pin the detection over water.
# Parameters: lat0=45.0855, lon0=-83.5684, heading=90, alt=15,
# range=50, along_track=120, image 640x480.
# Nadir-column centre row (320, 240) should stay at lat0 latitude
# (no across-track offset) and shift east by along_track amount.
# Pre-computed: lat=45.0855, lon=-83.5676366
# ---------------------------------------------------------------------------
def test_sample_sidebar_defaults_centre_row():
    meta = SonarMeta(lat0=45.0855, lon0=-83.5684, heading_deg=90.0,
                     altitude_m=15.0, range_m=50.0, along_track_m=120.0)
    mapper = PixelGeoMapper(img_h=480, img_w=640, meta=meta)
    lat, lon = mapper.pixel_to_latlon(320, 240)
    expected_lat = 45.0855
    expected_lon = -83.5676366
    dist = latlon_to_metres(lat, lon, expected_lat, expected_lon, 45.0855)
    assert dist < TOL_M, (
        f'sample defaults centre-row mismatch: got ({lat:.7f}, {lon:.7f}), '
        f'expected ({expected_lat}, {expected_lon}), dist={dist:.3f} m'
    )


# ---------------------------------------------------------------------------
# Case 4: illustrative-location gating logic.
# survey_cfg["source"] == "sample" must be truthy; anything else must not be.
# This does not import Streamlit — tests the dict-key gating logic directly.
# ---------------------------------------------------------------------------
def _should_show_illustrative_caption(survey_cfg):
    return (survey_cfg or {}).get('source') == 'sample'


def test_illustrative_caption_shown_for_sample():
    assert _should_show_illustrative_caption({'source': 'sample'}) is True


def test_illustrative_caption_not_shown_for_upload():
    assert _should_show_illustrative_caption({'source': 'upload'}) is False


def test_illustrative_caption_not_shown_for_none():
    assert _should_show_illustrative_caption(None) is False


def test_illustrative_caption_not_shown_for_empty():
    assert _should_show_illustrative_caption({}) is False


# ---------------------------------------------------------------------------
# Case 5: slant-range correction floor -- when across_m < altitude_m
# the ground distance falls back to abs(across_m), not zero.
# Ensures no silent NaN / zero coordinates from near-nadir pixels.
# ---------------------------------------------------------------------------
def test_slant_range_fallback_near_nadir():
    meta = SonarMeta(lat0=0.0, lon0=0.0, heading_deg=0.0,
                     altitude_m=20.0, range_m=5.0, along_track_m=100.0)
    mapper = PixelGeoMapper(img_h=480, img_w=640, meta=meta)
    # pixel at x=325 -- tiny offset from nadir, slant < altitude
    lat, lon = mapper.pixel_to_latlon(325, 0)
    assert not np.isnan(lat), 'lat must not be NaN near nadir'
    assert not np.isnan(lon), 'lon must not be NaN near nadir'
    # across_m = 5 * (5/320) = 0.078 m < altitude 20 m -> fallback applies
    assert lat != 0.0 or lon != 0.0, 'position must shift from origin'
