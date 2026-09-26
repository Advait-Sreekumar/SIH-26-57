# Tests for nav-aware geolocation (root-cause fix for on-land detections).
# Run: pytest pipeline/test_geolocate.py -v   (does NOT import streamlit)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

from geolocate import (
    M_PER_DEG_LAT,
    NavTrackGeoMapper,
    coordinate_units_plausible,
    m_per_deg_lon,
)

TOL_M = 1.0


def latlon_to_m(lat1, lon1, lat2, lon2, ref_lat):
    dn = (lat2 - lat1) * M_PER_DEG_LAT
    de = (lon2 - lon1) * m_per_deg_lon(ref_lat)
    return float(np.hypot(dn, de))


def north_track(n=64, lat0=45.0, lon0=-83.5, dlat=1e-4, rng=50.0, alt=15.0):
    return [
        {"lat": lat0 + i * dlat, "lon": lon0, "heading_deg": 0.0,
         "range_m": rng, "altitude_m": alt}
        for i in range(n)
    ]


def east_track(n=64, lat0=45.0, lon0=-83.5, dlon=1e-4, rng=50.0, alt=15.0):
    return [
        {"lat": lat0, "lon": lon0 + i * dlon, "heading_deg": 0.0,  # heading derived from track
         "range_m": rng, "altitude_m": alt}
        for i in range(n)
    ]


# 1. Known location -> expected sea coordinate (starboard on north track = east).
def test_known_starboard_offset_north_track():
    nav = north_track()
    m = NavTrackGeoMapper(nav, img_h=len(nav), img_w=640)
    mid_row = (len(nav) - 1) / 2.0
    r = m.locate(480, mid_row)  # across_px=160 -> slant 25 -> ground 20 m east
    assert r.valid
    lat_i = nav[0]["lat"] + mid_row * 1e-4   # interpolated latitude at mid row
    expected_lon = nav[0]["lon"] + 20 / m_per_deg_lon(lat_i)
    assert latlon_to_m(r.lat, r.lon, lat_i, expected_lon, lat_i) < TOL_M


# 3 & 4. Port vs starboard land on opposite sides.
def test_port_and_starboard_opposite_sides():
    nav = north_track()
    m = NavTrackGeoMapper(nav, img_h=len(nav), img_w=640)
    stbd = m.locate(480, 32)   # right of centre
    port = m.locate(160, 32)   # left of centre
    assert stbd.lon > nav[0]["lon"] > port.lon  # starboard east, port west
    assert abs(stbd.lat - port.lat) < 1e-6      # same along-track latitude


# 5. Different heading: starboard of an eastbound vessel is to the south.
def test_heading_east_starboard_is_south():
    nav = east_track()
    m = NavTrackGeoMapper(nav, img_h=len(nav), img_w=640)
    mid = (len(nav) - 1) / 2.0
    stbd = m.locate(480, mid)
    assert stbd.lat < nav[0]["lat"]             # south of track
    lon_i = nav[0]["lon"] + mid * 1e-4          # interpolated track lon at mid row
    assert abs(stbd.lon - lon_i) < 1e-6         # no east/west displacement (pure south)


# 6. Larger range scales the across-track offset.
def test_range_scales_offset():
    a = NavTrackGeoMapper(north_track(rng=50.0), 64, 640)
    b = NavTrackGeoMapper(north_track(rng=150.0), 64, 640)
    ra = a.locate(600, 32)
    rb = b.locate(600, 32)
    da = latlon_to_m(ra.lat, ra.lon, 45.0032, -83.5, 45.0)
    db = latlon_to_m(rb.lat, rb.lon, 45.0032, -83.5, 45.0)
    assert db > da  # wider range -> point further off-track


# 7. Curved track: a bottom-row detection uses the LAST ping's fix, not the first.
def test_curved_track_follows_path():
    n = 64
    nav = [{"lat": 45.0 + i * 1e-4, "lon": -83.5 + (i * 1e-4 if i > n // 2 else 0.0),
            "heading_deg": 0.0, "range_m": 50.0, "altitude_m": 15.0} for i in range(n)]
    m = NavTrackGeoMapper(nav, img_h=n, img_w=640)
    top = m.locate(320, 0)          # nadir at first ping
    bottom = m.locate(320, n - 1)   # nadir at last ping
    assert latlon_to_m(top.lat, top.lon, nav[0]["lat"], nav[0]["lon"], 45.0) < TOL_M
    assert latlon_to_m(bottom.lat, bottom.lon, nav[-1]["lat"], nav[-1]["lon"], 45.0) < TOL_M


# 2. Multiple anomalies all resolve and stay near the track.
def test_multiple_anomalies_near_track():
    nav = north_track()
    m = NavTrackGeoMapper(nav, img_h=len(nav), img_w=640)
    dets = [{"centroid_px": [x, y]} for x, y in [(200, 5), (480, 20), (320, 40), (600, 60)]]
    out = m.geotag(dets)
    assert all(d["geo_valid"] for d in out)
    for d in out:
        assert m._dist_to_track_m(d["lat"], d["lon"]) <= m.max_offtrack_m


# 8/9. Off-track coordinate is rejected, not silently moved.
def test_offtrack_coordinate_rejected():
    nav = north_track()
    m = NavTrackGeoMapper(nav, img_h=len(nav), img_w=640, max_offtrack_m=1.0)
    r = m.locate(600, 32)  # ~44 m off-track, exceeds 1 m limit
    assert r.valid is False
    assert r.lat is None and r.lon is None
    assert "off the survey track" in r.reason


def test_nadir_sits_on_track():
    nav = north_track()
    m = NavTrackGeoMapper(nav, img_h=len(nav), img_w=640)
    r = m.locate(320, 10)
    assert r.valid
    assert m._dist_to_track_m(r.lat, r.lon) < TOL_M


# CRS guard.
def test_units_guard_flags_utm_like_values():
    ok, _ = coordinate_units_plausible(north_track())
    assert ok
    utm = [{"lat": 5012345.0, "lon": 456789.0, "heading_deg": 0, "range_m": 50, "altitude_m": 15}]
    ok2, reason = coordinate_units_plausible(utm)
    assert ok2 is False and "degree ranges" in reason


def test_units_guard_flags_all_zero():
    z = [{"lat": 0.0, "lon": 0.0, "heading_deg": 0, "range_m": 50, "altitude_m": 15}]
    ok, reason = coordinate_units_plausible(z)
    assert ok is False
