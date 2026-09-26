"""Nav-aware geolocation for XTF surveys.

Root-cause fix for detections landing on shore:

The previous XTF path (xtf_io.nav_to_meta + geotag.PixelGeoMapper) collapsed the
whole survey to ONE origin (the middle ping) + ONE heading + a LINEAR along-track
extrapolation measured from the top of the image. Because the along-track origin
(image row 0) and the coordinate origin (middle ping) disagreed, every detection
was pushed ~half the track length along the heading — enough to land on shore.

This module instead positions each detection from the navigation fix of the *ping
it actually sits on*:

    image row  ->  fractional ping index  ->  interpolated (lat, lon, heading)
    image col  ->  signed across-track ground range (port/starboard)
    (lat, lon) + across-track offset perpendicular to that ping's heading

No single global origin, no linear along-track guess — the along-track position is
the ping's own recorded position, so the result follows the real (curved) track.

Coordinate conventions
----------------------
* Navigation fixes are WGS84 decimal degrees (EPSG:4326), lat = SensorYcoordinate,
  lon = SensorXcoordinate.
* Heading is degrees clockwise from true north (compass convention).
* Waterfall column layout: left half = PORT channel, right half = STARBOARD
  (matches xtf_io.read_xtf, which packs row[:n]=port, row[n:]=stbd). A column right
  of centre is therefore a starboard return; starboard bearing = heading + 90 deg.
* Local offsets use an equirectangular (small-area) approximation about each ping's
  latitude; over a single survey swath (<~1 km) the error is sub-metre.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

M_PER_DEG_LAT = 111320.0


def m_per_deg_lon(lat_deg: float) -> float:
    return M_PER_DEG_LAT * np.cos(np.radians(lat_deg))


def _headings_from_track(lat, lon):
    """Bearing (deg CW from north) along the track, one per fix, by finite diff."""
    n = len(lat)
    hdg = np.zeros(n)
    for i in range(n):
        j0, j1 = max(0, i - 1), min(n - 1, i + 1)
        dn = (lat[j1] - lat[j0]) * M_PER_DEG_LAT
        de = (lon[j1] - lon[j0]) * m_per_deg_lon(lat[i])
        if abs(dn) + abs(de) < 1e-9:
            hdg[i] = hdg[i - 1] if i else 0.0
        else:
            hdg[i] = np.degrees(np.arctan2(de, dn)) % 360.0
    return hdg


@dataclass
class GeoResult:
    lat: float | None
    lon: float | None
    valid: bool
    reason: str = ""


class NavTrackGeoMapper:
    """Per-ping nav-aware pixel -> lat/lon mapper for XTF waterfalls.

    Parameters
    ----------
    nav : list[dict]  with keys lat, lon, heading_deg, range_m, altitude_m
    img_h, img_w : shape of the (preprocessed) waterfall the detections index into.
    """

    def __init__(self, nav, img_h, img_w, max_offtrack_m=None):
        if not nav:
            raise ValueError("NavTrackGeoMapper requires at least one navigation fix")
        self.h = int(img_h)
        self.w = int(img_w)
        self.n = len(nav)
        self.lat = np.array([float(p["lat"]) for p in nav])
        self.lon = np.array([float(p["lon"]) for p in nav])
        self.range_m = np.array([float(p.get("range_m") or 0.0) for p in nav])
        self.alt = np.array([float(p.get("altitude_m") or 0.0) for p in nav])

        # Prefer recorded per-ping heading when it actually varies / is set;
        # otherwise derive it from the track geometry (more robust than a
        # constant 0.0 that many exporters leave in the header).
        rec = np.array([float(p.get("heading_deg") or 0.0) for p in nav])
        if np.nanstd(rec) < 1e-6:  # constant (often all-zero) -> derive from track
            self.heading = _headings_from_track(self.lat, self.lon)
        else:
            self.heading = rec % 360.0

        med_range = float(np.median(self.range_m[self.range_m > 0])) if np.any(self.range_m > 0) else 50.0
        # A detection can be at most ~one swath-range off the track laterally;
        # allow a generous margin before we call a coordinate implausible.
        self.max_offtrack_m = max_offtrack_m if max_offtrack_m is not None else med_range * 2.0 + 200.0

    # -- interpolation helpers ------------------------------------------------
    def _ping_at_row(self, y_px):
        if self.h <= 1 or self.n == 1:
            return 0.0
        f = (y_px / (self.h - 1)) * (self.n - 1)
        return float(np.clip(f, 0.0, self.n - 1))

    def _interp(self, arr, f):
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, self.n - 1)
        t = f - i0
        return float(arr[i0] * (1 - t) + arr[i1] * t)

    def _interp_heading(self, f):
        # circular interpolation so 359->1 does not swing through 180
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, self.n - 1)
        t = f - i0
        a0, a1 = np.radians(self.heading[i0]), np.radians(self.heading[i1])
        x = np.cos(a0) * (1 - t) + np.cos(a1) * t
        y = np.sin(a0) * (1 - t) + np.sin(a1) * t
        return float(np.degrees(np.arctan2(y, x)) % 360.0)

    # -- core transform -------------------------------------------------------
    def locate(self, x_px, y_px) -> GeoResult:
        f = self._ping_at_row(y_px)
        lat_i = self._interp(self.lat, f)
        lon_i = self._interp(self.lon, f)
        hdg_i = self._interp_heading(f)
        rng_i = self._interp(self.range_m, f) or 50.0
        alt_i = self._interp(self.alt, f)

        half = self.w / 2.0
        across_px = x_px - half
        # slant range at this column, then correct to ground range with altitude
        slant = abs(across_px) * (rng_i / max(half, 1.0))
        ground = np.sqrt(max(slant**2 - alt_i**2, 0.0))
        if ground <= 0:
            ground = slant  # near-nadir fallback: no negative/zero jump to origin
        signed = np.sign(across_px) * ground  # +starboard, -port

        # across-track points to starboard = heading + 90 deg (CW from north)
        bearing = np.radians(hdg_i + 90.0)
        de = signed * np.sin(bearing)  # east metres
        dn = signed * np.cos(bearing)  # north metres

        lat = lat_i + dn / M_PER_DEG_LAT
        lon = lon_i + de / m_per_deg_lon(lat_i)

        ok, reason = self._validate(lat, lon)
        if not ok:
            return GeoResult(None, None, False, reason)
        return GeoResult(round(float(lat), 7), round(float(lon), 7), True)

    # -- sanity checks (never silently move a coordinate) ---------------------
    def _validate(self, lat, lon):
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return False, "non-finite coordinate"
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return False, "coordinate outside valid lat/lon range"
        d = self._dist_to_track_m(lat, lon)
        if d > self.max_offtrack_m:
            return False, f"coordinate {d/1000:.1f} km off the survey track"
        return True, ""

    def _dist_to_track_m(self, lat, lon):
        dn = (self.lat - lat) * M_PER_DEG_LAT
        de = (self.lon - lon) * m_per_deg_lon(lat)
        return float(np.min(np.sqrt(dn**2 + de**2)))

    def geotag(self, detections):
        """Return detections with lat/lon set (None + geo_valid=False if rejected)."""
        out = []
        for d in detections:
            cx, cy = d["centroid_px"]
            r = self.locate(cx, cy)
            out.append({
                **d,
                "lat": r.lat, "lon": r.lon,
                "geo_valid": r.valid, "geo_reason": r.reason,
            })
        return out


def coordinate_units_plausible(nav):
    """Cheap CRS guard: are the recorded fixes plausibly WGS84 degrees?

    Returns (ok, reason). If a file stored UTM/other units, lat/lon fall far
    outside the degree ranges and we refuse to geolocate rather than fake it.
    """
    if not nav:
        return False, "no navigation fixes in file"
    lat = np.array([p["lat"] for p in nav])
    lon = np.array([p["lon"] for p in nav])
    if np.all(lat == 0) and np.all(lon == 0):
        return False, "navigation fixes are all zero"
    if np.any(np.abs(lat) > 90) or np.any(np.abs(lon) > 180):
        return False, "navigation values exceed degree ranges (file may use UTM/projected units)"
    return True, ""
