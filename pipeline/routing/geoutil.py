"""Small geodesy helpers shared across the routing services.

Everything here is plain great-circle / local-tangent math on WGS84 degrees;
no projection library needed. Distances in metres, bearings in degrees
clockwise from true north (compass convention, matching geolocate.py).
"""
from __future__ import annotations

import math

R_EARTH_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R_EARTH_M * math.asin(min(1.0, math.sqrt(a)))


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing (deg CW from north) from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


def bearing_to_uv(bearing_deg: float, speed: float) -> tuple[float, float]:
    """Split a speed along a compass bearing into (east, north) components.

    Returns (u_east, v_north) in the same units as ``speed``. A vector pointing
    at bearing 90 deg (due east) yields (+speed, 0); bearing 0 (north) -> (0, +speed).
    """
    b = math.radians(bearing_deg)
    return speed * math.sin(b), speed * math.cos(b)


def uv_to_speed_bearing(u_east: float, v_north: float) -> tuple[float, float]:
    """Inverse of :func:`bearing_to_uv`: (u,v) -> (magnitude, bearing_deg)."""
    mag = math.hypot(u_east, v_north)
    brg = math.degrees(math.atan2(u_east, v_north)) % 360.0
    return mag, brg
