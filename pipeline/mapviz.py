"""Pure helpers for the Survey Coverage Map and the indicative cleanup-route
map rendered in app.py.

No streamlit/folium imports live here on purpose: the point-building, bounds
and route-ordering logic is then unit-testable in isolation. This module is
MAP + UI code only — nothing in it touches the model, inference, confidence
scoring or the geotag math (it consumes the coordinates those produce).
"""
from __future__ import annotations

import math

# Review-status -> marker colour. Mirrors the badge scheme used across the app:
# confirmed green, rejected red, uncertain/annotate amber, unreviewed blue.
STATUS_HEX = {
    "confirm":    "#3fb950",
    "reject":     "#f85149",
    "uncertain":  "#d29922",
    "annotate":   "#d29922",
    "unreviewed": "#58a6ff",
}
# Possible natural feature (rock/shadow) — grey, overrides the status colour to
# match the existing Survey Coverage Map legend.
NATURAL_HEX = "#8b949e"

# Coloured status dot for the clickable detection controls (button labels can't
# be styled per-item in Streamlit, so we prefix a unicode dot in the marker
# colour family instead).
STATUS_DOT = {
    "confirm": "\U0001F7E2",     # green
    "reject": "\U0001F534",      # red
    "uncertain": "\U0001F7E1",   # amber
    "annotate": "\U0001F7E1",    # amber
    "unreviewed": "\U0001F535",  # blue
}
NATURAL_DOT = "⚪"           # white (possible natural feature)


def build_survey_points(geod, reviews=None):
    """One plottable point per detection that carries a valid geolocation.

    Every detection with non-None lat/lon is included — there is no dedup and no
    confirmed-only filter, so N distinct detections yield N points even when
    they lie only metres apart. Colour is by review status (or grey for a
    possible natural feature), matching the map legend.
    """
    reviews = reviews or {}
    points = []
    for d in geod:
        lat, lon = d.get("lat"), d.get("lon")
        if lat is None or lon is None:
            continue
        action = reviews.get(d["id"], {}).get("action", "unreviewed")
        natural = bool(d.get("likely_rock_or_shadow"))
        points.append({
            "det_id": d["id"],
            "lat": float(lat),
            "lon": float(lon),
            "confidence": round(float(d.get("confidence", 0.0)), 1),
            "review_status": action,
            "classification": "Possible natural feature" if natural else "Artificial anomaly",
            "color": NATURAL_HEX if natural else STATUS_HEX.get(action, STATUS_HEX["unreviewed"]),
        })
    return points


def clickable_detection_ids(geod):
    """Detection ids for the clickable list below the map — one per detection in
    the survey, in queue order, using the same ``id`` as the Detection Queue and
    Anomaly Detail so numbering stays consistent."""
    return [d["id"] for d in geod]


def bounds_of(points):
    """Axis-aligned ``[[min_lat, min_lon], [max_lat, max_lon]]`` covering every
    point. Accepts point dicts (lat/lon keys) or ``(lat, lon)`` tuples. Returns
    ``None`` for an empty input."""
    lats, lons = [], []
    for p in points:
        if isinstance(p, dict):
            lats.append(p["lat"]); lons.append(p["lon"])
        else:
            lats.append(p[0]); lons.append(p[1])
    if not lats:
        return None
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def _hav_m(a, b):
    """Haversine metres between (lat, lon) pairs — for ordering stops only."""
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = math.radians(b[0] - a[0])
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def nearest_neighbour_order(start, stops):
    """Greedy nearest-neighbour visiting order, as indices into ``stops``.
    ``start`` and ``stops`` are ``(lat, lon)`` tuples."""
    remaining = list(range(len(stops)))
    order = []
    cur = (start[0], start[1])
    while remaining:
        nxt = min(remaining, key=lambda i: _hav_m(cur, (stops[i][0], stops[i][1])))
        order.append(nxt)
        cur = (stops[nxt][0], stops[nxt][1])
        remaining.remove(nxt)
    return order


def build_route(port_latlon, stops):
    """Indicative straight-line multi-stop route.

    Returns ``(vertices, ordered_stops)`` where ``vertices`` is a list of
    ``[lat, lon]`` whose FIRST element is exactly the port coordinate, followed
    by the stops in nearest-neighbour visit order. ``ordered_stops`` is the stop
    objects in that same order. This is a straight-line indicative path only —
    it is not a navigable route and no chart/coastline data backs it.
    """
    plat, plon = float(port_latlon[0]), float(port_latlon[1])
    stop_ll = [(float(s["lat"]), float(s["lon"])) if isinstance(s, dict)
               else (float(s[0]), float(s[1])) for s in stops]
    order = nearest_neighbour_order((plat, plon), stop_ll)
    ordered_stops = [stops[i] for i in order]
    vertices = [[plat, plon]] + [[stop_ll[i][0], stop_ll[i][1]] for i in order]
    return vertices, ordered_stops
