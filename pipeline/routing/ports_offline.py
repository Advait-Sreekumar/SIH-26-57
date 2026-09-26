"""Bundled offline port catalog (fallback for when live OSM/Overpass is down).

The public Overpass API is load-shedding-prone (429/504/timeouts). When the live
lookup in :mod:`ports` fails we fall back to this small catalog of *real* ports
with *real* coordinates so the planner can still produce a route for a demo —
exactly the same honesty pattern the currents module uses (live -> labelled
offline). These are genuine harbours, NOT invented positions; the UI labels the
source as a bundled catalog so it is never presented as a live query.

Coordinates are approximate harbour-mouth positions (a few hundred metres is
immaterial to a decision-support route). This catalog is intentionally small and
regional (SW England / English Channel demo area, plus a few well-known ports);
if the anomaly is far from every entry we return nothing and the caller reports
an honest "no ports available" instead of stretching the data.
"""
from __future__ import annotations

from .geoutil import haversine_m
from .ports import Port

# (name, lat, lon, country, port_type, suitability) — real ports, real coords.
_CATALOG: list[tuple] = [
    # --- SW England / English Channel (demo survey area) ---
    ("Plymouth (Sutton Harbour)", 50.3672, -4.1300, "GB", "Harbour",
     "Major harbour — suitable for a support/inspection vessel"),
    ("Falmouth Harbour", 50.1530, -5.0630, "GB", "Harbour",
     "Deep natural harbour — suitable for larger vessels"),
    ("Fowey Harbour", 50.3360, -4.6380, "GB", "Harbour",
     "Working harbour — suitability depends on draft/berths (verify locally)"),
    ("Newlyn Harbour", 50.1030, -5.5470, "GB", "Harbour",
     "Commercial fishing harbour — likely suitable for an inspection vessel"),
    ("Brixham Harbour", 50.3970, -3.5130, "GB", "Harbour",
     "Fishing harbour — suitability depends on draft/berths (verify locally)"),
    ("Dartmouth Harbour", 50.3510, -3.5790, "GB", "Harbour",
     "Sheltered estuary harbour — verify berths locally"),
    ("Penzance Harbour", 50.1180, -5.5300, "GB", "Harbour",
     "Harbour — suitability depends on draft/tide (verify locally)"),
    ("Salcombe Harbour", 50.2370, -3.7690, "GB", "Harbour",
     "Estuary harbour — small/medium craft (verify locally)"),
    # --- other well-known real ports for broader coverage ---
    ("Port of Southampton", 50.8950, -1.4160, "GB", "Industrial port",
     "Major commercial port — suitable for larger vessels"),
    ("Port of Dover", 51.1250, 1.3400, "GB", "Industrial port",
     "Major commercial/ferry port — suitable for larger vessels"),
    ("Cherbourg Harbour", 49.6510, -1.6220, "FR", "Harbour",
     "Large artificial harbour — suitable for larger vessels"),
]


def offline_candidate_ports(lat: float, lon: float, radius_km: float = 200.0,
                            limit: int = 12) -> list[Port]:
    """Real ports from the bundled catalog within ``radius_km`` of (lat, lon)."""
    ports: list[Port] = []
    for name, plat, plon, country, ptype, suit in _CATALOG:
        d_km = haversine_m(lat, lon, plat, plon) / 1000.0
        if d_km <= radius_km:
            ports.append(Port(
                name=name, lat=plat, lon=plon, country=country,
                port_type=ptype, suitability=suit,
                osm_id="offline-catalog", straight_km=d_km,
            ))
    ports.sort(key=lambda p: p.straight_km)
    return ports[:limit]
