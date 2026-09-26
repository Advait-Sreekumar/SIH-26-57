"""Port discovery service.

Finds *real* maritime ports/harbours near a confirmed anomaly using a live
geographic API (OpenStreetMap via the Overpass API — keyless JSON over HTTPS).
Every port returned has real coordinates and a real OSM name; nothing is
invented. When the API is unreachable we return an empty list plus a reason so
the UI can honestly say "port data unavailable" rather than fabricate a port.

A port is NEVER selected here by straight-line distance alone — this module only
*discovers* candidates. Ranking (distance + navigable route + current
assistance + suitability) happens in ``optimizer.rank_ports`` (spec PART 4/5).
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

from .geoutil import haversine_m

_OVERPASS_URLS = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
# Lower per-mirror budget so all three mirrors are actually attempted within a
# reasonable wait when the public instances are load-shedding.
_HTTP_TIMEOUT = 20.0
_HEADERS = {"User-Agent": "SonarEye-SIH/1.0 (marine cleanup route planner)"}


@dataclass(frozen=True)
class Port:
    name: str
    lat: float
    lon: float
    country: str | None       # only if OSM tags it; never inferred/fabricated
    port_type: str            # factual descriptor derived from OSM tags
    suitability: str          # qualitative, tag-derived (documented as such)
    osm_id: str
    straight_km: float        # straight-line km to the anomaly (for reference only)


def _classify(tags: dict) -> tuple[str, str]:
    """Return (port_type, suitability) from OSM tags — purely tag-derived."""
    seamark = tags.get("seamark:harbour:category") or tags.get("seamark:type", "")
    if tags.get("industrial") == "port" or tags.get("landuse") == "industrial":
        return "Industrial port", "Larger vessels — likely suitable for a support/inspection vessel"
    if tags.get("harbour") == "yes" or seamark == "harbour":
        return "Harbour", "General harbour — suitability depends on draft/berths (verify locally)"
    if tags.get("leisure") == "marina" or "marina" in (tags.get("name", "").lower()):
        return "Marina", "Small-craft marina — may suit a small inspection vessel only"
    if tags.get("man_made") == "pier":
        return "Pier", "Pier/jetty — limited; verify locally"
    return "Port feature", "Type unspecified in source data — verify locally"


def _overpass_query(query: str) -> list:
    """POST an Overpass query, trying mirrors in turn. Real data only.

    Overpass public instances are load-shedding-prone (429/504); we try a few
    known mirrors before giving up. Raises the last error if all fail.
    """
    last_exc: Exception = requests.RequestException("no overpass endpoint tried")
    for url in _OVERPASS_URLS:
        try:
            resp = requests.post(url, data={"data": query},
                                 headers=_HEADERS, timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            continue
    raise last_exc


def find_candidate_ports(lat: float, lon: float, radius_km: float = 200.0,
                         limit: int = 12) -> tuple[list[Port], str, str]:
    """Discover real ports within ``radius_km`` of (lat, lon).

    Returns ``(ports, reason, source_note)``. ``reason`` is "" on success or a
    human-readable explanation when discovery failed/empty. ``source_note`` is
    "" for a live OSM result, or a labelled notice when the ports came from the
    bundled offline catalog (live lookup down) — the UI surfaces it so offline
    data is never presented as a live query. The straight-distance sort is ONLY
    a stable ordering for display; final route selection is done by the
    optimizer, not by this distance.
    """
    radius_m = int(radius_km * 1000)
    query = f"""
    [out:json][timeout:18];
    (
      node["harbour"](around:{radius_m},{lat},{lon});
      way["harbour"](around:{radius_m},{lat},{lon});
      node["seamark:type"="harbour"](around:{radius_m},{lat},{lon});
      node["industrial"="port"](around:{radius_m},{lat},{lon});
      way["industrial"="port"](around:{radius_m},{lat},{lon});
      way["landuse"="harbour"](around:{radius_m},{lat},{lon});
    );
    out tags center {limit * 4};
    """
    try:
        elements = _overpass_query(query)
    except (requests.RequestException, ValueError) as exc:
        return _offline_fallback(lat, lon, radius_km, limit, type(exc).__name__)

    ports: list[Port] = []
    seen: set[tuple[float, float]] = set()
    for el in elements:
        tags = el.get("tags", {}) or {}
        name = tags.get("name")
        if not name:
            continue  # unnamed features are not useful as a labelled departure port
        plat = el.get("lat") or el.get("center", {}).get("lat")
        plon = el.get("lon") or el.get("center", {}).get("lon")
        if plat is None or plon is None:
            continue
        key = (round(plat, 4), round(plon, 4))
        if key in seen:
            continue
        seen.add(key)
        ptype, suit = _classify(tags)
        ports.append(Port(
            name=name, lat=float(plat), lon=float(plon),
            country=tags.get("addr:country"),
            port_type=ptype, suitability=suit,
            osm_id=f"{el.get('type')}/{el.get('id')}",
            straight_km=haversine_m(lat, lon, float(plat), float(plon)) / 1000.0,
        ))

    if not ports:
        return [], (f"No named ports found within {radius_km:.0f} km of the anomaly "
                    "in OpenStreetMap."), ""
    ports.sort(key=lambda p: p.straight_km)
    return ports[:limit], "", ""


def _offline_fallback(lat: float, lon: float, radius_km: float, limit: int,
                      exc_name: str) -> tuple[list[Port], str, str]:
    """Live OSM lookup failed — try the bundled real-port catalog, labelled."""
    from .ports_offline import offline_candidate_ports
    ports = offline_candidate_ports(lat, lon, radius_km=radius_km, limit=limit)
    if not ports:
        return [], (f"Port data unavailable — live lookup failed ({exc_name}) and no "
                    f"bundled port lies within {radius_km:.0f} km of the anomaly."), ""
    note = (f"PORT DATA: live OpenStreetMap lookup unavailable ({exc_name}) — using "
            "bundled offline port catalog (real ports, cached reference data; not a "
            "live query).")
    return ports, "", note
