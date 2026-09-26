"""Coarse land / coastline geometry for land-masking marine routes.

Uses a bundled Natural Earth 1:110m land layer (``data/ne_110m_land.geojson``,
committed to the repo so this works fully offline). At 1:110m this is a
*coarse* coastline: it is good enough to keep a planning route off obvious
landmasses, but it is NOT survey-grade and does not know about shallows,
channels, or small islands. Routes built on it are therefore always labelled a
"Planning / visualization route", never a certified navigational route.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

_DATA = Path(__file__).resolve().parent / "data" / "ne_110m_land.geojson"


@lru_cache(maxsize=1)
def _land_union():
    """Return (prepared_union, raw_union) of all land polygons, or (None, None).

    Cached: the union is built once per process. If the bundled file is missing
    we return (None, None) and callers treat the whole area as open water
    (land masking simply disabled — the route is still labelled 'planning').
    """
    if not _DATA.exists():
        return None, None
    with open(_DATA, "r", encoding="utf-8") as fh:
        gj = json.load(fh)
    geoms = [shape(f["geometry"]) for f in gj.get("features", []) if f.get("geometry")]
    if not geoms:
        return None, None
    union = unary_union(geoms)
    return prep(union), union


def coastline_available() -> bool:
    return _land_union()[0] is not None


def is_land(lat: float, lon: float) -> bool:
    """True if (lat, lon) falls on land per the coarse coastline.

    Returns False when no coastline data is available (fail-open: we would
    rather draw a planning route than silently refuse to plan).
    """
    prepared, _ = _land_union()
    if prepared is None:
        return False
    return bool(prepared.contains(Point(lon, lat)))


def land_geometry():
    """Raw shapely (Multi)Polygon of all land, or None if unavailable."""
    return _land_union()[1]
