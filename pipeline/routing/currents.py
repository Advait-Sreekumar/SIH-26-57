"""Ocean-current data service.

Primary source: Open-Meteo Marine API (live, keyless JSON over HTTPS) — returns
an ocean-current velocity + direction for a lat/lon. When that is unreachable
or returns no data for the location (e.g. inland / lake water the ocean model
does not cover), we fall back to a deterministic *analytic demonstration* field
that is labelled as such everywhere it surfaces.

Honesty rules honoured here (spec PARTS 11, 17):
    * Live data is labelled "Forecast" with its source + timestamp.
    * The offline field is labelled "Demonstration current data (synthetic, not
      measured)". We NEVER present the synthetic field as live/measured, and we
      never claim "real-time".
    * We never silently fabricate a value and pass it off as real.

Direction convention: bearings are degrees CW from true north and describe the
direction the water is flowing TOWARDS (oceanographic convention). Open-Meteo's
``ocean_current_direction`` uses the same flow-towards convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from .geoutil import bearing_to_uv

_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
_HTTP_TIMEOUT = 12.0


@dataclass(frozen=True)
class CurrentVector:
    """A single sampled current, flowing TOWARDS ``bearing_deg`` at ``speed_ms``."""
    u_east_ms: float
    v_north_ms: float
    speed_ms: float
    bearing_deg: float


@dataclass(frozen=True)
class CurrentProvenance:
    kind: str          # "live-forecast" | "demonstration"
    source: str        # human-readable source name
    timestamp: str     # ISO-8601, or "" for the timeless analytic field
    resolution: str    # human-readable native resolution
    label: str         # the exact banner text the UI must show
    is_live: bool


class CurrentField:
    """A current sampler over the area of interest, with provenance metadata.

    For the live case we sample once at the anomaly and treat the field as
    locally uniform over the (small, <~200 km) planning area — the free
    Open-Meteo endpoint is point-query only, and one honest sampled vector beats
    many fabricated ones. For the demonstration case we evaluate a smooth
    analytic function per point so the field visibly varies across the map.
    """

    def __init__(self, provenance: CurrentProvenance, uniform: CurrentVector | None = None):
        self.provenance = provenance
        self._uniform = uniform  # set for live; None for analytic demo

    def sample(self, lat: float, lon: float) -> CurrentVector:
        if self._uniform is not None:
            return self._uniform
        return _analytic_demo_vector(lat, lon)


def _mk_vector(speed_ms: float, toward_bearing_deg: float) -> CurrentVector:
    u, v = bearing_to_uv(toward_bearing_deg, speed_ms)
    return CurrentVector(u, v, speed_ms, toward_bearing_deg % 360.0)


def _analytic_demo_vector(lat: float, lon: float) -> CurrentVector:
    """Deterministic, physically-plausible synthetic current (NOT measured).

    A gentle spatially-varying field: speed 0.10-0.55 m/s, direction rotating
    slowly with position so routes see genuine assist/resist depending on
    heading. Purely a function of position, so it is reproducible for a demo.
    """
    speed = 0.32 + 0.22 * math.sin(math.radians(lat * 3.0)) * math.cos(math.radians(lon * 3.0))
    speed = max(0.10, min(0.55, speed))
    toward = (90.0 + 40.0 * math.sin(math.radians(lat * 2.0 + lon * 2.0))) % 360.0
    return _mk_vector(speed, toward)


def _demo_field(reason: str) -> CurrentField:
    prov = CurrentProvenance(
        kind="demonstration",
        source="Synthetic analytic field (built in code)",
        timestamp="",
        resolution="analytic (continuous)",
        label=("CURRENT DATA UNAVAILABLE — using stored DEMONSTRATION current field "
               f"(synthetic, not measured). {reason}"),
        is_live=False,
    )
    return CurrentField(prov)


def get_current_field(lat: float, lon: float, prefer_live: bool = True) -> CurrentField:
    """Return a :class:`CurrentField` for the area around (lat, lon).

    Tries the live Open-Meteo Marine API first; on any failure or a null result
    falls back to the clearly-labelled demonstration field.
    """
    if not prefer_live:
        return _demo_field("Live lookup disabled for this run.")
    try:
        resp = requests.get(
            _MARINE_URL,
            params={
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "current": "ocean_current_velocity,ocean_current_direction",
            },
            headers={"User-Agent": "SonarEye-SIH/1.0 (cleanup route planner)"},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        cur = data.get("current", {})
        vel_kmh = cur.get("ocean_current_velocity")
        direction = cur.get("ocean_current_direction")
        if vel_kmh is None or direction is None:
            return _demo_field("Live model returned no current for this location.")
        speed_ms = float(vel_kmh) / 3.6
        vec = _mk_vector(speed_ms, float(direction))
        ts = cur.get("time", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M"))
        prov = CurrentProvenance(
            kind="live-forecast",
            source="Open-Meteo Marine API (ocean-current forecast)",
            timestamp=str(ts),
            resolution="global ocean model, single-point sample",
            label=(f"Live current: FORECAST from Open-Meteo Marine API, "
                   f"sampled {ts} UTC. Not a real-time measurement."),
            is_live=True,
        )
        return CurrentField(prov, uniform=vec)
    except (requests.RequestException, ValueError, KeyError) as exc:
        return _demo_field(f"Live lookup failed ({type(exc).__name__}).")
