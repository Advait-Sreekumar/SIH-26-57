"""Orchestration: turn a confirmed anomaly into a current-aware cleanup route.

This is the single entry point the UI calls. It wires the independent services
together and returns a fully-populated :class:`RoutePlan`; it contains no UI
code and no Streamlit imports (spec PART 20).

Pipeline:  anomaly -> discover real ports -> current field -> rank ports by
current-aware time -> build the recommended route (+ distance & min-resistance
alternatives for the same port) -> assemble a labelled, honest plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone

from . import ports as ports_svc
from .currents import CurrentField, CurrentProvenance, get_current_field
from .marine_graph import MarineGraph
from .optimizer import OBJECTIVES, PortRanking, RouteSolution, rank_ports, solve_route

DEFAULT_VESSEL_KN = 8.0            # documented default cruising speed (PART 15)
KN_TO_MS = 0.514444
ROUTE_CLASS_LABEL = ("Planning / visualization route — decision-support only, "
                     "NOT a certified navigational route.")


@dataclass
class RouteSegment:
    seq: int
    from_lat: float
    from_lon: float
    to_lat: float
    to_lon: float
    dist_km: float
    bearing_deg: float
    mean_current_ms: float
    mean_assist_ms: float


@dataclass
class RoutePlan:
    feasible: bool
    reason: str
    anomaly_lat: float | None
    anomaly_lon: float | None
    vessel_speed_kn: float
    primary_objective: str
    route_class_label: str
    generated_at: str
    warnings: list[str] = dc_field(default_factory=list)
    recommended_port: object = None                 # ports.Port | None
    recommended_reason: str = ""
    ranked_ports: list = dc_field(default_factory=list)   # list[PortRanking]
    routes: dict = dc_field(default_factory=dict)         # objective -> RouteSolution
    segments: list = dc_field(default_factory=list)       # list[RouteSegment] (recommended)
    current_provenance: CurrentProvenance | None = None

    @property
    def recommended_route(self) -> RouteSolution | None:
        return self.routes.get(self.primary_objective)


def _summarize(sol: RouteSolution, n_segments: int = 6) -> list[RouteSegment]:
    """Downsample the leg list into a handful of report-friendly segments."""
    if not sol.legs:
        return []
    legs = sol.legs
    step = max(1, len(legs) // n_segments)
    out: list[RouteSegment] = []
    seq = 0
    for start in range(0, len(legs), step):
        chunk = legs[start:start + step]
        dist = sum(l.dist_m for l in chunk)
        cur = sum(l.current_speed_ms * l.dist_m for l in chunk) / dist if dist else 0.0
        assist = sum(l.assist_ms * l.dist_m for l in chunk) / dist if dist else 0.0
        out.append(RouteSegment(
            seq=seq, from_lat=chunk[0].lat1, from_lon=chunk[0].lon1,
            to_lat=chunk[-1].lat2, to_lon=chunk[-1].lon2,
            dist_km=dist / 1000.0, bearing_deg=chunk[0].bearing_deg,
            mean_current_ms=cur, mean_assist_ms=assist,
        ))
        seq += 1
    return out


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def plan_cleanup(anomaly_lat, anomaly_lon, vessel_speed_kn: float = DEFAULT_VESSEL_KN,
                 prefer_live_current: bool = True, search_radius_km: float = 200.0,
                 grid: int = 55, max_ports: int = 6, progress=None) -> RoutePlan:
    """Plan a current-aware cleanup route to a confirmed anomaly.

    ``anomaly_lat/lon`` must already be a validated, in-water geolocation
    (from ``geolocate.NavTrackGeoMapper``). If they are missing/invalid we
    return an infeasible plan rather than inventing a position.

    ``progress`` is an optional ``callback(stage: str)`` invoked as each real
    stage begins, so a UI can show an honest progress checklist (spec PART 21).
    """
    def _p(stage: str):
        if progress is not None:
            progress(stage)

    warnings: list[str] = []
    vessel_ms = max(0.5, vessel_speed_kn * KN_TO_MS)

    if anomaly_lat is None or anomaly_lon is None:
        return RoutePlan(False, "No valid anomaly coordinate to plan from.",
                         None, None, vessel_speed_kn, "current_time",
                         ROUTE_CLASS_LABEL, _now(),
                         warnings=["Anomaly position could not be reliably determined."])
    if not (-90 <= anomaly_lat <= 90 and -180 <= anomaly_lon <= 180):
        return RoutePlan(False, "Anomaly coordinate outside valid lat/lon range.",
                         anomaly_lat, anomaly_lon, vessel_speed_kn, "current_time",
                         ROUTE_CLASS_LABEL, _now())

    # 1. current field (live w/ labelled offline fallback)
    _p("current")
    field: CurrentField = get_current_field(anomaly_lat, anomaly_lon,
                                             prefer_live=prefer_live_current)
    if not field.provenance.is_live:
        warnings.append(field.provenance.label)

    # 2. discover real candidate ports
    _p("ports")
    candidates, port_reason, port_source = ports_svc.find_candidate_ports(
        anomaly_lat, anomaly_lon, radius_km=search_radius_km, limit=max_ports)
    if not candidates:
        return RoutePlan(False, port_reason or "No candidate ports found.",
                         anomaly_lat, anomaly_lon, vessel_speed_kn, "current_time",
                         ROUTE_CLASS_LABEL, _now(),
                         warnings=warnings + [port_reason], current_provenance=field.provenance)
    if port_source:
        warnings.append(port_source)

    # 3. rank ports by current-aware travel time (never by straight-line distance)
    _p("ranking")
    def graph_builder(port):
        return MarineGraph.covering([(port.lat, port.lon), (anomaly_lat, anomaly_lon)],
                                    rows=grid, cols=grid, mask_land=True)

    ranked, recommended = rank_ports(candidates, (anomaly_lat, anomaly_lon),
                                     graph_builder, field, vessel_ms)
    if recommended is None:
        return RoutePlan(False, "No navigable route found from any candidate port.",
                         anomaly_lat, anomaly_lon, vessel_speed_kn, "current_time",
                         ROUTE_CLASS_LABEL, _now(), warnings=warnings,
                         ranked_ports=ranked, current_provenance=field.provenance)

    # 4. build all three objective routes for the recommended port
    _p("routing")
    port = recommended.port
    graph = graph_builder(port)
    routes: dict[str, RouteSolution] = {}
    for obj in OBJECTIVES:
        routes[obj] = solve_route(graph, (port.lat, port.lon),
                                  (anomaly_lat, anomaly_lon), field, vessel_ms, obj)

    if not graph.info.land_masked:
        warnings.append("Coastline data unavailable — route not checked against land.")

    _p("done")
    return RoutePlan(
        feasible=True, reason="", anomaly_lat=anomaly_lat, anomaly_lon=anomaly_lon,
        vessel_speed_kn=vessel_speed_kn, primary_objective="current_time",
        route_class_label=ROUTE_CLASS_LABEL, generated_at=_now(), warnings=warnings,
        recommended_port=port, recommended_reason=recommended.rank_reason,
        ranked_ports=ranked, routes=routes,
        segments=_summarize(routes["current_time"]),
        current_provenance=field.provenance,
    )
