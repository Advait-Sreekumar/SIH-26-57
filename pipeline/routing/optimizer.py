"""Current-aware route optimization.

Cost model (documented, physically sensible — spec PART 7):

    The vessel is assumed to steer along each edge's bearing through the water at
    a constant cruising speed ``V`` (m/s). The ocean current at the edge is a
    vector ``c``. The vessel's speed made good ALONG the edge (over ground) is

        SOG_along = V + (c . track_hat)

    i.e. the current's component projected onto the direction of travel. A
    current flowing WITH the track (positive projection) speeds the vessel up
    (assist); one flowing AGAINST it slows the vessel down (resist); a purely
    cross-track current has no first-order effect on along-track progress. This
    is why "current in the same direction = good" is NOT applied blindly — only
    the projected component matters, and its sign depends on relative heading.

    Edge travel time = edge_length / max(SOG_along, floor).

Route objectives:
    * ``current_time``   minimize current-aware travel time  (PRIMARY for demo)
    * ``distance``       minimize geometric distance (current ignored)
    * ``min_resistance`` prefer current-assisted water even if slightly longer

All reported values (distance, ETA, assist) are computed from this model; none
are fabricated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx

from .currents import CurrentField
from .geoutil import bearing_to_uv, haversine_m

_SOG_FLOOR_MS = 0.15   # never divide by ~0 if a strong current opposes a slow leg
OBJECTIVES = ("current_time", "distance", "min_resistance")


@dataclass
class Leg:
    lat1: float
    lon1: float
    lat2: float
    lon2: float
    dist_m: float
    bearing_deg: float
    current_speed_ms: float
    current_bearing_deg: float
    along_sog_ms: float
    time_s: float
    assist_ms: float           # + = current helped along track, - = opposed


@dataclass
class RouteSolution:
    legs: list[Leg]
    total_dist_m: float
    total_time_s: float
    mean_assist_ms: float
    objective: str
    feasible: bool
    reason: str = ""


def _along_track_sog(bearing_deg, cur: "CurrentVectorLike", vessel_ms):
    """SOG along ``bearing_deg`` = V + projection of current onto the track."""
    tx, ty = bearing_to_uv(bearing_deg, 1.0)         # unit track vector (E, N)
    proj = cur.u_east_ms * tx + cur.v_north_ms * ty  # current . track_hat
    return vessel_ms + proj, proj


# structural typing helper (CurrentVector from currents.py fits this)
class CurrentVectorLike:
    u_east_ms: float
    v_north_ms: float
    speed_ms: float
    bearing_deg: float


def _edge_weight_factory(graph, field: CurrentField, vessel_ms: float, objective: str):
    def weight(u, v, data):
        la1, lo1 = graph.latlon(u)
        la2, lo2 = graph.latlon(v)
        mid_lat, mid_lon = (la1 + la2) / 2.0, (lo1 + lo2) / 2.0
        cur = field.sample(mid_lat, mid_lon)
        dist = data["dist_m"]
        bearing = data["bearing_deg"]
        sog, proj = _along_track_sog(bearing, cur, vessel_ms)
        sog = max(sog, _SOG_FLOOR_MS)
        if objective == "distance":
            return dist
        if objective == "current_time":
            return dist / sog
        # min_resistance: distance inflated when current opposes, discounted when it assists
        assist_frac = max(-0.9, min(0.9, proj / max(vessel_ms, 0.1)))
        return dist * (1.0 - 0.6 * assist_frac)
    return weight


def solve_route(graph, start_latlon, goal_latlon, field: CurrentField,
                vessel_ms: float, objective: str = "current_time") -> RouteSolution:
    """A* route from start to goal over the marine graph under ``objective``."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}")
    start = graph.nearest_node(*start_latlon)
    goal = graph.nearest_node(*goal_latlon)
    if start is None or goal is None:
        return RouteSolution([], 0, 0, 0, objective, False, "no water nodes in area")

    max_cur = 0.8  # for an admissible time heuristic
    def heuristic(a, b):
        la1, lo1 = graph.latlon(a)
        la2, lo2 = graph.latlon(b)
        d = haversine_m(la1, lo1, la2, lo2)
        if objective == "current_time":
            return d / (vessel_ms + max_cur)
        if objective == "distance":
            return d
        return d * 0.4  # min_resistance: loose but admissible-ish lower bound

    weight = _edge_weight_factory(graph, field, vessel_ms, objective)
    try:
        path = nx.astar_path(graph.G, start, goal, heuristic=heuristic, weight=weight)
    except nx.NetworkXNoPath:
        return RouteSolution([], 0, 0, 0, objective, False,
                             "no navigable water path between port and anomaly")

    legs: list[Leg] = []
    tot_d = tot_t = tot_assist = 0.0
    for a, b in zip(path[:-1], path[1:]):
        la1, lo1 = graph.latlon(a)
        la2, lo2 = graph.latlon(b)
        data = graph.G[a][b]
        cur = field.sample((la1 + la2) / 2.0, (lo1 + lo2) / 2.0)
        sog, proj = _along_track_sog(data["bearing_deg"], cur, vessel_ms)
        sog = max(sog, _SOG_FLOOR_MS)
        t = data["dist_m"] / sog
        legs.append(Leg(la1, lo1, la2, lo2, data["dist_m"], data["bearing_deg"],
                        cur.speed_ms, cur.bearing_deg, sog, t, proj))
        tot_d += data["dist_m"]
        tot_t += t
        tot_assist += proj * data["dist_m"]
    mean_assist = tot_assist / tot_d if tot_d else 0.0
    return RouteSolution(legs, tot_d, tot_t, mean_assist, objective, True)


@dataclass
class PortRanking:
    port: object                 # ports.Port
    solution: RouteSolution
    rank_reason: str


def rank_ports(candidate_ports, anomaly_latlon, graph_builder, field: CurrentField,
               vessel_ms: float):
    """Rank ports by current-aware travel time to the anomaly (PRIMARY objective).

    ``graph_builder(port)`` returns a MarineGraph covering that port + anomaly.
    Returns (ranked_list, recommended_or_None). Nothing is selected by
    straight-line distance; the ranking key is the computed route time.
    """
    scored: list[PortRanking] = []
    for port in candidate_ports:
        graph = graph_builder(port)
        sol = solve_route(graph, (port.lat, port.lon), anomaly_latlon,
                          field, vessel_ms, "current_time")
        if not sol.feasible:
            scored.append(PortRanking(port, sol, f"No navigable route found ({sol.reason})."))
            continue
        scored.append(PortRanking(port, sol, ""))

    feasible = [s for s in scored if s.solution.feasible]
    feasible.sort(key=lambda s: s.solution.total_time_s)
    for i, s in enumerate(feasible):
        hrs = s.solution.total_time_s / 3600.0
        km = s.solution.total_dist_m / 1000.0
        if i == 0:
            s.rank_reason = (f"RECOMMENDED: shortest current-aware travel time "
                             f"({hrs:.1f} h over {km:.1f} km at the set cruising speed).")
        else:
            s.rank_reason = f"Alternative: {hrs:.1f} h / {km:.1f} km current-aware."
    ranked = feasible + [s for s in scored if not s.solution.feasible]
    recommended = feasible[0] if feasible else None
    return ranked, recommended
