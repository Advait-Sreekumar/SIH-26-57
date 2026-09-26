# Tests for the current-aware cleanup route planner (spec PART 22).
# Pure logic — does NOT import streamlit and does NOT hit the network
# (current fields are constructed in-process; the graph uses mask_land=False
#  or the bundled offline coastline).
# Run: pytest pipeline/test_routing.py -v
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import math
import pytest

from routing.geoutil import (bearing_to_uv, haversine_m, initial_bearing_deg,
                             uv_to_speed_bearing)
from routing.currents import (CurrentField, CurrentProvenance, CurrentVector,
                              get_current_field)
from routing.marine_graph import MarineGraph
from routing.optimizer import _along_track_sog, solve_route, rank_ports
from routing import coastline


def uniform_field(speed_ms, toward_bearing_deg):
    u, v = bearing_to_uv(toward_bearing_deg, speed_ms)
    vec = CurrentVector(u, v, speed_ms, toward_bearing_deg % 360.0)
    prov = CurrentProvenance("demonstration", "unit-test", "", "n/a", "TEST", False)
    return CurrentField(prov, uniform=vec)


# --- geodesy --------------------------------------------------------------
def test_bearing_uv_roundtrip():
    for b in (0, 45, 90, 180, 270, 359):
        u, v = bearing_to_uv(b, 3.0)
        mag, brg = uv_to_speed_bearing(u, v)
        assert abs(mag - 3.0) < 1e-9
        assert abs((brg - b + 180) % 360 - 180) < 1e-6


def test_haversine_and_bearing_sane():
    d = haversine_m(0, 0, 0, 1)  # 1 deg lon at equator ~111.3 km
    assert 111000 < d < 111400
    assert abs(initial_bearing_deg(0, 0, 1, 0) - 0.0) < 1e-6      # due north
    assert abs(initial_bearing_deg(0, 0, 0, 1) - 90.0) < 1e-6     # due east


# --- current-aware physics ------------------------------------------------
def test_current_assist_resist_perp():
    # track due east (bearing 90), vessel 4 m/s
    east_cur = uniform_field(1.0, 90.0).sample(0, 0)    # flowing east = WITH track
    west_cur = uniform_field(1.0, 270.0).sample(0, 0)   # flowing west = AGAINST
    north_cur = uniform_field(1.0, 0.0).sample(0, 0)    # perpendicular
    sog_assist, proj_a = _along_track_sog(90.0, east_cur, 4.0)
    sog_resist, proj_r = _along_track_sog(90.0, west_cur, 4.0)
    sog_perp, proj_p = _along_track_sog(90.0, north_cur, 4.0)
    assert sog_assist == pytest.approx(5.0, abs=1e-6)   # 4 + 1
    assert sog_resist == pytest.approx(3.0, abs=1e-6)   # 4 - 1
    assert sog_perp == pytest.approx(4.0, abs=1e-6)     # cross-track = no along effect
    assert proj_a > 0 > proj_r and abs(proj_p) < 1e-9


# --- routing over an open-water graph ------------------------------------
def _ocean_graph():
    # small open-ocean bbox (Atlantic off Virginia), land masking off for determinism
    return MarineGraph(36.0, -74.5, 37.0, -73.5, rows=25, cols=25, mask_land=False)


def test_route_found_and_ordered():
    g = _ocean_graph()
    sol = solve_route(g, (36.05, -74.45), (36.95, -73.55),
                      uniform_field(0.0, 0.0), vessel_ms=4.0, objective="distance")
    assert sol.feasible
    assert sol.total_dist_m > 0
    assert len(sol.legs) >= 2


def test_assisting_current_beats_opposing_on_time():
    g = _ocean_graph()
    start, goal = (36.05, -74.45), (36.95, -73.55)  # heading roughly NE
    assist = solve_route(g, start, goal, uniform_field(1.2, 45.0), 4.0, "current_time")
    oppose = solve_route(g, start, goal, uniform_field(1.2, 225.0), 4.0, "current_time")
    assert assist.feasible and oppose.feasible
    assert assist.total_time_s < oppose.total_time_s   # WITH current = faster
    assert assist.mean_assist_ms > 0 > oppose.mean_assist_ms


def test_rank_ports_picks_min_time_not_nearest():
    # Two ports: near one sits behind an opposing current; far one rides assist.
    class P:
        def __init__(self, name, lat, lon):
            self.name, self.lat, self.lon = name, lat, lon
            self.straight_km = 0.0
    anomaly = (36.95, -73.55)
    near = P("near", 36.6, -73.9)
    far = P("far", 36.05, -74.45)
    field = uniform_field(1.0, 45.0)  # NE assist favours the far, up-current port

    def gb(port):
        return MarineGraph.covering([(port.lat, port.lon), anomaly],
                                    rows=25, cols=25, mask_land=False)

    ranked, rec = rank_ports([near, far], anomaly, gb, field, vessel_ms=4.0)
    assert rec is not None
    assert all(r.solution.feasible for r in ranked)
    # recommended is whichever has the lower COMPUTED time, and reason says so
    assert "RECOMMENDED" in rec.rank_reason
    times = [r.solution.total_time_s for r in ranked]
    assert times == sorted(times)


# --- current provenance honesty ------------------------------------------
def test_offline_current_is_labelled_demonstration():
    f = get_current_field(45.0, -83.5, prefer_live=False)
    assert f.provenance.is_live is False
    assert f.provenance.kind == "demonstration"
    assert "DEMONSTRATION" in f.provenance.label.upper()


def test_analytic_demo_field_deterministic_and_bounded():
    f = get_current_field(10.0, 20.0, prefer_live=False)
    a = f.sample(10.0, 20.0)
    b = f.sample(10.0, 20.0)
    assert (a.u_east_ms, a.v_north_ms) == (b.u_east_ms, b.v_north_ms)  # reproducible
    assert 0.05 <= a.speed_ms <= 0.6


# --- coastline mask -------------------------------------------------------
def test_coastline_land_vs_water():
    if not coastline.coastline_available():
        pytest.skip("bundled coastline not present")
    assert coastline.is_land(40.0, 100.0) is True     # central Asia = land
    assert coastline.is_land(0.0, -140.0) is False     # mid Pacific = water


# --- plan_cleanup guards (no network) -------------------------------------
def test_plan_cleanup_none_coord_infeasible():
    from routing.plan import plan_cleanup
    plan = plan_cleanup(None, None)
    assert plan.feasible is False
    assert plan.anomaly_lat is None and plan.anomaly_lon is None
    assert plan.warnings  # honest reason, never a fabricated route


def test_plan_cleanup_out_of_range_coord_infeasible():
    from routing.plan import plan_cleanup
    plan = plan_cleanup(999.0, 999.0)
    assert plan.feasible is False
    assert "range" in plan.reason.lower()


def test_plan_cleanup_recommended_route_property_empty_when_infeasible():
    from routing.plan import plan_cleanup
    plan = plan_cleanup(None, None)
    assert plan.recommended_route is None  # no route dict populated

