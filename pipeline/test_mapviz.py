"""Unit tests for the pure map/route helpers in ``mapviz`` (no streamlit)."""
from __future__ import annotations

import mapviz


def _det(det_id, lat, lon, natural=False, conf=70.0):
    return {"id": det_id, "lat": lat, "lon": lon,
            "likely_rock_or_shadow": natural, "confidence": conf}


def test_three_detections_three_points_even_when_close():
    # Two of the three lie ~1 m apart (the real Barge_No_1_05 near-duplicate);
    # they must still be TWO distinct points, not collapsed to one.
    geod = [
        _det(0, 50.1000516, -4.1996705),
        _det(1, 50.0997147, -4.1995958),
        _det(2, 50.0997142, -4.1995829),  # ~1 m from det#1
    ]
    pts = mapviz.build_survey_points(geod, {})
    assert len(pts) == 3
    coords = {(round(p["lat"], 7), round(p["lon"], 7)) for p in pts}
    assert len(coords) == 3  # all distinct


def test_points_drop_only_missing_coords():
    geod = [_det(0, 50.1, -4.2), _det(1, None, None), _det(2, 50.2, -4.3)]
    pts = mapviz.build_survey_points(geod, {})
    assert [p["det_id"] for p in pts] == [0, 2]


def test_colour_by_review_status():
    geod = [_det(0, 50.1, -4.2), _det(1, 50.11, -4.21), _det(2, 50.12, -4.22)]
    reviews = {0: {"action": "confirm"}, 1: {"action": "reject"}}
    pts = {p["det_id"]: p for p in mapviz.build_survey_points(geod, reviews)}
    assert pts[0]["color"] == mapviz.STATUS_HEX["confirm"]
    assert pts[1]["color"] == mapviz.STATUS_HEX["reject"]
    assert pts[2]["color"] == mapviz.STATUS_HEX["unreviewed"]


def test_natural_feature_overrides_status_colour():
    geod = [_det(0, 50.1, -4.2, natural=True)]
    pts = mapviz.build_survey_points(geod, {0: {"action": "confirm"}})
    assert pts[0]["color"] == mapviz.NATURAL_HEX


def test_clickable_list_matches_detection_count():
    geod = [_det(0, 50.1, -4.2), _det(1, None, None), _det(2, 50.2, -4.3)]
    # One control per detection in the survey — including one with no coord.
    assert mapviz.clickable_detection_ids(geod) == [0, 1, 2]
    assert len(mapviz.clickable_detection_ids(geod)) == len(geod)


def test_bounds_includes_every_point():
    pts = [
        {"lat": 50.10, "lon": -4.20},
        {"lat": 50.12, "lon": -4.25},
        {"lat": 50.08, "lon": -4.18},
    ]
    b = mapviz.bounds_of(pts)
    (min_lat, min_lon), (max_lat, max_lon) = b
    for p in pts:
        assert min_lat <= p["lat"] <= max_lat
        assert min_lon <= p["lon"] <= max_lon
    assert min_lat == 50.08 and max_lat == 50.12
    assert min_lon == -4.25 and max_lon == -4.18


def test_bounds_empty_is_none():
    assert mapviz.bounds_of([]) is None


def test_route_first_vertex_is_exactly_port():
    port = (50.3672, -4.1300)  # Plymouth (Sutton Harbour)
    stops = [_det(0, 50.1000, -4.1997), _det(1, 50.0997, -4.1996)]
    vertices, ordered = mapviz.build_route(port, stops)
    assert vertices[0] == [50.3672, -4.1300]
    # [lat, lon] order preserved for every vertex
    assert vertices[0][0] == port[0] and vertices[0][1] == port[1]


def test_route_visits_every_stop_in_nn_order():
    port = (50.3672, -4.1300)
    # det#2 is closer to the port than det#0/#1 -> visited first.
    stops = [
        _det(0, 50.1000, -4.1997),
        _det(1, 50.0997, -4.1996),
        _det(2, 50.2000, -4.1500),
    ]
    vertices, ordered = mapviz.build_route(port, stops)
    # port + one vertex per stop
    assert len(vertices) == len(stops) + 1
    # every stop appears exactly once
    assert sorted(s["id"] for s in ordered) == [0, 1, 2]
    # nearest stop to the port comes first
    assert ordered[0]["id"] == 2
    # each non-port vertex is [lat, lon] of a stop, in visit order
    for v, s in zip(vertices[1:], ordered):
        assert v == [s["lat"], s["lon"]]


def test_route_single_stop():
    port = (50.3672, -4.1300)
    vertices, ordered = mapviz.build_route(port, [_det(0, 50.1, -4.2)])
    assert vertices == [[50.3672, -4.1300], [50.1, -4.2]]
    assert len(ordered) == 1
