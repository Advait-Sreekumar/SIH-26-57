"""Current-aware cleanup-route planning for SonarEye.

This package is a *decision-support* route planner: given a confirmed anomaly
position (from the nav-aware geolocation in ``pipeline/geolocate.py``), it
recommends a departure PORT and a marine route from that port to the anomaly,
taking ocean-current assistance/resistance into account.

It is deliberately split into independent services so no routing logic lives in
the Streamlit UI (see PART 20 of the spec):

    ports.py        real ports from a live geographic API (OSM/Overpass)
    currents.py     ocean-current vectors: live API + labelled offline fallback
    coastline.py    coarse land polygons for land masking (bundled Natural Earth)
    marine_graph.py navigable water grid graph over the area of interest
    optimizer.py    current-aware edge cost + A* + multi-route + port ranking

What this is NOT (see PART 23 — do not oversell):
    * not autonomous navigation, not collision avoidance,
    * not a certified navigational route,
    * not a guaranteed fastest / shortest / most fuel-efficient route.

Every geographic value returned traces to a real calculation or is explicitly
labelled as demonstration data (see ``CurrentField.provenance``).
"""
from __future__ import annotations

from .plan import RoutePlan, RouteSegment, plan_cleanup

__all__ = ["RoutePlan", "RouteSegment", "plan_cleanup"]
