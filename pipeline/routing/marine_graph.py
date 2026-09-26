"""Navigable marine graph.

Builds a regular lat/lon grid over the area covering the departure port and the
anomaly, drops grid cells that fall on land (coarse Natural Earth coastline),
and connects neighbouring water cells (8-connectivity) into a ``networkx``
graph. Each edge stores its length and compass bearing; the *cost* of traversing
an edge is computed later by the optimizer from the current field, so the graph
itself is current-agnostic and reusable.

Because the coastline is coarse (1:110m), a route on this graph is a
"Planning / visualization route", not a certified navigational route.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from . import coastline
from .geoutil import haversine_m, initial_bearing_deg


@dataclass
class GraphInfo:
    n_nodes: int
    n_water: int
    n_land_masked: int
    land_masked: bool          # was coastline masking actually applied?
    rows: int
    cols: int


class MarineGraph:
    def __init__(self, min_lat, min_lon, max_lat, max_lon, rows=55, cols=55,
                 mask_land=True):
        self.rows, self.cols = int(rows), int(cols)
        self.lats = np.linspace(min_lat, max_lat, self.rows)
        self.lons = np.linspace(min_lon, max_lon, self.cols)
        self.G = nx.Graph()
        self._latlon: dict[tuple[int, int], tuple[float, float]] = {}
        have_coast = coastline.coastline_available()
        self._masking = bool(mask_land and have_coast)
        n_land = 0

        water = np.zeros((self.rows, self.cols), dtype=bool)
        for i in range(self.rows):
            for j in range(self.cols):
                la, lo = float(self.lats[i]), float(self.lons[j])
                on_land = self._masking and coastline.is_land(la, lo)
                if on_land:
                    n_land += 1
                    continue
                water[i, j] = True
                self.G.add_node((i, j))
                self._latlon[(i, j)] = (la, lo)

        for i in range(self.rows):
            for j in range(self.cols):
                if not water[i, j]:
                    continue
                la, lo = self._latlon[(i, j)]
                for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.rows and 0 <= nj < self.cols and water[ni, nj]:
                        la2, lo2 = self._latlon[(ni, nj)]
                        d = haversine_m(la, lo, la2, lo2)
                        b = initial_bearing_deg(la, lo, la2, lo2)
                        self.G.add_edge((i, j), (ni, nj), dist_m=d, bearing_deg=b)

        self.info = GraphInfo(
            n_nodes=self.rows * self.cols,
            n_water=int(water.sum()),
            n_land_masked=n_land,
            land_masked=self._masking,
            rows=self.rows, cols=self.cols,
        )

    def latlon(self, node) -> tuple[float, float]:
        return self._latlon[node]

    def nearest_node(self, lat: float, lon: float):
        """Nearest *water* node to (lat, lon). Returns None if graph is empty."""
        best, best_d = None, float("inf")
        for node, (la, lo) in self._latlon.items():
            d = haversine_m(lat, lon, la, lo)
            if d < best_d:
                best, best_d = node, d
        return best

    @classmethod
    def covering(cls, points, pad_frac=0.35, rows=55, cols=55, mask_land=True):
        """Build a graph whose bbox covers ``points`` (list of (lat, lon)) + padding."""
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        dlat = max(max(lats) - min(lats), 0.05)
        dlon = max(max(lons) - min(lons), 0.05)
        return cls(
            min(lats) - pad_frac * dlat, min(lons) - pad_frac * dlon,
            max(lats) + pad_frac * dlat, max(lons) + pad_frac * dlon,
            rows=rows, cols=cols, mask_land=mask_land,
        )
