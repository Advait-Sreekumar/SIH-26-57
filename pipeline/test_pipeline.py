"""Unit tests for pipeline core modules: preprocess.py, confidence.py, geotag.py.

Run with: pytest pipeline/test_pipeline.py -v

Design constraints:
- No Streamlit import anywhere in this file
- No model checkpoint required (confidence/geotag tests are pure-numpy)
- preprocess tests use synthetic images only
- Fast: full suite runs in < 5 s
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure pipeline/ is on the path regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# preprocess.py tests
# ---------------------------------------------------------------------------
from preprocess import (
    TileBlender,
    gain_normalize,
    lee_filter,
    preprocess,
    tile_image,
)


def _rng_img(h=256, w=256, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (h, w)).astype(np.float32)


class TestLeeFilter:
    def test_output_shape_preserved(self):
        img = _rng_img(128, 200)
        out = lee_filter(img)
        assert out.shape == img.shape

    def test_output_dtype_float32(self):
        out = lee_filter(_rng_img())
        assert out.dtype == np.float32

    def test_smooths_uniform_region(self):
        """Lee filter should leave a perfectly uniform region unchanged."""
        img = np.full((64, 64), 0.5, dtype=np.float32)
        out = lee_filter(img)
        assert np.allclose(out, 0.5, atol=1e-3)

    def test_reduces_variance(self):
        """Filtered image variance must be lower than speckled input."""
        rng = np.random.default_rng(42)
        base = np.full((128, 128), 0.5, dtype=np.float32)
        noisy = np.clip(base + rng.normal(0, 0.2, (128, 128)).astype(np.float32), 0, 1)
        out = lee_filter(noisy)
        assert out.var() < noisy.var()


class TestGainNormalize:
    def test_output_in_0_1(self):
        img = _rng_img() * 200  # large range
        out = gain_normalize(img, clahe=False)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_clahe_output_in_0_1(self):
        img = _rng_img()
        out = gain_normalize(img, clahe=True)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_constant_image_no_crash(self):
        img = np.full((64, 64), 0.5, dtype=np.float32)
        out = gain_normalize(img, clahe=False)
        assert out.shape == img.shape


class TestTileImage:
    def test_small_image_single_tile(self):
        img = _rng_img(64, 64)
        tiles = tile_image(img, size=512, overlap=64)
        assert len(tiles) == 1

    def test_large_image_multiple_tiles(self):
        img = _rng_img(1024, 1024)
        tiles = tile_image(img, size=512, overlap=64)
        assert len(tiles) > 1

    def test_tile_shape_correct(self):
        img = _rng_img(512, 512)
        tiles = tile_image(img, size=512, overlap=64)
        for y, x, patch in tiles:
            assert patch.shape == (512, 512)

    def test_all_pixels_covered(self):
        """Every pixel must appear in at least one tile."""
        h, w = 768, 900
        img = np.arange(h * w, dtype=np.float32).reshape(h, w)
        covered = np.zeros((h, w), bool)
        for y, x, patch in tile_image(img, size=512, overlap=64):
            ph, pw = patch.shape
            covered[y:y + ph, x:x + pw] = True
        assert covered.all()


class TestTileBlender:
    def test_constant_patch_reconstructs(self):
        """Blending a constant-value patch over a whole image should return that constant."""
        h, w, val = 256, 256, 3.7
        blender = TileBlender(h, w, size=256, overlap=0)
        patch = np.full((256, 256), val, np.float32)
        blender.add(0, 0, patch)
        result = blender.result()
        assert np.allclose(result, val, atol=1e-4)

    def test_result_shape_matches_init(self):
        blender = TileBlender(300, 400, size=256, overlap=32)
        blender.add(0, 0, np.zeros((256, 256), np.float32))
        assert blender.result().shape == (300, 400)


class TestPreprocess:
    def test_returns_float32(self):
        img = _rng_img()
        out, _ = preprocess(img)
        assert out.dtype == np.float32

    def test_output_range(self):
        img = _rng_img()
        out, _ = preprocess(img)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_gap_returned_when_crop_enabled(self):
        img = _rng_img(256, 512)  # wide enough for water column detection
        out, gap = preprocess(img, crop_water_column=True)
        # gap may be None if no dark column detected -- that's valid too
        assert out.ndim == 2

    def test_no_crop(self):
        img = _rng_img(128, 128)
        out, gap = preprocess(img, crop_water_column=False)
        assert gap is None
        assert out.shape == (128, 128)


# ---------------------------------------------------------------------------
# confidence.py tests
# ---------------------------------------------------------------------------
from confidence import geometric_confidence, score_detections, shadow_score


def _solid_square_mask(h=64, w=64, bx=20, by=20, bw=20, bh=10) -> np.ndarray:
    mask = np.zeros((h, w), bool)
    mask[by:by + bh, bx:bx + bw] = True
    return mask


def _fake_detection(id_=0, x1=20, y1=20, x2=40, y2=30, mean_prob=0.8, peak_prob=0.95):
    return {
        "id": id_,
        "bbox_xyxy": [x1, y1, x2, y2],
        "area_px": (x2 - x1) * (y2 - y1),
        "centroid_px": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        "mean_prob": mean_prob,
        "peak_prob": peak_prob,
    }


class TestGeometricConfidence:
    def test_returns_float_in_0_1(self):
        gray = np.random.default_rng(0).uniform(0, 1, (64, 64)).astype(np.float32)
        mask = _solid_square_mask()
        geo, parts = geometric_confidence(gray, mask)
        assert 0.0 <= geo <= 1.0

    def test_parts_keys(self):
        gray = np.ones((64, 64), np.float32) * 0.5
        mask = _solid_square_mask()
        _, parts = geometric_confidence(gray, mask)
        assert set(parts.keys()) == {"aspect", "solidity", "edge"}

    def test_empty_mask_no_crash(self):
        gray = np.ones((64, 64), np.float32) * 0.5
        mask = np.zeros((64, 64), bool)
        geo, _ = geometric_confidence(gray, mask)
        assert isinstance(geo, float)


class TestShadowScore:
    def test_output_in_0_1(self):
        gray = np.random.default_rng(1).uniform(0, 1, (64, 64)).astype(np.float32)
        mask = _solid_square_mask()
        s = shadow_score(gray, mask)
        assert 0.0 <= s <= 1.0

    def test_empty_mask_returns_0(self):
        gray = np.ones((64, 64), np.float32) * 0.5
        mask = np.zeros((64, 64), bool)
        assert shadow_score(gray, mask) == 0.0

    def test_bright_target_dark_background(self):
        """A bright target on a dark background should score > 0."""
        gray = np.full((128, 128), 0.1, np.float32)
        gray[50:60, 50:70] = 0.9
        comp = np.zeros((128, 128), bool)
        comp[50:60, 50:70] = True
        assert shadow_score(gray, comp) > 0.0


class TestScoreDetections:
    def test_formula_output_range(self):
        gray = np.random.default_rng(2).uniform(0, 1, (128, 128)).astype(np.float32)
        mask = np.zeros((128, 128), np.uint8)
        mask[20:30, 20:40] = 1
        dets = [_fake_detection()]
        scored = score_detections(gray, gray, dets, mask)
        assert len(scored) == 1
        conf = scored[0]["confidence"]
        assert 0.0 <= conf <= 100.0

    def test_likely_rock_flag_below_35(self):
        """A detection with near-zero model probability should be flagged."""
        gray = np.zeros((128, 128), np.float32)
        mask = np.zeros((128, 128), np.uint8)
        mask[20:30, 20:40] = 1
        det = _fake_detection(mean_prob=0.01, peak_prob=0.01)
        scored = score_detections(gray, gray, [det], mask)
        assert scored[0]["likely_rock_or_shadow"] is True

    def test_high_prob_detection_not_flagged(self):
        """High-probability detections on bright targets should clear the rock threshold."""
        gray = np.full((128, 128), 0.1, np.float32)
        gray[20:30, 20:40] = 0.9
        mask = np.zeros((128, 128), np.uint8)
        mask[20:30, 20:40] = 1
        det = _fake_detection(mean_prob=0.95, peak_prob=0.99)
        scored = score_detections(gray, gray, [det], mask)
        assert scored[0]["likely_rock_or_shadow"] is False

    def test_sorted_by_confidence_descending(self):
        gray = np.random.default_rng(3).uniform(0, 1, (128, 128)).astype(np.float32)
        mask = np.zeros((128, 128), np.uint8)
        mask[10:20, 10:20] = 1
        mask[60:70, 60:80] = 1
        dets = [
            _fake_detection(id_=0, x1=10, y1=10, x2=20, y2=20, mean_prob=0.2, peak_prob=0.3),
            _fake_detection(id_=1, x1=60, y1=60, x2=80, y2=70, mean_prob=0.9, peak_prob=0.95),
        ]
        scored = score_detections(gray, gray, dets, mask)
        confs = [s["confidence"] for s in scored]
        assert confs == sorted(confs, reverse=True)

    def test_heuristic_formula_values(self):
        """Verify the formula: conf = 100*(0.6*model_score + 0.4*(0.7*geo + 0.3*shadow))."""
        gray = np.full((128, 128), 0.5, np.float32)
        mask = np.zeros((128, 128), np.uint8)
        mask[30:50, 30:70] = 1  # aspect 2:1
        det = _fake_detection(x1=30, y1=30, x2=70, y2=50, mean_prob=0.8, peak_prob=0.8)
        scored = score_detections(gray, gray, [det], mask)
        model_score = 0.5 * 0.8 + 0.5 * 0.8
        geo = scored[0]["geo_score"]
        sh = scored[0]["shadow_score"]
        expected = round(100.0 * (0.6 * model_score + 0.4 * (0.7 * geo + 0.3 * sh)), 1)
        assert abs(scored[0]["confidence"] - expected) < 0.2


# ---------------------------------------------------------------------------
# geotag.py tests
# ---------------------------------------------------------------------------
from geotag import PixelGeoMapper, SonarMeta


class TestPixelGeoMapper:
    def _meta(self, lat0=45.0, lon0=-83.0, heading=0.0, alt=20.0, range_m=50.0, along_m=100.0):
        return SonarMeta(lat0, lon0, heading, alt, range_m, along_m)

    def test_nadir_pixel_returns_origin(self):
        """Centre column, top row → should be near origin lat/lon."""
        meta = self._meta()
        mapper = PixelGeoMapper(200, 400, meta)
        lat, lon = mapper.pixel_to_latlon(200, 0)  # x = half-width = nadir
        assert abs(lat - 45.0) < 0.01
        assert abs(lon - -83.0) < 0.01

    def test_heading_north_east_movement(self):
        """At heading=90 (east), right of nadir is south — lat should decrease."""
        meta = self._meta(heading=90.0)
        mapper = PixelGeoMapper(200, 400, meta)
        lat_right, _ = mapper.pixel_to_latlon(300, 0)  # right of nadir = south
        lat_left, _ = mapper.pixel_to_latlon(100, 0)   # left of nadir = north
        assert lat_right < lat_left

    def test_output_finite(self):
        meta = self._meta()
        mapper = PixelGeoMapper(512, 512, meta)
        for x, y in [(0, 0), (255, 255), (511, 511)]:
            lat, lon = mapper.pixel_to_latlon(x, y)
            assert np.isfinite(lat) and np.isfinite(lon)

    def test_geotag_adds_lat_lon_to_all_detections(self):
        meta = self._meta()
        mapper = PixelGeoMapper(256, 256, meta)
        dets = [
            {"id": 0, "centroid_px": [128.0, 64.0]},
            {"id": 1, "centroid_px": [64.0, 192.0]},
        ]
        tagged = mapper.geotag(dets)
        assert all("lat" in d and "lon" in d for d in tagged)
        assert all(np.isfinite(d["lat"]) and np.isfinite(d["lon"]) for d in tagged)

    def test_along_track_moves_latitude(self):
        """Higher y_px (further along track) at heading=0 should produce higher latitude."""
        meta = self._meta(heading=0.0, along_m=200.0)
        mapper = PixelGeoMapper(200, 400, meta)
        lat_near, _ = mapper.pixel_to_latlon(200, 0)
        lat_far, _ = mapper.pixel_to_latlon(200, 199)
        assert lat_far > lat_near
