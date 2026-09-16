import json
from pathlib import Path

import numpy as np

M_PER_DEG_LAT = 111320.0


def read_xtf_navigation(xtf_path, max_pings=None):
    import pyxtf

    _, packets = pyxtf.xtf_read(str(xtf_path))
    pings = packets.get(pyxtf.XTFHeaderType.sonar, [])
    if max_pings:
        pings = pings[:max_pings]
    nav = []
    for ping_hdr in pings:
        lat, lon = float(ping_hdr.SensorYcoordinate), float(ping_hdr.SensorXcoordinate)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        alt = float(ping_hdr.SensorPrimaryAltitude)
        if alt <= 0:
            alt = float(ping_hdr.SensorDepth)
        nav.append(
            {
                "ping": int(ping_hdr.PingNumber),
                "lat": lat,
                "lon": lon,
                "heading_deg": float(ping_hdr.SensorHeading),
                "altitude_m": alt,
                "range_m": float(ping_hdr.RangeToFish),
                "sound_speed": float(ping_hdr.SoundVelocity or 1500.0),
            }
        )
    return nav


class SonarMeta:
    def __init__(self, lat0, lon0, heading_deg=0.0, altitude_m=20.0, range_m=50.0, along_track_m=100.0):
        self.lat0 = lat0
        self.lon0 = lon0
        self.heading_deg = heading_deg
        self.altitude_m = altitude_m
        self.range_m = range_m
        self.along_track_m = along_track_m


class PixelGeoMapper:
    def __init__(self, img_h, img_w, meta):
        self.h = img_h
        self.w = img_w
        self.meta = meta
        half = img_w / 2.0
        self.m_per_px_across = meta.range_m / max(half, 1.0)
        self.m_per_px_along = meta.along_track_m / max(img_h, 1.0)
        th = np.radians(meta.heading_deg)
        self.east = np.array([np.sin(th), np.cos(th)])
        self.north = np.array([np.cos(th), -np.sin(th)])

    def pixel_to_local_m(self, x_px, y_px):
        across = (x_px - self.w / 2.0) * self.m_per_px_across
        slant = abs(across) * 1.0
        ground = np.sqrt(max(slant**2 - self.meta.altitude_m**2, 0.0))
        ground = ground if ground > 0 else abs(across)
        ground = np.sign(across) * ground
        along = y_px * self.m_per_px_along
        return ground * self.east[0] + along * self.east[1], ground * self.north[0] + along * self.north[1]

    def pixel_to_latlon(self, x_px, y_px):
        e, n = self.pixel_to_local_m(x_px, y_px)
        lat = self.meta.lat0 + n / M_PER_DEG_LAT
        lon = self.meta.lon0 + e / (M_PER_DEG_LAT * np.cos(np.radians(self.meta.lat0)))
        return float(lat), float(lon)

    def geotag(self, detections):
        out = []
        for d in detections:
            cx, cy = d["centroid_px"]
            lat, lon = self.pixel_to_latlon(cx, cy)
            out.append({**d, "lat": round(lat, 7), "lon": round(lon, 7)})
        return out


_ACTION_TO_STATUS = {
    "confirm": "human-confirmed",
    "reject": "human-rejected",
    "uncertain": "human-uncertain",
    "annotate": "human-annotate",
}


def save_report(detections, out_base, run_id=None, reviews=None):
    out_base = Path(out_base)
    reviews = reviews or {}
    records = []
    for d in detections:
        det_id = d["id"]
        rev = reviews.get(det_id, {})
        action = rev.get("action")
        records.append(
            {
                "id": det_id,
                "lat": d.get("lat"),
                "lon": d.get("lon"),
                "bbox_xyxy": d["bbox_xyxy"],
                "area_px": d["area_px"],
                "confidence": d["confidence"],
                "model_prob": round(d["mean_prob"], 4),
                "geo_score": d["geo_score"],
                "shadow_score": d["shadow_score"],
                "likely_rock_or_shadow": d["likely_rock_or_shadow"],
                "review_status": _ACTION_TO_STATUS.get(action, "unreviewed"),
                "review_category": rev.get("category"),
                "review_note": rev.get("note"),
                "reviewed_at": rev.get("reviewed_at"),
            }
        )
    n_reviewed = sum(1 for r in records if r["review_status"] != "unreviewed")
    n_confirmed = sum(1 for r in records if r["review_status"] == "human-confirmed")
    payload = {
        "run_id": run_id,
        "n_detections": len(records),
        "n_reviewed": n_reviewed,
        "n_confirmed": n_confirmed,
        "detections": records,
    }
    json_path = out_base.with_suffix(".json")
    csv_path = out_base.with_suffix(".csv")
    json_path.write_text(json.dumps(payload, indent=2))
    import pandas as pd

    pd.DataFrame(records).to_csv(csv_path, index=False)
    return json_path, csv_path
