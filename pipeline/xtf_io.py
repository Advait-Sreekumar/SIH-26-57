from pathlib import Path

import cv2
import numpy as np

M_PER_DEG_LAT = 111320.0


def read_xtf(path):
    import pyxtf

    file_hdr, packets = pyxtf.xtf_read(str(path))
    sonar = packets.get(pyxtf.XTFHeaderType.sonar, [])
    if not sonar:
        raise ValueError(f"No sonar pings found in {path}")

    rows = []
    nav = []
    for ping_hdr in sonar:
        chan_list = list(zip(ping_hdr.ping_chan_headers, ping_hdr.data))
        chans = {}
        for chan_hdr, data in chan_list:
            chans[int(chan_hdr.ChannelNumber)] = data
        if 0 not in chans and 1 not in chans:
            continue
        port = np.abs(chans.get(0, np.array([0.0])))
        stbd = np.abs(chans.get(1, np.array([0.0])))
        n = max(len(port), len(stbd))
        row = np.zeros(2 * n, np.float32)
        row[:n] = port[:n] if len(port) else 0
        row[n:] = stbd[:n] if len(stbd) else 0
        rows.append(row)

        lat = float(ping_hdr.SensorYcoordinate)
        lon = float(ping_hdr.SensorXcoordinate)
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
            }
        )

    max_val = max(np.max(r) for r in rows) or 1.0
    waterfall = np.stack([r / max_val for r in rows]).astype(np.float32)
    return waterfall, nav, {"sonar_name": file_hdr.SonarName, "n_pings": len(rows), "n_channels": 2}


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def slant_range_correct(waterfall, altitude_m, range_m):
    half = waterfall.shape[1] // 2
    out = np.zeros_like(waterfall)
    idx = np.arange(half, dtype=np.float32)
    slant = idx / max(half - 1, 1) * range_m
    ground = np.sqrt(np.maximum(slant**2 - altitude_m**2, 0.0))
    gmax = ground[-1] if ground[-1] > 0 else range_m
    grid = np.linspace(0, gmax, half, dtype=np.float32)
    for side in (0, 1):
        cols = waterfall[:, side * half : (side + 1) * half]
        for r in range(waterfall.shape[0]):
            out[r, side * half : (side + 1) * half] = np.interp(grid, ground, cols[r])
    return out


def nav_to_meta(nav, waterfall_shape):
    from geotag import SonarMeta

    mid = nav[len(nav) // 2]
    lat0, lon0 = mid["lat"], mid["lon"]
    a, b = nav[0], nav[-1]
    heading = float(np.degrees(np.arctan2(b["lon"] - a["lon"], b["lat"] - a["lat"]))) % 360.0
    if abs(b["lat"] - a["lat"]) + abs(b["lon"] - a["lon"]) > 1e-12:
        heading = float(
            np.degrees(np.arctan2(haversine_step_east(a, b), haversine_step_north(a, b)))
        ) % 360.0
    total = 0.0
    for p, q in zip(nav[:-1], nav[1:]):
        total += haversine_m(p["lat"], p["lon"], q["lat"], q["lon"])
    along = total * (waterfall_shape[0] / max(len(nav) - 1, 1))
    range_m = float(np.median([n["range_m"] for n in nav])) or 50.0
    alt = float(np.median([n["altitude_m"] for n in nav])) or 15.0
    return SonarMeta(lat0=lat0, lon0=lon0, heading_deg=heading, altitude_m=alt, range_m=range_m, along_track_m=along)


def haversine_step_north(a, b):
    return (b["lat"] - a["lat"]) * M_PER_DEG_LAT


def haversine_step_east(a, b):
    return (b["lon"] - a["lon"]) * M_PER_DEG_LAT * np.cos(np.radians(a["lat"]))


def load_input(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        return img, None, {"type": "image", "name": path.name}
    if ext == ".xtf":
        waterfall, nav, info = read_xtf(path)
        info.update({"type": "xtf", "name": path.name, "nav": nav})
        return waterfall, nav, info
    raise ValueError(f"Unsupported input: {path}")
