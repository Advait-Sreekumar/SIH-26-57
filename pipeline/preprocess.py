import cv2
import numpy as np
from scipy.ndimage import uniform_filter


def lee_filter(img, window=7, eps=1e-6):
    img = img.astype(np.float32)
    mean = uniform_filter(img, window)
    sqr_mean = uniform_filter(img * img, window)
    var = sqr_mean - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    ci = std / np.maximum(mean, eps)
    ci2 = ci * ci
    w = np.maximum(0.0, 1.0 - ci2 / (ci2 + 1e-3))
    return (mean + w * (img - mean)).astype(np.float32)


def median_denoise(img, ksize=3):
    return cv2.medianBlur((img * 255).astype(np.uint8), ksize).astype(np.float32) / 255.0


def water_column_crop(img, center_frac=0.30, dark_quantile=0.15, min_gap_cols=5):
    h, w = img.shape
    x0, x1 = int(w * (0.5 - center_frac / 2)), int(w * (0.5 + center_frac / 2))
    col_med = np.median(img[:, x0:x1], axis=0)
    thr = np.quantile(col_med, dark_quantile)
    dark = col_med <= thr
    best_start, best_len, cur_start, cur_len = -1, 0, 0, 0
    for i, d in enumerate(dark):
        if d:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    if best_len < min_gap_cols:
        return img, 0, w
    gx0 = x0 + best_start
    gx1 = gx0 + best_len
    keep = np.ones(w, dtype=bool)
    keep[gx0:gx1] = False
    return img[:, keep], gx0, gx1


def gain_normalize(img, low_p=1.0, high_p=99.5, clahe=True):
    lo, hi = np.percentile(img, low_p), np.percentile(img, high_p)
    out = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    if clahe:
        eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = eq.apply((out * 255).astype(np.uint8)).astype(np.float32) / 255.0
    return out


def preprocess(img, denoise="lee", lee_window=7, median_k=3, crop_water_column=True, normalize=True):
    img = img.astype(np.float32)
    if denoise == "lee":
        img = lee_filter(img, lee_window)
    elif denoise == "median":
        img = median_denoise(img, median_k)
    gap = None
    if crop_water_column:
        img, gx0, gx1 = water_column_crop(img)
        gap = (gx0, gx1)
    if normalize:
        img = gain_normalize(img)
    return img, gap


def tile_image(img, size=512, overlap=64):
    h, w = img.shape
    step = size - overlap
    tiles = []
    ys = list(range(0, max(h - size, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - size, 0) + 1, step)) or [0]
    if ys[-1] + size < h:
        ys.append(h - size)
    if xs[-1] + size < w:
        xs.append(w - size)
    for y in ys:
        for x in xs:
            tiles.append((y, x, img[y : y + size, x : x + size]))
    return tiles


class TileBlender:
    def __init__(self, h, w, size=512, overlap=64):
        self.acc = np.zeros((h, w), np.float32)
        self.wsum = np.zeros((h, w), np.float32)
        self.size = size
        self.overlap = overlap
        ramp = np.ones(size, np.float32)
        if overlap > 0:
            r = np.linspace(0.05, 1.0, overlap, dtype=np.float32)
            ramp[:overlap] = r
            ramp[-overlap:] = r[::-1]
        self.ramp = ramp

    def add(self, y, x, patch_logits):
        wr = self.ramp[:, None] * self.ramp[None, :]
        self.acc[y : y + self.size, x : x + self.size] += patch_logits * wr
        self.wsum[y : y + self.size, x : x + self.size] += wr

    def result(self):
        return self.acc / np.maximum(self.wsum, 1e-6)
