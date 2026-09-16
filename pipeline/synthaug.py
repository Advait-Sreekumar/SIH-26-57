import cv2
import numpy as np


def speckle_noise(img, look=1.0):
    noise = np.random.gamma(shape=look, scale=1.0 / look, size=img.shape).astype(np.float32)
    return np.clip(img * noise, 0, 1)


def synthesize_shadow(img, n_shadows=2, strength=0.75, direction=None):
    h, w = img.shape
    out = img.copy()
    if direction is None:
        direction = np.random.uniform(0, 2 * np.pi)
    dx, dy = np.cos(direction), np.sin(direction)
    for _ in range(n_shadows):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        ax, ay = np.random.randint(h // 8, h // 3), np.random.randint(h // 10, h // 5)
        shadow = np.zeros((h, w), np.float32)
        cv2.ellipse(
            shadow,
            (int(cx + dx * ax * 0.9), int(cy + dy * ay * 0.9)),
            (ax, ay),
            np.degrees(np.arctan2(dy, dx)),
            0,
            360,
            1.0,
            -1,
        )
        shadow = cv2.GaussianBlur(shadow, (0, 0), max(ax, ay) / 4.0)
        out = out * (1 - strength * shadow)
    return np.clip(out, 0, 1)


def radial_distortion(img, k1=-0.15, k2=0.05):
    h, w = img.shape
    cx, cy = w / 2, h / 2
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    nx = (xs - cx) / cx
    ny = (ys - cy) / cy
    r2 = nx * nx + ny * ny
    f = 1 + k1 * r2 + k2 * r2 * r2
    map_x = f * nx * cx + cx
    map_y = f * ny * cy + cy
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def heave_pitch_roll(img, max_shift=12, max_shear=0.06):
    h, w = img.shape
    tx = np.random.uniform(-max_shift, max_shift)
    ty = np.random.uniform(-max_shift, max_shift)
    shear = np.random.uniform(-max_shear, max_shear)
    m = np.float32([[1, shear, tx], [0, 1, ty]])
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_sonar_synth(img, p_speckle=0.8, p_shadow=0.5, p_radial=0.3, p_heave=0.5):
    out = img.astype(np.float32)
    if np.random.rand() < p_speckle:
        out = speckle_noise(out, look=np.random.uniform(0.5, 3.0))
    if np.random.rand() < p_shadow:
        out = synthesize_shadow(out)
    if np.random.rand() < p_radial:
        out = radial_distortion(out, k1=np.random.uniform(-0.25, 0.1))
    if np.random.rand() < p_heave:
        out = heave_pitch_roll(out)
    return np.clip(out, 0, 1)
