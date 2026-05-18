import math

import numpy as np
from PIL import Image


def load_image(path):
    arr = np.array(Image.open(path)).astype(np.float32)
    return arr.mean(axis=2) if arr.ndim == 3 else arr


def minmax_norm(x):
    return ((x - x.min()) / max(x.max() - x.min(), 1e-8)).astype(np.float32)


def percentile_norm(x, pmin=0.1, pmax=99.9, eps=1e-8):
    lo = np.percentile(x, pmin)
    hi = np.percentile(x, pmax)
    return np.clip((x - lo) / max(hi - lo, eps), 0.0, 1.0).astype(np.float32)


def reflect_pad(x, multiple=16, margin=16):
    h, w = x.shape
    new_h = int(math.ceil(h / multiple) * multiple)
    new_w = int(math.ceil(w / multiple) * multiple)
    pad_h, pad_w = max(new_h - h, 0), max(new_w - w, 0)
    top = margin + pad_h // 2
    bottom = margin + pad_h - pad_h // 2
    left = margin + pad_w // 2
    right = margin + pad_w - pad_w // 2
    return np.pad(x, ((top, bottom), (left, right)), mode="reflect"), (
        top,
        bottom,
        left,
        right,
    )


def crop_pad(x, padinfo, scale=1):
    top, bottom, left, right = [p * scale for p in padinfo]
    h, w = x.shape
    return x[top : h - bottom, left : w - right]


def save_uint16_tiff(x, path, mode="clip"):
    if mode == "clip":
        y = np.clip(x, 0.0, 1.0)
    elif mode == "minmax":
        y = minmax_norm(x)
    else:
        raise ValueError(f"Unknown save mode: {mode}")
    Image.fromarray((y * 65535).astype(np.uint16)).save(path)


def save_image(path, x):
    y = np.clip(x / max(float(x.max()), 1e-8), 0.0, 1.0)
    Image.fromarray((y * 65535.0).astype(np.uint16)).save(path)
