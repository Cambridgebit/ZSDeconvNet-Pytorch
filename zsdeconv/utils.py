"""
Utility functions for ZSDeconvNet
"""

import os
import math
import numpy as np
from pathlib import Path
from PIL import Image
import torch


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def percentile_norm(x, pmin=0.1, pmax=99.9, eps=1e-8):
    """
    Percentile normalization

    Args:
        x: Input array
        pmin: Lower percentile
        pmax: Upper percentile
        eps: Epsilon for numerical stability

    Returns:
        Normalized array in [0, 1]
    """
    lo = np.percentile(x, pmin)
    hi = np.percentile(x, pmax)
    return np.clip((x - lo) / max(hi - lo, eps), 0.0, 1.0)


def load_image(path):
    """
    Load image as float32 array

    Args:
        path: Path to image file

    Returns:
        Image as float32 array (grayscale)
    """
    arr = np.array(Image.open(path)).astype(np.float32)
    return arr.mean(axis=2) if arr.ndim == 3 else arr


def save_image(path, x):
    """
    Save image as 16-bit TIFF

    Args:
        path: Output path
        x: Image array (will be normalized to [0, 1])
    """
    x = np.clip(x / max(x.max(), 1e-8), 0.0, 1.0)
    Image.fromarray((x * 65535.0).astype(np.uint16)).save(path)


def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def reflect_pad(x, multiple=16, margin=16):
    """
    Pad image with reflection

    Args:
        x: Input image (H, W)
        multiple: Make dimensions divisible by this
        margin: Additional margin

    Returns:
        Padded image and padding info
    """
    h, w = x.shape
    new_h = int(math.ceil(h / multiple) * multiple)
    new_w = int(math.ceil(w / multiple) * multiple)
    pad_h, pad_w = max(new_h - h, 0), max(new_w - w, 0)
    top, bottom = margin + pad_h // 2, margin + pad_h - pad_h // 2
    left, right = margin + pad_w // 2, margin + pad_w - pad_w // 2
    return np.pad(x, ((top, bottom), (left, right)), mode="reflect"), (
        top,
        bottom,
        left,
        right,
    )


def crop_pad(x, padinfo, scale=1):
    """
    Crop padding from image

    Args:
        x: Padded image
        padinfo: (top, bottom, left, right)
        scale: Scale factor (for upsampled outputs)

    Returns:
        Cropped image
    """
    top, bottom, left, right = [p * scale for p in padinfo]
    h, w = x.shape
    return x[top : h - bottom, left : w - right]
