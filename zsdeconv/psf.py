"""
PSF (Point Spread Function) utilities
"""

import numpy as np
import torch
from .utils import load_image, percentile_norm


def gaussian_psf(size=25, sigma=2.0):
    """
    Generate Gaussian PSF

    Args:
        size: PSF size (will be odd)
        sigma: Gaussian standard deviation

    Returns:
        Normalized Gaussian PSF
    """
    if size % 2 == 0:
        size += 1
    r = size // 2
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    psf = np.exp(-(x**2 / (2 * sigma**2) + y**2 / (2 * sigma**2)))
    psf = psf.astype(np.float32)
    psf /= np.maximum(psf.sum(), 1e-8)
    return psf


def load_psf_from_file(psf_path, size=25):
    """
    Load real PSF from file, crop center region

    CRITICAL: Do NOT use percentile_norm for PSF!
    It truncates weak signal and peak, breaking energy distribution.

    Args:
        psf_path: Path to PSF file
        size: Output size (default 25x25)

    Returns:
        Normalized PSF
    """
    psf = load_image(psf_path)
    # Note: No percentile_norm!

    h, w = psf.shape
    center_y, center_x = h // 2, w // 2
    half = size // 2

    top = max(0, center_y - half)
    bottom = min(h, center_y + half + 1)
    left = max(0, center_x - half)
    right = min(w, center_x + half + 1)

    psf = psf[top:bottom, left:right]

    if psf.shape[0] < size or psf.shape[1] < size:
        pad_h = size - psf.shape[0]
        pad_w = size - psf.shape[1]
        psf = np.pad(psf, ((0, pad_h), (0, pad_w)), mode="constant")

    # Normalize by sum (energy conservation)
    psf /= np.maximum(psf.sum(), 1e-8)
    return psf


def make_psf_tensor(mode, img, size=25, sigma=None, device=None, psf_path=None):
    """
    Create PSF tensor for training

    Args:
        mode: 'load' (real PSF), 'simulate' (Gaussian), or 'estimate'
        img: Reference image (for size estimation)
        size: PSF size
        sigma: Gaussian sigma (for simulate/estimate)
        device: Torch device
        psf_path: Path to real PSF file

    Returns:
        PSF tensor (1, 1, size, size)
    """
    if mode == "load":
        psf_path = (
            psf_path or "datasets/Microtubule/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif"
        )
        psf = load_psf_from_file(psf_path, size)
    elif mode == "estimate":
        sigma = sigma or 2.0
        psf = gaussian_psf(size, sigma)
    else:  # simulate
        sigma = sigma or 2.0
        psf = gaussian_psf(size, sigma)

    psf_t = torch.from_numpy(psf)[None, None].to(device)
    return psf_t / psf_t.sum().clamp_min(1e-8)
