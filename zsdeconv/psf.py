import numpy as np

from .utils import load_image


def load_psf(path, size=25, center="image"):
    if size % 2 == 0:
        raise ValueError("PSF size must be odd so convolution padding stays centered.")

    psf = load_image(path)
    h, w = psf.shape
    if center == "peak":
        cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    elif center == "image":
        cy, cx = h // 2, w // 2
    else:
        raise ValueError(f"Unknown PSF center mode: {center}")

    half = size // 2
    padded = np.pad(psf, ((half, half), (half, half)), mode="constant")
    cy += half
    cx += half
    psf = padded[cy - half : cy + half + 1, cx - half : cx + half + 1]

    total = psf.sum()
    if total <= 0:
        raise ValueError(f"PSF sum must be positive, got {total}.")
    return (psf / total).astype(np.float32)
