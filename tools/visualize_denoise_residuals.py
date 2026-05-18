import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path):
    arr = np.array(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr


def percentile_norm(x, pmin=0.1, pmax=99.9):
    lo = np.percentile(x, pmin)
    hi = np.percentile(x, pmax)
    return np.clip((x - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)


def spectrum(x):
    f = np.fft.fftshift(np.fft.fft2(x))
    return np.log1p(np.abs(f))


def find_pair(raw_dir, denoised_dir, stem):
    raw = Path(raw_dir) / f"{stem}.tif"
    den = Path(denoised_dir) / f"{stem}_n2v2.tif"
    if not raw.exists():
        raise FileNotFoundError(raw)
    if not den.exists():
        raise FileNotFoundError(den)
    return raw, den


def main():
    parser = argparse.ArgumentParser(description="Visualize residuals between raw and N2V2-denoised TIFFs.")
    parser.add_argument("--raw-dir", default="datasets_full/Microtubule/train_data")
    parser.add_argument("--denoised-dir", default="runs/n2v2_microtubule/denoised")
    parser.add_argument("--stem", default="01", help="File stem, e.g. 01 for 01.tif and 01_n2v2.tif.")
    parser.add_argument("--out", default=None, help="Output PNG path.")
    parser.add_argument("--pmin", type=float, default=0.1)
    parser.add_argument("--pmax", type=float, default=99.9)
    parser.add_argument("--residual-vmax", type=float, default=0.15)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    raw_path, den_path = find_pair(args.raw_dir, args.denoised_dir, args.stem)
    raw = percentile_norm(load_image(raw_path), args.pmin, args.pmax)
    den = percentile_norm(load_image(den_path), args.pmin, args.pmax)
    residual = raw - den
    residual_norm = percentile_norm(residual, 1, 99)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(raw, cmap="gray")
    axes[0, 0].set_title(f"Raw: {raw_path.name}")
    axes[0, 1].imshow(den, cmap="gray")
    axes[0, 1].set_title(f"Denoised: {den_path.name}")
    im = axes[0, 2].imshow(residual, cmap="RdBu_r", vmin=-args.residual_vmax, vmax=args.residual_vmax)
    axes[0, 2].set_title("Raw - Denoised")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(residual_norm, cmap="gray")
    axes[1, 0].set_title("Residual contrast-stretched")
    axes[1, 1].imshow(spectrum(raw), cmap="magma")
    axes[1, 1].set_title("Raw spectrum")
    axes[1, 2].imshow(spectrum(residual), cmap="magma")
    axes[1, 2].set_title("Residual spectrum")

    for ax in axes.ravel():
        ax.axis("off")

    mean_abs = float(np.mean(np.abs(residual)))
    std = float(np.std(residual))
    fig.suptitle(f"Residual mean_abs={mean_abs:.5f}, std={std:.5f}", y=0.98)
    plt.tight_layout()

    out = args.out
    if out is None:
        out = str(Path(args.denoised_dir) / f"{args.stem}_residual_check.png")
    plt.savefig(out, dpi=180)
    print(f"Saved residual visualization to {out}")


if __name__ == "__main__":
    main()
