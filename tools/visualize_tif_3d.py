import argparse

import numpy as np
from PIL import Image


def load_tif(path):
    arr = np.array(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr


def normalize(x):
    return (x - x.min()) / max(float(x.max() - x.min()), 1e-8)


def main():
    parser = argparse.ArgumentParser(description="Visualize a TIFF image as a 3D height map.")
    parser.add_argument("image", help="Input TIFF path.")
    parser.add_argument("--stride", type=int, default=4, help="Sample every N pixels for faster rendering.")
    parser.add_argument("--height-scale", type=float, default=80.0, help="Vertical scale multiplier.")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap.")
    parser.add_argument("--elev", type=float, default=55.0, help="Camera elevation.")
    parser.add_argument("--azim", type=float, default=-60.0, help="Camera azimuth.")
    parser.add_argument("--save", default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    img = normalize(load_tif(args.image))
    stride = max(1, args.stride)
    z = img[::stride, ::stride] * args.height_scale
    h, w = z.shape
    y, x = np.mgrid[:h, :w]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap=args.cmap, linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_title(args.image)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Intensity")
    ax.set_box_aspect((w, h, args.height_scale))
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"Saved 3D view to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
