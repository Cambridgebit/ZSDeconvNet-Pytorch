import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageFilter

from zsdeconv.models import DirectDeconvNet
from zsdeconv.utils import crop_pad, load_image, percentile_norm, reflect_pad, save_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rolling_ball_subtract(x, radius):
    if radius <= 0:
        return x
    img = Image.fromarray((percentile_norm(x) * 255).astype(np.uint8), mode="L")
    size = max(3, int(radius) * 2 + 1)
    if size % 2 == 0:
        size += 1
    background = img.filter(ImageFilter.MinFilter(size=size)).filter(ImageFilter.MaxFilter(size=size))
    bg = np.array(background).astype(np.float32) / 255.0
    y = percentile_norm(x) - bg
    return np.clip(y, 0.0, None).astype(np.float32)


def infer(
    image_path,
    checkpoint_path,
    output_path=None,
    pad_margin=16,
    rolling_ball_radius=0,
):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = DirectDeconvNet(
        base_ch=cfg.get("base_ch", 32),
        depth=cfg.get("depth", 4),
        n_conv=cfg.get("n_conv", 3),
        upsample=cfg.get("upsample", True),
        detail_branch=cfg.get("detail_branch", False),
        detail_scale=cfg.get("detail_scale", 0.2),
        gated_detail=cfg.get("gated_detail", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = percentile_norm(load_image(image_path))
    img_pad, padinfo = reflect_pad(img, multiple=2 ** cfg.get("depth", 4), margin=pad_margin)
    x = torch.from_numpy(img_pad[None, None]).float().to(device)

    with torch.no_grad():
        pred = model(x)

    scale = 2 if cfg.get("upsample", True) else 1
    pred_np = crop_pad(pred.squeeze().cpu().numpy(), padinfo, scale=scale)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(image_path), "output_deconv.tif")
    save_image(output_path, percentile_norm(pred_np))
    print(f"Result saved to {output_path}")

    if rolling_ball_radius > 0:
        root, ext = os.path.splitext(output_path)
        rb_output_path = f"{root}_rb{rolling_ball_radius}{ext or '.tif'}"
        rb_np = rolling_ball_subtract(pred_np, rolling_ball_radius)
        save_image(rb_output_path, percentile_norm(rb_np))
        print(f"Rolling-ball result saved to {rb_output_path}")

    print(f"Input shape: {img.shape}, output shape: {pred_np.shape}")
    return pred_np


def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-shot deconvolution inference on a TIFF image.")
    parser.add_argument("--image", default="datasets/RawSIMData_level_09.tif", help="Input grayscale TIFF.")
    parser.add_argument("--checkpoint", default="best_deconv_multi.pt", help="Model checkpoint.")
    parser.add_argument("--output", default=None, help="Output TIFF path.")
    parser.add_argument("--pad-margin", type=int, default=16)
    parser.add_argument("--rolling-ball-radius", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        pad_margin=args.pad_margin,
        rolling_ball_radius=args.rolling_ball_radius,
    )
