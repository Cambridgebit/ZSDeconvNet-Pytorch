import argparse
import os
from pathlib import Path

import torch

from zsdeconv.n2v2 import N2V2UNet, list_tifs
from zsdeconv.utils import crop_pad, load_image, percentile_norm, reflect_pad, save_uint16_tiff


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denoise_one(model, image_path, output_path, pmin, pmax, pad_margin=16):
    img = percentile_norm(load_image(image_path), pmin=pmin, pmax=pmax)
    img_pad, padinfo = reflect_pad(img, multiple=8, margin=pad_margin)
    x = torch.from_numpy(img_pad[None, None]).float().to(device)
    with torch.no_grad():
        pred = model(x)
    pred_np = crop_pad(pred.squeeze().cpu().numpy(), padinfo)
    save_uint16_tiff(pred_np, output_path, mode="clip")
    print(f"Saved denoised image to {output_path}")


def infer(args):
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = N2V2UNet(base_ch=cfg.get("base_ch", 48), depth=cfg.get("depth", 3)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    paths = list_tifs(args.image)
    if len(paths) > 1:
        output_dir = args.output or str(Path(args.image).with_name(f"{Path(args.image).name}_n2v2"))
        os.makedirs(output_dir, exist_ok=True)
        for path in paths:
            out = Path(output_dir) / f"{path.stem}_n2v2.tif"
            denoise_one(model, path, out, cfg.get("pmin", 0.1), cfg.get("pmax", 99.9), args.pad_margin)
    else:
        output = args.output
        if output is None:
            output = str(paths[0].with_name(f"{paths[0].stem}_n2v2.tif"))
        denoise_one(model, paths[0], output, cfg.get("pmin", 0.1), cfg.get("pmax", 99.9), args.pad_margin)


def parse_args():
    parser = argparse.ArgumentParser(description="Run N2V2-style denoising inference.")
    parser.add_argument("--image", default="datasets/Cell_001/RawSIMData/RawSIMData_level_01.tif", help="TIFF file or directory.")
    parser.add_argument("--checkpoint", default="runs/n2v2_microtubule/best_n2v2.pt")
    parser.add_argument("--output", default=None, help="Output TIFF path, or directory when --image is a directory.")
    parser.add_argument("--pad-margin", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    infer(parse_args())
