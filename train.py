import argparse
import csv
import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from zsdeconv.data import DeconvDataset
from zsdeconv.losses import DeconvLoss
from zsdeconv.models import DirectDeconvNet
from zsdeconv.psf import load_psf
from zsdeconv.utils import load_image, percentile_norm, save_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_loss_row(path, epoch, loss, lr, reg_scale):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["epoch", "loss", "lr", "reg_scale"])
        writer.writerow([epoch, f"{loss:.8f}", f"{lr:.8g}", f"{reg_scale:.6f}"])


def regularization_scale(epoch, total_epochs, warmup_epochs=5, final_scale=0.25, mode="cosine"):
    if total_epochs <= warmup_epochs:
        return 1.0
    if epoch <= warmup_epochs:
        return 1.0

    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    progress = min(max(progress, 0.0), 1.0)
    if mode == "linear":
        decay = 1.0 - progress
    elif mode == "cosine":
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unknown regularization decay mode: {mode}")
    return final_scale + (1.0 - final_scale) * decay


def build_checkpoint(model, psf, cfg, epoch, best_loss, image_path, psf_path):
    return {
        "model": model.state_dict(),
        "psf": psf.cpu(),
        "cfg": cfg,
        "epoch": epoch,
        "best_loss": best_loss,
        "image_path": image_path,
        "psf_path": psf_path,
        "normalization": "percentile_0.1_99.9",
    }


def train(
    image_path,
    psf_path,
    out_dir="./output",
    epochs=30,
    batch_size=16,
    lr=1e-4,
    patch_size=128,
    n_samples=2000,
    upsample=True,
    hess_w=0.15,
    tv_w=0.05,
    l1_w=0.0,
    psf_size=25,
    base_ch=32,
    depth=4,
    n_conv=3,
    seed=42,
    grad_clip=0.0,
    noise_std=0.1,
    psf_center="image",
    detail_branch=True,
    detail_scale=0.2,
    grad_reproj_w=0.0,
    energy_w=0.05,
    reg_warmup_epochs=5,
    reg_final_scale=0.25,
    reg_decay="cosine",
    hr_psf=True,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    img = percentile_norm(load_image(image_path))
    psf_np = load_psf(psf_path, size=psf_size, center=psf_center)
    psf = torch.from_numpy(psf_np)[None, None].to(device)

    dataset = DeconvDataset(img, patch_size=patch_size, n_samples=n_samples, noise_std=noise_std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    cfg = {
        "base_ch": base_ch,
        "depth": depth,
        "n_conv": n_conv,
        "upsample": upsample,
        "psf_size": psf_size,
        "detail_branch": detail_branch,
        "detail_scale": detail_scale,
        "hr_psf": hr_psf,
        "nonnegative_output": True,
    }
    run_config = {
        "image_path": image_path,
        "psf_path": psf_path,
        "out_dir": out_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patch_size": patch_size,
        "n_samples": n_samples,
        "hess_w": hess_w,
        "tv_w": tv_w,
        "l1_w": l1_w,
        "seed": seed,
        "grad_clip": grad_clip,
        "noise_std": noise_std,
        "psf_center": psf_center,
        "detail_branch": detail_branch,
        "detail_scale": detail_scale,
        "grad_reproj_w": grad_reproj_w,
        "energy_w": energy_w,
        "reg_warmup_epochs": reg_warmup_epochs,
        "reg_final_scale": reg_final_scale,
        "reg_decay": reg_decay,
        "hr_psf": hr_psf,
        "device": str(device),
        "model": cfg,
    }
    save_json(run_config, os.path.join(out_dir, "config.json"))
    save_image(os.path.join(out_dir, "psf_used.tif"), psf_np / psf_np.max())

    model = DirectDeconvNet(
        base_ch=base_ch,
        depth=depth,
        n_conv=n_conv,
        upsample=upsample,
        detail_branch=detail_branch,
        detail_scale=detail_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = DeconvLoss(
        psf,
        upsample=upsample,
        hess_w=hess_w,
        tv_w=tv_w,
        l1_w=l1_w,
        grad_reproj_w=grad_reproj_w,
        energy_w=energy_w,
        hr_psf=hr_psf,
        scale=2,
    )

    best_loss = float("inf")
    loss_csv = os.path.join(out_dir, "loss.csv")
    if os.path.exists(loss_csv):
        os.remove(loss_csv)

    print(f"Starting training on {device}, epochs={epochs}", flush=True)
    for epoch in range(1, epochs + 1):
        reg_scale = regularization_scale(epoch, epochs, reg_warmup_epochs, reg_final_scale, reg_decay)
        loss_fn.set_regularization_scale(reg_scale)
        model.train()
        losses = []
        for inp, gt in loader:
            inp = inp.to(device)
            gt = gt.to(device)
            pred = model(inp)
            loss = loss_fn(gt, pred)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(losses))
        current_lr = optimizer.param_groups[0]["lr"]
        append_loss_row(loss_csv, epoch, mean_loss, current_lr, reg_scale)

        print(
            f"epoch {epoch}/{epochs} | loss={mean_loss:.6f} | lr={current_lr:.2e} | reg_scale={reg_scale:.3f}",
            flush=True,
        )
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_ckpt = build_checkpoint(model, psf, cfg, epoch, best_loss, image_path, psf_path)
            torch.save(best_ckpt, os.path.join(out_dir, "best_deconv.pt"))
        last_ckpt = build_checkpoint(model, psf, cfg, epoch, best_loss, image_path, psf_path)
        torch.save(last_ckpt, os.path.join(out_dir, "last_deconv.pt"))

    print(f"Best model saved to {os.path.join(out_dir, 'best_deconv.pt')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a zero-shot microscopy deconvolution model.")
    parser.add_argument("--image", default="datasets/RawSIMData_level_09.tif", help="Input grayscale TIFF.")
    parser.add_argument("--psf", default="datasets/psf_emLambda525_dxy0.0313_NA1.3.tif", help="PSF TIFF.")
    parser.add_argument("--out-dir", default="./output", help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--hess-w", type=float, default=0.15)
    parser.add_argument("--tv-w", type=float, default=0.05)
    parser.add_argument("--l1-w", type=float, default=0.0)
    parser.add_argument("--psf-size", type=int, default=25)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-conv", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--psf-center", choices=["peak", "image"], default="image")
    parser.add_argument("--detail-scale", type=float, default=0.2)
    parser.add_argument("--grad-reproj-w", type=float, default=0.0)
    parser.add_argument("--energy-w", type=float, default=0.05, help="Weight for mean-intensity energy conservation.")
    parser.add_argument("--reg-warmup-epochs", type=int, default=5)
    parser.add_argument("--reg-final-scale", type=float, default=0.25)
    parser.add_argument("--reg-decay", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--lr-psf-forward", action="store_true", help="Use notebook-style LR PSF forward model.")
    parser.add_argument("--no-detail-branch", action="store_true", help="Disable the high-frequency detail branch.")
    parser.add_argument("--no-upsample", action="store_true", help="Disable 2x output upsampling.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        image_path=args.image,
        psf_path=args.psf,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size=args.patch_size,
        n_samples=args.n_samples,
        upsample=not args.no_upsample,
        hess_w=args.hess_w,
        tv_w=args.tv_w,
        l1_w=args.l1_w,
        psf_size=args.psf_size,
        base_ch=args.base_ch,
        depth=args.depth,
        n_conv=args.n_conv,
        seed=args.seed,
        grad_clip=args.grad_clip,
        noise_std=args.noise_std,
        psf_center=args.psf_center,
        detail_branch=not args.no_detail_branch,
        detail_scale=args.detail_scale,
        grad_reproj_w=args.grad_reproj_w,
        energy_w=args.energy_w,
        reg_warmup_epochs=args.reg_warmup_epochs,
        reg_final_scale=args.reg_final_scale,
        reg_decay=args.reg_decay,
        hr_psf=not args.lr_psf_forward,
    )
