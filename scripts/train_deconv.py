#!/usr/bin/env python3
"""
Deconvolution training script

Usage:
    python scripts/train_deconv.py --image datasets/Microtubule/train_data/01.tif \
        --psf datasets/Microtubule/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif --epochs 30
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from zsdeconv import (
    DirectDeconvNet,
    DeconvDataset,
    DeconvLoss,
    make_psf_tensor,
    percentile_norm,
    load_image,
    ensure_dir,
    set_seed,
    save_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train deconvolution model")
    parser.add_argument(
        "--image",
        type=str,
        default="datasets/Microtubule/train_data/01.tif",
        help="Input image path",
    )
    parser.add_argument(
        "--psf",
        type=str,
        default="datasets/Microtubule/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif",
        help="PSF file path",
    )
    parser.add_argument(
        "--out-dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--patch-size", type=int, default=128, help="Patch size")
    parser.add_argument("--n-samples", type=int, default=2000, help="Samples per epoch")
    parser.add_argument(
        "--upsample", action="store_true", default=True, help="2x upsampling"
    )
    parser.add_argument("--hess-w", type=float, default=0.05, help="Hessian weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def train(args):
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load image
    print(f"Loading image: {args.image}")
    img = percentile_norm(load_image(args.image))

    # Load PSF
    print(f"Loading PSF: {args.psf}")
    psf = make_psf_tensor("load", img, 25, device=device, psf_path=args.psf)

    # Create dataset
    print("Creating dataset...")
    dataset = DeconvDataset(img, args.patch_size, args.n_samples)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Create model
    print("Initializing deconvolution model...")
    model = DirectDeconvNet(upsample=args.upsample).to(device)

    optimizer = torch.optim.Adam(
        [
            *model.stage2.parameters(),
            model.refine1.weight,
            model.refine1.bias,
            model.refine2.weight,
            model.refine2.bias,
            model.out2.weight,
            model.out2.bias,
        ],
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = DeconvLoss(psf, args.upsample, hess_w=args.hess_w)

    best_path = os.path.join(args.out_dir, "best_deconv.pt")
    best_loss = float("inf")
    history = []

    print("=" * 60)
    print(f"Training deconvolution model, {args.epochs} epochs")
    print("=" * 60)

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for inp, gt in loader:
            inp, gt = inp.to(device), gt.to(device)
            deconv = model(inp)
            loss = loss_fn(gt, deconv)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        mean_loss = np.mean(losses)
        history.append(mean_loss)

        print(
            f"epoch {epoch + 1:03d}/{args.epochs:03d} | loss={mean_loss:.6f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "psf": psf.cpu(),
                    "cfg": {
                        "base_ch": 32,
                        "depth": 4,
                        "n_conv": 3,
                        "upsample": args.upsample,
                    },
                },
                best_path,
            )

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history, "b-", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Deconvolution Training Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150)
    plt.show()

    print(f"\nTraining complete! Model saved: {best_path}")
    return best_path


if __name__ == "__main__":
    args = parse_args()
    train(args)
