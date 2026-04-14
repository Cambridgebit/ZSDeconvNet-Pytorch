#!/usr/bin/env python3
"""
Denoise training script

Usage:
    python scripts/train_denoise.py --image datasets/Microtubule/train_data/01.tif --epochs 30
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from zsdeconv import (
    DenoiseUNet,
    PseudoPairDataset,
    percentile_norm,
    load_image,
    ensure_dir,
    set_seed,
    reflect_pad,
    crop_pad,
    save_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train denoising model")
    parser.add_argument(
        "--image",
        type=str,
        default="datasets/Microtubule/train_data/01.tif",
        help="Input image path",
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
    parser.add_argument("--depth", type=int, default=4, help="UNet depth")
    parser.add_argument("--base-ch", type=int, default=32, help="Base channels")
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

    # Create dataset
    print("Creating dataset...")
    dataset = PseudoPairDataset(img, args.patch_size, args.n_samples)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Create model
    print("Initializing denoising model...")
    model = DenoiseUNet(args.base_ch, args.depth).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = torch.nn.L1Loss()

    best_path = os.path.join(args.out_dir, "best_denoise.pt")
    best_loss = float("inf")
    history = []

    print("=" * 60)
    print(f"Training denoising model, {args.epochs} epochs")
    print("=" * 60)

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for inp, gt in loader:
            inp, gt = inp.to(device), gt.to(device)
            pred = model(inp)
            loss = criterion(pred, gt)

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
                    "cfg": {"base_ch": args.base_ch, "depth": args.depth, "n_conv": 3},
                },
                best_path,
            )

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history, "b-", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("Denoising Training Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150)
    plt.show()

    print(f"\nTraining complete! Model saved: {best_path}")
    return best_path


if __name__ == "__main__":
    args = parse_args()
    train(args)
