import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from zsdeconv.n2v2 import N2V2Dataset, N2V2UNet, list_tifs, masked_l1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_loss(path, epoch, loss, lr):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["epoch", "loss", "lr"])
        writer.writerow([epoch, f"{loss:.8f}", f"{lr:.8g}"])


def train(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    image_paths = list_tifs(args.data)
    dataset = N2V2Dataset(
        image_paths,
        patch_size=args.patch_size,
        n_samples=args.n_samples,
        mask_ratio=args.mask_ratio,
        neighborhood=args.neighborhood,
        pmin=args.pmin,
        pmax=args.pmax,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    cfg = {
        "base_ch": args.base_ch,
        "depth": args.depth,
        "normalization": "percentile",
        "pmin": args.pmin,
        "pmax": args.pmax,
    }
    model = N2V2UNet(base_ch=args.base_ch, depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    run_config = vars(args).copy()
    run_config["device"] = str(device)
    run_config["num_images"] = len(image_paths)
    run_config["model"] = cfg
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    loss_csv = os.path.join(args.out_dir, "loss.csv")
    if os.path.exists(loss_csv):
        os.remove(loss_csv)

    best_loss = float("inf")
    print(f"Training N2V2-style denoiser on {len(image_paths)} images, device={device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for corrupted, target, mask in loader:
            corrupted = corrupted.to(device)
            target = target.to(device)
            mask = mask.to(device)
            pred = model(corrupted)
            loss = masked_l1(pred, target, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(losses))
        lr_now = optimizer.param_groups[0]["lr"]
        append_loss(loss_csv, epoch, mean_loss, lr_now)
        print(f"epoch {epoch}/{args.epochs} | loss={mean_loss:.6f} | lr={lr_now:.2e}", flush=True)

        ckpt = {
            "model": model.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "best_loss": min(best_loss, mean_loss),
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last_n2v2.pt"))
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(ckpt, os.path.join(args.out_dir, "best_n2v2.pt"))

    print(f"Best model saved to {os.path.join(args.out_dir, 'best_n2v2.pt')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an N2V2-style self-supervised denoiser.")
    parser.add_argument("--data", default="datasets_full/Microtubule/train_data", help="TIFF file or directory.")
    parser.add_argument("--out-dir", default="runs/n2v2_microtubule")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--mask-ratio", type=float, default=0.02)
    parser.add_argument("--neighborhood", type=int, default=5)
    parser.add_argument("--base-ch", type=int, default=48)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step", type=int, default=20)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pmin", type=float, default=0.1)
    parser.add_argument("--pmax", type=float, default=99.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
