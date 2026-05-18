import argparse
import copy
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from zsdeconv.losses import DeconvLoss
from zsdeconv.models import DirectDeconvNet
from zsdeconv.n2v2 import list_tifs
from zsdeconv.psf import load_psf
from zsdeconv.utils import load_image, percentile_norm, save_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiImageDeconvDataset(Dataset):
    def __init__(self, image_paths, patch_size=128, n_samples=4000, noise_std=0.02, pmin=0.1, pmax=99.9):
        self.paths = [Path(p) for p in image_paths]
        self.images = [percentile_norm(load_image(p), pmin=pmin, pmax=pmax) for p in self.paths]
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.noise_std = noise_std
        for img, path in zip(self.images, self.paths):
            if img.shape[0] <= patch_size or img.shape[1] <= patch_size:
                raise ValueError(f"patch_size={patch_size} is too large for {path}: {img.shape}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img = random.choice(self.images)
        h, w = img.shape
        y = np.random.randint(0, h - self.patch_size)
        x = np.random.randint(0, w - self.patch_size)
        patch = img[y : y + self.patch_size, x : x + self.patch_size]
        noisy = patch + np.random.normal(0, self.noise_std, patch.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 1.0)
        return torch.from_numpy(noisy[None]), torch.from_numpy(patch[None])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_loss(path, epoch, loss, lr, reg_scale):
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


def build_checkpoint(model, psf, cfg, epoch, best_loss, data_dir, psf_path):
    return {
        "model": model.state_dict(),
        "psf": psf.cpu(),
        "cfg": cfg,
        "epoch": epoch,
        "best_loss": best_loss,
        "data_dir": data_dir,
        "psf_path": psf_path,
        "normalization": "percentile_0.1_99.9",
    }


def load_init_checkpoint(model, path):
    if not path:
        return
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    result = model.load_state_dict(state, strict=False)
    print(f"Initialized from {path}")
    if result.missing_keys:
        print(f"  missing keys: {len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"  unexpected keys: {len(result.unexpected_keys)}")


def set_detail_trainable(model, train_detail_only):
    if not train_detail_only:
        for param in model.parameters():
            param.requires_grad_(True)
        return
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("detail"))


def trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def detail_run_name(scale, l1_w):
    return f"detail_s{scale:g}_l1{l1_w:g}".replace(".", "p")


def parse_detail_sweep(items):
    if not items:
        return []
    pairs = []
    for item in items:
        if ":" not in item:
            raise ValueError("Each --detail-sweep item must use scale:l1_w, for example 0.06:0.04")
        scale, l1_w = item.split(":", 1)
        pairs.append((float(scale), float(l1_w)))
    return pairs


def train(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    image_paths = list_tifs(args.data)
    dataset = MultiImageDeconvDataset(
        image_paths,
        patch_size=args.patch_size,
        n_samples=args.n_samples,
        noise_std=args.noise_std,
        pmin=args.pmin,
        pmax=args.pmax,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    psf_np = load_psf(args.psf, size=args.psf_size, center=args.psf_center)
    psf = torch.from_numpy(psf_np)[None, None].to(device)

    cfg = {
        "base_ch": args.base_ch,
        "depth": args.depth,
        "n_conv": args.n_conv,
        "upsample": not args.no_upsample,
        "psf_size": args.psf_size,
        "detail_branch": not args.no_detail_branch,
        "detail_scale": args.detail_scale,
        "gated_detail": not args.no_gated_detail,
        "hr_psf": not args.lr_psf_forward,
        "nonnegative_output": True,
    }
    run_config = vars(args).copy()
    run_config["device"] = str(device)
    run_config["num_images"] = len(image_paths)
    run_config["model"] = cfg
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    save_image(os.path.join(args.out_dir, "psf_used.tif"), psf_np / psf_np.max())

    model = DirectDeconvNet(
        base_ch=args.base_ch,
        depth=args.depth,
        n_conv=args.n_conv,
        upsample=not args.no_upsample,
        detail_branch=not args.no_detail_branch,
        detail_scale=args.detail_scale,
        gated_detail=not args.no_gated_detail,
    ).to(device)
    load_init_checkpoint(model, args.init_checkpoint)
    set_detail_trainable(model, args.freeze_backbone_epochs > 0)
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    loss_fn = DeconvLoss(
        psf,
        upsample=not args.no_upsample,
        hess_w=args.hess_w,
        tv_w=args.tv_w,
        l1_w=args.l1_w,
        grad_reproj_w=args.grad_reproj_w,
        energy_w=args.energy_w,
        hr_psf=not args.lr_psf_forward,
        scale=2,
    )

    loss_csv = os.path.join(args.out_dir, "loss.csv")
    if os.path.exists(loss_csv):
        os.remove(loss_csv)

    best_loss = float("inf")
    print(f"Training multi-image deconv on {len(image_paths)} denoised images, device={device}")
    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            set_detail_trainable(model, False)
            optimizer = torch.optim.Adam(trainable_parameters(model), lr=args.lr * args.unfreeze_lr_scale)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
            print(
                f"Unfroze backbone at epoch {epoch}; lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

        model.train()
        reg_scale = regularization_scale(
            epoch,
            args.epochs,
            args.reg_warmup_epochs,
            args.reg_final_scale,
            args.reg_decay,
        )
        loss_fn.set_regularization_scale(reg_scale)
        losses = []
        for inp, gt in loader:
            inp = inp.to(device)
            gt = gt.to(device)
            pred = model(inp)
            loss = loss_fn(gt, pred)
            detail_reg = model.detail_regularization()
            if args.detail_l1_w > 0 and detail_reg is not None:
                loss = loss + args.detail_l1_w * reg_scale * detail_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(losses))
        lr_now = optimizer.param_groups[0]["lr"]
        append_loss(loss_csv, epoch, mean_loss, lr_now, reg_scale)
        print(
            f"epoch {epoch}/{args.epochs} | loss={mean_loss:.6f} | lr={lr_now:.2e} | reg_scale={reg_scale:.3f}",
            flush=True,
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                build_checkpoint(model, psf, cfg, epoch, best_loss, args.data, args.psf),
                os.path.join(args.out_dir, "best_deconv.pt"),
            )
        torch.save(
            build_checkpoint(model, psf, cfg, epoch, best_loss, args.data, args.psf),
            os.path.join(args.out_dir, "last_deconv.pt"),
        )

    print(f"Best model saved to {os.path.join(args.out_dir, 'best_deconv.pt')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train deconvolution on a directory of denoised TIFF images.")
    parser.add_argument("--data", default="runs/n2v2_microtubule/denoised")
    parser.add_argument("--psf", default="datasets/psf_emLambda525_dxy0.0313_NA1.3.tif")
    parser.add_argument("--out-dir", default="runs/deconv_multi_n2v2")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step", type=int, default=10)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--hess-w", type=float, default=0.08)
    parser.add_argument("--tv-w", type=float, default=0.015)
    parser.add_argument("--l1-w", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--psf-size", type=int, default=25)
    parser.add_argument("--psf-center", choices=["image", "peak"], default="image")
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-conv", type=int, default=3)
    parser.add_argument("--detail-scale", type=float, default=0.12)
    parser.add_argument("--detail-l1-w", type=float, default=0.01)
    parser.add_argument("--init-checkpoint", default=None, help="Initialize from a baseline checkpoint.")
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Train only the detail branch for the first N epochs after loading a baseline.",
    )
    parser.add_argument(
        "--unfreeze-lr-scale",
        type=float,
        default=0.25,
        help="LR multiplier after unfreezing the backbone.",
    )
    parser.add_argument(
        "--detail-sweep",
        nargs="*",
        default=None,
        help="Run multiple detail fine-tunes. Format: scale:l1_w, for example 0.04:0.04 0.06:0.06.",
    )
    parser.add_argument("--grad-reproj-w", type=float, default=0.0)
    parser.add_argument("--energy-w", type=float, default=0.05, help="Weight for mean-intensity energy conservation.")
    parser.add_argument("--reg-warmup-epochs", type=int, default=5)
    parser.add_argument("--reg-final-scale", type=float, default=0.25)
    parser.add_argument("--reg-decay", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pmin", type=float, default=0.1)
    parser.add_argument("--pmax", type=float, default=99.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-psf-forward", action="store_true")
    parser.add_argument("--no-detail-branch", action="store_true")
    parser.add_argument("--no-gated-detail", action="store_true")
    parser.add_argument("--no-upsample", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sweep = parse_detail_sweep(args.detail_sweep)
    if sweep:
        base_out_dir = args.out_dir
        for scale, l1_w in sweep:
            run_args = copy.deepcopy(args)
            run_args.detail_sweep = None
            run_args.no_detail_branch = False
            run_args.detail_scale = scale
            run_args.detail_l1_w = l1_w
            run_args.out_dir = os.path.join(base_out_dir, detail_run_name(scale, l1_w))
            print(f"\n=== Running {detail_run_name(scale, l1_w)} ===", flush=True)
            train(run_args)
    else:
        train(args)
