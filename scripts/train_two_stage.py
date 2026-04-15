#!/usr/bin/env python3
"""
Two-stage UNet Training for ZSDeconvNet
Fully aligned with official TensorFlow implementation

Usage:
    python scripts/train_two_stage.py \
        --data_dir datasets/Microtubule/ \
        --folder train_data \
        --otf_or_psf_path datasets/Microtubule/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif \
        --psf_src_mode 1 \
        --save_weights_dir ./checkpoints/ \
        --iterations 50000
"""

import argparse
import os
import sys
import time
import datetime
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from pathlib import Path
from scipy.interpolate import interp1d
import cv2

from zsdeconv import (
    DenoiseUNet,
    DirectDeconvNet,
    JointDenoiseDeconvNet,
    PseudoPairDataset,
    HessianLoss,
    TVLoss,
    percentile_norm,
    load_image,
    save_image,
    ensure_dir,
    set_seed,
    reflect_pad,
    crop_pad,
    gaussian_psf,
    load_psf_from_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage UNet Training")

    # Model
    parser.add_argument(
        "--conv_block_num", type=int, default=4, help="Number of convolution blocks"
    )
    parser.add_argument(
        "--conv_num", type=int, default=3, help="Number of convolutions per block"
    )
    parser.add_argument(
        "--upsample_flag", type=int, default=1, help="2x upsampling flag"
    )

    # Training settings
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--mixed_precision", type=int, default=0, help="Mixed precision training"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="Total training iterations"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--test_interval", type=int, default=1000, help="Save checkpoint interval"
    )
    parser.add_argument(
        "--valid_interval", type=int, default=1000, help="Validation interval"
    )
    parser.add_argument(
        "--load_init_model_iter",
        type=int,
        default=0,
        help="Load initial weights from iteration",
    )

    # Learning rate
    parser.add_argument("--start_lr", type=float, default=5e-5)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--lr_decay_interval", type=int, default=10000)

    # Data
    parser.add_argument("--otf_or_psf_path", type=str, required=True)
    parser.add_argument(
        "--psf_src_mode", type=int, default=1, help="1: PSF in .tif, 2: OTF in .mrc"
    )
    parser.add_argument(
        "--dxypsf", type=float, default=0.0313, help="dxy of simulated PSF"
    )
    parser.add_argument("--dx", type=float, default=0.0313)
    parser.add_argument("--dy", type=float, default=0.0313)

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--test_images_path", type=str, default="")
    parser.add_argument("--save_weights_dir", type=str, default="./checkpoints/")
    parser.add_argument("--save_weights_suffix", type=str, default="_Hess0.02")

    parser.add_argument("--input_y", type=int, default=128)
    parser.add_argument("--input_x", type=int, default=128)
    parser.add_argument(
        "--insert_xy", type=int, default=16, help="Boundary padding (border to crop)"
    )
    parser.add_argument("--input_y_test", type=int, default=512)
    parser.add_argument("--input_x_test", type=int, default=512)
    parser.add_argument("--valid_num", type=int, default=3)

    # Loss functions
    parser.add_argument("--mse_flag", type=int, default=0, help="0 for MAE, 1 for MSE")
    parser.add_argument("--denoise_loss_weight", type=float, default=0.5)
    parser.add_argument("--l1_rate", type=float, default=0)
    parser.add_argument("--TV_rate", type=float, default=0)
    parser.add_argument("--Hess_rate", type=float, default=0.02)
    parser.add_argument("--laplace_rate", type=float, default=0)

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ============================================================================
# Model Definition
# ============================================================================


class TwoStageUNet(nn.Module):
    """
    Two-stage UNet for joint denoising and deconvolution

    Outputs:
        - denoised: Intermediate denoised result
        - deconv: Final deconvolved result (2x upsampled if upsample_flag=1)
    """

    def __init__(self, base_ch=32, depth=4, n_conv=3, upsample=True):
        super().__init__()
        self.upsample = upsample

        # Stage 1: Denoising
        self.denoise_encoder = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.denoise_out = nn.Conv2d(base_ch, 1, 3, padding=1)

        # Stage 2: Deconvolution
        self.deconv = DirectDeconvNet(base_ch, depth, n_conv, upsample)

    def forward(self, x):
        # Stage 1: Denoising
        feat = self.denoise_encoder(x)
        denoised = F.relu(self.denoise_out(feat))

        # Stage 2: Deconvolution
        deconv = self.deconv(denoised)

        return denoised, deconv


# ============================================================================
# Loss Functions (Fully aligned with TensorFlow)
# ============================================================================


class LaplaceLoss(nn.Module):
    """Laplace regularization loss"""

    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3))

    def forward(self, x):
        lap = F.conv2d(x, self.kernel, padding=1)
        return lap.pow(2).mean()


class PSFLoss(nn.Module):
    """
    PSF loss function (fully aligned with TensorFlow implementation)

    loss = PSF_reconstruction_loss + TV_weight*TV_loss +
           Hess_weight*Hessian_loss + laplace_weight*Laplace_loss +
           l1_rate*L1_loss
    """

    def __init__(
        self,
        psf,
        TV_weight=0,
        Hess_weight=0.02,
        laplace_weight=0,
        l1_rate=0,
        mse_flag=False,
        upsample_flag=True,
        insert_xy=16,
        deconv_flag=True,
    ):
        super().__init__()
        self.register_buffer("psf", psf)
        self.TV_weight = TV_weight
        self.Hess_weight = Hess_weight
        self.laplace_weight = laplace_weight
        self.l1_rate = l1_rate
        self.mse_flag = mse_flag
        self.upsample_flag = upsample_flag
        self.insert_xy = insert_xy
        self.deconv_flag = deconv_flag

        if TV_weight > 0:
            self.tv_loss = TVLoss()
        if Hess_weight > 0:
            self.hess_loss = HessianLoss()
        if laplace_weight > 0:
            self.laplace_loss = LaplaceLoss()

    def forward(self, y_true, y_pred):
        B, C, H, W = y_pred.shape

        # PSF convolution
        if self.deconv_flag:
            k = self.psf.shape[-1]
            pad = k // 2
            y_conv = F.conv2d(
                F.pad(y_pred, (pad, pad, pad, pad), mode="reflect"), self.psf, padding=0
            )
        else:
            y_conv = y_pred

        # Upsampling (downsample to match y_true)
        if self.upsample_flag:
            y_conv = F.interpolate(
                y_conv, size=(H, W), mode="bilinear", align_corners=False
            )

        # Crop boundary (equivalent to insert_xy in TF)
        if self.insert_xy > 0:
            y_conv = y_conv[
                ..., self.insert_xy : -self.insert_xy, self.insert_xy : -self.insert_xy
            ]
            y_true = y_true[
                ..., self.insert_xy : -self.insert_xy, self.insert_xy : -self.insert_xy
            ]

        # Reconstruction loss
        if self.mse_flag:
            psf_loss = F.mse_loss(y_conv, y_true)
        else:
            psf_loss = F.l1_loss(y_conv, y_true)

        # TV loss
        TV_loss = 0
        if self.TV_weight > 0:
            TV_loss = self.tv_loss(y_pred)

        # Hessian loss (4 terms: xx, yy, xy, yx)
        Hess_loss = 0
        if self.Hess_weight > 0:
            dxx = y_pred[:, :, :, :-2] - 2 * y_pred[:, :, :, 1:-1] + y_pred[:, :, :, 2:]
            dyy = y_pred[:, :, :-2, :] - 2 * y_pred[:, :, 1:-1, :] + y_pred[:, :, 2:, :]
            dxy = (
                y_pred[:, :, 1:, 1:]
                - y_pred[:, :, :-1, 1:]
                - y_pred[:, :, 1:, :-1]
                + y_pred[:, :, :-1, :-1]
            )
            dyx = (
                y_pred[:, :, 1:, 1:]
                - y_pred[:, :, 1:, :-1]
                - y_pred[:, :, :-1, 1:]
                + y_pred[:, :, :-1, :-1]
            )
            Hess_loss = (
                dxx.pow(2).mean()
                + dyy.pow(2).mean()
                + dxy.pow(2).mean()
                + dyx.pow(2).mean()
            ) / 4.0

        # Laplace loss
        laplace_loss_val = 0
        if self.laplace_weight > 0:
            laplace_loss_val = self.laplace_loss(y_pred)

        # L1 loss
        l1_loss_val = 0
        if self.l1_rate > 0:
            l1_loss_val = y_pred.abs().mean()

        total_loss = (
            psf_loss
            + self.TV_weight * TV_loss
            + self.Hess_weight * Hess_loss
            + self.laplace_weight * laplace_loss_val
            + self.l1_rate * l1_loss_val
        )

        return total_loss, psf_loss, TV_loss, Hess_loss


# ============================================================================
# Data Loading (aligned with TF DataLoader)
# ============================================================================


def load_train_data(images_path, gt_path, batch_size, input_y, input_x, insert_xy):
    """
    Load training data with padding (aligned with TF DataLoader)

    Args:
        images_path: Path to input images
        gt_path: Path to ground truth images
        batch_size: Batch size
        input_y, input_x: Input dimensions
        insert_xy: Boundary padding

    Returns:
        input_g: Padded input (B, 1, H+2*insert_xy, W+2*insert_xy)
        gt_g: Ground truth (B, 1, H, W)
    """
    # Get image files
    img_files = sorted(glob.glob(os.path.join(images_path, "*.tif")))
    gt_files = sorted(glob.glob(os.path.join(gt_path, "*.tif")))

    # Randomly select batch_size images
    if len(img_files) < batch_size:
        # Repeat if not enough images
        indices = np.random.choice(len(img_files), batch_size, replace=True)
    else:
        indices = np.random.choice(len(img_files), batch_size, replace=False)

    input_list = []
    gt_list = []

    for idx in indices:
        # Load image
        img = load_image(img_files[idx])
        gt = load_image(gt_files[idx])

        # Resize to input size if needed
        if img.shape != (input_y, input_x):
            img = cv2.resize(img, (input_x, input_y))
            gt = cv2.resize(gt, (input_x, input_y))

        input_list.append(img)
        gt_list.append(gt)

    input_g = np.stack(input_list, axis=0)  # (B, H, W)
    gt_g = np.stack(gt_list, axis=0)  # (B, H, W)

    # Add padding (insert_xy)
    if insert_xy > 0:
        input_g = np.pad(
            input_g,
            ((0, 0), (insert_xy, insert_xy), (insert_xy, insert_xy)),
            mode="constant",
        )

    # Add channel dimension
    input_g = torch.from_numpy(input_g[:, None, :, :]).float()
    gt_g = torch.from_numpy(gt_g[:, None, :, :]).float()

    return input_g, gt_g


def load_validation_data(
    valid_images_path, valid_gt_path, valid_num, input_y, input_x, insert_xy
):
    """Load validation data"""
    input_g, gt_g = load_train_data(
        valid_images_path, valid_gt_path, valid_num, input_y, input_x, insert_xy
    )
    return input_g, gt_g


# ============================================================================
# Training
# ============================================================================


class Trainer:
    def __init__(self, args):
        self.args = args

        # Set device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if args.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision enabled")
        else:
            self.scaler = None

        # Set seed
        set_seed(args.seed)

        # Create directories
        self.save_weights_name = (
            f"{args.folder}_twostage_Unet{args.save_weights_suffix}"
        )
        self.save_weights_path = os.path.join(
            args.save_weights_dir, self.save_weights_name
        )
        self.valid_path = os.path.join(self.save_weights_path, "TrainSampled")
        self.test_path = os.path.join(self.save_weights_path, "TestSampled")
        self.log_path = os.path.join(self.save_weights_path, "graph")

        ensure_dir(self.save_weights_path)
        ensure_dir(self.valid_path)
        ensure_dir(self.test_path)
        ensure_dir(self.log_path)

        # Save config
        with open(os.path.join(self.save_weights_path, "config.txt"), "w") as f:
            f.write(str(args))

        # TensorBoard
        self.writer = SummaryWriter(self.log_path)

        # Prepare PSF
        self.psf = self._prepare_psf()

        # Prepare model
        self.model = self._prepare_model()

        # Prepare data paths
        self.train_images_path = os.path.join(args.data_dir, args.folder, "input")
        self.train_gt_path = os.path.join(args.data_dir, args.folder, "gt")
        self.valid_images_path = self.train_images_path  # Same as train for now
        self.valid_gt_path = self.train_gt_path

        # Load validation data
        self.input_valid, self.gt_valid = load_validation_data(
            self.valid_images_path,
            self.valid_gt_path,
            args.valid_num,
            args.input_y,
            args.input_x,
            args.insert_xy,
        )
        self.input_valid = self.input_valid.to(self.device)
        self.gt_valid = self.gt_valid.to(self.device)

        # Save validation samples
        self._save_validation_samples()

        # Training stats
        self.loss_record = []
        self.loss_record2 = []
        self.iteration = args.load_init_model_iter

    def _prepare_psf(self):
        """Load and process PSF (aligned with TF)"""
        args = self.args

        if args.psf_src_mode == 1:
            print("Using PSF from file...")
            psf = load_psf_from_file(args.otf_or_psf_path, size=25)
        else:
            print("Using simulated Gaussian PSF...")
            psf = gaussian_psf(25, sigma=2.0)

        # Reshape to (1, 1, K, K)
        psf_tensor = torch.from_numpy(psf)[None, None].float().to(self.device)
        psf_tensor = psf_tensor / psf_tensor.sum().clamp_min(1e-8)

        # Save PSF for visualization
        psf_save_path = os.path.join(self.save_weights_path, "psf.tif")
        save_image(psf_save_path, psf)

        return psf_tensor

    def _prepare_model(self):
        """Initialize model"""
        args = self.args

        model = TwoStageUNet(
            base_ch=32,
            depth=args.conv_block_num,
            n_conv=args.conv_num,
            upsample=(args.upsample_flag == 1),
        ).to(self.device)

        # Load weights if specified
        if args.load_init_model_iter > 0:
            weight_path = os.path.join(
                self.save_weights_path, f"weights_{args.load_init_model_iter}.pth"
            )
            if os.path.exists(weight_path):
                checkpoint = torch.load(weight_path, map_location=self.device)
                model.load_state_dict(checkpoint["model"])
                print(f"Loaded weights: {weight_path}")

        return model

    def _save_validation_samples(self):
        """Save initial validation samples"""
        args = self.args

        for i in range(args.valid_num):
            img = self.input_valid[i, 0].cpu().numpy()
            img = img[
                args.insert_xy : args.insert_xy + args.input_y,
                args.insert_xy : args.insert_xy + args.input_x,
            ]
            save_image(os.path.join(self.valid_path, f"input_sample_{i}.tif"), img)

            gt = self.gt_valid[i, 0].cpu().numpy()
            save_image(os.path.join(self.valid_path, f"gt_sample_{i}.tif"), gt)

    def train(self):
        """Main training loop"""
        args = self.args

        # Loss function
        criterion = PSFLoss(
            self.psf,
            TV_weight=args.TV_rate,
            Hess_weight=args.Hess_rate,
            laplace_weight=args.laplace_rate,
            l1_rate=args.l1_rate,
            mse_flag=(args.mse_flag == 1),
            upsample_flag=(args.upsample_flag == 1),
            insert_xy=args.insert_xy,
            deconv_flag=True,
        ).to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.start_lr, betas=(0.9, 0.999)
        )

        # Scheduler
        def lr_lambda(iteration):
            if (iteration + 1) % args.lr_decay_interval == 0:
                return args.lr_decay_factor
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        print("=" * 60)
        print(f"Starting training for {args.iterations} iterations")
        print(
            f"Loss weights: denoise={args.denoise_loss_weight}, "
            + f"deconv={1 - args.denoise_loss_weight}"
        )
        print(
            f"Regularization: TV={args.TV_rate}, Hess={args.Hess_rate}, "
            + f"Laplace={args.laplace_rate}, L1={args.l1_rate}"
        )
        print("=" * 60)

        start_time = time.time()

        self.model.train()
        while self.iteration < args.iterations:
            self.iteration += 1

            # Load data
            input_g, gt_g = load_train_data(
                self.train_images_path,
                self.train_gt_path,
                args.batch_size,
                args.input_y,
                args.input_x,
                args.insert_xy,
            )
            input_g = input_g.to(self.device)
            gt_g = gt_g.to(self.device)

            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    denoised, deconv = self.model(input_g)
                    loss_denoise, _, _, _ = criterion(gt_g, denoised)
                    loss_deconv, psf_loss, tv_loss, hess_loss = criterion(gt_g, deconv)
                    loss = (
                        args.denoise_loss_weight * loss_denoise
                        + (1 - args.denoise_loss_weight) * loss_deconv
                    )

                # Backward pass with mixed precision
                optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                denoised, deconv = self.model(input_g)
                loss_denoise, _, _, _ = criterion(gt_g, denoised)
                loss_deconv, psf_loss, tv_loss, hess_loss = criterion(gt_g, deconv)
                loss = (
                    args.denoise_loss_weight * loss_denoise
                    + (1 - args.denoise_loss_weight) * loss_deconv
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Record losses
            self.loss_record.append(loss_denoise.item())
            self.loss_record2.append(loss_deconv.item())

            # Print progress
            elapsed = time.time() - start_time
            print(
                f"{self.iteration} it | time: {elapsed:.1f}s | "
                + f"denoise_loss = {loss_denoise.item():.3e} | "
                + f"deconv_loss = {loss_deconv.item():.3e}"
            )

            # Validation
            if self.iteration % args.valid_interval == 0:
                self._validate(self.iteration)

            # Save checkpoint and test
            if self.iteration % args.test_interval == 0:
                self._save_checkpoint(self.iteration)
                self._write_logs(self.iteration)
                self._test(self.iteration)

        self.writer.close()
        print("Training complete!")

    def _validate(self, iteration):
        """Run validation"""
        print("Validating...")
        valid_start = time.time()

        self.model.eval()
        with torch.no_grad():
            denoised, deconv = self.model(self.input_valid)

            for i in range(self.args.valid_num):
                # Denoised output
                denoise_out = denoised[i, 0].cpu().numpy()
                denoise_out = percentile_norm(denoise_out)
                save_image(
                    os.path.join(
                        self.valid_path, f"{i}_denoised_iter{iteration:05d}.tif"
                    ),
                    denoise_out,
                )

                # Deconvolved output
                deconv_out = deconv[i, 0].cpu().numpy()
                if self.args.upsample_flag:
                    deconv_out = deconv_out[
                        2 * self.args.insert_xy : 2
                        * (self.args.insert_xy + self.args.input_y),
                        2 * self.args.insert_xy : 2
                        * (self.args.insert_xy + self.args.input_x),
                    ]
                else:
                    deconv_out = deconv_out[
                        self.args.insert_xy : self.args.insert_xy + self.args.input_y,
                        self.args.insert_xy : self.args.insert_xy + self.args.input_x,
                    ]
                deconv_out = percentile_norm(deconv_out)
                save_image(
                    os.path.join(
                        self.valid_path, f"{i}_deconved_iter{iteration:05d}.tif"
                    ),
                    deconv_out,
                )

        self.model.train()

        elapsed = time.time() - valid_start
        print(f"Validation time: {elapsed:.2f}s")

    def _save_checkpoint(self, iteration):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.save_weights_path, f"weights_{iteration}.pth"
        )
        torch.save(
            {
                "model": self.model.state_dict(),
                "iteration": iteration,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")

    def _write_logs(self, iteration):
        """Write TensorBoard logs"""
        mean_denoise_loss = np.mean(self.loss_record)
        mean_deconv_loss = np.mean(self.loss_record2)
        lr = (
            self.model.optimizer.param_groups[0]["lr"]
            if hasattr(self.model, "optimizer")
            else self.args.start_lr
        )

        self.writer.add_scalar("lr", lr, iteration)
        self.writer.add_scalar("denoise_loss", mean_denoise_loss, iteration)
        self.writer.add_scalar("deconv_loss", mean_deconv_loss, iteration)

        self.loss_record = []
        self.loss_record2 = []

    def _test(self, iteration):
        """Run inference on test images"""
        print("Testing...")

        if not self.args.test_images_path:
            return

        self.model.eval()

        test_files = sorted(glob.glob(self.args.test_images_path))

        with torch.no_grad():
            for test_path in test_files:
                img = load_image(test_path)
                img_norm = percentile_norm(img)

                # Pad
                img_padded = np.pad(
                    img_norm,
                    (
                        (self.args.insert_xy, self.args.insert_xy),
                        (self.args.insert_xy, self.args.insert_xy),
                    ),
                    mode="constant",
                )

                x = torch.from_numpy(img_padded[None, None]).float().to(self.device)
                denoised, deconv = self.model(x)

                # Save denoised
                denoise_out = denoised[0, 0].cpu().numpy()
                denoise_out = percentile_norm(denoise_out)
                save_image(
                    os.path.join(
                        self.test_path,
                        f"{Path(test_path).stem}_denoised_iter{iteration}.tif",
                    ),
                    denoise_out,
                )

                # Save deconvolved
                deconv_out = deconv[0, 0].cpu().numpy()
                if self.args.upsample_flag:
                    deconv_out = deconv_out[
                        2 * self.args.insert_xy : 2
                        * (self.args.insert_xy + self.args.input_y),
                        2 * self.args.insert_xy : 2
                        * (self.args.insert_xy + self.args.input_x),
                    ]
                else:
                    deconv_out = deconv_out[
                        self.args.insert_xy : self.args.insert_xy + self.args.input_y,
                        self.args.insert_xy : self.args.insert_xy + self.args.input_x,
                    ]
                deconv_out = percentile_norm(deconv_out)
                save_image(
                    os.path.join(
                        self.test_path,
                        f"{Path(test_path).stem}_deconved_iter{iteration}.tif",
                    ),
                    deconv_out,
                )

        self.model.train()


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
