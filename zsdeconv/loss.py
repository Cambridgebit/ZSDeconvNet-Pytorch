"""
Loss functions for ZSDeconvNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HessianLoss(nn.Module):
    """
    Hessian regularization to suppress artifacts

    Computes second-order derivatives in x, y, and diagonal directions.
    """

    def forward(self, x):
        dxx = x[:, :, :, :-2] - 2 * x[:, :, :, 1:-1] + x[:, :, :, 2:]
        dyy = x[:, :, :-2, :] - 2 * x[:, :, 1:-1, :] + x[:, :, 2:, :]
        dxy = x[:, :, 1:, 1:] - x[:, :, :-1, 1:] - x[:, :, 1:, :-1] + x[:, :, :-1, :-1]
        return (dxx.pow(2).mean() + dyy.pow(2).mean() + dxy.pow(2).mean()) / 3.0


class TVLoss(nn.Module):
    """
    Total Variation regularization

    Encourages piecewise smooth solutions.
    """

    def forward(self, x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx.pow(2).mean() + dy.pow(2).mean()


class DeconvLoss(nn.Module):
    """
    Deconvolution loss: PSF convolution reconstruction + regularization

    Args:
        psf: Point spread function tensor (1, 1, K, K)
        upsample: Whether output is 2x upsampled
        hess_w: Hessian regularization weight
        tv_w: TV regularization weight
        l1_w: L1 regularization weight
    """

    def __init__(self, psf, upsample=True, hess_w=0.02, tv_w=0.0, l1_w=0.0):
        super().__init__()
        self.register_buffer("psf", psf)
        self.upsample = upsample
        self.hess_w, self.tv_w, self.l1_w = hess_w, tv_w, l1_w
        self.hess = HessianLoss()
        self.tv = TVLoss()

    def forward(self, y_true, y_pred):
        k = self.psf.shape[-1]
        pad = k // 2
        # Convolve prediction with PSF
        y_conv = F.conv2d(
            F.pad(y_pred, (pad, pad, pad, pad), mode="reflect"), self.psf, padding=0
        )

        # Downsample if needed
        if self.upsample:
            y_conv = F.interpolate(
                y_conv, size=y_true.shape[-2:], mode="bilinear", align_corners=False
            )

        # Reconstruction loss
        loss = F.l1_loss(y_conv, y_true)

        # Regularization
        if self.hess_w > 0:
            loss = loss + self.hess_w * self.hess(y_pred)
        if self.tv_w > 0:
            loss = loss + self.tv_w * self.tv(y_pred)
        if self.l1_w > 0:
            loss = loss + self.l1_w * y_pred.abs().mean()

        return loss


class JointLoss(nn.Module):
    """
    Joint loss: denoising + deconvolution

    total_loss = λ_denoise * L_denoise + λ_deconv * L_deconv

    Args:
        psf: Point spread function tensor
        upsample: Whether output is 2x upsampled
        lambda_denoise: Denoising loss weight
        lambda_deconv: Deconvolution loss weight
        hess_w: Hessian regularization weight
        tv_w: TV regularization weight
    """

    def __init__(
        self,
        psf,
        upsample=True,
        lambda_denoise=1.0,
        lambda_deconv=1.0,
        hess_w=0.02,
        tv_w=0.0,
    ):
        super().__init__()
        self.lambda_denoise = lambda_denoise
        self.lambda_deconv = lambda_deconv

        self.denoise_criterion = nn.L1Loss()
        self.deconv_criterion = DeconvLoss(psf, upsample, hess_w, tv_w)

    def forward(self, y_true, denoised, deconv):
        loss_denoise = self.denoise_criterion(denoised, y_true)
        loss_deconv = self.deconv_criterion(y_true, deconv)
        total_loss = (
            self.lambda_denoise * loss_denoise + self.lambda_deconv * loss_deconv
        )
        return total_loss, loss_denoise, loss_deconv
