import torch.nn as nn
import torch.nn.functional as F


class HessianLoss(nn.Module):
    def forward(self, x):
        dxx = x[:, :, :, :-2] - 2 * x[:, :, :, 1:-1] + x[:, :, :, 2:]
        dyy = x[:, :, :-2, :] - 2 * x[:, :, 1:-1, :] + x[:, :, 2:, :]
        dxy = x[:, :, 1:, 1:] - x[:, :, :-1, 1:] - x[:, :, 1:, :-1] + x[:, :, :-1, :-1]
        return (dxx.pow(2).mean() + dyy.pow(2).mean() + dxy.pow(2).mean()) / 3.0


class TVLoss(nn.Module):
    def forward(self, x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx.pow(2).mean() + dy.pow(2).mean()


def resize_psf_for_scale(psf, scale):
    if scale == 1:
        return psf
    k = psf.shape[-1]
    new_k = (k - 1) * scale + 1
    psf_hr = F.interpolate(psf, size=(new_k, new_k), mode="bicubic", align_corners=True)
    psf_hr = psf_hr.clamp_min(0)
    return psf_hr / psf_hr.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def image_gradients(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


class DeconvLoss(nn.Module):
    def __init__(
        self,
        psf,
        upsample=True,
        hess_w=0.02,
        tv_w=0.0,
        l1_w=0.0,
        grad_reproj_w=0.0,
        energy_w=0.0,
        hr_psf=True,
        scale=2,
    ):
        super().__init__()
        self.register_buffer("psf", psf)
        self.upsample = upsample
        self.base_hess_w = hess_w
        self.base_tv_w = tv_w
        self.base_l1_w = l1_w
        self.base_grad_reproj_w = grad_reproj_w
        self.energy_w = energy_w
        self.reg_scale = 1.0
        self.hr_psf = hr_psf
        self.scale = scale
        self.hess = HessianLoss()
        self.tv = TVLoss()

    @property
    def hess_w(self):
        return self.base_hess_w * self.reg_scale

    @property
    def tv_w(self):
        return self.base_tv_w * self.reg_scale

    @property
    def l1_w(self):
        return self.base_l1_w * self.reg_scale

    @property
    def grad_reproj_w(self):
        return self.base_grad_reproj_w * self.reg_scale

    def set_regularization_scale(self, scale):
        self.reg_scale = float(scale)

    def reproject(self, y_pred, target_shape):
        psf = self.psf
        if self.upsample and self.hr_psf:
            psf = resize_psf_for_scale(psf, self.scale)

        k = psf.shape[-1]
        pad = k // 2
        y_conv = F.conv2d(F.pad(y_pred, (pad, pad, pad, pad), mode="reflect"), psf)
        if self.upsample:
            mode = "area" if self.hr_psf else "bilinear"
            if mode == "area":
                y_conv = F.interpolate(y_conv, size=target_shape, mode=mode)
            else:
                y_conv = F.interpolate(y_conv, size=target_shape, mode=mode, align_corners=False)
        return y_conv

    def forward(self, y_true, y_pred):
        y_conv = self.reproject(y_pred, y_true.shape[-2:])
        loss = F.l1_loss(y_conv, y_true)
        if self.energy_w > 0:
            pred_energy = y_conv.mean(dim=(-2, -1))
            true_energy = y_true.mean(dim=(-2, -1))
            loss = loss + self.energy_w * F.l1_loss(pred_energy, true_energy)
        if self.grad_reproj_w > 0:
            conv_dx, conv_dy = image_gradients(y_conv)
            true_dx, true_dy = image_gradients(y_true)
            grad_loss = F.l1_loss(conv_dx, true_dx) + F.l1_loss(conv_dy, true_dy)
            loss = loss + self.grad_reproj_w * grad_loss
        if self.hess_w > 0:
            loss = loss + self.hess_w * self.hess(y_pred)
        if self.tv_w > 0:
            loss = loss + self.tv_w * self.tv(y_pred)
        if self.l1_w > 0:
            loss = loss + self.l1_w * y_pred.abs().mean()
        return loss
