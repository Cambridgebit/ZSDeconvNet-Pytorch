"""
Model architectures for ZSDeconvNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with ReLU activations"""

    def __init__(self, in_ch, out_ch, n_conv=3):
        super().__init__()
        layers = []
        for i in range(n_conv):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """Encoder with skip connections"""

    def __init__(self, in_ch=1, base_ch=32, depth=4, n_conv=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for i in range(depth):
            out_ch = base_ch * (2**i)
            self.blocks.append(ConvBlock(ch, out_ch, n_conv))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch
        self.out_ch = ch

    def forward(self, x):
        skips = []
        for blk, pool in zip(self.blocks, self.pools):
            x = blk(x)
            skips.append(x)
            x = pool(x)
        return x, skips


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection"""

    def __init__(self, in_ch, skip_ch, out_ch, n_conv=3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        layers = [
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        cur = out_ch
        for _ in range(n_conv - 1):
            next_ch = max(out_ch // 2, 1)
            layers.extend(
                [nn.Conv2d(cur, next_ch, 3, padding=1), nn.ReLU(inplace=True)]
            )
            cur = next_ch
        self.conv = nn.Sequential(*layers)
        self.out_ch = cur

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.conv(torch.cat([x, skip], dim=1))


class UNetStage(nn.Module):
    """Single UNet stage (encoder + decoder)"""

    def __init__(self, in_ch=1, base_ch=32, depth=4, n_conv=3):
        super().__init__()
        self.encoder = Encoder(in_ch, base_ch, depth, n_conv)
        mid_ch = self.encoder.out_ch * 2
        self.mid = nn.Sequential(
            nn.Conv2d(self.encoder.out_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, self.encoder.out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoders = nn.ModuleList()
        cur_ch = self.encoder.out_ch
        for i in reversed(range(depth)):
            skip_ch = base_ch * (2**i)
            block = DecoderBlock(cur_ch, skip_ch, skip_ch, n_conv)
            self.decoders.append(block)
            cur_ch = block.out_ch
        self.out_ch = cur_ch

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.mid(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        return x


class DenoiseUNet(nn.Module):
    """
    Denoising UNet

    Args:
        base_ch: Base number of channels
        depth: UNet depth
        n_conv: Convolutions per block
    """

    def __init__(self, base_ch=32, depth=4, n_conv=3):
        super().__init__()
        self.encoder = UNetStage(1, base_ch, depth, n_conv)
        self.out = nn.Conv2d(self.encoder.out_ch, 1, 3, padding=1)

    def forward(self, x):
        feat = self.encoder(x)
        return F.relu(self.out(feat))


class DirectDeconvNet(nn.Module):
    """
    Direct deconvolution network

    Args:
        base_ch: Base number of channels
        depth: UNet depth
        n_conv: Convolutions per block
        upsample: 2x upsampling
    """

    def __init__(self, base_ch=32, depth=4, n_conv=3, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.stage2 = UNetStage(1, base_ch, depth, n_conv)
        self.refine1 = nn.Conv2d(self.stage2.out_ch, 128, 3, padding=1)
        self.refine2 = nn.Conv2d(128, 128, 3, padding=1)
        self.out2 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x):
        f2 = self.stage2(x)
        if self.upsample:
            f2 = F.interpolate(f2, scale_factor=2, mode="nearest")
        deconv = F.relu(self.out2(F.relu(self.refine2(F.relu(self.refine1(f2))))))
        return deconv


class JointDenoiseDeconvNet(nn.Module):
    """
    Joint denoising + deconvolution network

    Architecture:
        Input (noisy) → DenoiseUNet → Denoised → DeconvUNet → Deconvolved

    Args:
        base_ch: Base number of channels
        depth: UNet depth
        n_conv: Convolutions per block
        upsample: 2x upsampling
    """

    def __init__(self, base_ch=32, depth=4, n_conv=3, upsample=True):
        super().__init__()
        self.upsample = upsample

        # Denoising stage
        self.denoise = UNetStage(1, base_ch, depth, n_conv)
        self.out_denoise = nn.Conv2d(self.denoise.out_ch, 1, 3, padding=1)

        # Deconvolution stage
        self.deconv = UNetStage(1, base_ch, depth, n_conv)
        self.refine1 = nn.Conv2d(self.deconv.out_ch, 128, 3, padding=1)
        self.refine2 = nn.Conv2d(128, 128, 3, padding=1)
        self.out_deconv = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x):
        # Denoising
        feat_denoise = self.denoise(x)
        denoised = F.relu(self.out_denoise(feat_denoise))

        # Deconvolution (using denoised as input)
        feat_deconv = self.deconv(denoised)
        if self.upsample:
            feat_deconv = F.interpolate(feat_deconv, scale_factor=2, mode="nearest")
        deconv = F.relu(
            self.out_deconv(F.relu(self.refine2(F.relu(self.refine1(feat_deconv)))))
        )

        return denoised, deconv


class Noise2VoidUNet(nn.Module):
    """
    Noise2Void blind-spot UNet

    Args:
        base_ch: Base number of channels
        depth: UNet depth (shallower than standard)
        n_conv: Convolutions per block
    """

    def __init__(self, base_ch=64, depth=3, n_conv=2):
        super().__init__()
        self.encoder = UNetStage(1, base_ch, depth, n_conv)
        self.out = nn.Conv2d(self.encoder.out_ch, 1, 3, padding=1)

    def forward(self, x):
        feat = self.encoder(x)
        return self.out(feat)
