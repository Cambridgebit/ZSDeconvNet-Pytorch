import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import load_image, percentile_norm


def list_tifs(path):
    root = Path(path)
    if root.is_file():
        return [root]
    files = sorted(root.glob("*.tif")) + sorted(root.glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {root}")
    return files


class N2V2Dataset(Dataset):
    def __init__(
        self,
        image_paths,
        patch_size=128,
        n_samples=4000,
        mask_ratio=0.02,
        neighborhood=5,
        pmin=0.1,
        pmax=99.9,
    ):
        self.paths = [Path(p) for p in image_paths]
        self.images = [percentile_norm(load_image(p), pmin=pmin, pmax=pmax) for p in self.paths]
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.mask_ratio = mask_ratio
        self.neighborhood = neighborhood
        if neighborhood < 3 or neighborhood % 2 == 0:
            raise ValueError("neighborhood must be an odd integer >= 3")
        for img, path in zip(self.images, self.paths):
            if img.shape[0] <= patch_size or img.shape[1] <= patch_size:
                raise ValueError(f"patch_size={patch_size} is too large for {path}: {img.shape}")

    def __len__(self):
        return self.n_samples

    def sample_patch(self):
        img = random.choice(self.images)
        h, w = img.shape
        y = np.random.randint(0, h - self.patch_size)
        x = np.random.randint(0, w - self.patch_size)
        return img[y : y + self.patch_size, x : x + self.patch_size]

    def corrupt_blind_spots(self, patch):
        corrupted = patch.copy()
        mask = np.zeros_like(patch, dtype=np.float32)
        h, w = patch.shape
        n_mask = max(1, int(h * w * self.mask_ratio))
        ys = np.random.randint(0, h, size=n_mask)
        xs = np.random.randint(0, w, size=n_mask)
        radius = self.neighborhood // 2

        for y, x in zip(ys, xs):
            dy = dx = 0
            while dy == 0 and dx == 0:
                dy = np.random.randint(-radius, radius + 1)
                dx = np.random.randint(-radius, radius + 1)
            yy = np.clip(y + dy, 0, h - 1)
            xx = np.clip(x + dx, 0, w - 1)
            corrupted[y, x] = patch[yy, xx]
            mask[y, x] = 1.0

        return corrupted, mask

    def __getitem__(self, idx):
        patch = self.sample_patch()
        corrupted, mask = self.corrupt_blind_spots(patch)
        return (
            torch.from_numpy(corrupted[None].astype(np.float32)),
            torch.from_numpy(patch[None].astype(np.float32)),
            torch.from_numpy(mask[None].astype(np.float32)),
        )


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class N2V2UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=48, depth=3):
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for i in range(depth):
            out_ch = base_ch * (2**i)
            self.encoders.append(ConvBlock(ch, out_ch))
            self.pools.append(nn.AvgPool2d(2))
            ch = out_ch

        self.mid = ConvBlock(ch, ch * 2)
        ch = ch * 2

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = base_ch * (2**i)
            self.up_convs.append(nn.Conv2d(ch, skip_ch, 1))
            self.decoders.append(ConvBlock(skip_ch * 2, skip_ch))
            ch = skip_ch

        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.mid(x)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))

        return torch.sigmoid(self.out(x))


def masked_l1(pred, target, mask):
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp_min(1.0)
