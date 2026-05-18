import numpy as np
import torch
from torch.utils.data import Dataset


class DeconvDataset(Dataset):
    def __init__(self, img, patch_size=128, n_samples=2000, noise_std=0.1):
        self.img = img
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.h, self.w = img.shape
        if self.h <= patch_size or self.w <= patch_size:
            raise ValueError(
                f"patch_size={patch_size} must be smaller than image shape {(self.h, self.w)}"
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        h_start = np.random.randint(0, self.h - self.patch_size)
        w_start = np.random.randint(0, self.w - self.patch_size)
        patch = self.img[h_start : h_start + self.patch_size, w_start : w_start + self.patch_size]
        noisy = patch + np.random.normal(0, self.noise_std, patch.shape).astype(np.float32)
        noisy = np.clip(noisy, 0, 1)
        return torch.from_numpy(noisy[None]), torch.from_numpy(patch[None])
