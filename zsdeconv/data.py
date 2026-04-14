"""
Dataset classes for ZSDeconvNet
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PseudoPairDataset(Dataset):
    """
    Generate pseudo-pair data from single image (add Gaussian noise)

    Used for denoising training.

    Args:
        img: Input image (normalized to [0, 1])
        patch_size: Training patch size
        n_samples: Number of samples per epoch
    """

    def __init__(self, img, patch_size=128, n_samples=2000):
        self.img = img
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.h, self.w = img.shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        h_start = np.random.randint(0, self.h - self.patch_size)
        w_start = np.random.randint(0, self.w - self.patch_size)

        patch = self.img[
            h_start : h_start + self.patch_size, w_start : w_start + self.patch_size
        ]

        # Add Gaussian noise to create "noisy" input
        noisy = patch + np.random.normal(0, 0.1, patch.shape).astype(np.float32)
        noisy = np.clip(noisy, 0, 1)

        return (
            torch.from_numpy(noisy[None]),  # Noisy input
            torch.from_numpy(patch[None]),  # Clean target
        )


class DeconvDataset(Dataset):
    """
    Dataset for deconvolution training using denoised images

    Args:
        img: Denoised image (normalized to [0, 1])
        patch_size: Training patch size
        n_samples: Number of samples per epoch
    """

    def __init__(self, img, patch_size=128, n_samples=2000):
        self.img = img
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.h, self.w = img.shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        h_start = np.random.randint(0, self.h - self.patch_size)
        w_start = np.random.randint(0, self.w - self.patch_size)

        patch = self.img[
            h_start : h_start + self.patch_size, w_start : w_start + self.patch_size
        ]

        # Add Gaussian noise to create "noisy" input
        noisy = patch + np.random.normal(0, 0.1, patch.shape).astype(np.float32)
        noisy = np.clip(noisy, 0, 1)

        return (torch.from_numpy(noisy[None]), torch.from_numpy(patch[None]))


class Noise2VoidDataset(Dataset):
    """
    Noise2Void dataset with blind-spot masking

    Generates blind-spot masks and corrupted inputs.

    Args:
        img: Input image (normalized to [0, 1])
        patch_size: Training patch size
        n_samples: Number of samples per epoch
        mask_ratio: Ratio of pixels to keep (0.9 = 10% blind spots)
        window_size: Neighborhood window size for filling blind spots
    """

    def __init__(
        self, img, patch_size=128, n_samples=2000, mask_ratio=0.9, window_size=5
    ):
        self.img = img
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.mask_ratio = mask_ratio
        self.window_size = window_size
        self.h, self.w = img.shape

    def __len__(self):
        return self.n_samples

    def generate_blind_spot_mask(self, patch):
        """Generate blind-spot mask and corrupted input"""
        h, w = patch.shape
        mask = np.ones((h, w), dtype=np.float32)
        output = patch.copy()

        # Randomly select (1-mask_ratio) pixels as blind spots
        num_mask = int(h * w * (1 - self.mask_ratio))
        y_mask = np.random.randint(0, h, num_mask)
        x_mask = np.random.randint(0, w, num_mask)

        for y, x in zip(y_mask, x_mask):
            # Replace center pixel with neighbor value
            y_offset = np.random.randint(
                -self.window_size // 2, self.window_size // 2 + 1
            )
            x_offset = np.random.randint(
                -self.window_size // 2, self.window_size // 2 + 1
            )

            y_neigh = np.clip(y + y_offset, 0, h - 1)
            x_neigh = np.clip(x + x_offset, 0, w - 1)

            # Ensure not using itself
            if y_offset == 0 and x_offset == 0:
                y_offset = np.random.choice([-1, 1])
                y_neigh = np.clip(y + y_offset, 0, h - 1)

            output[y, x] = patch[y_neigh, x_neigh]
            mask[y, x] = 0.0  # Mark as blind spot

        return output, mask

    def __getitem__(self, idx):
        h_start = np.random.randint(0, self.h - self.patch_size)
        w_start = np.random.randint(0, self.w - self.patch_size)

        patch = self.img[
            h_start : h_start + self.patch_size, w_start : w_start + self.patch_size
        ]

        input_patch, mask = self.generate_blind_spot_mask(patch)

        return (
            torch.from_numpy(input_patch[None]),  # Corrupted input
            torch.from_numpy(patch[None]),  # Original target
            torch.from_numpy(mask[None]),  # Blind-spot mask
        )
