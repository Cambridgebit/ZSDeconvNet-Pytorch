"""Backward-compatible loss imports."""

from .losses import DeconvLoss, HessianLoss, TVLoss

__all__ = ["DeconvLoss", "HessianLoss", "TVLoss"]
