import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# An extra loss term: (pred_rgb, target_rgb, valid_mask or None) -> scalar
LossTerm = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], per_sample: bool = False) -> torch.Tensor:
    """Mean of `x` (over the `mask`ed pixels); with `per_sample`, one mean per batch element `(B,)`."""
    if per_sample:
        if mask is None:
            return x.flatten(1).mean(1)
        return (x * mask).flatten(1).sum(1) / (mask.flatten(1).sum(1) * x.shape[1] + 1e-6)
    if mask is None:
        return x.mean()
    return (x * mask).sum() / (mask.sum() * x.shape[1] + 1e-6)


def _pool(x: torch.Tensor, factor: int) -> torch.Tensor:
    return F.avg_pool2d(x, factor, ceil_mode=True) if factor > 1 else x


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    k = k / k.sum()
    C = x.shape[1]
    x = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


def image_gradients(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradients per channel, `(B, C, H, W)` -> `(B, 2C, H, W)` (x then y), scaled so a unit step is 1."""
    C = x.shape[1]
    k = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype) / 4.0
    kernel = torch.stack([k, k.t()]).unsqueeze(1).repeat(C, 1, 1, 1)  # (2C, 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, groups=C)


def local_sharpness(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    """Strongest gradient magnitude within `window` px of each pixel, `(B, C, H, W)` (per channel)."""
    g = image_gradients(x)
    C = x.shape[1]
    mag = torch.sqrt(g[:, :C] ** 2 + g[:, C:] ** 2 + 1e-8)
    return F.max_pool2d(mag, window, stride=1, padding=window // 2)


def local_std(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    """Per-channel standard deviation within `window` x `window` px of each pixel, `(B, C, H, W)`."""
    pad = window // 2
    mean = F.avg_pool2d(F.pad(x, (pad,) * 4, mode="replicate"), window, stride=1)
    mean_sq = F.avg_pool2d(F.pad(x * x, (pad,) * 4, mode="replicate"), window, stride=1)
    return torch.sqrt((mean_sq - mean * mean).clamp(min=0.0) + 1e-6)


def _coarse(x: torch.Tensor, factor: int, mode: str) -> torch.Tensor:
    """
    Low-resolution view of `x` at `factor`. "blur" pools by factor/4 and then Gaussian-blurs, so the
    view varies smoothly with stroke position (overlapping support of ~`factor` px). "pool" is plain
    non-overlapping average pooling, which only gives placement signal within a pooling cell.
    """
    if mode == "pool" or factor <= 1:
        return _pool(x, factor)
    p = max(1, factor // 4)
    return _gaussian_blur(_pool(x, p), sigma=0.5 * factor / p)


class MultiScalePaintLoss(nn.Module):
    """
    Reconstruction loss for stroke-based painting: a full-resolution term for detail plus terms on an
    blurred low-resolution pyramid for coarse color and placement.

    The low-resolution terms matter for optimization as much as for quality: a stroke placed in the wrong
    spot barely overlaps its target at full resolution (hard edges give almost no gradient), but in a
    blurred low-resolution view the two still overlap, so the stroke gets pulled towards where paint is
    needed. With `coarse_mode="blur"` (default) that pull is smooth over ~`factor` pixels.

    Args:
        detail_weight: weight of the full-resolution L1 term.
        scales: downsampling factors of the coarse terms, e.g. (4, 16).
        scale_weights: weight per coarse scale (defaults to 1.0 each).
        coarse_mode: "blur" (smooth, overlapping) or "pool" (non-overlapping average pooling).
        edge_weight: weight of two edge terms (0 disables them):
            "edge"  - 0.5 x L1 between the image gradients (Sobel): places edges correctly. On its own it
                      still prefers a smeared edge over a sharp one that is 1-2 px off (same hedging as
                      pixel L1), hence:
            "sharp" - L1 between the local maximum gradient magnitudes (5 px window): a sharp edge slightly
                      off position matches the target's local sharpness, a smeared edge does not.
        flat_weight: weight of an L1 term between the local standard deviations (5 px windows) of
            prediction and target ("flat"). Several strokes that don't quite match in color / opacity leave
            a flat target area mottled: small in pixel L1, very visible. 0 disables it.
        extra_terms: optional named terms added on top (e.g. a perceptual or adversarial loss later),
            as `name -> (weight, fn(pred, target, mask))`.
    """

    def __init__(
        self,
        detail_weight: float = 1.0,
        scales: Sequence[int] = (4, 16),
        scale_weights: Optional[Sequence[float]] = None,
        coarse_mode: str = "blur",
        edge_weight: float = 0.0,
        flat_weight: float = 0.0,
        extra_terms: Optional[Dict[str, Tuple[float, LossTerm]]] = None,
    ):
        super().__init__()
        self.detail_weight = detail_weight
        self.scales = tuple(scales)
        self.scale_weights = tuple(scale_weights) if scale_weights is not None else (1.0,) * len(self.scales)
        assert len(self.scale_weights) == len(self.scales)
        assert coarse_mode in ("blur", "pool")
        self.coarse_mode = coarse_mode
        self.edge_weight = edge_weight
        self.flat_weight = flat_weight
        self.extra_terms = dict(extra_terms or {})

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor] = None,
        per_sample: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns the total loss (a scalar, or `(B,)` per image with `per_sample`) and a dict of the
        individual (weighted) terms, as batch means, for logging. Extra terms are always batch scalars.
        """

        def mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
            return _masked_mean(x, mask, per_sample)

        terms: Dict[str, torch.Tensor] = {}
        if self.detail_weight > 0:
            terms["detail"] = self.detail_weight * mean((pred - target).abs(), valid_mask)

        if self.edge_weight > 0:
            diff = (image_gradients(pred) - image_gradients(target)).abs()
            terms["edge"] = 0.5 * self.edge_weight * mean(diff, valid_mask)
            sharp = (local_sharpness(pred) - local_sharpness(target)).abs()
            terms["sharp"] = self.edge_weight * mean(sharp, valid_mask)

        if self.flat_weight > 0:
            flat = (local_std(pred) - local_std(target)).abs()
            terms["flat"] = self.flat_weight * mean(flat, valid_mask)

        for factor, weight in zip(self.scales, self.scale_weights):
            if weight <= 0:
                continue
            if valid_mask is None:
                diff = _coarse(pred, factor, self.coarse_mode) - _coarse(target, factor, self.coarse_mode)
                terms[f"x{factor}"] = weight * mean(diff.abs(), None)
            else:
                # Downsample masked images and renormalize by the downsampled mask so padding does not leak in
                m = _coarse(valid_mask, factor, self.coarse_mode)
                diff = (
                    _coarse(pred * valid_mask, factor, self.coarse_mode)
                    - _coarse(target * valid_mask, factor, self.coarse_mode)
                ) / (m + 1e-6)
                terms[f"x{factor}"] = weight * mean(diff.abs(), (m > 0).to(diff.dtype))

        for name, (weight, fn) in self.extra_terms.items():
            terms[name] = weight * fn(pred, target, valid_mask)

        total = sum(terms.values())
        return total, {k: float(v.detach().mean()) for k, v in terms.items()}


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB for images in [0, 1], shape (B,)."""
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse + 1e-10)
