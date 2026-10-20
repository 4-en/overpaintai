"""Quality and efficiency measures of a finished painting (validation, comparisons)."""

from typing import Dict

import torch

from overpaintai.losses import image_gradients, local_sharpness, psnr

ACTIVE_OPACITY = 0.02  # strokes below this (opacity x gate x fade) are effectively not painted
FAINT_OPACITY = 0.15  # active but barely visible
FAINT_RADIUS_PX = 1.0  # active but thinner than a pixel


def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    g = image_gradients(x)
    C = x.shape[1]
    return torch.sqrt(g[:, :C] ** 2 + g[:, C:] ** 2 + 1e-8).mean(1, keepdim=True)


@torch.no_grad()
def painting_metrics(
    canvas: torch.Tensor, target: torch.Tensor, blank: torch.Tensor, opacity: torch.Tensor, radius_px: torch.Tensor
) -> Dict[str, float]:
    """
    Args:
        canvas, target, blank: (B, 3, H, W) painting, target and the empty canvas.
        opacity, radius_px: (B, N) rendered opacity (incl. gate / fade) and mean radius in pixels of all
            strokes of the paintings.
    Returns means over the batch:
        psnr, edge_err (L1 of image gradients), lineart_edge_err (same, on the target's strongest 5% edges),
        sharpness (local edge strength of the painting / of the target; 1 = as sharp),
        strokes (active strokes per image), faint_share (active strokes that are barely visible or thinner
        than a pixel), gain_per_100 (PSNR gained over the blank canvas per 100 active strokes).
    """
    p = psnr(canvas, target)
    p_blank = psnr(blank, target)
    grad_err = (image_gradients(canvas) - image_gradients(target)).abs().mean(1, keepdim=True)
    mag = _gradient_magnitude(target)
    thr = torch.quantile(mag.flatten(1).float(), 0.95, dim=1).view(-1, 1, 1, 1)
    strong = (mag >= thr).float()
    line_err = (grad_err * strong).flatten(1).sum(1) / strong.flatten(1).sum(1).clamp(min=1)
    sharp = local_sharpness(canvas).flatten(1).mean(1) / local_sharpness(target).flatten(1).mean(1).clamp(min=1e-6)

    active = opacity > ACTIVE_OPACITY
    faint = active & ((opacity < FAINT_OPACITY) | (radius_px < FAINT_RADIUS_PX))
    n_active = active.float().sum(1)
    return {
        "psnr": p.mean().item(),
        "edge_err": grad_err.mean().item(),
        "lineart_edge_err": line_err.mean().item(),
        "sharpness": sharp.mean().item(),
        "strokes": n_active.mean().item(),
        "faint_share": (faint.float().sum(1) / n_active.clamp(min=1)).mean().item(),
        "gain_per_100": ((p - p_blank) / (n_active.clamp(min=1) / 100.0)).mean().item(),
    }
