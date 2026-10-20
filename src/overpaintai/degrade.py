"""
Synthetic "partially painted" canvases: the target, degraded in soft-edged regions, as a start state for
training later painting steps without having to paint up to them first.

The degradations are chosen to look like the mistakes of a stroke painting rather than like photo
damage: besides local blur, regions are repainted with rough strokes in the target's local colors (so the
canvas really is a stroke painting there), shifted in color, flattened, left unpainted, or kept at a
coarser level of detail.
"""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.losses import _gaussian_blur
from overpaintai.synthetic import random_strokes

DEGRADATIONS = ("blur", "strokes", "color", "flatten", "missing", "lowres")


def _rand(gen: Optional[torch.Generator], *shape, device) -> torch.Tensor:
    return torch.rand(*shape, generator=gen, device=device)


def region_masks(
    B: int, H: int, W: int, count: int, size_range=(0.08, 0.35), gen=None, device=None
) -> torch.Tensor:
    """
    Soft blob masks `(B, 1, H, W)` in [0, 1]: the union of `count` elliptical regions per image (random
    centers, sizes as a fraction of the image side, aspect and soft falloff).
    """
    yy = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1)
    xx = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, W)
    c = _rand(gen, B, count, 2, device=device)
    lo, hi = size_range
    r = lo * (hi / lo) ** _rand(gen, B, count, 2, device=device)  # log-uniform radii (x, y)
    dx = (xx - c[..., 0].view(B, count, 1, 1)) / r[..., 0].view(B, count, 1, 1)
    dy = (yy - c[..., 1].view(B, count, 1, 1)) / r[..., 1].view(B, count, 1, 1)
    d = torch.sqrt(dx**2 + dy**2)
    m = torch.sigmoid((1.0 - d) * 6.0)  # soft edge
    return m.amax(dim=1, keepdim=True)


def _stroke_repaint(
    engine: DifferentiableCanvasEngine, rgb: torch.Tensor, target: torch.Tensor, n: int, gen, rough: float
) -> torch.Tensor:
    """Paints `n` random strokes over `rgb` with colors sampled from the locally averaged target."""
    B, _, S, _ = target.shape
    params, tools = random_strokes(engine, B, n, radius_range=(0.08, 0.015), generator=gen, device=target.device)
    # Colors from the target around each stroke's midpoint, plus a little error
    mid = 0.25 * params[..., 0:2] + 0.5 * params[..., 2:4] + 0.25 * params[..., 4:6]
    smooth = _gaussian_blur(target, sigma=max(1.0, 0.02 * S))
    grid = (mid * 2.0 - 1.0).unsqueeze(2)  # (B, n, 1, 2)
    color = F.grid_sample(smooth, grid, mode="bilinear", align_corners=True).squeeze(-1).transpose(1, 2)
    # Mostly brightness error (same for all channels) plus a little hue error: per-channel noise alone
    # turns grey and white areas pink / green
    shade = torch.randn(B, n, 1, generator=gen, device=target.device) * 0.06
    hue = torch.randn(B, n, 3, generator=gen, device=target.device) * 0.015
    noise = (shade + hue) * rough
    params[..., 11:14] = (color + noise).clamp(0.0, 1.0)
    params[..., 8:10] = 0.6 + 0.4 * _rand(gen, B, n, 2, device=target.device)
    layer = torch.cat([rgb, torch.ones_like(rgb[:, :1])], dim=1)
    return engine(layer, params, tools)[:, :3]


@torch.no_grad()
def degraded_canvas(
    target: torch.Tensor,
    engine: DifferentiableCanvasEngine,
    generator: Optional[torch.Generator] = None,
    strength: float = 1.0,
    kinds: Sequence[str] = DEGRADATIONS,
) -> torch.Tensor:
    """
    Paint layer `(B, 4, S, S)` that looks like a partially finished painting of `target` `(B, 3, S, S)`:
    the target with 1-3 randomly chosen degradations, each applied in a few soft regions (and the coarse
    ones sometimes everywhere). `strength` scales how much of the image is affected and how strongly.
    Missing regions are transparent (alpha 0), so the canvas background shows there.
    """
    B, _, S, _ = target.shape
    dev = target.device
    gen = generator
    rgb = target.clone()
    alpha = torch.ones_like(target[:, :1])
    n_kinds = len(kinds)

    for _ in range(int(torch.randint(1, 4, (1,), generator=gen, device=dev).item())):
        kind = kinds[int(torch.randint(0, n_kinds, (1,), generator=gen, device=dev).item())]
        count = max(1, round(strength * int(torch.randint(1, 5, (1,), generator=gen, device=dev).item())))
        mask = region_masks(B, S, S, count, gen=gen, device=dev)
        everywhere = _rand(gen, B, 1, 1, 1, device=dev) < 0.25 * strength
        if kind in ("blur", "lowres", "strokes"):  # coarse degradations sometimes cover the whole image
            mask = torch.where(everywhere, torch.ones_like(mask), mask)

        if kind == "blur":
            sigma = S * (0.004 + 0.03 * strength * _rand(gen, 1, device=dev).item())  # small to medium
            new = _gaussian_blur(rgb, sigma)
        elif kind == "strokes":
            n = max(8, round(strength * (20 + 60 * _rand(gen, 1, device=dev).item())))
            new = _stroke_repaint(engine, rgb, target, n, gen, rough=strength)
        elif kind == "color":
            shift = (_rand(gen, B, 3, 1, 1, device=dev) - 0.5) * 0.15 * strength  # off by up to ~0.08
            gain = 1.0 + (_rand(gen, B, 1, 1, 1, device=dev) - 0.5) * 0.3 * strength
            new = ((rgb - 0.5) * gain + 0.5 + shift).clamp(0.0, 1.0)
        elif kind == "flatten":  # texture and shading lost: a smooth patch of the region's average color
            new = _gaussian_blur(rgb, S * (0.03 + 0.05 * _rand(gen, 1, device=dev).item()))
        elif kind == "missing":
            alpha = alpha * (1.0 - mask * (0.5 + 0.5 * _rand(gen, B, 1, 1, 1, device=dev)))
            continue
        elif kind == "lowres":
            factor = 2 ** int(torch.randint(1, 4, (1,), generator=gen, device=dev).item())
            small = F.interpolate(rgb, scale_factor=1.0 / factor, mode="area")
            new = F.interpolate(small, size=(S, S), mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"unknown degradation {kind!r}")
        rgb = rgb + mask * (new - rgb)

    return torch.cat([rgb.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)], dim=1)

