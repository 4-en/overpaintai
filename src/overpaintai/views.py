"""
Views: square windows of the image that the stroke predictor works on.

The network always runs at a fixed working resolution R. A view is a box (x, y, s) in normalized image
coordinates (top-left corner and side length, all in [0, 1]); target and canvas are resampled from the box
to R x R, and strokes are predicted in view coordinates [0, 1] and mapped back to the image. Because the
network always sees R pixels per view, a stroke of a given size looks the same to it at any image
resolution, and the renderer draws the mapped strokes at whatever output size is wanted.

Coordinate convention (matches the engine): pixel j of an N-pixel axis sits at coordinate j / (N - 1).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from overpaintai.canvas_engine import DifferentiableCanvasEngine

WHOLE_IMAGE = (0.0, 0.0, 1.0)


@dataclass
class Strokes:
    """
    Strokes in physical units of some coordinate frame (a view, or the whole image).

    points:   (B, N, 3, 2) Bezier start / control / end points (x, y)
    radii:    (B, N, 2) start / end radius, in the same units as the points
    opacity:  (B, N, 2) start / end opacity in [0.05, 1]
    hardness: (B, N, 1) in [0, 1]
    color:    (B, N, 3) RGB in [0, 1]
    """

    points: torch.Tensor
    radii: torch.Tensor
    opacity: torch.Tensor
    hardness: torch.Tensor
    color: torch.Tensor


def whole_image_boxes(batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Boxes `(B, 3)` covering the whole image."""
    return torch.tensor(WHOLE_IMAGE, device=device).expand(batch_size, 3).clone()


def tile_boxes(tiles_per_side: int, overlap: float = 0.125, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Square tiles covering the image, `(T, 3)` with T = tiles_per_side^2. Each tile is enlarged by `overlap`
    (fraction of the tile size) on every side, so strokes can cross tile seams, and clamped into the image.
    """
    n = tiles_per_side
    base = 1.0 / n
    s = min(1.0, base * (1.0 + 2.0 * overlap))
    starts = torch.arange(n, dtype=torch.float32) * base - base * overlap
    starts = starts.clamp(0.0, 1.0 - s)
    ys, xs = torch.meshgrid(starts, starts, indexing="ij")
    return torch.stack([xs.flatten(), ys.flatten(), torch.full((n * n,), s)], dim=-1).to(device)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < 0.3:
        return x
    radius = max(1, math.ceil(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    k = k / k.sum()
    C = x.shape[1]
    x = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


def view_grid(boxes: torch.Tensor, R: int) -> torch.Tensor:
    """Image coordinates of the R x R view pixels, `(B, R, R, 2)` as (x, y) in [0, 1]."""
    u = torch.linspace(0.0, 1.0, R, device=boxes.device, dtype=boxes.dtype)
    x = boxes[:, 0, None, None] + boxes[:, 2, None, None] * u.view(1, 1, R)
    y = boxes[:, 1, None, None] + boxes[:, 2, None, None] * u.view(1, R, 1)
    return torch.stack([x.expand(-1, R, R), y.expand(-1, R, R)], dim=-1)


def resample_views(image: torch.Tensor, boxes: torch.Tensor, R: int) -> torch.Tensor:
    """
    Resamples the box of each image to R x R: `(B, C, H, W)` -> `(B, C, R, R)`.

    When the view is smaller than its box (downsampling), the image is Gaussian-prefiltered with
    sigma = 0.5 * (image pixels per view pixel) before bilinear sampling. That avoids aliasing and makes
    the view content (nearly) independent of the source resolution. Items are grouped by box size so
    each group is blurred once. A single image `(1, C, H, W)` can be shared by any number of boxes; it is
    then blurred once per box size and never copied per view.
    """
    B, C, H, W = boxes.shape[0], *image.shape[1:]
    shared = image.shape[0] == 1 and B > 1
    out = image.new_empty((B, C, R, R))
    grid = view_grid(boxes, R) * 2.0 - 1.0  # grid_sample coordinates (align_corners=True)
    sizes = boxes[:, 2]
    for s in torch.unique(sizes):
        idx = torch.nonzero(sizes == s).flatten()
        step = float(s) * (max(H, W) - 1) / max(R - 1, 1)  # image pixels per view pixel
        src = image if shared else image[idx]
        src = _gaussian_blur(src, 0.5 * step) if step > 1.0 else src
        if shared:
            src = src.expand(idx.numel(), -1, -1, -1)
        out[idx] = F.grid_sample(src, grid[idx], mode="bilinear", padding_mode="border", align_corners=True)
    return out


def view_inputs(
    target: torch.Tensor,
    canvas_rgb: torch.Tensor,
    paint_alpha: torch.Tensor,
    boxes: torch.Tensor,
    R: int,
) -> torch.Tensor:
    """
    Network input for each view, `(B, 12, R, R)`: target RGB, canvas RGB, signed error (target - canvas),
    paint-layer alpha, and the image-space (x, y) coordinates of each view pixel (mapped to [-1, 1]).
    Images may have batch 1 and be shared by all B boxes (e.g. many tiles of one image).
    """
    tgt = resample_views(target, boxes, R)
    cvs = resample_views(canvas_rgb, boxes, R)
    alpha = resample_views(paint_alpha, boxes, R)
    coords = view_grid(boxes, R).permute(0, 3, 1, 2) * 2.0 - 1.0
    return torch.cat([tgt, cvs, tgt - cvs, alpha, coords], dim=1)


def view_to_image(strokes: Strokes, boxes: torch.Tensor) -> Strokes:
    """Maps strokes from view coordinates to image coordinates: p_img = (x, y) + s * p_view."""
    offset = boxes[:, None, None, 0:2]
    s = boxes[:, 2, None, None]
    return Strokes(
        points=offset + s.unsqueeze(-1) * strokes.points,
        radii=s * strokes.radii,
        opacity=strokes.opacity,
        hardness=strokes.hardness,
        color=strokes.color,
    )


def image_to_view(strokes: Strokes, boxes: torch.Tensor) -> Strokes:
    """Inverse of `view_to_image`."""
    offset = boxes[:, None, None, 0:2]
    s = boxes[:, 2, None, None]
    return Strokes(
        points=(strokes.points - offset) / s.unsqueeze(-1),
        radii=strokes.radii / s,
        opacity=strokes.opacity,
        hardness=strokes.hardness,
        color=strokes.color,
    )


def to_engine_params(strokes: Strokes, engine: DifferentiableCanvasEngine, clamp_points: bool = True) -> torch.Tensor:
    """
    Encodes image-space strokes as the engine's `(B, N, 14)` parameters in [0, 1]. Radii are clamped to the
    engine's [min_radius, max_radius]; engines meant for high resolutions can use a smaller `min_radius`.
    With `clamp_points=False`, points may lie outside the canvas (e.g. when rendering a stroke piecewise
    into a crop that only covers part of it).
    """
    B, N = strokes.radii.shape[:2]
    r = (strokes.radii - engine.min_radius) / (engine.max_radius - engine.min_radius)
    a = (strokes.opacity - 0.05) / 0.95
    return torch.cat(
        [
            strokes.points.reshape(B, N, 6).clamp(0.0, 1.0) if clamp_points else strokes.points.reshape(B, N, 6),
            r.clamp(0.0, 1.0),
            a.clamp(0.0, 1.0),
            strokes.hardness,
            strokes.color,
        ],
        dim=-1,
    )


def from_engine_params(params: torch.Tensor, engine: DifferentiableCanvasEngine) -> Strokes:
    """Decodes the engine's `(B, N, 14)` parameters into image-space `Strokes`."""
    B, N = params.shape[:2]
    return Strokes(
        points=params[..., 0:6].reshape(B, N, 3, 2),
        radii=engine.min_radius + params[..., 6:8] * (engine.max_radius - engine.min_radius),
        opacity=0.05 + 0.95 * params[..., 8:10],
        hardness=params[..., 10:11],
        color=params[..., 11:14],
    )
