"""
Synthetic "partially painted" canvases: the target, degraded in soft-edged regions, as a start state for
training later painting steps without having to paint up to them first.

The degradations are chosen to look like the mistakes of a stroke painting rather than like photo
damage: besides local blur, regions are repainted with rough strokes in the target's local colors (so the
canvas really is a stroke painting there), shifted in color, flattened, left unpainted, or kept at a
coarser level of detail.

Touch-up pairs (`touchup_pair`) cover the other end: a nearly finished painting with a few small, sharp
differences from its target (lines, shapes, hatching, dots, small fills and gradients, missing fine
detail), so the model also practices precise local corrections on canvases where most of the image is
already right.
"""

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.losses import _gaussian_blur
from overpaintai.synthetic import radius_to_param, random_strokes

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
    Missing regions are transparent (alpha 0), so the canvas background shows there; with an opaque engine
    (`engine.opaque`) they are painted over with the background instead, at alpha 1.
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
            gone = mask * (0.5 + 0.5 * _rand(gen, B, 1, 1, 1, device=dev))
            if engine.opaque:
                rgb = rgb + gone * (engine.create_background(B, S, S, dev) - rgb)
            else:
                alpha = alpha * (1.0 - gone)
            continue
        elif kind == "lowres":
            factor = 2 ** int(torch.randint(1, 4, (1,), generator=gen, device=dev).item())
            small = F.interpolate(rgb, scale_factor=1.0 / factor, mode="area")
            new = F.interpolate(small, size=(S, S), mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"unknown degradation {kind!r}")
        rgb = rgb + mask * (new - rgb)

    return torch.cat([rgb.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)], dim=1)



DETAILS = ("line", "shape", "hatch", "dots", "fill", "gradient")
_SLOTS = 6  # strokes per stroke-based detail (unused slots are moved off the canvas)
_OFF = 4.0  # coordinate of unused strokes: far outside the canvas, they paint nothing


def _local_colors(target: torch.Tensor, xy: torch.Tensor, gen, subtle: torch.Tensor) -> torch.Tensor:
    """
    Detail colors `(B, n, 3)` at points `xy` `(B, n, 2)`: where `subtle`, the locally averaged target shifted a
    little (mostly in brightness, like a stroke painting's own mistakes), elsewhere a random color.
    """
    B, n, _ = xy.shape
    dev = target.device
    smooth = _gaussian_blur(target, sigma=max(1.0, 0.02 * target.shape[-1]))
    local = F.grid_sample(smooth, (xy * 2 - 1).unsqueeze(2), mode="bilinear", align_corners=True)
    local = local.squeeze(-1).transpose(1, 2)
    shade = torch.randn(B, n, 1, generator=gen, device=dev) * 0.12
    hue = torch.randn(B, n, 3, generator=gen, device=dev) * 0.04
    near = (local + shade + hue).clamp(0.0, 1.0)
    return torch.where(subtle.unsqueeze(-1), near, _rand(gen, B, n, 3, device=dev))


def _detail_strokes(engine, B: int, n: int, centers, sizes, colors, kind, gen) -> torch.Tensor:
    """
    Stroke params `(B, n * _SLOTS, 14)` for `n` details per image: each detail's kind index `kind` `(B, n)`
    (into DETAILS) decides its strokes; region kinds (fill, gradient) get none.
    """
    dev = centers.device
    S = engine.canvas_size
    k = torch.arange(_SLOTS, device=dev).view(1, 1, _SLOTS).float()
    c = centers.unsqueeze(2)  # (B, n, 1, 2)
    size = sizes.view(B, n, 1, 1)
    ang = _rand(gen, B, n, 1, device=dev) * 2 * math.pi
    d = torch.stack([ang.cos(), ang.sin()], -1)  # (B, n, 1, 2) main direction
    normal = torch.stack([-d[..., 1], d[..., 0]], -1)
    thin = (0.4 + 1.6 * _rand(gen, B, n, 1, device=dev)) / (S - 1)  # line radius: 0.4..2 px

    # line: a polyline of 2-4 connected segments that turn a little at each joint
    turn = (_rand(gen, B, n, _SLOTS, device=dev) - 0.5) * 1.2
    a = ang + turn.cumsum(-1)
    step = torch.stack([a.cos(), a.sin()], -1) * size  # (B, n, slots, 2)
    p0 = c + torch.cat([torch.zeros_like(step[:, :, :1]), step[:, :, :-1]], 2).cumsum(2)
    line = (p0, p0 + step, thin.expand(-1, -1, _SLOTS), k < 2 + (_rand(gen, B, n, 1, device=dev) * 3).floor())
    # shape: 1-3 short fat overlapping strokes (a blob)
    jit = (_rand(gen, B, n, _SLOTS, 2, device=dev) - 0.5) * size * 0.5
    seg = d * size * 0.3
    shape = (c + jit - seg, c + jit + seg, (size[..., 0] * (0.2 + 0.3 * _rand(gen, B, n, _SLOTS, device=dev))),
             k < 1 + (_rand(gen, B, n, 1, device=dev) * 3).floor())
    # hatch: 3-6 parallel thin strokes
    off = c + normal * (k.unsqueeze(-1) - 2.5) * size * 0.15
    hatch = (off - d * size / 2, off + d * size / 2, thin.expand(-1, -1, _SLOTS),
             k < 3 + (_rand(gen, B, n, 1, device=dev) * 4).floor())
    # dots: up to 6 small dabs in a cluster
    dot = c + (_rand(gen, B, n, _SLOTS, 2, device=dev) - 0.5) * size
    dots = (dot, dot, thin.expand(-1, -1, _SLOTS) * 1.5, k < 3 + (_rand(gen, B, n, 1, device=dev) * 4).floor())

    start, end, radius, used = [torch.zeros_like(x) for x in line]
    for i, (s0, s1, r, u) in enumerate((line, shape, hatch, dots)):
        pick = (kind == i).unsqueeze(-1)  # (B, n, 1)
        start = torch.where(pick.unsqueeze(-1), s0, start)
        end = torch.where(pick.unsqueeze(-1), s1, end)
        radius = torch.where(pick, r, radius)
        used = torch.where(pick, u, used)
    used = used.bool() & (kind < 4).unsqueeze(-1)
    off_canvas = torch.full_like(start, _OFF)
    start = torch.where(used.unsqueeze(-1), start, off_canvas)
    end = torch.where(used.unsqueeze(-1), end, off_canvas)
    mid = 0.5 * (start + end) + normal * size * (_rand(gen, B, n, _SLOTS, 1, device=dev) - 0.5) * 0.3
    r = radius_to_param(engine, radius).unsqueeze(-1).expand(-1, -1, -1, 2)
    opacity = (0.75 + 0.25 * _rand(gen, B, n, 1, 1, device=dev)).expand(-1, -1, _SLOTS, 2)
    hard = (0.5 + 0.5 * _rand(gen, B, n, 1, 1, device=dev)).expand(-1, -1, _SLOTS, 1)
    rgb = colors.unsqueeze(2).expand(-1, -1, _SLOTS, -1)
    params = torch.cat([start, mid, end, r, opacity, hard, rgb], -1)  # (B, n, slots, 14)
    return params.reshape(B, n * _SLOTS, 14)


def _region_details(B: int, S: int, n: int, centers, sizes, colors, kind, gen) -> torch.Tensor:
    """Small fills and gradients as an RGBA layer `(B, 4, S, S)` (straight color, alpha)."""
    dev = centers.device
    yy = torch.linspace(0.0, 1.0, S, device=dev).view(1, 1, S, 1)
    xx = torch.linspace(0.0, 1.0, S, device=dev).view(1, 1, 1, S)
    cx, cy = centers[..., 0].view(B, n, 1, 1), centers[..., 1].view(B, n, 1, 1)
    r = (sizes / 2).view(B, n, 1, 1)
    aspect = 0.5 + _rand(gen, B, n, 1, 1, device=dev)
    dist = torch.sqrt(((xx - cx) / (r * aspect)) ** 2 + ((yy - cy) * aspect / r) ** 2)
    sharp = 6.0 + 30.0 * _rand(gen, B, n, 1, 1, device=dev)  # soft to crisp edges
    mask = torch.sigmoid((1.0 - dist) * sharp) * (kind >= 4).view(B, n, 1, 1)  # (B, n, S, S)
    ang = _rand(gen, B, n, 1, 1, device=dev) * 2 * math.pi
    t = (((xx - cx) * ang.cos() + (yy - cy) * ang.sin()) / (2 * r) + 0.5).clamp(0.0, 1.0)
    other = _rand(gen, B, n, 3, device=dev)
    grad = (kind == 5).view(B, n, 1, 1, 1)
    c0 = colors.view(B, n, 3, 1, 1)
    col = torch.where(grad, c0 + t.unsqueeze(2) * (other.view(B, n, 3, 1, 1) - c0), c0.expand(-1, -1, -1, S, S))
    rgb = torch.zeros(B, 3, S, S, device=dev)
    alpha = torch.zeros(B, 1, S, S, device=dev)
    for i in range(n):  # source over, one region after another
        m = mask[:, i : i + 1]
        a_new = m + alpha * (1 - m)
        rgb = torch.where(a_new > 1e-6, (m * col[:, i] + (1 - m) * alpha * rgb) / a_new.clamp(min=1e-6), rgb)
        alpha = a_new
    return torch.cat([rgb, alpha], 1)


@torch.no_grad()
def touchup_pair(
    target: torch.Tensor,
    engine: DifferentiableCanvasEngine,
    generator: Optional[torch.Generator] = None,
    add_share: float = 0.5,
    strength: float = 1.0,
    target_hr: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    A nearly finished painting and its target, `(paint layer (B, 4, S, S), target (B, 3, S, S), target_hr)`: the target
    slightly softened (like a stroke painting of it, not a pixel copy), with fine detail missing in a few small
    regions, and a few small details (DETAILS, each 1.5-12% of the side, drawn with the engine's strokes; colors
    either close to the image's or random). Per image, with probability `add_share` the details are added to the
    target (the model must paint them), otherwise they sit on the canvas (the model must paint them away and
    restore the image there). `strength` scales the number of details. `target_hr` (B, 3, H, H): the same target at
    a higher resolution (`target` its area-downsampled version); the details are then drawn at that resolution, the
    returned `target_hr` gets them too and the returned target is its downsampled version (None without one).
    """
    B, _, S, _ = target.shape
    dev = target.device
    gen = generator
    base = _gaussian_blur(target, 0.3 + 0.7 * _rand(gen, 1, device=dev).item())
    lost = region_masks(B, S, S, int(torch.randint(1, 4, (1,), generator=gen, device=dev).item()),
                        size_range=(0.02, 0.1), gen=gen, device=dev)
    base = base + lost * (_gaussian_blur(target, 1.0 + 2.0 * _rand(gen, 1, device=dev).item()) - base)

    n = max(1, round(strength * int(torch.randint(2, 7, (1,), generator=gen, device=dev).item())))
    centers = 0.05 + 0.9 * _rand(gen, B, n, 2, device=dev)
    sizes = 0.015 * 8.0 ** _rand(gen, B, n, device=dev)  # 1.5..12% of the side, log-uniform
    kind = torch.randint(0, len(DETAILS), (B, n), generator=gen, device=dev)
    colors = _local_colors(target, centers, gen, _rand(gen, B, n, device=dev) < 0.5)
    strokes = _detail_strokes(engine, B, n, centers, sizes, colors, kind, gen)
    H = S if target_hr is None else target_hr.shape[-1]
    regions = _region_details(B, H, n, centers, sizes, colors, kind, gen)
    detail = engine(regions, strokes)  # strokes over the regions, on a transparent layer
    premult = torch.cat([detail[:, :3] * detail[:, 3:], detail[:, 3:]], 1)
    over_at = lambda img, pm: img * (1 - pm[:, 3:]) + pm[:, :3]
    small = premult if H == S else F.interpolate(premult, size=(S, S), mode="area")

    add = (_rand(gen, B, 1, 1, 1, device=dev) < add_share)
    canvas = torch.where(add, base, over_at(base, small))
    new_hr = None
    if target_hr is None:
        new_target = torch.where(add, over_at(target, small), target)
    else:
        new_hr = torch.where(add, over_at(target_hr, premult), target_hr).clamp(0.0, 1.0)
        new_target = torch.where(add, F.interpolate(new_hr, size=(S, S), mode="area"), target)
    layer = torch.cat([canvas.clamp(0.0, 1.0), torch.ones_like(canvas[:, :1])], dim=1)
    return layer, new_target.clamp(0.0, 1.0), new_hr
