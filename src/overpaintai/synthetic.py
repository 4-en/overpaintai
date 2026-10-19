import math
from typing import Optional, Sequence, Tuple

import torch

from overpaintai.canvas_engine import DifferentiableCanvasEngine


def radius_to_param(engine: DifferentiableCanvasEngine, radius: torch.Tensor) -> torch.Tensor:
    """Inverse of the engine's radius mapping: canvas-unit radius -> stroke parameter in [0, 1]."""
    return ((radius - engine.min_radius) / (engine.max_radius - engine.min_radius)).clamp(0.0, 1.0)


def random_strokes(
    engine: DifferentiableCanvasEngine,
    batch_size: int,
    num_strokes: int,
    tool_probs: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 0.0),
    radius_range: Tuple[float, float] = (0.12, 0.012),
    palette_size: Tuple[int, int] = (3, 6),
    num_washes: int = 0,
    wash_radius: Tuple[float, float] = (0.3, 0.75),
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples painting-like stroke sequences: `(B, K, 14)` stroke params and `(B, K, 5)` one-hot tools.

    Strokes go coarse to fine (radius shrinks log-linearly from `radius_range[0]` to `radius_range[1]`
    over the sequence, with jitter), are gently curved and tapered, and draw their colors from a small
    per-image palette with jitter, so images look like blocked-in paintings rather than confetti.
    The first `num_washes` strokes are large, mostly soft washes (radius in `wash_radius`, up to filling the
    canvas), like toning a canvas or blocking in a background before painting.
    """
    B, K = batch_size, num_strokes
    dev = device or engine.bg_color.device
    g = generator

    def rand(*shape):
        return torch.rand(*shape, generator=g, device=dev)

    # Coarse-to-fine radius schedule with jitter
    frac = torch.linspace(0.0, 1.0, K, device=dev).view(1, K)
    log_r = math.log(radius_range[0]) + frac * (math.log(radius_range[1]) - math.log(radius_range[0]))
    r0 = torch.exp(log_r + 0.35 * torch.randn(B, K, generator=g, device=dev))
    if num_washes > 0:
        w = min(num_washes, K)
        lo, hi = wash_radius
        r0[:, :w] = lo + (hi - lo) * rand(B, w)
    r2 = r0 * (0.5 + 0.7 * rand(B, K))
    if num_washes > 0:
        r2[:, :w] = r0[:, :w] * (0.7 + 0.4 * rand(B, w))

    # Curved strokes: random start, direction and length (in radii), sideways bend of the control point
    start = rand(B, K, 2)
    angle = 2.0 * math.pi * rand(B, K)
    length = r0 * (2.0 + 8.0 * rand(B, K))
    if num_washes > 0:
        length[:, :w] = r0[:, :w] * (0.3 + 1.5 * rand(B, w))  # washes are broad rather than long
    direction = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    end = (start + direction * length.unsqueeze(-1)).clamp(0.0, 1.0)
    normal = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)
    bend = (rand(B, K) - 0.5) * length
    ctrl = (0.5 * (start + end) + normal * bend.unsqueeze(-1)).clamp(0.0, 1.0)

    # Per-image palette; each stroke picks a palette color plus a little jitter
    n_colors = torch.randint(palette_size[0], palette_size[1] + 1, (B,), generator=g, device=dev)
    palette = rand(B, palette_size[1], 3)
    pick = (rand(B, K) * n_colors.view(B, 1)).long()
    color = torch.gather(palette, 1, pick.unsqueeze(-1).expand(B, K, 3))
    color = (color + 0.08 * torch.randn(B, K, 3, generator=g, device=dev)).clamp(0.0, 1.0)

    opacity = 0.6 + 0.4 * rand(B, K, 2)
    hardness = rand(B, K, 1)
    if num_washes > 0:
        opacity[:, :w] = 0.7 + 0.3 * rand(B, w, 2)
        hardness[:, :w] = 0.6 * rand(B, w, 1)

    params = torch.cat(
        [
            start, ctrl, end,
            radius_to_param(engine, r0).unsqueeze(-1), radius_to_param(engine, r2).unsqueeze(-1),
            opacity, hardness, color,
        ],
        dim=-1,
    )  # [x0, y0, x1, y1, x2, y2, r0, r2, a0, a2, hardness, R, G, B]

    probs = torch.tensor(tool_probs, device=dev, dtype=torch.float32)
    tool_idx = torch.multinomial(probs / probs.sum(), B * K, replacement=True, generator=g).view(B, K)
    tools = torch.nn.functional.one_hot(tool_idx, engine.NUM_TOOLS).float()
    return params, tools


@torch.no_grad()
def render_targets(
    engine: DifferentiableCanvasEngine,
    params: torch.Tensor,
    tools: torch.Tensor,
    size: int,
    background: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Renders stroke sequences onto a blank layer and composites them: `(B, 3, size, size)` RGB."""
    B = params.shape[0]
    layer = engine.create_blank_paint_layer(B, size, size, params.device)
    layer = engine(layer, params, tools)
    if background is None:
        background = engine.create_background(B, size, size, params.device)
    return engine.composite(layer, background)
