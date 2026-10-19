"""Image and plot helpers for fitting / training runs (PIL only, no plotting dependencies)."""

import math
from pathlib import Path
from typing import Sequence, Tuple

import torch
from PIL import Image, ImageDraw

PANEL_COLORS = ((200, 60, 40), (40, 90, 200), (40, 150, 70), (150, 70, 170))


def to_uint8(x: torch.Tensor):
    """(3, H, W) tensor in [0, 1] -> (H, W, 3) uint8 array."""
    return (x.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")


def save_frame(path: Path, target: torch.Tensor, canvas: torch.Tensor, text: str):
    """Rows of [target | painting | error] per image, with a caption."""
    B, _, H, W = target.shape
    err = (canvas - target).abs().mean(1, keepdim=True).expand(-1, 3, -1, -1).clamp(0, 1) * 2.0
    rows = [torch.cat([target[i], canvas[i], err[i].clamp(0, 1)], dim=2) for i in range(B)]
    grid = torch.cat(rows, dim=1)
    img = Image.fromarray(to_uint8(grid))
    scale = max(1, 256 // H)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    canvas_img = Image.new("RGB", (img.width, img.height + 18), "white")
    canvas_img.paste(img, (0, 18))
    ImageDraw.Draw(canvas_img).text((4, 3), text, fill="black")
    canvas_img.save(path)


def save_curve(
    path: Path,
    rows: list,
    panels: Sequence[Tuple[str, bool]] = (("loss", True), ("psnr", False)),
    x_key: str = "step",
    markers: Sequence[float] = (),
):
    """
    Minimal line plot: one panel per (key, log_scale) in `panels`, plotted over `x_key`. Rows missing a
    key are skipped for that panel; `markers` draws vertical lines (e.g. stage or epoch starts).
    """
    Wp, Hp, pad = 720, 260, 40
    img = Image.new("RGB", (Wp, Hp * len(panels)), "white")
    d = ImageDraw.Draw(img)
    for panel, (key, log_scale) in enumerate(panels):
        pts = [(r[x_key], r[key]) for r in rows if r.get(key) is not None and (not log_scale or r[key] > 0)]
        oy = panel * Hp
        d.rectangle([pad, oy + pad, Wp - pad, oy + Hp - pad], outline=(180, 180, 180))
        if not pts:
            d.text((pad, oy + 12), f"{key}: no data", fill="black")
            continue
        xs = [p[0] for p in pts]
        vals = [math.log10(p[1]) if log_scale else p[1] for p in pts]
        lo, hi = min(vals), max(vals)
        hi = hi if hi > lo else lo + 1.0
        x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1

        def xy(x, v):
            return (pad + (Wp - 2 * pad) * (x - x0) / (x1 - x0), oy + Hp - pad - (Hp - 2 * pad) * (v - lo) / (hi - lo))

        for m in markers:
            if x0 <= m <= x1:
                d.line([xy(m, lo), xy(m, hi)], fill=(225, 225, 225))
        color = PANEL_COLORS[panel % len(PANEL_COLORS)]
        if len(pts) > 1:
            d.line([xy(x, v) for x, v in zip(xs, vals)], fill=color, width=2)
        else:
            cx, cy = xy(xs[0], vals[0])
            d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color)
        label = f"{'log10 ' if log_scale else ''}{key}: {pts[-1][1]:.4g}  (range {lo:.3g} .. {hi:.3g})"
        d.text((pad, oy + 12), label, fill="black")
    img.save(path)
