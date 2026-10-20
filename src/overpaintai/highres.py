"""
High-resolution painting: coarse whole-image steps, then error-guided tiles of decreasing size.

The model always works on views resampled to its working resolution R, so a large canvas is painted the
way the README's "High-Resolution Inference Harness" describes:
  1. whole-image steps on the downscaled image (blocking in),
  2. levels of square tiles (core size 1/2, 1/4, ... of the image), each round picking the tiles with the
     highest (optionally importance-weighted) error, with non-maximum suppression,
  3. every tile view = its core plus a context margin; strokes anchored outside the core, or reaching past
     what the view shows, are faded out smoothly (never clipped) so tiles don't paint blind,
  4. strokes are rendered only into a square crop of the canvas around their bounding box, which gives the
     same result as a full-canvas render at a fraction of the cost.
Nothing here is trained; it only schedules the views a trained model paints.
"""

import io
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from overpaintai import views
from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.model import StrokePredictor


@dataclass
class HighResConfig:
    whole_steps: int = 6  # whole-image steps before tiling; step k gets progress k / whole_steps
    tile_progress: float = 6 / 7  # progress fed to tile steps (training uses k/total for its tile step)
    rounds_per_level: int = 2  # tile rounds per level
    max_tiles: int = 32  # tiles per round (one batch)
    margin: float = 0.25  # context margin on each side of a tile core, as a fraction of the core size
    error_threshold: float = 0.015  # tiles whose mean core error is below this are skipped
    nms_iou: float = 0.3  # candidate cores overlapping a chosen one more than this are suppressed
    detail_limit: float = 1.0  # stop once a view would have fewer than R * detail_limit source pixels
    min_core: Optional[float] = None  # explicit smallest core size (overrides the source-detail stop)
    scale_clamp: Tuple[float, float] = (0.3, 1.0)  # view-size range fed to the model (trained range)
    fade: bool = True  # fade strokes near / beyond the tile edge instead of keeping them as is
    analysis_res: int = 512  # resolution of the error map used for tile selection
    importance_gain: float = 2.0  # error weight = 1 + gain * importance mask


@dataclass
class PaintResult:
    layer: torch.Tensor  # (1, 4, S, S)
    canvas: torch.Tensor  # (1, 3, S, S)
    snapshots: List[Tuple[str, torch.Tensor]] = field(default_factory=list)  # (label, canvas)
    stats: List[Dict] = field(default_factory=list)
    strokes: List[torch.Tensor] = field(default_factory=list)  # engine-independent stroke records


class PaintCancelled(Exception):
    """Raised by `HighResPainter.paint` when its `should_stop` callback returns True."""


def load_padded(source: Union[str, Path, bytes, Image.Image], size: int, device, mode: str = "RGB"):
    """
    Loads an image (path, encoded bytes or PIL image) with its longer side resized to `size` and pads it
    to a square (edge-replicated, so the model sees plausible content there). Transparent images are
    composited onto white. Returns (image (1, C, S, S), valid mask (1, 1, S, S), (w, h) of the content,
    longer source side in pixels).
    """
    if isinstance(source, Image.Image):
        img = source
    else:
        img = Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
    if mode == "RGB" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    img = img.convert(mode)
    src_side = max(img.size)
    scale = size / src_side
    w, h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    img = img.resize((w, h), Image.LANCZOS if scale < 1 else Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype="float32") / 255.0)
    x = x.view(h, w, -1).permute(2, 0, 1).unsqueeze(0)
    x = F.pad(x, (0, size - w, 0, size - h), mode="replicate")
    valid = torch.zeros(1, 1, size, size)
    valid[..., :h, :w] = 1.0
    return x.to(device), valid.to(device), (w, h), src_side


def config_from_checkpoint(ckpt: Dict, **overrides) -> Tuple[HighResConfig, int, int]:
    """
    Harness settings matching how a checkpoint was trained (whole-image steps, tile-step progress, trained
    tile-size range), with `overrides` applied (any `HighResConfig` field; None values are ignored).
    Returns (config, view resolution, tool index).
    """
    train_args = ckpt.get("args", {})
    steps = int(train_args.get("steps", 6))
    tile_lo = float(str(train_args.get("tile_size", "0.3,0.6")).split(",")[0])
    cfg = HighResConfig(whole_steps=steps, tile_progress=steps / (steps + 1), scale_clamp=(tile_lo, 1.0))
    for key, value in overrides.items():
        if value is not None:
            if not hasattr(cfg, key):
                raise TypeError(f"unknown HighResConfig field {key!r}")
            setattr(cfg, key, value)
    return cfg, int(ckpt.get("view_res", 128)), int((ckpt.get("tools") or [0])[0])


def load_checkpoint_model(path: str, device: torch.device) -> Tuple[StrokePredictor, Dict]:
    """Rebuilds the model of a training checkpoint (EMA weights when available)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = StrokePredictor(**ckpt["model_config"])
    model.load_state_dict(ckpt.get("ema") or ckpt["model"])
    return model.to(device).eval(), ckpt


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class CropRenderer:
    """
    Renders image-space strokes onto a square paint layer of `size` px, each batch of strokes only inside a
    square crop of the canvas that contains their footprint (identical to a full-canvas render, since the
    renderer is resolution-independent, at a fraction of the cost). Shared by painting, video replay and
    high-resolution export. Footprints larger than `max_crop` px are rendered in squares of at most that
    size, which bounds memory on very large canvases (exact for the paint tool, whose result per pixel only
    depends on that pixel).
    """

    def __init__(
        self, size: int, num_stamps: int, min_radius: float, max_radius: float, device, tool: int = 0,
        max_crop: int = 2048, adaptive_stamps: bool = False,
    ):
        """
        adaptive_stamps: choose the curve samples per call from the strokes' length in pixels (about one
            segment per 16 px, between 8 and `num_stamps`) instead of always `num_stamps`; much cheaper for
            the many small detail strokes, visually the same (used for replay / export).
        """
        self.S = size
        self.max_crop = max_crop
        self.adaptive_stamps = adaptive_stamps
        self.num_stamps = num_stamps
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.device = torch.device(device)
        self.tool = tool
        self.engine = self._engine(size, 1.0)
        self.background = self.engine.create_background(1, size, size, self.device)
        self._crop_engines: Dict[int, DifferentiableCanvasEngine] = {}

    def _engine(self, n: int, scale: float) -> DifferentiableCanvasEngine:
        """Engine for an n-px canvas covering `scale` of the image side (radii in its own units)."""
        return DifferentiableCanvasEngine(
            canvas_size=n, num_stamps=self.num_stamps,
            min_radius=self.min_radius / scale, max_radius=self.max_radius / scale,
        ).to(self.device).eval()

    def _stamps_for(self, strokes: views.Strokes) -> int:
        if not self.adaptive_stamps:
            return self.num_stamps
        p = strokes.points[0]  # (N, 3, 2): control polygon length bounds the curve length
        length = ((p[:, 1] - p[:, 0]).norm(dim=-1) + (p[:, 2] - p[:, 1]).norm(dim=-1)).max().item() * (self.S - 1)
        stamps = 8 * math.ceil((length / 16.0 + 1) / 8)
        return int(min(self.num_stamps, max(8, stamps)))  # multiples of 8: few distinct shapes to compile

    def blank(self) -> torch.Tensor:
        return self.engine.create_blank_paint_layer(1, self.S, self.S, self.device)

    def composite(self, layer: torch.Tensor) -> torch.Tensor:
        return self.engine.composite(layer, self.background)

    @torch.no_grad()
    def render(self, layer: torch.Tensor, strokes: views.Strokes) -> torch.Tensor:
        """Renders image-space strokes `(1, N)` in order onto `layer` (in place) and returns it."""
        if strokes.radii.shape[1] == 0:
            return layer
        S = self.S
        # Points are kept on the canvas once, here; crops then take them as they are (unclamped), so a
        # stroke split over several crops keeps its exact shape
        strokes = views.Strokes(strokes.points.clamp(0.0, 1.0), strokes.radii, strokes.opacity,
                                strokes.hardness, strokes.color)
        r = strokes.radii.amax(-1)  # (1, N)
        soft = (strokes.hardness[..., 0] < 0.999).float()
        ext = r * (1.0 + 1.15 * soft) + 3.0 / (S - 1)  # soft profile reaches ~2.15 r
        pts = strokes.points  # (1, N, 3, 2)
        lo = (pts.amin(dim=2) - ext.unsqueeze(-1)).amin(dim=1)[0].clamp(0, 1)  # (2,)
        hi = (pts.amax(dim=2) + ext.unsqueeze(-1)).amax(dim=1)[0].clamp(0, 1)
        x0, y0 = math.floor(lo[0].item() * (S - 1)), math.floor(lo[1].item() * (S - 1))
        x1, y1 = math.ceil(hi[0].item() * (S - 1)), math.ceil(hi[1].item() * (S - 1))
        n = min(S, max(x1 - x0, y1 - y0) + 1)
        stamps = self._stamps_for(strokes)
        if n <= self.max_crop:
            self._render_square(layer, strokes, min(x0, S - n), min(y0, S - n), n, stamps)
        else:  # split huge footprints into squares (the last ones may reach past the canvas: padded)
            m = self.max_crop
            for py in range(y0, y1 + 1, m):
                for px in range(x0, x1 + 1, m):
                    self._render_square(layer, strokes, px, py, m, stamps)
        return layer

    def _render_square(self, layer: torch.Tensor, strokes: views.Strokes, x0: int, y0: int, n: int, stamps: int):
        """Renders `strokes` into the n x n square at pixel (x0, y0); parts outside the canvas are padding."""
        S = self.S
        box = torch.tensor([[x0 / (S - 1), y0 / (S - 1), (n - 1) / (S - 1)]], device=self.device)
        if n not in self._crop_engines:
            self._crop_engines[n] = self._engine(n, (n - 1) / (S - 1)) if n != S else self.engine
        eng = self._crop_engines[n]
        if eng.num_stamps != stamps:  # one engine per crop size; only its curve sampling changes
            eng.num_stamps = stamps
            eng.t = torch.linspace(0.0, 1.0, stamps, device=self.device).view(1, stamps, 1, 1)
        params = views.to_engine_params(views.image_to_view(strokes, box), eng, clamp_points=False)
        tools = torch.zeros(1, params.shape[1], eng.NUM_TOOLS, device=self.device)
        tools[..., self.tool] = 1.0
        h, w = min(n, S - y0), min(n, S - x0)
        if h == n and w == n:
            crop = layer[:, :, y0 : y0 + n, x0 : x0 + n].contiguous()
        else:
            crop = layer.new_zeros((1, 4, n, n))
            crop[:, :, :h, :w] = layer[:, :, y0 : y0 + h, x0 : x0 + w]
        layer[:, :, y0 : y0 + h, x0 : x0 + w] = eng(crop, params, tools)[:, :, :h, :w]  # in place (no autograd)


def pack_strokes(s: views.Strokes) -> torch.Tensor:
    """Image-space strokes `(1, N)` as `(N, 14)`: points (6), radii (2), opacity (2), hardness, color (3)."""
    N = s.radii.shape[1]
    return torch.cat([s.points.reshape(1, N, 6), s.radii, s.opacity, s.hardness, s.color], dim=-1)[0]


def unpack_strokes(t: torch.Tensor) -> views.Strokes:
    """Inverse of `pack_strokes`: `(N, 14)` -> `Strokes` with batch 1."""
    t = t.unsqueeze(0)
    N = t.shape[1]
    return views.Strokes(points=t[..., 0:6].reshape(1, N, 3, 2), radii=t[..., 6:8], opacity=t[..., 8:10],
                         hardness=t[..., 10:11], color=t[..., 11:14])


def render_strokes(
    strokes: torch.Tensor,
    canvas_size: int,
    content_wh: Tuple[int, int],
    output_side: int,
    num_stamps: int = 32,
    min_radius: float = 1e-4,
    max_radius: float = 0.75,
    device="cpu",
    chunk: int = 16,
    on_progress: Optional[Callable[[float], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> torch.Tensor:
    """
    Re-renders recorded strokes (`pack_strokes` format, coordinates of the square `canvas_size` canvas with
    `content_wh` content) so that the content's longer side is `output_side` px. Strokes are
    resolution-independent, so this is a real upscale: hard edges stay one pixel wide at the new size.
    Curve samples grow with the resolution so long strokes stay smooth. Returns the content `(1, 3, h, w)`.
    """
    w, h = content_wh
    S = max(16, round(canvas_size * output_side / max(w, h)))
    # Max curve samples grow with sqrt(resolution) (flattening error ~ segment length^2); per chunk the
    # renderer picks fewer for short strokes
    stamps = int(min(128, max(num_stamps, round(num_stamps * math.sqrt(S / max(canvas_size, 1))))))
    renderer = CropRenderer(S, stamps, min_radius, max_radius, device, adaptive_stamps=True)
    strokes = strokes.to(renderer.device).float()
    layer = renderer.blank()
    N = strokes.shape[0]
    for i in range(0, N, chunk):  # chunks follow the recording (<= 16 strokes per tile), keeping crops local
        renderer.render(layer, unpack_strokes(strokes[i : i + chunk]))
        if should_stop is not None and should_stop():
            raise PaintCancelled()
        if on_progress is not None and (i // chunk) % 20 == 0:
            on_progress(min(1.0, (i + chunk) / max(N, 1)))
    out_w, out_h = round(w * S / canvas_size), round(h * S / canvas_size)
    return renderer.composite(layer)[:, :, :out_h, :out_w]


def radius_limits(model: StrokePredictor, size: int) -> Tuple[float, float]:
    """Engine radius range for painting at `size` px: down to the model's smallest stroke in the smallest
    tile (but not below a quarter pixel), up to canvas-filling."""
    lo, hi = model.radius_range
    return max(0.25 / (size - 1), lo / 64.0), hi


class HighResPainter:
    """
    Args:
        model: trained `StrokePredictor` (eval mode).
        size: output canvas resolution S (square).
        view_res: the model's working resolution R.
        tool: engine tool index used for all strokes.
        num_stamps: curve samples per stroke for rendering (more = smoother long strokes).
        autocast: dtype for the network, or None.
    """

    def __init__(
        self,
        model: StrokePredictor,
        size: int,
        view_res: int = 128,
        tool: int = 0,
        num_stamps: int = 32,
        config: Optional[HighResConfig] = None,
        autocast: Optional[torch.dtype] = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.S = size
        self.R = view_res
        self.tool = tool
        self.num_stamps = num_stamps
        self.cfg = config or HighResConfig()
        self.autocast = autocast
        self.device = device or next(model.parameters()).device
        self.min_radius, self.max_radius = radius_limits(model, size)
        self.renderer = CropRenderer(size, num_stamps, self.min_radius, self.max_radius, self.device, tool)
        self.engine = self.renderer.engine
        self.background = self.renderer.background

    def composite(self, layer: torch.Tensor) -> torch.Tensor:
        return self.renderer.composite(layer)

    def render(self, layer: torch.Tensor, strokes: views.Strokes) -> torch.Tensor:
        return self.renderer.render(layer, strokes)

    # --- model ------------------------------------------------------------------------------------
    def _autocast(self):
        enabled = self.autocast is not None and self.device.type == "cuda"
        return torch.autocast(self.device.type, dtype=self.autocast or torch.bfloat16, enabled=enabled)

    @torch.no_grad()
    def predict(
        self, target: torch.Tensor, layer: torch.Tensor, boxes: torch.Tensor, progress: float,
        global_tokens: Optional[torch.Tensor] = None,
    ):
        T = boxes.shape[0]
        canvas = self.composite(layer)
        x = views.view_inputs(target, canvas, layer[:, 3:], boxes, self.R)  # one image, T views
        lo, hi = self.cfg.scale_clamp
        prog = torch.full((T,), progress, device=self.device)
        with self._autocast():
            return self.model(x, boxes, prog, global_tokens=global_tokens, view_size=boxes[:, 2].clamp(lo, hi))

    @torch.no_grad()
    def global_tokens(self, target: torch.Tensor, layer: torch.Tensor, progress: float) -> torch.Tensor:
        boxes = views.whole_image_boxes(1, self.device)
        x = views.view_inputs(target, self.composite(layer), layer[:, 3:], boxes, self.R)
        with self._autocast():
            return self.model.encode_global(x, torch.full((1,), progress, device=self.device))

    # --- tiles ------------------------------------------------------------------------------------
    def error_map(self, target_a: torch.Tensor, layer: torch.Tensor, weight_a: torch.Tensor) -> torch.Tensor:
        """Mean absolute error at the analysis resolution, times the importance / validity weight."""
        A = target_a.shape[-1]
        canvas_a = F.interpolate(self.composite(layer), size=(A, A), mode="area")
        return (target_a - canvas_a).abs().mean(1, keepdim=True) * weight_a  # (1, 1, A, A)

    def select_tiles(self, err: torch.Tensor, core: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Picks up to `max_tiles` core windows of side `core` (stride core/2) with the highest mean error,
        with greedy non-maximum suppression. Returns cores `(T, 3)` and their scores `(T,)`.
        """
        A = err.shape[-1]
        k = max(1, round(core * A))
        stride = max(1, k // 2)
        scores = F.avg_pool2d(err, k, stride=stride).flatten()  # windows at multiples of stride
        n_side = (A - k) // stride + 1
        order = torch.argsort(scores, descending=True)
        chosen, chosen_scores = [], []
        for i in order.tolist():
            if scores[i] < self.cfg.error_threshold or len(chosen) >= self.cfg.max_tiles:
                break
            yi, xi = divmod(i, n_side)
            box = (xi * stride / A, yi * stride / A, k / A)
            if all(self._iou(box, c) <= self.cfg.nms_iou for c in chosen):
                chosen.append(box)
                chosen_scores.append(scores[i].item())
        return torch.tensor(chosen, device=err.device).view(-1, 3), torch.tensor(chosen_scores)

    @staticmethod
    def _iou(a, b) -> float:
        ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[1] + a[2], b[1] + b[2]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / (a[2] ** 2 + b[2] ** 2 - inter)

    def view_boxes(self, cores: torch.Tensor) -> torch.Tensor:
        """Core boxes enlarged by the context margin on each side, shifted (not shrunk) into the image."""
        v = (cores[:, 2] * (1.0 + 2.0 * self.cfg.margin)).clamp(max=1.0)
        xy = cores[:, :2] - 0.5 * (v - cores[:, 2]).unsqueeze(-1)
        xy = torch.minimum(xy.clamp(min=0.0), (1.0 - v).unsqueeze(-1))
        return torch.cat([xy, v.unsqueeze(-1)], dim=-1)

    def fade(self, strokes: views.Strokes, anchors: torch.Tensor, cores: torch.Tensor, boxes: torch.Tensor):
        """
        Per-stroke opacity factor `(T, N)` in [0, 1] (view-space strokes): fades strokes whose anchor lies
        outside the tile core, and strokes whose footprint reaches past the visible view, smoothly over one
        margin width. Nothing is clipped, so faded strokes keep soft, natural edges.
        """
        s = boxes[:, 2:3]
        core_lo = (cores[:, :2] - boxes[:, :2]) / s  # core in view coords, (T, 2)
        core_hi = core_lo + cores[:, 2:3] / s
        width = (self.cfg.margin * cores[:, 2:3] / s).clamp(min=1e-3)  # margin in view units, (T, 1)

        out_core = torch.maximum(core_lo.unsqueeze(1) - anchors, anchors - core_hi.unsqueeze(1)).clamp(min=0)
        f_anchor = 1.0 - _smoothstep(out_core.amax(-1) / width)

        r = strokes.radii.amax(-1, keepdim=True)  # (T, N, 1)
        reach_lo = (strokes.points.amin(dim=2) - r).clamp(max=0).neg()  # beyond the view's 0 edge
        reach_hi = (strokes.points.amax(dim=2) + r - 1.0).clamp(min=0)
        f_reach = 1.0 - _smoothstep(torch.maximum(reach_lo, reach_hi).amax(-1) / width)
        return f_anchor * f_reach

    # --- full schedule --------------------------------------------------------------------------------
    def core_sizes(self, source_side: int) -> List[float]:
        """Tile core sizes per level: halving until views would hold fewer source pixels than R."""
        sizes = []
        core = 0.5
        while True:
            view = min(1.0, core * (1.0 + 2.0 * self.cfg.margin))
            if self.cfg.min_core is not None:
                if core < self.cfg.min_core - 1e-9:
                    break
            elif view * source_side < self.R * self.cfg.detail_limit:
                break
            sizes.append(core)
            core /= 2.0
        return sizes

    @torch.no_grad()
    def paint(
        self,
        target: torch.Tensor,
        source_side: int,
        valid: Optional[torch.Tensor] = None,
        importance: Optional[torch.Tensor] = None,
        log=print,
        on_progress: Optional[Callable[[Dict, Callable[[], torch.Tensor]], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        on_stage: Optional[Callable[[Dict], None]] = None,
        initial_layer: Optional[torch.Tensor] = None,
        whole_done: int = 0,
    ) -> PaintResult:
        """
        Args:
            target: (1, 3, S, S) target at the output resolution.
            source_side: longer side of the original image in pixels (limits how fine tiles go).
            valid: optional (1, 1, S, S) mask of real image pixels (padding = 0); excluded from selection.
            importance: optional (1, 1, S, S) map in [0, 1]; high values attract more / smaller tiles.
            on_progress: called after every whole-image step and every tile round as
                `on_progress(event, get_canvas)`; `event` holds the stage, step / round, strokes so far,
                elapsed seconds and an approximate overall `fraction`; `get_canvas()` composites the current
                canvas `(1, 3, S, S)` on demand (so callers can throttle previews cheaply).
            should_stop: checked at the same points; returning True raises `PaintCancelled`.
            on_stage: called with each stage's stats (stage, strokes, tiles, psnr, seconds) when it ends.
            initial_layer: (1, 4, S, S) paint layer to continue from (e.g. an earlier result) instead of a
                blank canvas; it is not modified.
            whole_done: whole-image steps already painted on `initial_layer`, so progress continues from
                there: step k gets progress (whole_done + k) / (whole_done + whole_steps).
        """
        cfg, A = self.cfg, min(self.cfg.analysis_res, self.S)
        t0 = time.time()
        layer = self.renderer.blank() if initial_layer is None else initial_layer.to(self.device).float().clone()
        result = PaintResult(layer=layer, canvas=self.composite(layer))
        target_a = F.interpolate(target, size=(A, A), mode="area")
        weight_a = torch.ones(1, 1, A, A, device=self.device)
        if valid is not None:
            weight_a = weight_a * F.interpolate(valid, size=(A, A), mode="area")
        if importance is not None:
            weight_a = weight_a * (1.0 + cfg.importance_gain * F.interpolate(importance, size=(A, A), mode="area"))
        n_strokes = 0
        cores_plan = self.core_sizes(source_side) if cfg.rounds_per_level > 0 else []
        total_units = max(1, cfg.whole_steps + len(cores_plan) * cfg.rounds_per_level)
        units_done = 0

        def progress(event: Dict, units: int):
            nonlocal units_done
            units_done = units
            if on_progress is not None:
                event.update(strokes=n_strokes, seconds=time.time() - t0,
                             fraction=min(1.0, units_done / total_units))
                on_progress(event, lambda: self.composite(result.layer))
            if should_stop is not None and should_stop():
                raise PaintCancelled()

        def record(label: str, tiles: int = 0):
            canvas = self.composite(result.layer)
            mse = ((canvas - target) ** 2)
            mse = (mse * valid).sum() / (valid.sum() * 3) if valid is not None else mse.mean()
            stat = {"stage": label, "strokes": n_strokes, "tiles": tiles, "psnr": (-10 * torch.log10(mse)).item(),
                    "seconds": time.time() - t0}
            result.stats.append(stat)
            result.snapshots.append((label, canvas))
            log(f"{label:>18s}: {n_strokes:5d} strokes, PSNR {stat['psnr']:.2f} dB, {stat['seconds']:.1f}s")
            if on_stage is not None:
                on_stage(stat)

        # 1. Whole-image steps
        whole = views.whole_image_boxes(1, self.device)
        whole_total = max(1, whole_done + cfg.whole_steps)
        for k in range(cfg.whole_steps):
            out = self.predict(target, result.layer, whole, (whole_done + k) / whole_total)
            strokes = views.view_to_image(out.strokes, whole)
            result.layer = self.render(result.layer, strokes)
            result.strokes.append(pack_strokes(strokes).cpu())
            n_strokes += strokes.radii.shape[1]
            progress({"stage": "whole image", "kind": "whole", "step": k + 1, "steps": cfg.whole_steps}, k + 1)
        if cfg.whole_steps > 0:
            record("whole image")

        # 2. Tile levels
        for level, core in enumerate(cores_plan):
            level_tiles = 0
            label = f"tiles 1/{round(1 / core)}"
            for rnd in range(cfg.rounds_per_level):
                cores, scores = self.select_tiles(self.error_map(target_a, result.layer, weight_a), core)
                if cores.shape[0] == 0:
                    break
                boxes = self.view_boxes(cores)
                g = self.global_tokens(target, result.layer, cfg.tile_progress).expand(cores.shape[0], -1, -1)
                out = self.predict(target, result.layer, boxes, cfg.tile_progress, global_tokens=g)
                fade = self.fade(out.strokes, out.anchors, cores, boxes) if cfg.fade else None
                img = views.view_to_image(out.strokes, boxes)
                for t in range(cores.shape[0]):  # highest-error tile first
                    keep = slice(None) if fade is None else fade[t] > 0.1
                    tile = views.Strokes(
                        points=img.points[t : t + 1, keep],
                        radii=img.radii[t : t + 1, keep],
                        opacity=(img.opacity[t : t + 1, keep] * (1.0 if fade is None else fade[t, keep, None]))
                        .clamp(0.05, 1.0),
                        hardness=img.hardness[t : t + 1, keep],
                        color=img.color[t : t + 1, keep],
                    )
                    result.layer = self.render(result.layer, tile)
                    result.strokes.append(pack_strokes(tile).cpu())
                    n_strokes += tile.radii.shape[1]
                level_tiles += cores.shape[0]
                progress({"stage": label, "kind": "tiles", "round": rnd + 1, "rounds": cfg.rounds_per_level,
                          "tiles": int(cores.shape[0])}, cfg.whole_steps + level * cfg.rounds_per_level + rnd + 1)
            record(label, level_tiles)

        result.canvas = self.composite(result.layer)
        return result
