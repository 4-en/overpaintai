"""
Frame predictor: the plan model's ideas (a plan of tokens, stroke latents refined over several layers that attend
to each other and to the plan) at U-Net speed, with high detail only where each stroke needs it.

1. Coarse features everywhere: a lossless stem (pixel-unshuffle, R -> R/2) and a U-Net that stops at R/4 (32 x 32
   at 128 px). No full-resolution stages: they cost ~45% of the U-Net's time and most of its training memory,
   but were only ever read at the anchors. Placement is a soft-argmax on the R/4 map; refinement does the rest.
2. Fine detail in each stroke's own frame: position precision only matters at a stroke's edge (a thin stroke's
   edge is half a pixel from its centre line, a wide stroke's lies r away), so every refinement layer samples the
   raw images along the stroke (9 positions) and across it: the interior (colour), an edge band at r +- 1 px
   (where paint stops) and just outside (what to follow), plus the end caps and a few learned points. The raw
   images are read at full resolution when the caller passes them (`images`), so precision is bounded by the
   image, not by the view.
3. With `target_unet`, the target has its own U-Net, run on the view box at `target_res` x the view resolution
   (256 px for a 128 px view, from the full-resolution image) and cached: it doesn't depend on the canvas, so it is
   computed once per view box (once for all whole-image steps of an unroll or painting, once per tile set) instead
   of every step. Its stages at 64 / 32 / 16 px are added to the canvas stream (canvas + error only) at the same
   resolutions, and its full-resolution output is what the stroke samples read besides the raw values: learned line
   and edge features finer than a view pixel, with the context of the whole view.
4. Each point also sees the predicted result: every stroke's coverage there (as the renderer computes it, without
   rendering) and the canvas after the joint step, so a stroke sees where it would actually land, whether it
   misses a line, cuts into an edge or overlaps another stroke.
5. With `stroke_proposals`, placement picks whole strokes: a small conv head on the R/4 map proposes per cell a
   stroke centred there (half-vector p0 -> chord middle, bend, radius) and a learned score that is added to the
   placement logits. Each query starts from the proposal at its anchor instead of a dot, so the first refinement
   layer already samples along a stroke of sensible length and direction (a cell's features see far beyond the
   cell: a cell on an edge can propose a long stroke along it). Zero-initialized: a model starts out as without.
   With `look_ahead`, the end caps also look 4 / 8 / 16 / 32 px past each end, and learned points may sit up to
   half a stroke beyond the ends, so a stroke can see whether to grow instead of hedging with tapered ends.

A stroke's latent attends over its own point tokens, then to the other strokes, then to the plan. Sampling
positions come from the stroke decoded without gradient (as in `PlanStrokePredictor`: through the positions the
loop stroke -> features -> stroke blew up); the decoder learns through the final stroke.

Inputs are the usual view tensor (`views.view_inputs`) without its paint-alpha channel: the frame model paints on
an opaque canvas.
"""

import math
import weakref
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from overpaintai.canvas_engine import joint_at, profile_at
from overpaintai.model import (
    COLOR_EPS, GATE_LEAK, FourierPosition, Stage, StrokeOutput, StrokePredictor, StrokeSamples, TransformerBlock,
    _groups, _gumbel,
)
from overpaintai.plan_model import _blur, _curve, _odd, _sample
from overpaintai.views import Strokes, resample_views, view_grid

ERASER = 4  # tool index of the eraser, the mask with `mask_tool`
# Fixed points across a stroke: (sign-free multiple of the radius, extra view pixels), mirrored to both sides
ACROSS = ((0.0, 0.0), (0.5, 0.0), (1.0, -1.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (1.5, 0.0))
CAPS = ((1.0, -1.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0))  # beyond each end, along the tangent
LOOK_AHEAD = ((1.0, 4.0), (1.0, 8.0), (1.0, 16.0), (1.0, 32.0))  # further ones with `look_ahead`
PROPOSAL_REACH = 0.35  # largest proposed half-length and bend (view units)
OVERSHOOT_PX = 16.0  # learned points beyond an end: (t - 1) x (chord + this many view pixels) past it
PROFILE_STAMPS = 12  # curve samples for the predicted coverage


def _frame_offsets() -> torch.Tensor:
    """The fixed across-stroke offsets `(A, 2)` as (radius multiple, view pixels), both sides, centre once."""
    out = [(0.0, 0.0)]
    for r, px in ACROSS[1:]:
        out += [(r, px), (-r, -px)]
    return torch.tensor(out)


def _signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * x.abs().sqrt()


class _Images:
    """
    What the strokes sample, at full resolution when given: target, canvas, error and the luma gradients of target
    and error, as one stack `(Bi, 10, H, W)` (Bi = B, or 1 shared by all views), plus the box mapping view
    coordinates to it.
    """

    def __init__(self, inputs: torch.Tensor, boxes: torch.Tensor, images: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        if images is None:  # the view itself
            target, canvas = inputs[:, 0:3].float(), inputs[:, 3:6].float()
            self.boxes = boxes.new_tensor([0.0, 0.0, 1.0]).expand(boxes.shape[0], 3).float()
        else:
            target, canvas = images[0].float(), images[1].float()
            self.boxes = boxes.float()
        H, W = target.shape[-2:]
        err = target - canvas
        luma = lambda x: (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        lt, le = luma(target), luma(err)
        # Central differences per source pixel, scaled to per view pixel below
        def grad(x):
            p = F.pad(x, (1, 1, 1, 1), mode="replicate")
            return 0.5 * (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]), 0.5 * (p[..., 2:, 1:-1] - p[..., :-2, 1:-1])
        self.stack = torch.cat([target, canvas, err, *grad(lt), *grad(le)], dim=1)  # (Bi, 13, H, W)
        self.B = boxes.shape[0]
        self.R = inputs.shape[-1]
        self.size = max(H, W)

    def sample(self, pts: torch.Tensor) -> torch.Tensor:
        """Values at view points `(B, N, Q, 2)` -> `(B, N, Q, 13)` (gradients per view pixel)."""
        B, N, Q, _ = pts.shape
        img = self.boxes[:, None, None, :2] + self.boxes[:, None, None, 2:3] * pts.float()  # image coordinates
        if self.stack.shape[0] == 1 and B > 1:  # one image for all views: one grid
            out = _sample(self.stack, img.reshape(1, 1, B * N * Q, 2)).view(B, N, Q, -1)
        else:
            out = _sample(self.stack, img.reshape(B, 1, N * Q, 2)).view(B, N, Q, -1)
        # view pixels per source pixel: the gradient per view pixel
        scale = (self.size - 1) * self.boxes[:, 2] / max(self.R - 1, 1)
        g = out[..., 9:] * scale.view(B, 1, 1, 1)
        return torch.cat([out[..., :9], g], dim=-1)


class FrameStrokePredictor(nn.Module):
    """
    Args (shared ones as in `StrokePredictor` / `PlanStrokePredictor`):
        global_blocks: self-attention blocks over the plan tokens (view grid + whole-image grid + plan slots).
        query_blocks: pairs of (self-attention among the stroke latents, cross-attention to the plan).
        refine_layers: refinement layers (stroke-frame samples -> latent; strokes; plan).
        frame_along: positions along each stroke where it samples across.
        learned_points: extra sample points per stroke and layer, offsets in the stroke's frame from its latent.
        point_dim: width of the coarse features read at each sample point.
        target_unet: a separate, cached U-Net for the target (see the module docstring); the per-step stream then
            only reads canvas and error.
        target_res: its input resolution as a multiple of the view's (1, 2, 4).
        target_base: its width at full resolution (doubling per level, at most the coarse path's widest).
        stroke_proposals: a dense stroke proposal per R/4 cell and a learned placement score (module docstring).
        look_ahead: end-cap samples far past the ends; learned points beyond the ends.
    """

    IN_CHANNELS = StrokePredictor.IN_CHANNELS
    uses_images = True  # callers pass the full-resolution target and canvas (`forward(..., images=...)`)

    def __init__(
        self,
        num_strokes: int = 16,
        widths: tuple = (32, 64, 128, 256),
        token_dim: int = 256,
        heads: int = 8,
        global_blocks: int = 3,
        query_blocks: int = 2,
        refine_layers: int = 3,
        view_tokens: int = 16,
        global_grid: int = 8,
        plan_slots: int = 8,
        frame_along: int = 9,
        learned_points: int = 8,
        point_dim: int = 32,
        max_offset: float = 0.35,
        radius_range: tuple = (0.004, 0.75),
        num_tools: int = 5,
        paintable_mask: bool = False,
        stroke_gate: bool = False,
        error_progress: bool = False,
        stationary: bool = False,
        hard_gate: bool = False,
        pressure: bool = True,
        target_unet: bool = False,
        target_res: int = 2,
        target_base: int = 16,
        stroke_proposals: bool = False,
        look_ahead: bool = False,
    ):
        super().__init__()
        self.num_strokes = num_strokes
        self.paintable_mask = paintable_mask
        self.stroke_gate = stroke_gate
        self.error_progress = error_progress
        self.stationary = stationary
        self.hard_gate = hard_gate
        self.pressure = pressure
        self.in_channels = self.IN_CHANNELS + int(paintable_mask)
        # View channels the per-step network reads: everything but the paint alpha (channel 9), and with its own
        # U-Net the target (channels 0-2) neither
        skip = {9} | ({0, 1, 2} if target_unet else set())
        self.register_buffer("channels", torch.tensor([c for c in range(self.in_channels) if c not in skip]),
                             persistent=False)
        self.target_unet = target_unet
        self.target_res = target_res
        self.view_tokens = view_tokens
        self.global_grid = global_grid
        self.max_offset = max_offset
        self.radius_range = radius_range
        self.learned_points = learned_points
        self.stroke_proposals = stroke_proposals
        self.look_ahead = look_ahead
        self.placement_noise = 0.0  # runtime switches, as in StrokePredictor
        self.sample_in_eval = False
        self.mask_tool = False  # set by build_model: the eraser is a per-step mask (for the predicted result)
        _, w1, w2, w3 = widths
        D = token_dim
        n_in = len(self.channels)

        n_cond = 2 + int(error_progress or stationary)
        self.cond_mlp = nn.Sequential(nn.Linear(n_cond, D), nn.SiLU(), nn.Linear(D, D))

        # Coarse path: lossless stem (every pixel kept, 2 x 2 -> channels), then two downsamples
        self.stem = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(4 * n_in, w1, 3, padding=1))
        self.enc1 = Stage(w1, w1, D)
        self.down2 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)
        self.enc2 = Stage(w2, w2, D)
        self.down3 = nn.Conv2d(w2, w3, 3, stride=2, padding=1)
        self.enc3 = Stage(w3, w3, D)

        # Target U-Net: levels from target_res x R down to R/8, each level's output with the whole view's context
        if target_unet:
            n = int(round(math.log2(target_res))) + 4
            if 2 ** (n - 4) != target_res:
                raise ValueError(f"target_res must be a power of two, got {target_res}")
            tw = [min(target_base * 2 ** i, w3) for i in range(n)]
            self.t_cond = nn.Sequential(nn.Linear(1, D), nn.SiLU(), nn.Linear(D, D))  # view size only
            self.t_stem = nn.Conv2d(3, tw[0], 3, padding=1)
            self.t_enc = nn.ModuleList([Stage(tw[i], tw[i], D) for i in range(n)])
            self.t_down = nn.ModuleList([nn.Conv2d(tw[i], tw[i + 1], 3, stride=2, padding=1) for i in range(n - 1)])
            self.t_up = nn.ModuleList([nn.Conv2d(tw[i + 1], tw[i], 3, padding=1) for i in range(n - 1)])
            self.t_dec = nn.ModuleList([Stage(2 * tw[i], tw[i], D) for i in range(n - 1)])
            # into the canvas stream at R/2, R/4, R/8; and the full-resolution map the stroke samples read
            self.t_fuse = nn.ModuleList([nn.Conv2d(tw[n - 3], w1, 1), nn.Conv2d(tw[n - 2], w2, 1),
                                         nn.Conv2d(tw[n - 1], w3, 1)])
            self.t_point = nn.Conv2d(tw[0], point_dim, 1)

        # Plan: view tokens, whole-image tokens and plan slots
        self.to_tokens = nn.Linear(w3, D)
        self.token_pos = FourierPosition(D)
        self.token_type = nn.Parameter(torch.zeros(3, D))  # 0: whole image, 1: view, 2: plan slot
        self.slots = nn.Parameter(torch.randn(plan_slots, D) * 0.02)
        self.global_blocks = nn.ModuleList([TransformerBlock(D, heads) for _ in range(global_blocks)])
        self.from_tokens = nn.Conv2d(D, w3, 1)
        nn.init.zeros_(self.from_tokens.weight)
        nn.init.zeros_(self.from_tokens.bias)

        # Decoder to R/4 only
        self.mid = Stage(w3, w3, D)
        self.up2 = nn.Conv2d(w3, w2, 3, padding=1)
        self.dec2 = Stage(2 * w2, w2, D)
        self.pixel_dim = P = w2
        self.pixel_norm = nn.GroupNorm(_groups(P), P)
        self.point_proj = nn.Conv2d(P, point_dim, 1)  # the coarse features each sample point reads

        # Stroke latents
        self.queries = nn.Parameter(torch.randn(num_strokes, D) * 0.02)
        self.query_cond = nn.Linear(D, D)
        self.query_blocks = nn.ModuleList()
        for _ in range(query_blocks):
            self.query_blocks.append(TransformerBlock(D, heads))
            self.query_blocks.append(TransformerBlock(D, heads, cross=True))

        # Initial placement on the R/4 map
        self.place_key = nn.Linear(D, P)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        side = math.ceil(math.sqrt(num_strokes))
        cells = (torch.arange(side, dtype=torch.float32) + 0.5) / side
        ref = torch.stack(torch.meshgrid(cells, cells, indexing="xy"), dim=-1).reshape(-1, 2)[:num_strokes]
        self.ref_logits = nn.Parameter(torch.log(ref) - torch.log1p(-ref))
        self.ref_log_sigma = nn.Parameter(torch.tensor(math.log(0.25)))
        self.error_gain = nn.Parameter(torch.tensor(1.0))
        init_dim = P + w3 + 3
        self.init_proj = nn.Sequential(nn.LayerNorm(init_dim), nn.Linear(init_dim, D))
        if stroke_proposals:
            # per cell from its features and log error: score, half-vector 2, bend, radius logit
            self.propose = nn.Sequential(nn.Conv2d(P + 1, P, 3, padding=1), nn.SiLU(),
                                         nn.Conv2d(P, P, 3, padding=1), nn.SiLU(), nn.Conv2d(P, 5, 1))
            self.proposal_proj = nn.Linear(4, D)  # the chosen proposal into the latent
            for layer in (self.propose[-1], self.proposal_proj):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Stroke-frame points: fixed grid (along x across, end caps) + learned offsets per layer
        self.register_buffer("t_along", torch.linspace(0.0, 1.0, frame_along), persistent=False)
        self.register_buffer("across", _frame_offsets(), persistent=False)
        self.register_buffer("caps", torch.tensor(CAPS + (LOOK_AHEAD if look_ahead else ())), persistent=False)
        # per point: target + canvas 6, error x4 and signed sqrt 6, gradients 4, own coverage and shape 2, predicted
        # error x4 and signed sqrt 6; coarse features; frame coordinates (t, radius multiple, pixels, cap, learned)
        # and sin / cos of t at 3 frequencies 11
        self.point_in = 24 + point_dim * (2 if target_unet else 1) + 11
        self.point_enc = nn.Sequential(nn.LayerNorm(self.point_in), nn.Linear(self.point_in, 4 * point_dim),
                                       nn.SiLU(), nn.Linear(4 * point_dim, D))
        self.refine = nn.ModuleList()
        for _ in range(refine_layers):
            self.refine.append(nn.ModuleDict(dict(
                learned=nn.Linear(D, 3 * learned_points),
                look=TransformerBlock(D, heads, cross=True),
                self_attn=TransformerBlock(D, heads), cross_attn=TransformerBlock(D, heads, cross=True),
            )))
            nn.init.zeros_(self.refine[-1]["learned"].weight)
            nn.init.zeros_(self.refine[-1]["learned"].bias)

        # Shared stroke decoder: anchor shift 2, point offsets 6, radii, opacity, hardness 1, color 3, tools, (gate)
        self.n_ends = 3 if pressure else 2
        n_out = 2 + 6 + 2 * self.n_ends + 1 + 3 + num_tools + int(stroke_gate)
        self.decoder = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, 2 * D), nn.SiLU(), nn.Linear(2 * D, D), nn.SiLU(), nn.Linear(D, n_out),
        )
        nn.init.normal_(self.decoder[-1].weight, std=1e-3)  # starts as a dot at the initial anchor
        nn.init.zeros_(self.decoder[-1].bias)

    # ------------------------------------------------------------------------------------------------
    _cond = StrokePredictor._cond
    uses_error_progress = StrokePredictor.uses_error_progress

    def encode_target(self, target: torch.Tensor, boxes: torch.Tensor, R: int,
                      view_size: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        The cacheable target features of views `boxes` (B, 3) of `target` (B or 1, 3, H, W), for a model working
        at view resolution `R`: (map the stroke samples read, at target_res x R; additions to the canvas stream at
        R/2, R/4, R/8). `view_size` (B,) as in `forward` (default: the boxes' sizes).
        """
        x = resample_views(target, boxes, self.target_res * R)
        vs = boxes[:, 2] if view_size is None else view_size
        cond = self.t_cond(torch.log2(vs.clamp(min=1e-4)).unsqueeze(-1).to(x.dtype))
        skips, h = [], self.t_stem(x)
        for i, stage in enumerate(self.t_enc):
            h = stage(h, cond)
            if i < len(self.t_down):
                skips.append(h)
                h = self.t_down[i](h)
        outs = [h]  # bottom up: R/8 first
        for i in reversed(range(len(self.t_dec))):
            h = self.t_dec[i](torch.cat([self.t_up[i](F.interpolate(h, scale_factor=2.0, mode="nearest")),
                                         skips[i]], 1), cond)
            outs.append(h)
        outs = outs[::-1]  # full resolution first
        n = len(outs)
        return (self.t_point(outs[0]), self.t_fuse[0](outs[n - 3]), self.t_fuse[1](outs[n - 2]),
                self.t_fuse[2](outs[n - 1]))

    def _encode(self, x: torch.Tensor, cond: torch.Tensor, target: Optional[Tuple[torch.Tensor, ...]] = None):
        x = x[:, self.channels]
        add = (0.0, 0.0, 0.0) if target is None else target[1:]
        h1 = self.enc1(self.stem(x) + add[0], cond)
        h2 = self.enc2(self.down2(h1) + add[1], cond)
        h3 = self.enc3(self.down3(h2) + add[2], cond)
        return h1, h2, h3

    def _target(self, inputs: torch.Tensor, boxes: torch.Tensor, images, target_features, view_size=None):
        """The target features for this call: given (cached by the caller) or computed (from the full-resolution
        target when given, else the view's own, upsampled)."""
        if not self.target_unet:
            return None
        if target_features is not None:
            return target_features
        if images is not None:
            return self.encode_target(images[0], boxes, inputs.shape[-1], view_size)
        unit = boxes.new_tensor([0.0, 0.0, 1.0]).expand(boxes.shape[0], 3)
        vs = boxes[:, 2] if view_size is None else view_size
        return self.encode_target(inputs[:, 0:3], unit, inputs.shape[-1], vs)

    def _grid_positions(self, boxes: torch.Tensor, G: int) -> torch.Tensor:
        centers = (torch.arange(G, device=boxes.device, dtype=boxes.dtype) + 0.5) / G
        x = boxes[:, 0, None, None] + boxes[:, 2, None, None] * centers.view(1, 1, G)
        y = boxes[:, 1, None, None] + boxes[:, 2, None, None] * centers.view(1, G, 1)
        return torch.stack([x.expand(-1, G, G), y.expand(-1, G, G)], dim=-1).reshape(-1, G * G, 2)

    def _tokens(self, h3: torch.Tensor, boxes: torch.Tensor, G: int, kind: int) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(h3, G).flatten(2).transpose(1, 2)
        return self.to_tokens(pooled) + self.token_pos(self._grid_positions(boxes, G)) + self.token_type[kind]

    def encode_global(
        self, whole_inputs: torch.Tensor, progress: torch.Tensor, error_progress: Optional[torch.Tensor] = None,
        images: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        target_features: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        """Whole-image tokens `(B, G*G, D)`, shared by all tile views of that image and step (`images`,
        `target_features`: as in `forward`, for the whole image)."""
        boxes = whole_inputs.new_tensor([0.0, 0.0, 1.0]).expand(whole_inputs.shape[0], 3)
        cond = self._cond(progress, boxes[:, 2], error_progress)
        target = self._target(whole_inputs, boxes, images, target_features)
        return self._tokens(self._encode(whole_inputs, cond, target)[2], boxes, self.global_grid, kind=0)

    # --- strokes ------------------------------------------------------------------------------------
    def proposal_strokes(self, raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Raw proposals `(..., 4)` -> (half-vector p0 - chord middle `(..., 2)`, bend: p1 off the chord middle
        along the chord's normal `(...)`, radius logit offset `(...)`), in view units."""
        return PROPOSAL_REACH * torch.tanh(raw[..., 0:2]), PROPOSAL_REACH * torch.tanh(raw[..., 2]), raw[..., 3]

    def proposal_radius(self, r_raw: torch.Tensor) -> torch.Tensor:
        """The radius a radius logit offset gives with the decoder's own correction at zero."""
        r_lo, r_hi = (math.log(r) for r in self.radius_range)
        r_init = (math.log(0.1) - r_lo) / (r_hi - r_lo)
        return torch.exp(r_lo + (r_hi - r_lo) * torch.sigmoid(r_raw + math.log(r_init / (1.0 - r_init))))

    def _decode(self, h: torch.Tensor, anchor0: torch.Tensor, imgs: _Images, proposal: Optional[torch.Tensor] = None):
        """
        The stroke a latent describes: (anchor, points, radii, opacity, hardness, color, tool logits, raw gate logit).
        The anchor is the initial anchor plus a shift and the middle of the chord p0 -> p2 (see `PlanStrokePredictor`).
        With a `proposal` `(B, N, 4)` (raw, see `proposal_strokes`), the decoder's outputs correct that stroke.
        The colour is a correction to the target averaged over the stroke's interior (centre line and half a
        radius to each side, full resolution), looked up at the detached stroke.
        """
        out = self.decoder(h).float()
        e = self.n_ends
        shift, offsets, r_raw, a_raw, h_raw, c_raw, rest = out.split(
            [2, 6, e, e, 1, 3, out.shape[-1] - 12 - 2 * e], dim=-1)
        anchor = anchor0 + 0.15 * torch.tanh(shift)
        off = self.max_offset * torch.tanh(offsets.unflatten(-1, (3, 2)))  # (B, N, 3, 2)
        half = 0.5 * (off[..., 0, :] - off[..., 2, :])
        bend = 0.5 * off[..., 1, :]
        if proposal is not None:
            p_half, p_bend, p_r = self.proposal_strokes(proposal.float())
            normal = torch.stack([-p_half[..., 1], p_half[..., 0]], -1) / (p_half.norm(dim=-1, keepdim=True) + 1e-3)
            half, bend, r_raw = half + p_half, bend + p_bend.unsqueeze(-1) * normal, r_raw + p_r.unsqueeze(-1)
        points = torch.stack([anchor + half, anchor + bend, anchor - half], 2)
        radii = self.proposal_radius(r_raw)  # starts at radius ~0.1 view units
        opacity = 0.05 + 0.95 * torch.sigmoid(a_raw)
        with torch.no_grad():
            interior = self._frame_points(points.detach(), radii.detach(), imgs.R, interior=True)[0]
            tc = imgs.sample(interior)[..., 0:3].mean(2).clamp(COLOR_EPS, 1.0 - COLOR_EPS)
        color = torch.sigmoid(torch.log(tc) - torch.log1p(-tc) + c_raw)
        tool_logits, gate_raw = (rest[..., :-1], rest[..., -1]) if self.stroke_gate else (rest, None)
        return anchor, points, radii, opacity, torch.sigmoid(h_raw), color, tool_logits, gate_raw

    def _frame(self, points: torch.Tensor, radii: torch.Tensor, t: torch.Tensor):
        """Curve position and unit tangent / normal `(B, N, K, 2)` and radius `(B, N, K)` at `t` `(K,)` or
        `(B, N, K)`."""
        p0, p1, p2 = (p.unsqueeze(2) for p in points.unbind(2))  # (B, N, 1, 2)
        tt = (t.view(1, 1, -1) if t.dim() == 1 else t).unsqueeze(-1)
        pos = (1 - tt) ** 2 * p0 + 2 * tt * (1 - tt) * p1 + tt ** 2 * p2
        d = 2 * (1 - tt) * (p1 - p0) + 2 * tt * (p2 - p1)
        norm = d.norm(dim=-1, keepdim=True)  # a dot (no direction) gets the x axis as its tangent
        tangent = torch.where(norm > 1e-6, d / norm.clamp(min=1e-6), torch.tensor([1.0, 0.0], device=d.device))
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
        r0, r2 = radii[..., 0:1], radii[..., 1:2]
        r1 = radii[..., 2:3] if radii.shape[-1] == 3 else 0.5 * (r0 + r2)
        t2 = tt[..., 0]
        return pos, normal, tangent, (1 - t2) ** 2 * r0 + 2 * t2 * (1 - t2) * r1 + t2 ** 2 * r2

    def _frame_points(self, points: torch.Tensor, radii: torch.Tensor, R: int, interior: bool = False):
        """
        The fixed sample points of each stroke `(B, N, Q, 2)` (view coordinates) and their frame coordinates
        `(Q, 4)`: t, radius multiple, view pixels, end cap. `interior`: only the centre line and +-r/2 (colour).
        """
        px = 1.0 / max(R - 1, 1)
        pos, normal, tangent, r = self._frame(points, radii, self.t_along)  # (B, N, T, 2) / (B, N, T)
        across = self.across[:3] if interior else self.across  # centre, +r/2, -r/2 first
        dist = r.unsqueeze(-1) * across[:, 0] + px * across[:, 1]  # (B, N, T, A)
        pts = pos.unsqueeze(3) + normal.unsqueeze(3) * dist.unsqueeze(-1)  # (B, N, T, A, 2)
        B, N, T, A, _ = pts.shape
        pts = pts.reshape(B, N, T * A, 2)
        tq = self.t_along.view(T, 1).expand(T, A)
        coords = torch.stack([tq, across[:, 0].expand(T, A), across[:, 1].expand(T, A), torch.zeros(T, A,
                              device=pts.device)], dim=-1).reshape(T * A, 4)
        if interior:
            return pts, coords
        # End caps: beyond the start (against the tangent) and the end
        cap_d = r[..., [0, -1]].unsqueeze(-1) * self.caps[:, 0] + px * self.caps[:, 1]  # (B, N, 2, C)
        sign = torch.tensor([-1.0, 1.0], device=pts.device).view(1, 1, 2, 1, 1)
        ends = pos[:, :, [0, -1]].unsqueeze(3) + sign * tangent[:, :, [0, -1]].unsqueeze(3) * cap_d.unsqueeze(-1)
        C = self.caps.shape[0]
        cap_coords = torch.stack([torch.tensor([0.0, 1.0], device=pts.device).view(2, 1).expand(2, C),
                                  self.caps[:, 0].expand(2, C), self.caps[:, 1].expand(2, C),
                                  torch.ones(2, C, device=pts.device)], dim=-1).reshape(2 * C, 4)
        return torch.cat([pts, ends.reshape(B, N, 2 * C, 2)], 2), torch.cat([coords, cap_coords], 0)

    def _learned_points(self, layer: nn.ModuleDict, h: torch.Tensor, points: torch.Tensor, radii: torch.Tensor, R: int):
        """Learned points `(B, N, L, 2)` (positions differentiable through the offsets, not the stroke) and their
        frame coordinates `(B, N, L, 3)`: t, radius multiple, view pixels."""
        L = self.learned_points
        raw = layer["learned"](h).float().view(*h.shape[:2], L, 3)
        # Spread along the stroke at the start (zero-initialized offsets -> t evenly spaced); with look-ahead t
        # spans (-0.5, 1.5), starting at the same places
        base = torch.linspace(0.1, 0.9, L, device=h.device)
        lo, span = (-0.5, 2.0) if self.look_ahead else (0.0, 1.0)
        t = lo + span * torch.sigmoid(raw[..., 0] + torch.logit((base - lo) / span).view(1, 1, L))
        k_r, k_px = 2.0 * torch.tanh(raw[..., 1]), 3.0 * torch.tanh(raw[..., 2])
        px = 1.0 / max(R - 1, 1)
        pos, normal, tangent, r = self._frame(points, radii, t.clamp(0.0, 1.0))
        if self.look_ahead:  # beyond an end: straight on along its tangent (the curve itself would bend away)
            over = (t - 1.0).clamp(min=0.0) - (-t).clamp(min=0.0)  # signed: + past p2, - before p0
            chord = (points[:, :, 2] - points[:, :, 0]).norm(dim=-1, keepdim=True)
            pos = pos + tangent * (over * (chord + OVERSHOOT_PX * px)).unsqueeze(-1)
        pts = pos + normal * (r * k_r + px * k_px).unsqueeze(-1)
        return pts, torch.stack([t, k_r, k_px], dim=-1)

    def _point_tokens(self, pts: torch.Tensor, coords: torch.Tensor, imgs: _Images, coarse: torch.Tensor,
                      strokes: Tuple, edge: float, fine: Optional[torch.Tensor] = None):
        """
        Point tokens `(B * N, Q, D)` for each stroke's own sample points `pts` `(B, N, Q, 2)`: image values, the
        predicted result there (from all strokes `strokes`), coarse features and frame coordinates `coords`
        `(B, N, Q, 5)`.
        """
        B, N, Q, _ = pts.shape
        with torch.no_grad(), torch.autocast(pts.device.type, enabled=False):
            vals = imgs.sample(pts.detach())  # (B, N, Q, 13)
            ctrl, radii, opacity, hardness, color, paint_w, mask_w = strokes
            flat = pts.detach().float().reshape(B, N * Q, 2)
            mask, shape = profile_at(flat, ctrl, radii, opacity, hardness, edge, PROFILE_STAMPS)  # (B, N, N*Q)
            canvas = vals[..., 3:6].reshape(B, N * Q, 3)
            pred = joint_at(canvas, mask, shape, color, paint_w, mask_w).view(B, N, Q, 3)
            own = torch.arange(N, device=pts.device)
            own_mask = mask.view(B, N, N, Q)[:, own, own]  # (B, N, Q): each stroke at its own points
            own_shape = shape.view(B, N, N, Q)[:, own, own]
            pred_err = vals[..., 0:3] - pred
            g = _signed_sqrt(vals[..., 9:13])
            feats = torch.cat([vals[..., 0:6], 4.0 * vals[..., 6:9], _signed_sqrt(vals[..., 6:9]), g,
                               own_mask.unsqueeze(-1), own_shape.unsqueeze(-1), 4.0 * pred_err,
                               _signed_sqrt(pred_err)], dim=-1)
        # Coarse features and learned positions keep their gradient (the learned offsets learn through them)
        cf = _sample(coarse, pts)  # (B, N, Q, point_dim)
        if fine is not None:  # the target U-Net's full-resolution features
            cf = torch.cat([cf, _sample(fine, pts).to(cf.dtype)], -1)
        t = coords[..., 0:1]
        ang = t * math.pi * torch.tensor([1.0, 2.0, 4.0], device=t.device)
        frame = torch.cat([coords, ang.sin(), ang.cos()], dim=-1)
        x = torch.cat([feats.to(cf.dtype), cf, frame.to(cf.dtype)], dim=-1)
        return self.point_enc(x).reshape(B * N, Q, -1)

    def _stroke_state(self, decoded, R: int) -> Tuple:
        """Provisional strokes for the predicted result, as the renderer would get them: (ctrl, radii (3), opacity x
        gate (3), hardness, color, paint tool weight, mask tool weight or None)."""
        _, points, radii, opacity, hardness, color, tool_logits, gate_raw = decoded
        if radii.shape[-1] == 2:
            radii = torch.cat([radii, radii.mean(-1, keepdim=True)], -1)
            opacity = torch.cat([opacity, opacity.mean(-1, keepdim=True)], -1)
        gate = torch.sigmoid(gate_raw + 3.0) if gate_raw is not None else torch.ones_like(opacity[..., 0])
        p_mask = tool_logits.float().softmax(-1)[..., ERASER] if self.mask_tool else torch.zeros_like(gate)
        return (points.float(), radii.float(), (opacity * gate.unsqueeze(-1)).float(), hardness[..., 0].float(),
                color.float(), 1.0 - p_mask, p_mask if self.mask_tool else None)

    # ------------------------------------------------------------------------------------------------
    def forward(
        self,
        inputs: torch.Tensor,
        boxes: torch.Tensor,
        progress: torch.Tensor,
        global_tokens: Optional[torch.Tensor] = None,
        view_size: Optional[torch.Tensor] = None,
        error_progress: Optional[torch.Tensor] = None,
        samples: Optional[StrokeSamples] = None,
        images: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        target_features: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> StrokeOutput:
        """
        Same contract as `StrokePredictor.forward`, plus `images`: (target, canvas) `(Bi, 3, H, W)` of the whole
        image (Bi = B, or 1 shared by all views) that `boxes` index into, read at full resolution along the strokes.
        Without them the strokes read the view itself. `target_features`: `encode_target` of these views, cached
        by the caller (computed here if omitted).
        """
        sampling = self.training or self.sample_in_eval
        B, _, R, _ = inputs.shape
        cond = self._cond(progress, boxes[:, 2] if view_size is None else view_size, error_progress)
        target = self._target(inputs, boxes, images, target_features, view_size)
        h1, h2, h3 = self._encode(inputs, cond, target)
        imgs = _Images(inputs, boxes, images)

        # Plan
        Gv = self.view_tokens
        view_tok = self._tokens(h3, boxes, Gv, kind=1)
        if global_tokens is None:  # whole-image step: the view itself at the coarser scale
            global_tokens = self._tokens(h3, boxes, self.global_grid, kind=0)
        slots = (self.slots + self.token_type[2]).unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([slots, global_tokens, view_tok], dim=1)
        for block in self.global_blocks:
            tokens = block(tokens)
        grid_tok = tokens[:, -Gv * Gv:].transpose(1, 2).reshape(B, -1, Gv, Gv)
        h3 = h3 + F.interpolate(self.from_tokens(grid_tok), size=h3.shape[-2:], mode="bilinear", align_corners=False)
        m = self.mid(h3, cond)
        d2 = self.dec2(torch.cat([self.up2(F.interpolate(m, size=h2.shape[-2:], mode="nearest")), h2], 1), cond)
        pixel = self.pixel_norm(d2)  # (B, P, R/4, R/4)
        coarse = self.point_proj(pixel)

        # Stroke latents
        h = self.queries.unsqueeze(0).expand(B, -1, -1) + self.query_cond(cond).unsqueeze(1)
        for i in range(0, len(self.query_blocks), 2):
            h = self.query_blocks[i](h)
            h = self.query_blocks[i + 1](h, context=tokens)

        # Initial placement: soft-argmax on the R/4 map + priors (home position, where the canvas is still wrong)
        Rc = pixel.shape[-1]
        key = self.place_key(h)
        logits = torch.einsum("bnp,bphw->bnhw", key, pixel).flatten(2) / math.sqrt(self.pixel_dim)
        logits = logits.float() * self.log_temperature.exp()
        uv = view_grid(boxes.new_tensor([0.0, 0.0, 1.0]).expand(B, 3), Rc).reshape(B, Rc * Rc, 2)
        ref = torch.sigmoid(self.ref_logits)
        d_sq = ((uv[:1].float().unsqueeze(1) - ref.view(1, -1, 1, 2)) ** 2).sum(-1)
        logits = logits - d_sq / (2.0 * self.ref_log_sigma.exp() ** 2)
        err = _blur(inputs[:, 6:9].float().abs().mean(1, keepdim=True), _odd(9 * R / 128))
        err = F.adaptive_avg_pool2d(err, Rc)
        log_err = torch.log(err + 1e-3)
        logits = logits + self.error_gain * log_err.flatten(2)
        proposal_map = None
        if self.stroke_proposals:  # a learned score on top of the priors, and a stroke per cell
            proposal_map = self.propose(torch.cat([pixel, log_err.to(pixel.dtype)], 1)).float()  # (B, 5, Rc, Rc)
            logits = logits + proposal_map[:, 0].flatten(1).unsqueeze(1)
        prob = logits.softmax(dim=-1)
        place_seed, place_prob = None, prob
        if self.placement_noise > 0 and (sampling or (samples is not None and samples.place_seed is not None)):
            place_seed = samples.place_seed if samples is not None and samples.place_seed is not None else \
                torch.randint(0, 2**62, (B,), device=logits.device)
            place_prob = (logits + self.placement_noise * _gumbel(place_seed, logits.shape[1:])).softmax(-1)
        anchor0 = torch.einsum("bnk,bkc->bnc", place_prob, uv.float())
        entropy = -(prob * prob.clamp(min=1e-12).log()).sum(-1)

        at0 = anchor0.detach().unsqueeze(2)
        with torch.no_grad(), torch.autocast(inputs.device.type, enabled=False):
            color0 = imgs.sample(at0)[..., 0:3]
        h = h + self.init_proj(torch.cat([_sample(pixel, at0), _sample(m, at0), color0.to(pixel.dtype)], -1)[:, :, 0])
        proposal = None
        if proposal_map is not None:  # the proposal where the query landed
            proposal = _sample(proposal_map[:, 1:], at0)[:, :, 0]  # (B, N, 4)
            h = h + self.proposal_proj(proposal.to(h.dtype))

        # Refinement in the strokes' frames
        edge = 2.0 * math.log(9.0) * (R - 1)  # the renderer's 1-px hard edge, in view units
        for layer in self.refine:
            with torch.no_grad():
                decoded = self._decode(h, anchor0, imgs, proposal)
            state = self._stroke_state(decoded, R)
            fixed, fixed_coords = self._frame_points(state[0], state[1], R)  # (B, N, Q0, 2), (Q0, 4)
            Q0 = fixed.shape[2]
            coords = torch.cat([fixed_coords, torch.zeros(Q0, 1, device=h.device)], -1).expand(B, self.num_strokes,
                                                                                                   Q0, 5)
            pts = fixed
            if self.learned_points > 0:
                lp, lc = self._learned_points(layer, h, state[0], state[1], R)
                L = lp.shape[2]
                lcoords = torch.cat([lc, torch.zeros_like(lc[..., :1]), torch.ones_like(lc[..., :1])], -1)
                pts = torch.cat([fixed.to(lp.dtype), lp], 2)
                coords = torch.cat([coords.to(lcoords.dtype), lcoords], 2)
            look = self._point_tokens(pts, coords, imgs, coarse, state, edge,
                                      None if target is None else target[0])  # (B*N, Q, D)
            hq = layer["look"](h.reshape(B * self.num_strokes, 1, -1), context=look.to(h.dtype))
            h = hq.view(B, self.num_strokes, -1)
            h = layer["self_attn"](h)
            h = layer["cross_attn"](h, context=tokens)

        # The output: the same decoder on the last latent
        anchor, points, radii, opacity, hardness, color, tool_logits, gate_raw = self._decode(
            h, anchor0, imgs, proposal)
        gate_logit, gate_on = None, None
        if self.stroke_gate:  # start with every stroke on (sigmoid(3) = 0.95)
            gate_logit = gate_raw + 3.0
            gate = torch.sigmoid(gate_logit)
            if self.training:
                gate = gate + GATE_LEAK * (gate_logit - gate_logit.detach())
            if self.hard_gate:
                if samples is not None and samples.gate_on is not None:
                    gate_on = samples.gate_on
                elif sampling:
                    gate_on = torch.bernoulli(gate.detach().clamp(0.0, 1.0))
                else:
                    gate_on = (gate_logit > 0).float()
                gate = gate_on + (gate - gate.detach())
        else:
            gate = torch.ones(points.shape[:2], device=points.device)

        strokes = Strokes(points=points, radii=radii, opacity=opacity, hardness=hardness, color=color)
        return StrokeOutput(
            strokes=strokes, tool_logits=tool_logits, anchors=anchor.float(), anchor_entropy=entropy,
            placement_logits=logits, gate=gate, gate_logit=gate_logit, samples=StrokeSamples(place_seed, gate_on),
            proposals=None if proposal_map is None else proposal_map[:, 1:],
        )


class TargetCache:
    """
    Cached `encode_target` features per (target tensor, view boxes): a target and its views don't change while a
    painting goes on, so whole-image steps (and repeated passes over the same tiles) reuse one encoding. Entries
    hold the features with their autograd graph when computed with gradients (one encoding shared by all steps of
    an unroll; their gradients add up), and are keyed by gradient mode, so features without a graph never stand in
    for ones that need it. A target is recognized by identity (weak reference): a new tensor is a new target.
    """

    def __init__(self, size: int = 8):
        self.size = size
        self.entries = {}

    def get(self, model: nn.Module, target: torch.Tensor, boxes: torch.Tensor, R: int,
            view_size: Optional[torch.Tensor] = None, repeat: int = 1, autocast=None):
        """Features of `boxes` (B * repeat, 3) on `target` (B or 1, ...; repeated per view when `repeat` > 1),
        or None for models without a target U-Net. `autocast`: context manager to compute them under."""
        if not getattr(model, "target_unet", False):
            return None
        vs = None if view_size is None else view_size.detach()
        key = (boxes.detach().float().cpu().numpy().tobytes(),
               None if vs is None else vs.float().cpu().numpy().tobytes(),
               torch.is_grad_enabled(), model.training, R, repeat)
        self.entries = {k: v for k, v in self.entries.items() if v[0]() is not None}  # targets still alive
        hit = self.entries.get(key)
        if hit is not None and hit[0]() is target:
            return hit[1]
        img = target.repeat_interleave(repeat, 0) if repeat > 1 else target
        # With gradients, only the outputs are kept until backward, which runs the U-Net once more: its high-res
        # activations would otherwise stay alive over the whole unroll (+6 GB for 16 views at 256 px)
        encode = model.encode_target
        if torch.is_grad_enabled():
            encode = lambda *a: torch.utils.checkpoint.checkpoint(model.encode_target, *a, use_reentrant=False)
        if autocast is None:
            feats = encode(img, boxes, R, view_size)
        else:
            with autocast:
                feats = encode(img, boxes, R, view_size)
        if len(self.entries) >= self.size:
            self.entries.pop(next(iter(self.entries)))
        self.entries[key] = (weakref.ref(target), feats)
        return feats
