"""
Plan-then-strokes predictor: the same inputs and outputs as `model.StrokePredictor`, a different way of
deciding the N strokes of a step.

1. Plan: the view's deepest features as a 16 x 16 token grid, the whole image as 8 x 8 tokens and a few learned
   plan slots attend to each other (6 blocks). The view tokens also flow back into the U-Net bottleneck.
2. Strokes: one latent per stroke (query blocks: coordinate, then read the plan), an initial position from a
   spatial soft-argmax over full-resolution pixel features (as in `StrokePredictor`), then refinement layers on
   the latents. Attention only ever sees the latents. One shared decoder turns any latent into a complete stroke:
   each layer decodes its current stroke (detached) only to sample features along it (what the stroke would
   cover), adds them to the latent, then lets the strokes attend to each other (who covers what) and to the plan.
   The output is the same decoder applied to the last latent, so every layer works on the stroke that will be
   painted.

Strokes get a pressure curve (start / end / middle radius and opacity). The anchor is the middle of the stroke's
chord (start and end lie on opposite sides of it) and the control point bends the curve by a bounded amount, so
the stroke always runs past its anchor, and the placement map, the features read at the anchor and the color
all describe the stroke that gets painted. (In `StrokePredictor` the anchor is the curve's midpoint, which made
~30% of strokes double back on themselves; a symmetric chord can't.) Colors start from the target averaged along
the stroke at the stroke's own scale: single pixels for thin strokes, up to ~33 px averages for wide ones.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from overpaintai.model import (
    COLOR_EPS, GATE_LEAK, FourierPosition, Stage, StrokeOutput, StrokePredictor, StrokeSamples, TransformerBlock, _gumbel,
    _groups,
)
from overpaintai.views import Strokes, view_grid

CURVE_T = (0.0, 0.25, 0.5, 0.75, 1.0)  # where features are sampled along a stroke


def _odd(x: float) -> int:
    return max(1, int(round(x)) | 1)


def _blur(x: torch.Tensor, k: int) -> torch.Tensor:
    """Box average over k x k pixels (k odd), borders averaged over the pixels inside."""
    return F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False) if k > 1 else x


def _sample(feat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Bilinear features at view points `(B, N, K, 2)` in [0, 1] -> `(B, N, K, C)`."""
    out = F.grid_sample(feat, (points * 2.0 - 1.0).to(feat.dtype), mode="bilinear", align_corners=True)
    return out.permute(0, 2, 3, 1)


def _curve(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Quadratic Bezier points `(B, N, K, 2)` at parameters `t` `(K,)`."""
    t = t.view(1, 1, -1, 1)
    return (1 - t) ** 2 * p0.unsqueeze(2) + 2 * t * (1 - t) * p1.unsqueeze(2) + t ** 2 * p2.unsqueeze(2)


class PlanStrokePredictor(nn.Module):
    """
    Args (see `StrokePredictor` for the shared ones: widths, token_dim, heads, max_offset, radius_range,
    num_tools, paintable_mask, stroke_gate, error_progress, stationary, hard_gate):
        global_blocks: self-attention blocks over the plan tokens (view grid + whole-image grid + plan slots).
        query_blocks: pairs of (self-attention among the stroke latents, cross-attention to the plan).
        refine_layers: refinement layers (see the module docstring).
        view_tokens: side of the view's token grid (16: one token per 8 x 8 px of a 128 px view).
        global_grid: side of the whole image's token grid.
        plan_slots: learned plan tokens (a summary of what the step should do, learned end to end).
        pressure: strokes with a middle radius / opacity (pressure curve, 16 engine params).
        stroke_points: 'chord' (the anchor is the chord's middle, see the module docstring) or 'free' (all three
            points are free offsets from the anchor; checkpoints from before this option. Unroll training moved
            those strokes up to ~30 px away from their anchors, where their colors and features don't apply).
        color_base: where a stroke's color starts, before the decoder's correction: 'scaled' (the target along the
            stroke, averaged over about the stroke's width) or 'blur9' (always ~9 px averages; checkpoints from
            before the option. That blurred the color of the thin late strokes on edges and detail).
    """

    IN_CHANNELS = StrokePredictor.IN_CHANNELS

    def __init__(
        self,
        num_strokes: int = 16,
        widths: tuple = (32, 64, 128, 256),
        token_dim: int = 256,
        heads: int = 8,
        global_blocks: int = 6,
        query_blocks: int = 4,
        refine_layers: int = 3,
        view_tokens: int = 16,
        global_grid: int = 8,
        plan_slots: int = 8,
        max_offset: float = 0.35,
        radius_range: tuple = (0.004, 0.75),
        num_tools: int = 5,
        paintable_mask: bool = False,
        stroke_gate: bool = False,
        error_progress: bool = False,
        stationary: bool = False,
        hard_gate: bool = False,
        pressure: bool = True,
        stroke_points: str = "chord",
        color_base: str = "scaled",
    ):
        super().__init__()
        self.num_strokes = num_strokes
        self.paintable_mask = paintable_mask
        self.stroke_gate = stroke_gate
        self.error_progress = error_progress
        self.stationary = stationary
        self.hard_gate = hard_gate
        self.pressure = pressure
        if stroke_points not in ("chord", "free"):
            raise ValueError(f"unknown stroke_points {stroke_points!r}")
        self.stroke_points = stroke_points
        if color_base not in ("scaled", "blur9"):
            raise ValueError(f"unknown color_base {color_base!r}")
        self.color_base = color_base
        self.in_channels = self.IN_CHANNELS + int(paintable_mask)
        self.view_tokens = view_tokens
        self.global_grid = global_grid
        self.max_offset = max_offset
        self.radius_range = radius_range
        self.placement_noise = 0.0  # runtime switches, as in StrokePredictor
        self.sample_in_eval = False
        w0, w1, w2, w3 = widths
        D = token_dim

        n_cond = 2 + int(error_progress or stationary)
        self.cond_mlp = nn.Sequential(nn.Linear(n_cond, D), nn.SiLU(), nn.Linear(D, D))

        # Local path: residual U-Net, three downsamples
        self.stem = nn.Conv2d(self.in_channels, w0, 3, padding=1)
        self.enc0 = Stage(w0, w0, D)
        self.down1 = nn.Conv2d(w0, w1, 3, stride=2, padding=1)
        self.enc1 = Stage(w1, w1, D)
        self.down2 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)
        self.enc2 = Stage(w2, w2, D)
        self.down3 = nn.Conv2d(w2, w3, 3, stride=2, padding=1)
        self.enc3 = Stage(w3, w3, D)

        # Plan: view tokens, whole-image tokens and plan slots, with image-space positions and a type each
        self.to_tokens = nn.Linear(w3, D)
        self.token_pos = FourierPosition(D)
        self.token_type = nn.Parameter(torch.zeros(3, D))  # 0: whole image, 1: view, 2: plan slot
        self.slots = nn.Parameter(torch.randn(plan_slots, D) * 0.02)
        self.global_blocks = nn.ModuleList([TransformerBlock(D, heads) for _ in range(global_blocks)])
        self.from_tokens = nn.Conv2d(D, w3, 1)
        nn.init.zeros_(self.from_tokens.weight)
        nn.init.zeros_(self.from_tokens.bias)

        # Decoder back to full resolution
        self.mid = Stage(w3, w3, D)
        self.up2 = nn.Conv2d(w3, w2, 3, padding=1)
        self.dec2 = Stage(2 * w2, w2, D)
        self.up1 = nn.Conv2d(w2, w1, 3, padding=1)
        self.dec1 = Stage(2 * w1, w1, D)
        self.up0 = nn.Conv2d(w1, w0, 3, padding=1)
        self.dec0 = Stage(2 * w0, 2 * w0, D)
        self.pixel_dim = P = 2 * w0
        self.pixel_norm = nn.GroupNorm(_groups(P), P)

        # Stroke latents: queries coordinate and read the plan
        self.queries = nn.Parameter(torch.randn(num_strokes, D) * 0.02)
        self.query_cond = nn.Linear(D, D)
        self.query_blocks = nn.ModuleList()
        for _ in range(query_blocks):
            self.query_blocks.append(TransformerBlock(D, heads))
            self.query_blocks.append(TransformerBlock(D, heads, cross=True))

        # Initial placement (as in StrokePredictor) with a weaker error prior on a wider error map
        self.place_key = nn.Linear(D, P)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        side = math.ceil(math.sqrt(num_strokes))
        cells = (torch.arange(side, dtype=torch.float32) + 0.5) / side
        ref = torch.stack(torch.meshgrid(cells, cells, indexing="xy"), dim=-1).reshape(-1, 2)[:num_strokes]
        self.ref_logits = nn.Parameter(torch.log(ref) - torch.log1p(-ref))
        self.ref_log_sigma = nn.Parameter(torch.tensor(math.log(0.25)))
        self.error_gain = nn.Parameter(torch.tensor(1.0))

        # Latent init from the features at the initial anchor + the target color at three scales
        point_dim = P + w2 + w3 + 9
        self.init_proj = nn.Sequential(nn.LayerNorm(point_dim), nn.Linear(point_dim, D))

        # Refinement: strokes look along their current curve, attend to each other and to the plan
        curve_dim = len(CURVE_T) * (P + w2)
        self.refine = nn.ModuleList()
        for _ in range(refine_layers):
            self.refine.append(nn.ModuleDict(dict(
                feat=nn.Sequential(nn.LayerNorm(curve_dim), nn.Linear(curve_dim, D)),
                self_attn=TransformerBlock(D, heads), cross_attn=TransformerBlock(D, heads, cross=True),
            )))

        # Shared stroke decoder (latent -> stroke): anchor shift 2, point offsets 6, radii, opacity, hardness 1,
        # color 3, tools, (gate)
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
    _encode = StrokePredictor._encode

    def _base_color(self, colors: torch.Tensor, look: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        """
        The target along the stroke `(B, N, 3)`, averaged over about the stroke's width: the three blur levels of
        `colors` (1, ~9 and ~33 px at 128 px) mixed by the stroke's diameter in pixels, linearly in log size.
        """
        levels = _sample(colors, look).float().mean(2).unflatten(-1, (3, 3))  # (B, N, level, rgb)
        if self.stroke_points == "free" or self.color_base == "blur9":
            return levels[..., 1, :]
        R = colors.shape[-1]
        sizes = torch.tensor([1.0, _odd(9 * R / 128), _odd(33 * R / 128)], device=colors.device).log()
        width = (2.0 * radii.detach().float().mean(-1) * (R - 1)).clamp(min=1.0).log().clamp(max=sizes[-1])
        lo = (width.unsqueeze(-1) >= sizes[1:2]).long().squeeze(-1)  # 0: between 1 and 9 px, 1: 9 and 33 px
        frac = ((width - sizes[lo]) / (sizes[lo + 1] - sizes[lo])).clamp(0.0, 1.0).unsqueeze(-1)
        pick = lambda i: levels.gather(2, i.view(*i.shape, 1, 1).expand(*i.shape, 1, 3)).squeeze(2)
        return (1.0 - frac) * pick(lo) + frac * pick(lo + 1)

    @staticmethod
    def _curve_t(device: torch.device) -> torch.Tensor:
        return torch.tensor(CURVE_T, device=device)

    def _grid_positions(self, boxes: torch.Tensor, G: int) -> torch.Tensor:
        centers = (torch.arange(G, device=boxes.device, dtype=boxes.dtype) + 0.5) / G
        x = boxes[:, 0, None, None] + boxes[:, 2, None, None] * centers.view(1, 1, G)
        y = boxes[:, 1, None, None] + boxes[:, 2, None, None] * centers.view(1, G, 1)
        return torch.stack([x.expand(-1, G, G), y.expand(-1, G, G)], dim=-1).reshape(-1, G * G, 2)

    def _tokens(self, h3: torch.Tensor, boxes: torch.Tensor, G: int, kind: int) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(h3, G).flatten(2).transpose(1, 2)
        return self.to_tokens(pooled) + self.token_pos(self._grid_positions(boxes, G)) + self.token_type[kind]

    def encode_global(
        self, whole_inputs: torch.Tensor, progress: torch.Tensor, error_progress: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Whole-image tokens `(B, G*G, D)`, shared by all tile views of that image and step."""
        boxes = whole_inputs.new_tensor([0.0, 0.0, 1.0]).expand(whole_inputs.shape[0], 3)
        cond = self._cond(progress, boxes[:, 2], error_progress)
        return self._tokens(self._encode(whole_inputs, cond)[3], boxes, self.global_grid, kind=0)

    def _decode(self, h: torch.Tensor, anchor0: torch.Tensor, colors: torch.Tensor):
        """
        The stroke a latent describes: (anchor, points, radii, opacity, hardness, color, tool logits, raw gate logit).
        The anchor is the initial anchor plus a small shift and the middle of the chord p0 -> p2 (half chord up to
        max_offset); the control point is at most max_offset / 2 from it, so the curve passes within max_offset / 4
        of the anchor. The color is a correction to the target along the stroke at about the stroke's width (see
        `_base_color`; looked up at the detached stroke: where to look, not a gradient path). `colors`: the target
        at 1, ~9 and ~33 px averages, stacked on channels.
        """
        out = self.decoder(h).float()
        e = self.n_ends
        shift, offsets, r_raw, a_raw, h_raw, c_raw, rest = out.split(
            [2, 6, e, e, 1, 3, out.shape[-1] - 12 - 2 * e], dim=-1)
        anchor = anchor0 + 0.1 * torch.tanh(shift)
        off = self.max_offset * torch.tanh(offsets.unflatten(-1, (3, 2)))  # (B, N, 3, 2)
        if self.stroke_points == "chord":
            half = 0.5 * (off[..., 0, :] - off[..., 2, :])
            points = torch.stack([anchor + half, anchor + 0.5 * off[..., 1, :], anchor - half], 2)
            look = _curve(*points.detach().unbind(2), self._curve_t(h.device))
        else:
            points = anchor.unsqueeze(2) + off
            look = anchor.detach().unsqueeze(2)
        r_lo, r_hi = (math.log(r) for r in self.radius_range)
        r_init = (math.log(0.1) - r_lo) / (r_hi - r_lo)  # start at radius ~0.1 view units
        radii = torch.exp(r_lo + (r_hi - r_lo) * torch.sigmoid(r_raw + math.log(r_init / (1.0 - r_init))))
        opacity = 0.05 + 0.95 * torch.sigmoid(a_raw)
        tc = self._base_color(colors, look, radii).clamp(COLOR_EPS, 1.0 - COLOR_EPS)
        color = torch.sigmoid(torch.log(tc) - torch.log1p(-tc) + c_raw)
        tool_logits, gate_raw = (rest[..., :-1], rest[..., -1]) if self.stroke_gate else (rest, None)
        return anchor, points, radii, opacity, torch.sigmoid(h_raw), color, tool_logits, gate_raw

    def forward(
        self,
        inputs: torch.Tensor,
        boxes: torch.Tensor,
        progress: torch.Tensor,
        global_tokens: Optional[torch.Tensor] = None,
        view_size: Optional[torch.Tensor] = None,
        error_progress: Optional[torch.Tensor] = None,
        samples: Optional[StrokeSamples] = None,
    ) -> StrokeOutput:
        """Same contract as `StrokePredictor.forward`."""
        sampling = self.training or self.sample_in_eval
        B, _, R, _ = inputs.shape
        cond = self._cond(progress, boxes[:, 2] if view_size is None else view_size, error_progress)
        h0, h1, h2, h3 = self._encode(inputs, cond)

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
        d1 = self.dec1(torch.cat([self.up1(F.interpolate(d2, size=h1.shape[-2:], mode="nearest")), h1], 1), cond)
        d0 = self.dec0(torch.cat([self.up0(F.interpolate(d1, size=h0.shape[-2:], mode="nearest")), h0], 1), cond)
        pixel = self.pixel_norm(d0)

        # Stroke latents
        h = self.queries.unsqueeze(0).expand(B, -1, -1) + self.query_cond(cond).unsqueeze(1)
        for i in range(0, len(self.query_blocks), 2):
            h = self.query_blocks[i](h)
            h = self.query_blocks[i + 1](h, context=tokens)

        # Initial placement: spatial soft-argmax + priors (home position, where the canvas is still wrong)
        key = self.place_key(h)
        logits = torch.einsum("bnp,bphw->bnhw", key, pixel).flatten(2) / math.sqrt(self.pixel_dim)
        logits = logits.float() * self.log_temperature.exp()
        uv = view_grid(boxes.new_tensor([0.0, 0.0, 1.0]).expand(B, 3), R).reshape(B, R * R, 2)
        ref = torch.sigmoid(self.ref_logits)
        d_sq = ((uv[:1].float().unsqueeze(1) - ref.view(1, -1, 1, 2)) ** 2).sum(-1)
        logits = logits - d_sq / (2.0 * self.ref_log_sigma.exp() ** 2)
        k_err = _odd(9 * R / 128)
        err = _blur(inputs[:, 6:9].float().abs().mean(1, keepdim=True), k_err)
        logits = logits + self.error_gain * torch.log(err.flatten(2) + 1e-3)
        prob = logits.softmax(dim=-1)
        place_seed, place_prob = None, prob
        if self.placement_noise > 0 and (sampling or (samples is not None and samples.place_seed is not None)):
            place_seed = samples.place_seed if samples is not None and samples.place_seed is not None else \
                torch.randint(0, 2**62, (B,), device=logits.device)
            place_prob = (logits + self.placement_noise * _gumbel(place_seed, logits.shape[1:])).softmax(-1)
        anchor0 = torch.einsum("bnk,bkc->bnc", place_prob, uv.float())
        entropy = -(prob * prob.clamp(min=1e-12).log()).sum(-1)

        # Target color at three scales (single pixel, ~9 px and ~33 px averages at 128 px)
        target = inputs[:, 0:3]
        target_9 = _blur(target, _odd(9 * R / 128))
        target_33 = _blur(target, _odd(33 * R / 128))
        colors = torch.cat([target, target_9, target_33], dim=1)

        at0 = anchor0.detach().unsqueeze(2)  # where to look (the anchor itself learns through the final stroke)
        point_feats = torch.cat([_sample(pixel, at0), _sample(d2, at0), _sample(m, at0), _sample(colors, at0)], -1)
        h = h + self.init_proj(point_feats[:, :, 0].to(h.dtype))

        t_curve = self._curve_t(inputs.device)

        # Each layer decodes its current strokes only to look along them: detached there (as reference points in
        # deformable DETR decoders). Through the sampling positions the gradient is the feature difference
        # between neighbouring pixels, unbounded, and stroke -> sampled features -> latent -> stroke is a loop
        # that blew up within ~100 teacher steps. The decoder learns through the final stroke below.
        for layer in self.refine:
            with torch.no_grad():
                points = self._decode(h, anchor0, colors)[1]
            pts = _curve(*points.unbind(2), t_curve)
            seen = torch.cat([_sample(pixel, pts), _sample(d2, pts)], -1).flatten(2)  # (B, N, K * (P + w2))
            h = h + layer["feat"](seen.to(h.dtype))
            h = layer["self_attn"](h)
            h = layer["cross_attn"](h, context=tokens)

        # The output: the same decoder on the last latent
        anchor, points, radii, opacity, hardness, color, tool_logits, gate_raw = self._decode(h, anchor0, colors)
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
        )


def build_model(config: dict) -> nn.Module:
    """The model a checkpoint's `model_config` describes (`arch`: 'unet' = `StrokePredictor`, the default for
    checkpoints from before the key existed, 'plan' = `PlanStrokePredictor` or 'frame' = `FrameStrokePredictor`)."""
    config = dict(config)
    arch = config.pop("arch", "unet")
    joint = config.pop("joint_strokes", False)
    opaque = config.pop("opaque_canvas", False)
    mask_tool = config.pop("mask_tool", False)
    if arch == "frame":
        if not joint:
            raise ValueError("the frame architecture paints joint steps (joint_strokes)")
        from overpaintai.frame_model import FrameStrokePredictor  # imports this module

        model = FrameStrokePredictor(**config)
    elif arch == "plan":
        config.setdefault("stroke_points", "free")  # plan checkpoints from before these options
        config.setdefault("color_base", "blur9")
        model = PlanStrokePredictor(**config)
    elif arch == "unet":
        model = StrokePredictor(**config)
    else:
        raise ValueError(f"unknown model architecture {arch!r}")
    # How the model's strokes are rendered, not part of the network: each step's strokes as one order-independent
    # operation (`DifferentiableCanvasEngine.forward_joint_step`) instead of one after another
    model.joint_strokes = bool(joint)
    # The canvas it paints on (`DifferentiableCanvasEngine(opaque=...)`) and whether the eraser is a per-step mask
    # (`eraser_mode="mask"`): engines that render its strokes are built to match
    model.opaque_canvas = bool(opaque)
    model.mask_tool = bool(mask_tool)
    return model
