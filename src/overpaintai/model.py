"""
Stroke predictor: (target, current canvas) seen through a view -> N strokes for that view.

Local path: a residual U-Net on the R x R view keeps pixel-level detail for accurate placement.
Global path: the deepest features are pooled to a fixed 8 x 8 token grid (whatever the input size), tagged
with Fourier encodings of their image-space position, and attended jointly with tokens of the whole image,
so even a zoomed-in tile view "sees" the entire picture. Stroke queries attend to those tokens to decide
what to paint, then place themselves on the full-resolution feature map via spatial soft-argmax.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from overpaintai.views import Strokes, view_grid


GATE_LEAK = 0.02  # minimum gradient of the stroke gate w.r.t. its logit in training
# Stroke colours start from the target colour clamped to [COLOR_EPS, 1 - COLOR_EPS] (in logit space). At 0.02 the
# base could not reach flat fills with a channel near 0 or 1 (e.g. sky blue, B = 1.0): the leftover error made
# joint models alternate between washing such areas lighter and repainting them
COLOR_EPS = 0.005
GATE_LOGIT_RANGE = 6.0  # logits beyond +-this are pulled back (sigmoid(-6) = 0.0025: off, but recoverable)


def _groups(channels: int) -> int:
    return min(32, channels // 4)


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> conv3x3, twice, with a residual connection and a FiLM-style conditioning bias."""

    def __init__(self, c_in: int, c_out: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(c_in), c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.cond = nn.Linear(cond_dim, c_out)
        self.norm2 = nn.GroupNorm(_groups(c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond(cond)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class Stage(nn.Module):
    def __init__(self, c_in: int, c_out: int, cond_dim: int, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlock(c_in if i == 0 else c_out, c_out, cond_dim) for i in range(depth)])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond)
        return x


class FourierPosition(nn.Module):
    """Encodes continuous (x, y) in [0, 1] with sin/cos at octave frequencies, then projects to `dim`."""

    def __init__(self, dim: int, octaves: int = 6):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(octaves)) * math.pi, persistent=False)
        self.proj = nn.Linear(4 * octaves, dim)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:  # (..., 2) -> (..., dim)
        ang = xy.unsqueeze(-1) * self.freqs  # (..., 2, octaves)
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)
        return self.proj(feats)


class TransformerBlock(nn.Module):
    """Pre-norm self-attention (or cross-attention when `context` is given) + MLP."""

    def __init__(self, dim: int, heads: int, cross: bool = False):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_kv = nn.LayerNorm(dim) if cross else None
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.norm_q(x)
        kv = q if context is None else self.norm_kv(context)
        x = x + self.attn(q, kv, kv, need_weights=False)[0]
        return x + self.mlp(self.norm_mlp(x))


@dataclass
class StrokeOutput:
    strokes: Strokes  # in view coordinates
    tool_logits: torch.Tensor  # (B, N, 5)
    anchors: torch.Tensor  # (B, N, 2) curve midpoints in view coordinates
    anchor_entropy: torch.Tensor  # (B, N) entropy of each query's placement map (nats)
    placement_logits: torch.Tensor  # (B, N, R*R) pre-softmax placement maps (for heatmap supervision)
    gate: torch.Tensor  # (B, N) in [0, 1]: how much of each stroke to paint (1 for models without a gate)
    gate_logit: Optional[torch.Tensor] = None  # (B, N) pre-sigmoid gate (for the range penalty in training)
    samples: Optional["StrokeSamples"] = None  # the random draws of this output (sampled placement / gates)


@dataclass
class StrokeSamples:
    """The random choices of one forward pass, so it can be replayed exactly (see `StrokePredictor.forward`)."""

    place_seed: Optional[torch.Tensor] = None  # (B,) long: seed of each view's placement noise
    gate_on: Optional[torch.Tensor] = None  # (B, N) float in {0, 1}: sampled hard gates


def _gumbel(seeds: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Standard Gumbel noise `(B, *shape)`, image b drawn from its own seed (so one image can be replayed)."""
    out = []
    for seed in seeds.tolist():
        g = torch.Generator(device=seeds.device).manual_seed(seed)
        u = torch.rand(shape, generator=g, device=seeds.device).clamp(1e-9, 1.0 - 1e-9)
        out.append(-torch.log(-torch.log(u)))
    return torch.stack(out)


class StrokePredictor(nn.Module):
    """
    Args:
        num_strokes: strokes (queries) predicted per view and step.
        widths: U-Net channels per level (full res, /2, /4, /8).
        token_dim: width of the global tokens, queries and attention blocks.
        token_grid: side of the fixed token grid (tokens per view = token_grid^2).
        max_offset: maximum endpoint distance from the anchor, in view units.
        radius_range: (min, max) stroke radius in view units; predicted on a log scale so fine and
            canvas-filling strokes are equally easy to hit.
        num_tools: size of the tool logits (all engine tools; unused ones can be masked).
        paintable_mask: expect a 13th input channel marking where the view may be painted (1) and what is
            context only (0), see `views.view_inputs(..., paintable=...)`.
        stroke_gate: predict a gate per stroke that scales its opacity, so a step can leave strokes out
            (trained with a per-stroke cost).
        error_progress: condition on a second progress value derived from the canvas error, in addition
            to the step-based progress (see `views.error_progress`).
        stationary: condition only on the canvas, not on the step: the progress slot gets the error-based
            progress instead of step / num_steps (the `progress` argument is ignored), plus a log-scale
            version of it that still tells a 90% from a 99% finished view. The same model then fits any
            number of steps (step 3 and step 3000 are the same problem).
        hard_gate: the stroke gate is either on or off instead of a dimmer: logit > 0 at inference, sampled
            with the gate probability in training, with the soft gate's gradient (straight-through). With a
            soft gate, a stroke cost makes dimming strokes cheaper than dropping them, which piles up faint
            strokes. Sampling matters: at full opacity a well-placed stroke has ~zero marginal benefit, so a
            deterministic gate only ever learns "off"; sampled off, the gradient sees what it is worth.
        The switches default to off so that checkpoints from before they existed load unchanged.
    """

    IN_CHANNELS = 12  # see views.view_inputs (13 with paintable_mask)

    def __init__(
        self,
        num_strokes: int = 16,
        widths: tuple = (32, 64, 128, 256),
        token_dim: int = 256,
        token_grid: int = 8,
        heads: int = 8,
        global_blocks: int = 2,
        query_blocks: int = 2,
        max_offset: float = 0.35,
        radius_range: tuple = (0.004, 0.75),
        num_tools: int = 5,
        paintable_mask: bool = False,
        stroke_gate: bool = False,
        error_progress: bool = False,
        stationary: bool = False,
        hard_gate: bool = False,
    ):
        super().__init__()
        # Runtime switches (not part of the architecture):
        # placement_noise > 0 adds Gumbel noise of that scale to the placement logits while sampling, so the
        # soft-argmax leans towards one of a map's good spots per draw (different attempts commit to different
        # plans, for --best-of) while positions stay continuous; sample_in_eval draws noise / gates at inference
        self.placement_noise = 0.0
        self.sample_in_eval = False
        self.num_strokes = num_strokes
        self.paintable_mask = paintable_mask
        self.stroke_gate = stroke_gate
        self.error_progress = error_progress
        self.stationary = stationary
        self.hard_gate = hard_gate
        self.in_channels = self.IN_CHANNELS + int(paintable_mask)
        self.token_grid = token_grid
        self.max_offset = max_offset
        self.radius_range = radius_range
        w0, w1, w2, w3 = widths
        cond_dim = token_dim

        # Conditioning: step progress in [0, 1], log2 of the view size (and the error-based progress);
        # stationary: error-based progress, log2 of the view size, log-scale error progress
        n_cond = 2 + int(error_progress or stationary)
        self.cond_mlp = nn.Sequential(nn.Linear(n_cond, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        # Local path: residual U-Net, three downsamples
        self.stem = nn.Conv2d(self.in_channels, w0, 3, padding=1)
        self.enc0 = Stage(w0, w0, cond_dim)
        self.down1 = nn.Conv2d(w0, w1, 3, stride=2, padding=1)
        self.enc1 = Stage(w1, w1, cond_dim)
        self.down2 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)
        self.enc2 = Stage(w2, w2, cond_dim)
        self.down3 = nn.Conv2d(w2, w3, 3, stride=2, padding=1)
        self.enc3 = Stage(w3, w3, cond_dim)

        # Global path: fixed token grid with image-space Fourier positions
        self.to_tokens = nn.Linear(w3, token_dim)
        self.token_pos = FourierPosition(token_dim)
        self.token_type = nn.Parameter(torch.zeros(2, token_dim))  # 0: global, 1: view
        self.global_blocks = nn.ModuleList([TransformerBlock(token_dim, heads) for _ in range(global_blocks)])
        self.from_tokens = nn.Conv2d(token_dim, w3, 1)
        nn.init.zeros_(self.from_tokens.weight)
        nn.init.zeros_(self.from_tokens.bias)

        # Decoder back to full resolution
        self.mid = Stage(w3, w3, cond_dim)
        self.up2 = nn.Conv2d(w3, w2, 3, padding=1)
        self.dec2 = Stage(2 * w2, w2, cond_dim)
        self.up1 = nn.Conv2d(w2, w1, 3, padding=1)
        self.dec1 = Stage(2 * w1, w1, cond_dim)
        self.up0 = nn.Conv2d(w1, w0, 3, padding=1)
        self.dec0 = Stage(2 * w0, 2 * w0, cond_dim)
        self.pixel_dim = 2 * w0

        # Stroke queries
        self.queries = nn.Parameter(torch.randn(num_strokes, token_dim) * 0.02)
        self.query_cond = nn.Linear(cond_dim, token_dim)
        self.query_blocks = nn.ModuleList()
        for _ in range(query_blocks):
            self.query_blocks.append(TransformerBlock(token_dim, heads))  # queries coordinate
            self.query_blocks.append(TransformerBlock(token_dim, heads, cross=True))  # read the image
        self.place_key = nn.Linear(token_dim, self.pixel_dim)
        self.pixel_norm = nn.GroupNorm(_groups(self.pixel_dim), self.pixel_dim)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        # Placement priors: a learnable home position per query (initially a grid) with a soft Gaussian
        # preference of learnable width, and a learnable bonus for pixels where the canvas is still wrong
        side = math.ceil(math.sqrt(num_strokes))
        cells = (torch.arange(side, dtype=torch.float32) + 0.5) / side
        ref = torch.stack(torch.meshgrid(cells, cells, indexing="xy"), dim=-1).reshape(-1, 2)[:num_strokes]
        self.ref_logits = nn.Parameter(torch.log(ref) - torch.log1p(-ref))
        self.ref_log_sigma = nn.Parameter(torch.tensor(math.log(0.25)))
        self.error_gain = nn.Parameter(torch.tensor(2.0))

        # Per-stroke attributes from the query + features sampled at the anchor
        feat_dim = token_dim + self.pixel_dim + w2 + w3 + 3
        self.attr_mlp = nn.Sequential(
            nn.Linear(feat_dim, 2 * token_dim), nn.SiLU(),
            nn.Linear(2 * token_dim, token_dim), nn.SiLU(),
            nn.Linear(token_dim, 4 + 2 + 2 + 1 + 3 + num_tools + int(stroke_gate)),  # gate last (see forward)
        )
        last = self.attr_mlp[-1]
        nn.init.normal_(last.weight, std=1e-3)
        nn.init.zeros_(last.bias)

    # ------------------------------------------------------------------------------------------------
    def _cond(
        self, progress: torch.Tensor, view_size: torch.Tensor, error_progress: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        log_size = torch.log2(view_size.clamp(min=1e-4))
        if self.stationary:
            e = torch.zeros_like(view_size) if error_progress is None else error_progress.to(view_size.dtype)
            # remaining error on a log scale: 0 at blank, 1 at 1/1024 of the blank canvas' error
            fine = (-torch.log2((1.0 - e).clamp(min=2.0**-10)) / 10.0).clamp(0.0, 1.0)
            return self.cond_mlp(torch.stack([e, log_size, fine], dim=-1))
        parts = [progress, log_size]
        if self.error_progress:
            parts.append(torch.zeros_like(progress) if error_progress is None else error_progress)
        return self.cond_mlp(torch.stack(parts, dim=-1))

    @property
    def uses_error_progress(self) -> bool:
        """Whether callers must pass `error_progress` (see `views.error_progress`)."""
        return self.error_progress or self.stationary

    def _token_positions(self, boxes: torch.Tensor) -> torch.Tensor:
        """Image-space centers of the token grid cells of each view, `(B, G*G, 2)`."""
        G = self.token_grid
        centers = (torch.arange(G, device=boxes.device, dtype=boxes.dtype) + 0.5) / G
        x = boxes[:, 0, None, None] + boxes[:, 2, None, None] * centers.view(1, 1, G)
        y = boxes[:, 1, None, None] + boxes[:, 2, None, None] * centers.view(1, G, 1)
        return torch.stack([x.expand(-1, G, G), y.expand(-1, G, G)], dim=-1).reshape(-1, G * G, 2)

    def _encode(self, x: torch.Tensor, cond: torch.Tensor):
        h0 = self.enc0(self.stem(x), cond)
        h1 = self.enc1(self.down1(h0), cond)
        h2 = self.enc2(self.down2(h1), cond)
        h3 = self.enc3(self.down3(h2), cond)
        return h0, h1, h2, h3

    def _tokens(self, h3: torch.Tensor, boxes: torch.Tensor, kind: int) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(h3, self.token_grid).flatten(2).transpose(1, 2)  # (B, G*G, C)
        return self.to_tokens(pooled) + self.token_pos(self._token_positions(boxes)) + self.token_type[kind]

    def encode_global(
        self, whole_inputs: torch.Tensor, progress: torch.Tensor, error_progress: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Global tokens `(B, G*G, D)` from the whole-image view (inputs built with box (0, 0, 1)); share them
        with all tile views of the same image and step via `forward(..., global_tokens=...)`.
        """
        boxes = whole_inputs.new_tensor([0.0, 0.0, 1.0]).expand(whole_inputs.shape[0], 3)
        cond = self._cond(progress, boxes[:, 2], error_progress)
        return self._tokens(self._encode(whole_inputs, cond)[3], boxes, kind=0)

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
    ) -> StrokeOutput:
        """
        Args:
            inputs: (B, 12 or 13, R, R) view inputs from `views.view_inputs` (any R; trained at a fixed R).
            boxes: (B, 3) view boxes (x, y, s) in image coordinates.
            progress: (B,) painting progress in [0, 1] (e.g. step / num_steps); ignored by stationary models.
            global_tokens: (B, G*G, D) from `encode_global`; if omitted, this view's own tokens are used
                as global context (correct for whole-image views).
            view_size: (B,) view size fed to the conditioning instead of `boxes[:, 2]`, e.g. clamped to
                the range seen in training when painting smaller tiles (token positions still use `boxes`).
            error_progress: (B,) error-based progress (`views.error_progress`), for models with
                `uses_error_progress`.
            samples: replay these random draws (from an earlier `StrokeOutput.samples` on the same inputs)
                instead of drawing new ones.
        """
        sampling = self.training or self.sample_in_eval
        B, _, R, _ = inputs.shape
        cond = self._cond(progress, boxes[:, 2] if view_size is None else view_size, error_progress)
        h0, h1, h2, h3 = self._encode(inputs, cond)

        view_tok = self._tokens(h3, boxes, kind=1)
        if global_tokens is None:
            global_tokens = view_tok - self.token_type[1] + self.token_type[0]
        n_global = global_tokens.shape[1]
        tokens = torch.cat([global_tokens, view_tok], dim=1)
        for block in self.global_blocks:
            tokens = block(tokens)

        # Inject attended view tokens back into the bottleneck so pixel features carry global context
        G = self.token_grid
        grid_tok = tokens[:, n_global:].transpose(1, 2).reshape(B, -1, G, G)
        h3 = h3 + F.interpolate(self.from_tokens(grid_tok), size=h3.shape[-2:], mode="bilinear", align_corners=False)
        m = self.mid(h3, cond)

        d2 = self.dec2(torch.cat([self.up2(F.interpolate(m, size=h2.shape[-2:], mode="nearest")), h2], 1), cond)
        d1 = self.dec1(torch.cat([self.up1(F.interpolate(d2, size=h1.shape[-2:], mode="nearest")), h1], 1), cond)
        d0 = self.dec0(torch.cat([self.up0(F.interpolate(d1, size=h0.shape[-2:], mode="nearest")), h0], 1), cond)
        pixel = self.pixel_norm(d0)  # (B, P, R, R)

        # Queries read the tokens
        q = self.queries.unsqueeze(0).expand(B, -1, -1) + self.query_cond(cond).unsqueeze(1)
        for i in range(0, len(self.query_blocks), 2):
            q = self.query_blocks[i](q)
            q = self.query_blocks[i + 1](q, context=tokens)

        # Placement: spatial soft-argmax of query-key . pixel-feature similarity, plus the priors
        key = self.place_key(q)  # (B, N, P)
        logits = torch.einsum("bnp,bphw->bnhw", key, pixel).flatten(2) / math.sqrt(self.pixel_dim)
        logits = logits.float() * self.log_temperature.exp()
        uv = view_grid(boxes.new_tensor([0.0, 0.0, 1.0]).expand(B, 3), R).reshape(B, R * R, 2)
        ref = torch.sigmoid(self.ref_logits)  # (N, 2)
        d_sq = ((uv[:1].float().unsqueeze(1) - ref.view(1, -1, 1, 2)) ** 2).sum(-1)  # (1, N, R*R)
        logits = logits - d_sq / (2.0 * self.ref_log_sigma.exp() ** 2)
        err = F.avg_pool2d(inputs[:, 6:9].float().abs().mean(1, keepdim=True), 5, stride=1, padding=2)
        logits = logits + self.error_gain * torch.log(err.flatten(2) + 1e-3)  # (B, 1, R*R) broadcast
        prob = logits.softmax(dim=-1)  # (B, N, R*R)
        place_seed, place_prob = None, prob
        if self.placement_noise > 0 and (sampling or (samples is not None and samples.place_seed is not None)):
            if samples is not None and samples.place_seed is not None:
                place_seed = samples.place_seed
            else:
                place_seed = torch.randint(0, 2**62, (B,), device=logits.device)
            place_prob = (logits + self.placement_noise * _gumbel(place_seed, logits.shape[1:])).softmax(-1)
        anchors = torch.einsum("bnk,bkc->bnc", place_prob, uv.float()).to(inputs.dtype)  # view coords
        entropy = -(prob * prob.clamp(min=1e-12).log()).sum(-1)

        # Features at the anchors from several decoder levels, plus the target color there
        grid = (anchors * 2.0 - 1.0).unsqueeze(2)  # (B, N, 1, 2)

        def sample(feat: torch.Tensor) -> torch.Tensor:
            out = F.grid_sample(feat, grid.to(feat.dtype), mode="bilinear", align_corners=True)
            return out.squeeze(-1).transpose(1, 2)  # (B, N, C)

        target_color = sample(inputs[:, 0:3])
        feats = torch.cat([q, sample(pixel), sample(d2), sample(m), target_color], dim=-1)
        out = self.attr_mlp(feats)

        gate_logit, gate_on = None, None
        if self.stroke_gate:  # start with every stroke on (sigmoid(3) = 0.95)
            gate_logit, out = out[..., -1].float() + 3.0, out[..., :-1]
            gate = torch.sigmoid(gate_logit)
            if self.training:
                # Leaky gradient (forward value unchanged): a saturated, switched-off gate still gets a
                # learning signal and can come back, instead of dying
                gate = gate + GATE_LEAK * (gate_logit - gate_logit.detach())
            if self.hard_gate:  # on / off; gradient of the soft gate (straight-through)
                if samples is not None and samples.gate_on is not None:
                    gate_on = samples.gate_on
                elif sampling:
                    gate_on = torch.bernoulli(gate.detach().clamp(0.0, 1.0))
                else:
                    gate_on = (gate_logit > 0).float()
                gate = gate_on + (gate - gate.detach())
        else:
            gate = torch.ones(out.shape[:-1], device=out.device)
        offsets, out = out[..., 0:4], out[..., 4:]
        r_raw, a_raw, h_raw, c_raw, tool_logits = out.split([2, 2, 1, 3, out.shape[-1] - 8], dim=-1)

        p0 = anchors + self.max_offset * torch.tanh(offsets[..., 0:2])
        p2 = anchors + self.max_offset * torch.tanh(offsets[..., 2:4])
        p1 = 2.0 * anchors - 0.5 * (p0 + p2)  # so that B(0.5) = 0.25 p0 + 0.5 p1 + 0.25 p2 = anchor
        r_lo, r_hi = (math.log(r) for r in self.radius_range)
        # Start at radius ~0.1 (view units), half-opaque, medium-hard, in the target's color
        r_init = (math.log(0.1) - r_lo) / (r_hi - r_lo)
        radii = torch.exp(r_lo + (r_hi - r_lo) * torch.sigmoid(r_raw + math.log(r_init / (1.0 - r_init))))
        opacity = 0.05 + 0.95 * torch.sigmoid(a_raw)
        hardness = torch.sigmoid(h_raw)
        tc = target_color.clamp(COLOR_EPS, 1.0 - COLOR_EPS)
        color = torch.sigmoid(torch.log(tc) - torch.log1p(-tc) + c_raw)

        strokes = Strokes(
            points=torch.stack([p0, p1, p2], dim=2).float(),
            radii=radii.float(),
            opacity=opacity.float(),
            hardness=hardness.float(),
            color=color.float(),
        )
        return StrokeOutput(
            strokes=strokes, tool_logits=tool_logits.float(), anchors=anchors.float(),
            anchor_entropy=entropy, placement_logits=logits, gate=gate, gate_logit=gate_logit,
            samples=StrokeSamples(place_seed, gate_on),
        )
