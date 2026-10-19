import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableCanvasEngine(nn.Module):
    """
    A resolution-independent, fully differentiable 2D digital painting engine in PyTorch.
    
    Uses continuous capsule/line-segment distance fields along the Quadratic Bezier curve
    (eliminating discrete circle-stamp beading artifacts) and a true RGBA paint layer
    `(B, 4, H, W)` separated from a static or custom background `(B, 3, H, W)`.

    Continuous Parameters (14 in [0, 1]):
      [x0, y0, x1, y1, x2, y2, r0, r2, a0, a2, hardness, R, G, B]

    Discrete Tool Modes (5 tools, all sharing the continuous `hardness` attribute):
      0: Paint Brush (Standard pigment deposition, from soft airbrush at h=0 to hard cel ink at h=1)
      1: Dry / Textured Brush (Pigment deposition modulated by procedural canvas/chalk grain)
      2: Smudge / Push (Brush carries a paint patch along the curve, depositing and picking up as it moves)
      3: Blur / Blend (In-place isotropic softening of colors and edges underneath the brush)
      4: Eraser (Subtractive alpha removal, from soft feathered eraser at h=0 to hard cutout at h=1)
    """

    NUM_CONT_PARAMS = 14  # x0, y0, x1, y1, x2, y2, r0, r2, a0, a2, hardness, R, G, B
    NUM_TOOLS = 5         # Paint, Dry/Chalk, Smudge, Blur/Blend, Eraser

    def __init__(
        self,
        canvas_size: int = 256,
        num_stamps: int = 32,
        min_radius: float = 0.005,
        max_radius: float = 0.25,
        hard_edge_px: float = 1.0,
        blur_ratio: float = 0.035,
        smudge_spacing: float = 0.25,
        smudge_length: float = 12.0,
        smudge_max_dabs: int = 128,
        bg_color: Tuple[float, float, float] = (0.97, 0.96, 0.93),
        bg_texture_strength: float = 0.04,
    ):
        super().__init__()
        self.canvas_size = canvas_size
        self.num_stamps = max(4, num_stamps)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.hard_edge_px = hard_edge_px
        self.blur_ratio = blur_ratio
        self.smudge_spacing = smudge_spacing
        self.smudge_length = smudge_length
        self.smudge_max_dabs = smudge_max_dabs
        self.bg_texture_strength = bg_texture_strength

        self.register_buffer("bg_color", torch.tensor(bg_color, dtype=torch.float32).view(1, 3, 1, 1))
        t = torch.linspace(0.0, 1.0, self.num_stamps, dtype=torch.float32).view(1, self.num_stamps, 1, 1)
        self.register_buffer("t", t)

        self._cached_hw = (0, 0)
        self._update_spatial_buffers(canvas_size, canvas_size, device=torch.device("cpu"))

    def _update_spatial_buffers(self, H: int, W: int, device: torch.device):
        """Creates normalized (x, y) coordinate grids, paper grain, default background, and 4-channel blur kernels."""
        if self._cached_hw == (H, W) and self.grid_x.device == device:
            return

        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, H, device=device, dtype=torch.float32),
            torch.linspace(0.0, 1.0, W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", xx.view(1, 1, H, W), persistent=False)
        self.register_buffer("grid_y", yy.view(1, 1, H, W), persistent=False)

        base_grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        self.register_buffer("base_grid", base_grid, persistent=False)

        # Multi-frequency procedural paper/chalk grain in [0.0, 1.0]
        grain = (
            0.50 * torch.sin(xx * 190.0 * math.pi) * torch.cos(yy * 190.0 * math.pi)
            + 0.30 * torch.sin((xx + yy) * 430.0 * math.pi)
            + 0.20 * torch.cos((xx - yy * 1.5) * 870.0 * math.pi)
        )
        grain = torch.sigmoid(grain * 3.5).view(1, 1, H, W)
        self.register_buffer("paper_grain", grain, persistent=False)

        # Default subtle woven canvas background texture (1, 3, H, W)
        weave = (
            0.5 * torch.sin(xx * 140.0 * math.pi)
            + 0.5 * torch.cos(yy * 140.0 * math.pi)
        ).view(1, 1, H, W)
        default_bg = (self.bg_color.to(device) - self.bg_texture_strength * (0.5 - 0.5 * weave)).clamp(0.0, 1.0)
        self.register_buffer("default_bg", default_bg, persistent=False)

        # Separable 1D Gaussian blur kernel for the 4-channel premultiplied RGBA Blur / Blend tool
        k_size = max(3, int(round(max(H, W) * self.blur_ratio)) | 1)
        sigma = k_size / 3.0
        coords = torch.arange(k_size, device=device, dtype=torch.float32) - (k_size - 1) / 2.0
        g1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        g1d = g1d / g1d.sum()
        self.register_buffer("blur_kh", g1d.view(1, 1, 1, k_size).expand(4, 1, 1, k_size).contiguous(), persistent=False)
        self.register_buffer("blur_kv", g1d.view(1, 1, k_size, 1).expand(4, 1, k_size, 1).contiguous(), persistent=False)
        self._blur_pad = k_size // 2

        self._cached_hw = (H, W)

    @staticmethod
    def _brush_profile(
        dist_sq: torch.Tensor, radius: torch.Tensor, hardness: torch.Tensor, edge_sharpness: float
    ) -> torch.Tensor:
        """Blends a soft Gaussian falloff (hardness 0) with a crisp disc edge (hardness 1)."""
        hard = torch.sigmoid(edge_sharpness * (radius - torch.sqrt(dist_sq + 1e-8)))
        soft = torch.exp(-2.0 * dist_sq / (radius * radius + 1e-6))
        return (1.0 - hardness) * soft + hardness * hard

    def _smudge(
        self,
        premult: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
        rt: torch.Tensor,
        at: torch.Tensor,
        hardness: torch.Tensor,
        w_smudge: torch.Tensor,
        edge_sharpness: float,
    ) -> torch.Tensor:
        """
        Physically-motivated smear of premultiplied RGBA `(B, 4, H, W)` along the stroke.

        The curve is resampled into closely spaced dabs. At each dab, the paint under the brush is
        replaced (at the smear rate times the dab profile) by the paint one step behind, translated
        forward by the dab-to-dab motion in whole pixels (exact copies, so nothing is blurred). Paint
        inside the footprint therefore travels with the brush:
          - starting on bare canvas, the brush carries nothing and digs an indent into paint it enters,
            gradually loading up with that paint and filling the indent as it goes,
          - starting on paint, it drags that paint along and leaves it as a fading trail.

        Carried paint drifts backwards relative to the brush by (1 - rate) per distance travelled and is
        left behind once it crosses the footprint (2 radii), so it is dragged about 2 / (1 - rate) radii.
        The rate is `opacity - (2 / smudge_length) * opacity^2`, which caps the drag distance at full
        opacity to `smudge_length` radii while barely changing low opacities. Blurring is the Blur / Blend tool's job.
        """
        B, _, H, W = premult.shape
        S = bx.shape[1]

        # Resample the curve into evenly spaced dabs (spacing relative to the smallest radius)
        with torch.no_grad():
            seg_len = torch.sqrt((bx[:, 1:] - bx[:, :-1]) ** 2 + (by[:, 1:] - by[:, :-1]) ** 2)
            length = seg_len.sum(dim=1).flatten()
            r_min = rt.amin(dim=1).flatten()
            n_dabs = int((length / (self.smudge_spacing * r_min)).max().ceil().item()) + 1
            n_dabs = max(2, min(self.smudge_max_dabs, n_dabs))

        def resample(v: torch.Tensor) -> torch.Tensor:  # (B, S, 1, 1) -> (B, n_dabs)
            return F.interpolate(v.view(B, 1, S), size=n_dabs, mode="linear", align_corners=True).view(B, n_dabs)

        dx, dy, dr, da = resample(bx), resample(by), resample(rt), resample(at)

        # Whole-pixel dab-to-dab motion; rounding absolute positions keeps rounding error from accumulating
        px = torch.round(dx.detach() * (W - 1))
        py = torch.round(dy.detach() * (H - 1))
        shift_x = (px[:, 1:] - px[:, :-1]) * (2.0 / max(W - 1, 1))
        shift_y = (py[:, 1:] - py[:, :-1]) * (2.0 / max(H - 1, 1))

        rate = (da - (2.0 / self.smudge_length) * da * da).clamp(0.0, 1.0)
        strength = (w_smudge.view(B, 1) * rate).view(B, n_dabs, 1, 1)

        canvas = premult
        for i in range(1, n_dabs):
            offset = torch.stack([shift_x[:, i - 1], shift_y[:, i - 1]], dim=-1).view(B, 1, 1, 2)
            dragged = F.grid_sample(
                canvas, self.base_grid - offset, mode="nearest", padding_mode="zeros", align_corners=True
            )
            d_sq = (self.grid_x - dx[:, i].view(B, 1, 1, 1)) ** 2 + (self.grid_y - dy[:, i].view(B, 1, 1, 1)) ** 2
            dab = self._brush_profile(d_sq, dr[:, i].view(B, 1, 1, 1), hardness.view(B, 1, 1, 1), edge_sharpness)
            canvas = canvas + dab * strength[:, i : i + 1] * (dragged - canvas)

        return canvas

    def _gaussian_blur_rgba(self, x_rgba: torch.Tensor) -> torch.Tensor:
        """Applies fast separable Gaussian smoothing across 4 premultiplied RGBA channels."""
        p = self._blur_pad
        out = F.pad(x_rgba, (p, p, p, p), mode="replicate")
        out = F.conv2d(out, self.blur_kh, groups=4)
        out = F.conv2d(out, self.blur_kv, groups=4)
        return out

    def create_blank_paint_layer(
        self, batch_size: int = 1, H: Optional[int] = None, W: Optional[int] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Returns a completely transparent RGBA paint layer of shape (B, 4, H, W) initialized to zeros."""
        H = H or self.canvas_size
        W = W or self.canvas_size
        device = device or self.bg_color.device
        return torch.zeros((batch_size, 4, H, W), dtype=torch.float32, device=device)

    def create_background(
        self,
        batch_size: int = 1,
        H: Optional[int] = None,
        W: Optional[int] = None,
        device: Optional[torch.device] = None,
        textured: bool = True,
    ) -> torch.Tensor:
        """Returns the static RGB background image `(B, 3, H, W)` (textured canvas weave or flat bg_color)."""
        H = H or self.canvas_size
        W = W or self.canvas_size
        device = device or self.bg_color.device
        self._update_spatial_buffers(H, W, device)
        if textured and self.bg_texture_strength > 0:
            return self.default_bg.expand(batch_size, 3, H, W).clone()
        return self.bg_color.to(device).expand(batch_size, 3, H, W).clone()

    def composite(
        self, paint_layer: torch.Tensor, background: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Composites the transparent RGBA `paint_layer` `(B, 4, H, W)` over `background` `(B, 3, H, W)`."""
        B, C, H, W = paint_layer.shape
        assert C == 4, f"Expected 4-channel RGBA paint_layer, got {C} channels."
        if background is None:
            self._update_spatial_buffers(H, W, paint_layer.device)
            background = self.default_bg.expand(B, 3, H, W)
        paint_rgb = paint_layer[:, :3]
        paint_a = paint_layer[:, 3:4]
        return ((1.0 - paint_a) * background + paint_a * paint_rgb).clamp(0.0, 1.0)

    def decode_raw_actions(
        self,
        raw_cont: torch.Tensor,
        raw_tool_logits: Optional[torch.Tensor] = None,
        gumbel_tau: float = 1.0,
        gumbel_hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps unconstrained neural network outputs R^14 into bounded [0, 1] stroke parameters.

        In training mode, tools are sampled with Gumbel-Softmax. With `gumbel_hard=True` the forward
        pass uses a one-hot tool (matching inference) while gradients flow through the soft relaxation.
        """
        stroke_params = torch.sigmoid(raw_cont)
        if raw_tool_logits is None:
            shape = raw_cont.shape[:-1] + (self.NUM_TOOLS,)
            tool_weights = torch.zeros(shape, device=raw_cont.device, dtype=raw_cont.dtype)
            tool_weights[..., 0] = 1.0
        else:
            if self.training:
                tool_weights = F.gumbel_softmax(raw_tool_logits, tau=gumbel_tau, hard=gumbel_hard, dim=-1)
            else:
                idx = raw_tool_logits.argmax(dim=-1)
                tool_weights = F.one_hot(idx, num_classes=self.NUM_TOOLS).to(raw_cont.dtype)
        return stroke_params, tool_weights

    def forward_single_step(
        self,
        paint_layer: torch.Tensor,
        stroke_params: torch.Tensor,
        tool_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Applies a single stroke per batch item onto the transparent RGBA `paint_layer` `(B, 4, H, W)`.
        
        Args:
            paint_layer: (B, 4, H, W) RGBA tensor in [0, 1]
            stroke_params: (B, 14) tensor in [0, 1]:
                [x0, y0, x1, y1, x2, y2, r0, r2, a0, a2, hardness, R, G, B]
            tool_weights: (B, 5) weights over tools:
                [Paint, Dry/Chalk, Smudge, Blur/Blend, Eraser]
        """
        B, C, H, W = paint_layer.shape
        assert C == 4, f"Expected 4-channel RGBA paint_layer (B, 4, H, W), got shape {tuple(paint_layer.shape)}"
        assert stroke_params.shape[-1] == self.NUM_CONT_PARAMS, (
            f"Expected {self.NUM_CONT_PARAMS} continuous stroke params, got {stroke_params.shape[-1]}"
        )
        device = paint_layer.device
        self._update_spatial_buffers(H, W, device)

        if tool_weights is None:
            tool_weights = torch.zeros(B, self.NUM_TOOLS, device=device, dtype=paint_layer.dtype)
            tool_weights[:, 0] = 1.0

        # 1. Unpack normalized stroke parameters (14 params)
        p = stroke_params.view(B, self.NUM_CONT_PARAMS, 1, 1, 1)
        x0, y0 = p[:, 0], p[:, 1]
        x1, y1 = p[:, 2], p[:, 3]
        x2, y2 = p[:, 4], p[:, 5]

        r0 = self.min_radius + p[:, 6] * (self.max_radius - self.min_radius)
        r2 = self.min_radius + p[:, 7] * (self.max_radius - self.min_radius)

        a0 = 0.05 + 0.95 * p[:, 8]
        a2 = 0.05 + 0.95 * p[:, 9]

        hardness = p[:, 10]  # Continuous edge hardness in [0, 1] across all tools
        stroke_rgb = stroke_params[:, 11:14].view(B, 3, 1, 1)

        # 2. Sample Quadratic Bezier curve nodes B(t)
        t = self.t  # (1, S, 1, 1)
        omt = 1.0 - t
        bx = (omt ** 2) * x0 + 2.0 * omt * t * x1 + (t ** 2) * x2  # (B, S, 1, 1)
        by = (omt ** 2) * y0 + 2.0 * omt * t * y1 + (t ** 2) * y2  # (B, S, 1, 1)

        rt = omt * r0 + t * r2  # (B, S, 1, 1)
        at = omt * a0 + t * a2  # (B, S, 1, 1)

        # 3. Continuous Line-Segment (Capsule) Distance Field
        ax, ay = bx[:, :-1], by[:, :-1]          # (B, S-1, 1, 1)
        bx_end, by_end = bx[:, 1:], by[:, 1:]    # (B, S-1, 1, 1)
        abx = bx_end - ax
        aby = by_end - ay
        ab_len_sq = abx * abx + aby * aby + 1e-8

        apx = self.grid_x - ax  # (B, S-1, H, W)
        apy = self.grid_y - ay  # (B, S-1, H, W)
        h = ((apx * abx + apy * aby) / ab_len_sq).clamp(0.0, 1.0)  # (B, S-1, H, W)

        closest_x = ax + h * abx
        closest_y = ay + h * aby
        dist_sq = (self.grid_x - closest_x) ** 2 + (self.grid_y - closest_y) ** 2

        r_seg = rt[:, :-1] + h * (rt[:, 1:] - rt[:, :-1])  # (B, S-1, H, W)
        a_seg = at[:, :-1] + h * (at[:, 1:] - at[:, :-1])  # (B, S-1, H, W)

        # 4. Unified Continuous Hardness Brush Profile
        # Hard edge falloff (10%-90% alpha) spans `hard_edge_px` output pixels at any resolution.
        # sigmoid(k * x) rises from 0.1 to 0.9 over 2*ln(9)/k; one pixel is 1/(N-1) in grid units.
        edge_sharpness = 2.0 * math.log(9.0) * (max(H, W) - 1) / self.hard_edge_px

        seg_profile = self._brush_profile(dist_sq, r_seg, hardness, edge_sharpness) * a_seg
        brush_mask, _ = seg_profile.max(dim=1, keepdim=True)  # (B, 1, H, W)

        # 5. Tool weights unpacking: [Paint, Dry/Chalk, Smudge, Blur/Blend, Eraser]
        w = tool_weights.view(B, self.NUM_TOOLS, 1, 1, 1)
        w_paint, w_dry, w_smudge, w_blur, w_eraser = w[:, 0], w[:, 1], w[:, 2], w[:, 3], w[:, 4]

        paint_mask = (w_paint * brush_mask + w_dry * (brush_mask * self.paper_grain)).clamp(0.0, 1.0)
        blur_mask = (w_blur * brush_mask).clamp(0.0, 1.0)
        eraser_mask = (w_eraser * brush_mask).clamp(0.0, 1.0)

        # 6. STEP A: Paint Deposition via Porter-Duff "Source Over" on RGBA
        c_rgb = paint_layer[:, :3]
        c_a = paint_layer[:, 3:4]

        a_after_paint = paint_mask + c_a * (1.0 - paint_mask)
        rgb_num = paint_mask * stroke_rgb + (1.0 - paint_mask) * c_a * c_rgb
        rgb_after_paint = torch.where(
            a_after_paint > 1e-5,
            rgb_num / (a_after_paint + 1e-6),
            c_rgb,
        )

        # 7. STEP B: Smudge / Push — dab-sequential paint transport in premultiplied RGBA
        premult_rgba = torch.cat([rgb_after_paint * a_after_paint, a_after_paint], dim=1)
        if tool_weights.requires_grad or bool((w_smudge > 0).any()):
            premult_rgba = self._smudge(premult_rgba, bx, by, rt, at, hardness, w_smudge, edge_sharpness)
        a_after_smudge = premult_rgba[:, 3:4].clamp(0.0, 1.0)
        rgb_after_smudge = torch.where(
            a_after_smudge > 1e-5,
            (premult_rgba[:, :3] / (a_after_smudge + 1e-6)).clamp(0.0, 1.0),
            rgb_after_paint,
        )

        # 8. STEP C: In-Place Isotropic Blur / Blend Tool
        post_smudge_premult = torch.cat([rgb_after_smudge * a_after_smudge, a_after_smudge], dim=1)
        blurred_post_smudge = self._gaussian_blur_rgba(post_smudge_premult)
        blended_blur_premult = (1.0 - blur_mask) * post_smudge_premult + blur_mask * blurred_post_smudge

        a_after_blur = blended_blur_premult[:, 3:4].clamp(0.0, 1.0)
        rgb_after_blur = torch.where(
            a_after_blur > 1e-5,
            (blended_blur_premult[:, :3] / (a_after_blur + 1e-6)).clamp(0.0, 1.0),
            rgb_after_smudge,
        )

        # 9. STEP D: True Subtractive Eraser (supports both soft and hard erasing via `hardness`)
        a_final = a_after_blur * (1.0 - eraser_mask)
        rgb_final = torch.where(a_final > 1e-5, rgb_after_blur, torch.zeros_like(rgb_after_blur))

        return torch.cat([rgb_final, a_final], dim=1).clamp(0.0, 1.0)

    def forward(
        self,
        paint_layer: torch.Tensor,
        stroke_sequence: torch.Tensor,
        tool_weights_sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unrolls K sequential strokes onto the RGBA `paint_layer` `(B, 4, H, W)`."""
        if stroke_sequence.dim() == 2:
            return self.forward_single_step(paint_layer, stroke_sequence, tool_weights_sequence)

        B, K, _ = stroke_sequence.shape
        curr_layer = paint_layer
        for k in range(K):
            tw_k = tool_weights_sequence[:, k] if tool_weights_sequence is not None else None
            curr_layer = self.forward_single_step(curr_layer, stroke_sequence[:, k], tw_k)
        return curr_layer


def reverse_strokes(stroke_params: torch.Tensor) -> torch.Tensor:
    """
    Returns the reversed-direction encoding of `(..., 14)` strokes: endpoints swapped, radius and
    opacity tapers swapped. For every tool except Smudge the reversed stroke renders identically, so
    parameter-space losses should take the minimum over both orderings.
    """
    idx = [4, 5, 2, 3, 0, 1, 7, 6, 9, 8, 10, 11, 12, 13]
    return stroke_params[..., idx]


def pad_to_square_with_mask(
    img: torch.Tensor, target_size: int = 256, pad_value: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int]]:
    B, C, H, W = img.shape
    scale = target_size / float(max(H, W))
    new_H, new_W = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
    resized = F.interpolate(img, size=(new_H, new_W), mode="bilinear", align_corners=False)

    pad_top = (target_size - new_H) // 2
    pad_bottom = target_size - new_H - pad_top
    pad_left = (target_size - new_W) // 2
    pad_right = target_size - new_W - pad_left

    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=pad_value)
    valid_mask = torch.zeros((B, 1, target_size, target_size), device=img.device, dtype=img.dtype)
    valid_mask[:, :, pad_top : pad_top + new_H, pad_left : pad_left + new_W] = 1.0

    return padded, valid_mask, (pad_top, pad_bottom, pad_left, pad_right)


def masked_reconstruction_loss(
    composited_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    diff = composited_rgb - target_rgb
    pixel_loss = 0.5 * diff.abs() + 0.5 * (diff ** 2)
    if valid_mask is not None:
        return (pixel_loss * valid_mask).sum() / (valid_mask.sum() * composited_rgb.shape[1] + 1e-6)
    return pixel_loss.mean()