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
    """

    NUM_CONT_PARAMS = 13  # x0, y0, x1, y1, x2, y2, r0, r2, a0, a2, R, G, B
    NUM_TOOLS = 5         # Hard, Soft, Dry/Chalk, Smudge, Eraser

    def __init__(
        self,
        canvas_size: int = 256,
        num_stamps: int = 32,
        min_radius: float = 0.005,
        max_radius: float = 0.25,
        hard_edge_sharpness: float = 300.0,
        smudge_pull_strength: float = 0.55,
        smudge_blur_ratio: float = 0.045,
        smudge_taps: int = 8,
        bg_color: Tuple[float, float, float] = (0.97, 0.96, 0.93),
        bg_texture_strength: float = 0.04,
    ):
        super().__init__()
        self.canvas_size = canvas_size
        self.num_stamps = max(4, num_stamps)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.hard_edge_sharpness = hard_edge_sharpness
        self.smudge_pull_strength = smudge_pull_strength
        self.smudge_blur_ratio = smudge_blur_ratio
        self.smudge_taps = smudge_taps
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

        # Separable 1D Gaussian blur kernel for 4-channel premultiplied RGBA smudge diffusion
        k_size = max(3, int(round(max(H, W) * self.smudge_blur_ratio)) | 1)
        sigma = k_size / 3.0
        coords = torch.arange(k_size, device=device, dtype=torch.float32) - (k_size - 1) / 2.0
        g1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        g1d = g1d / g1d.sum()
        self.register_buffer("blur_kh", g1d.view(1, 1, 1, k_size).expand(4, 1, 1, k_size).contiguous(), persistent=False)
        self.register_buffer("blur_kv", g1d.view(1, 1, k_size, 1).expand(4, 1, k_size, 1).contiguous(), persistent=False)
        self._blur_pad = k_size // 2

        self._cached_hw = (H, W)

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
        """
        Returns a completely transparent RGBA paint layer of shape (B, 4, H, W) initialized to zeros.
        """
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
        """
        Returns the static RGB background image `(B, 3, H, W)` (either textured canvas weave or flat bg_color).
        """
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
        """
        Composites the transparent RGBA `paint_layer` `(B, 4, H, W)` over the static `background` `(B, 3, H, W)`
        to produce the final visible RGB image `(B, 3, H, W)`.
        """
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps unconstrained neural network outputs R^13 into bounded [0, 1] stroke parameters.
        Also returns tool weights (soft temperature-scaled Softmax during training, hard one-hot during eval).
        """
        stroke_params = torch.sigmoid(raw_cont)
        if raw_tool_logits is None:
            shape = raw_cont.shape[:-1] + (self.NUM_TOOLS,)
            tool_weights = torch.zeros(shape, device=raw_cont.device, dtype=raw_cont.dtype)
            tool_weights[..., 0] = 1.0
        else:
            if self.training:
                tool_weights = F.softmax(raw_tool_logits / gumbel_tau, dim=-1)
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
        """
        B, C, H, W = paint_layer.shape
        assert C == 4, f"Expected 4-channel RGBA paint_layer (B, 4, H, W), got shape {tuple(paint_layer.shape)}"
        device = paint_layer.device
        self._update_spatial_buffers(H, W, device)

        if tool_weights is None:
            tool_weights = torch.zeros(B, self.NUM_TOOLS, device=device, dtype=paint_layer.dtype)
            tool_weights[:, 0] = 1.0

        # 1. Unpack normalized stroke parameters
        p = stroke_params.view(B, 13, 1, 1, 1)
        x0, y0 = p[:, 0], p[:, 1]
        x1, y1 = p[:, 2], p[:, 3]
        x2, y2 = p[:, 4], p[:, 5]

        r0 = self.min_radius + p[:, 6] * (self.max_radius - self.min_radius)
        r2 = self.min_radius + p[:, 7] * (self.max_radius - self.min_radius)

        a0 = 0.05 + 0.95 * p[:, 8]
        a2 = 0.05 + 0.95 * p[:, 9]

        stroke_rgb = stroke_params[:, 10:13].view(B, 3, 1, 1)

        # 2. Sample Quadratic Bezier curve nodes B(t) and tangent B'(t)
        t = self.t  # (1, S, 1, 1)
        omt = 1.0 - t
        bx = (omt ** 2) * x0 + 2.0 * omt * t * x1 + (t ** 2) * x2  # (B, S, 1, 1)
        by = (omt ** 2) * y0 + 2.0 * omt * t * y1 + (t ** 2) * y2  # (B, S, 1, 1)

        rt = omt * r0 + t * r2  # (B, S, 1, 1)
        at = omt * a0 + t * a2  # (B, S, 1, 1)

        # 3. Continuous Line-Segment (Capsule) Distance Field (eliminates discrete circle beading!)
        # Connect each node i to i+1 into S-1 continuous segments
        ax, ay = bx[:, :-1], by[:, :-1]          # (B, S-1, 1, 1)
        bx_end, by_end = bx[:, 1:], by[:, 1:]    # (B, S-1, 1, 1)
        abx = bx_end - ax
        aby = by_end - ay
        ab_len_sq = abx * abx + aby * aby + 1e-8

        # Project pixel (grid_x, grid_y) onto each segment i: projection parameter h in [0, 1]
        apx = self.grid_x - ax  # (B, S-1, H, W)
        apy = self.grid_y - ay  # (B, S-1, H, W)
        h = ((apx * abx + apy * aby) / ab_len_sq).clamp(0.0, 1.0)  # (B, S-1, H, W)

        # Closest point on segment i to pixel (x, y)
        closest_x = ax + h * abx
        closest_y = ay + h * aby
        dist_sq = (self.grid_x - closest_x) ** 2 + (self.grid_y - closest_y) ** 2
        dist = torch.sqrt(dist_sq + 1e-8)

        # Linearly interpolate radius r and opacity a at exact projection fraction h along each segment
        r_seg = rt[:, :-1] + h * (rt[:, 1:] - rt[:, :-1])  # (B, S-1, H, W)
        a_seg = at[:, :-1] + h * (at[:, 1:] - at[:, :-1])  # (B, S-1, H, W)

        # 4. Continuous Segment Profiles for Hard, Soft, and Textured brushes
        res_scale = 1.0 if self.training else max(1.0, max(H, W) / float(self.canvas_size))
        edge_sharpness = self.hard_edge_sharpness * res_scale

        hard_seg = torch.sigmoid(edge_sharpness * (r_seg - dist)) * a_seg
        soft_seg = torch.exp(-2.0 * dist_sq / (r_seg * r_seg + 1e-6)) * a_seg

        # Taking max across continuous capsule segments produces a 100% smooth tube with zero beading
        mask_hard, _ = hard_seg.max(dim=1, keepdim=True)  # (B, 1, H, W)
        mask_soft, _ = soft_seg.max(dim=1, keepdim=True)  # (B, 1, H, W)
        mask_dry = mask_hard * self.paper_grain           # (B, 1, H, W)

        # 5. Tool weights unpacking
        w = tool_weights.view(B, self.NUM_TOOLS, 1, 1, 1)
        w_hard, w_soft, w_dry, w_smudge, w_eraser = w[:, 0], w[:, 1], w[:, 2], w[:, 3], w[:, 4]

        paint_mask = (w_hard * mask_hard + w_soft * mask_soft + w_dry * mask_dry).clamp(0.0, 1.0)
        smudge_mask = (w_smudge * mask_soft).clamp(0.0, 1.0)
        eraser_mask = (w_eraser * mask_hard).clamp(0.0, 1.0)

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

        # 7. STEP B: Premultiplied RGBA Smudge along segment tangents
        premult_rgba = torch.cat([rgb_after_paint * a_after_paint, a_after_paint], dim=1)  # (B, 4, H, W)
        blurred_premult = self._gaussian_blur_rgba(premult_rgba)

        # Segment tangent vectors scaled by number of segments so total curve velocity matches B'(t)
        seg_tx = abx * (self.num_stamps - 1)  # (B, S-1, 1, 1)
        seg_ty = aby * (self.num_stamps - 1)  # (B, S-1, 1, 1)
        seg_weights = soft_seg / (soft_seg.sum(dim=1, keepdim=True) + 1e-6)
        local_tx = (seg_weights * seg_tx).sum(dim=1, keepdim=True)
        local_ty = (seg_weights * seg_ty).sum(dim=1, keepdim=True)

        accum_smudge = torch.zeros_like(premult_rgba)
        total_tap_w = 0.0
        for tap in range(self.smudge_taps):
            frac = tap / float(max(1, self.smudge_taps - 1))
            ox = local_tx * (frac * self.smudge_pull_strength)
            oy = local_ty * (frac * self.smudge_pull_strength)
            offset_grid = torch.cat([ox.permute(0, 2, 3, 1), oy.permute(0, 2, 3, 1)], dim=-1)
            sample_grid = (self.base_grid - offset_grid).clamp(-1.0, 1.0)
            sampled = F.grid_sample(blurred_premult, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)
            w_tap = 1.0 - 0.4 * frac
            accum_smudge = accum_smudge + w_tap * sampled
            total_tap_w += w_tap

        smudged_premult = accum_smudge / total_tap_w
        carried_rgb_premult = smudged_premult[:, :3]
        carried_a = smudged_premult[:, 3:4].clamp(0.0, 1.0)

        # Effective smudge blend weight is gated by how much paint was actually picked up (`carried_a`)
        # so dragging from bare canvas (carried_a = 0) into wet paint never erases or dilutes existing paint!
        eff_smudge = smudge_mask * carried_a
        carried_rgb = carried_rgb_premult / (carried_a + 1e-5)

        # Weight existing pigment by its actual alpha `a_after_paint` so bare canvas (alpha=0, rgb=0)
        # never darkens carried paint wisps with black zeros!
        rem_a = a_after_paint * (1.0 - eff_smudge)
        rgb_after_smudge = (rem_a * rgb_after_paint + eff_smudge * carried_rgb) / (rem_a + eff_smudge + 1e-6)
        a_after_smudge = torch.maximum(a_after_paint, eff_smudge).clamp(0.0, 1.0)

        # 8. STEP C: True Eraser (reduces paint Alpha channel, revealing the background underneath)
        a_final = a_after_smudge * (1.0 - eraser_mask)
        rgb_final = torch.where(a_final > 1e-5, rgb_after_smudge, torch.zeros_like(rgb_after_smudge))

        return torch.cat([rgb_final, a_final], dim=1).clamp(0.0, 1.0)

    def forward(
        self,
        paint_layer: torch.Tensor,
        stroke_sequence: torch.Tensor,
        tool_weights_sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unrolls K sequential strokes onto the RGBA `paint_layer` `(B, 4, H, W)`.
        """
        if stroke_sequence.dim() == 2:
            return self.forward_single_step(paint_layer, stroke_sequence, tool_weights_sequence)

        B, K, _ = stroke_sequence.shape
        curr_layer = paint_layer
        for k in range(K):
            tw_k = tool_weights_sequence[:, k] if tool_weights_sequence is not None else None
            curr_layer = self.forward_single_step(curr_layer, stroke_sequence[:, k], tw_k)
        return curr_layer


def pad_to_square_with_mask(
    img: torch.Tensor, target_size: int = 256, pad_value: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int]]:
    """
    Resizes an arbitrary (B, 3, H, W) image so its longest side is `target_size`,
    pads the shorter side with `pad_value` to form a 1:1 square,
    and returns a binary `valid_mask` (1 inside artwork, 0 in padding).
    """
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
    """
    Computes L1 + L2 loss on the composited RGB canvas inside `valid_mask`.
    """
    diff = composited_rgb - target_rgb
    pixel_loss = 0.5 * diff.abs() + 0.5 * (diff ** 2)
    if valid_mask is not None:
        return (pixel_loss * valid_mask).sum() / (valid_mask.sum() * composited_rgb.shape[1] + 1e-6)
    return pixel_loss.mean()