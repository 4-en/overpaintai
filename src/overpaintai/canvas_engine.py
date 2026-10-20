import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _brush_profile(
    dist_sq: torch.Tensor, radius: torch.Tensor, hardness: torch.Tensor, edge_sharpness: float
) -> torch.Tensor:
    """Blends a soft Gaussian falloff (hardness 0) with a crisp disc edge (hardness 1)."""
    hard = torch.sigmoid(edge_sharpness * (radius - torch.sqrt(dist_sq + 1e-8)))
    soft = torch.exp(-2.0 * dist_sq / (radius * radius + 1e-6))
    return (1.0 - hardness) * soft + hardness * hard


def _dab_masks(
    col_x: torch.Tensor,
    row_y: torch.Tensor,
    col_in: torch.Tensor,
    row_in: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    hardness: torch.Tensor,
    strength: torch.Tensor,
    edge_sharpness: float,
) -> torch.Tensor:
    """
    Smudge transfer masks of all dabs on their windows, `(B, n, Wn * Wn)`: brush profile centered at
    (`x`, `y`) `(B, n)` times `strength`. `col_x` / `row_y` `(B, n, Wn)` are the windows' normalized pixel
    coordinates; `col_in` / `row_in` zero out pixels outside the image.
    """
    B, n, Wn = col_x.shape
    d_sq = (col_x.unsqueeze(2) - x.view(B, n, 1, 1)) ** 2 + (row_y.unsqueeze(3) - y.view(B, n, 1, 1)) ** 2
    m = _brush_profile(d_sq, r.view(B, n, 1, 1), hardness.view(B, 1, 1, 1), edge_sharpness)
    m = m * (strength.view(B, n, 1, 1) * row_in.unsqueeze(3) * col_in.unsqueeze(2))
    return m.view(B, n, Wn * Wn)


# Fusing the mask math into a few kernels matters: it runs over every dab window
_dab_masks_compiled = torch.compile(_dab_masks, dynamic=True)


def _segment_profile(
    seg: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor, hardness: torch.Tensor, edge_sharpness: float
) -> torch.Tensor:
    """
    Profile of capsule segments `seg` `(B, N, H, W, 8)` (or broadcastable), with the last dimension
    holding [ax, ay, abx, aby, r_start, r_end, a_start, a_end], evaluated on the pixel grid.
    """
    ax, ay, abx, aby, r0, r1, a0, a1 = seg.unbind(dim=-1)
    apx = grid_x - ax
    apy = grid_y - ay
    h = ((apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-8)).clamp(0.0, 1.0)
    dist_sq = (apx - h * abx) ** 2 + (apy - h * aby) ** 2
    r_seg = r0 + h * (r1 - r0)
    a_seg = a0 + h * (a1 - a0)
    return _brush_profile(dist_sq, r_seg, hardness, edge_sharpness) * a_seg


def _best_segment(
    seg: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor, hardness: torch.Tensor, edge_sharpness: float
):
    """Max over all segments `(B, N, 8)` of their profiles, and the winning index, each `(B, 1, H, W)`."""
    B, N, _ = seg.shape
    prof = _segment_profile(seg.view(B, N, 1, 1, 8), grid_x, grid_y, hardness, edge_sharpness)
    return prof.max(dim=1, keepdim=True)


# Fusing profile + max avoids ~20 full-size intermediates per segment and stroke, which dominate rendering
# cost; used for the no-grad winner search on CUDA
_best_segment_compiled = torch.compile(_best_segment, dynamic=True)


def _window_index(base: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Flat indices `(B, 4, n)` of a window: per-item start `base` `(B,)` plus shared offsets `offs` `(n,)`."""
    idx = base.view(-1, 1, 1) + offs.view(1, 1, -1)
    return idx.expand(base.shape[0], 4, offs.shape[0])


def _run_dabs(canvas: torch.Tensor, masks: torch.Tensor, base: torch.Tensor, offs: torch.Tensor, diffs=None):
    """
    Applies dabs 1..n-1 in place to the flat padded canvas `(B, 4, L)`: each dab's window is blended
    towards the window at the previous dab by its mask. Optionally records `dragged - cur` per dab.
    """
    for i in range(1, masks.shape[1]):
        cur_idx = _window_index(base[:, i], offs)
        cur = canvas.gather(2, cur_idx)
        diff = canvas.gather(2, _window_index(base[:, i - 1], offs)) - cur
        if diffs is not None:
            diffs[:, i] = diff
        canvas.scatter_(2, cur_idx, cur + masks[:, i : i + 1] * diff)
    return canvas


class _SmudgeDabs(torch.autograd.Function):
    """
    The whole smudge dab loop as one autograd node. Saving per-dab autograd state for window-sized ops
    is dominated by full-canvas gradient bookkeeping, so the backward is written by hand: it walks the
    dabs in reverse, updating a single gradient canvas in place window by window, then maps the mask
    gradients back to the dab parameters with one vectorized vjp.
    """

    @staticmethod
    def forward(ctx, canvas, x, y, r, strength, hardness, base, offs, col_x, row_y, col_in, row_in, edge, mask_fn):
        with torch.no_grad():
            masks = mask_fn(col_x, row_y, col_in, row_in, x, y, r, hardness, strength, edge)
            diffs = canvas.new_empty((canvas.shape[0], masks.shape[1], 4, offs.shape[0]))
            out = _run_dabs(canvas.clone(), masks, base, offs, diffs)
        ctx.save_for_backward(x, y, r, strength, hardness, base, offs, col_x, row_y, col_in, row_in, masks, diffs)
        ctx.edge, ctx.mask_fn = edge, mask_fn
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, y, r, strength, hardness, base, offs, col_x, row_y, col_in, row_in, masks, diffs = ctx.saved_tensors
        G = grad_out.clone()
        g_masks = torch.zeros_like(masks)
        for i in range(masks.shape[1] - 1, 0, -1):
            cur_idx = _window_index(base[:, i], offs)
            g_new = G.gather(2, cur_idx)
            m = masks[:, i : i + 1]
            g_masks[:, i] = (g_new * diffs[:, i]).sum(dim=1)
            G.scatter_(2, cur_idx, g_new * (1.0 - m))  # window was overwritten: its grad is only via `cur`
            G.scatter_add_(2, _window_index(base[:, i - 1], offs), g_new * m)  # via `dragged`

        params = [t.detach().requires_grad_(t.requires_grad) for t in (x, y, r, strength, hardness)]
        grads = [None] * 5
        wanted = [p for p in params if p.requires_grad]
        if wanted:
            with torch.enable_grad():
                m_re = ctx.mask_fn(col_x, row_y, col_in, row_in, *params[:3], params[4], params[3], ctx.edge)
                got = iter(torch.autograd.grad(m_re, wanted, g_masks))
            grads = [next(got) if p.requires_grad else None for p in params]
        return (G, *grads) + (None,) * 8


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
        max_radius: float = 0.75,
        hard_edge_px: float = 1.0,
        blur_ratio: float = 0.035,
        smudge_spacing: float = 0.25,
        smudge_length: float = 12.0,
        smudge_max_dabs: int = 128,
        compile_smudge: bool = True,
        compile_masks: bool = True,
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
        self.compile_smudge = compile_smudge
        self.compile_masks = compile_masks
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

    def _stroke_mask(
        self,
        bx: torch.Tensor,
        by: torch.Tensor,
        rt: torch.Tensor,
        at: torch.Tensor,
        hardness: torch.Tensor,
        edge_sharpness: float,
        chunk: int = 8,
    ) -> torch.Tensor:
        """
        Brush mask `(B, 1, H, W)`: the max of the capsule profiles over all curve segments.

        The max only passes gradient to the winning segment, so the winner per pixel is found without
        autograd (in chunks, bounding peak memory), and only that segment's profile is recomputed
        differentiably. Values and gradients match materializing all `(B, S-1, H, W)` segment profiles,
        at a fraction of the saved memory.
        """
        B = bx.shape[0]
        _, _, H, W = self.grid_x.shape
        seg = torch.stack(
            [bx[:, :-1], by[:, :-1], bx[:, 1:] - bx[:, :-1], by[:, 1:] - by[:, :-1],
             rt[:, :-1], rt[:, 1:], at[:, :-1], at[:, 1:]],
            dim=-1,
        ).view(B, -1, 8)  # (B, S-1, 8)
        n_seg = seg.shape[1]

        with torch.no_grad():
            if self.compile_masks and seg.is_cuda:
                best_val, best_idx = _best_segment_compiled(seg, self.grid_x, self.grid_y, hardness, edge_sharpness)
            else:
                best_val = torch.full((B, 1, H, W), -1.0, device=seg.device, dtype=seg.dtype)
                best_idx = torch.zeros((B, 1, H, W), device=seg.device, dtype=torch.long)
                for c in range(0, n_seg, chunk):
                    part = seg[:, c : c + chunk].view(B, -1, 1, 1, 8)
                    val, idx = _segment_profile(part, self.grid_x, self.grid_y, hardness, edge_sharpness).max(
                        dim=1, keepdim=True)
                    better = val > best_val
                    best_val = torch.where(better, val, best_val)
                    best_idx = torch.where(better, idx + c, best_idx)

        if not (torch.is_grad_enabled() and (seg.requires_grad or hardness.requires_grad)):
            return best_val
        winner = torch.gather(seg, 1, best_idx.view(B, H * W, 1).expand(B, H * W, 8))  # (B, H*W, 8)
        return _segment_profile(winner.view(B, 1, H, W, 8), self.grid_x, self.grid_y, hardness, edge_sharpness)

    def _smudge(
        self,
        premult: torch.Tensor,
        ctrl: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
        hardness: torch.Tensor,
        w_smudge: torch.Tensor,
        edge_sharpness: float,
    ) -> torch.Tensor:
        """
        Physically-motivated smear of premultiplied RGBA `(B, 4, H, W)` along the stroke.

        The curve is split into closely spaced dabs. At each dab, the paint under the brush is
        replaced (at the smear rate times the dab profile) by the paint one step behind, translated
        forward by the dab-to-dab motion in whole pixels (exact copies, so nothing is blurred). Paint
        inside the footprint therefore travels with the brush:
          - starting on bare canvas, the brush carries nothing and digs an indent into paint it enters,
            gradually loading up with that paint and filling the indent as it goes,
          - starting on paint, it drags that paint along and leaves it as a fading trail.

        Carried paint drifts backwards relative to the brush by (1 - rate) per distance travelled and is
        left behind once it crosses the footprint (2 radii), so it is dragged about 2 / (1 - rate) radii.
        The rate is `opacity - (2 / smudge_length) * opacity^2`, which caps the drag distance at full
        opacity to `smudge_length` radii while barely changing low opacities. Blurring is the Blur /
        Blend tool's job.

        Args:
            ctrl: (B, 10) decoded [x0, y0, x1, y1, x2, y2, r0, r2, a0, a2] (radii in canvas units)
            bx, by: (B, S, 1, 1) curve samples, used to measure the stroke length
            hardness, w_smudge: (B,)
        """
        H, W = premult.shape[-2:]
        N = max(H, W) - 1

        with torch.no_grad():
            # Per-stroke dab count: spacing `smudge_spacing` * the stroke's smallest radius
            length = torch.sqrt((bx[:, 1:] - bx[:, :-1]) ** 2 + (by[:, 1:] - by[:, :-1]) ** 2).sum(dim=1).flatten()
            r_min = torch.minimum(ctrl[:, 6], ctrl[:, 7])
            n_dabs = (length / (self.smudge_spacing * r_min)).ceil().long() + 1
            n_dabs = n_dabs.clamp(2, self.smudge_max_dabs)

            # Per-stroke window half-size R (pixels): profiles are truncated where they fall below ~1e-4
            r_max = torch.maximum(ctrl[:, 6], ctrl[:, 7])
            hard_ext = r_max + 9.2 / edge_sharpness  # sigmoid < 1e-4 beyond this
            soft_ext = torch.maximum(2.15 * r_max, hard_ext)  # exp(-2 d^2 / r^2) < 1e-4 beyond 2.15 r
            ext = torch.where(hardness < 1.0 - 1e-6, soft_ext, hard_ext)
            R_px = (ext * N).ceil().long() + 1

            # Bucket strokes by window size (sqrt(2) steps) so small brushes don't pay for large ones
            bucket = torch.ceil(2.0 * torch.log2(R_px.float())).long()

        groups = self._smudge_groups(bucket.tolist(), n_dabs.tolist(), R_px.tolist())
        results = [
            self._smudge_windows(
                premult[g], ctrl[g], n_dabs[g], int(n_dabs[g].max()), int(R_px[g].max()),
                hardness[g], w_smudge[g], edge_sharpness,
            )
            for g in groups
        ]
        if len(groups) == 1:
            return results[0]
        inverse = torch.argsort(torch.cat(groups))
        return torch.cat(results, dim=0)[inverse]

    # Measured cost of one dab (forward + backward): a fixed launch overhead plus a per-window-pixel cost
    _DAB_OVERHEAD_PX = 120_000

    def _smudge_groups(self, bucket: list, n_dabs: list, R_px: list) -> list:
        """
        Partitions the batch into groups that are smudged together (sharing a dab count and window size).
        Contiguous runs of size buckets are merged when that is cheaper under the cost model
        `n_max * (overhead + sum_items (2R_max + 1)^2)`; found exactly by dynamic programming.
        """
        device = self.grid_x.device
        keys = sorted(set(bucket))
        if len(keys) == 1:
            return [torch.arange(len(bucket), device=device)]
        members = [[i for i, b in enumerate(bucket) if b == key] for key in keys]

        def cost(lo: int, hi: int) -> float:  # buckets lo..hi-1 as one group
            items = [i for m in members[lo:hi] for i in m]
            wn = 2 * max(R_px[i] for i in items) + 1
            return max(n_dabs[i] for i in items) * (self._DAB_OVERHEAD_PX + len(items) * wn * wn)

        k = len(keys)
        best = [0.0] + [float("inf")] * k
        split = [0] * (k + 1)
        for hi in range(1, k + 1):
            for lo in range(hi):
                c = best[lo] + cost(lo, hi)
                if c < best[hi]:
                    best[hi], split[hi] = c, lo
        groups, hi = [], k
        while hi > 0:
            lo = split[hi]
            groups.append(torch.tensor([i for m in members[lo:hi] for i in m], device=device))
            hi = lo
        if len(groups) == 1:  # everything merged: keep batch order (no reordering to undo)
            return [torch.arange(len(bucket), device=device)]
        return groups[::-1]

    def _smudge_windows(
        self,
        premult: torch.Tensor,
        ctrl: torch.Tensor,
        n_dabs: torch.Tensor,
        n_max: int,
        R: int,
        hardness: torch.Tensor,
        w_smudge: torch.Tensor,
        edge_sharpness: float,
    ) -> torch.Tensor:
        """
        Runs the smudge dabs for one bucket of strokes. Each dab only touches a (2R+1)^2 window around its
        whole-pixel center, so the canvas is updated window by window via exact integer gather/scatter;
        the paint "one step behind" is simply the window at the previous dab's center. Strokes with fewer
        than `n_max` dabs repeat their final dab, which is a no-op (zero motion).
        """
        B, _, H, W = premult.shape
        x0, y0, x1, y1, x2, y2, r0, r2, a0, a2 = ctrl.unbind(dim=-1)

        # Dab positions evaluated directly on the quadratic Bezier, per-stroke spacing
        j = torch.arange(n_max, device=premult.device, dtype=premult.dtype)
        t = (j.view(1, -1) / (n_dabs.view(-1, 1) - 1).to(premult.dtype)).clamp(max=1.0)  # (B, n_max)
        omt = 1.0 - t
        dx = omt * omt * x0[:, None] + 2.0 * omt * t * x1[:, None] + t * t * x2[:, None]
        dy = omt * omt * y0[:, None] + 2.0 * omt * t * y1[:, None] + t * t * y2[:, None]
        dr = omt * r0[:, None] + t * r2[:, None]
        da = omt * a0[:, None] + t * a2[:, None]

        rate = (da - (2.0 / self.smudge_length) * da * da).clamp(0.0, 1.0)
        strength = w_smudge.view(B, 1) * rate  # (B, n_max)

        Wn = 2 * R + 1
        Hp, Wp = H + 2 * R, W + 2 * R
        with torch.no_grad():
            # Whole-pixel dab centers; in the R-padded canvas a window's top-left corner is (cy, cx)
            cx = torch.round(dx * (W - 1)).long()  # (B, n_max)
            cy = torch.round(dy * (H - 1)).long()
            base = cy * Wp + cx
            k = torch.arange(Wn, device=premult.device)
            offs = (k.view(-1, 1) * Wp + k.view(1, -1)).flatten()  # (Wn * Wn,)

            # Normalized coordinates of each window's columns / rows, and whether they lie in the image
            cols = cx.unsqueeze(-1) - R + k  # (B, n_max, Wn)
            rows = cy.unsqueeze(-1) - R + k
            col_x = cols.to(premult.dtype) / (W - 1)
            row_y = rows.to(premult.dtype) / (H - 1)
            col_in = ((cols >= 0) & (cols < W)).to(premult.dtype)
            row_in = ((rows >= 0) & (rows < H)).to(premult.dtype)

        mask_fn = _dab_masks_compiled if (self.compile_smudge and premult.is_cuda) else _dab_masks
        canvas = F.pad(premult, (R, R, R, R)).reshape(B, 4, Hp * Wp)
        dab_params = (dx, dy, dr, strength, hardness)
        if torch.is_grad_enabled() and any(t.requires_grad for t in (canvas, *dab_params)):
            canvas = _SmudgeDabs.apply(
                canvas, *dab_params, base, offs, col_x, row_y, col_in, row_in, edge_sharpness, mask_fn
            )
        else:
            with torch.no_grad():
                masks = mask_fn(col_x, row_y, col_in, row_in, dx, dy, dr, hardness, strength, edge_sharpness)
                canvas = _run_dabs(canvas, masks, base, offs)

        return canvas.view(B, 4, Hp, Wp)[:, :, R : R + H, R : R + W]

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

        # 3-4. Continuous capsule distance field along the curve, with the unified hardness profile.
        # Hard edge falloff (10%-90% alpha) spans `hard_edge_px` output pixels at any resolution.
        # sigmoid(k * x) rises from 0.1 to 0.9 over 2*ln(9)/k; one pixel is 1/(N-1) in grid units.
        edge_sharpness = 2.0 * math.log(9.0) * (max(H, W) - 1) / self.hard_edge_px
        brush_mask = self._stroke_mask(bx, by, rt, at, hardness, edge_sharpness)

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

        # Tools whose weight is zero everywhere are skipped when no gradient needs them (e.g. paint-only
        # inference): the result is the same up to float rounding, and each skipped step saves several
        # full-canvas passes per stroke.
        need_all = tool_weights.requires_grad

        # 7. STEP B: Smudge / Push — dab-sequential paint transport in premultiplied RGBA
        if need_all or bool((w_smudge > 0).any()):
            premult_rgba = torch.cat([rgb_after_paint * a_after_paint, a_after_paint], dim=1)
            ctrl = torch.cat([p[:, 0:6].view(B, 6)] + [v.view(B, 1) for v in (r0, r2, a0, a2)], dim=1)
            premult_rgba = self._smudge(
                premult_rgba, ctrl, bx, by, hardness.view(B), w_smudge.view(B), edge_sharpness
            )
            a_after_smudge = premult_rgba[:, 3:4].clamp(0.0, 1.0)
            rgb_after_smudge = torch.where(
                a_after_smudge > 1e-5,
                (premult_rgba[:, :3] / (a_after_smudge + 1e-6)).clamp(0.0, 1.0),
                rgb_after_paint,
            )
        else:
            a_after_smudge, rgb_after_smudge = a_after_paint, rgb_after_paint

        # 8. STEP C: In-Place Isotropic Blur / Blend Tool
        if need_all or bool((w_blur > 0).any()):
            post_smudge_premult = torch.cat([rgb_after_smudge * a_after_smudge, a_after_smudge], dim=1)
            blurred_post_smudge = self._gaussian_blur_rgba(post_smudge_premult)
            blended_blur_premult = (1.0 - blur_mask) * post_smudge_premult + blur_mask * blurred_post_smudge

            a_after_blur = blended_blur_premult[:, 3:4].clamp(0.0, 1.0)
            rgb_after_blur = torch.where(
                a_after_blur > 1e-5,
                (blended_blur_premult[:, :3] / (a_after_blur + 1e-6)).clamp(0.0, 1.0),
                rgb_after_smudge,
            )
        else:
            a_after_blur, rgb_after_blur = a_after_smudge, rgb_after_smudge

        # 9. STEP D: True Subtractive Eraser (supports both soft and hard erasing via `hardness`)
        a_final = a_after_blur * (1.0 - eraser_mask)
        rgb_final = torch.where(a_final > 1e-5, rgb_after_blur, torch.zeros_like(rgb_after_blur))

        return torch.cat([rgb_final, a_final], dim=1).clamp(0.0, 1.0)

    def forward(
        self,
        paint_layer: torch.Tensor,
        stroke_sequence: torch.Tensor,
        tool_weights_sequence: Optional[torch.Tensor] = None,
        checkpoint_strokes: bool = False,
    ) -> torch.Tensor:
        """
        Unrolls K sequential strokes onto the RGBA `paint_layer` `(B, 4, H, W)`.

        With `checkpoint_strokes=True` (and autograd enabled), only the paint layer between strokes is
        kept for backward and each stroke is recomputed during backward: memory stops growing with the
        per-stroke intermediates, at the cost of roughly one extra forward pass.
        """
        if stroke_sequence.dim() == 2:
            return self.forward_single_step(paint_layer, stroke_sequence, tool_weights_sequence)

        B, K, _ = stroke_sequence.shape
        use_ckpt = checkpoint_strokes and torch.is_grad_enabled()
        curr_layer = paint_layer
        for k in range(K):
            tw_k = tool_weights_sequence[:, k] if tool_weights_sequence is not None else None
            if use_ckpt:
                curr_layer = checkpoint(
                    self.forward_single_step, curr_layer, stroke_sequence[:, k], tw_k, use_reentrant=False
                )
            else:
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