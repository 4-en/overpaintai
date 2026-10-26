import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# An extra loss term: (pred_rgb, target_rgb, valid_mask or None) -> scalar
LossTerm = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], per_sample: bool = False) -> torch.Tensor:
    """Mean of `x` (over the `mask`ed pixels); with `per_sample`, one mean per batch element `(B,)`."""
    if per_sample:
        if mask is None:
            return x.flatten(1).mean(1)
        return (x * mask).flatten(1).sum(1) / (mask.flatten(1).sum(1) * x.shape[1] + 1e-6)
    if mask is None:
        return x.mean()
    return (x * mask).sum() / (mask.sum() * x.shape[1] + 1e-6)


def _pool(x: torch.Tensor, factor: int) -> torch.Tensor:
    return F.avg_pool2d(x, factor, ceil_mode=True) if factor > 1 else x


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    k = k / k.sum()
    C = x.shape[1]
    x = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


def image_gradients(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradients per channel, `(B, C, H, W)` -> `(B, 2C, H, W)` (x then y), scaled so a unit step is 1."""
    C = x.shape[1]
    k = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype) / 4.0
    kernel = torch.stack([k, k.t()]).unsqueeze(1).repeat(C, 1, 1, 1)  # (2C, 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, groups=C)


def local_sharpness(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    """Strongest gradient magnitude within `window` px of each pixel, `(B, C, H, W)` (per channel)."""
    g = image_gradients(x)
    C = x.shape[1]
    mag = torch.sqrt(g[:, :C] ** 2 + g[:, C:] ** 2 + 1e-8)
    return F.max_pool2d(mag, window, stride=1, padding=window // 2)


def sharpness_pyramid(target: torch.Tensor, levels: int = 6) -> torch.Tensor:
    """
    Strongest lightness gradient within 3, 5, 9, 17, ... px (`2**(j+1) + 1` for level j) of each pixel,
    `(B, levels, H, W)`: how sharp the target is within reach of a stroke of radius ~`2**j` px.
    """
    g = image_gradients(target.mean(1, keepdim=True))
    mag = torch.sqrt(g[:, :1] ** 2 + g[:, 1:] ** 2 + 1e-8)
    return torch.cat([F.max_pool2d(mag, 2 ** (j + 1) + 1, stride=1, padding=2**j) for j in range(levels)], 1)


def stroke_edge_sharpness(pyramid: torch.Tensor, points: torch.Tensor, radii: torch.Tensor,
                          ref: float = 0.3) -> torch.Tensor:
    """
    How sharp the target is along each stroke's border, in [0, 1): sampled at 8 border points (both sides
    of its start, middle and end, plus the two end caps) at the pyramid level of a band ~radius / 2 wide, the
    stronger half of them averaged and mapped through `s / (s + ref)`. A stroke inside a soft area scores ~0
    even when an edge is nearby; one whose side or end lies on a crisp target edge scores high.
    `points` (B, N, 3, 2) control points and `radii` (B, N, 2) in image coordinates [0, 1]; returns (B, N).
    """
    B, L, H, W = pyramid.shape
    N = points.shape[1]
    p0, p1, p2 = points.unbind(2)
    r0, r2 = radii[..., 0], radii[..., 1]  # ends (a pressure curve's middle value is ignored here)
    centers = torch.stack([p0, 0.25 * p0 + 0.5 * p1 + 0.25 * p2, p2], 2)  # curve at t = 0, 0.5, 1
    tangents = torch.stack([p1 - p0, p2 - p0, p2 - p1], 2) + torch.tensor([1e-6, 0.0], device=points.device)
    tangents = tangents / tangents.norm(dim=-1, keepdim=True)  # a dot (zero tangent) falls back to +x
    normals = torch.stack([-tangents[..., 1], tangents[..., 0]], -1)
    r = torch.stack([r0, 0.5 * (r0 + r2), r2], 2).unsqueeze(-1)
    border = torch.cat([centers + r * normals, centers - r * normals,
                        (p0 - r0.unsqueeze(-1) * tangents[:, :, 0]).unsqueeze(2),
                        (p2 + r2.unsqueeze(-1) * tangents[:, :, 2]).unsqueeze(2)], 2)  # (B, N, 8, 2)
    grid = (border.reshape(B, N * 8, 1, 2) * 2.0 - 1.0).to(pyramid.dtype)
    samples = F.grid_sample(pyramid, grid, align_corners=True, padding_mode="border").view(B, L, N, 8)
    band = radii.mean(-1) * (max(H, W) - 1) / 4.0
    level = torch.log2(band.clamp(min=1.0)).clamp(max=L - 1)  # (B, N)
    pick = (1.0 - (level.unsqueeze(1) - torch.arange(L, device=level.device).view(1, L, 1)).abs()).clamp(min=0.0)
    per_point = (samples * pick.unsqueeze(-1)).sum(1)  # (B, N, 8)
    s = per_point.topk(4, dim=-1).values.mean(-1)
    return s / (s + ref)


def stroke_shape_bias(points: torch.Tensor, radii: torch.Tensor, visible: torch.Tensor, pixels: float,
                      length_weight: float = 0.0, size_weight: float = 0.0) -> torch.Tensor:
    """
    Painterly bias `(B,)` (lower = longer, larger strokes): minus the mean of `length_weight * log(1 + chord)` and
    `size_weight * log(radius)` over a step's strokes, weighted by how visible they are (`visible` (B, N), detached:
    a faint or switched-off stroke gains nothing from its shape, so making strokes faint is no way to collect it).
    Lengths in pixels of `pixels` per unit (`points` (B, N, 3, 2), `radii` (B, N, k)). The chord, not the arc, so
    a stroke can't gain length by doubling back. Logs: every doubling is worth the same, and the pull on a stroke
    that is already long fades (1 / length).
    """
    d = points[:, :, 2] - points[:, :, 0]
    chord = torch.sqrt((d * d).sum(-1) + 1e-12) * pixels  # (sqrt with eps: a dot's gradient stays finite)
    r = (radii.mean(-1) * pixels).clamp(min=0.25)
    w = visible.detach()
    w = w / w.sum(-1, keepdim=True).clamp(min=1e-6)
    return -(w * (length_weight * torch.log1p(chord) + size_weight * torch.log(r))).sum(-1)


def robust_color_error(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Saturating per-pixel error `(B, 1, H, W)`: `rho(e) = tau * e / (e + tau)` (slope 1 at 0, at most `tau`) of
    the lightness error plus half the mean of it over the color (RGB minus lightness) errors. A gray error
    of d counts ~d like L1; a pure hue change counts half of what L1 would give it.
    """
    def rho(e: torch.Tensor) -> torch.Tensor:
        return tau * e / (e + tau)

    d = pred - target
    luma = d.mean(1, keepdim=True)
    return rho(luma.abs()) + 0.5 * rho((d - luma).abs()).mean(1, keepdim=True)


def local_std(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    """Per-channel standard deviation within `window` x `window` px of each pixel, `(B, C, H, W)`."""
    pad = window // 2
    mean = F.avg_pool2d(F.pad(x, (pad,) * 4, mode="replicate"), window, stride=1)
    mean_sq = F.avg_pool2d(F.pad(x * x, (pad,) * 4, mode="replicate"), window, stride=1)
    return torch.sqrt((mean_sq - mean * mean).clamp(min=0.0) + 1e-6)


def _soft_sqrt(x: torch.Tensor, eps: float) -> torch.Tensor:
    """sign(x) sqrt|x|, smooth through 0 (bounded slope)."""
    return x * torch.rsqrt(x.abs() + eps)


def _window(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur with a wide `sigma`, computed at reduced resolution (pool by ~sigma/4) and upsampled."""
    f = int(sigma // 4)
    if f < 2:
        return _gaussian_blur(x, sigma)
    H, W = x.shape[-2:]
    low = _gaussian_blur(F.avg_pool2d(x, f, ceil_mode=True), sigma / f)
    return F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)


def texture_statistics(x: torch.Tensor, window: float = 16.0, bands: Sequence[float] = (0.7, 1.5, 3.0)) -> torch.Tensor:
    """
    Local texture statistics of `x` `(B, C, H, W)` -> `(B, 7 * len(bands), H, W)`: the image is split into
    frequency bands (differences of Gaussian blurs at `bands` px) and, per band, within a Gaussian window of
    std `window` px: the energy per color channel, the luma structure tensor (stroke direction and strength)
    and its anisotropic part (how coherent the marks are: brush strokes vs noise). It does not say where in
    the window a mark is, so cloned or slightly moved texture matches while blurred texture does not.
    """
    feats, prev = [], x
    for sigma in bands:
        low = _gaussian_blur(x, sigma)
        h, prev = prev - low, low
        feats.append(_window(h * h, window).add(1e-6).sqrt())
        y = h.mean(1, keepdim=True)
        gx = F.pad(y[..., :, 2:] - y[..., :, :-2], (1, 1, 0, 0), mode="replicate") / 2
        gy = F.pad(y[..., 2:, :] - y[..., :-2, :], (0, 0, 1, 1), mode="replicate") / 2
        jxx, jyy, jxy = _window(torch.cat([gx * gx, gy * gy, gx * gy], 1), window).split(1, 1)
        feats.append(_soft_sqrt(torch.cat([jxx, jyy, jxy], 1), 1e-8))
        aniso = ((jxx - jyy) ** 2 + 4 * jxy**2 + 1e-12).sqrt()
        feats.append(aniso * torch.rsqrt(jxx + jyy + 1e-4))
    return torch.cat(feats, 1)


def shift_tolerant_detail(pred: torch.Tensor, target: torch.Tensor, radius: int = 3, patch: int = 7,
                          sigma: float = 1.5) -> torch.Tensor:
    """
    `(B, 1, H, W)`: RMS difference of the fine bands (image minus its `sigma` px blur) per `patch` x `patch`
    patch, against the best-matching patch of the other image within +-`radius` px, averaged over both
    directions (prediction -> target and back, so missing and extra detail both count). Detail moved by up to
    `radius` px is free; blurred detail costs its amplitude; wrong detail costs more than none.
    """
    def run(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p, t = pred - _gaussian_blur(pred, sigma), target - _gaussian_blur(target, sigma)
        return 0.5 * (_best_shift(p, t, radius, patch) + _best_shift(t, p, radius, patch))

    if torch.is_grad_enabled() and (pred.requires_grad or target.requires_grad):
        # recomputed in the backward pass: the per-shift differences would cost ~1 GB per call at 16 x 128 px
        return torch.utils.checkpoint.checkpoint(run, pred, target, use_reentrant=False)
    return run(pred, target)


def _best_shift(a: torch.Tensor, b: torch.Tensor, radius: int, patch: int) -> torch.Tensor:
    """Per pixel, the RMS difference of `a`'s patch to the closest of `b`'s patches within +-radius px."""
    B, C, H, W = a.shape
    k = 2 * radius + 1
    shifted = F.unfold(F.pad(b, (radius,) * 4, mode="replicate"), k).view(B, C, k * k, H, W)
    d = (a.unsqueeze(2) - shifted).pow(2).mean(1)  # (B, shifts, H, W)
    pad = patch // 2
    d = F.pad(d, (pad,) * 4, mode="replicate")
    d = F.avg_pool2d(F.avg_pool2d(d, (patch, 1), 1), (1, patch), 1)  # separable box filter
    return d.min(1, keepdim=True).values.add(1e-6).sqrt()


def simplify(x: torch.Tensor, size: int = 9, passes: int = 2) -> torch.Tensor:
    """
    `x` with everything smaller than about `size` px removed and the edges of larger shapes kept hard:
    `passes` median filters of `size` x `size` (per channel). Unlike a blur, a hard edge stays hard and a soft
    one soft, so it reads like a painter's first block-in: silhouettes and big color areas, no eyes, lines or
    texture. Differentiable (the gradient goes to each window's median pixel).
    """
    p = size // 2
    B, C, H, W = x.shape
    for _ in range(passes):
        u = F.unfold(F.pad(x, (p,) * 4, mode="replicate"), size)  # (B, C * size^2, H * W)
        x = u.view(B, C, size * size, H * W).median(2).values.view(B, C, H, W)
    return x


def _coarse(x: torch.Tensor, factor: int, mode: str) -> torch.Tensor:
    """
    Low-resolution view of `x` at `factor`. "blur" pools by factor/4 and then Gaussian-blurs, so the
    view varies smoothly with stroke position (overlapping support of ~`factor` px). "pool" is plain
    non-overlapping average pooling, which only gives placement signal within a pooling cell.
    """
    if mode == "pool" or factor <= 1:
        return _pool(x, factor)
    p = max(1, factor // 4)
    return _gaussian_blur(_pool(x, p), sigma=0.5 * factor / p)


class MultiScalePaintLoss(nn.Module):
    """
    Reconstruction loss for stroke-based painting: a full-resolution term for detail plus terms on an
    blurred low-resolution pyramid for coarse color and placement.

    The low-resolution terms matter for optimization as much as for quality: a stroke placed in the wrong
    spot barely overlaps its target at full resolution (hard edges give almost no gradient), but in a
    blurred low-resolution view the two still overlap, so the stroke gets pulled towards where paint is
    needed. With `coarse_mode="blur"` (default) that pull is smooth over ~`factor` pixels.

    Args:
        detail_weight: weight of the full-resolution term: L1, or with `robust_tau` > 0 a saturating error.
        robust_tau: > 0 replaces the full-resolution L1 by `tau * e / (e + tau)` per pixel: like L1 for small
            errors, capped at `tau` for large ones, on lightness (full weight) and color (half weight)
            separately. Painting a wrong color over another wrong color then costs almost nothing, breaking a
            pixel that was right costs a lot, so a big stroke that gets most of its area right is worth it
            on a mostly-wrong canvas (early) and expensive on a mostly-right one (late). Being a function of
            the canvas (not of the per-step change) it cannot be gamed by breaking and repairing pixels. It
            also prefers a crisp edge slightly off (few pixels fully wrong) over a soft one (many pixels half
            wrong), where L1 is neutral-ish and L2 prefers soft. The coarse terms stay L1 for their pull.
        scales: downsampling factors of the coarse terms, e.g. (4, 16).
        scale_weights: weight per coarse scale (defaults to 1.0 each).
        coarse_mode: "blur" (smooth, overlapping) or "pool" (non-overlapping average pooling).
        edge_weight: weight of two edge terms (0 disables them):
            "edge"  - 0.5 x L1 between the image gradients (Sobel): places edges correctly. On its own it
                      still prefers a smeared edge over a sharp one that is 1-2 px off (same hedging as
                      pixel L1), hence:
            "sharp" - L1 between the local maximum gradient magnitudes (5 px window): a sharp edge slightly
                      off position matches the target's local sharpness, a smeared edge does not.
        flat_weight: weight of an L1 term between the local standard deviations (5 px windows) of
            prediction and target ("flat"). Several strokes that don't quite match in color / opacity leave
            a flat target area mottled: small in pixel L1, very visible. 0 disables it.
        value_weight: weight of a lightness-only L1 term ("value"), averaged over full resolution and the
            coarse scales (a squint test: value structure reads first, hue errors matter less). 0 disables it.
        forgive_tau: > 0 splits the edge / sharp / flat terms into "too soft / too flat" (always counted) and
            "too sharp / too busy" (counted only where the colors around it are already roughly right: weight
            `tau / (e + tau)` with e the 5 px mean color error, not differentiated). A crisp stroke over an
            area that was wrong anyway is then judged by its color, not punished for an edge the target
            doesn't have; once the area is right, a spurious edge costs fully again.
        texture_weight: weight of a texture term (`texture_statistics` of prediction vs target, L1): matches
            how much fine detail there is, which way it runs and how stroke-like it is, but not where each
            mark sits: it prefers cloned / slightly moved texture over blurred texture, unlike every
            pixel-exact term. Alone (or with only weak exact terms) it is also satisfied by busy, wrongly
            colored strokes; pair it with `shift_detail_weight` (see scripts/loss_check.py).
        texture_window: std of its window, px.
        shift_detail_weight: weight of a shift-tolerant fine-detail term (`shift_tolerant_detail`): the
            fine band must match the target's, but each patch may match anywhere within `shift_radius` px.
            Unlike the texture term, wrong detail (busy strokes, noise, broken lines) costs more than none.
        shift_radius: how far (px) detail may move for free.
        shift_sigma: blur scale (px) below which detail counts as fine.
        texture_bands: blur scales (px) that split the image into its frequency bands.
        simplify_size: > 0 judges each area by its large shapes until they are right, then by everything:
            per pixel `(1 - r) * block + r * (the full-resolution terms)`, the coarse terms unchanged. "block"
            compares the simplified images (`simplify`: shapes smaller than this many px removed, the edges of
            larger ones kept hard) by L1 and local sharpness: big shapes in place with hard edges where the
            target has them, small things ignored (S(pred) drops them too: not punished, just not counted).
            Readiness `r = readiness_tau / (e + readiness_tau)` (not differentiated), with e the mean color
            error of S(pred) vs S(target) over a window twice the size. r is never 0, so details always count
            a little; a larger tau brings them in earlier. With r = 1 everywhere this is the loss without
            simplification. The value is not monotone (an area that gets ready starts counting its details),
            so compare canvases by the loss without it. 0 disables it.
        simplify_passes: median passes of the simplification.
        readiness_tau: color error of the simplified images at which the details count half.
        simplify_weight: weight of the block-in term.
        extra_terms: optional named terms added on top (e.g. a perceptual or adversarial loss later),
            as `name -> (weight, fn(pred, target, mask))`.
    """

    def __init__(
        self,
        detail_weight: float = 1.0,
        scales: Sequence[int] = (4, 16),
        scale_weights: Optional[Sequence[float]] = None,
        coarse_mode: str = "blur",
        edge_weight: float = 0.0,
        flat_weight: float = 0.0,
        extra_terms: Optional[Dict[str, Tuple[float, LossTerm]]] = None,
        robust_tau: float = 0.0,
        value_weight: float = 0.0,
        forgive_tau: float = 0.0,
        texture_weight: float = 0.0,
        texture_window: float = 16.0,
        texture_bands: Sequence[float] = (0.7, 1.5, 3.0),
        shift_detail_weight: float = 0.0,
        shift_radius: int = 3,
        shift_sigma: float = 1.5,
        simplify_size: int = 0,
        simplify_passes: int = 2,
        readiness_tau: float = 0.05,
        simplify_weight: float = 1.0,
    ):
        super().__init__()
        self.detail_weight = detail_weight
        self.robust_tau = robust_tau
        self.value_weight = value_weight
        self.forgive_tau = forgive_tau
        self.texture_weight = texture_weight
        self.texture_window = texture_window
        self.texture_bands = tuple(texture_bands)
        self.shift_detail_weight = shift_detail_weight
        self.shift_radius = shift_radius
        self.shift_sigma = shift_sigma
        self.simplify_size = simplify_size
        self.simplify_passes = simplify_passes
        self.readiness_tau = readiness_tau
        self.simplify_weight = simplify_weight
        self.scales = tuple(scales)
        self.scale_weights = tuple(scale_weights) if scale_weights is not None else (1.0,) * len(self.scales)
        assert len(self.scale_weights) == len(self.scales)
        assert coarse_mode in ("blur", "pool")
        self.coarse_mode = coarse_mode
        self.edge_weight = edge_weight
        self.flat_weight = flat_weight
        self.extra_terms = dict(extra_terms or {})

    @torch.no_grad()
    def readiness(self, s_pred: torch.Tensor, s_target: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) in (0, 1]: how right the large shapes are around each pixel (simplified images)."""
        k = 2 * self.simplify_size + 1
        e = (s_pred - s_target).abs().mean(1, keepdim=True)
        e = F.avg_pool2d(F.pad(e, (k // 2,) * 4, mode="replicate"), k, 1)
        return self.readiness_tau / (e + self.readiness_tau)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor] = None,
        per_sample: bool = False, readiness: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns the total loss (a scalar, or `(B,)` per image with `per_sample`) and a dict of the
        individual (weighted) terms, as batch means, for logging. Extra terms are always batch scalars.
        With `simplify_size`, `readiness` (B, 1, H, W) replaces the readiness of `pred` (e.g. another canvas').
        """

        def mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
            return _masked_mean(x, mask, per_sample)

        terms: Dict[str, torch.Tensor] = {}
        logs: Dict[str, float] = {}
        ready = None  # (B, 1, H, W) readiness, with the simplified images
        if self.simplify_size > 0:
            with torch.no_grad():
                s_target = simplify(target, self.simplify_size, self.simplify_passes)
            s_pred = simplify(pred, self.simplify_size, self.simplify_passes)
            ready = readiness.detach() if readiness is not None else self.readiness(s_pred, s_target)
            logs["ready"] = float(ready.mean() if valid_mask is None else _masked_mean(ready, valid_mask))

        def fine(x: torch.Tensor) -> torch.Tensor:
            """A full-resolution term's map, counted by readiness."""
            return x if ready is None else ready * x

        if ready is not None:  # the block-in: large shapes in place, with the target's hard edges
            block = (s_pred - s_target).abs() + 0.5 * (local_sharpness(s_pred) - local_sharpness(s_target)).abs()
            terms["block"] = self.simplify_weight * mean((1.0 - ready) * block, valid_mask)

        if self.detail_weight > 0 and self.robust_tau > 0:
            terms["detail"] = self.detail_weight * mean(fine(robust_color_error(pred, target, self.robust_tau)), valid_mask)
        elif self.detail_weight > 0:
            terms["detail"] = self.detail_weight * mean(fine((pred - target).abs()), valid_mask)

        right = None  # (B, 1, H, W): 1 where the colors around are right, -> 0 where far off
        if self.forgive_tau > 0 and (self.edge_weight > 0 or self.flat_weight > 0):
            with torch.no_grad():
                e = F.avg_pool2d(F.pad((pred - target).abs().mean(1, keepdim=True), (2,) * 4, mode="replicate"), 5, 1)
                right = self.forgive_tau / (e + self.forgive_tau)

        def excess(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """|p - t|, with the part where p exceeds t (too sharp / busy) weighted by `right`."""
            if right is None:
                return (p - t).abs()
            return (t - p).clamp(min=0.0) + right * (p - t).clamp(min=0.0)

        if self.edge_weight > 0:
            gp, gt = image_gradients(pred), image_gradients(target)
            diff = (gp - gt).abs()
            if right is not None:  # a gradient stronger than the target's is a spurious edge
                diff = torch.where(gp.abs() > gt.abs(), right * diff, diff)
            terms["edge"] = 0.5 * self.edge_weight * mean(fine(diff), valid_mask)
            terms["sharp"] = self.edge_weight * mean(fine(excess(local_sharpness(pred), local_sharpness(target))),
                                                     valid_mask)

        if self.flat_weight > 0:
            terms["flat"] = self.flat_weight * mean(fine(excess(local_std(pred), local_std(target))), valid_mask)

        def coarse_l1(p: torch.Tensor, t: torch.Tensor, factor: int) -> torch.Tensor:
            if valid_mask is None:
                return mean((_coarse(p, factor, self.coarse_mode) - _coarse(t, factor, self.coarse_mode)).abs(), None)
            # Downsample masked images and renormalize by the downsampled mask so padding does not leak in
            m = _coarse(valid_mask, factor, self.coarse_mode)
            diff = (
                _coarse(p * valid_mask, factor, self.coarse_mode) - _coarse(t * valid_mask, factor, self.coarse_mode)
            ) / (m + 1e-6)
            return mean(diff.abs(), (m > 0).to(diff.dtype))

        for factor, weight in zip(self.scales, self.scale_weights):
            if weight > 0:
                terms[f"x{factor}"] = weight * coarse_l1(pred, target, factor)

        if self.value_weight > 0:
            vp, vt = pred.mean(1, keepdim=True), target.mean(1, keepdim=True)
            factors = (1,) + self.scales
            terms["value"] = self.value_weight * sum(
                mean(fine((vp - vt).abs()), valid_mask) if f == 1 else coarse_l1(vp, vt, f) for f in factors
            ) / len(factors)

        if self.texture_weight > 0:
            def tex(x: torch.Tensor) -> torch.Tensor:
                return texture_statistics(x, self.texture_window, self.texture_bands)
            terms["texture"] = self.texture_weight * mean(fine((tex(pred) - tex(target)).abs()), valid_mask)

        if self.shift_detail_weight > 0:
            d = shift_tolerant_detail(pred, target, self.shift_radius, sigma=self.shift_sigma)
            terms["shift_detail"] = self.shift_detail_weight * mean(fine(d), valid_mask)

        for name, (weight, fn) in self.extra_terms.items():
            terms[name] = weight * fn(pred, target, valid_mask)

        total = sum(terms.values())
        return total, {**{k: float(v.detach().mean()) for k, v in terms.items()}, **logs}


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB for images in [0, 1], shape (B,)."""
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse + 1e-10)
