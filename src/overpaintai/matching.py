"""Order-free supervision of predicted strokes against ground-truth strokes (Hungarian matching)."""

from typing import Dict, Tuple

import torch
from scipy.optimize import linear_sum_assignment

from overpaintai.views import Strokes


def stroke_features(s: Strokes, reverse: bool = False, weights: Dict[str, float] = None,
                    with_mid: bool = False) -> torch.Tensor:
    """
    Flat per-stroke feature vector `(B, N, 14)` used for matching: points, radii, opacity, hardness, color,
    each scaled by its weight. With `reverse=True` the stroke is traversed end-to-start (endpoints, radius
    and opacity tapers swapped), which renders identically for non-directional tools. With `with_mid`, the
    pressure curve's middle radius / opacity are features too (strokes without one: the mean of the ends).
    """
    w = {"points": 1.0, "radii": 0.25, "opacity": 0.5, "hardness": 0.25, "color": 1.0, **(weights or {})}
    points, radii, opacity = s.points, s.radii[..., :2], s.opacity[..., :2]
    if reverse:
        points, radii, opacity = points.flip(2), radii.flip(-1), opacity.flip(-1)
    if with_mid:  # the middle stays the middle when reversed
        mid_r = s.radii[..., 2:] if s.radii.shape[-1] == 3 else s.radii.mean(-1, keepdim=True)
        mid_a = s.opacity[..., 2:] if s.opacity.shape[-1] == 3 else s.opacity.mean(-1, keepdim=True)
        radii, opacity = torch.cat([radii, mid_r], -1), torch.cat([opacity, mid_a], -1)
    B, N = radii.shape[:2]
    return torch.cat(
        [
            w["points"] * points.reshape(B, N, 6),
            w["radii"] * torch.log(radii.clamp(min=1e-4)),  # log scale: relative size differences
            w["opacity"] * opacity,
            w["hardness"] * s.hardness,
            w["color"] * s.color,
        ],
        dim=-1,
    )


def matched_stroke_loss(
    pred: Strokes, target: Strokes, weights: Dict[str, float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    L1 distance between predicted and target strokes under the best one-to-one assignment, taking the
    better of the two traversal directions per pair. Shapes `(B, N, ...)` and `(B, M, ...)`; with N != M,
    only min(N, M) pairs are matched (extra predictions are unsupervised by this term).

    Returns the scalar loss (mean per-feature L1 of the matched pairs) and the assignment as a
    `(2, B, K)` tensor of (prediction index, target index) pairs, K = min(N, M).
    """
    mid = pred.radii.shape[-1] == 3  # a predicted pressure curve is matched against the target's (or linear)
    f_pred = stroke_features(pred, weights=weights, with_mid=mid)
    f_fwd = stroke_features(target, weights=weights, with_mid=mid)
    f_rev = stroke_features(target, reverse=True, weights=weights, with_mid=mid)
    # Pairwise L1 cost (B, N, M), min over target orientation
    cost = torch.minimum(torch.cdist(f_pred, f_fwd, p=1), torch.cdist(f_pred, f_rev, p=1))

    cost_cpu = cost.detach().float().cpu()
    rows, cols = [], []
    for b in range(cost_cpu.shape[0]):
        r, c = linear_sum_assignment(cost_cpu[b].numpy())
        rows.append(torch.as_tensor(r))
        cols.append(torch.as_tensor(c))
    rows = torch.stack(rows).to(cost.device)
    cols = torch.stack(cols).to(cost.device)
    batch = torch.arange(cost.shape[0], device=cost.device).unsqueeze(1)
    matched = cost[batch, rows, cols]  # (B, K)
    loss = matched.mean() / f_pred.shape[-1]
    return loss, torch.stack([rows, cols])
