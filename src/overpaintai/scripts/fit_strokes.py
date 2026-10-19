"""
Stroke fitting without a network: optimizes stroke parameters directly through the differentiable
renderer to reproduce target images. Answers whether (and how well) strokes can be placed by gradient
descent on the reconstruction loss, which any trained stroke predictor relies on.

Targets are synthetic paintings rendered by the engine itself (so an exact solution exists), or a
user-provided image via --image.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.data import load_image
from overpaintai.losses import MultiScalePaintLoss, psnr
from overpaintai.synthetic import radius_to_param, random_strokes, render_targets
from overpaintai.viz import save_curve, save_frame

TOOLS = {"paint": 0, "dry": 1, "smudge": 2, "blur": 3, "eraser": 4}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=str, default=None, help="Fit this image instead of synthetic targets")
    p.add_argument("--size", type=int, default=128, help="Canvas resolution")
    p.add_argument("--batch", type=int, default=4, help="Synthetic targets fitted in parallel")
    p.add_argument("--target-strokes", type=int, default=100, help="Strokes in each synthetic target")
    p.add_argument("--strokes", type=int, default=160, help="Total strokes to fit")
    p.add_argument("--stages", type=int, default=4, help="Coarse-to-fine stages strokes are added in")
    p.add_argument("--iters", type=int, default=250, help="Optimization steps per stage")
    p.add_argument("--lr", type=float, default=0.03, help="Adam learning rate (on unconstrained params)")
    p.add_argument("--tool", choices=sorted(TOOLS), default="paint", help="Tool used by all fitted strokes")
    p.add_argument("--radius-start", type=float, default=0.10, help="Initial radius of the first stage")
    p.add_argument("--radius-end", type=float, default=0.015, help="Initial radius of the last stage")
    p.add_argument("--init-opacity", type=float, default=0.9, help="Opacity new strokes start with")
    p.add_argument("--edge-px-start", type=float, default=6.0, help="Hard edge width at the start (px)")
    p.add_argument("--edge-px-end", type=float, default=1.0, help="Hard edge width at the end (px)")
    p.add_argument("--detail-weight", type=float, default=1.0, help="Full-resolution L1 weight")
    p.add_argument("--scales", type=str, default="4,16", help="Downsampling factors of coarse L1 terms")
    p.add_argument("--scale-weights", type=str, default="1,1", help="Weights of the coarse terms")
    p.add_argument("--coarse-mode", choices=("blur", "pool"), default="blur", help="Coarse term type")
    p.add_argument("--freeze-previous", action="store_true", help="Only optimize the current stage")
    p.add_argument("--no-checkpoint", action="store_true", help="Disable per-stroke checkpointing")
    p.add_argument("--log-every", type=int, default=25, help="Log / frame interval (steps)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/fit", help="Output directory")
    return p.parse_args()


def logit(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


@torch.no_grad()
def init_strokes(
    engine: DifferentiableCanvasEngine,
    canvas: torch.Tensor,
    target: torch.Tensor,
    n: int,
    radius: float,
    generator: torch.Generator,
    opacity: float = 0.9,
) -> torch.Tensor:
    """
    Error-guided initialization of `n` strokes per image, returned as raw (pre-sigmoid) params
    `(B, n, 14)`: positions are sampled proportionally to the (smoothed) squared error, colors are
    taken from the locally averaged target, brushes start at `radius` with a random direction.
    """
    B, _, H, W = target.shape
    k = max(3, int(round(radius * H)) | 1)
    err = F.avg_pool2d(((canvas - target) ** 2).mean(1, keepdim=True), k, stride=1, padding=k // 2)
    probs = (err.flatten(1) + 1e-8) / (err.flatten(1) + 1e-8).sum(1, keepdim=True)
    idx = torch.multinomial(probs, n, replacement=True, generator=generator)  # (B, n)
    ys, xs = (idx // W).float() / (H - 1), (idx % W).float() / (W - 1)

    smooth = F.avg_pool2d(target, k, stride=1, padding=k // 2)
    color = torch.gather(smooth.flatten(2), 2, idx.unsqueeze(1).expand(B, 3, n)).permute(0, 2, 1)

    dev = target.device
    angle = 2.0 * math.pi * torch.rand(B, n, generator=generator, device=dev)
    half = 1.5 * radius
    dx, dy = half * torch.cos(angle), half * torch.sin(angle)
    x0, y0, x2, y2 = (xs - dx).clamp(0, 1), (ys - dy).clamp(0, 1), (xs + dx).clamp(0, 1), (ys + dy).clamp(0, 1)
    r = radius_to_param(engine, torch.full((B, n), radius, device=dev))
    a = torch.full_like(xs, (opacity - 0.05) / 0.95)  # inverse of the engine's opacity mapping
    params = torch.stack(
        [x0, y0, xs, ys, x2, y2, r, r, a, a, torch.full_like(xs, 0.5)],
        dim=-1,
    )
    return logit(torch.cat([params, color], dim=-1))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    S = args.size
    engine = DifferentiableCanvasEngine(canvas_size=S, hard_edge_px=args.edge_px_start).to(device).train()

    if args.image:
        target = load_image(args.image, S, device)
        gt_strokes = None
    else:
        gt_params, gt_tools = random_strokes(engine, args.batch, args.target_strokes, generator=gen)
        engine.hard_edge_px = 1.0
        target = render_targets(engine, gt_params, gt_tools, S)
        gt_strokes = args.target_strokes
    B = target.shape[0]
    background = engine.create_background(B, S, S, device)

    scales = [int(s) for s in args.scales.split(",") if s]
    scale_weights = [float(w) for w in args.scale_weights.split(",") if w]
    loss_fn = MultiScalePaintLoss(args.detail_weight, scales, scale_weights, coarse_mode=args.coarse_mode)

    tool_w = torch.zeros(1, 1, engine.NUM_TOOLS, device=device)
    tool_w[..., TOOLS[args.tool]] = 1.0

    per_stage = [args.strokes // args.stages + (1 if s < args.strokes % args.stages else 0) for s in range(args.stages)]
    total_steps = args.iters * args.stages
    stage_params: list = []
    frozen_layer = engine.create_blank_paint_layer(B, S, S, device)
    optimizer = None
    log_rows, stage_starts = [], []
    step, t_start = 0, time.time()

    def render(edge_px: float, with_grad: bool = True):
        engine.hard_edge_px = edge_px
        active = stage_params[-1:] if args.freeze_previous else stage_params
        start_layer = frozen_layer if args.freeze_previous else engine.create_blank_paint_layer(B, S, S, device)
        raw = torch.cat(active, dim=1)
        with torch.set_grad_enabled(with_grad):
            params = torch.sigmoid(raw)
            layer = engine(
                start_layer, params, tool_w.expand(B, raw.shape[1], -1),
                checkpoint_strokes=not args.no_checkpoint,
            )
            return engine.composite(layer, background)

    for stage in range(args.stages):
        # Radius of this stage's new strokes: log-linear from radius_start to radius_end
        f = stage / max(1, args.stages - 1)
        radius = math.exp(math.log(args.radius_start) + f * (math.log(args.radius_end) - math.log(args.radius_start)))

        with torch.no_grad():
            current = render(engine.hard_edge_px, with_grad=False) if stage_params else background
            if args.freeze_previous and stage_params:
                frozen_layer = engine(
                    frozen_layer, torch.sigmoid(stage_params[-1]),
                    tool_w.expand(B, stage_params[-1].shape[1], -1),
                )
        new = init_strokes(engine, current, target, per_stage[stage], radius, gen, args.init_opacity)
        new.requires_grad_(True)
        stage_params.append(new)
        if optimizer is None or args.freeze_previous:
            optimizer = torch.optim.Adam([new], lr=args.lr)
        else:
            optimizer.add_param_group({"params": [new]})
        stage_starts.append(step)

        for it in range(args.iters):
            progress = step / max(1, total_steps - 1)
            edge_px = args.edge_px_start * (args.edge_px_end / args.edge_px_start) ** progress
            canvas = render(edge_px)
            loss, terms = loss_fn(canvas, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0 or (stage == args.stages - 1 and it == args.iters - 1):
                with torch.no_grad():
                    p = psnr(canvas, target).mean().item()
                n_strokes = sum(t.shape[1] for t in stage_params)
                row = {"step": step, "stage": stage, "strokes": n_strokes, "edge_px": edge_px,
                       "loss": loss.item(), "psnr": p, "time": time.time() - t_start, **terms}
                log_rows.append(row)
                terms_txt = " ".join(f"{k}={v:.4f}" for k, v in terms.items())
                print(f"step {step:5d} stage {stage} strokes {n_strokes:4d} edge {edge_px:4.2f}px "
                      f"loss {loss.item():.4f} ({terms_txt}) psnr {p:5.2f}dB  {row['time']:6.1f}s", flush=True)
                save_frame(out / "frames" / f"step_{step:05d}.png", target, canvas,
                           f"stage {stage} | {n_strokes} strokes | step {step} | PSNR {p:.2f} dB")
            step += 1

    # Final evaluation at a crisp 1px edge, independent of the annealing schedule
    with torch.no_grad():
        final = render(1.0, with_grad=False)
        final_psnr = psnr(final, target)
    save_frame(out / "final.png", target, final, f"final | {args.strokes} strokes | PSNR {final_psnr.mean():.2f} dB")
    save_curve(out / "curve.png", log_rows, markers=stage_starts)
    with open(out / "log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in log_rows for k in r)))
        writer.writeheader()
        writer.writerows(log_rows)
    torch.save({"params": torch.sigmoid(torch.cat(stage_params, dim=1)).cpu(), "tool": args.tool,
                "size": S, "args": vars(args)}, out / "strokes.pt")
    summary = {
        "final_psnr_mean": final_psnr.mean().item(),
        "final_psnr_per_image": final_psnr.tolist(),
        "fitted_strokes": args.strokes,
        "target_strokes": gt_strokes,
        "seconds": time.time() - t_start,
        "args": vars(args),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"final PSNR {final_psnr.mean():.2f} dB (per image: {', '.join(f'{v:.2f}' for v in final_psnr.tolist())}) "
          f"in {summary['seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
