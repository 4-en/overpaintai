"""
Train the stroke predictor.

Modes:
  teacher  Supervised pretraining on synthetic paintings with known strokes. The canvas holds all but the
           last k*N ground-truth strokes (k <= --teacher-tail, so the strokes to find are visible in the
           target) and the model predicts the next N; loss = Hungarian-matched stroke-parameter L1 +
           placement heatmap + render loss against the ground-truth next canvas.
  unroll   Render-loss training: paint K steps from a blank canvas with the model's own strokes and
           compare the paintings to the target (multi-scale loss). Works with synthetic targets or a
           folder of images; optionally adds a zoomed-in tile step to train the high-resolution path.
           With a canvas pool (--pool-size), most batches continue paintings from earlier batches
           instead of starting over, so the model also learns to work on deep canvases.

Typical PoC: teacher pretraining, then unroll fine-tuning initialized from it:
  overpaint-train --mode teacher --epochs 10 --out runs/teacher
  overpaint-train --mode unroll --init-from runs/teacher/best.pt --epochs 20 --out runs/unroll
  overpaint-train --mode unroll --data path/to/images --init-from runs/unroll/best.pt --out runs/real
Resume an interrupted run (model, EMA, optimizer, schedule, epoch, history) with --resume <ckpt>.
"""

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from overpaintai import views
from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.data import ImageFolderDataset, SyntheticSource, load_image
from overpaintai.degrade import degraded_canvas
from overpaintai.losses import MultiScalePaintLoss, psnr, sharpness_pyramid, stroke_edge_sharpness
from overpaintai.matching import matched_stroke_loss
from overpaintai.metrics import ACTIVE_OPACITY, painting_metrics
from overpaintai.model import GATE_LOGIT_RANGE, StrokeSamples, TransformerBlock
from overpaintai.plan_model import build_model
from overpaintai.painting import Painter
from overpaintai.viz import save_curve, save_frame, save_progress

TOOLS = {"paint": 0, "dry": 1, "smudge": 2, "blur": 3, "eraser": 4}

MODEL_PRESETS = {
    "tiny": dict(widths=(16, 32, 64, 128), token_dim=128, heads=4),
    "small": dict(widths=(32, 64, 128, 256), token_dim=256, heads=8),
    "base": dict(widths=(48, 96, 192, 384), token_dim=384, heads=8),
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("mode and data")
    g.add_argument("--mode", choices=("teacher", "unroll"), default="unroll")
    g.add_argument("--data", default="synthetic", help="'synthetic' or a folder of images (recursive)")
    g.add_argument("--image-size", type=int, default=128, help="Training image / canvas resolution "
                   "(rendering cost grows with its square; teacher mode gains nothing above --view-res)")
    g.add_argument("--view-res", type=int, default=128, help="Working resolution R the model sees")
    g.add_argument("--target-strokes", default="48,96", help="Synthetic: min,max strokes per painting")
    g.add_argument("--tools", default="paint", help="Comma-separated tools the model may use")
    g.add_argument("--val-fraction", type=float, default=0.05, help="Folder: share of images held out")
    g.add_argument("--val-size", type=int, default=32, help="Validation images")
    g.add_argument("--num-workers", type=int, default=4, help="Folder: DataLoader workers")

    g = p.add_argument_group("augmentation (folder data)")
    g.add_argument("--crop-scale", default="0.01,1.0",
                   help="Random square crop range as area fractions (0.01 = 10%% of the side); the side is "
                        "sampled log-uniformly, so all zoom levels are equally likely")
    g.add_argument("--hflip", type=float, default=0.5)
    g.add_argument("--vflip", type=float, default=0.0)
    g.add_argument("--color-jitter", type=float, default=0.1)
    g.add_argument("--no-augment", action="store_true")

    g = p.add_argument_group("model")
    g.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="small")
    g.add_argument("--strokes-per-step", type=int, default=16, help="N strokes predicted per view and step")
    g.add_argument("--arch", choices=("unet", "plan"), default="unet",
                   help="New models: 'unet' = StrokePredictor; 'plan' = PlanStrokePredictor (plan tokens, stroke "
                        "latents refined while seeing each other's provisional strokes, pressure curve, anchor "
                        "anywhere along the stroke; see plan_model.py)")
    g.add_argument("--global-blocks", type=int, default=None,
                   help="Transformer blocks over the plan / global tokens (default: 2 unet, 6 plan)")
    g.add_argument("--query-blocks", type=int, default=None,
                   help="Query block pairs (strokes coordinate, then read the plan; default: 2 unet, 4 plan). With "
                        "--init-from a shallower model, the extra blocks start as identity")
    g.add_argument("--refine-layers", type=int, default=3, help="Plan arch: stroke refinement layers")
    g.add_argument("--view-tokens", type=int, default=16, help="Plan arch: side of the view's token grid")
    g.add_argument("--plan-slots", type=int, default=8, help="Plan arch: learned plan tokens")
    g.add_argument("--no-pressure", action="store_true", help="Plan arch: no pressure curve (linear tapers)")
    g.add_argument("--no-stroke-gate", action="store_true", help="New models: no per-stroke gate output")
    g.add_argument("--no-paintable-mask", action="store_true", help="New models: no paintable-mask input")
    g.add_argument("--no-error-progress", action="store_true",
                   help="New --step-progress models: no error-based progress input")
    g.add_argument("--step-progress", action="store_true",
                   help="New models: condition on step / num_steps as before, instead of only on the canvas "
                        "(stationary: the same model fits any number of steps; needed for the canvas pool)")
    g.add_argument("--soft-gate", action="store_true",
                   help="New models: stroke gate as a dimmer in [0, 1] instead of on / off")
    g.add_argument("--joint-strokes", action="store_true",
                   help="New models: render each step's strokes as one order-independent operation (per pixel the "
                        "max coverage, colors mixed by coverage) instead of one after another; overlapping strokes "
                        "of a step merge seamlessly. Stored in the checkpoint, the harness renders the same way. "
                        "Works with --init-from a sequential model")

    g = p.add_argument_group("painting / loss")
    g.add_argument("--steps", type=int, default=6, help="K whole-image painting steps (unroll + validation)")
    g.add_argument("--tile-prob", type=float, default=0.25, help="Unroll: chance a batch adds a zoomed tile step")
    g.add_argument("--tile-size", default="0.3,0.6", help="Unroll: tile core side range (fraction of the image)")
    g.add_argument("--tile-margin", type=float, default=0.25,
                   help="Unroll: context margin around tile cores (fraction of the core, per side); strokes "
                        "outside the core are faded as in the high-res harness (with a paintable-mask model)")
    g.add_argument("--outside-penalty", type=float, default=0.0,
                   help="Unroll: extra cost for stroke footprint reaching past the paintable core (tile steps). "
                        "Experimental: the edge fade already pulls strokes inward, and at 0.1 the model learned "
                        "to shrink all tile strokes to the minimum radius instead")
    g.add_argument("--val-tile-steps", type=int, default=1,
                   help="Validation: extra 2x2-tile steps (with context margin for paintable-mask models)")
    g.add_argument("--val-long-steps", type=int, default=128,
                   help="Validation: also paint this many whole-image steps and log PSNR along the way, drift "
                        "from the best point and strokes still painted at the end; selects best.pt (0 = off)")
    g.add_argument("--intermediate-weight", type=float, default=0.3, help="Unroll: loss weight of earlier steps")
    g.add_argument("--step-ramp", type=float, default=0.0,
                   help="Unroll: earlier steps' weight = --intermediate-weight x ((k + 1) / steps) ** this, so the "
                        "first steps are judged more by where the painting ends up than by their own result "
                        "(room for a bold block-in that later steps refine). 0 = all earlier steps alike")
    g.add_argument("--start-mix", default="blank:1",
                   help="Unroll: start canvases per batch, as kind:weight pairs. 'blank' = empty canvas, "
                        "'warmup' = the model first paints 1..--rollout-max steps without gradients, "
                        "'degraded' = the target with synthetic painting mistakes (see degrade.py). "
                        "E.g. blank:0.4,warmup:0.3,degraded:0.3")
    g.add_argument("--rollout-max", type=int, default=12,
                   help="Unroll: max warm-up steps (and steps a degraded start counts as already painted)")
    g.add_argument("--degrade-strength", type=float, default=1.0, help="Unroll: how much degraded starts deviate")
    g.add_argument("--detach-canvas", action="store_true", help="Unroll: no gradient through earlier steps")
    g.add_argument("--grad-through-inputs", action="store_true",
                   help="Unroll: also backprop through the canvas the model looks at (unstable: explodes "
                        "over several steps); by default earlier steps get gradients via the renderer only")
    g.add_argument("--pool-size", type=int, default=0,
                   help="Unroll: paintings kept between batches and painted on further, e.g. 512 (0 = always "
                        "start over; needs a stationary model). ~0.5 MB GPU memory each at --image-size 128. "
                        "Experimental: at 128 px paintings saturate after a few dozen steps, and it has not beaten "
                        "fresh starts yet")
    g.add_argument("--pool-reset", type=float, default=0.05,
                   help="Unroll: chance per image and batch that a pool painting is replaced by a fresh start "
                        "(--start-mix). Sets the depth spread: median ~0.7/x batch visits, 10%% beyond 2.3/x")
    g.add_argument("--pool-min-gain", type=float, default=0.1,
                   help="Unroll: pool paintings whose last batch gained less PSNR than this (dB) are replaced "
                        "by fresh starts: finished paintings teach nothing but 'don't touch it', and with most "
                        "of a batch saying so, the model learned to paint timid soft strokes, then none")
    g.add_argument("--pool-fast-forward", type=int, default=0,
                   help="Unroll: extra steps without gradients on each painting before it goes back into the "
                        "pool (reaches deep canvases faster, at their cost)")
    g.add_argument("--detail-weight", type=float, default=1.0)
    g.add_argument("--edge-loss", action="store_true",
                   help="Add an L1 loss on image gradients (Sobel) that rewards sharp, correctly placed edges")
    g.add_argument("--edge-weight", type=float, default=1.0, help="Weight of the edge loss (with --edge-loss)")
    g.add_argument("--flat-loss", action="store_true",
                   help="Add a loss matching local variation (std) so flat target areas stay flat, not mottled")
    g.add_argument("--flat-weight", type=float, default=1.0, help="Weight of the flatness loss (with --flat-loss)")
    g.add_argument("--robust-loss", action="store_true",
                   help="Saturating full-resolution error instead of L1 (lightness counts double): painting over "
                        "already-wrong pixels is cheap, breaking right ones is expensive; favors bold early "
                        "strokes and crisp over soft edges")
    g.add_argument("--robust-tau", type=float, default=0.1,
                   help="Error at which the robust loss is half saturated (it is capped at this value); also the "
                        "color error at which --forgive-edges counts spurious edges half")
    g.add_argument("--hard-edge-reward", type=float, default=0.0,
                   help="Unroll: pushes strokes towards hard edges where the target is sharp along their border, "
                        "e.g. 0.05. Strokes in soft / flat areas are left free")
    g.add_argument("--value-weight", type=float, default=0.0,
                   help="Weight of a lightness-only loss over full resolution and the coarse scales, e.g. 1")
    g.add_argument("--forgive-edges", action="store_true",
                   help="Edges / texture the target doesn't have only count where the colors around are already "
                        "roughly right: a crisp stroke over a still-wrong area is judged by its color")
    g.add_argument("--stroke-cost", type=float, default=0.0,
                   help="Cost per active stroke (mean stroke gate) so steps leave out strokes they don't need. "
                        "With the relative loss it is a share of the remaining error: 0.02 = a batch's steps "
                        "with all strokes on must remove 2%% of the error left. Experimental: every cost tried "
                        "so far eventually switched all strokes off (off strokes get no gradient for position "
                        "and color, so turning them back on never pays off)")
    g.add_argument("--placement-spread", type=float, default=0.0,
                   help="Unroll: penalty for pairs of strokes whose placement maps nearly coincide (Bhattacharyya "
                        "overlap above 0.5; logged as `overlap`, the mean over pairs), e.g. 1. Without it the "
                        "plan model's strokes piled onto the same few pixels within a few hundred steps")
    g.add_argument("--no-relative-loss", action="store_true",
                   help="Unroll: plain loss instead of each image's loss relative to its loss at the start of "
                        "the batch (relative: deep canvases with little error left still get a strong signal)")
    g.add_argument("--worse-weight", type=float, default=1.0,
                   help="Unroll: extra loss for steps that make the painting worse than before the step")
    g.add_argument("--scales", default="4,16")
    g.add_argument("--scale-weights", default="1,1")
    g.add_argument("--teacher-tail", type=int, default=2,
                   help="Teacher: predict one of the last K groups of N strokes (1 = the final, fully visible "
                        "strokes); 0 = any group, including the first strokes of a blank canvas")
    g.add_argument("--match-weight", type=float, default=1.0, help="Teacher: stroke-parameter loss weight")
    g.add_argument("--teacher-relax", type=float, default=0.25,
                   help="Teacher: the stroke terms (matching, heatmap, gates on) fade out per image once its step gets "
                        "the pixels right with other strokes: weight = min(1, remaining / (relax x step error)), "
                        "with the step error = the error of not painting at all. 0.25: full weight until 75%% of "
                        "the step's error is gone, 0 when the result is exact. 0 disables (always full weight)")
    g.add_argument("--render-weight", type=float, default=1.0, help="Teacher: render loss weight")
    g.add_argument("--heatmap-weight", type=float, default=0.1,
                   help="Teacher: cross-entropy of placement maps against ground-truth stroke midpoints")
    g.add_argument("--heatmap-sigma", type=float, default=1.5, help="Teacher: heatmap target width (view px)")
    g.add_argument("--num-stamps", type=int, default=16, help="Curve samples per stroke in the renderer")
    g.add_argument("--no-checkpoint-strokes", action="store_true", help="Faster, but much more memory")
    g.add_argument("--best-of", type=int, default=1,
                   help="Unroll: paint each window this many times with different random draws (no gradient), "
                        "train on each image's best attempt (by its loss at the end of the window). Attempts "
                        "differ by --placement-noise and the hard gate's draws")
    g.add_argument("--placement-noise", type=float, default=0.0,
                   help="Training: Gumbel noise scale on the placement logits, so each stroke leans towards one "
                        "of its map's good spots per draw (e.g. 1); positions stay continuous, no noise at inference")
    g.add_argument("--checkpoint-steps", action="store_true",
                   help="Unroll: recompute each painting step in backward, keeping only the canvas between "
                        "steps: long unrolls (--steps 32+) fit in memory, with the full gradient, at ~1.3x compute")

    g = p.add_argument_group("optimization")
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--steps-per-epoch", type=int, default=500, help="Batches per epoch (folder: capped by size)")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--weight-decay", type=float, default=0.02)
    g.add_argument("--warmup-steps", type=int, default=500)
    g.add_argument("--min-lr-ratio", type=float, default=0.05, help="Cosine schedule floor")
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--ema", type=float, default=0.999, help="EMA decay for eval weights (0 disables)")
    g.add_argument("--amp", choices=("bf16", "none"), default="bf16", help="Autocast dtype for the network")

    g = p.add_argument_group("run")
    g.add_argument("--out", default="runs/train")
    g.add_argument("--resume", default=None, help="Continue a run from a checkpoint (all state)")
    g.add_argument("--init-from", default=None, help="Initialize model weights from a checkpoint only")
    g.add_argument("--val-every", type=int, default=1, help="Validate every N epochs")
    g.add_argument("--preview-image", nargs="+", default=None, metavar="PATH",
                   help="Image(s) painted after every epoch at --image-size; saved to <out>/previews/")
    g.add_argument("--preview-steps", type=int, default=None, help="Painting steps for previews (default: --steps)")
    g.add_argument("--save-every", type=int, default=0, help="Also keep epoch_XXX.pt every N epochs")
    g.add_argument("--log-every", type=int, default=25, help="Step log interval")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def _pair(text: str, cast=float):
    a, b = (cast(v) for v in text.split(","))
    return a, b


# ----------------------------------------------------------------------------------------------------
class EMA:
    """Exponential moving average of model weights, used for validation and the saved best model."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int):
        d = min(self.decay, (1 + step) / (10 + step))  # warm up the average early in training
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1.0 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def lr_lambda(warmup: int, total: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

    return f


def placement_overlap(logits: torch.Tensor, threshold: float = 0.5):
    """
    Overlap between the placement maps of a step's strokes, `logits` `(B, N, H*W)`: the Bhattacharyya coefficient
    of each pair (1 = the same map, 0 = disjoint). Returns (mean over pairs, mean of the part above `threshold`).
    """
    sq = logits.float().softmax(-1).sqrt()
    ov = sq @ sq.transpose(1, 2)
    n = ov.shape[1]
    pairs = ov[:, ~torch.eye(n, dtype=torch.bool, device=ov.device)]
    return pairs.mean(), (pairs - threshold).clamp(min=0.0).mean()


def box_mask(boxes: torch.Tensor, size: int) -> torch.Tensor:
    """(B, 3) boxes -> (B, 1, size, size) masks of the pixels inside each box."""
    coords = torch.linspace(0.0, 1.0, size, device=boxes.device)
    x0, y0, s = boxes[:, 0, None], boxes[:, 1, None], boxes[:, 2, None]
    in_x = (coords >= x0) & (coords <= x0 + s)
    in_y = (coords >= y0) & (coords <= y0 + s)
    return (in_y.unsqueeze(2) & in_x.unsqueeze(1)).unsqueeze(1).float()


def infinite(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        yield from loader


class CanvasPool:
    """
    Paintings (target, paint layer, steps painted so far) that unroll training keeps painting on across
    batches, as in the sample pool of "Growing Neural Cellular Automata": each batch continues some of them
    for a short unroll and puts them back, so the model sees canvases hundreds of steps deep, painted by
    itself, and learns to keep improving them (or to stop) instead of only the first steps of a painting.
    Kept on the training device.
    """

    def __init__(self, size: int, image_size: int, device: torch.device):
        self.targets = torch.zeros(size, 3, image_size, image_size, device=device)
        self.layers = torch.zeros(size, 4, image_size, image_size, device=device)
        self.depth = torch.zeros(size, dtype=torch.long, device=device)
        self.gain = torch.zeros(size, device=device)  # PSNR gained on the last visit
        self.filled = 0

    @property
    def size(self) -> int:
        return self.targets.shape[0]

    def draw(self, B: int, reset: float, gen: torch.Generator, min_gain: float = float("-inf")):
        """
        Slots `(B,)` for a batch and which of them start fresh `(B,)`: the next empty slots while the pool
        fills up (all fresh), then random slots, each replaced by a fresh start with probability `reset` or
        when it gained less than `min_gain` dB on its last visit (stagnant: finished, or stuck).
        """
        dev = self.targets.device
        if self.filled < self.size:
            slots = torch.arange(self.filled, self.filled + B, device=dev) % self.size
            self.filled = min(self.size, self.filled + B)
            return slots, torch.ones(B, dtype=torch.bool, device=dev)
        slots = torch.randperm(self.size, generator=gen, device=dev)[:B]
        fresh = (torch.rand(B, generator=gen, device=dev) < reset) | (self.gain[slots] < min_gain)
        return slots, fresh

    def get(self, slots: torch.Tensor):
        return self.targets[slots], self.layers[slots], self.depth[slots]

    def put(self, slots: torch.Tensor, targets: torch.Tensor, layers: torch.Tensor, depth: torch.Tensor,
            gain: torch.Tensor):
        self.targets[slots], self.layers[slots], self.depth[slots] = targets, layers.detach(), depth
        self.gain[slots] = gain

    def stats(self) -> Dict[str, float]:
        d = self.depth[: self.filled].float()
        return {"pool_depth_mean": d.mean().item(), "pool_depth_p90": torch.quantile(d, 0.9).item(),
                "pool_depth_max": d.max().item()}


# ----------------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        torch.manual_seed(args.seed)
        self.gen = torch.Generator(device=self.device).manual_seed(args.seed)

        resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
        if resume is not None and args.out == "runs/train":
            args.out = str(Path(args.resume).parent)  # continue in the original run directory
        self.out = Path(args.out)
        (self.out / "samples").mkdir(parents=True, exist_ok=True)

        self.tools = [TOOLS[t.strip()] for t in args.tools.split(",") if t.strip()]
        self.N = args.strokes_per_step
        self.S = args.image_size
        if args.mode == "teacher" and args.data != "synthetic":
            raise SystemExit("--mode teacher needs ground-truth strokes, i.e. --data synthetic")

        if resume:
            self.model_config = resume["model_config"]
        else:
            plan = args.arch == "plan"
            self.model_config = dict(
                num_strokes=self.N, **MODEL_PRESETS[args.model_size], stroke_gate=not args.no_stroke_gate,
                global_blocks=args.global_blocks or (6 if plan else 2),
                query_blocks=args.query_blocks or (4 if plan else 2),
                paintable_mask=not args.no_paintable_mask, stationary=not args.step_progress,
                error_progress=args.step_progress and not args.no_error_progress,  # stationary: built in
                hard_gate=not args.no_stroke_gate and not args.soft_gate,
            )
            if plan:
                self.model_config.update(arch="plan", refine_layers=args.refine_layers, view_tokens=args.view_tokens,
                                         plan_slots=args.plan_slots, pressure=not args.no_pressure,
                                         stroke_points="chord", color_base="scaled")
            if args.joint_strokes:
                self.model_config["joint_strokes"] = True
        # Gated strokes must be able to vanish completely: no opacity floor in the engine then
        min_opacity = 0.0 if self.model_config.get("stroke_gate") else 0.05
        self.engine = DifferentiableCanvasEngine(
            canvas_size=self.S, num_stamps=args.num_stamps, min_opacity=min_opacity
        ).to(self.device).train()
        self.model = build_model(self.model_config).to(self.device)
        self.model.placement_noise = args.placement_noise
        self.N = self.model.num_strokes
        self.ema = EMA(self.model, args.ema) if args.ema > 0 else None

        amp = torch.bfloat16 if args.amp == "bf16" else None
        ckpt_strokes = not args.no_checkpoint_strokes
        self.painter = Painter(self.model, self.engine, args.view_res, self.tools, amp, ckpt_strokes,
                               grad_through_inputs=args.grad_through_inputs)
        eval_model = self.ema.model if self.ema else self.model
        self.eval_painter = Painter(eval_model, self.engine, args.view_res, self.tools, amp, checkpoint_strokes=False)

        scales = [int(s) for s in args.scales.split(",") if s]
        weights = [float(w) for w in args.scale_weights.split(",") if w]
        self.loss_fn = MultiScalePaintLoss(
            args.detail_weight, scales, weights, edge_weight=args.edge_weight if args.edge_loss else 0.0,
            flat_weight=args.flat_weight if args.flat_loss else 0.0,
            robust_tau=args.robust_tau if args.robust_loss else 0.0, value_weight=args.value_weight,
            forgive_tau=args.robust_tau if args.forgive_edges else 0.0,
        )

        decay, no_decay = [], []
        for n, prm in self.model.named_parameters():
            (no_decay if prm.dim() < 2 or "norm" in n or n in ("queries", "token_type") else decay).append(prm)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.99),
        )

        self._setup_data()
        self.pool = None
        if args.mode == "unroll" and args.pool_size > 0:
            if self.model.stationary:
                self.pool = CanvasPool(max(args.pool_size, args.batch_size), self.S, self.device)
            else:
                print("note: canvas pool off (needs a stationary model; this one conditions on step progress)")
        self.start_kinds, weights = [], []
        for part in args.start_mix.split(","):
            kind, _, w = part.partition(":")
            if kind.strip() not in ("blank", "warmup", "degraded"):
                raise SystemExit(f"--start-mix: unknown start kind {kind!r}")
            self.start_kinds.append(kind.strip())
            weights.append(float(w or 1.0))
        self.start_weights = torch.tensor(weights, device=self.device)
        self.preview_targets = None
        if args.preview_image:
            self.preview_targets = torch.cat([load_image(p, self.S, self.device) for p in args.preview_image])
            (self.out / "previews").mkdir(exist_ok=True)
        total = args.epochs * self.steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda(args.warmup_steps, total, args.min_lr_ratio)
        )

        self.epoch, self.global_step, self.best_psnr = 0, 0, float("-inf")
        self.history: List[Dict] = []
        if resume is not None:
            self._load_resume(resume)
        elif args.init_from:
            self._init_from(args.init_from)

        n_params = sum(prm.numel() for prm in self.model.parameters())
        print(f"model {n_params / 1e6:.2f}M params | mode {args.mode} | data {args.data} | "
              f"{self.steps_per_epoch} steps/epoch | out {self.out}")
        (self.out / "args.json").write_text(json.dumps(vars(args), indent=2))

    # --- data -----------------------------------------------------------------------------------------
    def _setup_data(self):
        a = self.args
        if a.data == "synthetic":
            self.synth = SyntheticSource(
                self.engine, self.S, _pair(a.target_strokes, int), multiple_of=self.N,
                tool_probs=[1.0 if i in self.tools else 0.0 for i in range(5)],
            )
            self.steps_per_epoch = a.steps_per_epoch
            # Fixed validation paintings (seeded independently of training)
            vgen = torch.Generator(device=self.device).manual_seed(10_000 + a.seed)
            val = [self.synth.sample(min(16, a.val_size), vgen) for _ in range(math.ceil(a.val_size / 16))]
            self.val_targets = torch.cat([v.target for v in val])[: a.val_size]
            self.train_iter = None
        else:
            augment = not a.no_augment
            common = dict(val_fraction=a.val_fraction)
            train_set = ImageFolderDataset(
                a.data, self.S, "train", augment=augment, crop_scale=_pair(a.crop_scale),
                hflip=a.hflip, vflip=a.vflip, color_jitter=a.color_jitter, **common,
            )
            val_set = ImageFolderDataset(a.data, self.S, "val", augment=False, **common)
            loader = DataLoader(
                train_set, batch_size=a.batch_size, shuffle=True, drop_last=True,
                num_workers=a.num_workers, pin_memory=self.device.type == "cuda",
                persistent_workers=a.num_workers > 0,
            )
            if len(loader) == 0:
                raise SystemExit(f"Need at least --batch-size={a.batch_size} training images, found {len(train_set)}")
            self.steps_per_epoch = min(a.steps_per_epoch, len(loader))
            self.train_iter = infinite(loader)
            n_val = min(a.val_size, len(val_set))
            self.val_targets = torch.stack([val_set[i] for i in range(n_val)]).to(self.device)
            print(f"folder data: {len(train_set)} train / {len(val_set)} val images")

    def next_targets(self) -> torch.Tensor:
        if self.train_iter is not None:
            return next(self.train_iter).to(self.device, non_blocking=True)
        return self.synth.sample(self.args.batch_size, self.gen).target

    # --- training steps -------------------------------------------------------------------------------
    def teacher_batch(self) -> Dict[str, torch.Tensor]:
        a, B, N = self.args, self.args.batch_size, self.N
        batch = self.synth.sample(B, self.gen)
        M = batch.params.shape[1]
        # Early strokes are mostly painted over in the target, so predicting them from a blank canvas is
        # ill-posed (the best L1 answer is a hedged average). The last groups are (nearly) unoccluded and
        # can be read off the target - canvas residual; planning from blank is left to unroll training.
        groups = M // N
        lo = max(0, groups - a.teacher_tail) if a.teacher_tail > 0 else 0
        j = int(torch.randint(lo, groups, (1,), generator=self.gen, device=self.device).item())
        blank = self.engine.create_blank_paint_layer(B, self.S, self.S, self.device)
        with torch.no_grad():
            layer = self.engine(blank, batch.params[:, : j * N], batch.tools[:, : j * N]) if j > 0 else blank
            gt_params, gt_tools = batch.params[:, j * N : (j + 1) * N], batch.tools[:, j * N : (j + 1) * N]
            gt_canvas = self.painter.composite(self.engine(layer, gt_params, gt_tools, joint=self.painter.joint))

        boxes = views.whole_image_boxes(B, self.device)
        progress = torch.full((B,), j * N / M, device=self.device)
        out = self.painter.predict(batch.target, layer, boxes, progress)
        gt = views.from_engine_params(gt_params, self.engine)  # whole-image view: view = image coords
        match_loss, assign = matched_stroke_loss(out.strokes, gt, per_sample=True)  # (B,)

        gated = views.Strokes(out.strokes.points, out.strokes.radii, out.strokes.opacity * out.gate.unsqueeze(-1),
                              out.strokes.hardness, out.strokes.color)
        params = views.to_engine_params(views.view_to_image(gated, boxes), self.engine)
        canvas = self.painter.composite(self.painter.render(layer, params, self.painter.tool_weights(out.tool_logits)))
        render_loss, terms = self.loss_fn(canvas, gt_canvas)
        heat_loss = self.heatmap_loss(out.placement_logits, gt, assign)  # (B,)
        gate_loss = (1.0 - out.gate).mean(-1)  # every ground-truth stroke is real: matched strokes stay on
        # Different strokes that get the pixels right are not wrong: the stroke terms fade out per image as the
        # step's pixel error goes to zero (relative to the error of not painting); the pixel loss always counts
        with torch.no_grad():
            if a.teacher_relax > 0:
                left = (canvas - gt_canvas).abs().flatten(1).mean(1)
                step = (self.painter.composite(layer) - gt_canvas).abs().flatten(1).mean(1)
                relax = (left / (a.teacher_relax * step).clamp(min=1e-6)).clamp(max=1.0)
            else:
                relax = torch.ones(B, device=self.device)
        loss = a.render_weight * render_loss + (
            relax * (a.match_weight * match_loss + a.heatmap_weight * heat_loss + 0.1 * gate_loss)).mean()
        if out.gate_logit is not None:
            # Keep gate logits in range, as in unroll training. The gate's leaky gradient never fades, so the push
            # towards "on" grew the logits without bound (and with them the stroke head's shared layers, until
            # radii saturated at the minimum and teacher training blew up within a few hundred steps)
            over = (out.gate_logit.abs() - GATE_LOGIT_RANGE).clamp(min=0.0)
            loss = loss + 0.01 * (over * over).mean()

        metrics = {"match": match_loss.mean().detach(), "render": render_loss.detach(),
                   "heatmap": heat_loss.mean().detach(), "gate": out.gate.mean().detach(), "relax": relax.mean()}
        if len(self.tools) > 1:  # supervise tool choice on matched pairs
            b = torch.arange(B, device=self.device).unsqueeze(1)
            logits = out.tool_logits[b, assign[0]]
            target_tool = gt_tools[b, assign[1]].argmax(-1)
            tool_loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_tool.flatten())
            loss = loss + 0.1 * tool_loss
            metrics["tool_ce"] = tool_loss.detach()
        with torch.no_grad():
            metrics["psnr"] = psnr(canvas, gt_canvas).mean()
            metrics["entropy"] = out.anchor_entropy.mean()
            metrics["hardness"] = out.strokes.hardness.mean()
        return {"loss": loss, **metrics}

    def heatmap_loss(self, logits: torch.Tensor, gt: "views.Strokes", assign: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy between each matched query's placement map `(B, N, R*R)` and a Gaussian around its
        ground-truth stroke's curve midpoint B(0.5) (in view coordinates), per image `(B,)`. Gives placement a
        direct, strong gradient; soft-argmax alone barely moves a near-uniform map.
        """
        B, N, RR = logits.shape
        R = int(math.isqrt(RR))
        b = torch.arange(B, device=logits.device).unsqueeze(1)
        p0, p1, p2 = gt.points[b, assign[1]].unbind(dim=2)  # matched GT strokes, (B, K, 2)
        mid = 0.25 * p0 + 0.5 * p1 + 0.25 * p2
        u = torch.linspace(0.0, 1.0, R, device=logits.device)
        uv = torch.stack(torch.meshgrid(u, u, indexing="xy"), dim=-1).reshape(1, 1, RR, 2)
        sigma = self.args.heatmap_sigma / (R - 1)
        t = torch.exp(-((uv - mid.unsqueeze(2)) ** 2).sum(-1) / (2.0 * sigma * sigma))
        t = t / t.sum(-1, keepdim=True).clamp(min=1e-12)
        log_p = logits[b, assign[0]].log_softmax(dim=-1)
        return -(t * log_p).sum(-1).mean(-1)  # (B,)

    @torch.no_grad()
    def start_canvas(self, target: torch.Tensor, window: int):
        """
        Start paint layer for an unroll batch, per --start-mix: returns (layer, steps already "done", kind).
        Progress then runs from done / (done + window), so warm-up and degraded starts train later phases
        of a painting while keeping progress = fraction of the painting completed (as at inference).
        """
        a = self.args
        B, S = target.shape[0], target.shape[-1]
        blank = self.engine.create_blank_paint_layer(B, S, S, self.device)
        kind = self.start_kinds[int(torch.multinomial(self.start_weights, 1, generator=self.gen).item())]
        if kind == "blank" or a.rollout_max < 1:
            return blank, 0, kind
        done = int(torch.randint(1, a.rollout_max + 1, (1,), generator=self.gen, device=self.device).item())
        if kind == "degraded":
            return degraded_canvas(target, self.engine, self.gen, strength=a.degrade_strength), done, kind
        layer = blank  # warmup: the current model paints `done` steps on its own
        for i in range(done):
            progress = torch.full((B,), i / (done + window), device=self.device)
            layer = self.painter.step(target, layer, progress).layer
        return layer, done, kind

    @torch.no_grad()
    def batch_start(self, window: int):
        """
        Targets and start paint layers of an unroll batch, the steps already painted on each `(B,)`, and the
        pool slots to put the results back into (None without a pool). Without a pool every batch starts
        over (--start-mix); with one, most images continue a pool painting and a --pool-reset share starts
        fresh.
        """
        B = self.args.batch_size
        if self.pool is None:
            target = self.next_targets()
            layer, done, _ = self.start_canvas(target, window)
            return target, layer, torch.full((B,), done, device=self.device), None
        slots, fresh = self.pool.draw(B, self.args.pool_reset, self.gen, self.args.pool_min_gain)
        target, layer, depth = self.pool.get(slots)
        n = int(fresh.sum())
        if n:
            new_target = self.next_targets()[:n]
            new_layer, done, _ = self.start_canvas(new_target, window)
            target[fresh], layer[fresh], depth[fresh] = new_target, new_layer, done
        return target, layer, depth, slots

    def unroll_batch(self) -> Dict[str, torch.Tensor]:
        a = self.args
        tile = torch.rand((), generator=self.gen, device=self.device).item() < a.tile_prob
        window = a.steps + int(tile)
        target, layer, depth, slots = self.batch_start(window)
        B, S = target.shape[0], target.shape[-1]
        total = (depth + window).float()  # this painting's length so far: `depth` steps are already on the canvas
        loss, weight_sum, entropy, hardness, gates, outside, gate_range, worse = 0.0, 0.0, [], [], [], [], [], []
        soft_edges, overlap, crowding = [], [], []
        with torch.no_grad():
            pyramid = sharpness_pyramid(target) if a.hard_edge_reward > 0 else None
            start = self.painter.composite(layer)
            start_psnr = psnr(start, target)  # (B,)
            prev = self.loss_fn(start, target, per_sample=True)[0]  # (B,) loss before each step
        # Relative loss: each image's loss in units of its loss at the start of this batch, so a deep painting
        # with little error left counts as much as a blank one. The stroke cost is then relative too: a step
        # must remove a share of the error that is left, which keeps fine strokes worth it late in a painting
        # (an absolute cost outweighed everything a deep painting could still gain and switched all strokes off)
        scale = 1.0 / prev.clamp(min=1e-4) if not a.no_relative_loss else torch.ones_like(prev)

        tile_boxes = tile_cores = None
        if tile:  # one zoomed tile per image in the last step: a core plus context margin around it
            lo, hi = _pair(a.tile_size)
            c = lo + (hi - lo) * torch.rand(B, 1, generator=self.gen, device=self.device)
            xy = (1.0 - c) * torch.rand(B, 2, generator=self.gen, device=self.device)
            tile_cores = torch.cat([xy, c], dim=1)
            if self.model.paintable_mask:
                v = (c * (1.0 + 2.0 * a.tile_margin)).clamp(max=1.0)
                vxy = torch.minimum((xy - 0.5 * (v - c)).clamp(min=0.0), 1.0 - v)
                tile_boxes = torch.cat([vxy, v], dim=1).unsqueeze(1)
            else:  # models without a mask input: the tile view is the core itself, as before
                tile_boxes = tile_cores.unsqueeze(1)
                tile_cores = None

        def step_views(k: int):
            if tile and k == window - 1:
                return tile_boxes, tile_cores
            return None, None

        forced = self._best_of_draws(target, layer, depth, total, window, step_views) if a.best_of > 1 else None

        for k in range(window):
            progress = (depth + k).float() / total
            boxes, cores = step_views(k)
            step_args = (target, layer, progress, boxes, None if cores is None else cores.unsqueeze(1))
            draws = None if forced is None else forced[k]
            if a.checkpoint_steps:  # same random gate / tool samples in the recomputation (RNG state is kept)
                res = torch.utils.checkpoint.checkpoint(self.painter.step, *step_args, margin=a.tile_margin,
                                                        samples=draws, use_reentrant=False)
            else:
                res = self.painter.step(*step_args, margin=a.tile_margin, samples=draws)
            layer = res.layer
            entropy.append(res.output.anchor_entropy.mean().detach())
            hardness.append(res.output.strokes.hardness.mean().detach())
            gates.append(res.gate.mean())
            mean_ov, crowd = placement_overlap(res.output.placement_logits)
            overlap.append(mean_ov.detach())
            crowding.append(crowd)
            if pyramid is not None:  # softness of visible strokes where the target is sharp around them
                st = views.from_engine_params(res.params, self.engine)
                vis = st.opacity.mean(-1).detach()
                want = stroke_edge_sharpness(pyramid, st.points.detach(), st.radii.detach())
                soft_edges.append((vis * want * (1.0 - st.hardness[..., 0])).sum() / vis.sum().clamp(min=1e-6))
            if res.output.gate_logit is not None:  # keep gate logits in range so off-gates can come back
                over = (res.output.gate_logit.abs() - GATE_LOGIT_RANGE).clamp(min=0.0)
                gate_range.append((over * over).mean())
            if res.reach is not None:  # footprint past the core, in margin widths
                width = a.tile_margin * cores[:, 2:3] / boxes[:, 0, 2:3]
                # Not weighted by the gate: switching strokes off must not be a way around it (that made
                # tile steps collapse to all-off); only keeping strokes inside the core avoids it
                outside.append((res.reach / width.clamp(min=1e-3)).clamp(max=2.0).mean())

            last = k == window - 1
            canvas = self.painter.composite(layer)
            full, terms = self.loss_fn(canvas, target, per_sample=True)
            if a.worse_weight > 0:  # a step should never make the painting worse
                worse.append((full - prev).clamp(min=0.0))
            prev = full.detach()
            w = 1.0 if last else a.intermediate_weight * ((k + 1) / window) ** a.step_ramp
            if w > 0:
                step_loss = full
                if boxes is not None:  # emphasize the zoomed region the tile step painted
                    region = boxes[:, 0] if cores is None else cores
                    tile_loss, _ = self.loss_fn(canvas, target, box_mask(region, S), per_sample=True)
                    step_loss = 0.5 * (step_loss + tile_loss)
                loss = loss + w * step_loss
                weight_sum += w
            if a.detach_canvas and not last:
                layer = layer.detach()

        loss = (scale * loss).mean() / weight_sum
        if worse:
            loss = loss + a.worse_weight * (scale * torch.stack(worse).mean(0)).mean()
        mean_gate = torch.stack(gates).mean()
        if self.model.stroke_gate and a.stroke_cost > 0:
            loss = loss + a.stroke_cost * mean_gate
        if outside and a.outside_penalty > 0:
            loss = loss + a.outside_penalty * torch.stack(outside).mean()
        if gate_range:
            loss = loss + 0.01 * torch.stack(gate_range).mean()
        if soft_edges:
            loss = loss + a.hard_edge_reward * torch.stack(soft_edges).mean()
        if a.placement_spread > 0:
            loss = loss + a.placement_spread * torch.stack(crowding).mean()
        with torch.no_grad():
            final = self.painter.composite(layer)
            gain = psnr(final, target) - start_psnr  # (B,)
            metrics = {"psnr": psnr(final, target).mean(), "entropy": torch.stack(entropy).mean(),
                       "hardness": torch.stack(hardness).mean(), "start_psnr": start_psnr.mean(),
                       "gate": mean_gate.detach(), "depth": depth.float().mean(),
                       "overlap": torch.stack(overlap).mean()}
            if outside:
                metrics["outside"] = torch.stack(outside).mean().detach()
            if worse:
                metrics["worse"] = torch.stack(worse).mean().detach()
            if soft_edges:
                metrics["soft_edges"] = torch.stack(soft_edges).mean().detach()
            metrics["gain"] = gain.mean()  # improvement of this window over its start
            if forced is not None:  # how much better the chosen attempt was than the average one (loss units)
                sc = self._best_of_scores
                metrics["best_of_gain"] = ((sc.mean(0) - sc.amin(0)) * scale).mean()
            metrics.update({k: torch.tensor(v) for k, v in terms.items()})
            if slots is not None:  # back into the pool, to be continued by a later batch
                layer, depth = layer.detach(), depth + window
                for i in range(a.pool_fast_forward):
                    layer = self.painter.step(target, layer, (depth + i).float() / (depth + a.pool_fast_forward)).layer
                self.pool.put(slots, target, layer, depth + a.pool_fast_forward, gain)
        return {"loss": loss, **metrics}

    @torch.no_grad()
    def _best_of_draws(self, target, layer, depth, total, window, step_views) -> List[StrokeSamples]:
        """
        --best-of: paints the window --best-of times with independent random draws (stroke positions and
        gates), without gradients, and returns per step the draws of each image's best attempt (lowest
        loss at the end of the window). The training pass replays them, so it learns from the plan that
        worked out best over the whole window rather than from the best single step.
        """
        a = self.args
        attempts, scores = [], []
        for _ in range(a.best_of):
            cur, draws = layer, []
            for k in range(window):
                boxes, cores = step_views(k)
                res = self.painter.step(target, cur, (depth + k).float() / total, boxes,
                                        None if cores is None else cores.unsqueeze(1), margin=a.tile_margin)
                cur = res.layer
                draws.append(res.output.samples)
            attempts.append(draws)
            scores.append(self.loss_fn(self.painter.composite(cur), target, per_sample=True)[0])
        self._best_of_scores = torch.stack(scores)  # (K, B) loss of each attempt at the end of the window
        winner = self._best_of_scores.argmin(0)  # (B,)

        def pick(field: str, k: int):
            values = [getattr(att[k], field) for att in attempts]
            if values[0] is None:
                return None
            stacked = torch.stack(values)  # (K, B, ...)
            idx = winner.view(1, -1, *([1] * (stacked.dim() - 2))).expand(1, *stacked.shape[1:])
            return stacked.gather(0, idx)[0]

        return [StrokeSamples(pick("place_seed", k), pick("gate_on", k)) for k in range(window)]

    def train_step(self) -> Dict[str, float]:
        self.model.train()
        result = self.teacher_batch() if self.args.mode == "teacher" else self.unroll_batch()
        self.optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip or float("inf"))
        if torch.isfinite(grad_norm):
            self.optimizer.step()
            if self.ema:
                self.ema.update(self.model, self.global_step)
        self.scheduler.step()
        self.global_step += 1
        stats = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in result.items()}
        stats["grad_norm"] = float(grad_norm)
        stats["skipped"] = float(not torch.isfinite(grad_norm))  # non-finite gradient: no update
        stats["lr"] = self.scheduler.get_last_lr()[0]
        return stats

    # --- validation -----------------------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        a = self.args
        model = self.eval_painter.model
        model.eval()
        per_step, losses, finals, quality = [], [], [], []
        for chunk in self.val_targets.split(16):
            layers, steps = self.eval_painter.paint(chunk, a.steps, tile_steps=a.val_tile_steps, return_steps=True)
            canvases = [self.eval_painter.composite(lay) for lay in layers]
            per_step.append(torch.stack([psnr(c, chunk) for c in canvases]))  # (steps, b)
            losses.append(self.loss_fn(canvases[-1], chunk)[0] * chunk.shape[0])
            finals.append(canvases[-1])
            strokes = views.from_engine_params(torch.cat([r.params for r in steps], dim=1), self.engine)
            b, _, h, w = chunk.shape
            blank = self.eval_painter.composite(self.engine.create_blank_paint_layer(b, h, w, self.device))
            q = painting_metrics(canvases[-1], chunk, blank, strokes.opacity.mean(-1),
                                 strokes.radii.mean(-1) * (chunk.shape[-1] - 1))
            tile_steps = steps[a.steps:]
            if tile_steps:  # catches tile views that stop painting (e.g. collapsed stroke gates)
                q["tile_gate"] = torch.stack([r.gate.mean() for r in tile_steps]).mean().item()
            quality.append({k: v * chunk.shape[0] for k, v in q.items()})
        per_step = torch.cat(per_step, dim=1).mean(dim=1)
        n = self.val_targets.shape[0]
        metrics = {"val_psnr": per_step[-1].item(), "val_loss": (sum(losses) / n).item()}
        # Quality / efficiency measures (always logged, whatever the losses): sharpness, line art, strokes
        for key in quality[0]:
            if key != "psnr":
                metrics[f"val_{key}"] = sum(q[key] for q in quality) / n
        metrics.update({f"val_psnr_step{k + 1}": v.item() for k, v in enumerate(per_step)})
        save_frame(
            self.out / "samples" / f"epoch_{self.epoch:03d}.png", self.val_targets[:4], torch.cat(finals)[:4],
            f"epoch {self.epoch} | {self.args.mode} | {per_step.numel()} steps | val PSNR {metrics['val_psnr']:.2f} dB",
        )
        return metrics

    @torch.no_grad()
    def validate_long(self) -> Dict[str, float]:
        """
        Paints the validation images for --val-long-steps whole-image steps (inference runs far longer than
        the training unroll): PSNR after 8, 16, 32, ... steps and at the end, `drift` (how much PSNR the
        painting lost from its best step by the end; > 0 = later steps make it worse) and `end_strokes`
        (active strokes per step over the last 8 steps: a finished painting should need few).
        """
        L = self.args.val_long_steps
        marks = sorted({2**i for i in range(3, 20) if 2**i < L} | {L})
        painter = self.eval_painter
        painter.model.eval()
        sums = {f"val_long_psnr@{m}": 0.0 for m in marks}
        sums.update(val_long_drift=0.0, val_long_end_strokes=0.0)
        for chunk in self.val_targets.split(16):
            b, _, h, w = chunk.shape
            layer = self.engine.create_blank_paint_layer(b, h, w, self.device)
            best = torch.full((b,), float("-inf"), device=self.device)
            for k in range(L):
                res = painter.step(chunk, layer, torch.full((b,), k / L, device=self.device))
                layer = res.layer
                p = psnr(painter.composite(layer), chunk)
                best = torch.maximum(best, p)
                if k + 1 in marks:
                    sums[f"val_long_psnr@{k + 1}"] += p.sum().item()
                if k >= L - 8:
                    opacity = views.from_engine_params(res.params, self.engine).opacity.mean(-1)
                    sums["val_long_end_strokes"] += (opacity > ACTIVE_OPACITY).float().sum().item() / min(8, L)
            sums["val_long_drift"] += (best - p).sum().item()
        n = self.val_targets.shape[0]
        metrics = {k: v / n for k, v in sums.items()}
        metrics["val_long_psnr"] = metrics[f"val_long_psnr@{L}"]
        return metrics

    @torch.no_grad()
    def preview(self, name: str) -> Dict[str, float]:
        """Paints the --preview-image(s) with the current (EMA) weights; saves every step as one strip."""
        if self.preview_targets is None:
            return {}
        model = self.eval_painter.model
        was_training = model.training
        model.eval()
        steps = self.args.preview_steps or self.args.steps
        layers = self.eval_painter.paint(self.preview_targets, steps)
        canvases = [self.eval_painter.composite(lay) for lay in layers]
        p = psnr(canvases[-1], self.preview_targets).mean().item()
        caption = f"{name} | step {self.global_step} | {steps} steps x {self.N} strokes | PSNR {p:.2f} dB"
        for fname in (f"{name}.png", "latest.png"):
            save_progress(self.out / "previews" / fname, self.preview_targets, canvases, caption)
        model.train(was_training)
        return {"preview_psnr": p}

    # --- checkpoints ----------------------------------------------------------------------------------
    def save(self, path: Path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema": self.ema.model.state_dict() if self.ema else None,
                "model_config": self.model_config,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_psnr": self.best_psnr,
                "history": self.history,
                "args": vars(self.args),
                "view_res": self.args.view_res,
                "tools": self.tools,
            },
            path,
        )

    def _load_resume(self, ckpt: Dict):
        self.model.load_state_dict(ckpt["model"])
        if self.ema and ckpt.get("ema") is not None:
            self.ema.model.load_state_dict(ckpt["ema"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.best_psnr = ckpt["best_psnr"]
        self.history = ckpt.get("history", [])
        print(f"resumed from {self.args.resume}: epoch {ckpt['epoch']}, step {self.global_step}, "
              f"best val PSNR {self.best_psnr:.2f} dB")

    def _init_from(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("ema") or ckpt["model"]  # EMA weights are the better starting point
        own = self.model.state_dict()
        compatible, grown = {}, []
        for k, v in state.items():
            if k not in own:
                continue
            if own[k].shape == v.shape:
                compatible[k] = v
            elif own[k].dim() == v.dim() and all(n >= o for n, o in zip(own[k].shape, v.shape)):
                # Layer got bigger (new input channel / conditioning value / gate output): old weights in
                # place, new parts zero, so the model starts out behaving exactly like the old one
                t = torch.zeros_like(own[k])
                t[tuple(slice(0, o) for o in v.shape)] = v
                compatible[k] = t
                grown.append(k)
        if grown:
            print(f"grown layers (old weights copied, new parts zero): {', '.join(grown)}")
        skipped = sorted(set(own) - set(compatible))
        self.model.load_state_dict(compatible, strict=False)
        # Transformer blocks the old model didn't have (a deeper model): zero their residual outputs, so they
        # start as identity and the model starts out painting exactly like the old one
        new_blocks = []
        for name, module in self.model.named_modules():
            if isinstance(module, TransformerBlock) and not any(k.startswith(name + ".") for k in compatible):
                for layer in (module.attn.out_proj, module.mlp[-1]):
                    torch.nn.init.zeros_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
                new_blocks.append(name)
        if new_blocks:
            print(f"new transformer blocks (start as identity): {', '.join(new_blocks)}")
        if self.ema:
            self.ema.model.load_state_dict(self.model.state_dict())
        print(f"initialized from {path} ({len(compatible)} tensors" + (f", {len(skipped)} skipped)" if skipped else ")"))

    # --- main loop ------------------------------------------------------------------------------------
    def run(self):
        a = self.args
        step_log = self.out / "steps.csv"
        if self.preview_targets is not None and self.epoch == 0:
            p = self.preview("init")  # before any training (useful with --init-from)
            tqdm.write(f"preview before training: PSNR {p['preview_psnr']:.2f} dB -> {self.out / 'previews'}")
        for epoch in range(self.epoch, a.epochs):
            self.epoch = epoch
            sums: Dict[str, float] = {}
            t0 = time.time()
            bar = tqdm(range(self.steps_per_epoch), desc=f"epoch {epoch + 1}/{a.epochs} [{a.mode}]",
                       dynamic_ncols=True, leave=True)
            for i in bar:
                stats = self.train_step()
                for k, v in stats.items():
                    sums[k] = sums.get(k, 0.0) + v
                bar.set_postfix(loss=f"{stats['loss']:.4f}", psnr=f"{stats['psnr']:.2f}",
                                ent=f"{stats['entropy']:.2f}", lr=f"{stats['lr']:.1e}")
                if self.global_step % a.log_every == 0:
                    self._append_csv(step_log, {"step": self.global_step, "epoch": epoch, **stats})

            row = {"epoch": epoch + 1, "step": self.global_step, "time": time.time() - t0}
            row.update({f"train_{k}": v / self.steps_per_epoch for k, v in sums.items()})
            is_best = False
            if (epoch + 1) % a.val_every == 0 or epoch + 1 == a.epochs:
                row.update(self.validate())
                if a.val_long_steps > 0:
                    row.update(self.validate_long())
                score = row.get("val_long_psnr", row["val_psnr"])  # best.pt: the long painting, if measured
                if score > self.best_psnr:
                    self.best_psnr, is_best = score, True
            row.update(self.preview(f"epoch_{epoch + 1:03d}"))
            if self.pool is not None:
                row.update(self.pool.stats())
            self.history.append(row)

            self.save(self.out / "last.pt")
            if is_best:
                self.save(self.out / "best.pt")
            if a.save_every and (epoch + 1) % a.save_every == 0:
                self.save(self.out / f"epoch_{epoch + 1:03d}.pt")
            self._append_csv(self.out / "epochs.csv", row)
            save_curve(
                self.out / "curve.png", self.history,
                panels=(("train_loss", True), ("train_psnr", False), ("val_psnr", False), ("train_entropy", False)),
                x_key="epoch",
            )
            val_txt = (f" | val PSNR {row['val_psnr']:.2f} dB (loss {row['val_loss']:.4f}, edge err {row['val_edge_err']:.4f})"
                       f"{' * best' if is_best else ''}") if "val_psnr" in row else ""
            if "val_long_psnr" in row:
                val_txt += (f" | {a.val_long_steps} steps {row['val_long_psnr']:.2f} dB (drift "
                            f"{row['val_long_drift']:.2f}, {row['val_long_end_strokes']:.1f} strokes/step at the end)")
            preview_txt = f" | preview {row['preview_psnr']:.2f} dB" if "preview_psnr" in row else ""
            skipped = round(sums.get("skipped", 0.0))
            skip_txt = f" | WARNING: {skipped}/{self.steps_per_epoch} steps skipped (non-finite gradients)" if skipped else ""
            tqdm.write(f"epoch {epoch + 1}: train loss {row['train_loss']:.4f}, PSNR {row['train_psnr']:.2f} dB, "
                       f"entropy {row['train_entropy']:.2f}, hardness {row['train_hardness']:.2f}"
                       f"{val_txt}{preview_txt} | {row['time']:.0f}s{skip_txt}")
        metric = f"{a.val_long_steps}-step val PSNR" if a.val_long_steps > 0 else "val PSNR"
        print(f"done: best {metric} {self.best_psnr:.2f} dB -> {self.out / 'best.pt'}")

    @staticmethod
    def _append_csv(path: Path, row: Dict):
        new = not path.exists()
        if not new:  # keep a stable header; extra keys (e.g. val metrics on some epochs) get new columns
            with open(path) as fh:
                header = next(csv.reader(fh), [])
            if set(row) - set(header):
                with open(path) as fh:
                    rows = list(csv.DictReader(fh))
                header = header + [k for k in row if k not in header]
                with open(path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=header)
                    w.writeheader()
                    w.writerows(rows)
        else:
            header = list(row)
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            if new:
                w.writeheader()
            w.writerow(row)


def main(argv: Optional[List[str]] = None):
    Trainer(parse_args(argv)).run()


if __name__ == "__main__":
    main(sys.argv[1:])
