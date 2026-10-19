"""
Train the stroke predictor.

Modes:
  teacher  Supervised pretraining on synthetic paintings with known strokes. The canvas holds the first
           j*N ground-truth strokes (teacher forcing) and the model predicts the next N; loss = Hungarian-
           matched stroke-parameter L1 + render loss against the ground-truth next canvas.
  unroll   Render-loss training: paint K steps from a blank canvas with the model's own strokes and
           compare the paintings to the target (multi-scale loss). Works with synthetic targets or a
           folder of images; optionally adds a zoomed-in tile step to train the high-resolution path.

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
from torch.utils.data import DataLoader
from tqdm import tqdm

from overpaintai import views
from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.data import ImageFolderDataset, SyntheticSource
from overpaintai.losses import MultiScalePaintLoss, psnr
from overpaintai.matching import matched_stroke_loss
from overpaintai.model import StrokePredictor
from overpaintai.painting import Painter
from overpaintai.viz import save_curve, save_frame

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
    g.add_argument("--crop-scale", default="0.35,1.0", help="Random square crop area range (zoom)")
    g.add_argument("--hflip", type=float, default=0.5)
    g.add_argument("--vflip", type=float, default=0.0)
    g.add_argument("--color-jitter", type=float, default=0.1)
    g.add_argument("--no-augment", action="store_true")

    g = p.add_argument_group("model")
    g.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="small")
    g.add_argument("--strokes-per-step", type=int, default=16, help="N strokes predicted per view and step")

    g = p.add_argument_group("painting / loss")
    g.add_argument("--steps", type=int, default=6, help="K whole-image painting steps (unroll + validation)")
    g.add_argument("--tile-prob", type=float, default=0.25, help="Unroll: chance a batch adds a zoomed tile step")
    g.add_argument("--tile-size", default="0.3,0.6", help="Unroll: tile side range (fraction of the image)")
    g.add_argument("--val-tile-steps", type=int, default=0, help="Validation: extra 2x2-tile steps")
    g.add_argument("--intermediate-weight", type=float, default=0.3, help="Unroll: loss weight of earlier steps")
    g.add_argument("--detach-canvas", action="store_true", help="Unroll: no gradient through earlier steps")
    g.add_argument("--detail-weight", type=float, default=1.0)
    g.add_argument("--scales", default="4,16")
    g.add_argument("--scale-weights", default="1,1")
    g.add_argument("--match-weight", type=float, default=1.0, help="Teacher: stroke-parameter loss weight")
    g.add_argument("--render-weight", type=float, default=1.0, help="Teacher: render loss weight")
    g.add_argument("--heatmap-weight", type=float, default=0.1,
                   help="Teacher: cross-entropy of placement maps against ground-truth stroke midpoints")
    g.add_argument("--heatmap-sigma", type=float, default=1.5, help="Teacher: heatmap target width (view px)")
    g.add_argument("--num-stamps", type=int, default=16, help="Curve samples per stroke in the renderer")
    g.add_argument("--no-checkpoint-strokes", action="store_true", help="Faster, but much more memory")

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

        self.engine = DifferentiableCanvasEngine(canvas_size=self.S, num_stamps=args.num_stamps).to(self.device).train()
        self.model_config = (
            resume["model_config"] if resume else dict(num_strokes=self.N, **MODEL_PRESETS[args.model_size])
        )
        self.model = StrokePredictor(**self.model_config).to(self.device)
        self.N = self.model.num_strokes
        self.ema = EMA(self.model, args.ema) if args.ema > 0 else None

        amp = torch.bfloat16 if args.amp == "bf16" else None
        ckpt_strokes = not args.no_checkpoint_strokes
        self.painter = Painter(self.model, self.engine, args.view_res, self.tools, amp, ckpt_strokes)
        eval_model = self.ema.model if self.ema else self.model
        self.eval_painter = Painter(eval_model, self.engine, args.view_res, self.tools, amp, checkpoint_strokes=False)

        scales = [int(s) for s in args.scales.split(",") if s]
        weights = [float(w) for w in args.scale_weights.split(",") if w]
        self.loss_fn = MultiScalePaintLoss(args.detail_weight, scales, weights)

        decay, no_decay = [], []
        for n, prm in self.model.named_parameters():
            (no_decay if prm.dim() < 2 or "norm" in n or n in ("queries", "token_type") else decay).append(prm)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.99),
        )

        self._setup_data()
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
        j = int(torch.randint(0, M // N, (1,), generator=self.gen, device=self.device).item())
        blank = self.engine.create_blank_paint_layer(B, self.S, self.S, self.device)
        with torch.no_grad():
            layer = self.engine(blank, batch.params[:, : j * N], batch.tools[:, : j * N]) if j > 0 else blank
            gt_params, gt_tools = batch.params[:, j * N : (j + 1) * N], batch.tools[:, j * N : (j + 1) * N]
            gt_canvas = self.painter.composite(self.engine(layer, gt_params, gt_tools))

        boxes = views.whole_image_boxes(B, self.device)
        progress = torch.full((B,), j * N / M, device=self.device)
        out = self.painter.predict(batch.target, layer, boxes, progress)
        gt = views.from_engine_params(gt_params, self.engine)  # whole-image view: view = image coords
        match_loss, assign = matched_stroke_loss(out.strokes, gt)

        params = views.to_engine_params(views.view_to_image(out.strokes, boxes), self.engine)
        canvas = self.painter.composite(self.painter.render(layer, params, self.painter.tool_weights(out.tool_logits)))
        render_loss, terms = self.loss_fn(canvas, gt_canvas)
        heat_loss = self.heatmap_loss(out.placement_logits, gt, assign)
        loss = a.match_weight * match_loss + a.render_weight * render_loss + a.heatmap_weight * heat_loss

        metrics = {"match": match_loss.detach(), "render": render_loss.detach(), "heatmap": heat_loss.detach()}
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
        return {"loss": loss, **metrics}

    def heatmap_loss(self, logits: torch.Tensor, gt: "views.Strokes", assign: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy between each matched query's placement map `(B, N, R*R)` and a Gaussian around its
        ground-truth stroke's curve midpoint B(0.5) (in view coordinates). Gives placement a direct, strong
        gradient; soft-argmax alone barely moves a near-uniform map.
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
        return -(t * log_p).sum(-1).mean()

    def unroll_batch(self) -> Dict[str, torch.Tensor]:
        a = self.args
        target = self.next_targets()
        B, S = target.shape[0], target.shape[-1]
        tile = torch.rand((), generator=self.gen, device=self.device).item() < a.tile_prob
        total = a.steps + int(tile)
        layer = self.engine.create_blank_paint_layer(B, S, S, self.device)
        loss, weight_sum, entropy = 0.0, 0.0, []

        for k in range(total):
            progress = torch.full((B,), k / total, device=self.device)
            boxes = None
            if tile and k == total - 1:
                lo, hi = _pair(a.tile_size)
                s = lo + (hi - lo) * torch.rand(B, 1, generator=self.gen, device=self.device)
                xy = (1.0 - s) * torch.rand(B, 2, generator=self.gen, device=self.device)
                boxes = torch.cat([xy, s], dim=1).unsqueeze(1)  # (B, 1, 3): one tile per image
            res = self.painter.step(target, layer, progress, boxes)
            layer = res.layer
            entropy.append(res.output.anchor_entropy.mean().detach())

            last = k == total - 1
            w = 1.0 if last else a.intermediate_weight
            if w > 0:
                canvas = self.painter.composite(layer)
                step_loss, terms = self.loss_fn(canvas, target)
                if boxes is not None:  # emphasize the zoomed region the tile step painted
                    tile_loss, _ = self.loss_fn(canvas, target, box_mask(boxes[:, 0], S))
                    step_loss = 0.5 * (step_loss + tile_loss)
                loss = loss + w * step_loss
                weight_sum += w
            if a.detach_canvas and not last:
                layer = layer.detach()

        loss = loss / weight_sum
        with torch.no_grad():
            final = self.painter.composite(layer)
            metrics = {"psnr": psnr(final, target).mean(), "entropy": torch.stack(entropy).mean()}
            metrics.update({k: torch.tensor(v) for k, v in terms.items()})
        return {"loss": loss, **metrics}

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
        stats["lr"] = self.scheduler.get_last_lr()[0]
        return stats

    # --- validation -----------------------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        a = self.args
        model = self.eval_painter.model
        model.eval()
        per_step, losses, finals = [], [], []
        for chunk in self.val_targets.split(16):
            layers = self.eval_painter.paint(chunk, a.steps, tile_steps=a.val_tile_steps)
            canvases = [self.eval_painter.composite(lay) for lay in layers]
            per_step.append(torch.stack([psnr(c, chunk) for c in canvases]))  # (steps, b)
            losses.append(self.loss_fn(canvases[-1], chunk)[0] * chunk.shape[0])
            finals.append(canvases[-1])
        per_step = torch.cat(per_step, dim=1).mean(dim=1)
        n = self.val_targets.shape[0]
        metrics = {"val_psnr": per_step[-1].item(), "val_loss": (sum(losses) / n).item()}
        metrics.update({f"val_psnr_step{k + 1}": v.item() for k, v in enumerate(per_step)})
        save_frame(
            self.out / "samples" / f"epoch_{self.epoch:03d}.png", self.val_targets[:4], torch.cat(finals)[:4],
            f"epoch {self.epoch} | {self.args.mode} | {per_step.numel()} steps | val PSNR {metrics['val_psnr']:.2f} dB",
        )
        return metrics

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
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        skipped = sorted(set(own) - set(compatible))
        self.model.load_state_dict(compatible, strict=False)
        if self.ema:
            self.ema.model.load_state_dict(self.model.state_dict())
        print(f"initialized from {path} ({len(compatible)} tensors" + (f", {len(skipped)} skipped)" if skipped else ")"))

    # --- main loop ------------------------------------------------------------------------------------
    def run(self):
        a = self.args
        step_log = self.out / "steps.csv"
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
                if row["val_psnr"] > self.best_psnr:
                    self.best_psnr, is_best = row["val_psnr"], True
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
            val_txt = (f" | val PSNR {row['val_psnr']:.2f} dB (loss {row['val_loss']:.4f})"
                       f"{' * best' if is_best else ''}") if "val_psnr" in row else ""
            tqdm.write(f"epoch {epoch + 1}: train loss {row['train_loss']:.4f}, PSNR {row['train_psnr']:.2f} dB, "
                       f"entropy {row['train_entropy']:.2f}{val_txt} | {row['time']:.0f}s")
        print(f"done: best val PSNR {self.best_psnr:.2f} dB -> {self.out / 'best.pt'}")

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
