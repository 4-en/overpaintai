"""
Checks checkpoints for oscillation in long paintings: steps that make the painting worse (e.g. a model that
alternates between washing over an area and repairing it). Paints each image with whole-image steps through
the high-res harness, exactly as at inference, and reports per checkpoint:
  - final and best PSNR (a final far below the best = the painting got worse again),
  - the share of steps that lower the PSNR by more than --tolerance dB,
  - the sum of all drops (dB): how much painting was undone in total.

    python -m overpaintai.scripts.check_oscillation runs/a/best.pt runs/b/best.pt --image overpaintai_test.png \
        --data ~/Downloads/pixiv/ --val-images 8
"""

import argparse
from pathlib import Path

import torch

from overpaintai.data import ImageFolderDataset
from overpaintai.highres import HighResPainter, config_from_checkpoint, load_checkpoint_model, load_padded


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--image", action="append", default=[], help="Image to paint (repeatable)")
    p.add_argument("--data", default=None, help="Image folder: also paint the first --val-images of its val split")
    p.add_argument("--val-images", type=int, default=8)
    p.add_argument("--size", type=int, default=128, help="Canvas size (the oscillation also shows at 128 px)")
    p.add_argument("--steps", type=int, default=192, help="Whole-image steps")
    p.add_argument("--tolerance", type=float, default=0.05, help="dB a step may lose before it counts as worse")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def trace(painter: HighResPainter, target: torch.Tensor, valid: torch.Tensor, side: int) -> torch.Tensor:
    """PSNR after every whole-image step `(steps,)`."""
    out = []

    def on_progress(event, get_canvas):
        mse = (((get_canvas() - target) ** 2) * valid).sum() / (valid.sum() * 3)
        out.append((-10 * torch.log10(mse)).item())

    painter.paint(target, side, valid=valid, log=lambda *_: None, on_progress=on_progress)
    return torch.tensor(out)


@torch.no_grad()
def main():
    args = parse_args()
    dev = torch.device(args.device)
    images = []  # (name, target, valid, source side)
    for path in args.image:
        target, valid, _, side = load_padded(path, args.size, dev)
        images.append((Path(path).name, target, valid, side))
    if args.data:
        val = ImageFolderDataset(args.data, args.size, "val", augment=False)
        for i in range(min(args.val_images, len(val))):
            images.append((f"val{i}", val[i].unsqueeze(0).to(dev), torch.ones(1, 1, args.size, args.size, device=dev),
                           args.size))
    if not images:
        raise SystemExit("nothing to paint: pass --image and / or --data")

    for path in args.checkpoints:
        model, ckpt = load_checkpoint_model(path, dev)
        cfg, view_res, tool = config_from_checkpoint(ckpt, whole_steps=args.steps, rounds_per_level=0)
        painter = HighResPainter(model, args.size, view_res, tool, config=cfg, device=dev)
        print(f"{path} ({'joint' if painter.joint else 'sequential'} strokes)")
        totals = []
        for name, target, valid, side in images:
            ps = trace(painter, target, valid, side)
            d = ps[1:] - ps[:-1]
            row = (ps[-1].item(), ps.max().item(), 100 * (d < -args.tolerance).float().mean().item(),
                   (-d.clamp(max=0)).sum().item())
            totals.append(row)
            print(f"  {name:24s} final {row[0]:6.2f} best {row[1]:6.2f} dB | worse steps {row[2]:3.0f}% | "
                  f"drops {row[3]:6.2f} dB")
        if len(totals) > 1:
            mean = [sum(r[i] for r in totals) / len(totals) for i in range(4)]
            print(f"  {'mean':24s} final {mean[0]:6.2f} best {mean[1]:6.2f} dB | worse steps {mean[2]:3.0f}% | "
                  f"drops {mean[3]:6.2f} dB")


if __name__ == "__main__":
    main()
