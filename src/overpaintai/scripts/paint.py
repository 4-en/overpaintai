"""
Paint an image at high resolution with a trained stroke predictor: whole-image steps on the downscaled
image, then error-guided tiles of decreasing size (see overpaintai/highres.py).

  overpaint-paint --checkpoint runs/unroll/best.pt --image picture.png --size 2048
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from overpaintai.highres import HighResConfig, HighResPainter, load_checkpoint_model
from overpaintai.viz import to_uint8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Training checkpoint (best.pt / last.pt)")
    p.add_argument("--image", required=True)
    p.add_argument("--size", type=int, default=1024, help="Output canvas resolution (longer side)")
    p.add_argument("--out", default=None, help="Output folder (default: runs/paint/<image>_<size>)")
    p.add_argument("--whole-steps", type=int, default=None, help="Whole-image steps (default: training steps)")
    p.add_argument("--rounds", type=int, default=2, help="Tile rounds per level")
    p.add_argument("--max-tiles", type=int, default=32, help="Tiles per round")
    p.add_argument("--margin", type=float, default=0.25, help="Context margin per side (fraction of core)")
    p.add_argument("--threshold", type=float, default=0.015, help="Skip tiles with lower mean core error")
    p.add_argument("--min-tile", type=float, default=None,
                   help="Smallest tile core (fraction of the image); default: stop at the source's detail")
    p.add_argument("--no-tiles", action="store_true", help="Only the whole-image steps")
    p.add_argument("--no-fade", action="store_true", help="Keep strokes at tile edges unfaded")
    p.add_argument("--importance", default=None, help="Grayscale mask: bright areas get more / finer tiles")
    p.add_argument("--importance-gain", type=float, default=2.0)
    p.add_argument("--num-stamps", type=int, default=32, help="Curve samples per stroke when rendering")
    p.add_argument("--save-stages", action="store_true", help="Also save the full-res canvas after each stage")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_padded(path: str, size: int, device, mode: str = "RGB"):
    """
    Loads an image with its longer side resized to `size` and pads it to a square (edge-replicated, so the
    model sees plausible content there). Transparent images are composited onto white.
    Returns (image (1, C, S, S), valid mask (1, 1, S, S), (w, h) of the content, longer source side).
    """
    img = Image.open(path)
    if mode == "RGB" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    img = img.convert(mode)
    src_side = max(img.size)
    scale = size / src_side
    w, h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    img = img.resize((w, h), Image.LANCZOS if scale < 1 else Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype="float32") / 255.0)
    x = x.view(h, w, -1).permute(2, 0, 1).unsqueeze(0)
    pad = (0, size - w, 0, size - h)
    x = F.pad(x, pad, mode="replicate")
    valid = torch.zeros(1, 1, size, size)
    valid[..., :h, :w] = 1.0
    return x.to(device), valid.to(device), (w, h), src_side


def save_progress(path: Path, snapshots, target, wh, tile: int = 384):
    w, h = wh
    frames = [("target", target)] + list(snapshots)
    tw, th = tile, max(1, round(tile * h / w))
    sheet = Image.new("RGB", (tw * len(frames), th + 20), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(frames):
        small = F.interpolate(img[..., : round(h), : round(w)], size=(th, tw), mode="area")
        sheet.paste(Image.fromarray(to_uint8(small[0])), (i * tw, 20))
        draw.text((i * tw + 4, 4), label, fill="black")
    sheet.save(path)


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, ckpt = load_checkpoint_model(args.checkpoint, device)
    train_args = ckpt.get("args", {})
    steps = train_args.get("steps", 6)
    tile_lo = float(str(train_args.get("tile_size", "0.3,0.6")).split(",")[0])

    cfg = HighResConfig(
        whole_steps=args.whole_steps or steps,
        tile_progress=steps / (steps + 1),
        rounds_per_level=0 if args.no_tiles else args.rounds,
        max_tiles=args.max_tiles,
        margin=args.margin,
        error_threshold=args.threshold,
        min_core=args.min_tile,
        scale_clamp=(tile_lo, 1.0),
        fade=not args.no_fade,
        importance_gain=args.importance_gain,
    )
    tool = (ckpt.get("tools") or [0])[0]
    painter = HighResPainter(model, args.size, ckpt.get("view_res", 128), tool, args.num_stamps, cfg, device=device)

    target, valid, wh, src_side = load_padded(args.image, args.size, device)
    importance = None
    if args.importance:
        importance, _, _, _ = load_padded(args.importance, args.size, device, mode="L")
        importance = importance * valid

    out = Path(args.out or f"runs/paint/{Path(args.image).stem}_{args.size}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"painting {args.image} (source {src_side}px) at {args.size}px with {args.checkpoint}; "
          f"tile levels: {', '.join(f'1/{round(1 / c)}' for c in painter.core_sizes(src_side)) or 'none'}")
    result = painter.paint(target, src_side, valid=valid, importance=importance)

    w, h = wh
    crop = (slice(None), slice(None), slice(0, h), slice(0, w))
    Image.fromarray(to_uint8(result.canvas[crop][0])).save(out / "final.png")
    Image.fromarray(to_uint8(target[crop][0])).save(out / "target.png")
    save_progress(out / "progress.png", result.snapshots, target, wh)
    if args.save_stages:
        for i, (label, canvas) in enumerate(result.snapshots):
            name = label.replace(" ", "_").replace("/", "-")
            Image.fromarray(to_uint8(canvas[crop][0])).save(out / f"stage_{i}_{name}.png")
    torch.save({"strokes": torch.cat(result.strokes), "size": args.size, "content_wh": wh,
                "format": "points(6), radii(2), opacity(2), hardness, color(3); image coords"}, out / "strokes.pt")
    (out / "stats.json").write_text(json.dumps({"args": vars(args), "stages": result.stats}, indent=2))
    print(f"-> {out}/final.png ({result.stats[-1]['strokes']} strokes, PSNR {result.stats[-1]['psnr']:.2f} dB)")


if __name__ == "__main__":
    main()
