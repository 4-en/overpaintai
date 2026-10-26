"""
Checks how loss functions rank edited versions of an image: edits that still read the same (cloned or slightly
moved texture) should score closer to the original than edits that are wrong (blurred, structure changed).

An example is a folder with `original.<ext>` and edited copies of the same size, each labeled "good" or "bad",
either in a `labels.json` ({"clone_and_liquify": "good", "blur": "bad"}), by a `good_` / `bad_` file name
prefix, or by naming the one good edit `good.<ext>` (every other edit is then bad). Every folder below the
one given (and itself) that has an original is an example.

Each loss scores 128 px crops (centred on the edited areas) at several downscales, like training crops. The
table shows, per bad edit, its mean loss divided by the worst good edit's: above 1 = ranked right.

Paintings work as examples too: a target as `original`, a painting that should score better as `good` and
worse paintings (e.g. from a run whose loss the model exploited) as bad edits. Such pairs catch losses an
optimizer can exploit, which edits of the original cannot.

    python -m overpaintai.scripts.loss_check loss_test --synthetic \\
        --loss "mine:detail_weight=0,scales=32,shift_detail_weight=20,texture_weight=40,texture_window=12"

Loss specs are `name:key=value,...` with MultiScalePaintLoss arguments (lists separated by `/`, e.g.
scales=4/16).
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from overpaintai.losses import MultiScalePaintLoss, _gaussian_blur

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

LOSSES = {
    # the frame_unroll training loss (--edge-loss --edge-weight 0.5 --value-weight 0.5)
    "current": "detail_weight=1,scales=4/16,edge_weight=0.5,value_weight=0.5",
    "L1": "detail_weight=1,scales=",
    # loose detail: coarse colour anchor + shift-tolerant fine detail + texture statistics
    "loose": "detail_weight=0,scales=16/32,scale_weights=0.2/1,shift_detail_weight=20,texture_weight=40",
    "tex_only": "detail_weight=0,scales=32,texture_weight=160",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="An example folder, or a folder of example folders")
    p.add_argument("--loss", action="append", default=[], help="Extra loss `name:key=value,...` (repeatable)")
    p.add_argument("--only", action="store_true", help="Only the --loss losses, not the built-in ones")
    p.add_argument("--downscales", default="1,2,4", help="Downscales of the image before cropping")
    p.add_argument("--size", type=int, default=128, help="Crop size")
    p.add_argument("--crops", type=int, default=96, help="Crops per example and downscale")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true",
                   help="Also score two made-up bad edits over the edited areas: `noise` (texture replaced by "
                        "noise of the same local energy), `mush` (a strong blur) and `stray` (small dabs in "
                        "flat areas, colored from up to 16 px away: busy strokes that average out)")
    p.add_argument("--quiet", action="store_true", help="Only print the summary")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def parse_loss(spec: str) -> MultiScalePaintLoss:
    kwargs = {}
    for item in filter(None, spec.split(",")):
        key, value = item.split("=", 1)
        if key in ("scales", "scale_weights", "texture_bands"):
            kwargs[key] = [float(v) if key != "scales" else int(v) for v in value.split("/") if v]
        elif key in ("simplify_size", "simplify_passes", "shift_radius"):
            kwargs[key] = int(value)
        else:
            kwargs[key] = float(value)
    return MultiScalePaintLoss(**kwargs)


def load(path: Path, device) -> torch.Tensor:
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(img).permute(2, 0, 1)[None].to(device)


def find_examples(folder: Path) -> List[Path]:
    has_original = lambda d: any(f.stem == "original" and f.suffix.lower() in IMAGE_EXTS for f in d.iterdir())
    examples = [folder] if has_original(folder) else []
    examples += sorted(d for d in folder.rglob("*") if d.is_dir() and has_original(d))
    if not examples:
        raise SystemExit(f"no example in {folder}: a folder needs an original.<ext>")
    return examples


def load_example(folder: Path, device) -> Tuple[torch.Tensor, Dict[str, Tuple[str, torch.Tensor]]]:
    """(original, {edit name: (label, image)})"""
    labels_file = folder / "labels.json"
    labels = json.loads(labels_file.read_text()) if labels_file.exists() else {}
    original, edits, unlabeled = None, {}, []
    one_good = not labels and any(f.stem == "good" and f.suffix.lower() in IMAGE_EXTS for f in folder.iterdir())
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        if f.stem == "original":
            original = load(f, device)
            continue
        label = labels.get(f.stem) or labels.get(f.name)
        if label is None and one_good:
            label = "good" if f.stem == "good" else "bad"
        if label is None:
            label = next((p for p in ("good", "bad") if f.stem.startswith(p + "_")), None)
        if label not in ("good", "bad"):
            unlabeled.append(f.name)
            continue
        edits[f.stem] = (label, load(f, device))
    if unlabeled:
        raise SystemExit(f"{folder}: no good / bad label for {', '.join(unlabeled)} (labels.json or a "
                         f"good_ / bad_ prefix)")
    for name, (_, img) in edits.items():
        if img.shape != original.shape:
            raise SystemExit(f"{folder}/{name}: size {tuple(img.shape[-2:])} differs from the original's "
                             f"{tuple(original.shape[-2:])}")
    if not any(l == "good" for l, _ in edits.values()) or not any(l == "bad" for l, _ in edits.values()):
        raise SystemExit(f"{folder}: needs at least one good and one bad edit")
    return original, edits


def add_synthetic(original: torch.Tensor, edits: Dict[str, Tuple[str, torch.Tensor]], changed: torch.Tensor):
    """Made-up bad edits over the edited areas, on top of the first good edit."""
    base = next(img for label, img in edits.values() if label == "good")
    area = (_gaussian_blur(changed, 8) > 0.2).float()
    soft = _gaussian_blur(original, 1.5)
    energy = _gaussian_blur((original - soft) ** 2, 4).sqrt()
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(original.shape, generator=g).to(original)
    noise = noise - _gaussian_blur(noise, 1.5)
    noise = noise / noise.std()
    edits["noise"] = ("bad", area * (soft + energy * noise).clamp(0, 1) + (1 - area) * base)
    edits["mush"] = ("bad", area * _gaussian_blur(original, 3) + (1 - area) * base)
    edits["stray"] = ("bad", stray_dabs(original, base, g))


def stray_dabs(original: torch.Tensor, base: torch.Tensor, g: torch.Generator, reach: int = 16) -> torch.Tensor:
    """`base` with round 2-5 px dabs in the original's flat areas (~25% of them), each colored like the
    original `reach` px or less away."""
    H, W = original.shape[-2:]
    soft = _gaussian_blur(original, 1.5)
    flat = (_gaussian_blur((original - soft).abs().mean(1, keepdim=True), 3) < 0.01)[0, 0].cpu()
    ys, xs = flat.nonzero(as_tuple=True)
    n = int(flat.sum().item() * 0.25 / (math.pi * 3.5**2))
    if n == 0:
        return base
    pick = torch.randint(len(ys), (n,), generator=g)
    cy, cx = ys[pick].float(), xs[pick].float()
    r = 2 + 3 * torch.rand(n, generator=g)
    oy, ox = ((torch.rand(2, n, generator=g) * 2 - 1) * reach).round()
    sy = (cy + oy).clamp(0, H - 1).long()
    sx = (cx + ox).clamp(0, W - 1).long()
    color = original[0, :, sy.to(original.device), sx.to(original.device)].T.cpu()  # (n, 3)
    out = base.clone().cpu()
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    for i in range(n):
        y0, y1 = int(max(0, cy[i] - 6)), int(min(H, cy[i] + 7))
        x0, x1 = int(max(0, cx[i] - 6)), int(min(W, cx[i] + 7))
        d = ((yy[y0:y1, x0:x1] - cy[i]) ** 2 + (xx[y0:y1, x0:x1] - cx[i]) ** 2).sqrt()
        m = (r[i] + 0.5 - d).clamp(0, 1)
        out[0, :, y0:y1, x0:x1] = out[0, :, y0:y1, x0:x1] * (1 - m) + m * color[i].view(3, 1, 1)
    return out.to(base.device)


def crops(original, edits, downscale: int, size: int, n: int, seed: int, changed):
    """n crops of size px of the downscaled images, centred on edited areas: (original, {name: crops})"""
    down = (lambda x: F.interpolate(x, scale_factor=1 / downscale, mode="area")) if downscale > 1 else (lambda x: x)
    o, c = down(original), down(changed)
    if min(o.shape[-2:]) < size:
        return None
    weight = F.avg_pool2d(c, size, 1)[0, 0]  # edited share of each crop position
    p = (weight.flatten() + 1e-3).cpu()
    idx = torch.multinomial(p / p.sum(), n, replacement=True, generator=torch.Generator().manual_seed(seed))
    ys, xs = (idx // weight.shape[1]).tolist(), (idx % weight.shape[1]).tolist()
    cut = lambda x: torch.cat([x[..., y:y + size, x0:x0 + size] for y, x0 in zip(ys, xs)])
    return cut(o), {name: cut(down(img)) for name, (_, img) in edits.items()}


@torch.no_grad()
def main():
    args = parse_args()
    dev = torch.device(args.device)
    losses = {} if args.only else {k: parse_loss(v) for k, v in LOSSES.items()}
    for spec in args.loss:
        name, _, rest = spec.partition(":")
        losses[name] = parse_loss(rest)
    downscales = [int(s) for s in args.downscales.split(",") if s]

    worst = {name: float("inf") for name in losses}
    right = {name: [] for name in losses}
    say = (lambda *a: None) if args.quiet else print
    root = Path(args.folder).expanduser()
    for folder in find_examples(root):
        original, edits = load_example(folder, dev)
        changed = sum(((img - original).abs().mean(1, keepdim=True) > 0.03).float() for _, img in edits.values())
        changed = changed.clamp(max=1.0)
        if args.synthetic:
            add_synthetic(original, edits, changed)
        good = [n for n, (l, _) in edits.items() if l == "good"]
        bad = [n for n, (l, _) in edits.items() if l == "bad"]
        say(f"\n{folder.relative_to(root) if folder != root else folder.name}: good {', '.join(good)} | ratio = bad / worst good, above 1 = ranked right")
        sets = {s: crops(original, edits, s, args.size, args.crops, args.seed, changed) for s in downscales}
        sets = {s: v for s, v in sets.items() if v is not None}
        say(f"{'':12s}" + "".join(f" | x{s}: " + " ".join(f"{b[:8]:>8s}" for b in bad) for s in sets))
        for name, fn in losses.items():
            row = f"{name[:12]:12s}"
            for s, (o, e) in sets.items():
                score = {k: fn(x, o, per_sample=True)[0].mean().item() for k, x in e.items()}
                ref = max(score[g] for g in good)
                ratios = [score[b] / ref for b in bad]
                worst[name] = min(worst[name], *ratios)
                right[name] += [r > 1 for r in ratios]
                row += " |     " + " ".join(f"{r:8.2f}" for r in ratios)
            say(row)
    print("\nworst ratio over all examples, downscales and bad edits (above 1 = every ranking right), "
          "and the share of rankings right:")
    for name, w in worst.items():
        print(f"  {name:12s} {w:5.2f}  {100 * sum(right[name]) / len(right[name]):4.0f}%")


if __name__ == "__main__":
    main()
