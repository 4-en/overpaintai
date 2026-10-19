"""Training data: synthetic stroke paintings (with ground-truth strokes) and image folders."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.synthetic import random_strokes, render_targets

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------------------------------------------
# Synthetic paintings
# ----------------------------------------------------------------------------------------------------
@dataclass
class SyntheticBatch:
    target: torch.Tensor  # (B, 3, S, S)
    params: torch.Tensor  # (B, M, 14) ground-truth engine params, painting order
    tools: torch.Tensor  # (B, M, 5) one-hot


class SyntheticSource:
    """
    Generates batches of synthetic paintings on the fly (on the training device): random coarse-to-fine
    stroke sequences rendered by the engine, so the ground-truth strokes are known (teacher training).

    Args:
        stroke_range: (min, max) strokes per painting; each batch draws one count in that range, rounded
            to a multiple of `multiple_of` (the model's strokes per step), so it splits into whole steps.
        tool_probs: probability of each engine tool in the ground truth.
        washes: (min, max) large background washes at the start of each painting.
    """

    def __init__(
        self,
        engine: DifferentiableCanvasEngine,
        image_size: int,
        stroke_range: Tuple[int, int] = (48, 96),
        multiple_of: int = 16,
        tool_probs: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 0.0),
        radius_range: Tuple[float, float] = (0.12, 0.012),
        washes: Tuple[int, int] = (1, 3),
    ):
        self.engine = engine
        self.image_size = image_size
        self.stroke_range = stroke_range
        self.multiple_of = multiple_of
        self.tool_probs = tool_probs
        self.radius_range = radius_range
        self.washes = washes

    def sample(self, batch_size: int, generator: torch.Generator) -> SyntheticBatch:
        device = self.engine.bg_color.device
        lo, hi = self.stroke_range
        m = int(torch.randint(lo, hi + 1, (1,), generator=generator, device=device).item())
        m = max(self.multiple_of, round(m / self.multiple_of) * self.multiple_of)
        n_washes = int(torch.randint(self.washes[0], self.washes[1] + 1, (1,), generator=generator, device=device))
        params, tools = random_strokes(
            self.engine, batch_size, m, tool_probs=self.tool_probs, radius_range=self.radius_range,
            num_washes=n_washes, generator=generator, device=device,
        )
        target = render_targets(self.engine, params, tools, self.image_size)
        return SyntheticBatch(target=target, params=params, tools=tools)


# ----------------------------------------------------------------------------------------------------
# Image folders
# ----------------------------------------------------------------------------------------------------
def load_image(path: str, size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Loads an image as a `(1, 3, size, size)` tensor in [0, 1]: center-cropped to square, Lanczos-resized."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        side = min(w, h)
        img = img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
        img = img.resize((size, size), Image.LANCZOS)
    x = T.functional.pil_to_tensor(img).float() / 255.0
    return x.unsqueeze(0).to(device)


def list_images(root: str) -> list:
    files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
    if not files:
        raise FileNotFoundError(f"No images ({', '.join(sorted(IMAGE_EXTENSIONS))}) found under {root}")
    return files


def _in_val_split(path: Path, root: Path, val_fraction: float) -> bool:
    """Deterministic split by hashing the relative path, stable when files are added or removed."""
    h = hashlib.md5(str(path.relative_to(root)).encode()).digest()
    return int.from_bytes(h[:4], "little") / 2**32 < val_fraction


class ImageFolderDataset(Dataset):
    """
    Square training crops from all images under `root` (recursive).

    Augmentation (train split): random square crops covering a `crop_scale` fraction of the image area
    (zoom), horizontal / vertical flips, and color jitter. The validation split uses a fixed center
    crop. Images are padded to square only implicitly: crops are always square, cut from the image.
    """

    def __init__(
        self,
        root: str,
        image_size: int,
        split: str = "train",
        val_fraction: float = 0.05,
        augment: bool = True,
        crop_scale: Tuple[float, float] = (0.35, 1.0),
        hflip: float = 0.5,
        vflip: float = 0.0,
        color_jitter: float = 0.1,
        files: Optional[list] = None,
    ):
        assert split in ("train", "val", "all")
        self.root = Path(root)
        all_files = files if files is not None else list_images(root)
        if split == "all":
            self.files = all_files
        else:
            want_val = split == "val"
            self.files = [f for f in all_files if _in_val_split(f, self.root, val_fraction) == want_val]
            if not self.files:  # tiny folders: fall back to using everything for both splits
                self.files = all_files
        self.image_size = image_size

        to_tensor = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
        if augment and split == "train":
            ops = [
                T.RandomResizedCrop(image_size, scale=crop_scale, ratio=(1.0, 1.0), antialias=True),
                T.RandomHorizontalFlip(hflip),
                T.RandomVerticalFlip(vflip),
            ]
            if color_jitter > 0:
                c = color_jitter
                ops.append(T.RandomApply([T.ColorJitter(c, c, c, min(0.5, c / 4))], p=0.8))
            self.transform = T.Compose(to_tensor + ops)
        else:
            self.transform = T.Compose(to_tensor + [T.Resize(image_size, antialias=True), T.CenterCrop(image_size)])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.files[i]) as img:
            img = img.convert("RGB")
        return self.transform(img).clamp(0.0, 1.0)
