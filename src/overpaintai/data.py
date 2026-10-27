"""Training data: synthetic stroke paintings (with ground-truth strokes) and image folders."""

import hashlib
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.synthetic import random_strokes, render_targets

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Our own art folders hold very large pieces (splash arts up to ~100 MP): above Pillow's default decompression-bomb
# limit (89 MP warns, twice that fails). Allow up to ~400 MP; broken EXIF metadata only loses the metadata
Image.MAX_IMAGE_PIXELS = 200_000_000
warnings.filterwarnings("ignore", message="Corrupt EXIF data", category=UserWarning)


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


class LogUniformSquareCrop(torch.nn.Module):
    """
    Random square crop resized to `size`. The crop's side is sampled log-uniformly between sqrt(lo) and
    sqrt(hi) of the image's shorter side (`area_range` = (lo, hi) as fractions of that square's area), so
    every zoom level is equally likely: from native-detail crops (one source pixel per output pixel) to
    the whole image. Uniform-in-area sampling (torchvision's RandomResizedCrop) makes small crops rare.
    """

    def __init__(self, size: int, area_range: Tuple[float, float] = (0.01, 1.0)):
        super().__init__()
        self.size = size
        self.side_range = (math.sqrt(area_range[0]), math.sqrt(area_range[1]))

    def forward(self, img: torch.Tensor) -> torch.Tensor:  # (C, H, W)
        _, H, W = img.shape
        lo, hi = (math.log(v) for v in self.side_range)
        frac = math.exp(lo + (hi - lo) * torch.rand(()).item())
        side = max(1, min(H, W, round(frac * min(H, W))))
        y = int(torch.randint(0, H - side + 1, ()).item())
        x = int(torch.randint(0, W - side + 1, ()).item())
        crop = img[:, y : y + side, x : x + side].unsqueeze(0)
        return F.interpolate(crop, size=(self.size, self.size), mode="bilinear", antialias=True,
                             align_corners=False)[0]


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

    Augmentation (train split): random square crops whose side is log-uniform within `crop_scale` (fractions
    of the area of the shorter side's square, so 0.01 = 10% of the side; see `LogUniformSquareCrop`),
    horizontal / vertical flips, and color jitter. The validation split uses a fixed center
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
                LogUniformSquareCrop(image_size, crop_scale),
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
