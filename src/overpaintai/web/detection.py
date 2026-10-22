"""
Lightweight local mask generation for the web UI.

Detects subjects, faces, and people using lightweight neural networks running
locally on CPU or GPU:
  - Face detection: BlazeFace (Google Research / MediaPipe architecture in PyTorch,
    ~0.4 MB weights, sub-10ms inference).
  - People / semantic foreground segmentation: MobileNetV3-LRASPP (TorchVision,
    12.5 MB weights, fast segmentation on VOC foreground and person classes).
  - Visual saliency: Spectral residual frequency-domain saliency for general/art subjects.

All methods include graceful fallback to soft central / upper-middle priors on blank
or untextured images.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from scipy.signal import convolve2d
from torch import nn

# Cached models keyed by device string
_BLAZEFACE_MODELS: dict[str, BlazeFace | None] = {}
_LRASPP_MODELS: dict[str, nn.Module | None] = {}

ASSETS_DIR = Path(__file__).parent / "assets"
BLAZEFACE_WEIGHTS_PATH = ASSETS_DIR / "blazeface.pth"
BLAZEFACE_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/hollance/BlazeFace-PyTorch/master/blazeface.pth"
)


# --- BlazeFace Architecture ----------------------------------------------------


class BlazeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class BlazeFace(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.num_anchors = 896

        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),
            BlazeBlock(24, 24),
            BlazeBlock(24, 28),
            BlazeBlock(28, 32, stride=2),
            BlazeBlock(32, 36),
            BlazeBlock(36, 42),
            BlazeBlock(42, 48, stride=2),
            BlazeBlock(48, 56),
            BlazeBlock(56, 64),
            BlazeBlock(64, 72),
            BlazeBlock(72, 80),
            BlazeBlock(80, 88),
        )
        self.backbone2 = nn.Sequential(
            BlazeBlock(88, 96, stride=2),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
        )
        self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
        self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)
        self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
        self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)

        anchors = []
        for _, count, num_anchors in [(16, 16, 2), (8, 8, 6)]:
            for y in range(count):
                for x in range(count):
                    cx = (x + 0.5) / count
                    cy = (y + 0.5) / count
                    for _ in range(num_anchors):
                        anchors.append([cx, cy, 1.0, 1.0])
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        b = x.shape[0]

        x1 = self.backbone1(x)
        x2 = self.backbone2(x1)

        c1 = self.classifier_8(x1).permute(0, 2, 3, 1).reshape(b, 512, 1)
        c2 = self.classifier_16(x2).permute(0, 2, 3, 1).reshape(b, 384, 1)
        c = torch.cat((c1, c2), dim=1)

        r1 = self.regressor_8(x1).permute(0, 2, 3, 1).reshape(b, 512, 16)
        r2 = self.regressor_16(x2).permute(0, 2, 3, 1).reshape(b, 384, 16)
        r = torch.cat((r1, r2), dim=1)
        return r, c


def _intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    a = box_a.size(0)
    b = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:4].unsqueeze(1).expand(a, b, 2), box_b[:, 2:4].unsqueeze(0).expand(a, b, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(a, b, 2), box_b[:, :2].unsqueeze(0).expand(a, b, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def _jaccard(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    inter = _intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-6)


def get_blazeface(device: torch.device) -> BlazeFace | None:
    """Loads BlazeFace, checking local assets or torch cache, falling back to download if needed."""
    dev_str = str(device)
    if dev_str in _BLAZEFACE_MODELS:
        return _BLAZEFACE_MODELS[dev_str]

    weights_path = BLAZEFACE_WEIGHTS_PATH
    if not weights_path.exists():
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        cache_path = cache_dir / "blazeface.pth"
        if cache_path.exists():
            weights_path = cache_path
        else:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                req = urllib.request.Request(BLAZEFACE_DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    cache_path.write_bytes(resp.read())
                weights_path = cache_path
            except (OSError, urllib.error.URLError):
                _BLAZEFACE_MODELS[dev_str] = None
                return None

    try:
        net = BlazeFace()
        state = torch.load(weights_path, map_location=device, weights_only=True)
        net.load_state_dict(state, strict=False)
        net.to(device)
        net.eval()
        _BLAZEFACE_MODELS[dev_str] = net
        return net
    except (OSError, RuntimeError):
        _BLAZEFACE_MODELS[dev_str] = None
        return None


def detect_faces(
    net: BlazeFace,
    img: Image.Image,
    score_threshold: float = 0.65,
    iou_threshold: float = 0.3,
) -> list[dict[str, any]]:
    """Detects faces in an arbitrary image using BlazeFace with letterbox padding."""
    w, h = img.size
    scale = 128.0 / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR)

    pad_x = (128 - nw) // 2
    pad_y = (128 - nh) // 2

    square = Image.new("RGB", (128, 128), (0, 0, 0))
    square.paste(resized, (pad_x, pad_y))

    device = next(net.parameters()).device
    t = torch.from_numpy(np.array(square)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    t = t.to(device)

    with torch.no_grad():
        r, c = net(t)
        scores = c.clamp(-100.0, 100.0).sigmoid().squeeze(-1)[0]
        mask = scores >= score_threshold
        if not mask.any():
            return []

        raw_boxes = r[0, mask]
        sub_scores = scores[mask].unsqueeze(-1)
        anchors = net.anchors[mask]

        x_center = raw_boxes[:, 0] / 128.0 * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[:, 1] / 128.0 * anchors[:, 3] + anchors[:, 1]
        bw = raw_boxes[:, 2] / 128.0 * anchors[:, 2]
        bh = raw_boxes[:, 3] / 128.0 * anchors[:, 3]

        ymin = (y_center - bh / 2.0) * 128.0
        xmin = (x_center - bw / 2.0) * 128.0
        ymax = (y_center + bh / 2.0) * 128.0
        xmax = (x_center + bw / 2.0) * 128.0

        dets = torch.cat([torch.stack([ymin, xmin, ymax, xmax], dim=-1), sub_scores], dim=-1)

        remaining = torch.argsort(dets[:, 4], descending=True)
        results = []
        while len(remaining) > 0:
            best = dets[remaining[0]]
            ious = _jaccard(best[:4].unsqueeze(0), dets[remaining, :4]).squeeze(0)
            overlap = remaining[ious > iou_threshold]
            remaining = remaining[ious <= iou_threshold]

            box = best[:4]
            if len(overlap) > 1:
                w_coords = dets[overlap, :4]
                w_scores = dets[overlap, 4:5]
                box = (w_coords * w_scores).sum(dim=0) / w_scores.sum().clamp(min=1e-6)

            orig_ymin = (box[0].item() - pad_y) / scale
            orig_xmin = (box[1].item() - pad_x) / scale
            orig_ymax = (box[2].item() - pad_y) / scale
            orig_xmax = (box[3].item() - pad_x) / scale

            results.append({
                "box": (max(0.0, orig_xmin), max(0.0, orig_ymin), min(float(w), orig_xmax), min(float(h), orig_ymax)),
                "score": best[4].item(),
            })
    return results


# --- Segmentation & Saliency ----------------------------------------------------


def get_lraspp(device: torch.device) -> nn.Module | None:
    """Loads LRASPP MobileNetV3 Large from torchvision, caching the model."""
    dev_str = str(device)
    if dev_str in _LRASPP_MODELS:
        return _LRASPP_MODELS[dev_str]

    try:
        import torchvision.models.segmentation as seg

        weights = seg.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        model = seg.lraspp_mobilenet_v3_large(weights=weights).to(device)
        model.eval()
        _LRASPP_MODELS[dev_str] = model
        return model
    except (OSError, RuntimeError, ImportError):
        _LRASPP_MODELS[dev_str] = None
        return None


def run_lraspp_segmentation(model: nn.Module, img: Image.Image, device: torch.device) -> torch.Tensor:
    """Runs LRASPP and returns softmax probabilities of shape (21, H, W)."""
    w, h = img.size
    # Downscale for fast segmentation while preserving aspect ratio
    max_dim = 512
    scale = min(1.0, max_dim / max(w, h))
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    small = img.convert("RGB").resize((sw, sh), Image.Resampling.BILINEAR)

    import torchvision.transforms.functional as tf

    t = tf.to_tensor(small).unsqueeze(0).to(device)
    norm = tf.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        logits = model(norm)["out"]
        probs = torch.softmax(logits, dim=1)
        # Interpolate back to original image size
        probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)[0]
    return probs


def visual_saliency(img: Image.Image) -> np.ndarray:
    """Computes spatial background contrast combined with frequency-domain spectral residual saliency."""
    w, h = img.size
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    gray = np.array(img.convert("L"), dtype=np.float32)
    if float(gray.std()) < 5.0:
        return np.zeros((h, w), dtype=np.float32)

    # 1. Background border distance
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0)
    bg_color = np.median(border, axis=0)
    color_dist = np.linalg.norm(arr - bg_color, axis=-1)
    c_diff = float(color_dist.max() - color_dist.min())
    c_norm = (color_dist - color_dist.min()) / c_diff if c_diff > 5.0 else np.zeros((h, w), dtype=np.float32)

    # 2. Spectral residual frequency-domain analysis
    small_size = 64
    small = np.array(img.convert("L").resize((small_size, small_size), Image.Resampling.BILINEAR), dtype=np.float32)
    f = np.fft.fft2(small)
    mag = np.abs(f)
    phase = np.angle(f)
    log_mag = np.log(mag + 1e-6)

    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    avg_log_mag = convolve2d(log_mag, kernel, mode="same", boundary="symm")
    spectral_residual = np.exp(log_mag - avg_log_mag)

    f_res = spectral_residual * np.exp(1j * phase)
    sal = np.abs(np.fft.ifft2(f_res)) ** 2
    s_diff = float(sal.max() - sal.min())
    if s_diff > 1e-6:
        sal_norm = (sal - sal.min()) / s_diff
        sal_img = Image.fromarray((sal_norm * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        sal_map = np.array(sal_img, dtype=np.float32) / 255.0
    else:
        sal_map = np.zeros((h, w), dtype=np.float32)

    combined = 0.6 * c_norm + 0.4 * sal_map
    comb_img = Image.fromarray((combined * 255).astype(np.uint8))
    comb_img = comb_img.filter(ImageFilter.GaussianBlur(radius=max(2.0, 0.03 * min(w, h))))
    res = np.array(comb_img, dtype=np.float32) / 255.0
    res_diff = float(res.max() - res.min())
    return (res - res.min()) / res_diff if res_diff > 0.05 else np.zeros((h, w), dtype=np.float32)


# --- Mask Generators -----------------------------------------------------------


def generate_fallback_mask(size: tuple[int, int], mode: str) -> Image.Image:
    """Creates a smooth prior ellipse mask (fallback when no detector fires)."""
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if mode == "face":
        cx, cy, rx, ry = 0.5 * w, 0.38 * h, 0.18 * w, 0.2 * h
    else:
        cx, cy, rx, ry = 0.5 * w, 0.5 * h, 0.32 * w, 0.4 * h
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=0.08 * min(w, h)))


def generate_face_mask(img: Image.Image, device: torch.device) -> tuple[Image.Image, bool]:
    """Generates an importance mask focused on human faces or heads."""
    w, h = img.size
    net = get_blazeface(device)
    faces = detect_faces(net, img) if net is not None else []

    if faces:
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for face in faces:
            xmin, ymin, xmax, ymax = face["box"]
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            bw = xmax - xmin
            bh = ymax - ymin
            # Slightly expand to cover hair, jaw, and ears smoothly
            rx = bw * 0.75
            ry = bh * 0.85
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        blur_rad = max(4.0, 0.05 * min(w, h))
        return mask.filter(ImageFilter.GaussianBlur(radius=blur_rad)), False

    # If BlazeFace didn't detect a face, check if LRASPP sees a person and focus on their upper body/head
    lraspp = get_lraspp(device)
    if lraspp is not None:
        try:
            probs = run_lraspp_segmentation(lraspp, img, device)
            person_prob = probs[15].cpu().numpy()  # Class 15: person
            if person_prob.max() > 0.25:
                # Find the bounding box of the detected person
                ys, xs = np.where(person_prob > 0.2)
                if len(ys) > 0:
                    ymin, ymax = float(ys.min()), float(ys.max())
                    xmin, xmax = float(xs.min()), float(xs.max())
                    head_h = (ymax - ymin) * 0.35
                    cx = (xmin + xmax) / 2.0
                    cy = ymin + head_h * 0.5
                    rx = (xmax - xmin) * 0.4
                    ry = head_h * 0.6
                    mask = Image.new("L", (w, h), 0)
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
                    return mask.filter(ImageFilter.GaussianBlur(radius=max(4.0, 0.05 * min(w, h)))), False
        except (RuntimeError, ValueError):
            pass

    return generate_fallback_mask((w, h), "face"), True


def generate_people_mask(img: Image.Image, device: torch.device) -> tuple[Image.Image, bool]:
    """Generates an importance mask segmenting all people/bodies in the image."""
    w, h = img.size
    lraspp = get_lraspp(device)
    if lraspp is not None:
        try:
            probs = run_lraspp_segmentation(lraspp, img, device)
            person_prob = probs[15].cpu().numpy()  # Class 15: person
            if person_prob.max() > 0.18:
                # Smoothly map probabilities to [0, 255]
                p_norm = np.clip((person_prob - 0.12) / 0.55, 0.0, 1.0)
                mask = Image.fromarray((p_norm * 255).astype(np.uint8))
                blur_rad = max(2.0, 0.02 * min(w, h))
                return mask.filter(ImageFilter.GaussianBlur(radius=blur_rad)), False
        except (RuntimeError, ValueError):
            pass

    # Fallback to face detection if full-body segmentation missed it
    net = get_blazeface(device)
    faces = detect_faces(net, img) if net is not None else []
    if faces:
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for face in faces:
            xmin, ymin, xmax, ymax = face["box"]
            cx = (xmin + xmax) / 2.0
            bh = ymax - ymin
            bw = xmax - xmin
            # Upper body below face
            draw.ellipse([cx - bw * 0.8, ymin - bh * 0.2, cx + bw * 0.8, ymax + bh * 1.5], fill=255)
        return mask.filter(ImageFilter.GaussianBlur(radius=max(4.0, 0.06 * min(w, h)))), False

    return generate_fallback_mask((w, h), "subject"), True


def generate_subject_mask(img: Image.Image, device: torch.device) -> tuple[Image.Image, bool]:
    """Generates an importance mask for the main foreground subject."""
    w, h = img.size
    sal = visual_saliency(img)

    lraspp = get_lraspp(device)
    if lraspp is not None:
        try:
            probs = run_lraspp_segmentation(lraspp, img, device)
            # Probability of any foreground object (1 - background probability)
            fg_prob = (1.0 - probs[0]).cpu().numpy()
            if fg_prob.max() > 0.25:
                # Combine semantic foreground with visual saliency
                combined = 0.7 * fg_prob + 0.3 * sal
                c_norm = np.clip((combined - 0.15) / 0.65, 0.0, 1.0)
                mask = Image.fromarray((c_norm * 255).astype(np.uint8))
                blur_rad = max(2.0, 0.025 * min(w, h))
                return mask.filter(ImageFilter.GaussianBlur(radius=blur_rad)), False
        except (RuntimeError, ValueError):
            pass

    # If no semantic VOC class was detected, check if visual saliency has a clear subject
    if sal.max() > 0.35:
        mask = Image.fromarray((sal * 255).astype(np.uint8))
        blur_rad = max(2.0, 0.03 * min(w, h))
        return mask.filter(ImageFilter.GaussianBlur(radius=blur_rad)), False

    return generate_fallback_mask((w, h), "subject"), True


def generate_auto_mask(
    img: Image.Image,
    mode: str,
    device: torch.device | None = None,
) -> tuple[Image.Image, bool]:
    """
    Public entry point for automatic importance mask generation.

    Args:
        img: Input PIL Image.
        mode: Detection mode: "subject", "face", "people", or "person".
        device: Torch device to run detection models on (defaults to CPU/CUDA).

    Returns:
        (mask: PIL.Image in mode 'L', is_fallback: bool)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    m = mode.strip().lower()
    if m == "face":
        return generate_face_mask(img, device)
    if m in ("people", "person"):
        return generate_people_mask(img, device)
    if m == "subject":
        return generate_subject_mask(img, device)

    raise ValueError(f"Unknown mode: {mode!r}. Must be 'subject', 'face', or 'people'.")
