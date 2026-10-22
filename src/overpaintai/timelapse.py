"""
Speed-painting videos: replays recorded strokes (`highres.pack_strokes` format, image coordinates) at video
resolution and encodes the frames to MP4 with ffmpeg.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

from overpaintai.highres import CropRenderer, stroke_groups, unpack_strokes


def stroke_schedule(n_strokes: int, n_frames: int, gamma: float = 1.8) -> list:
    """
    Number of strokes visible in each frame: N * t^gamma for t from 0 to 1. With gamma > 1 the first (big,
    composition-defining) strokes get more screen time and the many small detail strokes go by quickly.
    """
    return [round(n_strokes * ((f + 1) / n_frames) ** gamma) for f in range(n_frames)]


def render_timelapse(
    strokes: torch.Tensor,
    canvas_size: int,
    content_wh: Tuple[int, int],
    out_path: Path,
    duration: float = 15.0,
    fps: int = 30,
    max_side: int = 1080,
    hold: float = 1.5,
    num_stamps: int = 32,
    min_radius: float = 1e-4,
    max_radius: float = 0.75,
    device="cpu",
    on_progress: Optional[Callable[[float], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    gamma: float = 1.8,
    groups: Optional[List[Tuple[int, bool]]] = None,
) -> Path:
    """
    Args:
        strokes: (N, 16) (or older (N, 14)) strokes in the painting order, coordinates of the square `canvas_size` canvas.
        content_wh: (w, h) of the image content on that canvas (the rest is padding and is cropped off).
        duration: video length in seconds, including `hold` seconds on the finished painting.
        max_side: longer side of the video in pixels (strokes are re-rendered at that resolution).
        on_progress: called with the fraction of frames done.
        should_stop: checked every frame; returning True aborts (raises InterruptedError).
        groups: (count, joint) per recorded render (`highres.stroke_groups`); default: chunks of 16, one stroke
            after another. A joint group (one order-independent operation) only appears as a whole.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed (needed to encode the video)")
    w, h = content_wh
    # Strokes are resolution-independent: render the content at max_side px, whatever the canvas size was
    V = max(16, round(canvas_size * max_side / max(w, h)))
    vw, vh = round(w * V / canvas_size) // 2 * 2, round(h * V / canvas_size) // 2 * 2  # even for yuv420p

    renderer = CropRenderer(V, num_stamps, min_radius, max_radius, device, adaptive_stamps=True)
    strokes = strokes.to(renderer.device).float()
    N = strokes.shape[0]
    hold_frames = max(1, round(hold * fps))
    paint_frames = max(1, round(duration * fps) - hold_frames)
    schedule = stroke_schedule(N, paint_frames, gamma)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{vw}x{vh}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-preset", "medium", "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(layer: torch.Tensor, repeat: int = 1):
        frame = renderer.composite(layer)[0, :, :vh, :vw]
        data = (frame.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy().tobytes()
        for _ in range(repeat):
            proc.stdin.write(data)

    try:
        layer = renderer.blank()
        done = 0  # strokes rendered
        if groups is None:  # tiles' strokes are stored in groups of <= 16: rendering them together keeps crops local
            groups = stroke_groups({"strokes": strokes})
        g, g_done = 0, 0  # current group, strokes of it rendered
        for f, target_count in enumerate(schedule):
            while done < target_count and g < len(groups):
                size, joint = groups[g]
                n = size - g_done if joint else min(size - g_done, target_count - done)
                renderer.render(layer, unpack_strokes(strokes[done : done + n]), joint)
                done, g_done = done + n, g_done + n
                if g_done == size:
                    g, g_done = g + 1, 0
            write(layer)
            if should_stop is not None and should_stop():
                raise InterruptedError("video export cancelled")
            if on_progress is not None and (f % 10 == 0 or f == paint_frames - 1):
                on_progress((f + 1) / (paint_frames + hold_frames))
        write(layer, hold_frames)
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.read().decode(errors='replace')[-500:]}")
    except BaseException:
        proc.kill()
        proc.wait()
        Path(out_path).unlink(missing_ok=True)
        raise
    if on_progress is not None:
        on_progress(1.0)
    return Path(out_path)

