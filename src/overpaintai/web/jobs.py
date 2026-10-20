"""
Painting jobs for the web interface: a queue processed by one worker thread (one GPU job at a time), with
progress events and throttled preview images that the HTTP layer streams to the browser.
"""

import csv
import io
import json
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

from overpaintai.highres import (
    HighResPainter,
    PaintCancelled,
    config_from_checkpoint,
    load_checkpoint_model,
    load_padded,
    render_strokes,
)
from overpaintai.timelapse import render_timelapse
from overpaintai.viz import to_uint8

PREVIEW_INTERVAL = 1.0  # seconds between live previews (stage ends always get one)
PREVIEW_MAX_SIDE = 1024


@dataclass
class JobSettings:
    checkpoint: str
    size: int = 1024
    whole_steps: Optional[int] = None  # None: as trained
    rounds: int = 2
    max_tiles: int = 32
    min_tile: Optional[float] = None  # None: stop at the source image's detail
    threshold: float = 0.015
    margin: float = 0.25
    fade: bool = True
    importance_gain: float = 2.0
    num_stamps: int = 32
    tiles: bool = True

    def validate(self):
        self.size = int(min(4096, max(128, self.size)))
        self.rounds = int(min(8, max(1, self.rounds)))
        self.max_tiles = int(min(256, max(1, self.max_tiles)))
        self.num_stamps = int(min(128, max(8, self.num_stamps)))
        if self.whole_steps is not None:
            self.whole_steps = int(min(256, max(0, self.whole_steps)))  # 0: tiles only (e.g. when continuing)
        if self.min_tile is not None and not 0 < self.min_tile <= 0.5:
            raise ValueError("min_tile must be in (0, 0.5]")


@dataclass
class Job:
    id: str
    settings: JobSettings
    out_dir: Path
    status: str = "queued"  # queued | running | done | error | cancelled
    created: float = field(default_factory=time.time)
    error: Optional[str] = None
    events: List[Dict] = field(default_factory=list)
    stats: List[Dict] = field(default_factory=list)
    preview: Optional[bytes] = None
    preview_version: int = 0
    cancel_requested: bool = False
    image_bytes: bytes = b""
    mask_bytes: Optional[bytes] = None
    kind: str = "paint"  # paint | video | export
    parent: Optional[str] = None  # paint job this one continues (paint) or replays (video)
    whole_done: int = 0  # whole-image steps painted so far, including all parents
    video_duration: float = 15.0
    video_max_side: int = 1080
    export_side: int = 4096
    total_strokes: int = 0  # strokes in the painting so far, including all parents

    def push(self, event: Dict):
        event = {"seq": len(self.events), "time": time.time(), **event}
        self.events.append(event)

    def summary(self) -> Dict:
        return {"id": self.id, "kind": self.kind, "parent": self.parent, "status": self.status, "error": self.error,
                "settings": asdict(self.settings), "stats": self.stats, "preview_version": self.preview_version,
                "events": len(self.events)}


# ----------------------------------------------------------------------------------------------------
def list_checkpoints(runs_dir: Path) -> List[Dict]:
    """best.pt / last.pt of every run, with epoch and PSNRs from the run's epochs.csv (cheap: no torch.load)."""
    found = []
    for ckpt in sorted(runs_dir.glob("*/*.pt")):
        if ckpt.name not in ("best.pt", "last.pt") or ckpt.parent.name == "web":
            continue
        info = {"path": str(ckpt), "run": ckpt.parent.name, "name": ckpt.name, "mtime": ckpt.stat().st_mtime}
        epochs = ckpt.parent / "epochs.csv"
        if epochs.exists():
            try:
                with open(epochs) as fh:
                    rows = [r for r in csv.DictReader(fh) if r.get("val_psnr")]
                if rows:
                    row = max(rows, key=lambda r: float(r["val_psnr"])) if ckpt.name == "best.pt" else rows[-1]
                    info["epoch"] = int(float(row["epoch"]))
                    info["val_psnr"] = float(row["val_psnr"])
                    if row.get("preview_psnr"):
                        info["preview_psnr"] = float(row["preview_psnr"])
            except (OSError, ValueError, KeyError):
                pass
        found.append(info)
    return sorted(found, key=lambda c: c["mtime"], reverse=True)


def checkpoint_defaults(path: Path) -> Dict:
    """Default settings for a checkpoint, from its run's args.json (as the harness would derive them)."""
    args_file = path.parent / "args.json"
    args = json.loads(args_file.read_text()) if args_file.exists() else {}
    cfg, view_res, _ = config_from_checkpoint({"args": args})
    return {"whole_steps": cfg.whole_steps, "view_res": view_res, "tile_range": list(cfg.scale_clamp),
            "train_image_size": args.get("image_size"), "data": args.get("data")}


def auto_importance(img: Image.Image, mode: str) -> Image.Image:
    """
    PLACEHOLDER for automatic importance masks (subject / face detection): returns a soft ellipse, wide and
    centered for "subject", smaller and in the upper middle for "face". Replace with a real detector.
    """
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if mode == "face":
        cx, cy, rx, ry = 0.5 * w, 0.38 * h, 0.18 * w, 0.2 * h
    else:
        cx, cy, rx, ry = 0.5 * w, 0.5 * h, 0.32 * w, 0.4 * h
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=0.08 * min(w, h)))


def _encode_preview(canvas: torch.Tensor, wh) -> bytes:
    w, h = wh
    img = canvas[..., :h, :w]
    scale = PREVIEW_MAX_SIDE / max(w, h)
    if scale < 1:
        img = F.interpolate(img, size=(max(1, round(h * scale)), max(1, round(w * scale))), mode="area")
    buf = io.BytesIO()
    Image.fromarray(to_uint8(img[0])).save(buf, format="JPEG", quality=88)
    return buf.getvalue()


# ----------------------------------------------------------------------------------------------------
class JobManager:
    def __init__(self, runs_dir: Path, out_dir: Path, device: torch.device):
        self.runs_dir = runs_dir.resolve()
        self.out_dir = out_dir
        self.device = device
        self.jobs: Dict[str, Job] = {}
        self.queue: "queue.Queue[str]" = queue.Queue()
        self._model = None  # (checkpoint path, model, checkpoint dict)
        self._worker = threading.Thread(target=self._work, daemon=True, name="overpaint-worker")
        self._worker.start()

    def resolve_checkpoint(self, path: str) -> Path:
        p = Path(path).resolve()
        if not p.is_relative_to(self.runs_dir) or p.suffix != ".pt" or not p.exists():
            raise ValueError(f"checkpoint must be an existing .pt file inside {self.runs_dir}")
        return p

    def _finished_paint_job(self, job_id: str) -> Job:
        parent = self.jobs.get(job_id)
        if parent is None or parent.kind != "paint" or parent.status != "done":
            raise ValueError("the job to continue from must be a finished painting")
        return parent

    def _enqueue(self, job: Job) -> Job:
        self.jobs[job.id] = job
        job.push({"type": "status", "status": "queued", "position": self.queue.qsize()})
        self.queue.put(job.id)
        return job

    def submit(
        self, image_bytes: Optional[bytes], mask_bytes: Optional[bytes], settings: JobSettings,
        parent: Optional[str] = None,
    ) -> Job:
        """
        Queues a painting. With `parent` (a finished painting job) it continues that painting: same image and
        resolution, starting from its paint layer, so further steps / finer tiles add to it.
        """
        settings.validate()
        settings.checkpoint = str(self.resolve_checkpoint(settings.checkpoint))
        if parent is not None:
            source = self._finished_paint_job(parent)
            settings.size = source.settings.size
            image_bytes = (source.out_dir / "source.img").read_bytes()
        if not image_bytes:
            raise ValueError("an image is required")
        Image.open(io.BytesIO(image_bytes)).verify()  # reject non-images early
        if mask_bytes:
            Image.open(io.BytesIO(mask_bytes)).verify()
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id, settings, self.out_dir / job_id, image_bytes=image_bytes, mask_bytes=mask_bytes or None,
                  parent=parent)
        return self._enqueue(job)

    def submit_export(self, paint_job: str, side: int) -> Job:
        """Queues a re-render of a finished painting's strokes at another resolution (longer side `side`)."""
        source = self._finished_paint_job(paint_job)
        side = int(min(8192, max(256, side)))
        job = Job(uuid.uuid4().hex[:12], source.settings, source.out_dir, kind="export", parent=paint_job,
                  export_side=side)
        return self._enqueue(job)

    def submit_video(self, paint_job: str, duration: float, max_side: int = 1080) -> Job:
        """Queues a speed-painting video of a finished painting (including everything it continued)."""
        source = self._finished_paint_job(paint_job)
        duration = float(min(60.0, max(3.0, duration)))
        # The video is written next to the painting it shows
        job = Job(uuid.uuid4().hex[:12], source.settings, source.out_dir, kind="video", parent=paint_job,
                  video_duration=duration, video_max_side=int(min(2160, max(128, max_side))))
        return self._enqueue(job)

    def cancel(self, job: Job):
        job.cancel_requested = True
        if job.status == "queued":
            job.status = "cancelled"
            job.push({"type": "status", "status": "cancelled"})

    # --- worker -------------------------------------------------------------------------------------
    def _work(self):
        while True:
            job = self.jobs.get(self.queue.get())
            if job is None or job.status != "queued":
                continue
            job.status = "running"
            job.push({"type": "status", "status": "running"})
            try:
                {"video": self._run_video, "export": self._run_export}.get(job.kind, self._run)(job)
                job.status = "done"
                job.push({"type": "status", "status": "done", "stats": job.stats, "total_strokes": job.total_strokes})
            except (PaintCancelled, InterruptedError):
                job.status = "cancelled"
                job.push({"type": "status", "status": "cancelled"})
            except Exception as exc:  # report any failure to the browser instead of killing the worker
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.push({"type": "status", "status": "error", "error": job.error})
            finally:
                job.image_bytes, job.mask_bytes = b"", None
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()  # give memory back (e.g. to a training run on the same GPU)

    def _load_model(self, path: str):
        if self._model is None or self._model[0] != path:
            self._model = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            model, ckpt = load_checkpoint_model(path, self.device)
            ckpt = {k: ckpt[k] for k in ("args", "view_res", "tools", "model_config", "epoch") if k in ckpt}
            self._model = (path, model, ckpt)
        return self._model[1], self._model[2]

    @torch.no_grad()
    def _run(self, job: Job):
        s = job.settings
        model, ckpt = self._load_model(s.checkpoint)
        cfg, view_res, tool = config_from_checkpoint(
            ckpt,
            whole_steps=s.whole_steps,
            rounds_per_level=s.rounds if s.tiles else 0,
            max_tiles=s.max_tiles,
            margin=s.margin,
            error_threshold=s.threshold,
            min_core=s.min_tile,
            fade=s.fade,
            importance_gain=s.importance_gain,
        )
        painter = HighResPainter(model, s.size, view_res, tool, s.num_stamps, cfg, device=self.device)
        target, valid, wh, src_side = load_padded(job.image_bytes, s.size, self.device)
        importance = None
        if job.mask_bytes:
            importance, _, _, _ = load_padded(job.mask_bytes, s.size, self.device, mode="L")
            importance = importance * valid

        initial_layer, prior_strokes, whole_done = None, None, 0
        if job.parent is not None:  # continue a finished painting
            parent = self.jobs[job.parent]
            initial_layer = torch.load(parent.out_dir / "layer.pt", map_location=self.device).float()
            prior_strokes = torch.load(parent.out_dir / "strokes.pt", weights_only=False)["strokes"]
            whole_done = parent.whole_done

        levels = [f"1/{round(1 / c)}" for c in painter.core_sizes(src_side)] if s.tiles else []
        job.push({"type": "plan", "whole_steps": cfg.whole_steps, "levels": levels, "rounds": cfg.rounds_per_level,
                  "size": s.size, "content": list(wh), "source_side": src_side, "continues": job.parent})
        last_preview = [0.0]

        def on_progress(event: Dict, get_canvas):
            stage_end = (event["kind"] == "whole" and event["step"] == event["steps"]) or (
                event["kind"] == "tiles" and event["round"] == event["rounds"])
            now = time.time()
            if stage_end or now - last_preview[0] >= PREVIEW_INTERVAL:
                job.preview = _encode_preview(get_canvas(), wh)
                job.preview_version += 1
                event["preview"] = job.preview_version
                last_preview[0] = now
            job.push({"type": "progress", **event})

        def on_stage(stat: Dict):
            job.stats.append(stat)
            job.push({"type": "stage", **stat})

        job.stats = []
        result = painter.paint(
            target, src_side, valid=valid, importance=importance, log=lambda *_: None,
            on_progress=on_progress, should_stop=lambda: job.cancel_requested, on_stage=on_stage,
            initial_layer=initial_layer, whole_done=whole_done,
        )
        job.stats = result.stats
        job.whole_done = whole_done + cfg.whole_steps
        w, h = wh
        out = job.out_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "source.img").write_bytes(job.image_bytes)  # for continuing this painting later
        torch.save(result.layer.half().cpu(), out / "layer.pt")
        Image.fromarray(to_uint8(result.canvas[0, :, :h, :w])).save(out / "final.png")
        Image.fromarray(to_uint8(target[0, :, :h, :w])).save(out / "target.png")
        new = torch.cat(result.strokes) if result.strokes else torch.zeros(0, 14)
        strokes = new if prior_strokes is None else torch.cat([prior_strokes, new])  # the whole history
        job.total_strokes = int(strokes.shape[0])
        torch.save({"strokes": strokes, "size": s.size, "content_wh": wh, "num_stamps": s.num_stamps,
                    "min_radius": painter.min_radius, "max_radius": painter.max_radius,
                    "format": "points(6), radii(2), opacity(2), hardness, color(3); image coords"}, out / "strokes.pt")
        (out / "stats.json").write_text(json.dumps({"settings": asdict(s), "stages": result.stats}, indent=2))
        job.preview = _encode_preview(result.canvas, wh)
        job.preview_version += 1
        job.push({"type": "preview", "preview": job.preview_version})

    @torch.no_grad()
    def _run_video(self, job: Job):
        data = torch.load(job.out_dir / "strokes.pt", weights_only=False)
        n = data["strokes"].shape[0]
        job.push({"type": "plan", "video": True, "strokes": n, "duration": job.video_duration})
        start = time.time()

        def on_progress(fraction: float):
            job.push({"type": "progress", "kind": "video", "stage": "rendering video", "fraction": fraction,
                      "strokes": n, "seconds": time.time() - start})

        render_timelapse(
            data["strokes"], data["size"], tuple(data["content_wh"]), job.out_dir / "timelapse.mp4",
            duration=job.video_duration, max_side=job.video_max_side, num_stamps=data.get("num_stamps", 32),
            min_radius=data.get("min_radius", 1e-4), max_radius=data.get("max_radius", 0.75), device=self.device,
            on_progress=on_progress, should_stop=lambda: job.cancel_requested,
        )

    @torch.no_grad()
    def _run_export(self, job: Job):
        data = torch.load(job.out_dir / "strokes.pt", weights_only=False)
        n = data["strokes"].shape[0]
        job.push({"type": "plan", "export": True, "strokes": n, "side": job.export_side})
        start = time.time()

        def on_progress(fraction: float):
            job.push({"type": "progress", "kind": "export", "stage": f"rendering {job.export_side}px",
                      "fraction": fraction, "strokes": n, "seconds": time.time() - start})

        img = render_strokes(
            data["strokes"], data["size"], tuple(data["content_wh"]), job.export_side,
            num_stamps=data.get("num_stamps", 32), min_radius=data.get("min_radius", 1e-4),
            max_radius=data.get("max_radius", 0.75), device=self.device,
            on_progress=on_progress, should_stop=lambda: job.cancel_requested,
        )
        arr = to_uint8(img[0])
        del img
        Image.fromarray(arr).save(job.out_dir / f"export_{job.export_side}.png", compress_level=3)
