"""
Web interface: FastAPI backend + plain HTML/CSS/JS frontend (static/) for painting images with the
high-resolution harness.

  overpaint-web --runs runs --port 8000          # then open http://127.0.0.1:8000
"""

import argparse
import asyncio
import io
import json
from dataclasses import fields
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from overpaintai.web.jobs import JobManager, JobSettings, auto_importance, checkpoint_defaults, list_checkpoints

STATIC = Path(__file__).parent / "static"


def create_app(runs_dir: str = "runs", device: str = None, out_dir: str = None) -> FastAPI:
    runs = Path(runs_dir)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    manager = JobManager(runs, Path(out_dir) if out_dir else runs / "web", dev)
    app = FastAPI(title="Overpaint AI")
    app.state.manager = manager
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def get_job(job_id: str):
        job = manager.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info():
        name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
        return {"device": str(dev), "device_name": name, "runs": str(runs)}

    @app.get("/api/checkpoints")
    def checkpoints():
        return list_checkpoints(runs)

    @app.get("/api/checkpoints/defaults")
    def defaults(path: str):
        try:
            return checkpoint_defaults(manager.resolve_checkpoint(path))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/jobs")
    async def submit(image: UploadFile = File(...), settings: str = Form(...), mask: UploadFile = File(None)):
        try:
            raw = json.loads(settings)
            known = {f.name for f in fields(JobSettings)}
            job_settings = JobSettings(**{k: v for k, v in raw.items() if k in known})
            mask_bytes = await mask.read() if mask is not None else None
            job = manager.submit(await image.read(), mask_bytes, job_settings)
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
            raise HTTPException(400, f"invalid request: {exc}")
        return job.summary()

    @app.get("/api/jobs/{job_id}")
    def status(job_id: str):
        return get_job(job_id).summary()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        job = get_job(job_id)
        manager.cancel(job)
        return job.summary()

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str, after: int = -1):
        """Server-Sent Events: every job event after `after`, until the job has finished."""
        job = get_job(job_id)

        async def stream():
            sent = after + 1
            while True:
                while sent < len(job.events):
                    yield f"data: {json.dumps(job.events[sent])}\n\n"
                    sent += 1
                if job.status in ("done", "error", "cancelled") and sent >= len(job.events):
                    break
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}/preview.jpg")
    def preview(job_id: str):
        job = get_job(job_id)
        if job.preview is None:
            raise HTTPException(404, "no preview yet")
        return Response(job.preview, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    def result_file(job_id: str, name: str, media: str):
        job = get_job(job_id)
        path = job.out_dir / name
        if job.status != "done" or not path.exists():
            raise HTTPException(404, "result not available")
        return FileResponse(path, media_type=media, filename=f"overpaint_{job.id}_{name}")

    @app.get("/api/jobs/{job_id}/result.png")
    def result(job_id: str):
        return result_file(job_id, "final.png", "image/png")

    @app.get("/api/jobs/{job_id}/strokes.pt")
    def strokes(job_id: str):
        return result_file(job_id, "strokes.pt", "application/octet-stream")

    @app.get("/api/jobs/{job_id}/stats.json")
    def stats(job_id: str):
        return result_file(job_id, "stats.json", "application/json")

    @app.post("/api/mask/auto")
    async def mask_auto(image: UploadFile = File(...), mode: str = Form("subject")):
        if mode not in ("subject", "face"):
            raise HTTPException(400, "mode must be 'subject' or 'face'")
        try:
            img = Image.open(io.BytesIO(await image.read()))
            img.load()
        except OSError as exc:
            raise HTTPException(400, f"invalid image: {exc}")
        buf = io.BytesIO()
        auto_importance(img, mode).save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png", headers={"X-Placeholder": "true"})

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs", help="Folder with training runs (checkpoints are listed from here)")
    p.add_argument("--out", default=None, help="Where results are written (default: <runs>/web)")
    p.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to allow access from other machines")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.runs, args.device, args.out), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
