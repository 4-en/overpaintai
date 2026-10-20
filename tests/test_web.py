import io
import json
import time

import pytest
import torch
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from overpaintai.model import StrokePredictor  # noqa: E402
from overpaintai.web.app import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    run = tmp_path / "runs" / "tiny"
    run.mkdir(parents=True)
    torch.manual_seed(0)
    config = dict(num_strokes=4, widths=(8, 16, 32, 64), token_dim=64, heads=4)
    model = StrokePredictor(**config)
    torch.save({"model": model.state_dict(), "ema": None, "model_config": config, "epoch": 0,
                "args": {"steps": 2, "tile_size": "0.3,0.6", "image_size": 64}, "view_res": 16, "tools": [0]},
               run / "best.pt")
    (run / "args.json").write_text(json.dumps({"steps": 2, "tile_size": "0.3,0.6", "image_size": 64}))
    (run / "epochs.csv").write_text("epoch,val_psnr,preview_psnr\n1,12.5,11.0\n")
    with TestClient(create_app(str(tmp_path / "runs"), device="cpu")) as c:
        yield c, str(run / "best.pt")


def _png(size=(48, 40), color=(200, 80, 40), mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def _wait(c, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = c.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error", "cancelled"):
            return status
        time.sleep(0.1)
    raise TimeoutError(job_id)


def test_page_checkpoints_and_defaults(client):
    c, ckpt = client
    assert "Overpaint" in c.get("/").text
    assert c.get("/static/app.js").status_code == 200
    listing = c.get("/api/checkpoints").json()
    assert listing[0]["path"] == ckpt and listing[0]["val_psnr"] == 12.5
    d = c.get("/api/checkpoints/defaults", params={"path": ckpt}).json()
    assert d["whole_steps"] == 2 and d["tile_range"] == [0.3, 1.0]
    assert c.get("/api/checkpoints/defaults", params={"path": "/etc/passwd"}).status_code == 400


def test_job_with_mask_runs_to_completion(client):
    c, ckpt = client
    settings = {"checkpoint": ckpt, "size": 128, "whole_steps": 2, "rounds": 1, "max_tiles": 2,
                "min_tile": 0.25, "threshold": 0.0, "num_stamps": 16, "unknown_field": 1}
    files = {"image": ("img.png", _png(), "image/png"), "mask": ("mask.png", _png(mode="L", color=255), "image/png")}
    job = c.post("/api/jobs", data={"settings": json.dumps(settings)}, files=files).json()
    status = _wait(c, job["id"])
    assert status["status"] == "done", status
    assert [s["stage"] for s in status["stats"]] == ["whole image", "tiles 1/2", "tiles 1/4"]

    events = [json.loads(line[6:]) for line in c.get(f"/api/jobs/{job['id']}/events").text.splitlines()
              if line.startswith("data: ")]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "status" and "plan" in kinds and "progress" in kinds and "stage" in kinds
    assert events[-1] == {**events[-1], "type": "status", "status": "done"}
    assert any(e.get("preview") for e in events if e["type"] == "progress")

    preview = Image.open(io.BytesIO(c.get(f"/api/jobs/{job['id']}/preview.jpg").content))
    result = Image.open(io.BytesIO(c.get(f"/api/jobs/{job['id']}/result.png").content))
    assert result.size == (128, round(128 * 40 / 48)) and preview.size == result.size  # aspect kept
    assert c.get(f"/api/jobs/{job['id']}/strokes.pt").status_code == 200
    assert "stages" in c.get(f"/api/jobs/{job['id']}/stats.json").json()


def test_cancel_and_invalid_requests(client):
    c, ckpt = client
    settings = {"checkpoint": ckpt, "size": 128, "whole_steps": 200, "tiles": False, "num_stamps": 16}
    job = c.post("/api/jobs", data={"settings": json.dumps(settings)},
                 files={"image": ("img.png", _png(), "image/png")}).json()
    c.post(f"/api/jobs/{job['id']}/cancel")
    assert _wait(c, job["id"])["status"] == "cancelled"
    assert c.get(f"/api/jobs/{job['id']}/result.png").status_code == 404

    bad = {"checkpoint": "/tmp/elsewhere.pt"}
    r = c.post("/api/jobs", data={"settings": json.dumps(bad)}, files={"image": ("i.png", _png(), "image/png")})
    assert r.status_code == 400
    r = c.post("/api/jobs", data={"settings": json.dumps({"checkpoint": ckpt})},
               files={"image": ("i.png", b"not an image", "image/png")})
    assert r.status_code == 400
    assert c.get("/api/jobs/nope").status_code == 404


def test_placeholder_auto_mask(client):
    c, _ = client
    r = c.post("/api/mask/auto", data={"mode": "face"}, files={"image": ("i.png", _png((80, 60)), "image/png")})
    assert r.status_code == 200 and r.headers["x-placeholder"] == "true"
    mask = Image.open(io.BytesIO(r.content))
    assert mask.size == (80, 60) and mask.mode == "L"
    assert mask.getpixel((40, 22)) > mask.getpixel((2, 58))  # brighter in the upper middle
    assert c.post("/api/mask/auto", data={"mode": "x"}, files={"image": ("i.png", _png(), "image/png")}).status_code == 400


def _paint(c, ckpt, **extra):
    settings = {"checkpoint": ckpt, "size": 128, "whole_steps": 2, "rounds": 1, "max_tiles": 2, "min_tile": 0.25,
                "threshold": 0.0, "num_stamps": 16, **extra}
    job = c.post("/api/jobs", data={"settings": json.dumps(settings)},
                 files={"image": ("img.png", _png(), "image/png")}).json()
    status = _wait(c, job["id"])
    assert status["status"] == "done", status
    return job["id"]


def test_continue_from_a_finished_painting(client, tmp_path):
    c, ckpt = client
    first = _paint(c, ckpt)
    before = torch.load(tmp_path / "runs" / "web" / first / "strokes.pt", weights_only=False)["strokes"]

    # Continue: no image needed, size comes from the parent, tiles only
    settings = {"checkpoint": ckpt, "size": 999, "whole_steps": 0, "rounds": 1, "max_tiles": 2, "min_tile": 0.125,
                "threshold": 0.0, "num_stamps": 16}
    job = c.post("/api/jobs", data={"settings": json.dumps(settings), "parent": first}).json()
    status = _wait(c, job["id"])
    assert status["status"] == "done", status
    assert status["parent"] == first and status["settings"]["size"] == 128
    assert [s["stage"] for s in status["stats"]] == ["tiles 1/2", "tiles 1/4", "tiles 1/8"]  # no whole stage

    after = torch.load(tmp_path / "runs" / "web" / job["id"] / "strokes.pt", weights_only=False)["strokes"]
    assert after.shape[0] > before.shape[0] and torch.equal(after[: before.shape[0]], before)  # history kept
    plan = next(e for e in c.get(f"/api/jobs/{job['id']}/events").text.splitlines() if '"plan"' in e)
    assert first in plan  # the plan event says which painting it continues

    # Only finished paintings can be continued
    r = c.post("/api/jobs", data={"settings": json.dumps(settings), "parent": "nope"})
    assert r.status_code == 400


def test_speed_painting_video_export(client):
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    c, ckpt = client
    painting = _paint(c, ckpt)
    video = c.post(f"/api/jobs/{painting}/video", data={"duration": 3, "max_side": 128}).json()
    assert video["kind"] == "video" and video["parent"] == painting
    assert _wait(c, video["id"])["status"] == "done"
    mp4 = c.get(f"/api/jobs/{video['id']}/video.mp4")
    assert mp4.status_code == 200 and mp4.content[4:8] == b"ftyp"  # an MP4 container
    assert c.post("/api/jobs/nope/video", data={"duration": 10}).status_code == 400


def test_stroke_schedule_is_slow_first_and_complete():
    from overpaintai.timelapse import stroke_schedule

    s = stroke_schedule(1000, 100)
    assert s[-1] == 1000 and s == sorted(s)
    assert s[49] < 300  # half the time shows well under half the strokes (big early strokes get more time)


def test_export_at_higher_resolution(client):
    c, ckpt = client
    painting = _paint(c, ckpt)
    job = c.post(f"/api/jobs/{painting}/export", data={"side": 300}).json()
    assert job["kind"] == "export"
    assert _wait(c, job["id"])["status"] == "done"
    png = Image.open(io.BytesIO(c.get(f"/api/jobs/{job['id']}/export.png").content))
    assert png.size[0] == 300 and abs(png.size[1] - 300 * 40 / 48) <= 1  # longer side as requested, aspect kept
    assert c.get(f"/api/jobs/{painting}/export.png").status_code == 404  # only export jobs have one
    assert c.post("/api/jobs/nope/export", data={"side": 4096}).status_code == 400
