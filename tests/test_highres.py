import torch

from overpaintai import views
from overpaintai.highres import HighResConfig, HighResPainter
from overpaintai.model import StrokePredictor


def _model():
    torch.manual_seed(0)
    return StrokePredictor(num_strokes=4, widths=(8, 16, 32, 64), token_dim=64, heads=4).eval()


def _strokes(N=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return views.Strokes(
        points=0.3 + 0.4 * torch.rand(1, N, 3, 2, generator=g),
        radii=0.01 + 0.05 * torch.rand(1, N, 2, generator=g),
        opacity=0.5 + 0.5 * torch.rand(1, N, 2, generator=g),
        hardness=torch.rand(1, N, 1, generator=g),
        color=torch.rand(1, N, 3, generator=g),
    )


def test_crop_render_matches_full_canvas_render():
    painter = HighResPainter(_model(), size=97, view_res=16, autocast=None)
    s = _strokes()
    layer = painter.engine.create_blank_paint_layer(1, 97, 97)
    cropped = painter.render(layer.clone(), s)
    full = painter.engine(layer, views.to_engine_params(s, painter.engine), torch.eye(5)[[0] * 5].unsqueeze(0))
    assert (cropped - full).abs().max() < 2e-3 and (cropped - full).abs().mean() < 1e-5
    assert cropped[..., :20, :20].abs().max() < 1e-6  # nothing (beyond soft tails) outside the footprint


def test_tile_selection_prefers_error_and_suppresses_overlap():
    painter = HighResPainter(_model(), size=64, view_res=16, autocast=None,
                             config=HighResConfig(max_tiles=3, error_threshold=0.01, nms_iou=0.1))
    err = torch.zeros(1, 1, 64, 64)
    err[..., 40:56, 8:24] = 1.0  # one hot spot, bottom left
    cores, scores = painter.select_tiles(err, core=0.25)
    assert cores.shape[0] >= 1 and scores[0] == scores.max()
    x, y, s = cores[0].tolist()
    assert x <= 8 / 64 + 1e-6 and x + s >= 24 / 64 - 0.1 and y >= 0.5  # covers the hot spot
    for i in range(cores.shape[0]):
        for j in range(i):
            assert painter._iou(cores[i].tolist(), cores[j].tolist()) <= 0.1 + 1e-6


def test_view_boxes_keep_size_inside_image():
    painter = HighResPainter(_model(), size=64, view_res=16, autocast=None, config=HighResConfig(margin=0.25))
    boxes = painter.view_boxes(torch.tensor([[0.0, 0.0, 0.25], [0.75, 0.4, 0.25]]))
    assert torch.allclose(boxes[:, 2], torch.full((2,), 0.375))
    assert (boxes[:, :2] >= 0).all() and (boxes[:, :2] + boxes[:, 2:] <= 1 + 1e-6).all()


def test_fade_keeps_core_strokes_and_fades_edge_strokes():
    painter = HighResPainter(_model(), size=64, view_res=16, autocast=None, config=HighResConfig(margin=0.25))
    cores = torch.tensor([[0.25, 0.25, 0.5]])
    boxes = painter.view_boxes(cores)  # view (0.125, 0.125, 0.75): core [1/6, 5/6] in view coords, margin 1/6
    anchors = torch.tensor([[[0.5, 0.5], [0.99, 0.5], [0.5, 0.5]]])  # 2nd: ~a full margin outside the core
    pts = anchors.unsqueeze(2).expand(1, 3, 3, 2).clone()
    radii = torch.tensor([[[0.05, 0.05], [0.005, 0.005], [0.8, 0.8]]])  # 3rd reaches far past the view
    s = views.Strokes(pts, radii, torch.ones(1, 3, 2), torch.ones(1, 3, 1), torch.ones(1, 3, 3))
    f = painter.fade(s, anchors, cores, boxes)[0]
    assert f[0] == 1.0 and f[1] < 0.05 and f[2] < 0.05


def test_paint_runs_whole_and_tile_levels():
    cfg = HighResConfig(whole_steps=2, rounds_per_level=1, max_tiles=4, error_threshold=0.0, min_core=0.25)
    painter = HighResPainter(_model(), size=64, view_res=16, config=cfg, autocast=None)
    target = torch.rand(1, 3, 64, 64)
    valid = torch.ones(1, 1, 64, 64)
    result = painter.paint(target, source_side=64, valid=valid, log=lambda *_: None)
    assert [s["stage"] for s in result.stats] == ["whole image", "tiles 1/2", "tiles 1/4"]
    assert result.canvas.shape == (1, 3, 64, 64)
    assert sum(r.shape[0] for r in result.strokes) == result.stats[-1]["strokes"] > 8


def test_paint_reports_progress_stages_and_can_be_cancelled():
    import pytest

    from overpaintai.highres import PaintCancelled

    cfg = HighResConfig(whole_steps=3, rounds_per_level=2, max_tiles=2, error_threshold=0.0, min_core=0.25)
    painter = HighResPainter(_model(), size=64, view_res=16, config=cfg, autocast=None)
    target = torch.rand(1, 3, 64, 64)
    events, stages = [], []

    def on_progress(event, get_canvas):
        events.append(dict(event))
        assert get_canvas().shape == (1, 3, 64, 64)

    painter.paint(target, 64, log=lambda *_: None, on_progress=on_progress, on_stage=stages.append)
    assert [e["kind"] for e in events[:3]] == ["whole"] * 3 and events[3]["kind"] == "tiles"
    fractions = [e["fraction"] for e in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert [s["stage"] for s in stages] == ["whole image", "tiles 1/2", "tiles 1/4"]

    calls = []
    with pytest.raises(PaintCancelled):
        painter.paint(target, 64, log=lambda *_: None, should_stop=lambda: calls.append(1) or len(calls) >= 2)
    assert len(calls) == 2  # stopped right at the second check


def test_split_rendering_matches_single_crop():
    from overpaintai.highres import CropRenderer

    torch.manual_seed(0)
    s = views.Strokes(points=torch.rand(1, 6, 3, 2) * 1.2 - 0.1, radii=0.05 + 0.3 * torch.rand(1, 6, 2),
                      opacity=0.5 + 0.5 * torch.rand(1, 6, 2), hardness=torch.rand(1, 6, 1), color=torch.rand(1, 6, 3))
    whole = CropRenderer(120, 32, 1e-3, 0.75, "cpu", max_crop=4096)
    split = CropRenderer(120, 32, 1e-3, 0.75, "cpu", max_crop=32)  # squares reach past the canvas edge
    a, b = whole.render(whole.blank(), s), split.render(split.blank(), s)
    # Compare what is visible (RGB of near-transparent pixels is numerically unstable either way)
    assert (whole.composite(a) - split.composite(b)).abs().max() < 1e-4
    assert (a[:, 3] - b[:, 3]).abs().max() < 1e-4


def test_render_strokes_reproduces_and_upscales_a_painting():
    from overpaintai.highres import render_strokes

    cfg = HighResConfig(whole_steps=3, rounds_per_level=1, max_tiles=2, error_threshold=0.0, min_core=0.25)
    painter = HighResPainter(_model(), size=64, view_res=16, config=cfg, autocast=None)
    target = torch.rand(1, 3, 64, 64)
    result = painter.paint(target, 64, log=lambda *_: None)
    strokes = torch.cat(result.strokes)
    kw = dict(num_stamps=32, min_radius=painter.min_radius, max_radius=painter.max_radius)
    same = render_strokes(strokes, 64, (64, 48), 64, **kw)
    assert same.shape == (1, 3, 48, 64)
    assert (same - result.canvas[..., :48, :]).abs().mean() < 0.01  # same painting (adaptive curve samples)
    big = render_strokes(strokes, 64, (64, 48), 256, **kw)
    assert big.shape == (1, 3, 192, 256)
    down = torch.nn.functional.interpolate(big, size=(48, 64), mode="area")
    assert (down - same).abs().mean() < 0.03  # the same picture, just at 4x the resolution


def test_harness_with_gated_masked_model_skips_invisible_strokes():
    torch.manual_seed(7)
    model = StrokePredictor(num_strokes=4, widths=(8, 16, 32, 64), token_dim=64, heads=4,
                            paintable_mask=True, stroke_gate=True, error_progress=True).eval()
    with torch.no_grad():  # switch half the strokes off via the gate output bias
        model.attr_mlp[-1].bias[-1] = -10.0
    cfg = HighResConfig(whole_steps=2, rounds_per_level=1, max_tiles=2, error_threshold=0.0, min_core=0.25)
    painter = HighResPainter(model, size=64, view_res=16, config=cfg, autocast=None)
    result = painter.paint(torch.rand(1, 3, 64, 64), 64, log=lambda *_: None)
    assert result.stats[-1]["strokes"] == 0  # all gated off -> nothing rendered
