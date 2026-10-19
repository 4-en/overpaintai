import torch

from overpaintai import views
from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.synthetic import random_strokes, render_targets


def _random_view_strokes(B=2, N=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return views.Strokes(
        points=0.2 + 0.6 * torch.rand(B, N, 3, 2, generator=g),
        radii=0.02 + 0.08 * torch.rand(B, N, 2, generator=g),
        opacity=0.5 + 0.5 * torch.rand(B, N, 2, generator=g),
        hardness=torch.rand(B, N, 1, generator=g),
        color=torch.rand(B, N, 3, generator=g),
    )


def test_view_image_transforms_round_trip():
    s = _random_view_strokes()
    boxes = torch.tensor([[0.25, 0.5, 0.5], [0.0, 0.0, 1.0]])
    back = views.image_to_view(views.view_to_image(s, boxes), boxes)
    for a, b in zip(vars(s).values(), vars(back).values()):
        assert torch.allclose(a, b, atol=1e-6)


def test_engine_params_round_trip():
    engine = DifferentiableCanvasEngine(canvas_size=32)
    s = _random_view_strokes()
    back = views.from_engine_params(views.to_engine_params(s, engine), engine)
    for a, b in zip(vars(s).values(), vars(back).values()):
        assert torch.allclose(a, b, atol=1e-6)


def test_stroke_drawn_in_a_view_matches_full_canvas_render():
    """Rendering view-space strokes on an R x R view canvas equals rendering the mapped strokes on the
    full canvas and cropping that view (the renderer is resolution-independent)."""
    S, R = 129, 33  # tile [0.5, 0.75] of a 129 px image is exactly pixels 64..96
    engine = DifferentiableCanvasEngine(canvas_size=S, min_radius=0.001).eval()
    box = torch.tensor([[0.5, 0.25, 0.25]])
    s = _random_view_strokes(B=1, N=3)
    s.radii = s.radii.clamp(max=0.2)
    tools = torch.zeros(1, 3, 5)
    tools[..., 0] = 1.0

    # Strokes in view coords, on a view-sized canvas; radii in view units -> scale the engine range
    view_engine = DifferentiableCanvasEngine(canvas_size=R, min_radius=0.001 / 0.25, max_radius=0.25 / 0.25).eval()
    view_layer = view_engine(view_engine.create_blank_paint_layer(1, R, R), views.to_engine_params(s, view_engine), tools)

    full = engine(engine.create_blank_paint_layer(1, S, S), views.to_engine_params(views.view_to_image(s, box), engine), tools)
    crop = full[:, :, 32:65, 64:97]
    assert (crop - view_layer).abs().max() < 0.05
    assert (crop - view_layer).abs().mean() < 2e-3


def test_resampled_view_is_nearly_resolution_independent():
    engine = DifferentiableCanvasEngine(canvas_size=256).eval()
    params, tools = random_strokes(engine, 2, 40, generator=torch.Generator().manual_seed(0))
    boxes = torch.tensor([[0.0, 0.0, 1.0], [0.3, 0.2, 0.5]])
    v256 = views.resample_views(render_targets(engine, params, tools, 256), boxes, 64)
    v512 = views.resample_views(render_targets(engine, params, tools, 512), boxes, 64)
    assert (v256 - v512).abs().mean() < 0.01


def test_tile_boxes_cover_the_image():
    boxes = views.tile_boxes(4, overlap=0.125)
    assert boxes.shape == (16, 3)
    assert (boxes[:, :2] >= 0).all() and (boxes[:, :2] + boxes[:, 2:] <= 1.0 + 1e-6).all()
    grid = views.view_grid(boxes, 16).reshape(-1, 2)
    covered = torch.zeros(32, 32, dtype=torch.bool)
    covered[(grid[:, 1] * 31).round().long(), (grid[:, 0] * 31).round().long()] = True
    assert covered.all()
