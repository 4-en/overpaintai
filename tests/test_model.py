import torch

from overpaintai import views
from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.model import StrokePredictor
from overpaintai.synthetic import random_strokes, render_targets

R = 32  # small working resolution keeps the tests fast; the architecture is size-agnostic


def _small_model(**kw):
    return StrokePredictor(num_strokes=4, widths=(8, 16, 32, 64), token_dim=64, heads=4, **kw)


def _inputs(engine, target, boxes):
    layer = engine.create_blank_paint_layer(target.shape[0], target.shape[-1], target.shape[-1])
    return views.view_inputs(target, engine.composite(layer), layer[:, 3:], boxes, R), layer


def test_output_shapes_and_ranges():
    torch.manual_seed(0)
    model = _small_model()
    engine = DifferentiableCanvasEngine(canvas_size=64)
    boxes = torch.tensor([[0.0, 0.0, 1.0], [0.25, 0.25, 0.5]])
    x, _ = _inputs(engine, torch.rand(2, 3, 64, 64), boxes)
    out = model(x, boxes, torch.tensor([0.0, 0.5]))
    s = out.strokes
    assert s.points.shape == (2, 4, 3, 2) and s.radii.shape == (2, 4, 2) and out.tool_logits.shape == (2, 4, 5)
    lo, hi = model.radius_range
    assert (s.radii >= lo).all() and (s.radii <= hi).all()
    assert (s.opacity >= 0.05).all() and (s.opacity <= 1).all() and (s.color >= 0).all() and (s.color <= 1).all()
    assert (out.anchors >= 0).all() and (out.anchors <= 1).all()


def test_anchor_is_the_curve_midpoint():
    torch.manual_seed(1)
    model = _small_model()
    boxes = views.whole_image_boxes(2)
    out = model(torch.randn(2, 12, R, R), boxes, torch.zeros(2))
    p0, p1, p2 = out.strokes.points.unbind(dim=2)
    assert torch.allclose(0.25 * p0 + 0.5 * p1 + 0.25 * p2, out.anchors, atol=1e-5)


def test_any_input_size_gives_fixed_global_tokens():
    model = _small_model()
    for size in (32, 48, 96):
        x = torch.randn(1, 12, size, size)
        tokens = model.encode_global(x, torch.zeros(1))
        assert tokens.shape == (1, model.token_grid**2, 64)
        out = model(x, views.whole_image_boxes(1), torch.zeros(1), global_tokens=tokens)
        assert out.strokes.points.shape == (1, 4, 3, 2)


def test_same_view_content_at_different_resolutions_gives_same_strokes():
    torch.manual_seed(2)
    model = _small_model().eval()
    engine = DifferentiableCanvasEngine(canvas_size=128).eval()
    params, tools = random_strokes(engine, 1, 30, generator=torch.Generator().manual_seed(3))
    boxes = views.whole_image_boxes(1)
    outs = []
    for size in (128, 256):
        x, _ = _inputs(engine, render_targets(engine, params, tools, size), boxes)
        outs.append(views.view_to_image(model(x, boxes, torch.zeros(1)).strokes, boxes))
    assert (outs[0].points - outs[1].points).abs().max() < 0.02
    assert (outs[0].color - outs[1].color).abs().max() < 0.02


def test_gradients_reach_every_parameter_through_the_renderer():
    torch.manual_seed(4)
    model = _small_model()
    engine = DifferentiableCanvasEngine(canvas_size=48)
    target = torch.rand(2, 3, 48, 48)
    boxes = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.3, 0.6]])
    progress = torch.tensor([0.3, 0.7])
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    tools = torch.zeros(2, 4, 5)
    tools[..., 0] = 1.0

    for _ in range(2):  # zero-initialized residual branches only pass gradient after one update
        opt.zero_grad()
        x, layer = _inputs(engine, target, boxes)
        out = model(x, boxes, progress)
        params = views.to_engine_params(views.view_to_image(out.strokes, boxes), engine)
        canvas = engine.composite(engine(layer, params, tools))
        loss = (canvas - target).abs().mean() + 1e-3 * out.tool_logits.square().mean()
        loss.backward()
        opt.step()

    missing = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, missing
