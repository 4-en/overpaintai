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


def test_new_model_features_shapes_and_ranges():
    torch.manual_seed(5)
    model = _small_model(paintable_mask=True, stroke_gate=True, error_progress=True)
    assert model.in_channels == 13
    boxes = views.whole_image_boxes(2)
    out = model(torch.randn(2, 13, R, R), boxes, torch.zeros(2), error_progress=torch.tensor([0.1, 0.8]))
    assert out.gate.shape == (2, 4) and (out.gate > 0).all() and (out.gate < 1).all()
    assert out.gate.mean() > 0.9  # strokes start switched on
    plain = _small_model()
    assert torch.equal(plain(torch.randn(1, 12, R, R), views.whole_image_boxes(1), torch.zeros(1)).gate,
                       torch.ones(1, 4))


def test_old_weights_load_into_grown_model_and_behave_the_same(tmp_path):
    from overpaintai.train import Trainer

    torch.manual_seed(6)
    old = _small_model()
    cfg = dict(num_strokes=4, widths=(8, 16, 32, 64), token_dim=64, heads=4)
    path = tmp_path / "old.pt"
    torch.save({"model": old.state_dict(), "ema": None, "model_config": cfg}, path)

    trainer = Trainer.__new__(Trainer)  # only the weight loading is under test
    trainer.model = _small_model(paintable_mask=True, stroke_gate=True, error_progress=True)
    trainer.ema = None
    trainer._init_from(str(path))
    new = trainer.model.eval()
    old.eval()

    x = torch.randn(2, 12, R, R)
    boxes = views.whole_image_boxes(2)
    a = old(x, boxes, torch.full((2,), 0.3))
    b = new(torch.cat([x, torch.ones(2, 1, R, R)], 1), boxes, torch.full((2,), 0.3),
            error_progress=torch.tensor([0.4, 0.9]))  # new inputs start with zero weight: ignored
    for f in ("points", "radii", "opacity", "hardness", "color"):
        assert torch.allclose(getattr(a.strokes, f), getattr(b.strokes, f), atol=1e-5), f
    assert torch.allclose(b.gate, torch.full_like(b.gate, torch.sigmoid(torch.tensor(3.0)).item()), atol=1e-5)


def test_old_checkpoints_still_load_for_painting(tmp_path):
    from overpaintai.highres import load_checkpoint_model

    cfg = dict(num_strokes=4, widths=(8, 16, 32, 64), token_dim=64, heads=4)  # no feature switches: old run
    torch.save({"model": _small_model().state_dict(), "ema": None, "model_config": cfg}, tmp_path / "c.pt")
    model, _ = load_checkpoint_model(str(tmp_path / "c.pt"), torch.device("cpu"))
    assert model.in_channels == 12 and not model.stroke_gate


def test_gate_keeps_a_gradient_when_switched_off():
    from overpaintai.model import GATE_LEAK

    model = _small_model(stroke_gate=True).train()
    with torch.no_grad():
        model.attr_mlp[-1].bias[-1] = -20.0  # saturated: sigmoid'(-17) ~ 4e-8
    out = model(torch.randn(1, 12, R, R), views.whole_image_boxes(1), torch.zeros(1))
    assert out.gate.max() < 1e-6  # forward: off
    grad, = torch.autograd.grad(out.gate.sum(), model.attr_mlp[-1].bias)
    assert grad[-1] >= GATE_LEAK * out.gate.numel() * 0.99  # backward: still a usable signal


def test_stationary_model_ignores_the_step_and_hard_gate_is_on_or_off():
    torch.manual_seed(7)
    model = _small_model(stroke_gate=True, stationary=True, hard_gate=True).eval()
    assert model.uses_error_progress
    x, boxes = torch.randn(2, 12, R, R), views.whole_image_boxes(2)
    e = torch.tensor([0.2, 0.95])
    a = model(x, boxes, torch.zeros(2), error_progress=e)
    b = model(x, boxes, torch.ones(2), error_progress=e)
    assert torch.equal(a.strokes.points, b.strokes.points)  # step progress has no effect
    c = model(x, boxes, torch.zeros(2), error_progress=torch.tensor([0.5, 0.5]))
    assert not torch.allclose(a.strokes.radii, c.strokes.radii)  # the canvas state does
    assert set(a.gate.unique().tolist()) <= {0.0, 1.0}

    model.train()
    with torch.no_grad():
        model.attr_mlp[-1].bias[-1] = -3.0  # logit 0: on with probability 0.5 in training
    on = torch.stack([model(x, boxes, torch.zeros(2), error_progress=e).gate.detach() for _ in range(20)])
    assert set(on.unique().tolist()) <= {0.0, 1.0} and 0.3 < on.mean() < 0.7  # sampled
    with torch.no_grad():
        model.attr_mlp[-1].bias[-1] = -20.0  # logit -17: off
    out = model(x, boxes, torch.zeros(2), error_progress=e)
    assert (out.gate == 0).all()
    grad, = torch.autograd.grad(out.gate.sum(), model.attr_mlp[-1].bias)
    assert grad[-1] > 0  # straight-through (+ leak): an off stroke still learns whether to switch on


def test_placement_noise_varies_training_draws_and_replays():
    torch.manual_seed(8)
    model = _small_model(stroke_gate=True, hard_gate=True)
    model.placement_noise = 1.0
    x, boxes = torch.randn(2, 12, R, R), views.whole_image_boxes(2)

    model.eval()  # no noise at inference: deterministic
    assert torch.equal(model(x, boxes, torch.zeros(2)).anchors, model(x, boxes, torch.zeros(2)).anchors)

    model.train()
    first = model(x, boxes, torch.zeros(2))
    other = model(x, boxes, torch.zeros(2))
    assert not torch.allclose(first.anchors, other.anchors)  # a new draw each time
    again = model(x, boxes, torch.zeros(2), samples=first.samples)
    assert torch.equal(first.anchors, again.anchors) and torch.equal(first.gate, again.gate)

    grad, = torch.autograd.grad(again.anchors.sum(), model.log_temperature)  # positions stay differentiable
    assert grad.abs() > 0


def test_deeper_model_from_shallow_weights_starts_identical(tmp_path):
    from overpaintai.train import Trainer

    torch.manual_seed(9)
    old = _small_model()
    cfg = dict(num_strokes=4, widths=(8, 16, 32, 64), token_dim=64, heads=4)
    path = tmp_path / "old.pt"
    torch.save({"model": old.state_dict(), "ema": None, "model_config": cfg}, path)

    trainer = Trainer.__new__(Trainer)  # only the weight loading is under test
    trainer.model = _small_model(global_blocks=4, query_blocks=3)
    trainer.ema = None
    trainer._init_from(str(path))
    x, boxes = torch.randn(2, 12, R, R), views.whole_image_boxes(2)
    a, b = old.eval()(x, boxes, torch.zeros(2)), trainer.model.eval()(x, boxes, torch.zeros(2))
    for f in ("points", "radii", "opacity", "hardness", "color"):
        assert torch.allclose(getattr(a.strokes, f), getattr(b.strokes, f), atol=1e-5), f
