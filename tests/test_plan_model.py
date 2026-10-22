import torch

from overpaintai import views
from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.plan_model import PlanStrokePredictor, build_model
from overpaintai.synthetic import random_strokes, render_targets

R = 32
SMALL = dict(num_strokes=4, widths=(8, 16, 32, 64), token_dim=64, heads=4, global_blocks=2, query_blocks=1,
             refine_layers=2, view_tokens=4, plan_slots=2)


def _model(**kw):
    return PlanStrokePredictor(**{**SMALL, **kw})


def _inputs(engine, target, boxes):
    layer = engine.create_blank_paint_layer(target.shape[0], target.shape[-1], target.shape[-1])
    return views.view_inputs(target, engine.composite(layer), layer[:, 3:], boxes, R), layer


def test_outputs_pressure_curve_and_points_around_the_anchor():
    torch.manual_seed(0)
    model = _model(stroke_gate=True, hard_gate=True, stationary=True).eval()
    boxes = views.whole_image_boxes(2)
    out = model(torch.randn(2, 12, R, R), boxes, torch.zeros(2), error_progress=torch.tensor([0.1, 0.6]))
    s = out.strokes
    assert s.points.shape == (2, 4, 3, 2) and s.radii.shape == (2, 4, 3) and s.opacity.shape == (2, 4, 3)
    lo, hi = model.radius_range
    assert (s.radii >= lo).all() and (s.radii <= hi).all() and (s.opacity >= 0.05).all() and (s.opacity <= 1).all()
    assert set(out.gate.unique().tolist()) <= {0.0, 1.0} and out.tool_logits.shape == (2, 4, 5)
    # the anchor is a reference point: all three Bezier points are bounded offsets from it
    reach = (s.points - out.anchors.unsqueeze(2)).abs()
    assert (reach <= model.max_offset + 0.1 + 1e-5).all()

    engine = DifferentiableCanvasEngine(canvas_size=48, min_opacity=0.0)
    params = views.to_engine_params(views.view_to_image(s, boxes), engine)
    assert params.shape == (2, 4, 16)  # pressure curve: 16 engine params
    assert torch.allclose(views.from_engine_params(params, engine).radii, s.radii.clamp(engine.min_radius), atol=1e-6)
    plain = _model(pressure=False).eval()(torch.randn(1, 12, R, R), views.whole_image_boxes(1), torch.zeros(1))
    assert plain.strokes.radii.shape == (1, 4, 2)


def test_strokes_run_past_their_anchor():
    torch.manual_seed(1)
    model = _model().eval()
    with torch.no_grad():  # long, bent strokes
        model.decoder[-1].weight.normal_(std=2.0)
    out = model(torch.randn(2, 12, R, R), views.whole_image_boxes(2), torch.zeros(2))
    p, a = out.strokes.points, out.anchors
    # the anchor is the chord's middle and the curve passes close by it, however long or bent the stroke
    assert torch.allclose(0.5 * (p[:, :, 0] + p[:, :, 2]), a, atol=1e-5)
    mid = 0.25 * p[:, :, 0] + 0.5 * p[:, :, 1] + 0.25 * p[:, :, 2]
    assert ((mid - a).norm(dim=-1) <= model.max_offset / 4 * 2**0.5 + 1e-5).all()
    assert (p[:, :, 0] - a).norm(dim=-1).max() > model.max_offset / 2  # long strokes exist
    # plan checkpoints from before the options keep their free offsets and 9 px colors
    old = build_model(dict(SMALL, arch="plan"))
    assert old.stroke_points == "free" and old.color_base == "blur9"


def test_color_starts_from_the_target_at_the_strokes_width():
    from overpaintai.plan_model import _blur

    model = _model().eval()
    target = torch.zeros(1, 3, 128, 128)
    target[:, :, :, 64:] = 1.0  # an edge at x = 0.5
    target[:, :, :, 66] = 0.0  # and a thin dark line just right of it
    colors = torch.cat([target, _blur(target, 9), _blur(target, 33)], 1)
    look = torch.tensor([[[[66 / 127, 0.5]], [[0.7, 0.5]]]])  # (1, 2 strokes, 1 point, 2)
    thin, wide = torch.full((1, 2, 3), 0.5 / 127), torch.full((1, 2, 3), 16 / 127)
    c_thin, c_wide = model._base_color(colors, look, thin), model._base_color(colors, look, wide)
    assert c_thin[0, 0].max() < 0.01 and c_thin[0, 1].min() > 0.99  # single pixels: the line itself
    assert 0.3 < c_wide[0, 0].mean() < 0.9  # ~33 px: the edge and the line averaged


def test_any_input_size_and_resolution_independent_strokes():
    torch.manual_seed(2)
    model = _model().eval()
    engine = DifferentiableCanvasEngine(canvas_size=128).eval()
    params, tools = random_strokes(engine, 1, 30, generator=torch.Generator().manual_seed(3))
    boxes = views.whole_image_boxes(1)
    outs = []
    for size in (128, 256):
        target = render_targets(engine, params, tools, size)
        layer = engine.create_blank_paint_layer(1, size, size)
        x = views.view_inputs(target, engine.composite(layer), layer[:, 3:], boxes, 64)
        outs.append(views.view_to_image(model(x, boxes, torch.zeros(1)).strokes, boxes))
    assert (outs[0].points - outs[1].points).abs().max() < 0.02
    assert (outs[0].color - outs[1].color).abs().max() < 0.02
    tokens = model.encode_global(torch.randn(1, 12, 48, 48), torch.zeros(1))
    assert tokens.shape == (1, model.global_grid ** 2, 64)
    out = model(torch.randn(1, 12, 48, 48), boxes, torch.zeros(1), global_tokens=tokens)
    assert out.strokes.points.shape == (1, 4, 3, 2)


def test_strokes_see_along_their_curve_and_each_other():
    torch.manual_seed(4)
    model = _model().eval()
    with torch.no_grad():  # strokes start as dots; give them some extent so there is a curve to look along
        model.decoder[-1].weight.normal_(std=0.5)
    x, boxes = torch.randn(1, 12, R, R), views.whole_image_boxes(1)
    out = model(x, boxes, torch.zeros(1))
    # stroke 0's output depends on stroke 1's latent (self-attention among the stroke latents) ...
    g, = torch.autograd.grad(out.strokes.points[0, 0].sum(), model.queries, retain_graph=True)
    assert g[1].abs().sum() > 0
    # ... and on what lies along the strokes' current curves (features sampled there enter the latents)
    g, = torch.autograd.grad(out.strokes.color.sum(), model.refine[0]["feat"][1].weight)
    assert g.abs().sum() > 0


def test_gradients_reach_every_parameter_through_the_renderer():
    torch.manual_seed(5)
    model = _model(stroke_gate=True)
    engine = DifferentiableCanvasEngine(canvas_size=48, min_opacity=0.0)
    target = torch.rand(2, 3, 48, 48)
    boxes = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.3, 0.6]])
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    tools = torch.zeros(2, 4, 5)
    tools[..., 0] = 1.0
    for _ in range(2):  # zero-initialized branches only pass gradient after one update
        opt.zero_grad()
        x, layer = _inputs(engine, target, boxes)
        out = model(x, boxes, torch.tensor([0.3, 0.7]))
        gated = views.Strokes(out.strokes.points, out.strokes.radii, out.strokes.opacity * out.gate.unsqueeze(-1),
                              out.strokes.hardness, out.strokes.color)
        canvas = engine.composite(engine(layer, views.to_engine_params(views.view_to_image(gated, boxes), engine),
                                         tools))
        loss = (canvas - target).abs().mean() + 1e-3 * out.tool_logits.square().mean()
        loss.backward()
        opt.step()
    missing = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, missing


def test_checkpoints_build_the_right_model_and_paint_with_the_harness(tmp_path):
    from overpaintai.highres import HighResConfig, HighResPainter, load_checkpoint_model, render_strokes
    from overpaintai.model import StrokePredictor

    old = dict(num_strokes=4, widths=(8, 16, 32, 64), token_dim=64, heads=4)
    assert type(build_model(old)) is StrokePredictor  # checkpoints from before `arch` existed
    cfg = dict(SMALL, arch="plan", stroke_gate=True, hard_gate=True, paintable_mask=True, stationary=True)
    torch.save({"model": build_model(cfg).state_dict(), "ema": None, "model_config": cfg}, tmp_path / "p.pt")
    model, _ = load_checkpoint_model(str(tmp_path / "p.pt"), torch.device("cpu"))
    assert isinstance(model, PlanStrokePredictor)

    hr = HighResConfig(whole_steps=2, rounds_per_level=1, max_tiles=2, error_threshold=0.0, min_core=0.25)
    painter = HighResPainter(model.eval(), size=64, view_res=16, config=hr, autocast=None)
    result = painter.paint(torch.rand(1, 3, 64, 64), source_side=64, valid=torch.ones(1, 1, 64, 64), log=lambda *_: None)
    strokes = torch.cat(result.strokes)
    assert strokes.shape[1] == 16 and result.canvas.shape == (1, 3, 64, 64)
    redrawn = render_strokes(strokes, 64, (64, 64), 64)
    assert (redrawn - result.canvas).abs().mean() < 0.02
