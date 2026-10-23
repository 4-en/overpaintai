import math

import pytest
import torch

from overpaintai.canvas_engine import DifferentiableCanvasEngine, reverse_strokes

SIZE = 64  # engines below use max_radius=0.25: the stroke params in these tests were written for it


def one_hot_tool(idx: int) -> torch.Tensor:
    tw = torch.zeros(1, DifferentiableCanvasEngine.NUM_TOOLS)
    tw[0, idx] = 1.0
    return tw


def test_reversed_stroke_renders_identically():
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.25).eval()
    layer = engine.create_blank_paint_layer(1, SIZE, SIZE)
    stroke = torch.tensor([[0.1, 0.3, 0.5, 0.6, 0.9, 0.4, 0.5, 0.2, 0.9, 0.4, 0.7, 0.8, 0.2, 0.1]])
    for tool in (0, 4):  # Paint and Eraser (eraser on top of a painted layer)
        base = layer if tool == 0 else engine.forward_single_step(layer, stroke)
        out = engine.forward_single_step(base, stroke, one_hot_tool(tool))
        out_rev = engine.forward_single_step(base, reverse_strokes(stroke), one_hot_tool(tool))
        assert torch.allclose(out, out_rev, atol=1e-4)
    assert torch.equal(reverse_strokes(reverse_strokes(stroke)), stroke)


def test_training_tool_weights_are_one_hot_with_gradients():
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.25).train()
    raw = torch.randn(4, 14)
    logits = torch.randn(4, engine.NUM_TOOLS, requires_grad=True)
    _, tw = engine.decode_raw_actions(raw, logits)
    assert torch.equal(tw.sum(-1), torch.ones(4))
    assert torch.equal((tw == 1.0).sum(-1), torch.ones(4, dtype=torch.long))

    target = torch.arange(engine.NUM_TOOLS, dtype=torch.float32)
    (tw * target).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def _bar(engine, layer, x0, x1, rgb):
    """Hard horizontal bar of paint across y=0.5 from x0 to x1."""
    stroke = torch.tensor([[x0, 0.5, (x0 + x1) / 2, 0.5, x1, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, *rgb]])
    return engine.forward_single_step(layer, stroke, one_hot_tool(0))


def _smudge(engine, layer, x0, x1, requires_grad=False):
    stroke = torch.tensor([[x0, 0.5, (x0 + x1) / 2, 0.5, x1, 0.5, 0.2, 0.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    stroke.requires_grad_(requires_grad)
    return engine.forward_single_step(layer, stroke, one_hot_tool(2)), stroke


def _px(x: float) -> int:
    return round(x * (SIZE - 1))


def test_smudge_from_bare_canvas_digs_indent():
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.25).eval()
    layer = _bar(engine, engine.create_blank_paint_layer(1, SIZE, SIZE), 0.5, 0.8, (0.9, 0.2, 0.1))
    out, _ = _smudge(engine, layer, 0.1, 0.6)
    row = _px(0.5)
    # Paint the brush passed through (inside the bar, behind the final dab) is pushed away
    assert layer[0, 3, row, _px(0.45)] > 0.99
    assert out[0, 3, row, _px(0.45)] < 0.05
    # Paint well outside the brush footprint is untouched (compared premultiplied: RGB of nearly
    # transparent pixels is numerically unstable through the premultiply round trip)
    def premult(x):
        return torch.cat([x[:, :3] * x[:, 3:], x[:, 3:]], dim=1)[..., _px(0.9) :]

    assert torch.allclose(premult(out), premult(layer), atol=1e-5)


def test_smudge_carries_paint_along_the_stroke():
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.25).eval()
    layer = engine.create_blank_paint_layer(1, SIZE, SIZE)
    layer = _bar(engine, layer, 0.15, 0.35, (0.1, 0.4, 0.9))  # blue
    layer = _bar(engine, layer, 0.6, 0.85, (0.9, 0.2, 0.1))   # red
    out, _ = _smudge(engine, layer, 0.25, 0.75)
    blue = torch.tensor([0.1, 0.4, 0.9])
    # Blue dragged from the start reaches the red bar undiluted (no blur), then mixes in gradually
    assert torch.allclose(out[0, :3, _px(0.5), _px(0.6)], blue, atol=0.02)
    assert out[0, 3, _px(0.5), _px(0.6)] > 0.99
    assert out[0, 2, _px(0.5), _px(0.7)] > 0.5  # still mostly blue inside the red


def test_smudge_drag_distance_is_finite_at_full_opacity():
    def drag(smudge_length):
        engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.25, smudge_length=smudge_length).eval()
        layer = _bar(engine, engine.create_blank_paint_layer(1, SIZE, SIZE), 0.05, 0.2, (0.1, 0.4, 0.9))
        out, _ = _smudge(engine, layer, 0.15, 0.95)
        return out[0, 3, _px(0.5)]

    unlimited, default = drag(float("inf")), drag(12.0)
    assert unlimited[_px(0.9)] > 0.95  # without a limit, paint rides along the whole stroke
    assert default[_px(0.45)] > 0.95  # with it, a trail is still left near the source...
    assert default[_px(0.9)] < 0.5 * unlimited[_px(0.9)]  # ...but the load runs out with distance


def test_smudge_is_differentiable():
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.25).train()
    layer = _bar(engine, engine.create_blank_paint_layer(1, SIZE, SIZE), 0.5, 0.8, (0.9, 0.2, 0.1))
    out, stroke = _smudge(engine, layer, 0.1, 0.6, requires_grad=True)
    out[:, 3].sum().backward()
    assert stroke.grad[0, :6].abs().sum() > 0  # curve geometry
    assert stroke.grad[0, 8:11].abs().sum() > 0  # opacity and hardness


def test_smudge_backward_matches_numerical_gradient():
    from overpaintai.canvas_engine import _dab_masks, _SmudgeDabs

    torch.manual_seed(0)
    B, n, R, H, W = 2, 6, 3, 12, 12
    Wn, Hp, Wp = 2 * R + 1, H + 2 * R, W + 2 * R
    dt = torch.float64
    x = (0.3 + 0.06 * torch.arange(n, dtype=dt)).repeat(B, 1) + 0.01 * torch.rand(B, n, dtype=dt)
    y = torch.full((B, n), 0.5, dtype=dt) + 0.02 * torch.rand(B, n, dtype=dt)
    r = torch.full((B, n), 0.12, dtype=dt)
    strength = torch.rand(B, n, dtype=dt) * 0.9
    hardness = torch.tensor([0.3, 0.7], dtype=dt)
    cx, cy = torch.round(x * (W - 1)).long(), torch.round(y * (H - 1)).long()
    k = torch.arange(Wn)
    offs = (k.view(-1, 1) * Wp + k.view(1, -1)).flatten()
    cols, rows = cx.unsqueeze(-1) - R + k, cy.unsqueeze(-1) - R + k
    window = (
        cols.to(dt) / (W - 1), rows.to(dt) / (H - 1),
        ((cols >= 0) & (cols < W)).to(dt), ((rows >= 0) & (rows < H)).to(dt),
    )
    canvas = torch.nn.functional.pad(torch.rand(B, 4, H, W, dtype=dt), (R, R, R, R)).reshape(B, 4, Hp * Wp)
    inputs = [t.clone().requires_grad_(True) for t in (canvas, x, y, r, strength, hardness)]

    def f(c, x, y, r, s, h):
        return _SmudgeDabs.apply(c, x, y, r, s, h, cy * Wp + cx, offs, *window, 40.0, _dab_masks)

    assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-6)


def test_stroke_result_does_not_depend_on_batch():
    torch.manual_seed(1)
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.25, compile_smudge=False).eval()
    strokes = torch.sigmoid(torch.randn(5, 14) - torch.tensor([0.0] * 6 + [1.0, 1.0] + [0.0] * 6))
    tools = torch.zeros(5, 5)
    tools[:, 2] = 1.0  # smudge: per-stroke dab counts and window-size groups
    layer = engine.forward_single_step(
        engine.create_blank_paint_layer(5, SIZE, SIZE), torch.sigmoid(torch.randn(5, 14))
    )
    full = engine.forward_single_step(layer, strokes, tools)
    single = torch.cat([engine.forward_single_step(layer[i : i + 1], strokes[i : i + 1], tools[i : i + 1]) for i in range(5)])
    assert torch.allclose(full, single, atol=1e-5)


def test_checkpointed_unroll_matches_plain_unroll():
    torch.manual_seed(2)
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.25, compile_smudge=False).train()
    raw = torch.randn(2, 3, 14)
    logits = torch.randn(2, 3, engine.NUM_TOOLS)

    def run(checkpoint_strokes):
        r, lg = raw.clone().requires_grad_(True), logits.clone().requires_grad_(True)
        out = engine(
            engine.create_blank_paint_layer(2, SIZE, SIZE), torch.sigmoid(r), torch.softmax(lg, -1),
            checkpoint_strokes=checkpoint_strokes,
        )
        engine.composite(out).mean().backward()
        return out.detach(), r.grad, lg.grad

    for a, b in zip(run(False), run(True)):
        assert torch.allclose(a, b, atol=1e-6)


def test_pressure_curve_defaults_to_linear_and_can_swell():
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.25).eval()
    layer = engine.create_blank_paint_layer(1, SIZE, SIZE)
    straight = torch.tensor([[0.15, 0.5, 0.5, 0.5, 0.85, 0.5, 0.2, 0.2, 0.9, 0.9, 1.0, 0.1, 0.2, 0.3]])
    mean_mid = torch.cat([straight, torch.tensor([[0.2, 0.9]])], dim=-1)  # middle = mean of the ends
    assert torch.allclose(engine.forward_single_step(layer, straight), engine.forward_single_step(layer, mean_mid),
                          atol=1e-5)

    swell = torch.cat([straight, torch.tensor([[0.6, 0.9]])], dim=-1)  # thicker in the middle
    width = lambda out, x: (out[0, 3, :, x] > 0.5).sum().item()  # painted rows in column x
    out = engine.forward_single_step(layer, swell)
    assert width(out, SIZE // 2) >= width(out, int(0.2 * SIZE)) + 3

    reversed_swell = reverse_strokes(swell)
    assert torch.allclose(out, engine.forward_single_step(layer, reversed_swell), atol=1e-4)


def _premult(layer):
    return torch.cat([layer[:, :3] * layer[:, 3:], layer[:, 3:]], 1)


def test_joint_step_is_order_independent_and_merges_strokes():
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.25, min_opacity=0.0).eval()
    torch.manual_seed(1)
    layer = engine(engine.create_blank_paint_layer(2, SIZE, SIZE), torch.rand(2, 3, 16))  # some paint below
    strokes = torch.rand(2, 5, 16)
    tools = torch.zeros(2, 5, engine.NUM_TOOLS)
    tools[..., 0], tools[:, 3, 0], tools[:, 3, 1] = 1.0, 0.0, 1.0  # one dry brush stroke among them
    out = engine(layer, strokes, tools, joint=5)
    perm = torch.tensor([3, 0, 4, 2, 1])
    assert torch.allclose(out, engine(layer, strokes[:, perm], tools[:, perm], joint=5), atol=1e-5)
    # a single stroke renders as before, for every tool (premultiplied: the RGB of near-transparent pixels is
    # numerically unstable either way)
    for tool in range(engine.NUM_TOOLS):
        one = torch.zeros(2, 1, engine.NUM_TOOLS)
        one[..., tool] = 1.0
        assert torch.allclose(_premult(engine(layer, strokes[:, :1], one, joint=1)),
                              _premult(engine(layer, strokes[:, :1], one)), atol=1e-5), tool

    # two overlapping strokes of the same color at half opacity: one coat (max), not two (source over twice)
    blank = engine.create_blank_paint_layer(1, SIZE, SIZE)
    bar = lambda x0, x1, a, rgb: [x0, 0.5, (x0 + x1) / 2, 0.5, x1, 0.5, 0.4, 0.4, a, a, 1.0, *rgb]
    same = torch.tensor([[bar(0.1, 0.6, 0.5, (1, 0, 0)), bar(0.4, 0.9, 0.5, (1, 0, 0))]])
    mid = (slice(None), 3, SIZE // 2, SIZE // 2)
    assert abs(engine(blank, same, joint=2)[mid].item() - 0.5) < 1e-3
    assert abs(engine(blank, same)[mid].item() - 0.75) < 1e-3
    # different colors where they overlap: coverage-weighted mean, whichever comes first
    mixed = torch.tensor([[bar(0.1, 0.6, 1.0, (1, 0, 0)), bar(0.4, 0.9, 1.0, (0, 0, 1))]])
    rgb = engine(blank, mixed, joint=2)[0, :3, SIZE // 2, SIZE // 2]
    assert torch.allclose(rgb, torch.tensor([0.5, 0.0, 0.5]), atol=1e-3)


def test_joint_groups_and_checkpointing():
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.25).train()
    raw = torch.randn(2, 6, 16)

    def run(checkpoint_strokes):
        r = raw.clone().requires_grad_(True)
        out = engine(engine.create_blank_paint_layer(2, SIZE, SIZE), torch.sigmoid(r), joint=4,
                     checkpoint_strokes=checkpoint_strokes)
        engine.composite(out).mean().backward()
        return out.detach(), r.grad

    (out, grad), (out_c, grad_c) = run(False), run(True)
    assert torch.allclose(out, out_c, atol=1e-6) and torch.allclose(grad, grad_c, atol=1e-6)
    assert (grad.abs().sum(-1) > 0).all()  # every stroke gets a gradient
    # groups of 4 then 2: the same as rendering the two groups one after another
    s = torch.sigmoid(raw)
    blank = engine.create_blank_paint_layer(2, SIZE, SIZE)
    two = engine(engine(blank, s[:, :4], joint=4), s[:, 4:], joint=2)
    assert torch.allclose(out, two, atol=1e-6)


def test_joint_smudge_blur_and_eraser_only_edit_the_canvas_before_the_step():
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.25, min_opacity=0.0).eval()
    torch.manual_seed(2)
    layer = engine(engine.create_blank_paint_layer(2, SIZE, SIZE), torch.rand(2, 6, 16))
    strokes = torch.rand(2, 5, 16)
    tools = torch.eye(engine.NUM_TOOLS)[[0, 2, 3, 4, 2]].expand(2, -1, -1)  # paint, 2 smudges, blur, eraser
    out = engine(layer, strokes, tools, joint=5)
    perm = torch.tensor([4, 2, 0, 3, 1])
    assert torch.allclose(out, engine(layer, strokes[:, perm], tools[:, perm], joint=5), atol=1e-5)
    # the step's paint lands on top, unsmudged, unblurred, unerased: the same as editing first, painting after
    edits = engine(layer, strokes[:, 1:], tools[:, 1:], joint=4)
    assert torch.allclose(_premult(out), _premult(engine(edits, strokes[:, :1], tools[:, :1], joint=1)), atol=1e-5)
    # each smudge reads the canvas before the step, not the other smudge's result: the changes add up
    one, two = (engine(layer, strokes[:, k : k + 1], tools[:, k : k + 1], joint=1) for k in (1, 4))
    both = engine(layer, strokes[:, [1, 4]], tools[:, [1, 4]], joint=2)
    expected = (_premult(one) + _premult(two) - _premult(layer)).clamp(0, 1)
    assert (_premult(both) - expected).abs().max() < 1e-4


def test_joint_soft_strokes_side_by_side_leave_no_seam():
    engine = DifferentiableCanvasEngine(canvas_size=128, min_opacity=0.0).eval()
    blank = engine.create_blank_paint_layer(1, 128, 128)
    bar = lambda y, a: [0.1, y, 0.5, y, 0.9, y, 0.08, 0.08, a, a, 0.0, 0.8, 0.2, 0.2]  # soft, radius ~8 px
    rows = slice(57, 71)  # between the two stroke centres (one radius apart)
    s = torch.tensor([[bar(0.45, 1.0), bar(0.515, 1.0)]])
    # full opacity: the soft edges fill the gap as if painted one after another (max coverage left a pale seam)
    assert torch.allclose(engine(blank, s, joint=2)[0, 3, rows], engine(blank, s)[0, 3, rows], atol=1e-4)
    # half opacity on the same spot: still one coat, not two
    same = torch.tensor([[bar(0.5, 0.5), bar(0.5, 0.5)]])
    assert abs(engine(blank, same, joint=2)[0, 3, 63, 64].item() - 0.5) < 0.01


def test_joint_gradients_stay_finite_in_the_far_tails_of_soft_strokes():
    engine = DifferentiableCanvasEngine(canvas_size=128, min_opacity=0.0).train()
    blank = engine.create_blank_paint_layer(1, 128, 128)
    # soft dots whose tails reach ~1e-25 .. 1e-45 across the canvas (squares and reciprocals of that under- /
    # overflow in float32): any NaN or inf there makes the whole gradient unusable
    dot = lambda x, y, rgb: [x, y, x, y, x, y, 0.01, 0.01, 1.0, 1.0, 0.0, *rgb, 0.01, 1.0]
    raw = torch.tensor([[dot(0.2, 0.2, (0.9, 0.1, 0.1)), dot(0.3, 0.25, (0.1, 0.1, 0.9))]]).logit(eps=1e-4)
    for x in range(1, 8):  # sweep the tail distance
        r = (raw + torch.tensor([0.0] * 16).index_fill(0, torch.tensor([0, 2, 4]), 0.1 * x)).requires_grad_(True)
        out = engine(blank, torch.sigmoid(r), joint=2)
        engine.composite(out).mean().backward()
        assert r.grad.isfinite().all(), x


def test_opaque_canvas_paints_like_a_transparent_one():
    torch.manual_seed(7)
    clear = DifferentiableCanvasEngine(canvas_size=SIZE, min_opacity=0.0).eval()
    opaque = DifferentiableCanvasEngine(canvas_size=SIZE, min_opacity=0.0, opaque=True).eval()
    blank = opaque.create_blank_paint_layer(2, SIZE, SIZE)
    assert (blank[:, 3] == 1).all() and torch.allclose(opaque.composite(blank), clear.create_background(2, SIZE, SIZE))
    strokes = torch.rand(2, 6, 16)
    for joint in (0, 6):
        a = clear.composite(clear(clear.create_blank_paint_layer(2, SIZE, SIZE), strokes, joint=joint))
        b = opaque.composite(opaque(blank, strokes, joint=joint))
        assert torch.allclose(a, b, atol=1e-5), joint


def test_mask_keeps_the_canvas_and_cuts_a_sharp_edge():
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.5, min_opacity=0.0, opaque=True,
                                        eraser_mode="mask").eval()
    layer = engine.create_blank_paint_layer(1, SIZE, SIZE)
    before = engine.composite(layer)
    # a big round red dab, and a big hard mask whose straight-ish edge cuts through its middle
    dab = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.8, 1.0]  # r ~ 0.4
    mask = [1.3, -0.5, 1.3, 0.5, 1.3, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    tools = torch.zeros(1, 2, engine.NUM_TOOLS)
    tools[0, 0, 0], tools[0, 1, 4] = 1.0, 1.0
    out = engine.composite(engine(layer, torch.tensor([[dab, mask]]), tools, joint=2))
    # the radius-0.5 mask at x = 1.3 covers x > 0.8: left of its edge red, right of it the canvas as before
    edge = 1.3 - (engine.min_radius + 1.0 * (engine.max_radius - engine.min_radius))  # x of the mask's edge
    col = lambda x: int(round(x * (SIZE - 1)))
    y = SIZE // 2
    assert out[0, 0, y, col(edge) - 3] > 0.95 and out[0, 1, y, col(edge) - 3] < 0.05
    assert torch.allclose(out[0, :, y, col(edge) + 3], before[0, :, y, col(edge) + 3], atol=1e-4)
    # the cut stays sharp: one pixel from red to the canvas
    assert out[0, 1, y, col(edge) - 2] < 0.1 and abs(out[0, 1, y, col(edge) + 2] - before[0, 1, y, col(edge) + 2]) < 0.05
    # masks exist only within joint steps
    with pytest.raises(ValueError):
        engine(layer, torch.tensor([[mask]]), tools[:, 1:])


def test_profile_and_joint_result_at_points_match_a_render():
    from overpaintai import views
    from overpaintai.canvas_engine import joint_at, profile_at

    torch.manual_seed(8)
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, max_radius=0.3, min_opacity=0.0, num_stamps=16,
                                        opaque=True, eraser_mode="mask").eval()
    params = torch.rand(1, 5, 16)
    params[..., 6:8] *= 0.4
    tools = torch.zeros(1, 5, engine.NUM_TOOLS)
    tools[..., 0] = 1.0
    tools[0, 4, 0], tools[0, 4, 4] = 0.0, 1.0  # one mask
    layer = engine(engine.create_blank_paint_layer(1, SIZE, SIZE), torch.rand(1, 3, 16))
    out = engine.composite(engine(layer, params, tools, joint=5))
    s = views.from_engine_params(params, engine)
    # pixel centres as points
    idx = torch.randint(0, SIZE, (1, 300, 2))
    pts = idx.float() / (SIZE - 1)
    edge = 2.0 * math.log(9.0) * (SIZE - 1) / engine.hard_edge_px
    mask, shape = profile_at(pts, s.points, s.radii, s.opacity, s.hardness[..., 0], edge, stamps=16)
    full = torch.stack([engine._brush(params[:, k], SIZE, SIZE)[0][0, 0] for k in range(5)])  # (5, H, W)
    at = lambda img: img[..., idx[0, :, 1], idx[0, :, 0]]
    assert torch.allclose(mask[0], at(full), atol=1e-4)
    canvas = at(engine.composite(layer)[0]).t().unsqueeze(0)
    pred = joint_at(canvas, mask, shape, s.color, tools[..., 0], tools[..., 4])
    assert torch.allclose(pred[0], at(out[0]).t(), atol=1e-3)
