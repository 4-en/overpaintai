import torch

from overpaintai.canvas_engine import DifferentiableCanvasEngine, reverse_strokes

SIZE = 64


def one_hot_tool(idx: int) -> torch.Tensor:
    tw = torch.zeros(1, DifferentiableCanvasEngine.NUM_TOOLS)
    tw[0, idx] = 1.0
    return tw


def test_reversed_stroke_renders_identically():
    engine = DifferentiableCanvasEngine(canvas_size=SIZE).eval()
    layer = engine.create_blank_paint_layer(1, SIZE, SIZE)
    stroke = torch.tensor([[0.1, 0.3, 0.5, 0.6, 0.9, 0.4, 0.5, 0.2, 0.9, 0.4, 0.7, 0.8, 0.2, 0.1]])
    for tool in (0, 4):  # Paint and Eraser (eraser on top of a painted layer)
        base = layer if tool == 0 else engine.forward_single_step(layer, stroke)
        out = engine.forward_single_step(base, stroke, one_hot_tool(tool))
        out_rev = engine.forward_single_step(base, reverse_strokes(stroke), one_hot_tool(tool))
        assert torch.allclose(out, out_rev, atol=1e-4)
    assert torch.equal(reverse_strokes(reverse_strokes(stroke)), stroke)


def test_training_tool_weights_are_one_hot_with_gradients():
    engine = DifferentiableCanvasEngine(canvas_size=SIZE).train()
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
    engine = DifferentiableCanvasEngine(canvas_size=SIZE).eval()
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
    engine = DifferentiableCanvasEngine(canvas_size=SIZE).eval()
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
        engine = DifferentiableCanvasEngine(canvas_size=SIZE, smudge_length=smudge_length).eval()
        layer = _bar(engine, engine.create_blank_paint_layer(1, SIZE, SIZE), 0.05, 0.2, (0.1, 0.4, 0.9))
        out, _ = _smudge(engine, layer, 0.15, 0.95)
        return out[0, 3, _px(0.5)]

    unlimited, default = drag(float("inf")), drag(12.0)
    assert unlimited[_px(0.9)] > 0.95  # without a limit, paint rides along the whole stroke
    assert default[_px(0.45)] > 0.95  # with it, a trail is still left near the source...
    assert default[_px(0.9)] < 0.5 * unlimited[_px(0.9)]  # ...but the load runs out with distance


def test_smudge_is_differentiable():
    engine = DifferentiableCanvasEngine(canvas_size=SIZE).train()
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
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, compile_smudge=False).eval()
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
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, compile_smudge=False).train()
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
