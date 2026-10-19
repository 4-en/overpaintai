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


def test_smudge_pull_is_in_canvas_units():
    """With pull strength 1.0, a horizontal smudge drags paint by ~the stroke's tangent length."""
    engine = DifferentiableCanvasEngine(canvas_size=SIZE, smudge_pull_strength=1.0, smudge_taps=1).eval()
    layer = engine.create_blank_paint_layer(1, SIZE, SIZE)
    # Hard vertical bar of paint at x ~= 0.3
    bar = torch.tensor([[0.3, 0.0, 0.3, 0.5, 0.3, 1.0, 0.05, 0.05, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]])
    layer = engine.forward_single_step(layer, bar)
    # Smudge rightwards along y=0.5; tangent length |B'(t)| = 0.2 everywhere
    smudge = torch.tensor([[0.2, 0.5, 0.3, 0.5, 0.4, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    out = engine.forward_single_step(layer, smudge, one_hot_tool(2))
    new_alpha = (out - layer)[0, 3, SIZE // 2]
    xs = torch.linspace(0.0, 1.0, SIZE)
    peak_x = xs[new_alpha.argmax()].item()
    assert abs(peak_x - (0.3 + 0.2)) < 2.0 / SIZE
