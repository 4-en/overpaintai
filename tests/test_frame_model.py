import math

import torch

from overpaintai import views
from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.frame_model import FrameStrokePredictor
from overpaintai.plan_model import build_model

R = 32
SMALL = dict(num_strokes=4, widths=(8, 16, 32, 64), token_dim=64, heads=4, global_blocks=1, query_blocks=1,
             refine_layers=2, view_tokens=4, plan_slots=2, learned_points=3, point_dim=8)


def _model(**kw):
    return FrameStrokePredictor(**{**SMALL, **kw})


def _inputs(engine, target, boxes, R=R):
    layer = engine.create_blank_paint_layer(target.shape[0], target.shape[-1], target.shape[-1])
    return views.view_inputs(target, engine.composite(layer), layer[:, 3:], boxes, R), layer


def _spread(model, std=0.5):
    with torch.no_grad():  # strokes start as dots; give them extent so there is a frame to sample
        model.decoder[-1].weight.normal_(std=std)


def test_outputs_and_any_input_size():
    torch.manual_seed(0)
    model = _model(stroke_gate=True, hard_gate=True, stationary=True, paintable_mask=True).eval()
    boxes = views.whole_image_boxes(2)
    x = torch.rand(2, 13, R, R)
    out = model(x, boxes, torch.zeros(2), error_progress=torch.tensor([0.1, 0.6]))
    s = out.strokes
    assert s.points.shape == (2, 4, 3, 2) and s.radii.shape == (2, 4, 3) and s.opacity.shape == (2, 4, 3)
    lo, hi = model.radius_range
    assert (s.radii >= lo).all() and (s.radii <= hi).all() and (s.color >= 0).all() and (s.color <= 1).all()
    assert out.placement_logits.shape == (2, 4, (R // 4) ** 2)  # placement on the R/4 map
    # the paint alpha (channel 9) is not read: an opaque canvas has none
    x2 = x.clone()
    x2[:, 9] = torch.rand(2, R, R)
    assert torch.allclose(model(x2, boxes, torch.zeros(2), error_progress=torch.tensor([0.1, 0.6])).strokes.points,
                          s.points)
    for size in (48, 64):
        tokens = model.encode_global(torch.rand(1, 13, size, size), torch.zeros(1), torch.zeros(1))
        assert tokens.shape == (1, model.global_grid ** 2, 64)


def test_frame_points_sit_at_the_strokes_edge():
    model = _model().eval()
    p = torch.tensor([[[[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]]]])  # a horizontal stroke, y = 0.5
    r = torch.full((1, 1, 3), 0.1)
    pts, coords = model._frame_points(p, r, R)
    px = 1.0 / (R - 1)
    side = coords[:, 3] == 0
    # across the stroke: distance from the centre line = radius multiple x r + pixels x px
    dist = (pts[0, 0, side, 1] - 0.5).abs()
    want = (coords[side, 1] * 0.1 + coords[side, 2] * px).abs()
    assert torch.allclose(dist, want, atol=1e-6)
    assert ((coords[side, 1].abs() == 1.0) & (coords[side, 2] == 0)).sum() == 2 * len(model.t_along)  # the edge
    # end caps lie beyond the ends, on the stroke's line
    caps = pts[0, 0, ~side]
    assert torch.allclose(caps[:, 1], torch.full_like(caps[:, 1], 0.5), atol=1e-6)
    assert ((caps[:, 0] < 0.2) | (caps[:, 0] > 0.8)).sum() >= 6
    # a thin stroke's edge band is pixels around its centre line
    thin = model._frame_points(p, torch.full((1, 1, 3), 0.5 * px), R)[0]
    assert (thin[0, 0, side, 1] - 0.5).abs().max() < 3 * px


def test_strokes_read_full_resolution_images_along_their_frame():
    torch.manual_seed(1)
    model = _model().eval()
    _spread(model)
    boxes = views.whole_image_boxes(1)
    x = torch.rand(1, 12, R, R)
    target = torch.rand(1, 3, 256, 256)
    canvas = torch.rand(1, 3, 256, 256)
    a = model(x, boxes, torch.zeros(1), images=(target, canvas)).strokes
    # detail finer than a view pixel (every second source pixel at 256 for a 32 px view) changes the strokes
    detail = target.clone()
    detail[..., ::2, ::2] = 1.0 - detail[..., ::2, ::2]
    b = model(x, boxes, torch.zeros(1), images=(detail, canvas)).strokes
    assert (a.color - b.color).abs().max() > 1e-3
    # one image shared by several views works like a copy per view
    boxes2 = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.3, 0.5]])
    x2 = torch.rand(2, 12, R, R)
    shared = model(x2, boxes2, torch.zeros(2), images=(target, canvas)).strokes
    copied = model(x2, boxes2, torch.zeros(2), images=(target.expand(2, -1, -1, -1), canvas.expand(2, -1, -1, -1)))
    assert torch.allclose(shared.points, copied.strokes.points, atol=1e-5)


def test_strokes_see_each_other_and_their_samples():
    torch.manual_seed(2)
    model = _model().eval()
    _spread(model)
    out = model(torch.rand(1, 12, R, R), views.whole_image_boxes(1), torch.zeros(1))
    g, = torch.autograd.grad(out.strokes.points[0, 0].sum(), model.queries, retain_graph=True)
    assert g[1].abs().sum() > 0  # stroke 0 depends on stroke 1's latent
    g, = torch.autograd.grad(out.strokes.color.sum(), model.point_enc[1].weight)
    assert g.abs().sum() > 0  # what the frame samples see enters the latents


def test_proposals_and_look_ahead_start_out_as_the_model_without_them():
    torch.manual_seed(5)
    old = _model().eval()
    _spread(old)
    new = _model(stroke_proposals=True).eval()
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith(("propose.", "proposal_proj.")) for k in missing)
    new.propose[-1].weight.data.zero_(), new.propose[-1].bias.data.zero_()  # as --init-from leaves them
    x, boxes = torch.rand(2, 12, R, R), views.whole_image_boxes(2)
    a = old(x, boxes, torch.zeros(2)).strokes
    b = new(x, boxes, torch.zeros(2)).strokes
    assert torch.allclose(a.points, b.points, atol=1e-6) and torch.allclose(a.radii, b.radii, atol=1e-6)
    # learned points with look-ahead start where they were (only the caps are new)
    ahead = _model(look_ahead=True)
    h = torch.randn(1, 4, 64)
    p = torch.tensor([[[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]]]).expand(1, 4, 3, 2)
    r = torch.full((1, 4, 3), 0.05)
    lp, _ = ahead._learned_points(ahead.refine[0], h, p, r, R)
    lp0, _ = old._learned_points(old.refine[0], h, p, r, R)
    assert torch.allclose(lp, lp0, atol=1e-6)


def test_look_ahead_sees_far_past_the_ends():
    model = _model(look_ahead=True).eval()
    p = torch.tensor([[[[0.3, 0.5], [0.5, 0.5], [0.7, 0.5]]]])
    r = torch.full((1, 1, 3), 0.02)
    pts, coords = model._frame_points(p, r, R)
    px = 1.0 / (R - 1)
    caps = pts[0, 0, coords[:, 3] == 1]
    assert torch.allclose(caps[:, 1], torch.full_like(caps[:, 1], 0.5), atol=1e-6)  # on the stroke's line
    assert (caps[:, 0] > 0.7 + 0.02 + 31 * px).any() and (caps[:, 0] < 0.3 - 0.02 - 31 * px).any()
    # learned points can go past the ends, straight on along the tangent
    layer = model.refine[0]
    with torch.no_grad():
        layer["learned"].bias.view(-1, 3)[:, 0] = 20.0  # t -> 1.5
    lp, lc = model._learned_points(layer, torch.zeros(1, 1, 64), p, r, R)
    assert torch.allclose(lc[..., 0], torch.full_like(lc[..., 0], 1.5), atol=1e-4)
    want = 0.7 + 0.5 * (0.4 + 16 * px)
    assert torch.allclose(lp[..., 0], torch.full_like(lp[..., 0], want), atol=1e-4)


def test_strokes_start_from_the_proposal_where_they_land():
    torch.manual_seed(6)
    model = _model(stroke_proposals=True).eval()
    with torch.no_grad():
        model.propose[-1].bias.copy_(torch.tensor([0.0, math.atanh(0.2 / 0.35), 0.0, 0.0, 0.0]))  # half (0.2, 0)
    s = model(torch.rand(1, 12, R, R), views.whole_image_boxes(1), torch.zeros(1)).strokes
    chord = s.points[:, :, 0] - s.points[:, :, 2]
    assert torch.allclose(chord, torch.tensor([0.4, 0.0]).expand_as(chord), atol=0.01)  # decoder adds ~nothing
    # the score map moves the queries: a high score in one corner pulls them there
    with torch.no_grad():
        model.propose[-1].bias.zero_()
    x = torch.rand(1, 12, R, R)
    base = model(x, views.whole_image_boxes(1), torch.zeros(1)).anchors
    score = model.propose[-1]
    hook = score.register_forward_hook(lambda m, i, o: o + torch.cat(
        [20.0 * (torch.arange(o.shape[-1]) < 2).float().view(1, 1, 1, -1).expand(1, 1, o.shape[-2], -1),
         torch.zeros_like(o[:, 1:])], 1))
    pulled = model(x, views.whole_image_boxes(1), torch.zeros(1)).anchors
    hook.remove()
    assert (pulled[..., 0] < base[..., 0]).all() and (pulled[..., 0] < 0.15).all()


def test_gradients_reach_every_parameter_through_the_renderer():
    torch.manual_seed(3)
    model = _model(stroke_gate=True, stroke_proposals=True, look_ahead=True)
    model.mask_tool = True
    engine = DifferentiableCanvasEngine(canvas_size=48, min_opacity=0.0, opaque=True, eraser_mode="mask")
    target = torch.rand(2, 3, 48, 48)
    boxes = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.3, 0.6]])
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    for _ in range(2):  # zero-initialized branches only pass gradient after one update
        opt.zero_grad()
        x, layer = _inputs(engine, target, boxes)
        out = model(x, boxes, torch.tensor([0.3, 0.7]), images=(target, engine.composite(layer)))
        gated = views.Strokes(out.strokes.points, out.strokes.radii, out.strokes.opacity * out.gate.unsqueeze(-1),
                              out.strokes.hardness, out.strokes.color)
        tools = torch.nn.functional.gumbel_softmax(out.tool_logits[..., [0, 4]], hard=True)
        tool_w = torch.zeros(2, 4, 5)
        tool_w = tool_w.index_add(-1, torch.tensor([0, 4]), tools)
        params = views.to_engine_params(views.view_to_image(gated, boxes), engine)
        canvas = engine.composite(engine(layer, params, tool_w, joint=4))
        loss = (canvas - target).abs().mean() + 1e-3 * out.tool_logits.square().mean()
        loss.backward()
        opt.step()
    missing = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, missing


def test_checkpoints_build_and_paint_with_the_harness(tmp_path):
    from overpaintai.highres import HighResConfig, HighResPainter, load_checkpoint_model, render_strokes, stroke_record

    torch.manual_seed(4)
    cfg = dict(SMALL, arch="frame", joint_strokes=True, opaque_canvas=True, mask_tool=True, stroke_gate=True,
               hard_gate=True, paintable_mask=True, stationary=True)
    try:
        build_model(dict(cfg, joint_strokes=False))
        raise AssertionError("frame models need joint strokes")
    except ValueError:
        pass
    torch.save({"model": build_model(cfg).state_dict(), "ema": None, "model_config": cfg}, tmp_path / "f.pt")
    model, _ = load_checkpoint_model(str(tmp_path / "f.pt"), torch.device("cpu"))
    assert isinstance(model, FrameStrokePredictor) and model.opaque_canvas and model.mask_tool
    _spread(model, 1.0)
    forward = model.forward

    def every_other_a_mask(*args, **kw):  # tool choices as a trained model might make them
        out = forward(*args, **kw)
        out.tool_logits = out.tool_logits.clone()
        out.tool_logits[:, 0::2, 4] += 10.0
        out.tool_logits[:, 1::2, 0] += 10.0
        return out

    model.forward = every_other_a_mask

    hr = HighResConfig(whole_steps=2, rounds_per_level=1, max_tiles=2, error_threshold=0.0, min_core=0.25)
    painter = HighResPainter(model.eval(), size=64, view_res=R, config=hr, autocast=None)
    assert painter.tools == [0, 4] and painter.renderer.engine.opaque
    result = painter.paint(torch.rand(1, 3, 64, 64), source_side=64, valid=torch.ones(1, 1, 64, 64),
                           log=lambda *_: None)
    record = stroke_record(result)
    assert record["strokes"].shape[1] == 17 and record["opaque"] and record["mask_tool"]
    assert (record["strokes"][:, 16] == 4).any() and (record["strokes"][:, 16] == 0).any()
    redrawn = render_strokes(record["strokes"], 64, (64, 64), 64, min_radius=painter.min_radius,
                             max_radius=painter.max_radius, groups=[(int(n), True) for n in record["groups"]],
                             opaque=True, mask_tool=True)
    assert (redrawn - result.canvas).abs().mean() < 0.01
    assert math.isfinite(result.stats[-1]["psnr"])


def _counting(model):
    calls = []
    encode = model.encode_target

    def counted(*a, **k):
        calls.append(1)
        return encode(*a, **k)

    model.encode_target = counted
    return calls


def test_target_unet_is_encoded_once_per_unroll_and_learns_through_every_step():
    from overpaintai.painting import Painter

    torch.manual_seed(5)
    engine = DifferentiableCanvasEngine(canvas_size=64, min_opacity=0.0, opaque=True)
    target = torch.rand(2, 3, 64, 64)
    grads = []
    for ckpt in (False, True):
        torch.manual_seed(6)
        model = _model(target_unet=True, stroke_gate=True)
        model.joint_strokes = True
        _spread(model)
        calls = _counting(model)
        painter = Painter(model, engine, view_res=R, checkpoint_strokes=False)
        boxes = views.whole_image_boxes(2)
        painter.target_features(target, boxes)  # before the steps, as the unroll does
        layer = engine.create_blank_paint_layer(2, 64, 64)
        loss = 0.0
        for k in range(3):
            p = torch.full((2,), k / 3)
            if ckpt:
                res = torch.utils.checkpoint.checkpoint(painter.step, target, layer, p, use_reentrant=False)
            else:
                res = painter.step(target, layer, p)
            layer = res.layer
            loss = loss + (painter.composite(layer) - target).abs().mean()
        loss.backward()
        # one encoding for all steps and their recomputation, plus its own recomputation in backward
        assert len(calls) == 2, calls
        grads.append(model.t_stem.weight.grad.clone())
        assert grads[-1].abs().sum() > 0
    assert torch.allclose(grads[0], grads[1], atol=1e-5)  # checkpointed steps share the same encoding
    # a new target tensor is a new encoding; gradient mode is part of the key
    with torch.no_grad():
        painter.step(target.clone(), layer.detach(), torch.zeros(2))
        painter.step(target, layer.detach(), torch.zeros(2))
    assert len(calls) == 4


def test_harness_reuses_the_target_encoding():
    from overpaintai.highres import HighResConfig, HighResPainter

    torch.manual_seed(7)
    cfg = dict(SMALL, target_unet=True, stroke_gate=True, hard_gate=True, paintable_mask=True, stationary=True)
    model = _model(**{k: v for k, v in cfg.items() if k not in SMALL}).eval()
    model.joint_strokes = True
    calls = _counting(model)
    hr = HighResConfig(whole_steps=4, rounds_per_level=1, max_tiles=2, error_threshold=0.0, min_core=0.25,
                       tile_repeats=2)
    painter = HighResPainter(model, size=64, view_res=R, config=hr, autocast=None)
    result = painter.paint(torch.rand(1, 3, 64, 64), source_side=64, log=lambda *_: None)
    tile_rounds = sum(1 for st in result.stats if st["tiles"] > 0)
    # whole image once (shared with the tile rounds' global tokens), then once per tile set (both passes)
    assert len(calls) == 1 + tile_rounds, (len(calls), tile_rounds)


def test_teacher_proposal_loss_is_zero_for_the_ground_truth_in_either_direction():
    from types import SimpleNamespace

    from overpaintai.train import Trainer

    model = _model(stroke_proposals=True)
    p0, p1, p2 = torch.tensor([0.3, 0.4]), torch.tensor([0.5, 0.45]), torch.tensor([0.6, 0.6])
    half = 0.5 * (p0 - p2)
    normal = torch.stack([-half[1], half[0]]) / half.norm()
    bend = ((p1 - 0.5 * (p0 + p2)) * normal).sum()
    radius = torch.tensor(0.03)
    r_lo, r_hi = (math.log(r) for r in model.radius_range)
    u, u0 = (math.log(radius) - r_lo) / (r_hi - r_lo), (math.log(0.1) - r_lo) / (r_hi - r_lo)
    r_raw = math.log(u / (1 - u)) - math.log(u0 / (1 - u0))
    trainer = SimpleNamespace(model=model)
    for sign in (1.0, -1.0):  # the proposal as one direction or the other
        raw = torch.tensor([*torch.atanh(sign * half / 0.35), torch.atanh(sign * bend / 0.35), r_raw])
        maps = raw.view(1, 4, 1, 1).expand(1, 4, 8, 8).contiguous()
        gt = views.Strokes(points=torch.stack([p0, p1, p2]).view(1, 1, 3, 2), radii=radius.view(1, 1, 1).expand(1, 1, 3),
                           opacity=torch.ones(1, 1, 3), hardness=torch.ones(1, 1, 1), color=torch.ones(1, 1, 3))
        loss, length = Trainer.proposal_loss(trainer, maps, gt)
        assert loss.abs().max() < 1e-5 and torch.isclose(length, 2 * half.norm(), atol=1e-5)
        wrong, _ = Trainer.proposal_loss(trainer, maps.flip(1) * 0.5, gt)
        assert wrong.min() > 0.01
