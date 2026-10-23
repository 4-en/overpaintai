import math

import torch

from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.losses import MultiScalePaintLoss, psnr
from overpaintai.synthetic import radius_to_param, random_strokes, render_targets


def test_loss_is_zero_for_identical_images_and_reports_terms():
    loss_fn = MultiScalePaintLoss(detail_weight=1.0, scales=(4, 16))
    x = torch.rand(2, 3, 32, 32)
    total, terms = loss_fn(x, x)
    assert total.item() == 0.0
    assert set(terms) == {"detail", "x4", "x16"}


def test_coarse_terms_see_misplaced_strokes_that_detail_barely_does():
    """A dot shifted beyond its own width: full-res L1 is maxed out, but coarse terms still improve
    as it moves closer, so they provide a placement signal."""
    def dot(cx):
        img = torch.zeros(1, 3, 64, 64)
        img[..., 30:34, cx : cx + 4] = 1.0
        return img

    target = dot(10)
    coarse = MultiScalePaintLoss(detail_weight=0.0, scales=(16,))
    detail = MultiScalePaintLoss(detail_weight=1.0, scales=())
    far, near = dot(40), dot(20)
    assert detail(near, target)[0] == detail(far, target)[0]  # no overlap either way
    assert coarse(near, target)[0] < coarse(far, target)[0]
    # Plain pooling cannot tell them apart once the dot leaves its pooling cell
    pooled = MultiScalePaintLoss(detail_weight=0.0, scales=(16,), coarse_mode="pool")
    assert pooled(near, target)[0] == pooled(far, target)[0]


def test_edge_loss_prefers_sharp_edges_over_blur():
    def edge_at(c):
        x = torch.zeros(1, 3, 32, 32)
        x[..., c:] = 1.0  # one hard vertical edge
        return x

    def blur(x, k):
        return torch.nn.functional.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)

    target = edge_at(16)
    edge = MultiScalePaintLoss(detail_weight=0.0, scales=(), edge_weight=1.0)
    assert edge(target, target)[0] < 1e-6
    assert edge(blur(target, 3), target)[1].keys() == {"edge", "sharp"}
    # A sharp edge slightly off position beats a smeared edge (plain gradient L1 alone would not)
    assert edge(edge_at(17), target)[0] < edge(blur(target, 3), target)[0]
    assert edge(edge_at(18), target)[0] < edge(blur(target, 7), target)[0]
    assert edge(edge_at(20), target)[0] > edge(edge_at(18), target)[0]  # larger misplacement still costs
    assert MultiScalePaintLoss(edge_weight=0.0)(blur(target, 3), target)[1].keys() == {"detail", "x4", "x16"}


def test_masked_loss_ignores_padding():
    loss_fn = MultiScalePaintLoss(scales=(4,))
    pred, target = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., :8] = 1.0
    target_changed = target.clone()
    target_changed[..., 8:] = 0.0  # only inside the padding
    assert torch.allclose(loss_fn(pred, target, mask)[0], loss_fn(pred, target_changed, mask)[0])


def test_psnr_of_identical_images_is_high():
    x = torch.rand(2, 3, 8, 8)
    assert (psnr(x, x) > 90).all()


def test_random_strokes_render_valid_targets():
    engine = DifferentiableCanvasEngine(canvas_size=32).eval()
    gen = torch.Generator().manual_seed(0)
    params, tools = random_strokes(engine, 3, 20, generator=gen)
    assert params.shape == (3, 20, 14) and tools.shape == (3, 20, 5)
    assert params.min() >= 0.0 and params.max() <= 1.0
    assert torch.equal(tools.argmax(-1), torch.zeros(3, 20, dtype=torch.long))  # paint only by default
    # Coarse to fine: early strokes are larger than late ones on average
    assert params[:, :5, 6].mean() > params[:, -5:, 6].mean()
    img = render_targets(engine, params, tools, 32)
    assert img.shape == (3, 3, 32, 32) and img.min() >= 0.0 and img.max() <= 1.0


def test_radius_to_param_inverts_engine_mapping():
    engine = DifferentiableCanvasEngine(canvas_size=32)
    r = torch.tensor([0.01, 0.05, 0.2])
    p = radius_to_param(engine, r)
    assert torch.allclose(engine.min_radius + p * (engine.max_radius - engine.min_radius), r)


def test_degraded_canvas_is_a_partial_painting_of_the_target():
    from overpaintai.degrade import DEGRADATIONS, degraded_canvas

    engine = DifferentiableCanvasEngine(canvas_size=32).eval()
    gen = torch.Generator().manual_seed(0)
    params, tools = random_strokes(engine, 3, 20, generator=gen)
    target = render_targets(engine, params, tools, 32)
    background = engine.create_background(3, 32, 32)
    for kind in DEGRADATIONS:
        layer = degraded_canvas(target, engine, gen, kinds=(kind,))
        assert layer.shape == (3, 4, 32, 32) and layer.min() >= 0 and layer.max() <= 1
        canvas = engine.composite(layer, background)
        start = psnr(canvas, target)
        assert (start > 12).all(), kind  # close to the target overall ...
    layer = degraded_canvas(target, engine, gen, kinds=("missing",))
    assert (layer[:, 3] < 0.5).any()  # ... and "missing" leaves unpainted (transparent) areas


def test_touchup_pairs_differ_from_their_target_only_in_small_places():
    from overpaintai.degrade import touchup_pair

    engine = DifferentiableCanvasEngine(canvas_size=64).eval()
    gen = torch.Generator().manual_seed(0)
    params, tools = random_strokes(engine, 8, 30, generator=gen)
    target = render_targets(engine, params, tools, 64)
    background = engine.create_background(8, 64, 64)
    for add_share in (0.0, 1.0):
        layer, new_target, _ = touchup_pair(target, engine, gen, add_share=add_share)
        assert layer.shape == (8, 4, 64, 64) and new_target.shape == target.shape
        assert (layer[:, 3] == 1).all()  # a finished painting: nothing left unpainted
        canvas = engine.composite(layer, background)
        changed = ((canvas - new_target).abs().amax(1) > 0.1).float().mean((1, 2))
        assert (changed < 0.25).all() and (changed > 0).any()  # local differences, most of the image is right
        if add_share == 0.0:  # details on the canvas: the target is the original image
            assert torch.equal(new_target, target)
        else:  # details added to the target; the canvas is the (softened) original
            assert ((new_target - target).abs().amax((1, 2, 3)) > 0.05).any()
            assert (psnr(canvas, target) > 22).all()


def _strokes_texture(seed: int, size: int = 96) -> torch.Tensor:
    """Random short dark dashes on a light ground, like painted foliage."""
    g = torch.Generator().manual_seed(seed)
    img = torch.full((1, 3, size, size), 0.8)
    yy, xx = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    for _ in range(140):
        cy, cx = torch.rand(2, generator=g) * size
        a = torch.rand(1, generator=g).item() * 0.6 + 1.2  # mostly vertical
        u = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)
        v = -(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)
        dash = ((u.abs() < 4) & (v.abs() < 1.2)).float()
        img = img * (1 - dash) + dash * torch.tensor([0.2, 0.4, 0.1]).view(1, 3, 1, 1)
    return img


def test_texture_loss_prefers_moved_texture_over_blurred_texture():
    from overpaintai.losses import MultiScalePaintLoss, _gaussian_blur

    target = _strokes_texture(0)
    moved = torch.roll(target, (3, 5), (2, 3))  # the same marks, elsewhere
    blurred = _gaussian_blur(target, 1.5)
    l1 = MultiScalePaintLoss(1.0, ())
    tex = MultiScalePaintLoss(0.0, (32,), texture_weight=160)
    assert l1(moved, target)[0] > l1(blurred, target)[0]  # pixel-exact: blur looks better
    assert tex(moved, target)[0] < tex(blurred, target)[0]
    assert tex(target, target)[0] < 1e-6


def test_texture_loss_gradients_are_finite_on_flat_images():
    from overpaintai.losses import MultiScalePaintLoss

    tex = MultiScalePaintLoss(0.0, (32,), texture_weight=160, shift_detail_weight=20)
    for pred, target in [(torch.full((2, 3, 64, 64), 0.5), _strokes_texture(1, 64).expand(2, -1, -1, -1)),
                         (torch.full((2, 3, 64, 64), 0.5), torch.full((2, 3, 64, 64), 0.5))]:
        pred = pred.clone().requires_grad_(True)
        loss, terms = tex(pred, target, torch.ones(2, 1, 64, 64))
        loss.backward()
        assert torch.isfinite(pred.grad).all() and pred.grad.abs().max() < 10
        assert "texture" in terms


def test_loss_check_ranks_a_folder_of_examples(tmp_path, monkeypatch, capsys):
    import sys
    from PIL import Image
    from overpaintai.losses import _gaussian_blur
    from overpaintai.scripts import loss_check

    target = _strokes_texture(2, 160)
    save = lambda x, name: Image.fromarray((x[0].permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(
        tmp_path / name)
    save(target, "original.png")
    save(torch.roll(target, (2, 3), (2, 3)), "good_moved.png")
    save(_gaussian_blur(target, 2.0), "bad_blur.png")
    monkeypatch.setattr(sys, "argv", ["loss_check", str(tmp_path), "--downscales", "1", "--crops", "4",
                                      "--device", "cpu"])
    loss_check.main()
    out = capsys.readouterr().out
    worst = {line.split()[0]: float(line.split()[1]) for line in out.splitlines()[-4:] if line.startswith("  ")}
    assert worst["L1"] < 1 < worst["loose"]


def test_shift_tolerant_detail_frees_moved_detail_and_charges_wrong_detail():
    from overpaintai.losses import _gaussian_blur, shift_tolerant_detail

    target = _strokes_texture(3)
    g = torch.Generator().manual_seed(0)
    moved = torch.roll(target, (2, -3), (2, 3))
    blurred = _gaussian_blur(target, 2.0)
    busy = blurred + 0.5 * (_strokes_texture(4) - blurred.mean())  # other marks: right amount, wrong places
    cost = {k: shift_tolerant_detail(x, target).mean().item()
            for k, x in {"same": target, "moved": moved, "blurred": blurred, "busy": busy}.items()}
    assert cost["same"] < 1e-2 and cost["moved"] < 0.5 * cost["blurred"] < cost["busy"]


def test_shift_tolerant_detail_gradients_match_without_checkpointing():
    import torch.utils.checkpoint
    from overpaintai import losses

    target = _strokes_texture(5, 48)
    pred = (target.roll(1, 3) * 0.8 + 0.1).requires_grad_(True)
    losses.shift_tolerant_detail(pred, target).mean().backward()
    g_ckpt = pred.grad.clone()
    pred.grad = None
    orig = torch.utils.checkpoint.checkpoint
    torch.utils.checkpoint.checkpoint = lambda fn, *a, **k: fn(*a)
    try:
        losses.shift_tolerant_detail(pred, target).mean().backward()
    finally:
        torch.utils.checkpoint.checkpoint = orig
    assert torch.isfinite(g_ckpt).all() and torch.allclose(g_ckpt, pred.grad, atol=1e-6)
