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
