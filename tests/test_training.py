import math
import json

import numpy as np
import pytest
import torch
from PIL import Image

from overpaintai import views
from overpaintai.data import ImageFolderDataset, LogUniformSquareCrop
from overpaintai.matching import matched_stroke_loss
from overpaintai.train import box_mask, lr_lambda, main


def _strokes(B=2, N=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return views.Strokes(
        points=torch.rand(B, N, 3, 2, generator=g),
        radii=0.01 + 0.1 * torch.rand(B, N, 2, generator=g),
        opacity=0.05 + 0.95 * torch.rand(B, N, 2, generator=g),
        hardness=torch.rand(B, N, 1, generator=g),
        color=torch.rand(B, N, 3, generator=g),
    )


def test_matching_is_order_and_direction_invariant():
    gt = _strokes()
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    pred = views.Strokes(
        points=gt.points[:, perm].flip(2),  # permuted and every stroke reversed
        radii=gt.radii[:, perm].flip(-1),
        opacity=gt.opacity[:, perm].flip(-1),
        hardness=gt.hardness[:, perm],
        color=gt.color[:, perm],
    )
    loss, assign = matched_stroke_loss(pred, gt)
    assert loss.item() < 1e-6
    assert torch.equal(assign[1][0], perm)  # prediction i is matched to target perm[i]


def test_matching_loss_has_gradients_and_handles_unequal_counts():
    pred = _strokes(N=6, seed=1)
    for t in vars(pred).values():
        t.requires_grad_(True)
    loss, assign = matched_stroke_loss(pred, _strokes(N=4, seed=2))
    loss.backward()
    assert assign.shape == (2, 2, 4)
    assert pred.points.grad.abs().sum() > 0


def test_box_mask_and_lr_schedule():
    m = box_mask(torch.tensor([[0.0, 0.0, 0.5]]), 9)
    assert m[0, 0, :5, :5].all() and m[0, 0, 5:].sum() == 0 and m[0, 0, :, 5:].sum() == 0
    f = lr_lambda(warmup=10, total=110, min_ratio=0.1)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(60) == pytest.approx(0.55) and f(110) == pytest.approx(0.1) and f(500) == pytest.approx(0.1)


@pytest.fixture
def image_folder(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(12):
        sub = tmp_path / ("a" if i % 2 else "b")
        sub.mkdir(exist_ok=True)
        arr = (rng.random((40 + 3 * i, 56, 3)) * 255).astype("uint8")
        Image.fromarray(arr).save(sub / f"img{i}.{'png' if i % 3 else 'jpg'}")
    (tmp_path / "notes.txt").write_text("not an image")
    return tmp_path


def test_image_folder_split_and_augmentation(image_folder):
    train = ImageFolderDataset(str(image_folder), 32, "train", val_fraction=0.3)
    val = ImageFolderDataset(str(image_folder), 32, "val", val_fraction=0.3)
    assert len(train) + len(val) == 12 and len(val) > 0
    assert not set(train.files) & set(val.files)
    assert set(val.files) == set(ImageFolderDataset(str(image_folder), 32, "val", val_fraction=0.3).files)
    x = train[0]
    assert x.shape == (3, 32, 32) and x.min() >= 0 and x.max() <= 1
    assert torch.equal(val[0], val[0])  # validation crops are deterministic


def test_log_uniform_crops_cover_all_zoom_levels():
    torch.manual_seed(0)
    crop = LogUniformSquareCrop(16, (0.01, 1.0))
    img = torch.arange(200 * 300, dtype=torch.float32).view(1, 200, 300).expand(3, -1, -1)
    assert crop(img).shape == (3, 16, 16)
    sides = []
    for _ in range(400):
        lo, hi = (np.log(v) for v in crop.side_range)
        sides.append(np.exp(lo + (hi - lo) * torch.rand(()).item()))
    sides = np.array(sides)
    assert 0.1 <= sides.min() and sides.max() <= 1.0
    assert 0.25 < np.median(sides) < 0.40  # geometric mean of 0.1 and 1 is ~0.32
    assert (sides < 0.2).mean() > 0.2  # small crops are common, not ~3% as with uniform-in-area


TINY = (
    "--model-size tiny --image-size 32 --view-res 16 --strokes-per-step 4 --target-strokes 8,12 --steps 2 "
    "--batch-size 4 --steps-per-epoch 3 --val-size 4 --warmup-steps 1 --log-every 1 --num-workers 0 "
    "--device cpu --amp none --val-long-steps 8"
).split()


def test_training_modes_checkpoints_and_resume(tmp_path, image_folder):
    teacher = tmp_path / "teacher"
    main(TINY + ["--mode", "teacher", "--epochs", "1", "--out", str(teacher)])
    for name in ("best.pt", "last.pt", "epochs.csv", "steps.csv", "curve.png", "args.json", "samples/epoch_000.png"):
        assert (teacher / name).exists(), name

    unroll = tmp_path / "unroll"
    main(TINY + ["--mode", "unroll", "--tile-prob", "1", "--epochs", "1", "--init-from", str(teacher / "best.pt"),
                 "--edge-loss", "--out", str(unroll)])
    main(TINY + ["--mode", "unroll", "--epochs", "2", "--resume", str(unroll / "last.pt"), "--out", str(unroll)])
    ckpt = torch.load(unroll / "last.pt", weights_only=False)
    assert ckpt["epoch"] == 1 and ckpt["global_step"] == 6 and len(ckpt["history"]) == 2
    assert "val_psnr" in ckpt["history"][-1]

    mixed = tmp_path / "mixed"
    main(TINY + ["--mode", "unroll", "--start-mix", "blank:1,warmup:1,degraded:1", "--rollout-max", "2",
                 "--epochs", "1", "--steps-per-epoch", "6", "--out", str(mixed)])
    hist = torch.load(mixed / "last.pt", weights_only=False)["history"][-1]
    assert "train_start_psnr" in hist and "train_gain" in hist

    folder = tmp_path / "folder"
    preview = next(image_folder.rglob("*.png"))
    main(TINY + ["--mode", "unroll", "--data", str(image_folder), "--val-fraction", "0.3", "--epochs", "1",
                 "--preview-image", str(preview), "--preview-steps", "3", "--out", str(folder)])
    for name in ("init.png", "epoch_001.png", "latest.png"):
        assert (folder / "previews" / name).exists(), name
    assert "preview_psnr" in torch.load(folder / "last.pt", weights_only=False)["history"][-1]
    assert json.loads((folder / "args.json").read_text())["data"] == str(image_folder)

    with pytest.raises(SystemExit):
        main(TINY + ["--mode", "teacher", "--data", str(image_folder), "--out", str(tmp_path / "bad")])


def test_new_losses_tile_cores_and_quality_metrics(tmp_path):
    out = tmp_path / "features"
    main(TINY + ["--mode", "unroll", "--tile-prob", "1", "--flat-loss", "--edge-loss", "--stroke-cost", "0.05",
                 "--outside-penalty", "0.2", "--epochs", "1", "--out", str(out)])
    hist = torch.load(out / "last.pt", weights_only=False)["history"][-1]
    for key in ("train_flat", "train_edge", "train_gate", "train_outside", "val_sharpness", "val_strokes",
                "val_faint_share", "val_gain_per_100", "val_lineart_edge_err", "val_tile_gate"):
        assert key in hist, key
    cfg = torch.load(out / "last.pt", weights_only=False)["model_config"]
    assert cfg["stroke_gate"] and cfg["paintable_mask"] and cfg["stationary"] and cfg["hard_gate"]


def test_canvas_pool_continues_paintings_and_long_validation(tmp_path):
    from overpaintai.train import Trainer, parse_args

    trainer = Trainer(parse_args(TINY + ["--mode", "unroll", "--pool-size", "8", "--pool-reset", "0", "--pool-min-gain", "-1000", "--tile-prob", "0",
                                         "--start-mix", "blank:1,degraded:1", "--out", str(tmp_path / "pool")]))
    assert trainer.pool is not None
    for _ in range(2):  # batch size 4: two batches fill the pool, all fresh
        trainer.train_step()
    assert trainer.pool.filled == 8
    first_depth = trainer.pool.depth.clone()
    stats = [trainer.train_step() for _ in range(4)]  # no resets: every batch continues pool paintings
    assert all(st["depth"] > 0 for st in stats)
    assert trainer.pool.depth.sum() == first_depth.sum() + 4 * 4 * 2  # 4 batches x 4 images x 2 steps
    assert "worse" in stats[0]
    long = trainer.validate_long()
    for key in ("val_long_psnr", "val_long_psnr@8", "val_long_drift", "val_long_end_strokes"):
        assert key in long, key
    assert long["val_long_drift"] >= 0

    old = Trainer(parse_args(TINY + ["--mode", "unroll", "--step-progress", "--out", str(tmp_path / "old")]))
    assert old.pool is None and not old.model.stationary  # step-conditioned models can't use the pool


def test_per_sample_loss_matches_the_batch_loss():
    from overpaintai.losses import MultiScalePaintLoss

    loss_fn = MultiScalePaintLoss(edge_weight=1.0, flat_weight=1.0)
    pred, target = torch.rand(3, 3, 32, 32), torch.rand(3, 3, 32, 32)
    mask = box_mask(torch.tensor([[0.1, 0.2, 0.5]] * 3), 32)
    for m in (None, mask):
        total, _ = loss_fn(pred, target, m)
        per, _ = loss_fn(pred, target, m, per_sample=True)
        assert per.shape == (3,) and torch.allclose(per.mean(), total, rtol=1e-4)


def test_painting_metrics_sanity():
    from overpaintai.metrics import painting_metrics

    t = torch.rand(2, 3, 32, 32)
    blank = torch.ones_like(t)
    m = painting_metrics(t.clone(), t, blank, torch.tensor([[0.5, 0.01, 0.1], [0.9, 0.9, 0.9]]),
                         torch.tensor([[3.0, 3.0, 0.5], [4.0, 4.0, 4.0]]))
    assert m["psnr"] > 90 and abs(m["sharpness"] - 1) < 1e-4 and m["edge_err"] < 1e-6
    assert m["strokes"] == 2.5  # (2 active + 3 active) / 2: the 0.01-opacity stroke does not count
    assert abs(m["faint_share"] - 0.25) < 1e-6  # image 1: 1 of 2 active strokes is faint (thin); image 2: none


def test_robust_loss_saturates_and_counts_lightness_over_hue():
    from overpaintai.losses import robust_color_error

    t = torch.full((1, 3, 4, 4), 0.5)
    err = lambda d: robust_color_error(t + torch.tensor(d).view(1, 3, 1, 1), t, 0.1).mean().item()
    assert abs(err([0.01] * 3) - 0.01) < 2e-3  # small gray error: like L1
    assert err([0.5] * 3) < 0.1 and err([0.5] * 3) - err([0.3] * 3) < 0.01  # wrong vs. other wrong: ~same
    assert err([0.1, -0.1, 0.0]) < 0.5 * err([0.1] * 3)  # a hue change at equal lightness counts less
    # a big stroke 60% right / 40% wrong on a mostly-wrong canvas beats doing nothing
    before = torch.cat([torch.full((1, 3, 6, 1), 0.9), torch.full((1, 3, 4, 1), 0.9)], 2)
    after = torch.cat([torch.full((1, 3, 6, 1), 0.5), torch.full((1, 3, 4, 1), 0.1)], 2)
    target = torch.full((1, 3, 10, 1), 0.5)
    assert robust_color_error(after, target, 0.1).mean() < robust_color_error(before, target, 0.1).mean()


def test_stroke_edge_sharpness_looks_at_the_stroke_border():
    from overpaintai.losses import sharpness_pyramid, stroke_edge_sharpness

    target = torch.zeros(1, 3, 64, 64)
    target[..., 32:] = 1.0  # one vertical edge at x ~ 0.5
    pyr = sharpness_pyramid(target)
    r = 8 / 63
    edge_x = 31.5 / 63
    side = torch.tensor([[edge_x - r, 0.3], [edge_x - r, 0.5], [edge_x - r, 0.7]])  # vertical, right side on the edge
    near = torch.tensor([[edge_x - 4 / 63, 0.5]] * 3)  # a dot over the edge, 4 px from its center and border
    far = torch.tensor([[0.1, 0.5]] * 3)
    s = stroke_edge_sharpness(pyr, torch.stack([side, near, far]).unsqueeze(0), torch.full((1, 3, 2), r))
    assert s[0, 0] > 0.5 and s[0, 1] < 0.3 and s[0, 2] < 0.05


def test_value_loss_and_forgiving_edges():
    from overpaintai.losses import MultiScalePaintLoss

    target = torch.full((1, 3, 32, 32), 0.5)
    hue = target + torch.tensor([0.2, -0.2, 0.0]).view(1, 3, 1, 1)  # same lightness, other hue
    value = MultiScalePaintLoss(0.0, (4,), scale_weights=(0.0,), value_weight=1.0)
    assert value(hue, target)[0] < 1e-6 and value(target + 0.2, target)[0] > 0.19

    edged = target.clone()
    edged[..., 16:] = 0.6  # a crisp edge the flat target doesn't have
    wrong = edged.clone() + 0.4  # the same edge over an area that is far off in color
    strict = MultiScalePaintLoss(0.0, (), edge_weight=1.0)
    lenient = MultiScalePaintLoss(0.0, (), edge_weight=1.0, forgive_tau=0.1)
    assert torch.allclose(lenient(edged, target)[0], strict(edged, target)[0], rtol=0.6)  # nearly right: still counts
    assert lenient(wrong, target)[0] < 0.3 * strict(wrong, target)[0]  # far off anyway: mostly forgiven
    soft = MultiScalePaintLoss(0.0, (), edge_weight=1.0, forgive_tau=0.1)
    assert soft(target, edged)[0] > 0.9 * strict(target, edged)[0]  # too soft is never forgiven


def test_hardness_reward_and_step_ramp_train(tmp_path):
    from overpaintai.train import Trainer, parse_args

    trainer = Trainer(parse_args(TINY + ["--mode", "unroll", "--robust-loss", "--hard-edge-reward", "0.1",
                                         "--step-ramp", "2", "--edge-loss", "--flat-loss", "--forgive-edges",
                                         "--value-weight", "1", "--out", str(tmp_path / "r")]))
    stats = trainer.train_step()
    assert "soft_edges" in stats and 0.0 <= stats["soft_edges"] <= 1.0 and math.isfinite(stats["loss"])


def test_checkpointed_steps_give_the_same_gradient(tmp_path):
    from overpaintai.train import Trainer, parse_args

    grads = []
    for extra in ([], ["--checkpoint-steps"]):
        torch.manual_seed(0)
        trainer = Trainer(parse_args(TINY + ["--mode", "unroll", "--tile-prob", "0", "--seed", "3",
                                             "--out", str(tmp_path / f"c{len(extra)}")] + extra))
        trainer.model.train()
        trainer.unroll_batch()["loss"].backward()
        grads.append(torch.cat([q.grad.flatten() for q in trainer.model.parameters() if q.grad is not None]))
    assert torch.allclose(grads[0], grads[1], rtol=1e-3, atol=1e-6)


def test_best_of_replays_each_images_best_attempt(tmp_path):
    from overpaintai.losses import psnr  # noqa: F401
    from overpaintai.train import Trainer, parse_args

    trainer = Trainer(parse_args(TINY + ["--mode", "unroll", "--placement-noise", "1", "--best-of", "3",
                                         "--tile-prob", "0", "--out", str(tmp_path / "b")]))
    trainer.model.train()
    target, layer, depth, _ = trainer.batch_start(trainer.args.steps)
    total = (depth + trainer.args.steps).float()
    forced = trainer._best_of_draws(target, layer, depth, total, trainer.args.steps, lambda k: (None, None))
    with torch.no_grad():
        for k in range(trainer.args.steps):
            layer = trainer.painter.step(target, layer, (depth + k).float() / total, samples=forced[k]).layer
        replay = trainer.loss_fn(trainer.painter.composite(layer), target, per_sample=True)[0]
    assert torch.allclose(replay, trainer._best_of_scores.amin(0), rtol=1e-4, atol=1e-6)
    stats = trainer.train_step()
    assert stats["best_of_gain"] >= 0 and math.isfinite(stats["loss"])


def test_plan_architecture_trains_in_teacher_and_unroll_mode(tmp_path):
    plan = ["--arch", "plan", "--global-blocks", "1", "--query-blocks", "1", "--refine-layers", "1",
            "--view-tokens", "2", "--plan-slots", "2"]
    teacher = tmp_path / "teacher"
    main(TINY + plan + ["--mode", "teacher", "--epochs", "1", "--out", str(teacher)])
    cfg = torch.load(teacher / "best.pt", weights_only=False)["model_config"]
    assert cfg["arch"] == "plan" and cfg["pressure"] and cfg["refine_layers"] == 1

    unroll = tmp_path / "unroll"
    main(TINY + ["--mode", "unroll", "--init-from", str(teacher / "best.pt"), "--tile-prob", "1", "--edge-loss",
                 "--robust-loss", "--hard-edge-reward", "0.05", "--checkpoint-steps", "--placement-spread", "1",
                 "--epochs", "1", "--out", str(unroll)] + plan)
    ckpt = torch.load(unroll / "last.pt", weights_only=False)
    hist = ckpt["history"][-1]
    assert "val_long_psnr" in hist and math.isfinite(hist["train_loss"]) and 0 <= hist["train_overlap"] <= 1
    assert ckpt["model_config"]["stroke_points"] == "chord"


def test_placement_overlap_finds_strokes_on_the_same_spot():
    from overpaintai.train import placement_overlap

    logits = torch.full((1, 3, 64), -20.0)
    logits[0, 0, 5] = logits[0, 1, 5] = logits[0, 2, 40] = 20.0  # strokes 0 and 1 on the same pixel
    mean, crowd = placement_overlap(logits.requires_grad_())
    assert abs(mean.item() - 1 / 3) < 1e-4 and abs(crowd.item() - 1 / 6) < 1e-4
    flat = placement_overlap(torch.zeros(1, 3, 64))[0]  # identical uniform maps overlap fully
    assert abs(flat.item() - 1.0) < 1e-4


def test_teacher_stroke_terms_fade_once_the_pixels_are_right(tmp_path):
    from overpaintai.train import Trainer, parse_args

    pred, gt = _strokes(N=4, seed=3), _strokes(N=4, seed=4)
    per, _ = matched_stroke_loss(pred, gt, per_sample=True)
    total, _ = matched_stroke_loss(pred, gt)
    assert per.shape == (pred.points.shape[0],) and torch.allclose(per.mean(), total)

    trainer = Trainer(parse_args(TINY + ["--mode", "teacher", "--out", str(tmp_path / "t")]))
    stats = trainer.train_step()
    assert 0.0 <= stats["relax"] <= 1.0 and math.isfinite(stats["loss"])
    off = Trainer(parse_args(TINY + ["--mode", "teacher", "--teacher-relax", "0", "--out", str(tmp_path / "o")]))
    assert off.train_step()["relax"] == 1.0
