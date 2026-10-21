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
