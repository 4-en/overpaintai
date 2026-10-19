import json

import numpy as np
import pytest
import torch
from PIL import Image

from overpaintai import views
from overpaintai.data import ImageFolderDataset
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


TINY = (
    "--model-size tiny --image-size 32 --view-res 16 --strokes-per-step 4 --target-strokes 8,12 --steps 2 "
    "--batch-size 4 --steps-per-epoch 3 --val-size 4 --warmup-steps 1 --log-every 1 --num-workers 0 "
    "--device cpu --amp none"
).split()


def test_training_modes_checkpoints_and_resume(tmp_path, image_folder):
    teacher = tmp_path / "teacher"
    main(TINY + ["--mode", "teacher", "--epochs", "1", "--out", str(teacher)])
    for name in ("best.pt", "last.pt", "epochs.csv", "steps.csv", "curve.png", "args.json", "samples/epoch_000.png"):
        assert (teacher / name).exists(), name

    unroll = tmp_path / "unroll"
    main(TINY + ["--mode", "unroll", "--tile-prob", "1", "--epochs", "1", "--init-from", str(teacher / "best.pt"),
                 "--out", str(unroll)])
    main(TINY + ["--mode", "unroll", "--epochs", "2", "--resume", str(unroll / "last.pt"), "--out", str(unroll)])
    ckpt = torch.load(unroll / "last.pt", weights_only=False)
    assert ckpt["epoch"] == 1 and ckpt["global_step"] == 6 and len(ckpt["history"]) == 2
    assert "val_psnr" in ckpt["history"][-1]

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
