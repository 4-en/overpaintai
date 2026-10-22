import torch
from PIL import Image, ImageDraw

from overpaintai.web.detection import (
    detect_faces,
    generate_auto_mask,
    generate_face_mask,
    generate_people_mask,
    generate_subject_mask,
    get_blazeface,
    visual_saliency,
)


def test_blank_image_fallbacks():
    blank = Image.new("RGB", (100, 80), (128, 128, 128))
    dev = torch.device("cpu")

    # Face fallback
    mask_face, fb_face = generate_face_mask(blank, dev)
    assert fb_face is True
    assert mask_face.size == (100, 80) and mask_face.mode == "L"
    # Upper-middle ellipse should be brightest in upper half
    assert mask_face.getpixel((50, 30)) > mask_face.getpixel((50, 70))

    # Subject fallback
    mask_subj, fb_subj = generate_subject_mask(blank, dev)
    assert fb_subj is True
    assert mask_subj.size == (100, 80) and mask_subj.mode == "L"
    # Centered ellipse should be brightest in center
    assert mask_subj.getpixel((50, 40)) > mask_subj.getpixel((5, 5))

    # People fallback
    mask_peop, fb_peop = generate_people_mask(blank, dev)
    assert fb_peop is True
    assert mask_peop.size == (100, 80) and mask_peop.mode == "L"


def test_visual_saliency_and_subject_detection():
    # High-contrast subject in the center
    img = Image.new("RGB", (128, 128), (10, 10, 10))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 88, 88], fill=(240, 220, 200))

    sal = visual_saliency(img)
    assert sal.shape == (128, 128)
    assert sal[64, 64] > sal[5, 5]

    mask, is_fb = generate_subject_mask(img, torch.device("cpu"))
    assert is_fb is False
    assert mask.size == (128, 128)
    assert mask.getpixel((64, 64)) > mask.getpixel((5, 5))


def test_blazeface_model_loading_and_detection():
    dev = torch.device("cpu")
    net = get_blazeface(dev)
    assert net is not None

    # Empty image should return 0 detections
    empty = Image.new("RGB", (128, 128), (0, 0, 0))
    dets = detect_faces(net, empty)
    assert len(dets) == 0


def test_generate_auto_mask_dispatch():
    img = Image.new("RGB", (64, 64), (100, 100, 100))
    dev = torch.device("cpu")

    for mode in ("subject", "face", "people", "person"):
        m, _ = generate_auto_mask(img, mode, dev)
        assert m.size == (64, 64) and m.mode == "L"

    try:
        generate_auto_mask(img, "invalid_mode", dev)
        assert False, "Should raise ValueError for invalid mode"
    except ValueError:
        pass
