from PIL import Image
import torch
from overpaintai.canvas_engine import DifferentiableCanvasEngine


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running 14-param RGBA canvas test on {device}...")

    engine = DifferentiableCanvasEngine(
        canvas_size=256,
        num_stamps=48,
        bg_color=(0.96, 0.94, 0.89),
        bg_texture_strength=0.06,
    ).to(device)
    engine.eval()

    background = engine.create_background(batch_size=1, H=256, W=256, device=device, textured=True)
    empty_paint = engine.create_blank_paint_layer(batch_size=1, H=256, W=256, device=device)

    # Underpaint two non-overlapping adjacent strokes (Blue left, Red right) with hard Paint Brush (tool 0, hardness=1.0)
    # 14-param format: [x0, y0, x1, y1, x2, y2, r0, r2, a0, a2, hardness, R, G, B]
    tool_paint = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], device=device)
    blue_stroke = torch.tensor([[0.18, 0.5, 0.28, 0.5, 0.36, 0.5, 0.181, 0.181, 1.0, 1.0, 1.0, 0.15, 0.45, 0.85]], device=device)
    red_stroke = torch.tensor([[0.64, 0.5, 0.72, 0.5, 0.82, 0.5, 0.181, 0.181, 0.75, 0.75, 1.0, 0.85, 0.25, 0.15]], device=device)

    paint_blue = engine.forward_single_step(empty_paint, blue_stroke, tool_paint)
    paint_two_color = engine.forward_single_step(paint_blue, red_stroke, tool_paint)

    # 6-Panel Showcase:
    # 0: Paint Brush (Hard, hardness=1.0)
    # 1: Paint Brush (Soft Airbrush, hardness=0.0)
    # 2: Dry / Textured Brush (hardness=0.8)
    # 3: Smudge / Push (drags blue into red and out onto bare canvas, hardness=0.2)
    # 4: Blur / Blend (softens the seam and edges in place without dragging, hardness=0.2)
    # 5: Eraser (soft-to-hard cutout across both shapes, hardness=0.85)
    panel_configs = [
        (0, 1.0, False),  # Hard Paint Brush
        (0, 0.0, False),  # Soft Paint Brush
        (1, 0.8, False),  # Dry / Chalk Brush
        (2, 0.2, True),   # Smudge / Push
        (3, 0.2, True),   # Blur / Blend
        (4, 0.85, True),  # Eraser
    ]

    composited_panels = []
    rgba_panels = []

    for tool_idx, hardness_val, use_underpaint in panel_configs:
        tw = torch.zeros(1, 5, device=device)
        tw[0, tool_idx] = 1.0

        stroke = torch.tensor([[
            0.15, 0.22, 0.50, 0.68, 0.88, 0.28,
            0.214, 0.082, 0.95, 0.95, hardness_val,
            0.95, 0.25, 0.20
        ]], device=device)

        start_layer = paint_two_color.clone() if use_underpaint else empty_paint.clone()
        out_paint_rgba = engine.forward_single_step(start_layer, stroke, tw)

        out_rgb = engine.composite(out_paint_rgba, background)
        arr_rgb = (out_rgb.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
        composited_panels.append(Image.fromarray(arr_rgb, mode="RGB"))

        arr_rgba = (out_paint_rgba.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
        rgba_panels.append(Image.fromarray(arr_rgba, mode="RGBA"))

    num_panels = len(panel_configs)
    strip_rgb = Image.new("RGB", (256 * num_panels, 256))
    strip_rgba = Image.new("RGBA", (256 * num_panels, 256))
    for i in range(num_panels):
        strip_rgb.paste(composited_panels[i], (i * 256, 0))
        strip_rgba.paste(rgba_panels[i], (i * 256, 0))

    strip_rgb.save("output_test.png")
    strip_rgba.save("output_test_transparent_layer.png")
    print("Saved output_test.png and output_test_transparent_layer.png")


if __name__ == "__main__":
    main()
