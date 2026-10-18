from PIL import Image
import torch
from overpaintai.canvas_engine import DifferentiableCanvasEngine


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running RGBA canvas test on {device}...")

    # Warm, textured canvas background so the true Eraser and transparent RGBA paint layer are clearly visible
    engine = DifferentiableCanvasEngine(
        canvas_size=256,
        num_stamps=48,
        bg_color=(0.96, 0.94, 0.89),
        bg_texture_strength=0.06,
    ).to(device)
    engine.eval()

    # 1. Fixed textured background (1, 3, 256, 256) and blank transparent RGBA paint layer (1, 4, 256, 256)
    background = engine.create_background(batch_size=1, H=256, W=256, device=device, textured=True)
    empty_paint = engine.create_blank_paint_layer(batch_size=1, H=256, W=256, device=device)

    # 2. Underpaint two adjacent strokes (Blue left, Red right) onto the RGBA paint layer
    tool_hard = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], device=device)
    blue_stroke = torch.tensor([[0.15, 0.5, 0.35, 0.5, 0.52, 0.5, 0.6, 0.6, 1.0, 1.0, 0.15, 0.45, 0.85]], device=device)
    red_stroke = torch.tensor([[0.48, 0.5, 0.65, 0.5, 0.85, 0.5, 0.6, 0.6, 1.0, 1.0, 0.85, 0.25, 0.15]], device=device)

    paint_blue = engine.forward_single_step(empty_paint, blue_stroke, tool_hard)
    paint_two_color = engine.forward_single_step(paint_blue, red_stroke, tool_hard)

    # Coral test stroke across the canvas
    stroke = torch.tensor([[0.15, 0.20, 0.50, 0.65, 0.85, 0.25, 0.70, 0.25, 0.95, 0.95, 0.95, 0.25, 0.20]], device=device)

    composited_panels = []
    rgba_panels = []

    for idx in range(5):
        tw = torch.zeros(1, 5, device=device)
        tw[0, idx] = 1.0

        start_layer = paint_two_color.clone() if idx in (3, 4) else empty_paint.clone()
        out_paint_rgba = engine.forward_single_step(start_layer, stroke, tw)

        # Composite the RGBA paint layer over the fixed textured background
        out_rgb = engine.composite(out_paint_rgba, background)

        arr_rgb = (out_rgb.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
        composited_panels.append(Image.fromarray(arr_rgb, mode="RGB"))

        arr_rgba = (out_paint_rgba.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
        rgba_panels.append(Image.fromarray(arr_rgba, mode="RGBA"))

    strip_rgb = Image.new("RGB", (256 * 5, 256))
    strip_rgba = Image.new("RGBA", (256 * 5, 256))
    for i in range(5):
        strip_rgb.paste(composited_panels[i], (i * 256, 0))
        strip_rgba.paste(rgba_panels[i], (i * 256, 0))

    strip_rgb.save("output_test.png")
    strip_rgba.save("output_test_transparent_layer.png")
    print("Saved output_test.png (composited over canvas) and output_test_transparent_layer.png (raw RGBA paint layer)")


if __name__ == "__main__":
    main()