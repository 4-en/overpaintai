import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from overpaintai.canvas_engine import DifferentiableCanvasEngine


class PaintGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("overpaint-ai: Interactive Tool Tester")
        self.root.geometry("840x660")
        self.root.resizable(False, False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.canvas_size = 256
        self.display_size = 512

        self.engine = DifferentiableCanvasEngine(
            canvas_size=self.canvas_size,
            num_stamps=16,
            bg_color=(0.97, 0.96, 0.93),
            bg_texture_strength=0.06,
        ).to(self.device)
        self.engine.eval()

        self.background = self.engine.create_background(1, self.canvas_size, self.canvas_size, self.device, textured=True)
        self.paint_layer = self.engine.create_blank_paint_layer(1, self.canvas_size, self.canvas_size, self.device)

        self.drawing = False
        self.last_x, self.last_y = 0.0, 0.0

        # UI State variables
        self.tool_var = tk.IntVar(value=0)       # 0: Paint, 1: Dry/Chalk, 2: Smudge, 3: Blur/Blend, 4: Eraser
        self.color_rgb = (0.15, 0.45, 0.85)
        self.radius_val = tk.DoubleVar(value=0.04)
        self.opacity_val = tk.DoubleVar(value=0.90)
        self.hardness_val = tk.DoubleVar(value=0.80)

        self._build_ui()
        self.update_canvas_display()

    def _build_ui(self):
        left_frame = ttk.Frame(self.root, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        right_frame = ttk.Frame(self.root, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas_widget = tk.Canvas(
            left_frame, width=self.display_size, height=self.display_size, bg="#f0f0f0", cursor="cross"
        )
        self.canvas_widget.pack()
        self.canvas_widget.bind("<ButtonPress-1>", self.on_press)
        self.canvas_widget.bind("<B1-Motion>", self.on_drag)
        self.canvas_widget.bind("<ButtonRelease-1>", self.on_release)

        ttk.Label(right_frame, text="Tools (All use Hardness)", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        tools = [
            ("Paint Brush (Soft <-> Hard)", 0),
            ("Dry Chalk / Bristle", 1),
            ("Smudge / Material Pusher", 2),
            ("Blur / Blend (In-Place Soften)", 3),
            ("Eraser (Soft <-> Hard)", 4),
        ]
        for name, val in tools:
            ttk.Radiobutton(right_frame, text=name, variable=self.tool_var, value=val).pack(anchor=tk.W, pady=2)

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        ttk.Label(right_frame, text="Color Palette", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        palette_frame = ttk.Frame(right_frame)
        palette_frame.pack(anchor=tk.W, pady=5)

        colors = [
            ("Blue", (0.15, 0.45, 0.85)),
            ("Red", (0.85, 0.25, 0.15)),
            ("Green", (0.15, 0.65, 0.35)),
            ("Yellow", (0.90, 0.75, 0.15)),
            ("Black", (0.10, 0.10, 0.10)),
        ]
        for name, rgb in colors:
            btn = tk.Button(
                palette_frame,
                text=name,
                width=8,
                bg=self._rgb_to_hex(rgb),
                fg="white" if rgb != (0.90, 0.75, 0.15) else "black",
                command=lambda c=rgb: set_color(c),
            )
            btn.pack(side=tk.LEFT, padx=2)

        def set_color(rgb):
            self.color_rgb = rgb

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        ttk.Label(right_frame, text="Brush Size", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Scale(right_frame, from_=0.01, to=0.12, variable=self.radius_val, orient=tk.HORIZONTAL, length=220).pack(
            anchor=tk.W, pady=4
        )

        ttk.Label(right_frame, text="Hardness (0 = Soft Airbrush, 1 = Crisp Edge)", font=("Arial", 10, "bold")).pack(
            anchor=tk.W, pady=(8, 0)
        )
        ttk.Scale(right_frame, from_=0.0, to=1.0, variable=self.hardness_val, orient=tk.HORIZONTAL, length=220).pack(
            anchor=tk.W, pady=4
        )

        ttk.Label(right_frame, text="Opacity / Flow", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(8, 0))
        ttk.Scale(right_frame, from_=0.1, to=1.0, variable=self.opacity_val, orient=tk.HORIZONTAL, length=220).pack(
            anchor=tk.W, pady=4
        )

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        ttk.Button(right_frame, text="Load Test Dual-Pill Scene", command=self.load_test_scene).pack(fill=tk.X, pady=4)
        ttk.Button(right_frame, text="Clear Canvas", command=self.clear_canvas).pack(fill=tk.X, pady=4)

    def _rgb_to_hex(self, rgb):
        return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"

    def pixel_to_norm(self, px, py):
        nx = max(0.0, min(1.0, px / float(self.display_size)))
        ny = max(0.0, min(1.0, py / float(self.display_size)))
        return nx, ny

    def on_press(self, event):
        self.drawing = True
        self.last_x, self.last_y = self.pixel_to_norm(event.x, event.y)
        self._apply_segment(self.last_x, self.last_y, self.last_x + 0.001, self.last_y + 0.001)

    def on_drag(self, event):
        if not self.drawing:
            return
        curr_x, curr_y = self.pixel_to_norm(event.x, event.y)
        dx = curr_x - self.last_x
        dy = curr_y - self.last_y
        dist = (dx * dx + dy * dy) ** 0.5

        min_step = 0.015
        if dist >= min_step:
            self._apply_segment(self.last_x, self.last_y, curr_x, curr_y)
            self.last_x, self.last_y = curr_x, curr_y

    def on_release(self, event):
        self.drawing = False

    def _apply_segment(self, x0, y0, x2, y2):
        x1 = (x0 + x2) * 0.5
        y1 = (y0 + y2) * 0.5
        # Slider is the actual normalized radius; the engine expects r in [0, 1] mapped onto [min, max]
        e = self.engine
        r = (self.radius_val.get() - e.min_radius) / (e.max_radius - e.min_radius)
        a = self.opacity_val.get()
        h = self.hardness_val.get()
        R, G, B = self.color_rgb

        # 14-param stroke format: [x0, y0, x1, y1, x2, y2, r0, r2, a0, a2, hardness, R, G, B]
        stroke_tensor = torch.tensor([[
            x0, y0, x1, y1, x2, y2,
            r, r, a, a, h, R, G, B
        ]], device=self.device)

        tool_idx = self.tool_var.get()
        tool_tensor = torch.zeros(1, 5, device=self.device)
        tool_tensor[0, tool_idx] = 1.0

        with torch.no_grad():
            self.paint_layer = self.engine.forward_single_step(
                self.paint_layer, stroke_tensor, tool_tensor
            )
        self.update_canvas_display()

    def load_test_scene(self):
        with torch.no_grad():
            paint = self.engine.create_blank_paint_layer(1, self.canvas_size, self.canvas_size, self.device)
            tool_paint = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], device=self.device)
            blue_stroke = torch.tensor([[0.18, 0.5, 0.28, 0.5, 0.36, 0.5, 0.55, 0.55, 1.0, 1.0, 1.0, 0.15, 0.45, 0.85]], device=self.device)
            red_stroke = torch.tensor([[0.64, 0.5, 0.72, 0.5, 0.82, 0.5, 0.55, 0.55, 0.70, 0.70, 1.0, 0.85, 0.25, 0.15]], device=self.device)
            paint = self.engine.forward_single_step(paint, blue_stroke, tool_paint)
            self.paint_layer = self.engine.forward_single_step(paint, red_stroke, tool_paint)
        self.update_canvas_display()

    def clear_canvas(self):
        with torch.no_grad():
            self.paint_layer = self.engine.create_blank_paint_layer(1, self.canvas_size, self.canvas_size, self.device)
        self.update_canvas_display()

    def update_canvas_display(self):
        with torch.no_grad():
            rgb_out = self.engine.composite(self.paint_layer, self.background)
        arr = (rgb_out.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
        img = Image.fromarray(arr, mode="RGB").resize((self.display_size, self.display_size), resample=Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas_widget.create_image(0, 0, anchor=tk.NW, image=self.photo)


def main():
    root = tk.Tk()
    app = PaintGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()