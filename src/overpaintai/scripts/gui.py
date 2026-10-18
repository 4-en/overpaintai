import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from overpaintai.canvas_engine import DifferentiableCanvasEngine


class PaintGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("overpaint-ai: Interactive Tool Tester")
        self.root.geometry("820x620")
        self.root.resizable(False, False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.canvas_size = 256
        self.display_size = 512  # Scale up for easy mouse drawing

        # Initialize DifferentiableCanvasEngine
        self.engine = DifferentiableCanvasEngine(
            canvas_size=self.canvas_size,
            num_stamps=32,
            bg_color=(0.97, 0.96, 0.93),
            bg_texture_strength=0.06,
        ).to(self.device)
        self.engine.eval()

        self.background = self.engine.create_background(1, self.canvas_size, self.canvas_size, self.device, textured=True)
        self.paint_layer = self.engine.create_blank_paint_layer(1, self.canvas_size, self.canvas_size, self.device)

        # Stroke drawing state
        self.drawing = False
        self.x0, self.y0 = 0.0, 0.0
        self.x1, self.y1 = 0.5, 0.5
        self.x2, self.y2 = 0.0, 0.0

        # UI State variables
        self.tool_var = tk.IntVar(value=0)  # 0: Hard, 1: Soft, 2: Chalk, 3: Smudge, 4: Eraser
        self.color_rgb = (0.15, 0.45, 0.85)  # Default Blue
        self.radius_val = tk.DoubleVar(value=0.04)  # Brush size
        self.opacity_val = tk.DoubleVar(value=0.90)  # Flow / Opacity

        self._build_ui()
        self.update_canvas_display()

    def _build_ui(self):
        # Main layout frames
        left_frame = ttk.Frame(self.root, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        right_frame = ttk.Frame(self.root, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas display widget
        self.canvas_widget = tk.Canvas(left_frame, width=self.display_size, height=self.display_size, bg="#f0f0f0", cursor="cross")
        self.canvas_widget.pack()
        self.canvas_widget.bind("<ButtonPress-1>", self.on_press)
        self.canvas_widget.bind("<B1-Motion>", self.on_drag)
        self.canvas_widget.bind("<ButtonRelease-1>", self.on_release)

        # Controls panel on the right
        ttk.Label(right_frame, text="Tools", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        tools = [
            ("Hard Round Brush", 0),
            ("Soft Airbrush", 1),
            ("Dry Chalk / Bristle", 2),
            ("Smudge / Blender", 3),
            ("Hard Eraser", 4),
        ]
        for name, val in tools:
            ttk.Radiobutton(right_frame, text=name, variable=self.tool_var, value=val).pack(anchor=tk.W, pady=2)

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

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
            btn = tk.Button(palette_frame, text=name, width=8, bg=self._rgb_to_hex(rgb), fg="white" if rgb != (0.90, 0.75, 0.15) else "black", command=lambda c=rgb: set_color(c))
            btn.pack(side=tk.LEFT, padx=2)

        def set_color(rgb):
            self.color_rgb = rgb

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        ttk.Label(right_frame, text="Brush Size", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Scale(right_frame, from_=0.01, to=0.15, variable=self.radius_val, orient=tk.HORIZONTAL, length=200).pack(anchor=tk.W, pady=5)

        ttk.Label(right_frame, text="Opacity / Flow", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 0))
        ttk.Scale(right_frame, from_=0.1, to=1.0, variable=self.opacity_val, orient=tk.HORIZONTAL, length=200).pack(anchor=tk.W, pady=5)

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        ttk.Button(right_frame, text="Clear Canvas", command=self.clear_canvas).pack(fill=tk.X, pady=5)

    def _rgb_to_hex(self, rgb):
        return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"

    def pixel_to_norm(self, px, py):
        nx = max(0.0, min(1.0, px / float(self.display_size)))
        ny = max(0.0, min(1.0, py / float(self.display_size)))
        return nx, ny

    def on_press(self, event):
        self.drawing = True
        self.x0, self.y0 = self.pixel_to_norm(event.x, event.y)
        self.x1, self.y1 = self.x0, self.y0
        self.x2, self.y2 = self.x0, self.y0

    def on_drag(self, event):
        if not self.drawing:
            return
        nx, ny = self.pixel_to_norm(event.x, event.y)
        # Update control point to midpoint and end point to current mouse position
        self.x1 = (self.x0 + nx) * 0.5
        self.y1 = (self.y0 + ny) * 0.5
        self.x2, self.y2 = nx, ny

    def on_release(self, event):
        if not self.drawing:
            return
        self.drawing = False
        nx, ny = self.pixel_to_norm(event.x, event.y)
        self.x2, self.y2 = nx, ny
        if abs(self.x2 - self.x0) < 0.002 and abs(self.y2 - self.y0) < 0.002:
            # Make sure it's not a zero-length stroke
            self.x2 = min(1.0, self.x0 + 0.01)
            self.y2 = min(1.0, self.y0 + 0.01)

        self.apply_stroke()

    def apply_stroke(self):
        r = self.radius_val.get()
        a = self.opacity_val.get()
        R, G, B = self.color_rgb

        # Stroke format: [x0, y0, x1, y1, x2, y2, r0, r2, a0, a2, R, G, B]
        stroke_tensor = torch.tensor([[
            self.x0, self.y0, self.x1, self.y1, self.x2, self.y2,
            r, r, a, a, R, G, B
        ]], device=self.device)

        tool_idx = self.tool_var.get()
        tool_tensor = torch.zeros(1, 5, device=self.device)
        tool_tensor[0, tool_idx] = 1.0

        with torch.no_grad():
            self.paint_layer = self.engine.forward_single_step(
                self.paint_layer, stroke_tensor, tool_tensor
            )

        self.update_canvas_display()

    def clear_canvas(self):
        with torch.no_grad():
            self.paint_layer = self.engine.create_blank_paint_layer(1, self.canvas_size, self.canvas_size, self.device)
        self.update_canvas_display()

    def update_canvas_display(self):
        with torch.no_grad():
            rgb_out = self.engine.composite(self.paint_layer, self.background)
        
        # Convert to PIL Image and resize up to display size using nearest/bilinear
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