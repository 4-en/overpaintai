# Overpaint AI

Differentiable stroke-based rendering engine in PyTorch for refining diffusion-generated images and performing resolution-independent digital art upscaling.

## Overview

Raw outputs from latent diffusion models often exhibit characteristic artifacts: waxy surface gradients, high-frequency pixel noise, inconsistent edge sharpness, and muddied blending. `overpaint-ai` addresses this by re-rendering images through a sequential, differentiable 2D painting bottleneck. Because strokes are parameterized in continuous normalized coordinates `[0, 1]`, a stroke sequence predicted at `256x256` can be evaluated analytically at `2048x2048` or `4096x4096` with crisp geometry.

## Canvas Engine Specification

The rendering core (`DifferentiableCanvasEngine`) separates the painting state into two tensors:
- **RGBA Paint Layer `(B, 4, H, W)`**: Stores straight RGB pigment (`0:3`) and paint opacity/thickness (`3`).
- **Background Layer `(B, 3, H, W)`**: Static RGB canvas color or procedural canvas weave texture.

### Continuous Stroke Parameters (14 Parameters)

Every stroke is defined by a vector in `[0, 1]^14`:
1. `x0, y0`: Quadratic Bezier start point.
2. `x1, y1`: Quadratic Bezier control point.
3. `x2, y2`: Quadratic Bezier end point.
4. `r0, r2`: Start and end brush radius (pressure taper).
5. `a0, a2`: Start and end opacity/flow taper.
6. `hardness`: Continuous edge hardness in `[0, 1]`, shared across all tools (`0.0` = soft Gaussian airbrush falloff, `1.0` = crisp cel/ink edge).
7. `R, G, B`: Stroke pigment color.

### Discrete Tool Modes (5 Tools)

All 5 tools utilize the continuous capsule distance field and respect the stroke's `hardness` parameter:
- `0: Paint Brush` - Porter-Duff "Source Over" pigment deposition (transitions continuously from soft airbrush to hard round brush via `hardness`).
- `1: Dry / Textured Brush` - Pigment deposition modulated by procedural paper and bristle grain.
- `2: Smudge / Push` - Dab-sequential smear in premultiplied RGBA space. At each dab along the curve, the paint under the brush is replaced by the paint one step behind, moved forward by whole pixels, so pigment travels with the brush without blurring. Dragging in from bare canvas pushes paint away (an indent); dragging out of paint carries it along and deposits it. Stroke opacity controls how long paint is carried (lower = shorter, fading smear trail); `smudge_length` caps the drag distance at full opacity (default 12 brush radii), so the brush gradually loads up with paint it pushes into and runs out of paint it drags.
- `3: Blur / Blend` - In-place isotropic Gaussian softening of color and alpha boundaries underneath the brush footprint without directional pigment displacement.
- `4: Eraser` - Subtractive alpha removal (`A_next = A * (1 - eraser_mask)`), supporting both feathered soft erasing and razor-sharp cutouts to reveal the underlying background.

### Performance Notes

- The brush mask only keeps the winning curve segment per pixel for backward, and the smudge tool updates small per-dab windows with a hand-written backward pass, so memory scales with brush footprint rather than canvas size times curve samples.
- For long unrolls, pass `checkpoint_strokes=True` to `engine(...)`: only the paint layer between strokes is kept, and strokes are recomputed during backward (~1.5x time, memory roughly independent of `K`).
- The smudge dab math is fused with `torch.compile` on CUDA (`compile_smudge=True`); the first smudge stroke at a new shape triggers a few seconds of compilation.

## Predictor Architecture

To balance global composition planning with 1:1 pixel accuracy for fine details:
- **Global Path**: Pretrained `ConvNeXt` encoder + Feature Pyramid Network (FPN) operating on the 9-channel state (`Target RGB`, `Composited Canvas RGB`, `Error Map`).
- **High-Resolution Detail Bypass**: Parallel `stride=1` shallow convolutional stem that preserves 1-pixel edges and lineart without downsampling.
- **Prediction Head**: Spatial Soft-Argmax for coordinate regression (`x0, y0, x1, y1, x2, y2`) paired with pooled feature projection for brush attributes (`r0, r2, a0, a2, hardness, R, G, B`) and tool logits.
- **Training Strategy**: Multi-step unrolled backpropagation ($K$ sequential strokes per gradient step) using temperature-annealed Softmax over tool selections.

## Installation & Usage

Install in editable mode from the repository root:

```bash
pip install -e .
```

Run the automated 6-panel visual test suite:

```bash
overpaint-test
```

Fit strokes directly to targets by gradient descent through the renderer (no network; synthetic targets by default, or `--image path`). Writes progress frames, a loss/PSNR curve and the fitted strokes to `--out`:

```bash
overpaint-fit --strokes 160 --stages 4 --out runs/fit
```

Train the stroke predictor (`overpaint-train --help` lists all options). Supervised pretraining on synthetic paintings with known strokes, then render-loss training over K painting steps, on synthetic data or a folder of images:

```bash
overpaint-train --mode teacher --epochs 20 --out runs/teacher
overpaint-train --mode unroll --init-from runs/teacher/best.pt --epochs 20 --out runs/unroll
overpaint-train --mode unroll --data path/to/images --init-from runs/unroll/best.pt --out runs/real
overpaint-train --resume runs/real/last.pt --epochs 40   # continue a run
```

To train later painting phases too, unroll batches can start from other canvases than a blank one: `--start-mix blank:0.4,warmup:0.3,degraded:0.3` mixes empty canvases, canvases the model first painted itself for up to `--rollout-max` steps (no gradients), and the target with synthetic painting mistakes (`overpaintai/degrade.py`: local blur, rough stroke repaints in local colors, color shifts, flattened or unpainted regions, lower detail). Progress conditioning always means "fraction of the painting done", so these starts count as steps already painted.

Each run writes `best.pt` (best validation PSNR, EMA weights included), `last.pt`, `epochs.csv` / `steps.csv`, `curve.png` and validation samples per epoch to `--out`.

Launch the interactive real-time mouse painting GUI:

```bash
overpaint-gui
```

## Roadmap

1. **Phase 1: Differentiable Canvas Engine (Completed)**
   - Continuous capsule Bezier distance field (zero circle-stamp beading).
   - 4-channel RGBA paint layer with Porter-Duff compositing and premultiplied dab-sequential smudge.
   - Unified continuous `hardness` attribute and in-place `Blur / Blend` tool.
   - Interactive Tkinter GUI and CLI test suite.
2. **Phase 2: Hybrid Stroke Predictor & Unrolled Training Loop**
   - Implement `HybridStrokePredictor` (`ConvNeXt` + FPN + full-resolution detail bypass).
   - Implement $K$-step unrolled backpropagation training loop with masked L1/L2 and perceptual losses.
3. **Phase 3: Arbitrary Aspect Ratio & High-Resolution Evaluation**
   - Aspect-ratio-preserving square padding with binary loss masking (`pad_to_square_with_mask`).
   - 4K analytical stroke replay and upscaling benchmark.

## Future Experiments (Post-Base Model)

Once the base model and unrolled training loop are validated, the following extensions are planned:
- **Glazing & Blending Mode Tools**: Adding multiplicative/screen/overlay glazing operations (`Multiply`, `Color Dodge`, `Overlay`) so the model can adjust lighting, cast shadows, and unify color casts across existing brushwork without painting over underlying stroke texture.
- **Custom Brush Shapes & Bristle Tip Footprints**: Expanding beyond an isotropic circular cross-section to support anisotropic flat/chisel brushes, bristle mark footprints, and irregular blocking-in stamps oriented along the curve tangent $B'(t)$.
- **Dynamic Brush Bank & Learnable Brush Embeddings**: Providing the model with a library of specialized digital art brushes (e.g., rake/bristle brushes, palette knives, stipple/sponge stamps, foliage/cloud blockers) conditioned via learnable **brush embeddings**, allowing the predictor to dynamically select and modulate custom brush tips per stroke.
- **Wet-on-Wet Impasto Mixing**: Simultaneous pigment deposition and tangent advection within a single stroke to simulate loaded bristle brushes mixing directly with wet canvas paint.

## High-Resolution Inference Harness

Inference-only scheduling for large canvases (e.g. 2048×2048), implemented in `overpaintai/highres.py`. The model only ever sees a view box `(x, y, s)` resampled to its working resolution, so how boxes are chosen is free; none of this requires retraining.

```bash
overpaint-paint --checkpoint runs/unroll/best.pt --image picture.png --size 2048
overpaint-paint ... --min-tile 0.0625            # tile finer than the source's own detail
overpaint-paint ... --importance face_mask.png  # spend more tiles where the mask is bright
```

Writes `final.png`, `target.png`, `progress.png` (target + canvas after each stage), `stats.json` (strokes / PSNR / time per stage) and `strokes.pt` (all strokes in image coordinates, re-renderable at any size) to `runs/paint/<image>_<size>/`. Tiling stops by default once a view would hold fewer source pixels than the model's working resolution. Implemented so far: whole-image steps, tile levels with error-guided selection and NMS, importance masks, context margins with soft fading at tile edges, and crop rendering (each stroke is drawn only into a square crop around its footprint). Non-square images are padded (edge-replicated, excluded from selection and PSNR) and cropped back; transparent images are composited onto white.

- **Coarse-to-fine schedule**: whole-image views first (blocking in), then tiles of decreasing size until one view pixel ≈ one target pixel. Every tile gets the global tokens of the whole image, so it knows the overall composition.
- **Error-guided tile selection**: instead of a fixed grid, blur the error map `|target − canvas|` at the current tile scale, pick the windows with the highest summed error, suppress heavily overlapping candidates (non-maximum suppression), paint them as one batch, re-measure, repeat. Regions that are already good get no more strokes; difficult regions get several passes.
- **Importance weighting**: multiply the error map by an importance map before selecting, e.g. from a face detector (including illustration/anime face detectors), subject segmentation (rembg, SAM), a saliency model, or a user-painted mask. The subject gets more and smaller tiles while the background stays looser.
- **Adaptive stopping**: stop when no (importance-weighted) region's error is above a threshold, so simple images finish early and detailed ones get more strokes.

### Web interface
`pip install -e ".[web]"`, then `overpaint-web` (open http://127.0.0.1:8000; `--host 0.0.0.0` for other machines, `--runs` for the checkpoint folder). Pick a checkpoint from `runs/`, open an image, set resolution, whole-image steps and tile settings, and optionally paint an importance mask over the image (brush / erase, upload a grayscale mask, or generate one for the subject / face, which is currently a placeholder ellipse). The painting updates live about once per second and after each stage; the result PNG, `strokes.pt` and `stats.json` can be downloaded and are also kept in `runs/web/<job id>/`. **Keep painting** continues a finished painting from its paint layer (more whole-image steps and / or further tile rounds and finer tiles), keeping the full stroke history. **Export video** replays all strokes of the painting (including continuations) as a 10-30 s speed-painting MP4 at 1080 px (needs `ffmpeg`); the first, composition-defining strokes get more screen time than the many small detail strokes. Jobs run one at a time on the GPU.

### Tile edges: don't paint blind, don't clip
A tile's strokes must not reach (far) past what the model can see: outside the view it can't judge what it would paint over, and could destroy finished work. Hard-clipping strokes at the tile border is not an option either, since it leaves visible hard edges and seams.
- **Context margin**: the model sees the tile plus a margin of surrounding canvas, but stroke anchors are restricted to the inner core (e.g. by masking the placement logits outside it).
- **Soft fade instead of clipping**: attenuate each stroke's opacity smoothly with how far its footprint (endpoints + radius) extends beyond the core, so strokes near the border fade out rather than being cut. In training this also gives a gradient that pulls strokes inward.
- **Train with the same rule**: apply the margin and fade to tile steps during training, so the model learns to keep its strokes inside the safe area instead of having them suppressed only at inference.
- **Overlapping tiles**: neighbouring tiles overlap, so seam regions are painted from both sides; tiles that overlap heavily go into different rounds, so they don't paint the same spot from the same stale canvas.

### Still open
- **Smaller tiles in training**: unroll trains tile views at 0.3–0.6 of the image (`--tile-size`), while 2048 px inference uses tiles down to ~1/16. The harness clamps the view size fed to the model to the trained range (the content still comes from the true, smaller box); training on smaller tiles would remove that mismatch.
- **Edge rule in training**: the margin + fade rule is only applied at inference so far.
- **Adaptive stopping per region** and automatic importance maps (face / subject detection) are not implemented yet; masks can be passed with `--importance`.
