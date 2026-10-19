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
- `2: Smudge / Push` - Multi-tap tangent advection in premultiplied RGBA space. Transports existing pigment along the curve tangent without diluting exiting paint or carving holes on bare canvas.
- `3: Blur / Blend` - In-place isotropic Gaussian softening of color and alpha boundaries underneath the brush footprint without directional pigment displacement.
- `4: Eraser` - Subtractive alpha removal (`A_next = A * (1 - eraser_mask)`), supporting both feathered soft erasing and razor-sharp cutouts to reveal the underlying background.

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

Launch the interactive real-time mouse painting GUI:

```bash
overpaint-gui
```

## Roadmap

1. **Phase 1: Differentiable Canvas Engine (Completed)**
   - Continuous capsule Bezier distance field (zero circle-stamp beading).
   - 4-channel RGBA paint layer with Porter-Duff compositing and premultiplied tangent smudge.
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
