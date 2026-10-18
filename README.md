# Overpaint AI

**WIP**

Differentiable stroke-based rendering for AI artifact cleanup and digital art upscaling.

## Concept

Diffusion models denoise images globally in pixel space, frequently generating waxy surfaces, noisy micro-textures, and melted lineart.

`overpaint-ai` reconstructs images through a sequential digital painting engine (Bézier curves, pressure tapers, smudging, and erasing) running directly on the GPU. Limiting the total stroke budget acts as an information bottleneck: it strips high-frequency diffusion artifacts while keeping composition, forms, and sharp edges intact.

Because stroke parameters use normalized coordinates in $[0, 1]$, actions predicted on small inputs can be rendered onto high-resolution canvas grids ($2048\text{px}+$) for resolution-independent upscaling.

## Core Components

### 1. Differentiable Canvas Engine (`canvas_engine.py`)
A PyTorch module executing parametric stroke operations directly in tensor space:
* **Geometry**: Quadratic Bézier curves parameterized by three control points $(P_0, P_1, P_2)$ with linear interpolation for radius $(r_0 \rightarrow r_2)$ and opacity $(a_0 \rightarrow a_2)$.
* **Tools**:
  * `Hard Round`: Sharp sigmoid distance falloff for ink and cel edges.
  * `Soft Airbrush`: Gaussian falloff for smooth gradients and shadows.
  * `Dry / Chalk`: Stroke mask modulated by procedural paper tooth.
  * `Smudge`: Bilinear sampling along stroke tangents via `grid_sample`.
  * `Eraser`: Subtractive rendering back to the background color.
* **Optimization**: Vectorized operations allow $K$-step unrolled backpropagation. Gradients flow directly from pixel loss to stroke parameters without reinforcement learning.

### 2. Predictor Architecture
* **Input (9–11 channels)**: `Target`, `Canvas`, `Target - Canvas`, and optional coordinate channels.
* **Global Encoder**: Pretrained ConvNeXt-Tiny downsampling to an $8 \times 8$ bottleneck for macro-composition.
* **Detail Bypass**: Parallel shallow stride-1 convolution stem preserving 1-pixel lines.
* **Coordinate Head**: Spatial Soft-Argmax converting 2D attention heatmaps directly into sub-pixel $(x, y)$ control points.

## Implementation Plan

1. **Engine Verification**: Optimize raw stroke tensors directly against target images using `Adam` to calibrate brush dynamics and loss terms.
2. **Predictor Training**: Train the model on $128 \times 128$ illustration crops using $K=4\text{ to }8$ unrolled steps per iteration.
3. **Coarse-to-Fine Execution**:
   * Global pass ($1 \times 1$) with bounded large brush radii for blocking in.
   * Local patch passes ($2 \times 2$ and $4 \times 4$ grids) with small radii for lineart and focal points.
4. **Arbitrary-Scale Rendering**: Export stroke sequences to arbitrary resolution canvases with proportional edge sharpness and texture scaling.
