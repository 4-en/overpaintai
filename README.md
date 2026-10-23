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
8. Optional pressure curve, `[0, 1]^16`: `r_mid, a_mid` appended. Radius and opacity then follow a quadratic profile along the stroke, with control values (start, mid, end) like the curve's control point, so a stroke can swell, taper at both ends or fade in and out. Without them the profile is linear (mid = mean of the ends gives exactly that), so 14-parameter strokes, old checkpoints and recorded `strokes.pt` files render as before.

### Discrete Tool Modes (5 Tools)

All 5 tools utilize the continuous capsule distance field and respect the stroke's `hardness` parameter:
- `0: Paint Brush` - Porter-Duff "Source Over" pigment deposition (transitions continuously from soft airbrush to hard round brush via `hardness`).
- `1: Dry / Textured Brush` - Pigment deposition modulated by procedural paper and bristle grain.
- `2: Smudge / Push` - Dab-sequential smear in premultiplied RGBA space. At each dab along the curve, the paint under the brush is replaced by the paint one step behind, moved forward by whole pixels, so pigment travels with the brush without blurring. Dragging in from bare canvas pushes paint away (an indent); dragging out of paint carries it along and deposits it. Stroke opacity controls how long paint is carried (lower = shorter, fading smear trail); `smudge_length` caps the drag distance at full opacity (default 12 brush radii), so the brush gradually loads up with paint it pushes into and runs out of paint it drags.
- `3: Blur / Blend` - In-place isotropic Gaussian softening of color and alpha boundaries underneath the brush footprint without directional pigment displacement.
- `4: Eraser` - Subtractive alpha removal (`A_next = A * (1 - eraser_mask)`), supporting both feathered soft erasing and razor-sharp cutouts to reveal the underlying background.
  - **Or a mask** (`eraser_mode="mask"`, `--tools paint,mask`, joint steps only): the union of a step's mask strokes keeps the canvas from before the step, so that step's paint, blur and smudge don't reach those pixels (like masking tape or a selection). Where a mask edge crosses a paint edge you get a sharp corner, even with round brushes; one mask across a line's end gives a flat end instead of a round cap.

**Opaque canvas** (`opaque=True`, default for `--arch frame`): a blank layer is the background itself at alpha 1 instead of transparent. Paint composites exactly the same; blur and smudge carry the paper's colour, and "missing" regions of degraded training starts are painted over with the background instead of made transparent. Models store `opaque_canvas` / `mask_tool` in their `model_config`, and every engine that renders their strokes (training, harness, export, timelapse via `strokes.pt`) is built to match.

### Joint Steps (order-independent strokes)

`engine(layer, strokes, tools, joint=N)` applies each group of `N` consecutive strokes as one operation (`forward_joint_step`), so the result doesn't depend on their order within the group:
1. **Smudge, blur and eraser edit the canvas as it was before the step**, never the paint of the same step. Each smudge stroke smears that canvas on its own and their changes add up (overlapping smudges don't drag each other's results along); blur and eraser make one pass each with the max of their strokes' masks.
2. **Paint and dry brush go on top:** per pixel, the coverage is the union of the strokes' brush shapes (`1 - prod(1 - shape)`, as if painted one after another) times their shape-weighted mean opacity (`sum(mask) / sum(shape)`), and the color is the coverage-weighted mean of their colors. That one layer is then painted "source over". Strokes on the same spot don't build up (two 0.5-opacity strokes give 0.5, not a darker 0.75 band), while soft edges side by side fill the gap between them as in normal painting, so a step can continue or widen a stroke without seams; different colors blend by coverage. Build-up stays possible across steps. A first version took the max of the masks as coverage: between two soft strokes it only had the stronger tail, which left a pale seam where the background showed through (coverage 0.68 instead of 0.85 one radius apart).

No tool reads another stroke of the same step, so the whole step is order-independent. A single stroke renders exactly as before, except that a smudge, blur or eraser stroke no longer acts on paint from its own step.

Models trained with `--joint-strokes` store `joint_strokes` in their `model_config`. Training, validation, the high-res harness and the web UI then render each view's strokes (a whole-image step or one tile) as one group. `strokes.pt` records the groups (`groups`, `joint`), so exports and timelapses replay them the same way; a joint group appears in a timelapse as a whole. Older records replay as before. It is also faster: 16 strokes x 16 images at 128 px take ~2.4x less time than one after another (forward and backward, checkpointed), with a higher per-group peak memory (~1.1 GB vs 130 MiB; the brush shapes are a second mask pass).

### Performance Notes

- The brush mask only keeps the winning curve segment per pixel for backward, and the smudge tool updates small per-dab windows with a hand-written backward pass, so memory scales with brush footprint rather than canvas size times curve samples.
- For long unrolls, pass `checkpoint_strokes=True` to `engine(...)`: only the paint layer between strokes is kept, and strokes are recomputed during backward (~1.5x time, memory roughly independent of `K`).
- The smudge dab math is fused with `torch.compile` on CUDA (`compile_smudge=True`); the first smudge stroke at a new shape triggers a few seconds of compilation.

## Predictor Architecture

Three predictors with the same inputs (12-13 channel 128 px view: target, canvas, error, paint alpha, coordinates, paintable mask) and outputs (N = 16 strokes per view and step), selected with `--arch` (stored in the checkpoint; checkpoints without it are `unet`):

- **`unet`** (`model.py`, `StrokePredictor`, 13.4M parameters as `small`): a residual U-Net. Its 16x16 bottleneck is pooled to 8x8 tokens, attended together with the whole image's tokens (2 blocks) and fed back into the bottleneck. 16 stroke queries coordinate and read the tokens (2 block pairs), then place themselves by a spatial soft-argmax over full-resolution pixel features. The attributes come from the features at that one point.
- **`plan`** (`plan_model.py`, `PlanStrokePredictor`, 26.3M parameters): plan first, then strokes that see each other.
  - **Plan:** the view's 16x16 bottleneck as tokens, the whole image as 8x8 tokens and 8 learned plan slots (6 self-attention blocks), fed back into the U-Net.
  - **Stroke latents:** query blocks (4 pairs), an initial position (soft-argmax as above, with a weaker, wider "paint where it's wrong" prior), then 3 refinement layers. Attention only ever works on the latents. One shared decoder turns any latent into a complete stroke. Each layer decodes its current strokes (detached) only to sample features at 5 points along them, adds those to the latents, then lets the strokes attend to each other (who covers what) and to the plan.
  - **Output:** the same decoder on the last latent, so every layer works on the stroke that will be painted. It has a pressure curve and a color that starts from the target along the stroke, averaged over about the stroke's width (single pixels for thin strokes, up to 33 px averages for wide ones; a fixed 9 px average blurred the colors of the thin late strokes on edges: switching a trained checkpoint over gave +0.36 dB at 128 steps without retraining). The anchor is the middle of the stroke's chord: start and end lie on opposite sides of it, and the control point bends the curve by at most half the maximum offset, so the curve passes within a quarter of it from the anchor. The curve no longer has to pass through the anchor exactly at its midpoint (that constraint made ~30% of strokes double back on themselves; a symmetric chord can't).
    - An earlier version made the anchor a free reference point: all three points bounded offsets of up to 45 px from it. Unroll training then moved the strokes away through the offsets (median 21-32 px from their anchors after the first step), while the placement map, the features and the base color stayed at the anchor. Colors went wrong (error 0.25-0.33, against 0.16 for the canvas they covered), and the model faded the strokes (opacity 0.08) and gated them off (gate 0.99 -> 0.41). The 128-step painting stopped improving after ~8 steps. Plan checkpoints from before (no `stroke_points` in their config) still load with this geometry (`free`).
    - With the anchor on the stroke, the next failure showed: within 400 unroll steps, all 16 placement maps converged on the same few pixels (pairwise Bhattacharyya overlap 0.75-0.89, nearest neighbour 1.5-3 px; the U-Net models are at 0.2-0.3). The free offsets had hidden this. `--placement-spread 1` penalizes pairs whose maps overlap more than 0.5. In a 400-step check from `plan_teacher_2`, it took the 64-step validation from 14.2 to 18.6 dB, and strokes shrank as the painting filled in (10.7 px at step 1, 1.6 px by step 16), as the U-Net's do. `overlap` (the mean over pairs) is always logged.
  - A first version kept the anchor on the curve at a predicted parameter t, solving the control point from it; that divides by 2t(1-t), t ran to its bound for every stroke and the control points left the canvas.
  - **Stability:** the per-layer decoded strokes are detached where they only decide where to look, as reference points are in deformable DETR decoders. Through the sampling positions the gradient is the feature difference between neighbouring pixels (unbounded), and stroke -> sampled features -> latent -> stroke is a feedback loop. The decoder learns through the final stroke.
  - **Cost:** ~1.2-1.3x the unet per training step (6-step unroll 1.9 vs 1.5 s, 32-step 8.1 vs 6.8 s per batch of 16).
- **`frame`** (`frame_model.py`, `FrameStrokePredictor`, 20.7M parameters; needs `--joint-strokes`): the plan model's ideas at the unet's speed, with high detail only where each stroke needs it.
  - **Why:** the full-resolution stages (`enc0` / `dec0`) took ~45% of the unet's time and most of its training memory, but their features were only read at 16 anchors. Strokes only knew what was at their anchor, so long strokes were risky (thin strokes came out as 4-5 px dashes) and step-1 placement was 1-2 px off.
  - **Coarse features everywhere:** a lossless stem (pixel-unshuffle, 128 -> 64 px), a U-Net down to 16 px and back up to 32 px only, the plan tokens as in `plan` (3 blocks), placement by soft-argmax on the 32x32 map. The paint-alpha channel is not read.
  - **Fine detail in each stroke's own frame:** position precision only matters at a stroke's edge, half a pixel from a thin stroke's centre line, r from a wide one's. In 3 refinement layers every stroke samples the raw images at 9 positions along it and across it: the interior (colour), an edge band at r +- 1 px (where paint stops), just outside (what to follow), plus its end caps and 8 learned points (offsets in the stroke's frame). The callers pass the full-resolution target and canvas (`forward(..., images=...)`), so in the high-res harness precision is bounded by the image, not the 128 px view. Each point reads target, canvas, error, luma gradients (where an edge is, to sub-pixel), the coarse features, and the **predicted result**: every stroke's coverage at that point, evaluated like the renderer but without rendering (`canvas_engine.profile_at` / `joint_at`), so a stroke sees whether it would miss a line, cut into an edge or overlap another stroke. The latent attends over its own ~140 point tokens, then to the other strokes, then to the plan. The colour starts from the target averaged over the stroke's interior at full resolution.
  - **Cached target U-Net** (`--target-unet`, on by default): the target doesn't change while a painting goes on, so it has its own U-Net, on the view box at 2x the view resolution (`--target-res 2`: 256 px, from the full-resolution image) with a decoder back up to it, conditioned only on the view size. It runs once per view box: once for all whole-image steps of an unroll or painting (also giving the tile rounds' whole-image tokens), once per tile set (reused by `--tile-repeats`). Its outputs at 64 / 32 / 16 px are added to the per-step stream, which then only reads canvas and error (through the lossless stem, so the error is compared at full detail at 64 px); its 256 px output is read by the stroke samples besides the raw values. In training it is encoded before the unroll's steps (outside their checkpoints), its gradients add up over all steps, and it is checkpointed itself (keeping its 256 px activations over a 12-step unroll cost +6 GB). Per step 22 ms with cached features (16 views; 25 ms without the target U-Net), plus 31 ms once per view box.
    - `--target-size` decouples what it reads from the canvas: with `--image-size 128 --target-size 256`, images are loaded (synthetic paintings rendered, touch-up details drawn) at 256 px, the U-Net encodes those, and canvas, loss, starts and the stroke samples' raw values use the area-downsampled 128 px version. A 12-step unroll then takes 2.1 s instead of 4.3 s at `--image-size 256` (4.1 vs 8.3 GB), the same as without high-res targets. The loss is still measured at 128 px, so 256 px detail pays off where it improves the 128 px result (exact edge positions, thin lines recognized as lines instead of grey); in the harness the canvas is full resolution anyway.
  - **Cost** (16 views at 128 px): inference 23-25 ms vs 23 ms (unet); a 12-step unroll at `--image-size 256` 3.8 s vs 3.9 s per batch, peak memory 8.0 vs 8.7 GB (with `--tools paint,mask` 4.24 vs 4.22 s, 8.3 vs 9.1 GB); with the target U-Net 4.02 s and 8.3 GB (these figures without it). The predicted coverage is fused with `torch.compile`; unfused it cost 15 ms per forward.

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

**Long paintings.** Inference runs hundreds to thousands of steps, far more than a training unroll. Three things make the model fit that:
- **Stationary model** (default for new models): it conditions on the canvas only, not on "step k of K". The progress input is the error-based progress (1 − current error / blank-canvas error) plus a log-scale version of it (tells 90% from 99% done), so step 3 and step 3000 are the same problem. `--step-progress` restores the old step conditioning (plus `--no-error-progress` to drop its extra error input).
- **Canvas pool** (`--pool-size 512`, off by default, needs a stationary model): unroll batches mostly continue paintings from earlier batches and put them back, as in the sample pool of *Growing Neural Cellular Automata*, so the model trains on canvases hundreds of steps deep that it painted itself (continuing, fixing its mistakes, stopping). Each image is replaced by a fresh `--start-mix` start with probability `--pool-reset` (0.05), and as soon as a visit gains less than `--pool-min-gain` dB (0.1): finished paintings only teach "don't touch it", and with most of each batch saying so the model first learned timid soft strokes, then switched all strokes off. At `--image-size 128` this keeps depths at a few dozen steps (paintings saturate there); real depth needs training on high-res canvases with tile views (planned). In a 3-epoch comparison from `with_edge_loss` it reached the same 128-step validation PSNR as without the pool (21.3–21.4 vs 21.2 dB) but painted softer strokes and did slightly worse through the high-res harness, so it stays off until training on high-res canvases. `--pool-fast-forward N` adds N gradient-free steps per visit to get deeper faster. Epochs log `pool_depth_mean/p90/max`. The pool is not saved in checkpoints (a resumed run refills it).
- **Relative loss** (default; `--no-relative-loss` for the plain one): each image's loss is measured relative to its loss at the start of the batch, so a deep painting with little error left gets as much signal as a blank one, plus `--worse-weight` × how much a step made the painting worse. The stroke cost is in the same units (a share of the error left), so fine strokes stay worth it late in a painting; an absolute cost outweighed everything a deep painting could still gain. Gradient norms are larger in these units (~15), so `--grad-clip 1` is always active, which Adam tolerates.
Validation also paints `--val-long-steps 128` whole-image steps and logs `val_long_psnr` (and at 8, 16, 32, ... steps), `val_long_drift` (PSNR lost from the best step by the end) and `val_long_end_strokes` (strokes still painted per step at the end); `best.pt` is then chosen by `val_long_psnr`.

New models (`--init-from` an older checkpoint copies its weights; new inputs / outputs start at zero, so training starts from the old behaviour) additionally have:
- a **stroke gate** per stroke, with an optional `--stroke-cost` per active stroke so steps leave out strokes they don't need (off by default: every cost tried so far eventually switched *all* strokes off, a trap the gate can't get out of, because off strokes get no gradient for position and color, so switching them back on never pays off; needs a better estimate of what a stroke is worth). The gate is on or off: sampled with the gate probability in training (straight-through gradient), thresholded at inference. A deterministic on/off gate only learns "off" (a well-placed stroke at full opacity has ~zero marginal benefit); sampled off, the gradient sees what the stroke is worth. `--soft-gate` makes it a dimmer as before, but then the cost makes dimming strokes cheaper than dropping them, which piles up faint strokes;
- a **paintable-mask** input: unroll tile steps get a core plus context margin (`--tile-margin`); strokes outside the core are faded as in the high-res harness, which also pulls them inward (`--outside-penalty` adds an explicit cost, off by default: at 0.1 the model learned to shrink tile strokes to the minimum radius); validation includes a tile step and logs `val_tile_gate` to catch tile views that stop painting;
- the stationary conditioning above.
Each can be switched off for new models (`--no-stroke-gate`, `--soft-gate`, `--no-paintable-mask`, `--step-progress`). Teacher mode keeps gate logits in range like unroll mode: before, the push towards "on" (with the gate's never-fading leaky gradient) grew the logits without bound, which saturated the stroke head (radii at the minimum) and blew up teacher training for any gated model within a few hundred steps at lr 3e-4. Loss options: `--edge-loss` (sharp, correctly placed edges) and `--flat-loss` (local variation matching, so flat areas stay flat instead of mottled). Looser detail: every pixel-exact loss prefers a blurred painting over one with cloned or slightly moved texture. `--texture-weight` matches texture statistics (per frequency band: energy, direction and coherence of the fine detail in soft windows), but alone a model exploits it with busy, wrongly colored strokes; `--shift-detail-weight` matches the fine detail patch by patch while allowing each patch to move up to `--shift-radius` px, so wrong detail costs more than none. Recommended together: `--detail-weight 0 --scales 16,32 --scale-weights 0.2,1 --shift-detail-weight 20 --texture-weight 40` (about 27 ms per loss call at 16 x 128 px, vs 3 ms). With these losses `best.pt` is picked by the validation loss instead of PSNR (`--select-by`), since PSNR prefers the soft, hedged paintings they train away from. `python -m overpaintai.scripts.loss_check loss_test --synthetic` scores losses on labeled edits of images and on painting pairs (see its docstring for the folder layout). Validation also logs low-band PSNR (both blurred by 2 px: are colors and shapes in place), the detail error (fine detail missing or wrong, allowing moves of up to 3 px), sharpness (painting vs target edge strength), line-art edge error, active strokes per image, the share of barely visible strokes and the PSNR gain per 100 strokes.

**Soft early strokes.** Early, big strokes come out soft even where the target edge is crisp. A probe (only the hardness of a trained model's first strokes optimized, positions fixed) showed the loss itself wants them soft, at 128 px and equally at 512 px: it is not the training resolution but hedging. A soft edge has a lower expected error than a crisp one that might be a few pixels off (L1 ~0.43, L2 ~0.38 optimal hardness; the edge loss only moves it to ~0.5). Three options against it:
- `--robust-loss` (`--robust-tau 0.1`): the full-resolution term becomes a saturating error `tau * e / (e + tau)` (L1-like for small errors, capped at tau), on lightness at full weight and on color at half. Painting a wrong color over another wrong color then costs almost nothing, breaking a pixel that was right costs a lot, so a big stroke that gets 60% of its area right is clearly worth it on a mostly-wrong (early) canvas and expensive on a mostly-right (late) one, as a painter blocks in and then refines. It is a score of the canvas, not of the per-step change, so breaking a pixel in one step and repairing it in the next gains nothing (an asymmetric "reward gains more than losses" per step could be farmed that way). A saturating error also prefers a crisp edge slightly off (few pixels fully wrong) to a soft one (many half wrong), but in the probe only a little (optimal hardness 0.43 -> 0.46 for first strokes, 0.48 -> 0.53 at step 3): it is mainly for the bold-strokes trade, crisp edges need the reward below. The coarse terms stay L1 for their pull.
- `--hard-edge-reward` (e.g. 0.05): pushes visible strokes towards hard where the target is sharp along their border (8 points on both sides and the end caps, in a band ~radius / 2 wide); strokes inside soft or flat areas are left free, even with an edge nearby. Logged as `soft_edges`. A first version that looked for the strongest edge anywhere within the stroke's radius (reward 0.1) made nearly every stroke hard (hardness 0.45 -> 0.96), glows and blurry backgrounds included, for -0.4 dB; the block-in did look much more like a painter's.
- `--forgive-edges`: the edge, sharp and flat terms count "too soft / too flat" always, but "too sharp / too busy" only where the colors around are already roughly right (weight `tau / (e + tau)`, e = 5 px mean color error, tau = `--robust-tau`, not differentiated so the model can't dodge it by being wrong). A crisp stroke over a still-wrong area is judged by its color; once the area is right, a spurious edge costs fully again.
- `--value-weight` (e.g. 0.5): a lightness-only L1 term over full resolution and the coarse scales, a squint test: value structure reads first, hue errors matter less.

Ablations (3 epochs from `with_edge_loss`, all with `--robust-loss --hard-edge-reward 0.05 --value-weight 0.5`; baseline = same run with none of these):

| run | hardness | sharpness | PSNR step 1 | 8 / 32 / 128 steps |
|---|---|---|---|---|
| baseline (soft) | 0.45 | 0.68 | 15.42 | 18.76 / 20.41 / 21.17 |
| + forgive, ramp 2, tau 0.1 | 0.85 | 0.83 | 13.85 | 17.92 / 19.63 / 20.24 |
| ramp 0 | 0.86 | 0.81 | 14.91 | 18.09 / 19.72 / 20.30 |
| tau 0.25 | 0.85 | 0.76 | 14.68 | 18.41 / 20.11 / 20.97 |
| no forgive | 0.85 | 0.71 | 14.72 | 18.48 / 20.21 / 21.17 |

`--forgive-edges` cost most of the quality (more faint strokes, messier detail, bare paper left early) and is not recommended; without it the hard-edged model matches the soft one at 128 steps. The step ramp hardly matters, a larger `--robust-tau` helps when forgiving is on.

**Model vs. renderer ceiling.** Greedy direct stroke fitting (64 strokes at a time, 150 Adam steps each, error-guided init, radius 0.12 -> 0.01 over the painting) on 16 validation images reaches 21.96 / 23.04 / 24.30 / 26.22 / 30.35 dB at 128 / 256 / 512 / 1024 / 2048 strokes and is still accelerating; the models reach ~19.5 / 20.5 / 21.3 / 22.0 / 22.3 dB. The renderer is not the limit, the predictor is. Measured stroke schedule of a model trained on 6-step unrolls: median radius 11 px at step 1, 2.8 px at step 4, 1.4 px at step 16, then sub-pixel dabs whose opacity falls to 0.15 by step 128. It compresses its coarse-to-fine schedule into the training horizon (96 strokes) and has nothing useful left to do afterwards.

`--checkpoint-steps` recomputes each painting step in backward (full gradient, 32 steps in 4.6 GB at batch 16, ~5.7 s per batch). 400 batches of 32-step unrolls from `abl_noforgive`: 19.38 / 20.41 / 21.34 / 22.18 / 22.75 dB (the same 400 batches at 6 steps: ... / 21.79 at 2048), opacity stays higher for longer (0.81 at step 16, 0.45 at step 64, vs 0.62 / 0.23). The horizon explains part of the plateau; the schedule stretches to about the training horizon and then decays again.

**Committing instead of hedging, judged over the whole window.** A stroke's position is the mean of its placement map (soft-argmax); with an entropy of ~5 nats that is an average over ~150 candidate pixels, and with two good spots it lands between them, where only a small, faint stroke is safe. `--best-of K` paints each training window K times with independent random draws, without gradients, and trains on each image's best attempt, replaying its draws exactly (`StrokeSamples`). Attempts are ranked by the loss at the *end* of the window, so a block-in that only pays off later wins over a safe detail stroke: no fitted or optimizer strokes, only the model's own. `best_of_gain` logs how much better the chosen attempt was than the average one (relative loss units). The draws come from the hard gate and `--placement-noise` (Gumbel noise on the placement logits in training, replayable through a per-image seed): each draw leans towards one of a map's good spots, while the position stays continuous and gets the renderer's gradient as before; no noise at inference. On `long32` (trained without noise), noise 1 costs ~0.05 dB, 4 seeds differ by ~0.3 dB per image, and the best of 4 is slightly above the noise-free painting.

A first version drew one pixel from the map instead (straight-through gradient of the mean). It collapsed within 200-400 batches from `long32`, with or without best-of (32-step validation 20.4 -> 13.0 / 12.4 dB): the 16 strokes of a step piled onto 3-6 spots. With a drawn pixel, the renderer's gradient no longer moves a stroke, it only shifts the map's mean, and the shared "paint where the error is largest" prior pulled every map onto the same few pixels; the soft-argmax had kept strokes apart through that direct gradient.

Trials from `long32`, 400 batches of 32-step unrolls each, PSNR on 16 validation images at 128 / 256 / 512 / 1024 / 2048 strokes:

| run | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| direct fitting (ceiling) | 21.96 | 23.04 | 24.30 | 26.22 | 30.35 |
| `long32` (start) | 19.38 | 20.41 | 21.34 | 22.18 | 22.75 |
| plain continuation (`cont32`) | 19.31 | 20.44 | 21.45 | 22.30 | 22.79 |
| `--placement-noise 1 --best-of 4` | 19.22 | 20.40 | 21.36 | 22.21 | 22.82 |
| `--global-blocks 6 --query-blocks 4` | 19.14 | 20.21 | 21.14 | 21.84 | 22.37 |

Best-of-4 is a wash at 1.7x the training cost: attempts differ by only ~0.3 dB per image, so the winner is picked mostly by chance. The deeper attention (8 extra blocks starting as identity, 19.75M parameters) is worse after this short budget at lr 1e-4 (placement entropy fell to 3.8 vs 4.9: the new blocks change placement but haven't paid off yet); not a verdict on depth, a fair test needs longer training or a higher learning rate for the new blocks.
- `--step-ramp` (e.g. 2): earlier unroll steps weigh `--intermediate-weight x ((k + 1) / steps) ** ramp`, so the first steps are judged more by where the painting ends up than by their own result.

Each run writes `best.pt` (best validation PSNR, the long one if measured; EMA weights included), `last.pt`, `epochs.csv` / `steps.csv`, `curve.png` and validation samples per epoch to `--out`.

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
- **Per-Pixel Max Alpha Within a Step**: implemented as joint steps (see the engine's *Joint Steps* and `--joint-strokes`); the A/B against standard compositing is still open.
- **Paintable-Area Mask Input**: an extra input channel per view marking where the model may paint (1) and what is context only (0): tile cores with surrounding context, padding of non-square images, tiles at the image border, and later user-locked regions. Whole-image views get an all-ones mask. In training, tile steps / zoomed crops get a random core inside the view; strokes reaching outside are faded exactly as in the inference harness (so training and inference agree and the gradient pulls them inward), plus a small penalty on footprint outside the mask so strokes don't waste strength fading at the border. The harness fade then becomes a rarely needed safety net.
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
`pip install -e ".[web]"`, then `overpaint-web` (open http://127.0.0.1:8000; `--host 0.0.0.0` for other machines, `--runs` for the checkpoint folder). Pick a checkpoint from `runs/`, open an image, set resolution, whole-image steps and tile settings, and optionally paint an importance mask over the image (brush / erase, upload a grayscale mask, or generate one for the subject / face, which is currently a placeholder ellipse). The painting updates live about once per second and after each stage; the result PNG, `strokes.pt` and `stats.json` can be downloaded and are also kept in `runs/web/<job id>/`. **Keep painting** continues a finished painting from its paint layer (more whole-image steps and / or further tile rounds and finer tiles), keeping the full stroke history. **Export at resolution** re-renders all strokes of the painting at up to 8192 px (longer side): strokes are resolution-independent, so this is a real upscale with crisp stroke edges, not an interpolated one (e.g. ~35 s for 4096 px, ~2 min for 8192 px on an RTX 3090). **Export video** replays all strokes of the painting (including continuations) as a 10-30 s speed-painting MP4 at 1080 px (needs `ffmpeg`); the first, composition-defining strokes get more screen time than the many small detail strokes. Jobs run one at a time on the GPU.

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
