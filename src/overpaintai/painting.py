"""
Painting with a stroke predictor: one step = look at views of (target, canvas) -> predict strokes ->
render them onto the full-resolution paint layer. Shared by training, validation and inference.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from overpaintai import views
from overpaintai.canvas_engine import DifferentiableCanvasEngine
from overpaintai.model import StrokeOutput, StrokePredictor


@dataclass
class StepResult:
    layer: torch.Tensor  # (B, 4, H, W) paint layer after the step
    output: StrokeOutput  # raw model output (view coordinates), batch = B * views per image
    params: torch.Tensor  # (B, N_total, 14) engine params that were rendered
    tool_weights: torch.Tensor  # (B, N_total, 5)


class Painter:
    """
    Couples a `StrokePredictor` with a `DifferentiableCanvasEngine`.

    Args:
        view_res: working resolution R the model sees.
        tools: engine tool indices the model may use (0 = paint, ...). With a single tool, strokes use it
            directly; with several, tools are sampled with hard Gumbel-Softmax in training and argmax
            in eval.
        autocast: dtype for the network (e.g. torch.bfloat16) or None. The renderer always runs in fp32.
        checkpoint_strokes: recompute strokes in backward (see `DifferentiableCanvasEngine.forward`).
        grad_through_inputs: let gradients flow into the canvas the model *looks at*. Off by default: over a
            multi-step unroll this chains the network's input Jacobian once per step (like an RNN) and
            explodes. Earlier strokes still get gradients through the renderer, where later strokes are
            composited on top of them.
    """

    def __init__(
        self,
        model: StrokePredictor,
        engine: DifferentiableCanvasEngine,
        view_res: int = 128,
        tools: Sequence[int] = (0,),
        autocast: Optional[torch.dtype] = None,
        checkpoint_strokes: bool = True,
        gumbel_tau: float = 1.0,
        grad_through_inputs: bool = False,
    ):
        self.model = model
        self.engine = engine
        self.view_res = view_res
        self.tools = list(tools)
        self.autocast = autocast
        self.checkpoint_strokes = checkpoint_strokes
        self.gumbel_tau = gumbel_tau
        self.grad_through_inputs = grad_through_inputs
        self._background = {}

    def background(self, B: int, S: int, device: torch.device) -> torch.Tensor:
        key = (S, device)
        if key not in self._background:
            self._background[key] = self.engine.create_background(1, S, S, device)
        return self._background[key].expand(B, -1, -1, -1)

    def composite(self, layer: torch.Tensor) -> torch.Tensor:
        return self.engine.composite(layer, self.background(layer.shape[0], layer.shape[-1], layer.device))

    def _autocast(self, device: torch.device):
        enabled = self.autocast is not None and device.type == "cuda"
        return torch.autocast(device.type, dtype=self.autocast or torch.bfloat16, enabled=enabled)

    def tool_weights(self, logits: torch.Tensor) -> torch.Tensor:
        n_tools = self.engine.NUM_TOOLS
        if len(self.tools) == 1:
            w = torch.zeros(*logits.shape[:-1], n_tools, device=logits.device)
            w[..., self.tools[0]] = 1.0
            return w
        mask = torch.full((n_tools,), float("-inf"), device=logits.device)
        mask[self.tools] = 0.0
        masked = logits + mask
        if self.model.training:
            return F.gumbel_softmax(masked, tau=self.gumbel_tau, hard=True, dim=-1)
        return F.one_hot(masked.argmax(-1), n_tools).to(logits.dtype)

    def global_tokens(self, target: torch.Tensor, layer: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        B = target.shape[0]
        if not self.grad_through_inputs:
            layer = layer.detach()
        boxes = views.whole_image_boxes(B, target.device)
        x = views.view_inputs(target, self.composite(layer), layer[:, 3:], boxes, self.view_res)
        with self._autocast(target.device):
            return self.model.encode_global(x, progress)

    def predict(
        self,
        target: torch.Tensor,
        layer: torch.Tensor,
        boxes: torch.Tensor,
        progress: torch.Tensor,
        global_tokens: Optional[torch.Tensor] = None,
    ) -> StrokeOutput:
        """Model output for one view per image (`boxes` (B, 3))."""
        if not self.grad_through_inputs:
            layer = layer.detach()
        x = views.view_inputs(target, self.composite(layer), layer[:, 3:], boxes, self.view_res)
        with self._autocast(target.device):
            return self.model(x, boxes, progress, global_tokens=global_tokens)

    def render(self, layer: torch.Tensor, params: torch.Tensor, tool_w: torch.Tensor) -> torch.Tensor:
        return self.engine(layer, params, tool_w, checkpoint_strokes=self.checkpoint_strokes)

    def step(
        self,
        target: torch.Tensor,
        layer: torch.Tensor,
        progress: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
    ) -> StepResult:
        """
        One painting step. `boxes` is (B, 3) for one view per image (default: whole image), or (B, T, 3)
        for T tile views per image; tiles get the global tokens of the whole image and their strokes are
        rendered in tile order onto the same canvas.
        """
        B = target.shape[0]
        if boxes is None:
            boxes = views.whole_image_boxes(B, target.device)
        if boxes.dim() == 2:
            out = self.predict(target, layer, boxes, progress)
            flat_boxes, T = boxes, 1
        else:
            T = boxes.shape[1]
            g = self.global_tokens(target, layer, progress).repeat_interleave(T, dim=0)
            flat_boxes = boxes.reshape(B * T, 3)
            out = self.predict(
                target.repeat_interleave(T, 0), layer.repeat_interleave(T, 0), flat_boxes,
                progress.repeat_interleave(T, 0), global_tokens=g,
            )
        strokes = views.view_to_image(out.strokes, flat_boxes)
        params = views.to_engine_params(strokes, self.engine)
        tool_w = self.tool_weights(out.tool_logits)
        N = params.shape[1]
        params = params.reshape(B, T * N, -1)
        tool_w = tool_w.reshape(B, T * N, -1)
        return StepResult(layer=self.render(layer, params, tool_w), output=out, params=params, tool_weights=tool_w)

    @torch.no_grad()
    def paint(
        self,
        target: torch.Tensor,
        num_steps: int,
        tile_steps: int = 0,
        tiles_per_side: int = 2,
        layer: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Inference: `num_steps` whole-image steps, then `tile_steps` steps on a grid of tiles. Returns the
        paint layer after every step (the last one is the result).
        """
        B, _, S, _ = target.shape
        layer = self.engine.create_blank_paint_layer(B, S, S, target.device) if layer is None else layer
        total = num_steps + tile_steps
        tiles = views.tile_boxes(tiles_per_side, device=target.device).unsqueeze(0).expand(B, -1, -1)
        layers = []
        for k in range(total):
            progress = torch.full((B,), k / total, device=target.device)
            boxes = None if k < num_steps else tiles
            layer = self.step(target, layer, progress, boxes).layer
            layers.append(layer)
        return layers
