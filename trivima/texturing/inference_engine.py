"""
Inference engine — runtime coordinator for the AI texturing pipeline.

Orchestrates: buffer render → dirty mask → AI model → cell write-back → temporal blend

Three modes:
  - realtime:   Pix2PixHD-Lite GAN, <10ms/frame, 100+ FPS
  - production:  StreamDiffusion + ControlNet, 15-50ms/frame
  - cinematic:   Full SDXL diffusion, 2-10s/frame

Fallback: if AI model is too slow for the current frame, reuse previous
frame's light values (stored in cells). This guarantees frame rate never
drops below the raw cell renderer's speed.
"""

import time
import numpy as np
from typing import Optional, Literal
from enum import Enum

from .buffer_renderer import CellBufferRendererGPU, RenderBuffers
from .cell_writeback import writeback_light_to_cells
from .temporal import TemporalConsistencyManager


class TexturingMode(Enum):
    REALTIME = "realtime"
    PRODUCTION = "production"
    CINEMATIC = "cinematic"
    OFF = "off"  # no AI texturing, flat shading only


class TexturingEngine:
    """Runtime coordinator for the full AI texturing pipeline.

    Frame pipeline:
      1. Render input buffers from cell grid        ~1ms
      2. Compute dirty mask from camera delta        ~0.1ms
      3. Run AI model on dirty cells                 ~5-8ms (realtime) / 15-50ms (production)
      4. Write-back light values to cells            ~0.5ms
      5. Per-cell temporal blend                     ~0.2ms
      Total realtime budget: ~7-10ms
    """

    def __init__(
        self,
        mode: TexturingMode = TexturingMode.REALTIME,
        resolution: int = 512,
        device: str = "cuda",
    ):
        self.mode = mode
        self.resolution = resolution
        self.device = device

        # Components
        self.buffer_renderer = CellBufferRendererGPU(resolution, resolution)
        self.temporal = TemporalConsistencyManager()

        # AI models (lazy loaded)
        self._gan_model = None
        self._controlnet_pipeline = None

        # Frame state
        self._prev_camera_pos = None
        self._prev_camera_forward = None
        self._frame_count = 0
        self._last_buffers: Optional[RenderBuffers] = None

        # Performance tracking
        self._timings = {
            "buffer_render": 0.0,
            "dirty_mask": 0.0,
            "ai_model": 0.0,
            "writeback": 0.0,
            "temporal": 0.0,
            "total": 0.0,
        }

    def load_model(self):
        """Load the AI model for the current mode."""
        if self.mode == TexturingMode.REALTIME:
            self._load_gan()
        elif self.mode in (TexturingMode.PRODUCTION, TexturingMode.CINEMATIC):
            self._load_controlnet()

    def _load_gan(self):
        """Load Pix2PixHD-Lite GAN."""
        import torch
        from .models.pix2pix_lite import Pix2PixLiteGenerator

        self._gan_model = Pix2PixLiteGenerator().to(self.device)
        self._gan_model.eval()

        # Try to load checkpoint
        import os
        ckpt_path = "data/checkpoints/pix2pix_lite.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self._gan_model.load_state_dict(ckpt.get("generator", ckpt))
            print(f"[Texturing] Loaded GAN from {ckpt_path}")
        else:
            print("[Texturing] No GAN checkpoint found — using random weights (will look bad)")

    def _load_controlnet(self):
        """Load ControlNet + StreamDiffusion pipeline."""
        from .models.controlnet_adapter import ControlNetTexturingPipeline

        self._controlnet_pipeline = ControlNetTexturingPipeline(device=self.device)
        self._controlnet_pipeline.load()

    def process_frame(
        self,
        grid,
        camera_pos: np.ndarray,
        camera_forward: np.ndarray,
        camera_up: np.ndarray,
        dt: float = 1.0 / 60.0,
        visible_cell_ids: Optional[list] = None,
    ) -> Optional[np.ndarray]:
        """Process one frame through the AI texturing pipeline.

        Args:
            grid: CellGrid
            camera_pos: (3,) camera position
            camera_forward: (3,) camera forward direction
            camera_up: (3,) camera up direction
            dt: time since last frame
            visible_cell_ids: list of visible cell indices (from frustum culling)

        Returns:
            (H, W, 3) float32 photorealistic RGB in [0,1], or None if mode is OFF
        """
        if self.mode == TexturingMode.OFF:
            return None

        t_total_start = time.perf_counter()

        # 1. Render input buffers
        t0 = time.perf_counter()
        buffers = self.buffer_renderer.render(
            grid, camera_pos, camera_forward, camera_up
        )
        self._timings["buffer_render"] = time.perf_counter() - t0
        self._last_buffers = buffers

        # 2. Compute dirty mask
        t0 = time.perf_counter()
        if visible_cell_ids is None:
            # Use all cells with valid cell_ids in the buffer
            visible_cell_ids = list(set(buffers.cell_ids[buffers.cell_ids >= 0].tolist()))

        dirty_cells = self.temporal.compute_dirty_mask(
            grid, camera_pos, camera_forward, visible_cell_ids
        )
        self._timings["dirty_mask"] = time.perf_counter() - t0

        # 3. Run AI model
        t0 = time.perf_counter()
        ai_output = self._run_ai_model(buffers)
        self._timings["ai_model"] = time.perf_counter() - t0

        if ai_output is not None:
            # 4. Write-back to cells (confidence-weighted)
            # Low-confidence cells get MORE weight from AI texturing output
            # because their gradient-based shading is unreliable.
            # High-confidence cells can blend AI output with gradient shading.
            t0 = time.perf_counter()
            wb_stats = writeback_light_to_cells(
                ai_output, buffers, camera_pos, grid,
                smoothing_alpha=1.0,  # temporal blending handles smoothing
                confidence_boost_low=True,  # boost AI weight for low-conf cells
            )
            self._timings["writeback"] = time.perf_counter() - t0

            # 5. Per-cell temporal blend
            t0 = time.perf_counter()
            # Extract updated cell light values for temporal blending
            updated = {}
            for cid in dirty_cells:
                if hasattr(grid, 'get_cell_light'):
                    light = grid.get_cell_light(cid)
                    updated[cid] = np.array(light, dtype=np.float32)

            if updated:
                blend_stats = self.temporal.blend_and_update(
                    grid, updated, camera_pos, camera_forward, dt
                )
            self._timings["temporal"] = time.perf_counter() - t0

        self._timings["total"] = time.perf_counter() - t_total_start

        # Update frame state
        self._prev_camera_pos = camera_pos.copy()
        self._prev_camera_forward = camera_forward.copy()
        self._frame_count += 1

        return ai_output

    def _run_ai_model(self, buffers: RenderBuffers) -> Optional[np.ndarray]:
        """Run the AI texturing model on the input buffers.

        Returns (H, W, 3) float32 RGB in [0,1] or None on failure.
        """
        if self.mode == TexturingMode.REALTIME:
            return self._run_gan(buffers)
        elif self.mode in (TexturingMode.PRODUCTION, TexturingMode.CINEMATIC):
            return self._run_controlnet(buffers)
        return None

    def _run_gan(self, buffers: RenderBuffers) -> Optional[np.ndarray]:
        """Run Pix2PixHD-Lite GAN."""
        if self._gan_model is None:
            return None

        import torch

        # Prepare 8-channel input
        model_input = buffers.to_model_input()  # (H, W, 8)
        tensor = torch.from_numpy(model_input).permute(2, 0, 1).unsqueeze(0)  # (1, 8, H, W)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            rgb, light = self._gan_model(tensor)

        # Convert from [-1,1] to [0,1]
        output = (rgb.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
        return np.clip(output, 0, 1).astype(np.float32)

    def _run_controlnet(self, buffers: RenderBuffers) -> Optional[np.ndarray]:
        """Run ControlNet + StreamDiffusion."""
        if self._controlnet_pipeline is None:
            return None

        import torch

        depth = torch.from_numpy(buffers.depth).unsqueeze(0).unsqueeze(0).to(self.device)
        normals = torch.from_numpy(buffers.normals).permute(2, 0, 1).unsqueeze(0).to(self.device)
        labels = torch.from_numpy(buffers.labels.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)

        # Normalize
        d_max = depth.max()
        if d_max > 0:
            depth = depth / d_max
        l_max = labels.max()
        if l_max > 0:
            labels = labels / l_max

        output = self._controlnet_pipeline.generate(depth, normals, labels)
        return output.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

    def get_timings(self) -> dict:
        """Return timing breakdown for the last frame."""
        return self._timings.copy()

    def get_temporal_stats(self) -> dict:
        """Return temporal consistency statistics."""
        return self.temporal.get_stats()
