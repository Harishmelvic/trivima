"""
Buffer renderer — rasterizes cell grid into flat 2D buffers for AI texturing input.

Produces 5 aligned buffers:
  - albedo (H×W×3): flat surface color, no lighting
  - depth (H×W×1): metric depth normalized to [0,1]
  - normals (H×W×3): world-space normals encoded as RGB
  - labels (H×W×1): semantic label index per pixel
  - cell_ids (H×W×1, int32): index of the cell rendered at each pixel

The cell_ids buffer is critical for the write-back step — it maps each pixel
in the AI model's output back to the 3D cell it corresponds to.

GPU path: buffer_renderer.cu (CUDA kernel, <1ms for 512×512 at 80K cells)
CPU fallback: this module provides a pure-Python rasterizer for development.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RenderBuffers:
    """Output of the buffer renderer — all buffers are aligned to the same camera."""
    albedo: np.ndarray       # (H, W, 3) float32, [0,1]
    depth: np.ndarray        # (H, W) float32, metric depth in meters
    normals: np.ndarray      # (H, W, 3) float32, world-space normal [-1,1]
    labels: np.ndarray       # (H, W) int32, semantic label index
    cell_ids: np.ndarray     # (H, W) int32, cell index or -1 for empty pixels
    width: int
    height: int

    def to_model_input(self) -> np.ndarray:
        """Stack into 8-channel tensor for the AI texturing model.
        Channels: albedo_r, albedo_g, albedo_b, depth_norm, normal_x, normal_y, normal_z, label_norm
        """
        h, w = self.height, self.width

        # Normalize depth to [0,1] based on min/max
        d = self.depth.copy()
        valid = d > 0
        if valid.any():
            d_min, d_max = d[valid].min(), d[valid].max()
            if d_max > d_min:
                d = np.where(valid, (d - d_min) / (d_max - d_min), 0.0)
            else:
                d = np.where(valid, 0.5, 0.0)

        # Normalize labels to [0,1] — divide by max label
        l = self.labels.astype(np.float32)
        l_max = l.max()
        if l_max > 0:
            l = l / l_max

        return np.stack([
            self.albedo[..., 0],    # R
            self.albedo[..., 1],    # G
            self.albedo[..., 2],    # B
            d,                       # depth normalized
            self.normals[..., 0],   # Nx
            self.normals[..., 1],   # Ny
            self.normals[..., 2],   # Nz
            l,                       # label normalized
        ], axis=-1).astype(np.float32)  # (H, W, 8)


class CellBufferRenderer:
    """CPU fallback renderer — rasterizes cell grid to buffers via simple ray marching.

    For production, this is replaced by buffer_renderer.cu (CUDA kernel).
    """

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height

    def render(self, grid, camera_pos: np.ndarray, camera_forward: np.ndarray,
               camera_up: np.ndarray, fov_deg: float = 60.0) -> RenderBuffers:
        """Render the cell grid from the given camera pose.

        Args:
            grid: CellGridCPU (native) or Python grid wrapper
            camera_pos: (3,) camera position in world space
            camera_forward: (3,) normalized forward direction
            camera_up: (3,) normalized up direction
            fov_deg: horizontal field of view in degrees

        Returns:
            RenderBuffers with all 5 aligned buffers
        """
        w, h = self.width, self.height
        albedo = np.zeros((h, w, 3), dtype=np.float32)
        depth = np.zeros((h, w), dtype=np.float32)
        normals = np.zeros((h, w, 3), dtype=np.float32)
        labels = np.full((h, w), -1, dtype=np.int32)
        cell_ids = np.full((h, w), -1, dtype=np.int32)

        # Build camera basis
        fwd = camera_forward / (np.linalg.norm(camera_forward) + 1e-8)
        right = np.cross(fwd, camera_up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, fwd)

        fov_rad = np.radians(fov_deg)
        aspect = w / h
        half_w = np.tan(fov_rad / 2)
        half_h = half_w / aspect

        # Ray march parameters
        cell_size = 0.05  # base cell size
        max_dist = 20.0   # max ray distance
        step = cell_size * 0.5  # half-cell steps for sub-cell precision

        for py in range(h):
            for px in range(w):
                # Normalized device coordinates [-1, 1]
                u = (2.0 * px / w - 1.0) * half_w
                v = (1.0 - 2.0 * py / h) * half_h

                ray_dir = fwd + u * right + v * up
                ray_dir = ray_dir / (np.linalg.norm(ray_dir) + 1e-8)

                # March along ray
                t = 0.0
                while t < max_dist:
                    pos = camera_pos + ray_dir * t

                    # Query cell grid at this position
                    idx = grid.find_at_position(pos[0], pos[1], pos[2])
                    if idx is not None and idx >= 0:
                        geo = grid.get_geo(idx)
                        if geo.density > 0.3:  # hit a non-empty cell
                            vis = grid.get_vis(idx)
                            albedo[py, px] = [vis.albedo_r, vis.albedo_g, vis.albedo_b]
                            depth[py, px] = t
                            normals[py, px] = [geo.normal_x, geo.normal_y, geo.normal_z]
                            labels[py, px] = vis.semantic_label
                            cell_ids[py, px] = idx
                            break
                    t += step

        return RenderBuffers(
            albedo=albedo, depth=depth, normals=normals,
            labels=labels, cell_ids=cell_ids,
            width=w, height=h
        )


class CellBufferRendererGPU:
    """GPU buffer renderer — wraps the CUDA kernel in buffer_renderer.cu.

    Falls back to CPU renderer if CUDA is not available.
    """

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self._gpu_available = False

        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_available = True
        except ImportError:
            pass

        if not self._gpu_available:
            self._fallback = CellBufferRenderer(width, height)

    def render(self, grid, camera_pos, camera_forward, camera_up,
               fov_deg: float = 60.0) -> RenderBuffers:
        if not self._gpu_available:
            return self._fallback.render(grid, camera_pos, camera_forward, camera_up, fov_deg)

        # TODO: Call CUDA kernel via trivima_native.render_buffers()
        # For now, use CPU fallback
        return self._fallback.render(grid, camera_pos, camera_forward, camera_up, fov_deg)
