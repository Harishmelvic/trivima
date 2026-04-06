"""
Export Gaussians to standard formats.

- PLY: standard 3DGS format, compatible with viewers and SC-GS
- Point cloud PLY: dense point sampling for external tools
"""

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def export_gaussians_ply(gaussians: dict, output_path: str):
    """Export Gaussians to PLY format compatible with 3DGS viewers and SC-GS.

    The PLY format stores:
      x, y, z: position
      nx, ny, nz: normal
      f_dc_0, f_dc_1, f_dc_2: DC color (SH band 0)
      opacity: log-opacity
      scale_0, scale_1, scale_2: log-scale
      rot_0, rot_1, rot_2, rot_3: quaternion
    """
    pos = gaussians["positions"].numpy() if isinstance(gaussians["positions"], torch.Tensor) else gaussians["positions"]
    colors = gaussians["colors"].numpy() if isinstance(gaussians["colors"], torch.Tensor) else gaussians["colors"]
    scales = gaussians["scales"].numpy() if isinstance(gaussians["scales"], torch.Tensor) else gaussians["scales"]
    rots = gaussians["rotations"].numpy() if isinstance(gaussians["rotations"], torch.Tensor) else gaussians["rotations"]
    opacs = gaussians["opacities"].numpy() if isinstance(gaussians["opacities"], torch.Tensor) else gaussians["opacities"]

    n = len(pos)

    # SH color: convert RGB [0,1] to SH DC coefficient
    # DC = (color - 0.5) / C0 where C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    sh_dc = (colors - 0.5) / C0

    # Normals (optional, from gaussians dict or zeros)
    if "normals" in gaussians:
        normals = gaussians["normals"].numpy() if isinstance(gaussians["normals"], torch.Tensor) else gaussians["normals"]
    else:
        normals = np.zeros((n, 3), dtype=np.float32)

    # Build structured array
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]

    elements = np.empty(n, dtype=dtype)
    elements['x'] = pos[:, 0]
    elements['y'] = pos[:, 1]
    elements['z'] = pos[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    elements['f_dc_0'] = sh_dc[:, 0]
    elements['f_dc_1'] = sh_dc[:, 1]
    elements['f_dc_2'] = sh_dc[:, 2]
    elements['opacity'] = opacs  # already in logit space
    elements['scale_0'] = scales[:, 0]  # already log scale
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    elements['rot_0'] = rots[:, 0]
    elements['rot_1'] = rots[:, 1]
    elements['rot_2'] = rots[:, 2]
    elements['rot_3'] = rots[:, 3]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)
    print(f"  Exported {n:,} Gaussians to {output_path}")


def export_point_cloud_ply(gaussians: dict, output_path: str, samples_per_gaussian: int = 10):
    """Export dense point cloud by sampling points from Gaussians.

    Each Gaussian contributes multiple points based on its scale.
    """
    pos = gaussians["positions"].numpy() if isinstance(gaussians["positions"], torch.Tensor) else gaussians["positions"]
    colors = gaussians["colors"].numpy() if isinstance(gaussians["colors"], torch.Tensor) else gaussians["colors"]

    n = len(pos)

    # Sample points around each Gaussian center
    all_points = []
    all_colors = []

    for i in range(n):
        # Sample within the Gaussian's extent
        scale = np.exp(gaussians["scales"][i].numpy() if isinstance(gaussians["scales"], torch.Tensor) else gaussians["scales"][i])
        samples = np.random.randn(samples_per_gaussian, 3).astype(np.float32) * scale * 0.5
        samples += pos[i]
        all_points.append(samples)
        all_colors.append(np.tile(colors[i], (samples_per_gaussian, 1)))

    points = np.concatenate(all_points)
    cols = np.concatenate(all_colors)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    elements = np.empty(len(points), dtype=dtype)
    elements['x'] = points[:, 0]
    elements['y'] = points[:, 1]
    elements['z'] = points[:, 2]
    elements['red'] = (cols[:, 0] * 255).clip(0, 255).astype(np.uint8)
    elements['green'] = (cols[:, 1] * 255).clip(0, 255).astype(np.uint8)
    elements['blue'] = (cols[:, 2] * 255).clip(0, 255).astype(np.uint8)

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)
    print(f"  Exported {len(points):,} point cloud to {output_path}")
