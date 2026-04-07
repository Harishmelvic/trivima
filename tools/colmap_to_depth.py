"""
Project a COLMAP / SfM point cloud into a camera image plane → sparse depth map.

Used by landmark dataset packers (MegaDepth, MegaScenes, Phototourism).
COLMAP gives us:
    - 3D points in world coordinates (positions of triangulated SfM features)
    - Per-image extrinsics (R, t) world-to-camera
    - Per-camera intrinsics (fx, fy, cx, cy)
    - Visibility info: which 3D points each image observed

We project the 3D points into each image to get a sparse depth map.
For MegaDepth specifically, dense depth maps are pre-computed by the authors —
in that case use them directly and skip this function.
"""

from __future__ import annotations

import numpy as np


def project_colmap_points_to_image(
    points_world: np.ndarray,
    world_to_cam: np.ndarray,
    K: np.ndarray,
    image_h: int,
    image_w: int,
    min_depth: float = 0.1,
    max_depth: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        points_world: (N, 3) 3D SfM points in world frame
        world_to_cam: (4, 4) world-to-camera extrinsic
        K: (3, 3) camera intrinsics in pixel space
        image_h, image_w: depth map size
    Returns:
        depth: (H, W) float32 depth in scene units, 0 where invalid
        mask:  (H, W) bool — True where a SfM point projected
    """
    if points_world.shape[0] == 0:
        return (
            np.zeros((image_h, image_w), dtype=np.float32),
            np.zeros((image_h, image_w), dtype=bool),
        )

    pts_h = np.concatenate(
        [points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1
    )
    cam = (world_to_cam @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    front = (z > min_depth) & (z < max_depth)
    cam = cam[front]
    if cam.shape[0] == 0:
        return (
            np.zeros((image_h, image_w), dtype=np.float32),
            np.zeros((image_h, image_w), dtype=bool),
        )
    proj = (K @ cam.T).T
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]
    z = cam[:, 2]

    u_i = np.round(u).astype(np.int32)
    v_i = np.round(v).astype(np.int32)
    valid = (u_i >= 0) & (u_i < image_w) & (v_i >= 0) & (v_i < image_h)
    u_i, v_i, z = u_i[valid], v_i[valid], z[valid]

    depth = np.zeros((image_h, image_w), dtype=np.float32)
    order = np.argsort(-z)
    u_i, v_i, z = u_i[order], v_i[order], z[order]
    depth[v_i, u_i] = z

    mask = depth > 0
    return depth, mask
