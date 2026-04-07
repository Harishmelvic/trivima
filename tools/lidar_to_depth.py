"""
Project a LiDAR point cloud into a camera image plane → sparse depth map.

Used by all outdoor packers (KITTI, KITTI-360, nuScenes, Waymo) — they all
have LiDAR + camera pose calibration but each in a slightly different format.
This is the common conversion.
"""

from __future__ import annotations

import numpy as np


def project_lidar_to_image(
    points_lidar: np.ndarray,
    lidar_to_cam: np.ndarray,
    K: np.ndarray,
    image_h: int,
    image_w: int,
    min_depth: float = 0.5,
    max_depth: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        points_lidar: (N, 3) LiDAR points in LiDAR frame
        lidar_to_cam: (4, 4) extrinsic from LiDAR to camera
        K: (3, 3) camera intrinsics in pixel space
        image_h, image_w: target depth-map size
        min_depth, max_depth: clipping range
    Returns:
        depth: (H, W) float32 depth in meters, 0 where invalid
        mask:  (H, W) bool — True where a LiDAR return was projected
    """
    if points_lidar.shape[0] == 0:
        return (
            np.zeros((image_h, image_w), dtype=np.float32),
            np.zeros((image_h, image_w), dtype=bool),
        )

    # Homogeneous transform LiDAR → camera
    pts_h = np.concatenate(
        [points_lidar, np.ones((points_lidar.shape[0], 1), dtype=np.float32)], axis=1
    )
    cam = (lidar_to_cam @ pts_h.T).T[:, :3]  # (N, 3)

    # Keep points in front of camera
    z = cam[:, 2]
    front = (z > min_depth) & (z < max_depth)
    cam = cam[front]
    if cam.shape[0] == 0:
        return (
            np.zeros((image_h, image_w), dtype=np.float32),
            np.zeros((image_h, image_w), dtype=bool),
        )

    # Project
    proj = (K @ cam.T).T  # (N, 3)
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]
    z = cam[:, 2]

    # Round to integer pixel coords
    u_i = np.round(u).astype(np.int32)
    v_i = np.round(v).astype(np.int32)
    valid = (u_i >= 0) & (u_i < image_w) & (v_i >= 0) & (v_i < image_h)
    u_i, v_i, z = u_i[valid], v_i[valid], z[valid]

    # Z-buffer: keep nearest depth at each pixel
    depth = np.zeros((image_h, image_w), dtype=np.float32)
    # Sort by depth descending so nearer points overwrite farther ones
    order = np.argsort(-z)
    u_i = u_i[order]
    v_i = v_i[order]
    z = z[order]
    depth[v_i, u_i] = z

    mask = depth > 0
    return depth, mask


def densify_sparse_depth(depth: np.ndarray, mask: np.ndarray, max_radius: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Optionally densify a sparse LiDAR depth map by morphological dilation.

    Returns the dilated depth + mask. Useful for visualization but the training loop
    works fine on the raw sparse depth.
    """
    try:
        import cv2
    except ImportError:
        return depth, mask

    kernel = np.ones((max_radius * 2 + 1, max_radius * 2 + 1), np.uint8)
    dense_mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    dense_depth = depth.copy()
    # Fill holes by nearest neighbor (per pixel) — simple approach
    if dense_mask.any() and not mask.all():
        from scipy.ndimage import distance_transform_edt

        idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        dense_depth = depth[tuple(idx)]
        dense_depth[~dense_mask] = 0
    return dense_depth, dense_mask
