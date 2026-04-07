"""
Pack KITTI raw sequences into the unified .torch format.

KITTI raw layout:
    /content/data/kitti_raw/
        2011_09_26/
            2011_09_26_drive_0001_sync/
                image_02/data/0000000000.png      # left color cam
                velodyne_points/data/0000000000.bin
            calib_cam_to_cam.txt
            calib_velo_to_cam.txt
            calib_imu_to_velo.txt

We use:
    - image_02 (left color camera) as the RGB source
    - Velodyne LiDAR projected into image_02 as the depth source
    - Per-frame poses from oxts/ (GPS+IMU) — sequence-level continuous pose

Usage:
    python tools/kitti_to_torch.py \\
        --kitti_root /content/data/kitti_raw \\
        --drive 2011_09_26/2011_09_26_drive_0001_sync \\
        --frames 100 \\
        --out /content/data/torch_packed/kitti_drive_0001.torch
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lidar_to_depth import project_lidar_to_image


def parse_calib_file(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = np.array([float(x) for x in v.strip().split()], dtype=np.float32)
            except ValueError:
                pass
    return out


def load_kitti_calib(date_dir: str) -> dict:
    cam_to_cam = parse_calib_file(os.path.join(date_dir, "calib_cam_to_cam.txt"))
    velo_to_cam = parse_calib_file(os.path.join(date_dir, "calib_velo_to_cam.txt"))

    # Camera 02 (left color) projection matrix (3, 4)
    P02 = cam_to_cam["P_rect_02"].reshape(3, 4)
    K = P02[:, :3]
    # rectification rotation
    R_rect_00 = cam_to_cam["R_rect_00"].reshape(3, 3)
    # velo → cam0
    R_v2c = velo_to_cam["R"].reshape(3, 3)
    T_v2c = velo_to_cam["T"].reshape(3)
    Tr_velo_to_cam0 = np.eye(4, dtype=np.float32)
    Tr_velo_to_cam0[:3, :3] = R_v2c
    Tr_velo_to_cam0[:3, 3] = T_v2c
    R_rect = np.eye(4, dtype=np.float32)
    R_rect[:3, :3] = R_rect_00
    Tr_velo_to_cam2 = R_rect @ Tr_velo_to_cam0
    # KITTI has a tx baseline for cam2 — already encoded in P02. We compute the cam2 origin offset:
    # P02 = K [R | t], where t = -K * baseline. Extract baseline:
    tx = P02[0, 3] / K[0, 0]
    cam2_offset = np.eye(4, dtype=np.float32)
    cam2_offset[0, 3] = tx
    Tr_velo_to_cam2 = cam2_offset @ Tr_velo_to_cam2

    return {
        "K": K.astype(np.float32),
        "Tr_velo_to_cam2": Tr_velo_to_cam2.astype(np.float32),
        "image_size": tuple(cam_to_cam["S_rect_02"].astype(int)),  # (W, H)
    }


def load_oxts_pose(path: str, scale: float) -> np.ndarray:
    """Load a single oxts file and return a 4x4 cam-rig pose in metric local frame."""
    with open(path) as f:
        vals = [float(x) for x in f.read().split()]
    lat, lon, alt = vals[0], vals[1], vals[2]
    roll, pitch, yaw = vals[3], vals[4], vals[5]
    # Mercator to metric (KITTI's standard transform)
    er = 6378137.0
    tx = scale * lon * np.pi * er / 180.0
    ty = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    tz = alt

    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R = R_z @ R_y @ R_x
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = [tx, ty, tz]
    return pose


def pack_drive(kitti_root: str, drive: str, n_frames: int, out_path: str, target_size: int = 256):
    drive_path = os.path.join(kitti_root, drive)
    date_dir = os.path.dirname(drive_path)

    calib = load_kitti_calib(date_dir)
    K_full = calib["K"]
    Tr_v2c = calib["Tr_velo_to_cam2"]
    src_w, src_h = calib["image_size"]

    img_dir = os.path.join(drive_path, "image_02", "data")
    velo_dir = os.path.join(drive_path, "velodyne_points", "data")
    oxts_dir = os.path.join(drive_path, "oxts", "data")

    img_files = sorted(Path(img_dir).glob("*.png"))
    if not img_files:
        raise FileNotFoundError(f"No images in {img_dir}")
    if len(img_files) > n_frames:
        # Evenly sample
        step = len(img_files) // n_frames
        img_files = img_files[::step][:n_frames]

    # Get oxts scale from the FIRST oxts file (KITTI convention)
    first_oxts = oxts_dir + "/" + img_files[0].stem + ".txt"
    if os.path.exists(first_oxts):
        with open(first_oxts) as f:
            lat0 = float(f.read().split()[0])
        scale = np.cos(lat0 * np.pi / 180.0)
        # Reference pose = first frame, used to make poses relative
        ref_pose = load_oxts_pose(first_oxts, scale)
        ref_inv = np.linalg.inv(ref_pose)
    else:
        scale = 1.0
        ref_inv = np.eye(4, dtype=np.float32)
        print("  WARNING: oxts not found, poses will be at identity")

    images_jpg = []
    depths = []
    masks = []
    cams = []

    for img_file in img_files:
        stem = img_file.stem
        velo_file = os.path.join(velo_dir, stem + ".bin")
        oxts_file = os.path.join(oxts_dir, stem + ".txt")
        if not os.path.exists(velo_file):
            continue

        # Load and resize image
        img = Image.open(img_file).convert("RGB")
        img_resized = img.resize((target_size, target_size), Image.LANCZOS)
        buf = io.BytesIO()
        img_resized.save(buf, format="JPEG", quality=92)
        images_jpg.append(torch.tensor(np.frombuffer(buf.getvalue(), dtype=np.uint8)))

        # Load LiDAR
        velo = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)[:, :3]
        # Project to original image resolution first
        depth_full, mask_full = project_lidar_to_image(
            velo, Tr_v2c, K_full, src_h, src_w, min_depth=1.0, max_depth=80.0
        )
        # Then nearest-resize to target
        d_t = torch.from_numpy(depth_full).unsqueeze(0).unsqueeze(0)
        m_t = torch.from_numpy(mask_full.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        import torch.nn.functional as F

        d_r = F.interpolate(d_t, size=(target_size, target_size), mode="nearest").squeeze()
        m_r = F.interpolate(m_t, size=(target_size, target_size), mode="nearest").squeeze().bool()
        depths.append(d_r)
        masks.append(m_r)

        # Pose: world (= first frame) → camera
        if os.path.exists(oxts_file):
            cam_pose = load_oxts_pose(oxts_file, scale)
            local_c2w = ref_inv @ cam_pose  # cam2world relative to first frame
        else:
            local_c2w = np.eye(4, dtype=np.float32)
        w2c = np.linalg.inv(local_c2w)

        # Build the 18-vec
        # KITTI K is in PIXELS in the original resolution; we need normalized for our schema.
        fx = K_full[0, 0] / src_w
        fy = K_full[1, 1] / src_h
        cx = K_full[0, 2] / src_w
        cy = K_full[1, 2] / src_h
        cam = torch.zeros(18, dtype=torch.float32)
        cam[0] = fx
        cam[1] = fy
        cam[2] = cx
        cam[3] = cy
        cam[4] = 1.0
        cam[5] = 80.0
        cam[6:18] = torch.from_numpy(w2c[:3, :4].reshape(-1).astype(np.float32))
        cams.append(cam)

    if not depths:
        raise RuntimeError(f"No frames packed from {drive}")

    depths_t = torch.stack(depths)
    masks_t = torch.stack(masks)
    cams_t = torch.stack(cams)

    valid = depths_t[masks_t]
    scene_scale = float(valid.quantile(0.95).item()) if valid.numel() > 0 else 50.0

    pkg = {
        "images": images_jpg,
        "depths": depths_t,
        "depth_mask": masks_t,
        "cameras": cams_t,
        "domain": "outdoor",
        "scene_scale": scene_scale,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(pkg, out_path)
    print(f"  Saved {len(images_jpg)} frames to {out_path}  (scene_scale={scene_scale:.2f}m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--drive", required=True, help="e.g. 2011_09_26/2011_09_26_drive_0001_sync")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Make tools/ importable
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    pack_drive(args.kitti_root, args.drive, args.frames, args.out, args.size)


if __name__ == "__main__":
    main()
