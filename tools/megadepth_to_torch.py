"""
Pack MegaDepth scenes into the unified .torch format.

MegaDepth provides per-image dense depth maps from COLMAP MVS reconstruction.
Layout (after extracting MegaDepth_v1.tar.gz):
    /content/data/MegaDepth_v1/
        0001/
            dense0/
                imgs/
                    *.jpg              # photos of one landmark
                depths/
                    *.h5               # dense depth in HDF5
                ...
        0002/
        ...

Each h5 file contains a 'depth' dataset; pose info comes from the MegaDepth_SfM
auxiliary archive (sparse model in COLMAP format).

Simplified usage: this packer assumes you have BOTH the dense depth h5 files AND
COLMAP cameras.txt + images.txt for each scene. Pass them per-scene.

Usage:
    python tools/megadepth_to_torch.py \\
        --scene_root /content/data/MegaDepth_v1/0001/dense0 \\
        --colmap_dir /content/data/MegaDepth_SfM/0001/sparse/manhattan/0 \\
        --max_views 60 \\
        --out /content/data/torch_packed/megadepth_0001.torch
"""

from __future__ import annotations

import argparse
import io
import os
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from PIL import Image


CameraInfo = namedtuple("CameraInfo", ["model", "width", "height", "params"])
ImageInfo = namedtuple("ImageInfo", ["id", "qvec", "tvec", "camera_id", "name"])


def qvec_to_rot(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )
    return R


def read_colmap_cameras(path: str) -> dict:
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cid = int(parts[0])
            model = parts[1]
            w = int(parts[2])
            h = int(parts[3])
            params = np.array([float(x) for x in parts[4:]], dtype=np.float32)
            cams[cid] = CameraInfo(model, w, h, params)
    return cams


def read_colmap_images(path: str) -> dict:
    imgs = {}
    with open(path) as f:
        lines = [l for l in f if l.strip() and not l.startswith("#")]
    # Each image is 2 lines; second line we ignore (point2D info)
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        iid = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
        tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float32)
        cam_id = int(parts[8])
        name = parts[9]
        imgs[iid] = ImageInfo(iid, qvec, tvec, cam_id, name)
    return imgs


def colmap_to_K(cam: CameraInfo) -> np.ndarray:
    """Convert a COLMAP camera (PINHOLE / SIMPLE_PINHOLE / SIMPLE_RADIAL) to a 3x3 K."""
    if cam.model in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
    elif cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        fx = fy = cam.params[0]
        cx, cy = cam.params[1], cam.params[2]
    else:
        # Fallback: just use first 4 params
        fx = cam.params[0]
        fy = cam.params[0] if len(cam.params) < 2 else cam.params[1]
        cx = cam.width / 2
        cy = cam.height / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def pack_megadepth_scene(
    scene_root: str,
    colmap_dir: str,
    max_views: int,
    out_path: str,
    target_size: int = 256,
):
    try:
        import h5py
    except ImportError:
        raise SystemExit("h5py not installed: pip install h5py")

    cams = read_colmap_cameras(os.path.join(colmap_dir, "cameras.txt"))
    imgs = read_colmap_images(os.path.join(colmap_dir, "images.txt"))

    img_dir = os.path.join(scene_root, "imgs")
    depth_dir = os.path.join(scene_root, "depths")

    images_jpg = []
    depths = []
    masks = []
    cams_out = []

    img_list = list(imgs.values())[:max_views]
    for info in img_list:
        img_path = os.path.join(img_dir, info.name)
        if not os.path.exists(img_path):
            continue
        # Depth file: same stem as image, .h5 extension
        stem = Path(info.name).stem
        depth_path = os.path.join(depth_dir, stem + ".h5")
        if not os.path.exists(depth_path):
            continue

        img = Image.open(img_path).convert("RGB")
        src_w, src_h = img.size
        img_resized = img.resize((target_size, target_size), Image.LANCZOS)
        buf = io.BytesIO()
        img_resized.save(buf, format="JPEG", quality=92)
        images_jpg.append(torch.tensor(np.frombuffer(buf.getvalue(), dtype=np.uint8)))

        with h5py.File(depth_path, "r") as f:
            depth = np.array(f["depth"], dtype=np.float32)
        # depth is at original image resolution; resize to target via nearest
        d_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        import torch.nn.functional as F

        d_r = F.interpolate(d_t, size=(target_size, target_size), mode="nearest").squeeze()
        m_r = d_r > 0
        depths.append(d_r)
        masks.append(m_r)

        # Camera
        cam = cams[info.camera_id]
        K_full = colmap_to_K(cam)
        # Normalize against COLMAP's image dims
        fx = K_full[0, 0] / cam.width
        fy = K_full[1, 1] / cam.height
        cx = K_full[0, 2] / cam.width
        cy = K_full[1, 2] / cam.height

        R = qvec_to_rot(info.qvec)
        t = info.tvec
        ext = np.zeros((3, 4), dtype=np.float32)
        ext[:3, :3] = R
        ext[:3, 3] = t

        cam_vec = torch.zeros(18, dtype=torch.float32)
        cam_vec[0] = fx
        cam_vec[1] = fy
        cam_vec[2] = cx
        cam_vec[3] = cy
        cam_vec[4] = 0.5
        cam_vec[5] = 1000.0
        cam_vec[6:18] = torch.from_numpy(ext.reshape(-1))
        cams_out.append(cam_vec)

    if not depths:
        raise RuntimeError(f"No frames packed from {scene_root}")

    depths_t = torch.stack(depths)
    masks_t = torch.stack(masks)
    cams_t = torch.stack(cams_out)

    valid = depths_t[masks_t]
    scene_scale = float(valid.quantile(0.95).item()) if valid.numel() > 0 else 100.0

    pkg = {
        "images": images_jpg,
        "depths": depths_t,
        "depth_mask": masks_t,
        "cameras": cams_t,
        "domain": "landmark",
        "scene_scale": scene_scale,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(pkg, out_path)
    print(f"  Saved {len(images_jpg)} frames to {out_path}  (scene_scale={scene_scale:.2f}m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_root", required=True)
    ap.add_argument("--colmap_dir", required=True)
    ap.add_argument("--max_views", type=int, default=60)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pack_megadepth_scene(args.scene_root, args.colmap_dir, args.max_views, args.out, args.size)


if __name__ == "__main__":
    main()
