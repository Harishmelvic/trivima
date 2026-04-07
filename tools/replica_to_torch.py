"""
Pack a Replica scene into the unified .torch format using Habitat-Sim renders.

Usage (inside Colab after habitat-sim is installed):
    python tools/replica_to_torch.py \\
        --replica_root /content/data/Replica \\
        --scene apartment_0 \\
        --views 50 \\
        --out /content/data/torch_packed/replica_apartment_0.torch

Or batch all scenes:
    python tools/replica_to_torch.py --replica_root /content/data/Replica --all --views 50

The output schema matches trivima/multiview/pointcloud_dataset.py:
    images:      list[byte tensor JPEG]
    depths:      tensor (N, H, W) meters
    depth_mask:  tensor (N, H, W) bool
    cameras:     tensor (N, 18)   pixelSplat schema
    domain:      'indoor'
    scene_scale: float (auto-estimated from depth percentiles)
"""

from __future__ import annotations

import argparse
import io
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def render_one_view(sim, sensor_h: int, sensor_w: int):
    """Run a single Habitat-Sim observation and return RGB + depth + pose."""
    obs = sim.get_sensor_observations()
    rgb = obs["color_sensor"][..., :3]  # (H, W, 3) uint8
    depth = obs["depth_sensor"]          # (H, W) float meters
    state = sim.agents[0].get_state()
    sensor_state = state.sensor_states["color_sensor"]
    pos = np.asarray(sensor_state.position, dtype=np.float32)   # (3,)
    rot = sensor_state.rotation                                  # quaternion (w, x, y, z)
    return rgb, depth, pos, rot


def quat_to_R(q) -> np.ndarray:
    """Habitat quaternion (numpy-quaternion: w,x,y,z accessors) → 3x3 rotation matrix."""
    # numpy-quaternion exposes w,x,y,z directly
    w = float(q.w)
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return R


def cam_to_pixelsplat_18(R_w2c: np.ndarray, t_w2c: np.ndarray, fx: float, fy: float, cx: float, cy: float, near: float = 0.1, far: float = 100.0) -> torch.Tensor:
    """Build the 18-vector camera entry expected by pointcloud_dataset.py.

    pixelSplat normalized intrinsics convention: fx/fy/cx/cy as fraction of image width/height.
    Here we pass NORMALIZED values so the dataset's _scale_intrinsics() picks them up correctly.
    """
    out = torch.zeros(18, dtype=torch.float32)
    out[0] = fx
    out[1] = fy
    out[2] = cx
    out[3] = cy
    out[4] = near
    out[5] = far
    ext = np.zeros((3, 4), dtype=np.float32)
    ext[:3, :3] = R_w2c
    ext[:3, 3] = t_w2c
    out[6:18] = torch.from_numpy(ext.reshape(-1))
    return out


def pack_scene(scene_dir: str, n_views: int, out_path: str, sensor_size: int = 256):
    try:
        import habitat_sim
    except ImportError:
        raise SystemExit("habitat_sim not installed. Run: pip install habitat-sim")

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = os.path.join(scene_dir, "mesh.ply")
    if not os.path.exists(backend_cfg.scene_id):
        # Some Replica releases use mesh.glb
        for alt in ["mesh.glb", "habitat/mesh_semantic.ply", "habitat/mesh.ply"]:
            ap = os.path.join(scene_dir, alt)
            if os.path.exists(ap):
                backend_cfg.scene_id = ap
                break
        else:
            raise FileNotFoundError(f"No mesh found in {scene_dir}")
    backend_cfg.enable_physics = False

    color_cfg = habitat_sim.CameraSensorSpec()
    color_cfg.uuid = "color_sensor"
    color_cfg.resolution = [sensor_size, sensor_size]
    color_cfg.position = [0.0, 1.5, 0.0]  # eye height
    color_cfg.sensor_type = habitat_sim.SensorType.COLOR

    depth_cfg = habitat_sim.CameraSensorSpec()
    depth_cfg.uuid = "depth_sensor"
    depth_cfg.resolution = [sensor_size, sensor_size]
    depth_cfg.position = [0.0, 1.5, 0.0]
    depth_cfg.sensor_type = habitat_sim.SensorType.DEPTH

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [color_cfg, depth_cfg]

    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)

    # Compute intrinsics from default 90° HFOV
    hfov_rad = np.deg2rad(90.0)
    fx_px = sensor_size / (2.0 * np.tan(hfov_rad / 2.0))
    fy_px = fx_px
    cx_px = sensor_size / 2.0
    cy_px = sensor_size / 2.0
    # Normalized for pixelSplat schema
    fx = fx_px / sensor_size
    fy = fy_px / sensor_size
    cx = cx_px / sensor_size
    cy = cy_px / sensor_size

    images_jpg = []
    depths = []
    masks = []
    cams = []

    pf = sim.pathfinder
    if not pf.is_loaded:
        print("  WARNING: navmesh not loaded, sampling random positions inside bounding box")

    rng = random.Random(0)
    n_done = 0
    attempts = 0
    while n_done < n_views and attempts < n_views * 10:
        attempts += 1
        if pf.is_loaded:
            pos = pf.get_random_navigable_point()
        else:
            bbox = pf.get_bounds() if pf.is_loaded else None
            if bbox is None:
                pos = np.random.uniform(-2, 2, 3).astype(np.float32)
            else:
                pos = np.random.uniform(bbox[0], bbox[1]).astype(np.float32)
        # Random yaw
        yaw = rng.uniform(0, 2 * np.pi)
        agent_state = sim.agents[0].get_state()
        agent_state.position = pos
        # Build a yaw-only quaternion
        import quaternion as qt  # comes with habitat-sim

        agent_state.rotation = qt.from_rotation_vector([0.0, yaw, 0.0])
        sim.agents[0].set_state(agent_state)

        rgb, depth, sp, sr = render_one_view(sim, sensor_size, sensor_size)
        # Skip degenerate renders
        if depth.mean() < 0.05 or depth.max() > 50.0:
            continue

        # Encode RGB to JPEG bytes
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
        images_jpg.append(torch.tensor(np.frombuffer(buf.getvalue(), dtype=np.uint8)))
        d = torch.from_numpy(depth.astype(np.float32))
        depths.append(d)
        masks.append((d > 0))

        # Build cam2world from sensor pose, then invert to world2cam
        R_c2w = quat_to_R(sr)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = sp
        w2c = np.linalg.inv(c2w)
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        cams.append(cam_to_pixelsplat_18(R_w2c, t_w2c, fx, fy, cx, cy))
        n_done += 1

    sim.close()
    if n_done == 0:
        raise RuntimeError(f"No valid views rendered for {scene_dir}")

    depths_t = torch.stack(depths)
    masks_t = torch.stack(masks)
    cams_t = torch.stack(cams)

    # Auto scene scale: 95th percentile of valid depths
    valid = depths_t[masks_t]
    scene_scale = float(valid.quantile(0.95).item()) if valid.numel() > 0 else 5.0

    pkg = {
        "images": images_jpg,
        "depths": depths_t,
        "depth_mask": masks_t,
        "cameras": cams_t,
        "domain": "indoor",
        "scene_scale": scene_scale,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(pkg, out_path)
    print(f"  Saved {n_done} views to {out_path}  (scene_scale={scene_scale:.2f}m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", required=True, help="Path containing Replica scene folders")
    ap.add_argument("--scene", help="Scene name like 'apartment_0' (omit with --all)")
    ap.add_argument("--all", action="store_true", help="Pack all scenes in replica_root")
    ap.add_argument("--views", type=int, default=50)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", help="Output .torch path (only valid without --all)")
    ap.add_argument("--out_dir", default="/content/data/torch_packed", help="Output dir for --all mode")
    args = ap.parse_args()

    if args.all:
        scenes = [
            p
            for p in sorted(Path(args.replica_root).iterdir())
            if p.is_dir() and not p.name.startswith(".")
        ]
        for s in scenes:
            out = os.path.join(args.out_dir, f"replica_{s.name}.torch")
            print(f"\n=== {s.name} ===")
            try:
                pack_scene(str(s), args.views, out, args.size)
            except Exception as e:
                print(f"  FAILED: {e}")
    else:
        if not args.scene or not args.out:
            raise SystemExit("Need --scene and --out (or use --all)")
        pack_scene(os.path.join(args.replica_root, args.scene), args.views, args.out, args.size)


if __name__ == "__main__":
    main()
