"""Generate synthetic views from Lyra 3DGS - fixed camera positions."""
import numpy as np
import torch
import math
import os
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization

def load_gaussians(ply_path):
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    n = len(v)
    means = torch.tensor(np.stack([v["x"], v["y"], v["z"]], axis=-1), dtype=torch.float32)
    quats = torch.tensor(np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1), dtype=torch.float32)
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    scales = torch.tensor(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1), dtype=torch.float32)
    opacities = torch.tensor(np.array(v["opacity"]), dtype=torch.float32)
    C0 = 0.28209479177387814
    colors = torch.tensor(np.clip(np.stack([
        np.array(v["f_dc_0"]) * C0 + 0.5,
        np.array(v["f_dc_1"]) * C0 + 0.5,
        np.array(v["f_dc_2"]) * C0 + 0.5,
    ], axis=-1), 0, 1), dtype=torch.float32)
    return means, quats, scales, opacities, colors, n

def make_camera(pos, target, up=[0, 1, 0]):
    pos = np.array(pos, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    fwd = target - pos
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up_new = np.cross(right, fwd)
    R = np.eye(3, dtype=np.float32)
    R[0, :] = right
    R[1, :] = -up_new
    R[2, :] = fwd
    t = -R @ pos
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = R
    m[:3, 3] = t
    return torch.tensor(m)

ply_path = "/workspace/lyra/outputs/demo/lyra_room/static_view_indices_fixed_0_1/room_photo/gaussians_orig/gaussians_0.ply"
output_dir = "/workspace/lyra/synthetic_views"
os.makedirs(output_dir, exist_ok=True)

print("Loading...")
means, quats, scales, opacities, colors, n = load_gaussians(ply_path)

# Scene bounds from high-opacity points
center = np.array([0.03, 0.03, 1.70])
radius = 3.0  # orbit radius in meters

W, H = 1280, 704
fx = fy = 500.0
K = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], dtype=torch.float32)
device = "cuda"

views = []

# Orbit at eye level around the room center
for angle in range(0, 360, 10):
    rad = math.radians(angle)
    x = center[0] + radius * math.sin(rad)
    z = center[2] + radius * math.cos(rad)
    y = center[1]
    views.append(("orbit_%03d" % angle, [x, y, z]))

# Elevated views
for angle in range(0, 360, 20):
    rad = math.radians(angle)
    x = center[0] + radius * 0.8 * math.sin(rad)
    z = center[2] + radius * 0.8 * math.cos(rad)
    y = center[1] + 1.5
    views.append(("elevated_%03d" % angle, [x, y, z]))

print("Rendering %d views..." % len(views))
for i, (name, pos) in enumerate(views):
    viewmat = make_camera(pos, center.tolist())
    renders, alphas, _ = rasterization(
        means=means.to(device), quats=quats.to(device),
        scales=torch.exp(scales.to(device)),
        opacities=torch.sigmoid(opacities.to(device)),
        colors=colors.to(device),
        viewmats=viewmat.unsqueeze(0).to(device),
        Ks=K.unsqueeze(0).to(device),
        width=W, height=H, packed=False)

    rgb = renders[0].clamp(0, 1).cpu().numpy()
    alpha = alphas[0, :, :, 0].cpu().numpy()
    coverage = 100 * (alpha > 0.01).sum() / (W * H)

    Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(output_dir, name + ".png"))
    if i % 10 == 0:
        print("  [%d/%d] %s: %.1f%%" % (i, len(views), name, coverage))

# Copy original
import shutil
shutil.copy("/workspace/lyra/room_photo.jpg", os.path.join(output_dir, "original_000.png"))

total = len(os.listdir(output_dir))
print("Done! %d images in %s" % (total, output_dir))
