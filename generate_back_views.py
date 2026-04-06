"""
Generate synthetic back/side views from Lyra's 3DGS.

Takes the Gaussian PLY, renders from multiple camera positions
around and behind the room, saves as images for COLMAP input.
"""
import numpy as np
import torch
import math
import os
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization

def load_gaussians(ply_path):
    """Load Gaussian PLY into tensors."""
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
    """Build view matrix looking from pos toward target."""
    pos = np.array(pos, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    fwd = target - pos
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, fwd)

    R = np.eye(3, dtype=np.float32)
    R[0, :] = right
    R[1, :] = -up
    R[2, :] = fwd
    t = -R @ pos

    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = R
    m[:3, 3] = t
    return torch.tensor(m)


def render_view(means, quats, scales, opacities, colors, viewmat, K, W, H, device):
    """Render one view."""
    renders, alphas, _ = rasterization(
        means=means.to(device),
        quats=quats.to(device),
        scales=torch.exp(scales.to(device)),
        opacities=torch.sigmoid(opacities.to(device)),
        colors=colors.to(device),
        viewmats=viewmat.unsqueeze(0).to(device),
        Ks=K.unsqueeze(0).to(device),
        width=W, height=H, packed=False,
    )
    rgb = renders[0].clamp(0, 1).cpu().numpy()
    alpha = alphas[0, :, :, 0].cpu().numpy()
    return (rgb * 255).astype(np.uint8), alpha


def main():
    ply_path = "/workspace/lyra/outputs/demo/lyra_room/static_view_indices_fixed_0_1/room_photo/gaussians_orig/gaussians_0.ply"
    output_dir = "/workspace/lyra/synthetic_views"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Gaussians...")
    means, quats, scales, opacities, colors, n = load_gaussians(ply_path)
    print("Loaded %d Gaussians" % n)

    # Find scene center and bounds
    center = means.mean(dim=0).numpy()
    extent = (means.max(dim=0).values - means.min(dim=0).values).numpy()
    print("Center: (%.2f, %.2f, %.2f)" % (center[0], center[1], center[2]))
    print("Extent: (%.2f, %.2f, %.2f)" % (extent[0], extent[1], extent[2]))

    radius = max(extent) * 0.8
    W, H = 1280, 704
    fx = fy = 600.0
    K = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], dtype=torch.float32)

    device = "cuda"

    # Generate views: orbit around the scene center
    # Front, sides, back, top-down, diagonal
    views = []

    # Orbit at eye level
    for angle in range(0, 360, 15):
        rad = math.radians(angle)
        x = center[0] + radius * math.cos(rad)
        z = center[2] + radius * math.sin(rad)
        y = center[1]
        views.append(("orbit_%03d" % angle, [x, y, z], center.tolist()))

    # Elevated orbit (looking down)
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        x = center[0] + radius * 0.7 * math.cos(rad)
        z = center[2] + radius * 0.7 * math.sin(rad)
        y = center[1] + radius * 0.5
        views.append(("elevated_%03d" % angle, [x, y, z], center.tolist()))

    print("Rendering %d synthetic views..." % len(views))
    for i, (name, pos, target) in enumerate(views):
        viewmat = make_camera(pos, target)
        rgb, alpha = render_view(means, quats, scales, opacities, colors, viewmat, K, W, H, device)
        coverage = 100 * (alpha > 0.01).sum() / (W * H)
        Image.fromarray(rgb).save(os.path.join(output_dir, "%s.png" % name))
        if i % 10 == 0:
            print("  [%d/%d] %s: %.1f%% coverage" % (i, len(views), name, coverage))

    # Also copy the original photo
    import shutil
    shutil.copy("/workspace/lyra/room_photo.jpg", os.path.join(output_dir, "original_000.png"))

    print("Done! %d views saved to %s" % (len(views) + 1, output_dir))


if __name__ == "__main__":
    main()
