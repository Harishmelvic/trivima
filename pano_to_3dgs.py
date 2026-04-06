"""
Panorama → Depth Pro → 3D Point Cloud → 3DGS

Slices panorama into overlapping perspective views,
runs Depth Pro on each, backprojects to 3D, trains 3DGS.
"""
import os, sys, math, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace/ml-depth-pro/src")

# ============================================================
# Step 1: Slice panorama into perspective views
# ============================================================
def slice_panorama(pano_path, num_views=8, fov_deg=90, output_dir="/workspace/pano_views"):
    """Slice a wide panorama into overlapping perspective views."""
    os.makedirs(output_dir, exist_ok=True)
    pano = Image.open(pano_path).convert("RGB")
    pw, ph = pano.size
    print("Panorama: %dx%d" % (pw, ph))

    view_w = int(ph * math.tan(math.radians(fov_deg/2)) * 2)
    view_w = min(view_w, ph * 2)  # cap at 2:1 aspect
    step = (pw - view_w) / max(num_views - 1, 1)

    views = []
    for i in range(num_views):
        x = int(i * step)
        x = min(x, pw - view_w)
        crop = pano.crop((x, 0, x + view_w, ph))
        path = os.path.join(output_dir, "view_%02d.png" % i)
        crop.save(path)
        # Camera yaw: map x position to angle
        center_x = x + view_w / 2
        yaw = (center_x / pw - 0.5) * 360  # degrees
        views.append({"path": path, "yaw": yaw, "idx": i, "x": x, "w": view_w})
        print("  View %d: x=%d, yaw=%.1f, size=%dx%d" % (i, x, yaw, view_w, ph))

    return views


# ============================================================
# Step 2: Depth Pro on each view
# ============================================================
def estimate_depths(views, device="cuda"):
    """Run Depth Pro on each view."""
    from depth_pro import create_model_and_transforms, load_rgb

    model, transform = create_model_and_transforms(device=device)
    model.eval()

    for v in views:
        img, _, f_px = load_rgb(v["path"])
        prediction = model.infer(transform(img), f_px=f_px)
        depth = prediction["depth"].cpu().numpy()
        focal = prediction["focallength_px"].item()
        v["depth"] = depth
        v["focal"] = focal
        v["image"] = np.array(Image.open(v["path"]).convert("RGB"))
        h, w = v["image"].shape[:2]
        v["h"] = h
        v["w"] = w
        print("  View %d: depth %.2f-%.2f m, focal=%.0f" % (v["idx"], depth[depth>0.1].min(), depth.max(), focal))

    del model
    torch.cuda.empty_cache()
    return views


# ============================================================
# Step 3: Backproject to 3D point cloud
# ============================================================
def backproject_views(views, subsample=3):
    """Backproject all views to a unified 3D point cloud."""
    all_positions = []
    all_colors = []

    for v in views:
        depth = v["depth"]
        image = v["image"]
        focal = v["focal"]
        h, w = v["h"], v["w"]
        yaw_rad = math.radians(v["yaw"])

        cx, cy = w / 2.0, h / 2.0
        u, vv = np.meshgrid(
            np.arange(0, w, subsample, dtype=np.float32),
            np.arange(0, h, subsample, dtype=np.float32)
        )
        d = depth[::subsample, ::subsample]
        valid = d > 0.1

        # Local camera coordinates
        px = (u - cx) * d / focal
        py = -(vv - cy) * d / focal
        pz = -d  # looking along -Z

        # Rotate by yaw to world coordinates
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        wx = px * cos_y + pz * sin_y
        wz = -px * sin_y + pz * cos_y
        wy = py

        positions = np.stack([wx[valid], wy[valid], wz[valid]], axis=-1)
        colors = image[::subsample, ::subsample][valid].astype(np.float32) / 255.0

        all_positions.append(positions)
        all_colors.append(colors)
        print("  View %d: %d points" % (v["idx"], len(positions)))

    positions = np.concatenate(all_positions).astype(np.float32)
    colors = np.concatenate(all_colors).astype(np.float32)
    print("Total point cloud: %d points" % len(positions))
    return positions, colors


# ============================================================
# Step 4: Initialize and render 3DGS
# ============================================================
def points_to_gaussians_and_render(positions, colors, views, output_dir="/workspace/pano_3dgs"):
    """Convert point cloud to Gaussians and render."""
    from gsplat import rasterization

    os.makedirs(output_dir, exist_ok=True)
    n = len(positions)
    print("Initializing %d Gaussians..." % n)

    means = torch.tensor(positions, device="cuda", dtype=torch.float32)
    # Scale from local density
    scale_val = 0.01  # 1cm base
    scales = torch.full((n, 3), math.log(scale_val), device="cuda", dtype=torch.float32)
    scales[:, 2] = math.log(scale_val * 0.3)  # thin in depth
    quats = torch.zeros((n, 4), device="cuda", dtype=torch.float32)
    quats[:, 0] = 1.0  # identity rotation
    opacities = torch.full((n,), 2.0, device="cuda", dtype=torch.float32)  # high opacity (logit)
    gs_colors = torch.tensor(colors, device="cuda", dtype=torch.float32)

    # Render from each view's camera
    for v in views:
        yaw_rad = math.radians(v["yaw"])
        h, w = v["h"], v["w"]
        focal = v["focal"]

        # View matrix
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        R = np.array([
            [cos_y, 0, -sin_y],
            [0, -1, 0],
            [sin_y, 0, cos_y],
        ], dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)

        viewmat = torch.eye(4, device="cuda", dtype=torch.float32)
        viewmat[:3, :3] = torch.tensor(R)
        viewmat[:3, 3] = torch.tensor(t)

        K = torch.tensor([
            [focal, 0, w/2],
            [0, focal, h/2],
            [0, 0, 1],
        ], device="cuda", dtype=torch.float32)

        renders, alphas, _ = rasterization(
            means=means, quats=quats, scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities), colors=gs_colors,
            viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
            width=w, height=h, packed=False,
        )

        rgb = renders[0].clamp(0, 1).cpu().numpy()
        alpha = alphas[0, :, :, 0].cpu().numpy()
        coverage = 100 * (alpha > 0.01).sum() / (w * h)

        img = Image.fromarray((rgb * 255).astype(np.uint8))
        img.save(os.path.join(output_dir, "render_%02d.png" % v["idx"]))
        print("  View %d (yaw=%.0f): %.1f%% coverage" % (v["idx"], v["yaw"], coverage))

    print("Renders saved to %s" % output_dir)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    pano_path = "/workspace/pano_fixed_final.png"
    if not os.path.exists(pano_path):
        pano_path = "/workspace/pano_fixed_step3.png"

    print("=" * 60)
    print("  Panorama → 3DGS Pipeline")
    print("=" * 60)

    t0 = time.time()

    print("\n[1/4] Slicing panorama...")
    views = slice_panorama(pano_path, num_views=8, fov_deg=80)

    print("\n[2/4] Depth Pro on each view...")
    views = estimate_depths(views)

    print("\n[3/4] Backprojecting to 3D...")
    positions, colors = backproject_views(views, subsample=2)

    print("\n[4/4] Rendering with gsplat...")
    points_to_gaussians_and_render(positions, colors, views)

    print("\nTotal time: %.1fs" % (time.time() - t0))
