"""
Panorama → 3DGS v2 — fixed camera math.

The panorama is a horizontal strip. Each view slice has a known yaw angle.
Camera sits at the center of the room looking outward at each yaw.
"""
import os, sys, math, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace/ml-depth-pro/src")

TARGET_W = 640
TARGET_H = 480


def slice_panorama(pano_path, num_views=12, output_dir="/workspace/pano_views"):
    os.makedirs(output_dir, exist_ok=True)
    pano = Image.open(pano_path).convert("RGB")
    pw, ph = pano.size
    print("Panorama: %dx%d" % (pw, ph))

    # Each view is a vertical strip resized to TARGET
    strip_w = pw // num_views
    views = []
    for i in range(num_views):
        x = i * strip_w
        crop = pano.crop((x, 0, min(x + strip_w + strip_w//2, pw), ph))
        crop = crop.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        path = os.path.join(output_dir, "view_%02d.png" % i)
        crop.save(path)
        # Yaw: panorama maps linearly to angle range
        yaw = (i / num_views) * 245 - 122.5  # spread across ~245 degrees
        views.append({"path": path, "yaw": yaw, "idx": i})
        print("  View %d: yaw=%.1f" % (i, yaw))
    return views


def estimate_depths(views, device="cuda"):
    from depth_pro import create_model_and_transforms, load_rgb

    model, transform = create_model_and_transforms(device=device)
    model.eval()

    for v in views:
        img, _, f_px = load_rgb(v["path"])
        prediction = model.infer(transform(img), f_px=f_px)
        v["depth"] = prediction["depth"].cpu().numpy()
        v["focal"] = prediction["focallength_px"].item()
        v["image"] = np.array(Image.open(v["path"]).convert("RGB").resize((TARGET_W, TARGET_H)))
        print("  View %d: depth %.2f-%.2f m" % (v["idx"], v["depth"][v["depth"]>0.1].min(), v["depth"].max()))

    del model
    torch.cuda.empty_cache()
    return views


def backproject_all(views, subsample=2):
    all_pos = []
    all_col = []

    for v in views:
        depth = v["depth"]
        image = v["image"]
        focal = v["focal"]
        h, w = depth.shape
        yaw_rad = math.radians(v["yaw"])

        # Adjust depth dimensions to match image
        if depth.shape != (TARGET_H, TARGET_W):
            from PIL import Image as PILImage
            depth_img = PILImage.fromarray(depth)
            depth = np.array(depth_img.resize((TARGET_W, TARGET_H), PILImage.BILINEAR))

        cx, cy = TARGET_W / 2.0, TARGET_H / 2.0
        u, vv = np.meshgrid(
            np.arange(0, TARGET_W, subsample, dtype=np.float32),
            np.arange(0, TARGET_H, subsample, dtype=np.float32)
        )
        d = depth[::subsample, ::subsample]
        valid = d > 0.1

        # Camera-local coords
        lx = (u - cx) * d / focal
        ly = -(vv - cy) * d / focal
        lz = -d

        # Rotate by yaw around Y axis
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        wx = lx * cos_y - lz * sin_y
        wz = lx * sin_y + lz * cos_y
        wy = ly

        pos = np.stack([wx[valid], wy[valid], wz[valid]], axis=-1)
        col = image[::subsample, ::subsample][valid].astype(np.float32) / 255.0

        all_pos.append(pos)
        all_col.append(col)

    positions = np.concatenate(all_pos).astype(np.float32)
    colors = np.concatenate(all_col).astype(np.float32)
    print("Total: %d points" % len(positions))
    return positions, colors


def render_views(positions, colors, views, output_dir="/workspace/pano_3dgs_v2"):
    from gsplat import rasterization
    os.makedirs(output_dir, exist_ok=True)

    n = len(positions)
    means = torch.tensor(positions, device="cuda", dtype=torch.float32)
    scale_val = 0.008
    scales = torch.full((n, 3), math.log(scale_val), device="cuda", dtype=torch.float32)
    quats = torch.zeros((n, 4), device="cuda", dtype=torch.float32)
    quats[:, 0] = 1.0
    opacities = torch.full((n,), 2.0, device="cuda", dtype=torch.float32)
    gs_colors = torch.tensor(colors, device="cuda", dtype=torch.float32)

    # Also render novel views not in training set
    all_views = list(views)
    # Add intermediate views
    for i in range(len(views) - 1):
        mid_yaw = (views[i]["yaw"] + views[i+1]["yaw"]) / 2
        all_views.append({"yaw": mid_yaw, "idx": 100 + i, "path": None})

    W, H = TARGET_W, TARGET_H
    focal = views[0]["focal"]

    for v in all_views:
        yaw_rad = math.radians(v["yaw"])

        # Camera at origin, looking along yaw direction
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        # Camera axes: right, down, forward
        fwd = np.array([sin_y, 0, -cos_y], dtype=np.float32)  # forward direction
        right = np.array([cos_y, 0, sin_y], dtype=np.float32)  # right
        up = np.array([0, -1, 0], dtype=np.float32)  # down (image convention)

        R = np.stack([right, up, fwd], axis=0)  # 3x3
        t = np.zeros(3, dtype=np.float32)

        viewmat = torch.eye(4, device="cuda", dtype=torch.float32)
        viewmat[:3, :3] = torch.tensor(R)
        viewmat[:3, 3] = torch.tensor(t)

        K = torch.tensor([
            [focal, 0, W/2],
            [0, focal, H/2],
            [0, 0, 1],
        ], device="cuda", dtype=torch.float32)

        renders, alphas, _ = rasterization(
            means=means, quats=quats, scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities), colors=gs_colors,
            viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
            width=W, height=H, packed=False,
        )

        rgb = renders[0].clamp(0, 1).cpu().numpy()
        alpha = alphas[0, :, :, 0].cpu().numpy()
        coverage = 100 * (alpha > 0.01).sum() / (W * H)

        img = Image.fromarray((rgb * 255).astype(np.uint8))
        prefix = "render" if v["idx"] < 100 else "novel"
        img.save(os.path.join(output_dir, "%s_%02d_yaw%+.0f.png" % (prefix, v["idx"] % 100, v["yaw"])))
        print("  %s %d (yaw=%+.0f): %.1f%%" % (prefix, v["idx"] % 100, v["yaw"], coverage))


if __name__ == "__main__":
    pano_path = "/workspace/pano_fixed_final.png"
    print("=" * 60)
    print("  Panorama → 3DGS v2 (fixed cameras)")
    print("=" * 60)

    t0 = time.time()

    print("\n[1/4] Slicing panorama...")
    views = slice_panorama(pano_path, num_views=10)

    print("\n[2/4] Depth Pro...")
    views = estimate_depths(views)

    print("\n[3/4] Backprojecting...")
    positions, colors = backproject_all(views, subsample=2)

    print("\n[4/4] Rendering...")
    render_views(positions, colors, views)

    print("\nDone in %.1fs" % (time.time() - t0))
