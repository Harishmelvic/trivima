"""
Train proper 3DGS on panorama views using gsplat optimization.
No COLMAP needed - we provide camera poses directly.
"""
import os, sys, math, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/workspace/ml-depth-pro/src")
from gsplat import rasterization

TARGET_W = 640
TARGET_H = 480


def load_views(pano_path, num_views=10):
    """Load panorama, slice into views with known cameras."""
    pano = Image.open(pano_path).convert("RGB")
    pw, ph = pano.size
    strip_w = pw // num_views

    views = []
    for i in range(num_views):
        x = i * strip_w
        crop = pano.crop((x, 0, min(x + strip_w + strip_w//2, pw), ph))
        img = np.array(crop.resize((TARGET_W, TARGET_H), Image.LANCZOS))
        yaw = (i / num_views) * 245 - 122.5
        views.append({"image": img, "yaw": yaw, "idx": i})
    return views


def get_initial_points(views):
    """Get initial 3D points from Depth Pro."""
    from depth_pro import create_model_and_transforms
    model, transform = create_model_and_transforms(device="cuda")
    model.eval()

    all_pos, all_col = [], []
    for v in views:
        img_pil = Image.fromarray(v["image"])
        img_pil.save("/tmp/temp_view.png")

        from depth_pro import load_rgb
        img_t, _, f_px = load_rgb("/tmp/temp_view.png")
        pred = model.infer(transform(img_t), f_px=f_px)
        depth = pred["depth"].cpu().numpy()
        focal = pred["focallength_px"].item()
        v["focal"] = focal

        if depth.shape != (TARGET_H, TARGET_W):
            depth = np.array(Image.fromarray(depth).resize((TARGET_W, TARGET_H), Image.BILINEAR))

        yaw_rad = math.radians(v["yaw"])
        cx, cy = TARGET_W/2, TARGET_H/2
        u, vv = np.meshgrid(np.arange(0, TARGET_W, 3, dtype=np.float32),
                            np.arange(0, TARGET_H, 3, dtype=np.float32))
        d = depth[::3, ::3]
        valid = d > 0.1

        lx = (u - cx) * d / focal
        ly = -(vv - cy) * d / focal
        lz = -d

        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        wx = lx * cos_y - lz * sin_y
        wz = lx * sin_y + lz * cos_y

        all_pos.append(np.stack([wx[valid], ly[valid], wz[valid]], axis=-1))
        all_col.append(v["image"][::3, ::3][valid].astype(np.float32) / 255.0)

    del model
    torch.cuda.empty_cache()

    return np.concatenate(all_pos).astype(np.float32), np.concatenate(all_col).astype(np.float32)


def make_viewmat(yaw_deg):
    yaw = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    fwd = np.array([sin_y, 0, -cos_y], dtype=np.float32)
    right = np.array([cos_y, 0, sin_y], dtype=np.float32)
    up = np.array([0, -1, 0], dtype=np.float32)
    R = np.stack([right, up, fwd], axis=0)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = R
    return torch.tensor(m, device="cuda")


def train_3dgs(positions, colors, views, iterations=500):
    """Optimize Gaussians against training views."""
    n = len(positions)
    print("Training %d Gaussians for %d iterations..." % (n, iterations))

    # Trainable parameters
    means = torch.tensor(positions, device="cuda", dtype=torch.float32, requires_grad=True)
    scales = torch.full((n, 3), math.log(0.01), device="cuda", dtype=torch.float32, requires_grad=True)
    quats = torch.zeros((n, 4), device="cuda", dtype=torch.float32)
    quats[:, 0] = 1.0
    quats = quats.requires_grad_(True)
    opacities = torch.full((n,), 2.0, device="cuda", dtype=torch.float32, requires_grad=True)
    gs_colors = torch.tensor(colors, device="cuda", dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([
        {"params": [means], "lr": 1e-4},
        {"params": [scales], "lr": 5e-3},
        {"params": [quats], "lr": 1e-3},
        {"params": [opacities], "lr": 5e-2},
        {"params": [gs_colors], "lr": 2.5e-3},
    ])

    focal = views[0]["focal"]
    K = torch.tensor([[focal, 0, TARGET_W/2], [0, focal, TARGET_H/2], [0, 0, 1]],
                     device="cuda", dtype=torch.float32)

    # Training images
    gt_images = []
    viewmats = []
    for v in views:
        gt = torch.tensor(v["image"].astype(np.float32) / 255.0, device="cuda")
        gt_images.append(gt)
        viewmats.append(make_viewmat(v["yaw"]))

    for step in range(iterations):
        # Random view
        vi = step % len(views)
        gt = gt_images[vi]
        vm = viewmats[vi]

        # Render
        renders, alphas, _ = rasterization(
            means=means,
            quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=torch.sigmoid(gs_colors),
            viewmats=vm.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=TARGET_W, height=TARGET_H,
            packed=False,
        )

        pred = renders[0]  # (H, W, 3)

        # L1 + SSIM loss
        loss = F.l1_loss(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print("  Step %d/%d: loss=%.4f" % (step+1, iterations, loss.item()))

    return means, scales, quats, opacities, gs_colors, K


def render_final(means, scales, quats, opacities, gs_colors, K, views, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Render training + novel views
    yaws = [v["yaw"] for v in views]
    # Add novel views between training views
    for i in range(len(yaws) - 1):
        yaws.append((yaws[i] + yaws[i+1]) / 2)

    for i, yaw in enumerate(sorted(yaws)):
        vm = make_viewmat(yaw)
        with torch.no_grad():
            renders, alphas, _ = rasterization(
                means=means,
                quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
                scales=torch.exp(scales),
                opacities=torch.sigmoid(opacities),
                colors=torch.sigmoid(gs_colors),
                viewmats=vm.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=TARGET_W, height=TARGET_H,
                packed=False,
            )
        rgb = renders[0].clamp(0, 1).cpu().numpy()
        alpha = alphas[0, :, :, 0].cpu().numpy()
        coverage = 100 * (alpha > 0.01).sum() / (TARGET_W * TARGET_H)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(
            os.path.join(output_dir, "final_%02d_yaw%+.0f.png" % (i, yaw)))
        print("  yaw=%+.0f: %.1f%%" % (yaw, coverage))


if __name__ == "__main__":
    pano_path = "/workspace/pano_fixed_final.png"
    print("=" * 60)
    print("  Train 3DGS from Panorama")
    print("=" * 60)

    t0 = time.time()

    print("\n[1/4] Loading views...")
    views = load_views(pano_path, num_views=10)

    print("\n[2/4] Depth Pro → 3D points...")
    positions, colors = get_initial_points(views)
    print("  %d initial points" % len(positions))

    print("\n[3/4] Optimizing 3DGS...")
    means, scales, quats, opacities, gs_colors, K = train_3dgs(
        positions, colors, views, iterations=300)

    print("\n[4/4] Rendering final views...")
    render_final(means, scales, quats, opacities, gs_colors, K, views, "/workspace/pano_3dgs_trained")

    print("\nDone in %.1fs" % (time.time() - t0))
