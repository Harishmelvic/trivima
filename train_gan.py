#!/usr/bin/env python3
"""
Trivima GAN Training — Real Photo Pairs
=========================================
Downloads indoor room photos, runs Depth Pro to generate cell buffers,
then trains the Pix2PixHD-Lite GAN on (flat_buffer, real_photo) pairs.

Usage:
    python train_gan.py --step all --num_images 100 --epochs 10 --batch_size 8
    python train_gan.py --step data --num_images 100
    python train_gan.py --step train --epochs 10
    python train_gan.py --step eval --checkpoint checkpoints/best.pt
"""

import argparse, os, sys, math, time, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Config
# ============================================================
DATA_DIR = "data/gan_pairs"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
EVAL_DIR = "logs/eval_samples"
IMG_SIZE = 256  # Training resolution


# ============================================================
# Step 1: Download indoor room photos
# ============================================================
def download_photos(num_images=100, output_dir="data/room_photos"):
    """Download indoor room photos from HuggingFace MIT Indoor Scenes."""
    os.makedirs(output_dir, exist_ok=True)

    existing = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"  Found {len(existing)} photos in {output_dir}")
    if len(existing) == 0:
        print("  ERROR: No photos found. Download room photos to data/room_photos/ first.")
        print("  Use icrawler or manual download.")
        sys.exit(1)
    return output_dir


# ============================================================
# Step 2: Generate training pairs using Depth Pro
# ============================================================
class VoxelRenderer:
    """Persistent EGL renderer for generating training pair voxel renders.

    Created once, reused for all training images. Avoids the EGL+CUDA
    context creation/destruction conflict that causes core dumps.
    """

    _instance = None

    @classmethod
    def get(cls, render_size=256):
        if cls._instance is None:
            cls._instance = cls(render_size)
        return cls._instance

    def __init__(self, render_size=256):
        import moderngl
        self.ctx = moderngl.create_context(standalone=True, backend="egl")
        self.W = self.H = render_size

        VS = """
#version 330
in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;
uniform mat4 u_projection;
uniform mat4 u_view;
out vec3 v_color;
out vec3 v_normal;
void main() {
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
    v_color = in_color;
    v_normal = in_normal;
}
"""
        FS = """
#version 330
in vec3 v_color;
in vec3 v_normal;
out vec4 frag_color;
void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(vec3(0.3, 0.8, 0.4));
    float d = max(dot(N, L), 0.0) * 0.55;
    frag_color = vec4(v_color * (0.18 + d), 1.0);
}
"""
        self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.fbo = self.ctx.simple_framebuffer((self.W, self.H))
        self.view = np.eye(4, dtype=np.float32)

    def render(self, cell_pos, cell_col, cell_nrm, cell_size, fov_deg):
        """Render cells as flat-shaded voxels. Returns (H, W, 3) uint8."""
        import moderngl

        cube_v = np.array([
            [0.5,0.5,0.5],[-0.5,0.5,0.5],[-0.5,-0.5,0.5],[0.5,0.5,0.5],[-0.5,-0.5,0.5],[0.5,-0.5,0.5],
            [-0.5,0.5,-0.5],[0.5,0.5,-0.5],[0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],
            [0.5,0.5,-0.5],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,-0.5],[0.5,-0.5,0.5],[0.5,-0.5,-0.5],
            [-0.5,0.5,0.5],[-0.5,0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5,0.5,0.5],[-0.5,-0.5,-0.5],[-0.5,-0.5,0.5],
            [-0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,-0.5],[-0.5,0.5,0.5],[0.5,0.5,-0.5],[-0.5,0.5,-0.5],
            [-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[0.5,-0.5,0.5],[-0.5,-0.5,-0.5],[0.5,-0.5,0.5],[-0.5,-0.5,0.5],
        ], dtype=np.float32)
        cube_n = np.array(
            [[0,0,1]]*6+[[0,0,-1]]*6+[[1,0,0]]*6+[[-1,0,0]]*6+[[0,1,0]]*6+[[0,-1,0]]*6,
            dtype=np.float32)

        n = len(cell_pos)
        sizes = np.full(n, cell_size, dtype=np.float32)
        scaled = cube_v[None,:,:] * sizes[:,None,None]
        translated = scaled + cell_pos[:,None,:]
        colors_exp = np.broadcast_to(cell_col[:,None,:], (n, 36, 3))
        normals_exp = np.broadcast_to(cube_n[None,:,:], (n, 36, 3))

        verts = np.zeros((n * 36, 9), dtype=np.float32)
        verts[:, 0:3] = translated.reshape(-1, 3)
        verts[:, 3:6] = colors_exp.reshape(-1, 3)
        verts[:, 6:9] = normals_exp.reshape(-1, 3)

        # Projection from FOV
        f_val = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0,0] = f_val; proj[1,1] = f_val
        proj[2,2] = -1.0004; proj[2,3] = -0.02; proj[3,2] = -1.0

        self.fbo.use()
        self.fbo.clear(0.12, 0.12, 0.18, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog["u_projection"].write(proj.T.tobytes())
        self.prog["u_view"].write(self.view.T.tobytes())

        # Batched rendering (EGL ~3M vertex limit)
        MAX_VERTS = 75000 * 36
        for start in range(0, len(verts), MAX_VERTS):
            batch = verts[start:start + MAX_VERTS]
            vbo = self.ctx.buffer(batch.tobytes())
            vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 3f", "in_position", "in_color", "in_normal")])
            vao.render(moderngl.TRIANGLES)
            vbo.release()
            vao.release()

        self.ctx.finish()
        pixels = np.frombuffer(self.fbo.read(), dtype=np.uint8).reshape(self.H, self.W, 3)
        return np.flipud(pixels)


def generate_pairs(photo_dir, output_dir=DATA_DIR, device="cuda", force=False):
    """Two-pass pair generation to avoid EGL+CUDA conflict.

    Pass 1: Depth Pro (CUDA) -> save cell data + target photos
    Pass 2: Voxel renderer (EGL) -> render cells -> save buffers
    """
    cells_dir = os.path.join(output_dir, "cells")
    os.makedirs(os.path.join(output_dir, "buffers"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "targets"), exist_ok=True)
    os.makedirs(cells_dir, exist_ok=True)

    photos = sorted([f for f in os.listdir(photo_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  Generating pairs from {len(photos)} photos (two-pass)")

    # === PASS 1: Depth Pro (CUDA) — save cells + targets ===
    print("  [Pass 1/2] Depth Pro -> cells...")
    from trivima.perception.depth_pro import DepthProEstimator
    from trivima.perception.depth_smoothing import bilateral_depth_smooth

    model = DepthProEstimator(device=device)
    model.load()

    n_pass1 = 0
    for i, fname in enumerate(photos):
        fpath = os.path.join(photo_dir, fname)
        cell_path = os.path.join(cells_dir, f"{i:04d}.npz")
        tgt_path = os.path.join(output_dir, "targets", f"{i:04d}.npy")

        if not force and os.path.exists(cell_path) and os.path.exists(tgt_path):
            n_pass1 += 1
            continue

        try:
            image = np.array(Image.open(fpath).convert("RGB"))
            h, w = image.shape[:2]
            max_side = 1024
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = np.array(Image.fromarray(image).resize((new_w, new_h), Image.LANCZOS))
                h, w = new_h, new_w

            result = model.estimate(image)
            depth = result["depth"]
            focal = result["focal_length"]
            smoothed = bilateral_depth_smooth(depth, image, spatial_sigma=2.5, color_sigma=25.0)

            # Backproject + bin into cells
            cx, cy = w / 2.0, h / 2.0
            cs = 0.02
            u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            valid = smoothed > 0.1
            px = (u - cx) * smoothed / focal
            py = -(v - cy) * smoothed / focal
            pz = -smoothed

            positions = np.stack([px[valid], py[valid], pz[valid]], axis=-1).astype(np.float32)
            colors = image[valid].astype(np.float32) / 255.0

            cell_idx = np.floor(positions / cs).astype(np.int32)
            bins = {}
            for pi in range(len(positions)):
                key = tuple(cell_idx[pi])
                if key not in bins:
                    bins[key] = {"ps": np.zeros(3, dtype=np.float64),
                                 "cs": np.zeros(3, dtype=np.float64), "n": 0}
                bins[key]["ps"] += positions[pi]
                bins[key]["cs"] += colors[pi]
                bins[key]["n"] += 1

            n_cells = len(bins)
            cell_pos = np.zeros((n_cells, 3), dtype=np.float32)
            cell_col = np.zeros((n_cells, 3), dtype=np.float32)
            cell_nrm = np.zeros((n_cells, 3), dtype=np.float32)
            for ci, (key, cell) in enumerate(bins.items()):
                cell_pos[ci] = (cell["ps"] / cell["n"]).astype(np.float32)
                cell_col[ci] = np.clip(cell["cs"] / cell["n"], 0, 1).astype(np.float32)
                to_cam = -cell_pos[ci]
                nm = np.linalg.norm(to_cam)
                cell_nrm[ci] = to_cam / nm if nm > 1e-6 else np.array([0,0,1], dtype=np.float32)

            fov_deg = math.degrees(2.0 * math.atan(h / (2.0 * focal)))

            # Save cells for Pass 2
            np.savez_compressed(cell_path, pos=cell_pos, col=cell_col, nrm=cell_nrm,
                                fov=np.array([fov_deg]))

            # Save target photo
            target = np.array(Image.fromarray(image).resize(
                (IMG_SIZE, IMG_SIZE), Image.LANCZOS)).astype(np.float32) / 255.0
            np.save(tgt_path, target)
            n_pass1 += 1

            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(photos)}] {n_cells} cells")

        except Exception as e:
            print(f"    FAILED {fname}: {e}")
            continue

    model.unload()
    torch.cuda.empty_cache()
    print(f"  Pass 1 done: {n_pass1} cell files saved")

    # === PASS 2: Buffer Projection (EGL) — project cells into clean 2D maps ===
    print("  [Pass 2/2] Projecting cells into smooth buffers...")
    from trivima.texturing.gpu_buffer_renderer import GPUBufferRenderer
    renderer = GPUBufferRenderer(width=IMG_SIZE, height=IMG_SIZE)

    n_pass2 = 0
    cell_files = sorted([f for f in os.listdir(cells_dir) if f.endswith('.npz')])
    for f in cell_files:
        idx = f.replace('.npz', '')
        buf_path = os.path.join(output_dir, "buffers", f"{idx}.npy")
        cell_path = os.path.join(cells_dir, f)

        if not force and os.path.exists(buf_path):
            n_pass2 += 1
            continue

        try:
            data = np.load(cell_path)
            cell_pos = data["pos"]
            cell_col = data["col"]
            cell_nrm = data["nrm"]
            fov_deg = float(data["fov"][0])

            # Project cells as point splats — clean smooth buffers, no cubes
            result = renderer.render(cell_pos, cell_col, cell_nrm, 0.02, fov_deg)
            buf_8ch = result["combined_8ch"]

            np.save(buf_path, buf_8ch)
            n_pass2 += 1

        except Exception as e:
            print(f"    FAILED render {f}: {e}")
            continue

    print(f"  Pass 2 done: {n_pass2} buffer projections saved")

    total = min(n_pass1, n_pass2)
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump({"total": total, "pass1": n_pass1, "pass2": n_pass2}, f, indent=2)
    return total


# ============================================================
# Step 3: Dataset + DataLoader
# ============================================================
class GANPairDataset(Dataset):
    """Loads pre-generated (buffer, target) pairs with augmentation."""

    def __init__(self, data_dir=DATA_DIR, augment=True):
        self.buf_dir = os.path.join(data_dir, "buffers")
        self.tgt_dir = os.path.join(data_dir, "targets")
        self.augment = augment

        self.indices = sorted([
            f.replace(".npy", "") for f in os.listdir(self.buf_dir) if f.endswith(".npy")
        ])
        print(f"  Dataset: {len(self.indices)} pairs")

    def __len__(self):
        # With augmentation, generate multiple crops per image
        return len(self.indices) * (4 if self.augment else 1)

    def __getitem__(self, idx):
        real_idx = idx % len(self.indices)
        buf = np.load(os.path.join(self.buf_dir, f"{self.indices[real_idx]}.npy"))
        tgt = np.load(os.path.join(self.tgt_dir, f"{self.indices[real_idx]}.npy"))

        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                buf = buf[:, ::-1, :].copy()
                tgt = tgt[:, ::-1, :].copy()
                # Flip normal X component
                buf[..., 4] = -buf[..., 4]

            # Random brightness/contrast jitter on target
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                tgt = np.clip(tgt * brightness, 0, 1)

            # Random color jitter on target
            if np.random.random() > 0.3:
                jitter = np.random.uniform(0.9, 1.1, size=3).astype(np.float32)
                tgt = np.clip(tgt * jitter, 0, 1)

        # Convert to tensor: (C, H, W), scale target to [-1, 1]
        buf_tensor = torch.from_numpy(buf.transpose(2, 0, 1)).float()
        tgt_tensor = torch.from_numpy(tgt.transpose(2, 0, 1)).float() * 2.0 - 1.0

        return buf_tensor, tgt_tensor


# ============================================================
# Step 4: Training loop
# ============================================================
def train(epochs=10, batch_size=8, lr_g=2e-4, lr_d=1e-4, data_dir=DATA_DIR, device="cuda"):
    """Train the Pix2PixHD-Lite GAN."""
    from trivima.texturing.models.pix2pix_lite import Pix2PixLiteTrainer

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    dataset = GANPairDataset(data_dir, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    print(f"  Training: {len(dataset)} samples, {len(loader)} batches/epoch")
    print(f"  Config: epochs={epochs}, batch_size={batch_size}, lr_g={lr_g}, lr_d={lr_d}")

    trainer = Pix2PixLiteTrainer(lr_g=lr_g, lr_d=lr_d, device=device)

    # Count parameters
    g_params = sum(p.numel() for p in trainer.generator.parameters())
    d_params = sum(p.numel() for p in trainer.discriminator.parameters())
    print(f"  Generator: {g_params/1e6:.1f}M params")
    print(f"  Discriminator: {d_params/1e6:.1f}M params")

    best_l1 = float("inf")
    history = []

    for epoch in range(epochs):
        t_epoch = time.time()
        epoch_losses = {"loss_d": [], "loss_g": [], "loss_g_adv": [], "loss_g_l1": [], "loss_g_perceptual": []}

        trainer.generator.train()
        trainer.discriminator.train()

        for batch_idx, (condition, target) in enumerate(loader):
            condition = condition.to(device)
            target = target.to(device)

            losses = trainer.train_step(condition, target)

            for k, v in losses.items():
                epoch_losses[k].append(v)

            if (batch_idx + 1) % 10 == 0:
                avg_l1 = np.mean(epoch_losses["loss_g_l1"][-10:])
                avg_d = np.mean(epoch_losses["loss_d"][-10:])
                print(f"    [{epoch+1}/{epochs}] batch {batch_idx+1}/{len(loader)} "
                      f"D={avg_d:.3f} G_l1={avg_l1:.3f}")

        # Epoch summary
        epoch_time = time.time() - t_epoch
        avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        val_l1 = avg["loss_g_l1"] / 100.0  # Undo lambda_l1 scaling
        history.append({"epoch": epoch + 1, **avg, "val_l1": val_l1, "time_s": epoch_time})

        print(f"  Epoch {epoch+1}/{epochs}: "
              f"D={avg['loss_d']:.3f} G={avg['loss_g']:.3f} "
              f"L1={val_l1:.4f} Perc={avg['loss_g_perceptual']:.3f} "
              f"({epoch_time:.1f}s)")

        # Save checkpoint
        if val_l1 < best_l1:
            best_l1 = val_l1
            trainer.save(os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  -> New best L1: {best_l1:.4f}")

        trainer.save(os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pt"))

        # Save a few eval samples every 2 epochs
        if (epoch + 1) % 2 == 0 or epoch == 0:
            save_eval_samples(trainer, dataset, epoch + 1, device)

    # Save training history
    with open(os.path.join(LOG_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    trainer.save(os.path.join(CHECKPOINT_DIR, "final.pt"))
    print(f"\n  Training complete. Best L1: {best_l1:.4f}")
    return best_l1


def save_eval_samples(trainer, dataset, epoch, device, num_samples=4):
    """Save side-by-side comparison images."""
    eval_dir = os.path.join(EVAL_DIR, f"epoch_{epoch:03d}")
    os.makedirs(eval_dir, exist_ok=True)

    trainer.generator.eval()
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset.indices))):
            buf_np = np.load(os.path.join(dataset.buf_dir, f"{dataset.indices[i]}.npy"))
            tgt_np = np.load(os.path.join(dataset.tgt_dir, f"{dataset.indices[i]}.npy"))

            buf_tensor = torch.from_numpy(buf_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            tgt_tensor = torch.from_numpy(tgt_np.transpose(2, 0, 1)).float() * 2.0 - 1.0

            fake_rgb, _ = trainer.generator(buf_tensor)
            fake_np = ((fake_rgb[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)

            # Side-by-side: input_albedo | GAN output | ground truth
            input_vis = buf_np[..., :3]  # Just albedo channels for visualization
            comparison = np.concatenate([input_vis, fake_np, tgt_np], axis=1)
            comparison = (comparison * 255).clip(0, 255).astype(np.uint8)

            Image.fromarray(comparison).save(os.path.join(eval_dir, f"sample_{i:02d}.png"))

    trainer.generator.train()


# ============================================================
# Step 5: Evaluation
# ============================================================
def evaluate(checkpoint_path, data_dir=DATA_DIR, device="cuda"):
    """Evaluate trained GAN on the dataset."""
    from trivima.texturing.models.pix2pix_lite import Pix2PixLiteTrainer

    print(f"  Loading checkpoint: {checkpoint_path}")
    trainer = Pix2PixLiteTrainer(device=device)
    trainer.load(checkpoint_path)
    trainer.generator.eval()

    dataset = GANPairDataset(data_dir, augment=False)
    os.makedirs(EVAL_DIR, exist_ok=True)

    total_l1 = 0.0
    total_ssim = 0.0
    total_samples = 0
    inference_times = []

    with torch.no_grad():
        for i in range(min(len(dataset.indices), 20)):  # Eval on up to 20 samples
            buf_np = np.load(os.path.join(dataset.buf_dir, f"{dataset.indices[i]}.npy"))
            tgt_np = np.load(os.path.join(dataset.tgt_dir, f"{dataset.indices[i]}.npy"))

            buf_tensor = torch.from_numpy(buf_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            # Measure inference time
            torch.cuda.synchronize()
            t0 = time.time()
            fake_rgb, fake_light = trainer.generator(buf_tensor)
            torch.cuda.synchronize()
            inf_ms = (time.time() - t0) * 1000
            inference_times.append(inf_ms)

            fake_np = ((fake_rgb[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)

            # L1
            l1 = np.mean(np.abs(fake_np - tgt_np))
            total_l1 += l1

            # Simple SSIM approximation (structural similarity)
            ssim_val = compute_ssim(fake_np, tgt_np)
            total_ssim += ssim_val

            total_samples += 1

            # Save comparison
            input_vis = buf_np[..., :3]
            comparison = np.concatenate([input_vis, fake_np, tgt_np], axis=1)
            comparison = (comparison * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(comparison).save(os.path.join(EVAL_DIR, f"eval_{i:02d}.png"))

    avg_l1 = total_l1 / max(total_samples, 1)
    avg_ssim = total_ssim / max(total_samples, 1)
    avg_inf = np.mean(inference_times)

    print(f"\n  === Evaluation Results ===")
    print(f"  Samples:        {total_samples}")
    print(f"  Avg L1:         {avg_l1:.4f}")
    print(f"  Avg SSIM:       {avg_ssim:.4f}")
    print(f"  Avg Inference:  {avg_inf:.1f}ms")
    print(f"  Comparisons saved to {EVAL_DIR}/")

    results = {
        "avg_l1": float(avg_l1),
        "avg_ssim": float(avg_ssim),
        "avg_inference_ms": float(avg_inf),
        "num_samples": total_samples,
    }
    with open(os.path.join(LOG_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def compute_ssim(img1, img2, window_size=7):
    """Simple SSIM between two (H, W, 3) float images in [0, 1]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Convert to grayscale
    g1 = 0.299 * img1[..., 0] + 0.587 * img1[..., 1] + 0.114 * img1[..., 2]
    g2 = 0.299 * img2[..., 0] + 0.587 * img2[..., 1] + 0.114 * img2[..., 2]

    from scipy.ndimage import uniform_filter
    mu1 = uniform_filter(g1, window_size)
    mu2 = uniform_filter(g2, window_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = uniform_filter(g1 ** 2, window_size) - mu1_sq
    sigma2_sq = uniform_filter(g2 ** 2, window_size) - mu2_sq
    sigma12 = uniform_filter(g1 * g2, window_size) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Trivima GAN Training")
    parser.add_argument("--step", type=str, default="all",
                        choices=["data", "train", "eval", "all"])
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true",
                        help="Force regenerate training pairs (delete old)")
    args = parser.parse_args()

    print("=" * 60)
    print("Trivima GAN Training — Real Photo Pairs")
    print("=" * 60)

    if args.step in ("data", "all"):
        print("\n[Step 1] Downloading room photos...")
        photo_dir = download_photos(num_images=args.num_images)

        print("\n[Step 2] Generating training pairs (voxel render -> photo)...")
        if args.force:
            import shutil
            for sub in ["buffers", "targets"]:
                p = os.path.join(DATA_DIR, sub)
                if os.path.exists(p):
                    shutil.rmtree(p)
                    print(f"  Cleared {p}")
        n_pairs = generate_pairs(photo_dir, device=args.device, force=args.force)
        print(f"  Total pairs: {n_pairs}")
        torch.cuda.empty_cache()

    if args.step in ("train", "all"):
        print("\n[Step 3] Training GAN...")
        best_l1 = train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_g=args.lr_g,
            lr_d=args.lr_d,
            device=args.device,
        )

        # Decision point
        print("\n  === Decision Point ===")
        if best_l1 < 0.05:
            print("  val_l1 < 0.05 -> GAN learns well. Proceed to longer training.")
        elif best_l1 < 0.10:
            print("  val_l1 0.05-0.10 -> GAN learns slowly. Try more epochs or data.")
        else:
            print("  val_l1 > 0.10 -> Check loss curves and eval samples before proceeding.")

    if args.step in ("eval", "all"):
        ckpt = args.checkpoint or os.path.join(CHECKPOINT_DIR, "best.pt")
        if os.path.exists(ckpt):
            print(f"\n[Step 4] Evaluating...")
            evaluate(ckpt, device=args.device)
        else:
            print(f"\n  No checkpoint found at {ckpt}")

    print("\nDone!")


if __name__ == "__main__":
    main()
