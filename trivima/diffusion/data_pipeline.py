"""
Data pipeline for training the Voxel DiT.

Two data sources:
  1. Synthetic: 3D-FRONT rooms → render photos + voxelize → (photo, grid) pairs
  2. Real: existing pipeline (Depth Pro + SAM + fill) on real photos → pairs

Each pair:
  photo: (3, 256, 256) — RGB room photo
  grid:  (8, N, N, N) — voxel grid
    channels: RGB(3) + density(1) + normal(3) + label(1)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image


class VoxelPairDataset(Dataset):
    """Dataset of (photo, voxel_grid) pairs for DiT training."""

    def __init__(self, data_dir: str, grid_size: int = 64, img_size: int = 256):
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.img_size = img_size

        self.photo_dir = os.path.join(data_dir, "photos")
        self.grid_dir = os.path.join(data_dir, "grids")

        self.indices = sorted([
            f.replace(".npy", "") for f in os.listdir(self.grid_dir)
            if f.endswith(".npy")
        ]) if os.path.exists(self.grid_dir) else []

        print(f"  VoxelPairDataset: {len(self.indices)} pairs from {data_dir}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        name = self.indices[idx]

        # Load photo
        photo_path = os.path.join(self.photo_dir, f"{name}.jpg")
        if not os.path.exists(photo_path):
            photo_path = os.path.join(self.photo_dir, f"{name}.png")
        photo = Image.open(photo_path).convert("RGB").resize(
            (self.img_size, self.img_size), Image.LANCZOS)
        photo = np.array(photo).astype(np.float32) / 255.0
        photo = torch.from_numpy(photo.transpose(2, 0, 1))  # (3, H, W)

        # Load voxel grid
        grid = np.load(os.path.join(self.grid_dir, f"{name}.npy"))
        grid = torch.from_numpy(grid).float()  # (C, N, N, N)

        # Normalize grid to [-1, 1] for diffusion
        grid = grid * 2.0 - 1.0

        return photo, grid


def cells_to_voxel_grid(
    cell_pos: np.ndarray,
    cell_col: np.ndarray,
    cell_nrm: np.ndarray,
    cell_labels: np.ndarray = None,
    grid_size: int = 64,
    cell_size: float = None,
) -> np.ndarray:
    """Convert cell data to a dense N×N×N×8 voxel grid.

    Maps the sparse cell positions into a regular grid.
    Empty voxels get zero density.

    Returns: (8, N, N, N) float32 array normalized to [0, 1]
    """
    n = len(cell_pos)
    if n == 0:
        return np.zeros((8, grid_size, grid_size, grid_size), dtype=np.float32)

    # Compute bounds and cell size
    pos_min = cell_pos.min(axis=0)
    pos_max = cell_pos.max(axis=0)
    extent = pos_max - pos_min
    if cell_size is None:
        cell_size = max(extent) / (grid_size - 1)

    # Map positions to grid indices
    grid_idx = np.floor((cell_pos - pos_min) / max(cell_size, 1e-6)).astype(int)
    grid_idx = np.clip(grid_idx, 0, grid_size - 1)

    # Fill grid
    grid = np.zeros((8, grid_size, grid_size, grid_size), dtype=np.float32)

    for i in range(n):
        ix, iy, iz = grid_idx[i]
        # RGB color
        grid[0:3, ix, iy, iz] = cell_col[i]
        # Density = 1.0 (occupied)
        grid[3, ix, iy, iz] = 1.0
        # Normal
        grid[4:7, ix, iy, iz] = (cell_nrm[i] + 1.0) / 2.0  # [-1,1] → [0,1]
        # Label
        if cell_labels is not None:
            grid[7, ix, iy, iz] = min(cell_labels[i] / 20.0, 1.0)

    return grid


def generate_synthetic_pair(room_mesh_path: str, grid_size: int = 64):
    """Generate a (photo, voxel_grid) pair from a 3D-FRONT room mesh.

    1. Load room mesh
    2. Render a photo from a random camera position
    3. Voxelize the mesh into a grid
    4. Return (photo, grid)
    """
    # This requires 3D-FRONT data to be downloaded
    # For now, return a synthetic test pair
    raise NotImplementedError(
        "3D-FRONT mesh processing requires downloading the dataset. "
        "Use generate_real_pair() with existing pipeline instead."
    )


def generate_real_pair(
    image_path: str,
    grid_size: int = 64,
    cell_size: float = 0.02,
    device: str = "cuda",
) -> tuple:
    """Generate a (photo, voxel_grid) pair using the existing pipeline.

    Pipeline: Depth Pro → backproject → cells → SAM → fill → grid

    Returns:
        (photo_tensor, grid_tensor) ready for training
    """
    import torch as th
    from trivima.perception.depth_pro import DepthProEstimator
    from trivima.perception.depth_smoothing import bilateral_depth_smooth
    from trivima.construction.volume_fill import fill_volume

    # Load and run Depth Pro
    model = DepthProEstimator(device=device)
    model.load()
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]

    result = model.estimate(image)
    depth = result["depth"]
    focal = result["focal_length"]
    model.unload()
    th.cuda.empty_cache()

    smoothed = bilateral_depth_smooth(depth, image, spatial_sigma=2.5, color_sigma=25.0)

    # Backproject
    import math
    cx, cy = w / 2.0, h / 2.0
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = smoothed > 0.1
    px = (u - cx) * smoothed / focal
    py = -(v - cy) * smoothed / focal
    pz = -smoothed

    positions = np.stack([px[valid], py[valid], pz[valid]], axis=-1).astype(np.float32)
    colors = image[valid].astype(np.float32) / 255.0

    # Bin into cells
    cs = cell_size
    cell_idx = np.floor(positions / cs).astype(np.int32)
    bins = {}
    for i in range(len(positions)):
        key = tuple(cell_idx[i])
        if key not in bins:
            bins[key] = {"ps": np.zeros(3, dtype=np.float64),
                         "cs": np.zeros(3, dtype=np.float64), "n": 0}
        bins[key]["ps"] += positions[i]
        bins[key]["cs"] += colors[i]
        bins[key]["n"] += 1

    n_cells = len(bins)
    cell_pos = np.zeros((n_cells, 3), dtype=np.float32)
    cell_col = np.zeros((n_cells, 3), dtype=np.float32)
    cell_nrm = np.zeros((n_cells, 3), dtype=np.float32)
    for i, (key, cell) in enumerate(bins.items()):
        cell_pos[i] = (cell["ps"] / cell["n"]).astype(np.float32)
        cell_col[i] = np.clip(cell["cs"] / cell["n"], 0, 1).astype(np.float32)
        to_cam = -cell_pos[i]
        nm = np.linalg.norm(to_cam)
        cell_nrm[i] = to_cam / nm if nm > 1e-6 else np.array([0, 0, 1], dtype=np.float32)

    # Volume fill
    cell_pos, cell_col, cell_nrm = fill_volume(
        cell_pos, cell_col, cell_nrm, cell_size=cs, default_depth=0.12)

    # Convert to dense grid
    grid = cells_to_voxel_grid(cell_pos, cell_col, cell_nrm, grid_size=grid_size)

    # Photo tensor
    photo = np.array(Image.fromarray(image).resize((256, 256), Image.LANCZOS))
    photo = photo.astype(np.float32) / 255.0

    return photo, grid


def batch_generate_pairs(
    photo_dir: str,
    output_dir: str,
    grid_size: int = 64,
    device: str = "cuda",
    max_images: int = None,
):
    """Process all photos in a directory into training pairs.

    This is the data engine — runs the existing pipeline on real photos
    to generate training data for the diffusion transformer.
    """
    os.makedirs(os.path.join(output_dir, "photos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "grids"), exist_ok=True)

    photos = sorted([f for f in os.listdir(photo_dir)
                     if f.endswith(('.jpg', '.jpeg', '.png'))])
    if max_images:
        photos = photos[:max_images]

    print(f"  Generating {len(photos)} pairs...")

    for i, fname in enumerate(photos):
        grid_path = os.path.join(output_dir, "grids", f"{i:04d}.npy")
        photo_path = os.path.join(output_dir, "photos", f"{i:04d}.jpg")

        if os.path.exists(grid_path):
            continue

        try:
            photo, grid = generate_real_pair(
                os.path.join(photo_dir, fname),
                grid_size=grid_size, device=device)

            np.save(grid_path, grid)
            Image.fromarray((photo * 255).astype(np.uint8)).save(photo_path)

            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(photos)}] done")

        except Exception as e:
            print(f"    FAILED {fname}: {e}")
            continue

    n_done = len([f for f in os.listdir(os.path.join(output_dir, "grids"))
                  if f.endswith(".npy")])
    print(f"  Generated {n_done} pairs in {output_dir}")
    return n_done
