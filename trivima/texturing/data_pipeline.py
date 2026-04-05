"""
Training data pipeline — generates paired (cell_render, photograph) samples
from ScanNet and Matterport3D datasets.

Process:
  1. Voxelize ScanNet/Matterport3D meshes at 5cm resolution
  2. Compute per-voxel albedo, normal, depth, semantic label
  3. Render as flat cell buffers from original camera poses
  4. Pair with original photographs as ground truth
  5. Apply augmentation to reach 200K+ effective training pairs

This runs as a background job starting in Week 1 (prepare_training_data.py).
By Week 3, data is ready for GAN and ControlNet training.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional
import json


@dataclass
class TrainingPair:
    """One training sample: cell render buffers paired with ground truth photo."""
    condition: np.ndarray   # (H, W, 8) float32 — 8-channel cell buffer input
    target: np.ndarray      # (H, W, 3) float32 — ground truth photograph [0,1]
    scene_id: str
    camera_idx: int


class ScanNetVoxelizer:
    """Voxelizes ScanNet meshes into cell-grid-compatible representations."""

    def __init__(self, cell_size: float = 0.05, resolution: int = 512):
        self.cell_size = cell_size
        self.resolution = resolution

    def process_scene(self, scene_path: str) -> dict:
        """Process one ScanNet scene into voxelized cell data.

        Args:
            scene_path: path to ScanNet scene directory (contains .ply mesh, .sens file)

        Returns:
            dict with keys: positions, albedo, normals, labels, camera_poses
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("open3d required: pip install open3d")

        scene_dir = Path(scene_path)
        mesh_path = scene_dir / f"{scene_dir.name}_vh_clean_2.ply"
        label_path = scene_dir / f"{scene_dir.name}_vh_clean_2.labels.ply"

        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        # Load mesh
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()

        vertices = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.ones_like(vertices) * 0.5
        normals = np.asarray(mesh.vertex_normals)

        # Voxelize: bin vertices into cells
        cell_indices = np.floor(vertices / self.cell_size).astype(np.int32)
        unique_cells, inverse = np.unique(cell_indices, axis=0, return_inverse=True)

        # Compute per-cell averages
        n_cells = len(unique_cells)
        cell_albedo = np.zeros((n_cells, 3), dtype=np.float32)
        cell_normals = np.zeros((n_cells, 3), dtype=np.float32)
        cell_counts = np.zeros(n_cells, dtype=np.int32)

        for i in range(len(vertices)):
            cidx = inverse[i]
            cell_albedo[cidx] += colors[i]
            cell_normals[cidx] += normals[i]
            cell_counts[cidx] += 1

        mask = cell_counts > 0
        cell_albedo[mask] /= cell_counts[mask, None]
        norms = np.linalg.norm(cell_normals[mask], axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        cell_normals[mask] /= norms

        cell_positions = (unique_cells + 0.5) * self.cell_size

        return {
            "positions": cell_positions,
            "albedo": cell_albedo,
            "normals": cell_normals,
            "counts": cell_counts,
            "cell_indices": unique_cells,
        }

    def render_cell_buffers(
        self,
        scene_data: dict,
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
    ) -> np.ndarray:
        """Render voxelized scene as flat cell buffers from a camera pose.

        Returns (H, W, 8) float32 buffer matching the AI model's expected input format.
        """
        h, w = self.resolution, self.resolution
        buffers = np.zeros((h, w, 8), dtype=np.float32)

        positions = scene_data["positions"]
        albedo = scene_data["albedo"]
        normals = scene_data["normals"]

        # Project 3D cells onto 2D image plane
        # Transform to camera space
        R = camera_extrinsics[:3, :3]
        t = camera_extrinsics[:3, 3]
        cam_pos = positions @ R.T + t

        # Filter cells behind camera
        valid = cam_pos[:, 2] > 0.1
        cam_pos = cam_pos[valid]
        valid_albedo = albedo[valid]
        valid_normals = normals[valid]

        if len(cam_pos) == 0:
            return buffers

        # Project to pixel coordinates
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        px = (cam_pos[:, 0] * fx / cam_pos[:, 2] + cx).astype(np.int32)
        py = (cam_pos[:, 1] * fy / cam_pos[:, 2] + cy).astype(np.int32)
        depths = cam_pos[:, 2]

        # Rasterize with z-buffer
        z_buffer = np.full((h, w), np.inf, dtype=np.float32)

        for i in range(len(px)):
            x, y = px[i], py[i]
            if 0 <= x < w and 0 <= y < h:
                if depths[i] < z_buffer[y, x]:
                    z_buffer[y, x] = depths[i]
                    buffers[y, x, 0:3] = valid_albedo[i]       # albedo RGB
                    buffers[y, x, 3] = depths[i]                # depth
                    buffers[y, x, 4:7] = valid_normals[i]       # normals
                    buffers[y, x, 7] = 1.0                       # label placeholder

        # Normalize depth channel to [0,1]
        valid_depth = z_buffer < np.inf
        if valid_depth.any():
            d_min, d_max = z_buffer[valid_depth].min(), z_buffer[valid_depth].max()
            if d_max > d_min:
                buffers[:, :, 3] = np.where(valid_depth, (z_buffer - d_min) / (d_max - d_min), 0)

        return buffers


class TrainingDataGenerator:
    """Generates training pairs from ScanNet/Matterport3D."""

    def __init__(
        self,
        scannet_root: str = "data/scannet",
        matterport_root: str = "data/matterport3d",
        output_dir: str = "data/training_pairs",
        resolution: int = 512,
        cell_size: float = 0.05,
    ):
        self.scannet_root = Path(scannet_root)
        self.matterport_root = Path(matterport_root)
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.voxelizer = ScanNetVoxelizer(cell_size, resolution)

    def generate_scannet_pairs(self, max_scenes: int = 1500) -> int:
        """Generate pairs from ScanNet scenes.

        Returns number of pairs generated.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pair_count = 0

        scenes = sorted(self.scannet_root.glob("scene*"))[:max_scenes]
        print(f"[DataPipeline] Processing {len(scenes)} ScanNet scenes...")

        for scene_dir in scenes:
            try:
                scene_data = self.voxelizer.process_scene(str(scene_dir))

                # Load camera poses
                pose_dir = scene_dir / "pose"
                if not pose_dir.exists():
                    continue

                intrinsics = self._load_scannet_intrinsics(scene_dir)
                poses = sorted(pose_dir.glob("*.txt"))

                # Sample every 10th pose for diversity
                for pose_file in poses[::10]:
                    cam_idx = int(pose_file.stem)
                    extrinsics = np.loadtxt(str(pose_file)).reshape(4, 4)

                    # Skip invalid poses
                    if np.any(np.isinf(extrinsics)):
                        continue

                    # Render cell buffers
                    condition = self.voxelizer.render_cell_buffers(
                        scene_data, intrinsics, extrinsics
                    )

                    # Load corresponding RGB frame
                    rgb_path = scene_dir / "color" / f"{cam_idx}.jpg"
                    if not rgb_path.exists():
                        continue

                    from PIL import Image
                    target = np.array(
                        Image.open(str(rgb_path)).resize((self.resolution, self.resolution))
                    ).astype(np.float32) / 255.0

                    # Save pair
                    pair_id = f"{scene_dir.name}_{cam_idx:06d}"
                    np.savez_compressed(
                        str(self.output_dir / f"{pair_id}.npz"),
                        condition=condition,
                        target=target,
                    )
                    pair_count += 1

            except Exception as e:
                print(f"[DataPipeline] Error processing {scene_dir.name}: {e}")
                continue

        print(f"[DataPipeline] Generated {pair_count} ScanNet pairs")
        return pair_count

    def _load_scannet_intrinsics(self, scene_dir: Path) -> np.ndarray:
        """Load camera intrinsics from ScanNet scene."""
        intrinsic_path = scene_dir / "intrinsic" / "intrinsic_color.txt"
        if intrinsic_path.exists():
            return np.loadtxt(str(intrinsic_path)).reshape(4, 4)[:3, :3]
        # Default intrinsics for ScanNet
        return np.array([
            [577.87, 0, 319.5],
            [0, 577.87, 239.5],
            [0, 0, 1],
        ], dtype=np.float32)

    def create_augmented_pairs(self, augment_factor: int = 3) -> int:
        """Apply data augmentation to existing pairs.

        Augmentations:
          - Random color jitter on the cell buffers (not ground truth)
          - Random light direction change (rotate normal buffers)
          - Horizontal flip

        Returns number of augmented pairs created.
        """
        existing = list(self.output_dir.glob("*.npz"))
        aug_count = 0

        print(f"[DataPipeline] Augmenting {len(existing)} pairs × {augment_factor}...")

        for npz_path in existing:
            data = np.load(str(npz_path))
            condition = data["condition"]
            target = data["target"]

            for aug_idx in range(augment_factor):
                aug_cond = condition.copy()
                aug_target = target.copy()

                # Color jitter on albedo channels only
                if np.random.random() > 0.5:
                    jitter = np.random.uniform(0.8, 1.2, 3).astype(np.float32)
                    aug_cond[:, :, 0:3] = np.clip(aug_cond[:, :, 0:3] * jitter, 0, 1)

                # Random horizontal flip
                if np.random.random() > 0.5:
                    aug_cond = aug_cond[:, ::-1, :].copy()
                    aug_target = aug_target[:, ::-1, :].copy()
                    # Flip normal x component
                    aug_cond[:, :, 4] *= -1

                pair_id = f"{npz_path.stem}_aug{aug_idx}"
                np.savez_compressed(
                    str(self.output_dir / f"{pair_id}.npz"),
                    condition=aug_cond,
                    target=aug_target,
                )
                aug_count += 1

        print(f"[DataPipeline] Created {aug_count} augmented pairs")
        return aug_count

    def create_dataloader(self, batch_size: int = 16, shuffle: bool = True):
        """Create a PyTorch DataLoader from the generated pairs."""
        import torch
        from torch.utils.data import Dataset, DataLoader

        class PairDataset(Dataset):
            def __init__(self, data_dir):
                self.files = sorted(Path(data_dir).glob("*.npz"))

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                data = np.load(str(self.files[idx]))
                condition = torch.from_numpy(data["condition"]).permute(2, 0, 1).float()
                target = torch.from_numpy(data["target"]).permute(2, 0, 1).float()
                # Scale target to [-1, 1] for GAN training
                target = target * 2.0 - 1.0
                return condition, target

        dataset = PairDataset(self.output_dir)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
