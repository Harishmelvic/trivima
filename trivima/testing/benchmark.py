"""
Benchmark harness — automated testing across 20+ room photographs.

Metrics measured:
  - FPS (realtime mode): average over 1000 frames of continuous navigation
  - SSIM vs ground truth: structural similarity (target > 0.85)
  - LPIPS vs ground truth: perceptual distance (target < 0.15)
  - Temporal flicker: mean absolute pixel diff between consecutive frames (target < 3%)
  - Energy conservation: % cells violating before correction (target < 5%)
  - Mass conservation: total mass drift over 1000 frames (target < 0.1%)
  - Collision accuracy: % of known solid positions correctly identified (target > 98%)
  - Memory usage: peak GPU memory during navigation (target < 4GB)
"""

import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json


@dataclass
class SceneResult:
    """Benchmark results for one test scene."""
    scene_id: str
    image_path: str

    # Performance
    avg_fps: float = 0.0
    min_fps: float = 0.0
    p95_fps: float = 0.0
    avg_frame_ms: float = 0.0

    # Quality (vs ground truth)
    ssim: float = 0.0
    lpips: float = 0.0
    psnr: float = 0.0

    # Temporal
    temporal_flicker_pct: float = 0.0

    # Conservation
    energy_violation_pct: float = 0.0
    mass_drift_pct: float = 0.0
    shadow_violation_pct: float = 0.0

    # Collision
    collision_accuracy: float = 0.0

    # Memory
    peak_gpu_mb: float = 0.0

    # Cell grid
    total_cells: int = 0
    visible_cells_avg: int = 0

    # Pass/fail against targets
    @property
    def passes_must_have(self) -> bool:
        return (
            self.avg_fps >= 20.0 and
            self.collision_accuracy >= 0.98 and
            self.total_cells > 0
        )

    @property
    def passes_nice_to_have(self) -> bool:
        return (
            self.avg_fps >= 30.0 and
            self.ssim >= 0.85 and
            self.energy_violation_pct < 5.0
        )


@dataclass
class BenchmarkReport:
    """Aggregate benchmark results across all test scenes."""
    scenes: List[SceneResult] = field(default_factory=list)
    timestamp: str = ""

    @property
    def pass_rate(self) -> float:
        if not self.scenes:
            return 0.0
        passing = sum(1 for s in self.scenes if s.passes_must_have)
        return passing / len(self.scenes)

    @property
    def avg_fps(self) -> float:
        return np.mean([s.avg_fps for s in self.scenes]) if self.scenes else 0

    @property
    def avg_ssim(self) -> float:
        return np.mean([s.ssim for s in self.scenes]) if self.scenes else 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "num_scenes": len(self.scenes),
            "pass_rate": self.pass_rate,
            "avg_fps": self.avg_fps,
            "avg_ssim": self.avg_ssim,
            "scenes": [vars(s) for s in self.scenes],
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class CameraPath:
    """Predefined camera path for reproducible benchmarks."""

    def __init__(self, waypoints: List[np.ndarray], duration_seconds: float = 30.0):
        self.waypoints = waypoints
        self.duration = duration_seconds

    def sample(self, t: float) -> tuple:
        """Sample camera position and forward direction at time t in [0, duration].

        Returns (position, forward) as (3,) numpy arrays.
        """
        frac = np.clip(t / self.duration, 0, 1)
        idx_float = frac * (len(self.waypoints) - 1)
        idx = int(idx_float)
        blend = idx_float - idx

        if idx >= len(self.waypoints) - 1:
            pos = self.waypoints[-1]
            fwd = self.waypoints[-1] - self.waypoints[-2]
        else:
            pos = (1 - blend) * self.waypoints[idx] + blend * self.waypoints[idx + 1]
            fwd = self.waypoints[idx + 1] - self.waypoints[idx]

        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm > 1e-6:
            fwd = fwd / fwd_norm
        else:
            fwd = np.array([0, 0, -1], dtype=np.float32)

        return pos, fwd

    @staticmethod
    def circular(center: np.ndarray, radius: float = 3.0, height: float = 1.6,
                 num_points: int = 100) -> "CameraPath":
        """Create a circular camera path around a center point."""
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        waypoints = []
        for a in angles:
            x = center[0] + radius * np.cos(a)
            z = center[2] + radius * np.sin(a)
            waypoints.append(np.array([x, height, z], dtype=np.float32))
        return CameraPath(waypoints)


class PipelineBenchmark:
    """Main benchmark runner."""

    def __init__(
        self,
        test_images_dir: str = "data/sample_images",
        output_dir: str = "output/benchmark",
        num_frames: int = 1000,
    ):
        self.test_images_dir = Path(test_images_dir)
        self.output_dir = Path(output_dir)
        self.num_frames = num_frames

    def run_fps_test(self, grid, texturing_engine, camera_path: CameraPath) -> dict:
        """Measure frame rate along a camera path.

        Returns dict with fps stats.
        """
        frame_times = []
        dt = camera_path.duration / self.num_frames

        for i in range(self.num_frames):
            t = i * dt
            pos, fwd = camera_path.sample(t)
            up = np.array([0, 1, 0], dtype=np.float32)

            t0 = time.perf_counter()
            texturing_engine.process_frame(grid, pos, fwd, up, dt=dt)
            elapsed = time.perf_counter() - t0
            frame_times.append(elapsed)

        frame_times = np.array(frame_times)
        fps = 1.0 / frame_times

        return {
            "avg_fps": float(np.mean(fps)),
            "min_fps": float(np.min(fps)),
            "p5_fps": float(np.percentile(fps, 5)),
            "p95_fps": float(np.percentile(fps, 95)),
            "avg_frame_ms": float(np.mean(frame_times) * 1000),
        }

    def run_quality_test(
        self,
        grid,
        texturing_engine,
        ground_truth_images: List[np.ndarray],
        camera_poses: List[tuple],
    ) -> dict:
        """Compare AI-textured output against ground truth photographs.

        Returns dict with quality metrics.
        """
        ssim_scores = []
        lpips_scores = []
        psnr_scores = []

        for gt_image, (pos, fwd) in zip(ground_truth_images, camera_poses):
            up = np.array([0, 1, 0], dtype=np.float32)
            output = texturing_engine.process_frame(grid, pos, fwd, up)

            if output is None:
                continue

            # Resize output to match GT if needed
            if output.shape[:2] != gt_image.shape[:2]:
                from PIL import Image
                output_pil = Image.fromarray((output * 255).astype(np.uint8))
                output_pil = output_pil.resize((gt_image.shape[1], gt_image.shape[0]))
                output = np.array(output_pil).astype(np.float32) / 255.0

            ssim_scores.append(self._compute_ssim(output, gt_image))
            psnr_scores.append(self._compute_psnr(output, gt_image))

            # LPIPS requires torch
            try:
                lpips_scores.append(self._compute_lpips(output, gt_image))
            except ImportError:
                pass

        return {
            "ssim": float(np.mean(ssim_scores)) if ssim_scores else 0,
            "psnr": float(np.mean(psnr_scores)) if psnr_scores else 0,
            "lpips": float(np.mean(lpips_scores)) if lpips_scores else 0,
        }

    def run_temporal_test(self, grid, texturing_engine, camera_path: CameraPath,
                          num_frames: int = 100) -> dict:
        """Measure temporal flicker along a slow camera path."""
        prev_frame = None
        diffs = []
        dt = camera_path.duration / num_frames

        for i in range(num_frames):
            t = i * dt
            pos, fwd = camera_path.sample(t)
            up = np.array([0, 1, 0], dtype=np.float32)

            output = texturing_engine.process_frame(grid, pos, fwd, up, dt=dt)
            if output is None:
                continue

            if prev_frame is not None:
                # Mean absolute pixel difference
                diff = np.abs(output - prev_frame).mean()
                diffs.append(diff)

            prev_frame = output.copy()

        return {
            "temporal_flicker_pct": float(np.mean(diffs) * 100) if diffs else 0,
            "max_flicker_pct": float(np.max(diffs) * 100) if diffs else 0,
        }

    def run_conservation_test(self, grid, validator, texturing_engine,
                              camera_path: CameraPath, num_frames: int = 1000) -> dict:
        """Track conservation metrics over time."""
        energy_violations = []
        mass_drifts = []
        dt = camera_path.duration / num_frames

        for i in range(num_frames):
            t = i * dt
            pos, fwd = camera_path.sample(t)
            up = np.array([0, 1, 0], dtype=np.float32)

            texturing_engine.process_frame(grid, pos, fwd, up, dt=dt)

            # Run validation
            visible = list(range(min(grid.size(), 1000)))  # sample
            report = validator.validate_frame(grid, visible)
            validator.apply_corrections(grid)

            energy_violations.append(report.energy_pct_violating)
            mass_drifts.append(abs(report.mass_violation))

        return {
            "avg_energy_violation_pct": float(np.mean(energy_violations)),
            "max_energy_violation_pct": float(np.max(energy_violations)),
            "final_mass_drift_pct": float(mass_drifts[-1] * 100) if mass_drifts else 0,
            "max_mass_drift_pct": float(np.max(mass_drifts) * 100) if mass_drifts else 0,
        }

    def run_collision_test(self, grid, known_solid_positions: List[np.ndarray],
                           known_empty_positions: List[np.ndarray]) -> dict:
        """Test collision detection against known geometry."""
        correct = 0
        total = 0

        for pos in known_solid_positions:
            idx = grid.find_at_position(pos[0], pos[1], pos[2])
            if idx is not None and idx >= 0:
                geo = grid.get_geo(idx)
                if geo.is_solid():
                    correct += 1
            total += 1

        for pos in known_empty_positions:
            idx = grid.find_at_position(pos[0], pos[1], pos[2])
            if idx is None or idx < 0 or grid.get_geo(idx).is_empty():
                correct += 1
            total += 1

        return {
            "collision_accuracy": correct / max(total, 1),
            "total_tested": total,
        }

    # --- Metric computation ---

    @staticmethod
    def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Structural Similarity Index."""
        try:
            from skimage.metrics import structural_similarity
            return structural_similarity(img1, img2, channel_axis=2, data_range=1.0)
        except ImportError:
            # Fallback: simple MSE-based approximation
            mse = np.mean((img1 - img2) ** 2)
            return max(0, 1.0 - mse * 10)  # rough approximation

    @staticmethod
    def _compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        mse = np.mean((img1 - img2) ** 2)
        if mse < 1e-10:
            return 100.0
        return float(10 * np.log10(1.0 / mse))

    @staticmethod
    def _compute_lpips(img1: np.ndarray, img2: np.ndarray) -> float:
        """Perceptual distance using LPIPS."""
        import torch
        import lpips

        loss_fn = lpips.LPIPS(net='alex')
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            d = loss_fn(t1, t2)
        return float(d.item())

    def generate_report(self, results: List[SceneResult]) -> BenchmarkReport:
        """Generate aggregate report from individual scene results."""
        from datetime import datetime

        report = BenchmarkReport(
            scenes=results,
            timestamp=datetime.now().isoformat(),
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        report.save(str(self.output_dir / "benchmark_report.json"))

        return report
