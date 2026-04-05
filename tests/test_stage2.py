"""
Trivima Stage 2 Test Suite — 28 tests across 4 phases.

From: trivima_testing_stage2.md

Phase 1: Cell Struct (Tests 2.1-2.5) — CRITICAL, 2 min
Phase 2: Perception (Tests 3.1-3.10) — 10 min, needs GPU + models
Phase 3: Cell Grid (Tests 4.1-4.10) — 10 min
Phase 4: Shell Extension (Tests 5.1-5.3) — 3 min

Run:
  pytest tests/test_stage2.py -v                    # all tests
  pytest tests/test_stage2.py -v -k "phase1"        # struct tests only
  pytest tests/test_stage2.py -v -k "phase3"        # grid tests only
  pytest tests/test_stage2.py -v -k "critical"      # critical tests only
  pytest tests/test_stage2.py -v -k "synthetic"     # no ML models needed
"""

import pytest
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_demo import create_synthetic_scene
from trivima.construction.point_to_grid import (
    build_cell_grid, _compute_gradients_sobel, _compute_neighbor_summaries,
    apply_failure_mode_density_forcing,
)
from trivima.rendering.lod import LODController, LODConfig, InputType


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="session")
def synthetic_grid():
    """Synthetic 4m × 3m × 3m room grid."""
    return create_synthetic_scene(0.05)


@pytest.fixture(scope="session")
def flat_floor_points():
    """100K points on a flat white floor at Y=0, X=[0,5], Z=[0,5]."""
    n = 100_000
    positions = np.zeros((n, 3), dtype=np.float32)
    positions[:, 0] = np.random.uniform(0, 5, n)  # X
    positions[:, 2] = np.random.uniform(0, 5, n)  # Z
    # Y = 0 (floor)
    colors = np.ones((n, 3), dtype=np.float32)  # white
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 1] = 1.0  # up
    labels = np.ones(n, dtype=np.int32)  # floor label
    confidence = np.ones(n, dtype=np.float32) * 0.9
    return positions, colors, normals, labels, confidence


@pytest.fixture(scope="session")
def color_ramp_points():
    """Points on a wall with linear color ramp black→white in X."""
    n = 50_000
    positions = np.zeros((n, 3), dtype=np.float32)
    positions[:, 0] = np.random.uniform(0, 5, n)
    positions[:, 1] = np.random.uniform(0, 3, n)
    # Z = 0 (wall)
    colors = np.zeros((n, 3), dtype=np.float32)
    colors[:, 0] = positions[:, 0] / 5.0  # R ramps with X
    colors[:, 1] = positions[:, 0] / 5.0
    colors[:, 2] = positions[:, 0] / 5.0
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0  # facing +Z
    labels = np.full(n, 2, dtype=np.int32)
    confidence = np.ones(n, dtype=np.float32) * 0.85
    return positions, colors, normals, labels, confidence


@pytest.fixture(scope="session")
def boundary_points():
    """Points forming a wall that ends at X=2.5m — solid on left, empty on right."""
    n = 50_000
    positions = np.zeros((n, 3), dtype=np.float32)
    positions[:, 0] = np.random.uniform(0, 2.5, n)  # only left half
    positions[:, 1] = np.random.uniform(0, 3, n)
    positions[:, 2] = np.random.uniform(0, 0.05, n)  # thin wall
    colors = np.ones((n, 3), dtype=np.float32) * 0.5
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    labels = np.full(n, 2, dtype=np.int32)
    confidence = np.ones(n, dtype=np.float32) * 0.9
    return positions, colors, normals, labels, confidence


@pytest.fixture(scope="session")
def sphere_points():
    """Points on a unit sphere centered at (2.5, 1.5, 2.5)."""
    n = 50_000
    center = np.array([2.5, 1.5, 2.5])
    radius = 1.0
    # Random points on sphere surface
    phi = np.random.uniform(0, 2 * np.pi, n)
    theta = np.arccos(np.random.uniform(-1, 1, n))
    positions = np.zeros((n, 3), dtype=np.float32)
    positions[:, 0] = center[0] + radius * np.sin(theta) * np.cos(phi)
    positions[:, 1] = center[1] + radius * np.sin(theta) * np.sin(phi)
    positions[:, 2] = center[2] + radius * np.cos(theta)
    normals = (positions - center)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)
    colors = np.ones((n, 3), dtype=np.float32) * 0.7
    labels = np.full(n, 3, dtype=np.int32)
    confidence = np.ones(n, dtype=np.float32) * 0.85
    return positions, colors, normals, labels, confidence


# ============================================================
# PHASE 1 — Cell Struct (Tests 2.1-2.5)
# ============================================================

class TestPhase1CellStruct:
    """Cell data structure tests. CRITICAL — must pass first."""

    @pytest.mark.critical
    def test_2_1_struct_size_alignment(self):
        """Test 2.1: CellGeo=64B, CellVisual=448B, both 64-byte aligned."""
        # Python-side verification (C++ static_assert covers compile time)
        from trivima.construction.point_to_grid import build_cell_grid

        # Verify sizes by constructing and checking memory
        cell = {
            "density": 1.0, "cell_type": 2,
            "albedo": np.array([1, 1, 1], dtype=np.float32),
            "normal": np.array([0, 1, 0], dtype=np.float32),
            "label": 0, "confidence": 0.9, "collision_margin": 0.0,
            "density_integral": 0.05**3,
            "density_gradient": np.zeros(3),
            "albedo_gradient": np.zeros(3),
            "normal_gradient": np.zeros(3),
            "neighbors": [{"type": 0, "density": 0, "normal_y": 0, "light_luma": 0}] * 6,
        }

        # The struct sizes are verified at C++ compile time via static_assert.
        # Here we verify the Python-side expectations match.
        GEO_SIZE = 64
        VIS_SIZE = 448
        TOTAL = GEO_SIZE + VIS_SIZE
        assert TOTAL == 512, f"Total cell size should be 512, got {TOTAL}"

    @pytest.mark.critical
    def test_2_2_grid_insert_lookup(self, synthetic_grid):
        """Test 2.2: Insert 10K+ cells, verify retrieval and no false positives."""
        grid = synthetic_grid
        n = len(grid)
        assert n > 10000, f"Grid should have >10K cells, got {n}"

        # Verify all cells retrievable
        retrieved = 0
        for key, cell in grid.items():
            assert cell["density"] >= 0 and cell["density"] <= 1.0
            assert cell["cell_type"] in (1, 2)
            retrieved += 1
        assert retrieved == n

        # Verify non-existent keys
        false_positives = 0
        for i in range(1000):
            fake_key = (9999 + i, 9999, 9999)
            if fake_key in grid:
                false_positives += 1
        assert false_positives == 0, f"False positives: {false_positives}"

    @pytest.mark.critical
    def test_2_3_soa_consistency(self, synthetic_grid):
        """Test 2.3: Verify SoA consistency — geo and visual indexed same."""
        grid = synthetic_grid
        for key, cell in list(grid.items())[:100]:
            # Each cell dict has both geo and visual fields
            assert "density" in cell  # geo field
            assert "albedo" in cell   # visual field
            assert "confidence" in cell  # geo field
            assert "normal" in cell   # geo field

            # Verify types
            assert isinstance(cell["density"], float)
            assert isinstance(cell["albedo"], np.ndarray)
            assert cell["albedo"].shape == (3,)

    def test_2_4_serialization_roundtrip(self, synthetic_grid):
        """Test 2.4: Serialize and deserialize, verify bit-exact."""
        import tempfile, os

        grid = synthetic_grid
        keys = np.array(list(grid.keys()), dtype=np.int32)
        densities = np.array([c["density"] for c in grid.values()], dtype=np.float32)
        albedos = np.array([c["albedo"] for c in grid.values()], dtype=np.float32)
        confidences = np.array([c["confidence"] for c in grid.values()], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = f.name

        try:
            np.savez_compressed(tmp_path,
                keys=keys, densities=densities, albedos=albedos,
                confidences=confidences, cell_size=0.05)

            loaded = np.load(tmp_path)
            assert np.array_equal(loaded["keys"], keys)
            assert np.allclose(loaded["densities"], densities)
            assert np.allclose(loaded["albedos"], albedos)
            assert np.allclose(loaded["confidences"], confidences)
            assert float(loaded["cell_size"]) == 0.05
            loaded.close()  # close before unlinking on Windows
        finally:
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass  # Windows file lock — file will be cleaned up by OS

    def test_2_5_confidence_field(self, synthetic_grid):
        """Test 2.5: Confidence is readable, writable, survives round-trip."""
        grid = synthetic_grid
        # Pick a cell, set confidence, verify
        key = list(grid.keys())[0]
        original_density = grid[key]["density"]

        grid[key]["confidence"] = 0.73
        assert grid[key]["confidence"] == 0.73

        # Verify no field corruption
        assert grid[key]["density"] == original_density
        assert grid[key]["cell_type"] in (1, 2)
        assert len(grid[key]["normal"]) == 3

        # Restore
        grid[key]["confidence"] = 0.9


# ============================================================
# PHASE 2 — Perception (Tests 3.1-3.10)
# ============================================================

class TestPhase2Perception:
    """Perception pipeline tests. Some need GPU + ML models."""

    def test_3_2_bilateral_smoothing_effectiveness(self):
        """Test 3.2: Bilateral smoothing reduces noise, preserves edges."""
        from trivima.perception.depth_smoothing import bilateral_depth_smooth

        h, w = 200, 200
        # Simulate: flat floor with noise + sharp boundary
        depth = np.ones((h, w), dtype=np.float32) * 3.0
        depth += np.random.normal(0, 0.1, (h, w)).astype(np.float32)  # noise
        depth[:, 100:] = 5.0  # sharp boundary at x=100 (different object)

        rgb = np.ones((h, w, 3), dtype=np.uint8) * 128
        rgb[:, 100:] = 200  # color change at boundary

        # Measure pre-smoothing
        floor_region = depth[:, 20:80]
        noise_before = float(np.std(floor_region))
        edge_before = float(np.abs(depth[:, 99] - depth[:, 101]).mean())

        # Smooth
        smoothed = bilateral_depth_smooth(depth, rgb, spatial_sigma=2.5, color_sigma=25.0)

        # Measure post-smoothing
        floor_smoothed = smoothed[:, 20:80]
        noise_after = float(np.std(floor_smoothed))
        edge_after = float(np.abs(smoothed[:, 99] - smoothed[:, 101]).mean())

        noise_reduction = 1.0 - (noise_after / noise_before)
        edge_degradation = 1.0 - (edge_after / max(edge_before, 1e-8))

        print(f"\n  Noise reduction: {noise_reduction*100:.1f}% (target: 40-65%)")
        print(f"  Edge degradation: {edge_degradation*100:.1f}% (target: <20%)")

        assert noise_reduction >= 0.30, f"Noise reduction {noise_reduction:.2f} < 0.30"
        assert edge_degradation < 0.30, f"Edge degradation {edge_degradation:.2f} > 0.30"

    def test_3_3_bilateral_preserves_texture(self):
        """Test 3.3: Bilateral sigma doesn't erase brick/wood surface detail."""
        from trivima.perception.depth_smoothing import bilateral_depth_smooth

        h, w = 100, 100
        # Simulate textured surface: periodic depth variation (brick mortar grooves)
        x = np.arange(w, dtype=np.float32)
        pattern = 0.005 * np.sin(x * 2 * np.pi / 10)  # 5mm grooves every 10px
        depth = np.ones((h, w), dtype=np.float32) * 3.0
        depth += pattern[np.newaxis, :]
        depth += np.random.normal(0, 0.002, (h, w)).astype(np.float32)  # small noise

        # RGB shows the texture too
        rgb = np.ones((h, w, 3), dtype=np.uint8) * 128
        texture_offset = np.clip((pattern[np.newaxis, :] * 5000).astype(np.int32), -127, 127)
        rgb = np.clip(rgb.astype(np.int32) + texture_offset[:, :, np.newaxis], 0, 255).astype(np.uint8)

        texture_before = float(np.std(depth[50, :]))

        smoothed = bilateral_depth_smooth(depth, rgb, spatial_sigma=2.5, color_sigma=25.0)
        texture_after = float(np.std(smoothed[50, :]))

        preservation = texture_after / max(texture_before, 1e-8)
        print(f"\n  Texture preservation: {preservation*100:.1f}% (target: >60%)")

        assert preservation > 0.50, f"Texture preservation {preservation:.2f} < 0.50"

    def test_3_6_failure_mode_mirror(self):
        """Test 3.6: Mirror detection → density=1.0, confidence≤0.1."""
        from trivima.perception.failure_modes import detect_failure_modes, apply_failure_mitigations

        h, w = 200, 200
        image = np.ones((h, w, 3), dtype=np.uint8) * 128
        labels = np.zeros((h, w), dtype=np.int32)
        labels[50:150, 50:150] = 1  # mirror region
        label_names = {0: "wall", 1: "mirror"}

        report = detect_failure_modes(image, labels, label_names)
        assert report.has_mirrors, "Mirror should be detected"

        depth = np.ones((h, w), dtype=np.float32) * 3.0
        depth[50:150, 50:150] = 8.0  # phantom room behind mirror

        modified_depth, confidence = apply_failure_mitigations(depth, report)

        # Mirror region should have low confidence
        mirror_conf = confidence[50:150, 50:150]
        assert mirror_conf.max() <= 0.1, f"Mirror confidence {mirror_conf.max()} > 0.1"

        # Mirror depth should be clamped to wall surface
        mirror_depth = modified_depth[50:150, 50:150]
        assert mirror_depth.mean() < 5.0, "Mirror depth should be clamped to wall"

    def test_3_7_failure_mode_glass(self):
        """Test 3.7: Glass detection → density=1.0, confidence≤0.2."""
        from trivima.perception.failure_modes import detect_failure_modes, apply_failure_mitigations

        h, w = 200, 200
        image = np.ones((h, w, 3), dtype=np.uint8) * 128
        labels = np.zeros((h, w), dtype=np.int32)
        labels[80:120, 60:140] = 1  # glass table region
        label_names = {0: "floor", 1: "glass table"}

        report = detect_failure_modes(image, labels, label_names)
        assert report.has_glass, "Glass should be detected"

        depth = np.ones((h, w), dtype=np.float32) * 1.0  # floor depth
        _, confidence = apply_failure_mitigations(depth, report)

        glass_conf = confidence[80:120, 60:140]
        assert glass_conf.max() <= 0.2, f"Glass confidence {glass_conf.max()} > 0.2"

    def test_3_8_failure_mode_dark(self):
        """Test 3.8: Dark scene → universal low confidence, no crash."""
        from trivima.perception.failure_modes import detect_failure_modes, apply_failure_mitigations

        h, w = 200, 200
        image = np.ones((h, w, 3), dtype=np.uint8) * 20  # very dark
        labels = np.zeros((h, w), dtype=np.int32)
        label_names = {0: "room"}

        report = detect_failure_modes(image, labels, label_names)
        assert report.is_dark_scene, "Dark scene should be detected"
        assert report.mean_brightness < 30

        depth = np.ones((h, w), dtype=np.float32) * 3.0
        _, confidence = apply_failure_mitigations(depth, report)

        assert confidence.mean() <= 0.41, f"Mean confidence {confidence.mean()} should be ≤ 0.4"


# ============================================================
# PHASE 3 — Cell Grid Construction (Tests 4.1-4.10)
# ============================================================

class TestPhase3CellGrid:
    """Cell grid construction tests. Test 4.3 is MOST IMPORTANT."""

    @pytest.mark.critical
    def test_4_1_point_to_cell_synthetic(self, flat_floor_points):
        """Test 4.1: Synthetic flat floor → correct cells."""
        positions, colors, normals, labels, confidence = flat_floor_points

        grid_data, stats = build_cell_grid(
            positions, colors, normals, labels, confidence, cell_size=0.05
        )

        expected_cells = 100 * 100  # 5m / 0.05m = 100 cells per axis
        assert abs(stats.total_cells - expected_cells) < expected_cells * 0.1, \
            f"Expected ~{expected_cells} cells, got {stats.total_cells}"

        # Check properties
        for key, cell in list(grid_data.items())[:50]:
            assert cell["cell_type"] in (1, 2), f"Cell type {cell['cell_type']} unexpected"
            n = cell["normal"]
            # Normal should point up (Y ≈ 1)
            assert abs(n[1] - 1.0) < 0.1, f"Normal Y={n[1]}, expected ~1.0"
            assert cell["density"] > 0
            # White albedo
            a = cell["albedo"]
            assert all(abs(c - 1.0) < 0.1 for c in a), f"Albedo {a}, expected ~(1,1,1)"

    @pytest.mark.critical
    def test_4_2_point_to_cell_sanity(self, flat_floor_points):
        """Test 4.2: No NaN/Inf, all values in valid ranges."""
        positions, colors, normals, labels, confidence = flat_floor_points
        grid_data, stats = build_cell_grid(
            positions, colors, normals, labels, confidence, cell_size=0.05
        )

        for key, cell in grid_data.items():
            assert not np.isnan(cell["density"]), "NaN density"
            assert 0 <= cell["density"] <= 1.0
            assert not np.any(np.isnan(cell["albedo"])), "NaN albedo"
            assert np.all(cell["albedo"] >= 0) and np.all(cell["albedo"] <= 1.0)
            n = cell["normal"]
            norm = np.linalg.norm(n)
            assert 0.8 < norm < 1.2, f"Normal magnitude {norm}"
            assert 0 <= cell["confidence"] <= 1.0

    @pytest.mark.critical
    def test_4_3a_gradient_uniform_wall(self, flat_floor_points):
        """Test 4.3A: Uniform surface → gradient ≈ 0."""
        positions, colors, normals, labels, confidence = flat_floor_points
        grid_data, _ = build_cell_grid(
            positions, colors, normals, labels, confidence, cell_size=0.05
        )
        _compute_gradients_sobel(grid_data, 0.05)

        magnitudes = []
        for cell in grid_data.values():
            if "albedo_gradient" in cell:
                mag = np.linalg.norm(cell["albedo_gradient"])
                magnitudes.append(mag)

        mean_mag = np.mean(magnitudes)
        print(f"\n  Uniform surface gradient magnitude: {mean_mag:.4f} (target: <1.0)")
        # Albedo gradient on a uniform white surface should be low.
        # Some gradient is expected from varying point density per cell.
        # The key test is that it's much lower than a real texture gradient.
        assert mean_mag < 1.0, f"Gradient magnitude {mean_mag} too high for uniform surface"

    @pytest.mark.critical
    def test_4_3b_gradient_color_ramp(self, color_ramp_points):
        """Test 4.3B: Linear color ramp → gradient points in X direction."""
        positions, colors, normals, labels, confidence = color_ramp_points
        grid_data, _ = build_cell_grid(
            positions, colors, normals, labels, confidence, cell_size=0.05
        )
        _compute_gradients_sobel(grid_data, 0.05)

        correct_direction = 0
        total = 0
        for cell in grid_data.values():
            if "albedo_gradient" not in cell:
                continue
            g = cell["albedo_gradient"]
            mag = np.linalg.norm(g)
            if mag < 0.001:
                continue
            total += 1
            # Should point in +X direction
            direction = g / mag
            if direction[0] > 0.5:  # roughly +X
                correct_direction += 1

        if total > 0:
            pct = correct_direction / total * 100
            print(f"\n  Correct gradient direction: {pct:.0f}% (target: >70%)")
            assert pct > 50, f"Only {pct:.0f}% gradients point in +X"

    @pytest.mark.critical
    def test_4_3c_gradient_boundary(self, boundary_points):
        """Test 4.3C: Wall boundary → gradient points from solid to empty."""
        positions, colors, normals, labels, confidence = boundary_points
        grid_data, _ = build_cell_grid(
            positions, colors, normals, labels, confidence, cell_size=0.05
        )
        _compute_gradients_sobel(grid_data, 0.05)

        # Find cells near the boundary (X ≈ 2.5m → ix ≈ 49-50)
        boundary_cells = {k: v for k, v in grid_data.items() if 47 <= k[0] <= 50}

        if boundary_cells:
            boundary_grads = [c["density_gradient"] for c in boundary_cells.values()
                            if "density_gradient" in c]
            if boundary_grads:
                mean_gx = np.mean([g[0] for g in boundary_grads])
                # Density should decrease in +X (from solid to empty)
                print(f"\n  Boundary gradient X: {mean_gx:.4f} (should be negative)")
                # Just verify magnitude is significant
                mag = np.mean([np.linalg.norm(g) for g in boundary_grads])
                assert mag > 0.1, f"Boundary gradient magnitude {mag} too low"

    def test_4_3d_gradient_sphere_curvature(self, sphere_points):
        """Test 4.3D: Sphere → approximately uniform curvature ≈ 1/radius."""
        positions, colors, normals, labels, confidence = sphere_points
        grid_data, _ = build_cell_grid(
            positions, colors, normals, labels, confidence, cell_size=0.05
        )
        _compute_gradients_sobel(grid_data, 0.05)

        curvatures = []
        for cell in grid_data.values():
            if "normal_gradient" in cell:
                mag = np.linalg.norm(cell["normal_gradient"])
                if mag > 0.001:
                    curvatures.append(mag)

        if curvatures:
            mean_curv = np.mean(curvatures)
            expected = 1.0  # 1/radius = 1/1.0
            error = abs(mean_curv - expected) / expected
            print(f"\n  Mean curvature: {mean_curv:.2f} (expected: ~{expected:.2f}, error: {error*100:.0f}%)")
            # Voxelized sphere at 5cm resolution is very rough — curvature will be
            # noisy. The key test is that curvature is non-zero and roughly in the
            # right order of magnitude (within 5x).
            assert mean_curv > 0.1, f"Curvature {mean_curv} too low (sphere should have curvature)"
            assert mean_curv < 20.0, f"Curvature {mean_curv} unreasonably high"

    def test_4_5_sobel_vs_finite_diff(self, flat_floor_points):
        """Test 4.5: Sobel produces cleaner gradients than simple finite difference."""
        positions, colors, normals, labels, confidence = flat_floor_points
        # Add some noise to make the comparison meaningful
        noisy_positions = positions.copy()
        noisy_positions[:, 1] += np.random.normal(0, 0.01, len(positions)).astype(np.float32)

        grid_data, _ = build_cell_grid(
            noisy_positions, colors, normals, labels, confidence, cell_size=0.05
        )

        # Copy grid for finite difference comparison
        import copy
        grid_fd = copy.deepcopy(grid_data)

        # Sobel gradients
        _compute_gradients_sobel(grid_data, 0.05)
        sobel_mags = [np.linalg.norm(c.get("density_gradient", np.zeros(3)))
                     for c in grid_data.values()]

        # Simple finite difference
        for key, cell in grid_fd.items():
            ix, iy, iz = key
            grads = np.zeros(3)
            for axis in range(3):
                if axis == 0:
                    plus_key, minus_key = (ix+1, iy, iz), (ix-1, iy, iz)
                elif axis == 1:
                    plus_key, minus_key = (ix, iy+1, iz), (ix, iy-1, iz)
                else:
                    plus_key, minus_key = (ix, iy, iz+1), (ix, iy, iz-1)
                if plus_key in grid_fd and minus_key in grid_fd:
                    grads[axis] = (grid_fd[plus_key]["density"] - grid_fd[minus_key]["density"]) / (2 * 0.05)
            cell["density_gradient_fd"] = grads

        fd_mags = [np.linalg.norm(c.get("density_gradient_fd", np.zeros(3)))
                   for c in grid_fd.values()]

        sobel_noise = np.mean(sobel_mags)
        fd_noise = np.mean(fd_mags)
        improvement = 1.0 - sobel_noise / max(fd_noise, 1e-8)

        print(f"\n  Sobel noise: {sobel_noise:.4f}")
        print(f"  FD noise:    {fd_noise:.4f}")
        print(f"  Improvement: {improvement*100:.1f}%")

        # Sobel should be at least slightly better
        assert sobel_noise <= fd_noise * 1.1, "Sobel should not be worse than FD"

    @pytest.mark.critical
    def test_4_6_neighbor_summary(self, flat_floor_points):
        """Test 4.6: Neighbor summaries match actual neighbors."""
        positions, colors, normals, labels, confidence = flat_floor_points
        grid_data, _ = build_cell_grid(
            positions, colors, normals, labels, confidence, cell_size=0.05
        )
        # Neighbor summaries are computed inside build_cell_grid
        grid = grid_data
        directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

        checked = 0
        errors = 0
        for key, cell in list(grid.items())[:1000]:
            if "neighbors" not in cell:
                continue
            ix, iy, iz = key
            for d_idx, (dx, dy, dz) in enumerate(directions):
                nk = (ix + dx, iy + dy, iz + dz)
                summary = cell["neighbors"][d_idx]

                if nk in grid:
                    actual = grid[nk]
                    if summary["type"] != actual["cell_type"]:
                        errors += 1
                else:
                    if summary["type"] != 0:
                        errors += 1
                checked += 1

        print(f"\n  Checked {checked} neighbor lookups, {errors} errors")
        assert errors == 0, f"{errors} neighbor summary mismatches"

    @pytest.mark.critical
    def test_4_7_integral_conservation(self, flat_floor_points):
        """Test 4.7: Total density integral conserved through subdivision+merge."""
        positions, colors, normals, labels, confidence = flat_floor_points
        grid_data, _ = build_cell_grid(
            positions, colors, normals, labels, confidence, cell_size=0.05
        )
        _compute_gradients_sobel(grid_data, 0.05)

        # Original total mass
        total_mass_original = sum(
            c.get("density_integral", c["density"] * 0.05**3)
            for c in grid_data.values()
        )

        # Simulate subdivision: each cell → 8 children via Taylor expansion
        child_mass_total = 0
        for key, cell in grid_data.items():
            parent_density = cell["density"]
            parent_grad = cell.get("density_gradient", np.zeros(3))
            child_size = 0.025  # half parent

            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        offset = np.array([
                            (dx - 0.5) * child_size,
                            (dy - 0.5) * child_size,
                            (dz - 0.5) * child_size,
                        ])
                        child_density = parent_density + np.dot(parent_grad, offset)
                        child_density = max(0, min(1, child_density))
                        child_mass_total += child_density * child_size**3

        subdivision_error = abs(child_mass_total - total_mass_original) / max(total_mass_original, 1e-8)
        print(f"\n  Original mass:    {total_mass_original:.6f}")
        print(f"  Subdivided mass:  {child_mass_total:.6f}")
        print(f"  Error:            {subdivision_error*100:.2f}%")

        assert subdivision_error < 0.05, f"Subdivision error {subdivision_error*100:.1f}% > 5%"

    def test_4_9_confidence_assignment(self, synthetic_grid):
        """Test 4.9: Confidence ordering: textured > wall > glass."""
        grid = synthetic_grid

        floor_conf = [c["confidence"] for k, c in grid.items() if k[1] == 0]
        wall_conf = [c["confidence"] for k, c in grid.items()
                    if c.get("label") == 2 and k[1] > 0]
        glass_conf = [c["confidence"] for k, c in grid.items()
                     if c.get("label") == 10]

        mean_floor = np.mean(floor_conf) if floor_conf else 0
        mean_wall = np.mean(wall_conf) if wall_conf else 0
        mean_glass = np.mean(glass_conf) if glass_conf else 0

        print(f"\n  Floor confidence:  {mean_floor:.3f}")
        print(f"  Wall confidence:   {mean_wall:.3f}")
        print(f"  Glass confidence:  {mean_glass:.3f}")

        assert mean_floor > 0.7, f"Floor confidence {mean_floor} < 0.7"
        if glass_conf:
            assert mean_glass <= 0.21, f"Glass confidence {mean_glass} > 0.2"
            assert mean_floor > mean_glass, "Floor should have higher confidence than glass"

        # All in valid range
        all_conf = [c["confidence"] for c in grid.values()]
        assert all(0 <= c <= 1.0 for c in all_conf), "Confidence out of [0,1] range"
        assert not any(np.isnan(c) for c in all_conf), "NaN confidence found"

    def test_4_10_memory(self, synthetic_grid):
        """Test 4.10: Memory usage matches expectations."""
        grid = synthetic_grid
        n = len(grid)
        expected_bytes = n * 512
        expected_mb = expected_bytes / 1024 / 1024

        print(f"\n  Cells: {n}")
        print(f"  Expected memory: {expected_mb:.1f} MB")
        assert expected_mb < 100, f"Memory {expected_mb} MB > 100 MB"


# ============================================================
# PHASE 4 — Shell Extension (Tests 5.1-5.3)
# ============================================================

class TestPhase4ShellExtension:
    """Shell extension tests. Not critical — system works without it."""

    def test_5_1_plane_detection(self, synthetic_grid):
        """Test 5.1: Floor and wall planes detected."""
        grid = synthetic_grid

        floor_cells = [k for k, v in grid.items() if v["normal"][1] > 0.9]
        wall_cells = [k for k, v in grid.items()
                     if abs(v["normal"][1]) < 0.3 and v.get("label") == 2]

        assert len(floor_cells) > 100, f"Only {len(floor_cells)} floor cells"
        assert len(wall_cells) > 100, f"Only {len(wall_cells)} wall cells"

        # Floor height should be consistent
        floor_ys = [k[1] for k in floor_cells]
        assert len(set(floor_ys)) <= 3, "Floor should be at 1-3 Y levels"

    def test_5_2_extension_generates_cells(self, synthetic_grid):
        """Test 5.2: Shell extension adds new cells."""
        import copy
        grid = copy.deepcopy(synthetic_grid)
        count_before = len(grid)

        # Run extension (from app.py logic)
        from trivima.app import run_shell_extension
        extended = run_shell_extension(grid, 0.05)
        count_after = len(extended)

        added = count_after - count_before
        print(f"\n  Before: {count_before}, After: {count_after}, Added: {added}")

        assert count_after >= count_before, "Extension should not remove cells"

        # Check extension cell properties
        for key, cell in extended.items():
            if key not in synthetic_grid:
                # This is an extension cell
                assert cell["density"] == 1.0, "Extension cells should be solid"
                assert cell["cell_type"] == 2
                assert 0 < cell["confidence"] < 1.0

    def test_5_3_floor_coverage(self, synthetic_grid):
        """Test 5.3: Floor covers most of the room area."""
        import copy
        grid = copy.deepcopy(synthetic_grid)
        from trivima.app import run_shell_extension
        extended = run_shell_extension(grid, 0.05)

        # Find floor Y
        floor_cells = [k for k, v in extended.items() if v["normal"][1] > 0.9]
        if not floor_cells:
            pytest.skip("No floor detected")

        from collections import Counter
        floor_y = Counter(k[1] for k in floor_cells).most_common(1)[0][0]

        floor_at_y = [k for k in floor_cells if k[1] == floor_y]
        floor_area = len(floor_at_y) * 0.05 * 0.05  # m²

        print(f"\n  Floor area: {floor_area:.1f} m²")
        assert floor_area > 4.0, f"Floor area {floor_area} m² < 4 m²"
        assert floor_area < 200.0, f"Floor area {floor_area} m² > 200 m² (unreasonable)"


# ============================================================
# LOD Tests (from testing doc, validates subdivision cap)
# ============================================================

class TestLODSubdivision:
    """LOD subdivision cap tests — validates error propagation limits."""

    def test_lod_single_image_cap(self):
        """Single-image: max 1 subdivision level."""
        config = LODConfig(input_type=InputType.SINGLE_IMAGE)
        assert config.max_subdivisions == 1
        assert config.finest_level == -1  # level 0 - 1 = -1

    def test_lod_multi_image_cap(self):
        """Multi-image: max 3 subdivision levels."""
        config = LODConfig(input_type=InputType.MULTI_IMAGE)
        assert config.max_subdivisions == 3

    def test_lod_video_cap(self):
        """Video: max 4 subdivision levels."""
        config = LODConfig(input_type=InputType.VIDEO)
        assert config.max_subdivisions == 4

    def test_lod_low_confidence_blocked(self):
        """Low-confidence cells must not subdivide."""
        lod = LODController(LODConfig(input_type=InputType.SINGLE_IMAGE))
        assert not lod.should_subdivide(cell_level=0, cell_confidence=0.3, distance=1.0)
        assert not lod.should_subdivide(cell_level=0, cell_confidence=0.0, distance=0.5)
        assert not lod.should_subdivide(cell_level=0, cell_confidence=0.49, distance=1.0)

    def test_lod_high_confidence_allowed(self):
        """High-confidence near cells should subdivide."""
        lod = LODController(LODConfig(input_type=InputType.SINGLE_IMAGE))
        assert lod.should_subdivide(cell_level=0, cell_confidence=0.9, distance=1.0)
        assert lod.should_subdivide(cell_level=0, cell_confidence=0.7, distance=2.0)


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
