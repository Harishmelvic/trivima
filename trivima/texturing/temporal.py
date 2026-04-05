"""
Per-cell temporal consistency — blends AI texturing output in 3D cell space.

NOT screen-space EMA (which causes ghosting from parallax misalignment).
Instead, blends light values per cell using view angle delta as alpha.

Key insight: cells have stable 3D positions, so temporal blending in cell
space doesn't suffer from the screen-space parallax problem.

Algorithm:
  1. For each cell updated by the AI model:
     - Compute alpha = clamp(1 - dot(old_view_dir, new_view_dir), 0.1, 1.0)
     - cell.light = lerp(cell.light_prev, new_ai_light, alpha)
  2. Track ∂L/∂t per cell — if |∂L/∂t| < threshold, cell is stable
  3. Dirty mask: only re-light cells where view angle changed significantly
  4. Typically 10-30% of cells need re-lighting per frame during slow navigation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Set, Optional


@dataclass
class CellTemporalState:
    """Per-cell temporal tracking data."""
    prev_light: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    prev_view_dir: np.ndarray = field(default_factory=lambda: np.array([0, 0, -1], dtype=np.float32))
    light_temporal_deriv: float = 0.0  # |∂L/∂t|
    frames_since_update: int = 0
    is_stable: bool = False


class TemporalConsistencyManager:
    """Manages per-cell temporal blending and dirty mask computation.

    Usage:
        manager = TemporalConsistencyManager()

        # Each frame:
        dirty = manager.compute_dirty_mask(grid, camera_pos, camera_forward)
        # ... run AI model only on dirty cells ...
        manager.blend_and_update(grid, ai_results, camera_pos, camera_forward, dt)
    """

    def __init__(
        self,
        stability_threshold: float = 0.02,    # |∂L/∂t| below this → stable
        view_change_threshold: float = 0.98,   # dot(old_view, new_view) below this → dirty
        light_gradient_threshold: float = 0.1, # cells near shadow edges are always dirty
        min_alpha: float = 0.1,                # minimum blend factor (never fully ignore new data)
        max_alpha: float = 1.0,                # maximum blend factor (fast motion → full replace)
    ):
        self.stability_threshold = stability_threshold
        self.view_change_threshold = view_change_threshold
        self.light_gradient_threshold = light_gradient_threshold
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

        # Per-cell temporal state
        self._cell_state: dict[int, CellTemporalState] = {}

    def compute_dirty_mask(
        self,
        grid,
        camera_pos: np.ndarray,
        camera_forward: np.ndarray,
        visible_cell_ids: list,
    ) -> Set[int]:
        """Determine which visible cells need re-lighting by the AI model.

        A cell is dirty if:
          - View angle changed significantly (dot < threshold)
          - Cell is near a shadow edge (high light gradient magnitude)
          - Cell has never been lit
          - Cell has been stable too long (periodic refresh)

        Args:
            grid: CellGrid
            camera_pos: current camera position (3,)
            camera_forward: current camera forward direction (3,)
            visible_cell_ids: list of cell indices currently visible

        Returns:
            Set of cell indices that need re-lighting
        """
        dirty = set()
        camera_forward = camera_forward / (np.linalg.norm(camera_forward) + 1e-8)

        for cell_id in visible_cell_ids:
            state = self._cell_state.get(cell_id)

            # Never been lit → dirty
            if state is None:
                dirty.add(cell_id)
                continue

            # Compute current view direction for this cell
            cell_pos = grid.get_cell_center(cell_id)
            view_dir = cell_pos - camera_pos
            view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)

            # View angle changed significantly
            view_dot = np.dot(state.prev_view_dir, view_dir)
            if view_dot < self.view_change_threshold:
                dirty.add(cell_id)
                continue

            # Near shadow edge (high light gradient)
            light_grad_mag = grid.get_light_gradient_magnitude(cell_id)
            if light_grad_mag > self.light_gradient_threshold:
                dirty.add(cell_id)
                continue

            # Periodic refresh (every 60 frames for stable cells)
            if state.frames_since_update > 60:
                dirty.add(cell_id)
                continue

            # Cell is stable — skip it
            state.frames_since_update += 1

        return dirty

    def blend_and_update(
        self,
        grid,
        updated_cells: dict,  # cell_id -> new_light (3,)
        camera_pos: np.ndarray,
        camera_forward: np.ndarray,
        dt: float,
    ) -> dict:
        """Blend new AI lighting with previous values and update temporal state.

        Args:
            grid: CellGrid
            updated_cells: dict mapping cell_id → new light RGB array (3,)
            camera_pos: current camera position
            camera_forward: current camera forward direction
            dt: time since last frame in seconds

        Returns:
            dict with stats: num_blended, avg_alpha, pct_stable
        """
        alphas = []
        num_stable = 0

        for cell_id, new_light in updated_cells.items():
            state = self._cell_state.get(cell_id)

            if state is None:
                # First time — no blending, just set
                state = CellTemporalState()
                self._cell_state[cell_id] = state
                state.prev_light = new_light.copy()
                alpha = 1.0
            else:
                # Compute view direction for this cell
                cell_pos = grid.get_cell_center(cell_id)
                view_dir = cell_pos - camera_pos
                view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)

                # Alpha based on view angle change
                view_dot = np.clip(np.dot(state.prev_view_dir, view_dir), -1.0, 1.0)
                alpha = np.clip(1.0 - view_dot, self.min_alpha, self.max_alpha)

                # Blend: lerp(old, new, alpha)
                blended = (1.0 - alpha) * state.prev_light + alpha * new_light

                # Compute temporal derivative
                if dt > 0:
                    light_change = np.linalg.norm(blended - state.prev_light)
                    state.light_temporal_deriv = light_change / dt
                    state.is_stable = state.light_temporal_deriv < self.stability_threshold

                new_light = blended
                state.prev_light = blended.copy()

                # Update view direction
                state.prev_view_dir = view_dir.copy()

            # Write to cell grid
            grid.set_cell_light(cell_id, new_light[0], new_light[1], new_light[2])

            # Write temporal derivative to cell
            grid.set_light_temporal_deriv(cell_id, state.light_temporal_deriv)

            state.frames_since_update = 0
            alphas.append(alpha)
            if state.is_stable:
                num_stable += 1

        total = len(updated_cells) if updated_cells else 1
        return {
            "num_blended": len(updated_cells),
            "avg_alpha": float(np.mean(alphas)) if alphas else 0,
            "pct_stable": num_stable / total * 100,
        }

    def get_stats(self) -> dict:
        """Return summary statistics about temporal state."""
        total = len(self._cell_state)
        if total == 0:
            return {"total_tracked": 0, "pct_stable": 0, "avg_deriv": 0}

        stable = sum(1 for s in self._cell_state.values() if s.is_stable)
        avg_deriv = np.mean([s.light_temporal_deriv for s in self._cell_state.values()])

        return {
            "total_tracked": total,
            "pct_stable": stable / total * 100,
            "avg_deriv": float(avg_deriv),
        }
