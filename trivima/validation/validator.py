"""
Frame validator — orchestrates all conservation checks per frame.

Runs asynchronously, one frame behind rendering:
  Frame N: render + display
  Frame N-1: validate + queue corrections
  Frame N+1: apply gradual corrections

This hides validation latency from the render loop.
"""

import numpy as np
import time
from typing import List, Optional
from .conservation import (
    EnergyConservationChecker,
    MassConservationChecker,
    ShadowDirectionChecker,
    GradualCorrector,
    ConservationReport,
)


class FrameValidator:
    """Orchestrates all conservation checks for the cell grid.

    Usage:
        validator = FrameValidator(grid)

        # Each frame (runs async, 1 frame behind):
        report = validator.validate_frame(visible_cells, light_positions)
        validator.apply_corrections(grid)  # apply pending gradual corrections
    """

    def __init__(
        self,
        grid,
        energy_tolerance: float = 0.05,
        mass_tolerance: float = 0.001,
        shadow_threshold: float = 0.5,
        correction_spread_frames: int = 4,
    ):
        self.energy_checker = EnergyConservationChecker(
            per_cell_tolerance=energy_tolerance
        )
        self.mass_checker = MassConservationChecker(
            tolerance=mass_tolerance
        )
        self.shadow_checker = ShadowDirectionChecker(
            consistency_threshold=shadow_threshold
        )
        self.corrector = GradualCorrector(
            spread_frames=correction_spread_frames
        )

        # Set initial reference mass
        self.mass_checker.set_reference(grid)

        # History for tracking
        self._frame_count = 0
        self._history: List[ConservationReport] = []
        self._last_validation_ms = 0.0

    def validate_frame(
        self,
        grid,
        visible_cell_ids: list,
        light_positions: Optional[List[np.ndarray]] = None,
    ) -> ConservationReport:
        """Run all conservation checks for one frame.

        Args:
            grid: CellGrid
            visible_cell_ids: list of visible cell indices
            light_positions: list of (3,) light source positions in world space

        Returns:
            ConservationReport with all violations and stats
        """
        t0 = time.perf_counter()
        report = ConservationReport()
        report.total_cells_checked = len(visible_cell_ids)

        # 1. Energy conservation
        energy_violations = self.energy_checker.check(grid, visible_cell_ids)
        report.energy_violations = energy_violations
        report.energy_pct_violating = (
            len(energy_violations) / max(len(visible_cell_ids), 1) * 100
        )
        report.energy_max_violation = (
            max(v.magnitude for v in energy_violations) if energy_violations else 0.0
        )

        # 2. Mass conservation
        report.mass_violation = self.mass_checker.check(grid)

        # 3. Shadow direction (only if light positions are known)
        if light_positions:
            shadow_violations = self.shadow_checker.check(
                grid, visible_cell_ids, light_positions
            )
            report.shadow_violations = shadow_violations
            report.shadow_pct_violating = (
                len(shadow_violations) / max(len(visible_cell_ids), 1) * 100
            )

        # Queue corrections for gradual application
        self.corrector.queue_corrections(energy_violations)
        if light_positions:
            self.corrector.queue_corrections(report.shadow_violations)

        self._last_validation_ms = (time.perf_counter() - t0) * 1000
        self._frame_count += 1
        self._history.append(report)

        # Keep only last 100 reports
        if len(self._history) > 100:
            self._history = self._history[-100:]

        return report

    def apply_corrections(self, grid) -> int:
        """Apply one frame's gradual corrections.

        Call this each frame BEFORE rendering.
        Returns number of corrections applied.
        """
        return self.corrector.apply_frame(grid)

    def get_summary(self) -> dict:
        """Get summary stats across recent frames."""
        if not self._history:
            return {
                "frames_validated": 0,
                "avg_energy_pct": 0,
                "avg_mass_drift": 0,
                "avg_shadow_pct": 0,
                "pending_corrections": 0,
                "last_validation_ms": 0,
            }

        recent = self._history[-30:]  # last 30 frames
        return {
            "frames_validated": self._frame_count,
            "avg_energy_pct": np.mean([r.energy_pct_violating for r in recent]),
            "avg_mass_drift": np.mean([abs(r.mass_violation) for r in recent]),
            "avg_shadow_pct": np.mean([r.shadow_pct_violating for r in recent]),
            "pending_corrections": self.corrector.pending_count,
            "last_validation_ms": self._last_validation_ms,
        }

    def inject_deliberate_error(self, grid, cell_id: int, error_type: str = "energy"):
        """Inject a deliberate error to test if validation catches it.

        Used during Week 4 testing to verify conservation checks work.

        Args:
            grid: CellGrid
            cell_id: which cell to corrupt
            error_type: "energy" (boost light 2×), "mass" (zero density), "shadow" (flip gradient)
        """
        vis = grid.get_vis(cell_id)

        if error_type == "energy":
            # Double the light value — should trigger energy conservation
            grid.set_cell_light(cell_id, vis.light_r * 2, vis.light_g * 2, vis.light_b * 2)

        elif error_type == "mass":
            # Zero out density — should trigger mass conservation
            geo = grid.get_geo(cell_id)
            geo.density = 0.0
            geo.density_integral = 0.0

        elif error_type == "shadow":
            # Flip light gradient direction — should trigger shadow direction check
            grid.set_cell_light_gradient(cell_id, -vis.light_gx, -vis.light_gy, -vis.light_gz)
