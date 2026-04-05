"""
Conservation validation — checks physical laws using integral data built into cells.

Three conservation laws from the theory (Chapter 8):
  1. Energy: reflected light ≤ incoming light per cell
  2. Mass: total density integral constant between frames
  3. Shadow direction: light gradient aligns with light source positions

Corrections are applied gradually over 3-5 frames to avoid visible artifacts.
Runs async, one frame behind rendering.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ConservationViolation:
    """One detected conservation violation."""
    cell_id: int
    law: str            # "energy", "mass", "shadow"
    magnitude: float    # how far off (0 = no violation, 1 = 100% off)
    correction: np.ndarray  # suggested correction to apply


@dataclass
class ConservationReport:
    """Results of all conservation checks for one frame."""
    energy_violations: List[ConservationViolation] = field(default_factory=list)
    mass_violation: float = 0.0  # scene-wide mass drift as fraction
    shadow_violations: List[ConservationViolation] = field(default_factory=list)

    # Summary stats
    energy_pct_violating: float = 0.0
    energy_max_violation: float = 0.0
    shadow_pct_violating: float = 0.0
    total_cells_checked: int = 0

    @property
    def is_clean(self) -> bool:
        return (len(self.energy_violations) == 0 and
                abs(self.mass_violation) < 0.001 and
                len(self.shadow_violations) == 0)


class EnergyConservationChecker:
    """Per-cell energy conservation: reflected + absorbed ≤ incoming.

    For each cell:
      incoming = sum(neighbor flux toward this cell) + emissive
      reflected = albedo × light_integral
      absorbed = (1 - albedo) × light_integral

    Violation if reflected + absorbed > incoming × (1 + tolerance).

    Correction: scale light_value until conservation holds.
    """

    def __init__(self, per_cell_tolerance: float = 0.05, scene_tolerance: float = 0.01):
        self.per_cell_tolerance = per_cell_tolerance
        self.scene_tolerance = scene_tolerance

    def check(self, grid, visible_cell_ids: list) -> List[ConservationViolation]:
        violations = []

        for cid in visible_cell_ids:
            vis = grid.get_vis(cid)
            geo = grid.get_geo(cid)

            # Skip empty cells
            if geo.is_empty():
                continue

            albedo_luma = (vis.albedo_r + vis.albedo_g + vis.albedo_b) / 3.0
            light_integral = vis.light_integral

            if light_integral <= 0:
                continue

            # Compute incoming flux from neighbors
            incoming = 0.0
            for n in range(6):
                neighbor = vis.neighbors[n]
                incoming += neighbor.light_luma * 0.1  # approximate flux

            # Allow for ambient light contribution
            incoming = max(incoming, 0.01)

            reflected = albedo_luma * light_integral
            absorbed = (1.0 - albedo_luma) * light_integral
            total_outgoing = reflected + absorbed

            # Check violation
            if total_outgoing > incoming * (1.0 + self.per_cell_tolerance):
                excess = total_outgoing - incoming
                magnitude = excess / (total_outgoing + 1e-8)

                # Correction: scale light down
                correction_factor = incoming / (total_outgoing + 1e-8)
                light = np.array([vis.light_r, vis.light_g, vis.light_b])
                corrected_light = light * correction_factor

                violations.append(ConservationViolation(
                    cell_id=cid,
                    law="energy",
                    magnitude=magnitude,
                    correction=corrected_light,
                ))

        return violations


class MassConservationChecker:
    """Scene-wide mass conservation: total density integral must be constant.

    The total density integral is computed once at scene construction.
    Any drift indicates a bug in the cell grid update logic.
    """

    def __init__(self, tolerance: float = 0.001):
        self.tolerance = tolerance
        self._reference_mass: Optional[float] = None

    def set_reference(self, grid):
        """Compute and store the reference total mass."""
        total = 0.0
        for i in range(grid.size()):
            total += grid.get_geo(i).density_integral
        self._reference_mass = total

    def check(self, grid) -> float:
        """Check mass conservation. Returns drift as fraction of reference.

        Returns 0.0 if within tolerance, positive value if mass increased,
        negative if mass decreased.
        """
        if self._reference_mass is None:
            self.set_reference(grid)
            return 0.0

        current = 0.0
        for i in range(grid.size()):
            current += grid.get_geo(i).density_integral

        if abs(self._reference_mass) < 1e-8:
            return 0.0

        drift = (current - self._reference_mass) / self._reference_mass
        return drift


class ShadowDirectionChecker:
    """Shadow direction consistency: light gradient should align with light sources.

    For cells near shadow edges (high light gradient magnitude), the gradient
    direction should roughly point from shadow toward light source.

    Correction: rotate light gradient toward expected direction (30% correction).
    """

    def __init__(
        self,
        consistency_threshold: float = 0.5,
        gradient_magnitude_threshold: float = 0.1,
        correction_strength: float = 0.3,
    ):
        self.consistency_threshold = consistency_threshold
        self.gradient_magnitude_threshold = gradient_magnitude_threshold
        self.correction_strength = correction_strength

    def check(
        self,
        grid,
        visible_cell_ids: list,
        light_positions: List[np.ndarray],
    ) -> List[ConservationViolation]:
        if not light_positions:
            return []

        violations = []

        for cid in visible_cell_ids:
            vis = grid.get_vis(cid)

            # Light gradient magnitude
            grad = np.array([vis.light_gx, vis.light_gy, vis.light_gz])
            grad_mag = np.linalg.norm(grad)

            if grad_mag < self.gradient_magnitude_threshold:
                continue  # not near a shadow edge

            grad_dir = grad / (grad_mag + 1e-8)

            # Expected direction: from cell toward nearest light
            cell_pos = grid.get_cell_center(cid)
            best_consistency = -1.0
            best_expected_dir = grad_dir

            for light_pos in light_positions:
                expected_dir = light_pos - cell_pos
                expected_dir = expected_dir / (np.linalg.norm(expected_dir) + 1e-8)
                consistency = np.dot(grad_dir, expected_dir)

                if consistency > best_consistency:
                    best_consistency = consistency
                    best_expected_dir = expected_dir

            if best_consistency < self.consistency_threshold:
                # Shadow direction off — blend toward expected
                corrected_dir = (
                    (1.0 - self.correction_strength) * grad_dir +
                    self.correction_strength * best_expected_dir
                )
                corrected_dir = corrected_dir / (np.linalg.norm(corrected_dir) + 1e-8)
                corrected_grad = corrected_dir * grad_mag

                violations.append(ConservationViolation(
                    cell_id=cid,
                    law="shadow",
                    magnitude=1.0 - best_consistency,
                    correction=corrected_grad,
                ))

        return violations


class GradualCorrector:
    """Applies conservation corrections gradually over multiple frames.

    Instead of applying the full correction instantly (which can cause
    visible artifacts), we spread it over 3-5 frames.
    """

    def __init__(self, spread_frames: int = 4):
        self.spread_frames = spread_frames
        self._pending: dict = {}  # cell_id -> (correction, frames_remaining, law)

    def queue_corrections(self, violations: List[ConservationViolation]):
        """Queue new violations for gradual correction."""
        for v in violations:
            # Per-frame correction = total / spread_frames
            per_frame = v.correction / self.spread_frames
            self._pending[v.cell_id] = (per_frame, self.spread_frames, v.law)

    def apply_frame(self, grid) -> int:
        """Apply one frame's worth of corrections. Returns number of corrections applied."""
        applied = 0
        to_remove = []

        for cid, (per_frame, remaining, law) in self._pending.items():
            if law == "energy":
                # Scale light toward corrected value
                vis = grid.get_vis(cid)
                current = np.array([vis.light_r, vis.light_g, vis.light_b])
                adjusted = current + (per_frame - current) / remaining
                grid.set_cell_light(cid, adjusted[0], adjusted[1], adjusted[2])
            elif law == "shadow":
                # Adjust light gradient direction
                grid.set_cell_light_gradient(cid, per_frame[0], per_frame[1], per_frame[2])

            remaining -= 1
            if remaining <= 0:
                to_remove.append(cid)
            else:
                self._pending[cid] = (per_frame, remaining, law)
            applied += 1

        for cid in to_remove:
            del self._pending[cid]

        return applied

    @property
    def pending_count(self) -> int:
        return len(self._pending)
