"""
Adaptive Level-of-Detail controller for the cell grid.

Selects resolution tier per cell based on distance from camera.
Enforces subdivision limits from error propagation analysis:

  Single-image input: max 1 Taylor subdivision level (5cm → 2.5cm)
    - Child prediction error ±1.25cm ≈ child cell size → diminishing returns beyond this
    - Finer detail comes from neural texture features + AI texturing

  Multi-image input: max 3 Taylor subdivision levels (5cm → 2.5cm → 1.25cm → 0.6cm)
    - Multi-view reduces gradient error proportionally → deeper subdivision is reliable

  Beyond max subdivision: cells are NOT further subdivided. The renderer uses
  neural texture decode at sub-cell positions for visual detail.

See: single_image_precision_theory.md, Chapter 10 (Error Propagation)
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple, Optional


class InputType(IntEnum):
    """Determines subdivision limits based on input data quality."""
    SINGLE_IMAGE = 1     # max 1 subdivision level
    MULTI_IMAGE = 2      # max 3 subdivision levels
    VIDEO = 3            # max 4 subdivision levels (sub-cm precision)


# Subdivision limits per input type (from error propagation analysis)
MAX_SUBDIVISION_LEVELS = {
    InputType.SINGLE_IMAGE: 1,   # 5cm → 2.5cm, then neural features
    InputType.MULTI_IMAGE: 3,    # 5cm → 2.5cm → 1.25cm → 0.6cm
    InputType.VIDEO: 4,          # 5cm → ... → 0.3cm
}

# Distance thresholds for LOD tiers (meters from camera)
# Tier 2 (2cm): < near_threshold
# Tier 1 (5cm): near_threshold to mid_threshold
# Tier 0 (20cm): > mid_threshold
DEFAULT_NEAR_THRESHOLD = 3.0    # meters
DEFAULT_MID_THRESHOLD = 15.0    # meters


@dataclass
class LODConfig:
    """Configuration for the LOD controller."""
    input_type: InputType = InputType.SINGLE_IMAGE
    near_threshold: float = DEFAULT_NEAR_THRESHOLD
    mid_threshold: float = DEFAULT_MID_THRESHOLD
    max_visible_cells: int = 200_000
    base_level: int = 0          # level 0 = 5cm

    @property
    def max_subdivisions(self) -> int:
        return MAX_SUBDIVISION_LEVELS[self.input_type]

    @property
    def finest_level(self) -> int:
        """Most negative level allowed (finest resolution)."""
        return self.base_level - self.max_subdivisions


@dataclass
class LODDecision:
    """Per-cell LOD decision."""
    cell_index: int
    current_level: int
    desired_level: int
    action: str  # "keep", "subdivide", "merge"


class LODController:
    """Selects per-cell resolution and enforces subdivision limits.

    Usage:
        lod = LODController(LODConfig(input_type=InputType.SINGLE_IMAGE))

        # Each frame:
        decisions = lod.compute(grid, camera_pos)
        subdivide_list = [d for d in decisions if d.action == "subdivide"]
        merge_list = [d for d in decisions if d.action == "merge"]
    """

    def __init__(self, config: LODConfig = LODConfig()):
        self.config = config
        self._visible_count = 0

    def desired_level(self, distance: float) -> int:
        """Compute desired cell level based on distance from camera.

        Returns level (0 = 5cm base, negative = finer, positive = coarser).
        Clamped to the subdivision limit from error propagation analysis.
        """
        if distance < self.config.near_threshold:
            # Want finest allowed resolution
            raw_level = self.config.base_level - self.config.max_subdivisions
        elif distance < self.config.mid_threshold:
            # Base resolution
            raw_level = self.config.base_level
        else:
            # Coarse (10-20cm cells)
            if distance > self.config.mid_threshold * 1.5:
                raw_level = self.config.base_level + 2  # 20cm
            else:
                raw_level = self.config.base_level + 1  # 10cm

        # Clamp: never finer than allowed by input quality
        return max(raw_level, self.config.finest_level)

    def should_subdivide(self, cell_level: int, cell_confidence: float,
                         distance: float) -> bool:
        """Check if a cell should subdivide.

        Blocks subdivision for:
          - Cells already at finest allowed level
          - Low-confidence cells (gradient data is unreliable)
          - Distance too far
        """
        target = self.desired_level(distance)

        if cell_level <= target:
            return False  # already fine enough or finer

        # Low-confidence cells should NOT subdivide via Taylor expansion
        # because their gradients are unreliable (glass, mirrors, dark regions)
        if cell_confidence < 0.5:
            return False

        return True

    def should_merge(self, cell_level: int, distance: float) -> bool:
        """Check if a cell (and its siblings) should merge back to parent."""
        target = self.desired_level(distance)
        return cell_level < target  # cell is finer than needed

    def compute(self, grid, camera_pos: np.ndarray) -> List[LODDecision]:
        """Compute LOD decisions for all cells in the grid.

        Args:
            grid: CellGrid (native or Python wrapper)
            camera_pos: (3,) camera position in world space

        Returns:
            List of LODDecision indicating what to do with each cell
        """
        decisions = []
        subdivide_count = 0
        total_visible = 0

        for i in range(grid.size()):
            key = grid.key(i)
            geo = grid.geo(i)

            # Compute distance from camera to cell center
            center = grid.cell_center_pos(key)
            dx = center.x - camera_pos[0]
            dy = center.y - camera_pos[1]
            dz = center.z - camera_pos[2]
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5

            target_level = self.desired_level(dist)

            if key.level > target_level and self.should_subdivide(
                key.level, geo.confidence, dist
            ):
                # Budget check: don't exceed max visible cells
                # Each subdivision creates 8 children from 1 parent = +7 cells
                if total_visible + subdivide_count * 7 < self.config.max_visible_cells:
                    decisions.append(LODDecision(i, key.level, target_level, "subdivide"))
                    subdivide_count += 1
                else:
                    decisions.append(LODDecision(i, key.level, target_level, "keep"))
            elif key.level < target_level and self.should_merge(key.level, dist):
                decisions.append(LODDecision(i, key.level, target_level, "merge"))
            else:
                decisions.append(LODDecision(i, key.level, target_level, "keep"))

            total_visible += 1

        self._visible_count = total_visible
        return decisions

    def get_stats(self) -> dict:
        return {
            "input_type": self.config.input_type.name,
            "max_subdivisions": self.config.max_subdivisions,
            "finest_level": self.config.finest_level,
            "visible_cells": self._visible_count,
            "budget": self.config.max_visible_cells,
        }
