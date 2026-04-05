"""
Surface support field — queries whether a position has a surface to hold a placed object.

From unified_pipeline_theory.md §1.2:
  - Scans cell grid for surface cells with upward normals (normal_y > 0.85)
  - Clusters into floor (lowest, largest) and elevated surfaces (tables, shelves)
  - Each surface: height, XZ extent, semantic label, load capacity
  - Confidence-weighted: high-conf cells dominate plane fitting

Query API:
  query_support(grid, x, y, z) → SupportResult
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import Counter


# Semantic categories and their estimated load capacity (kg)
LOAD_CAPACITY = {
    "floor": 1000.0,
    "table": 50.0,
    "dining table": 50.0,
    "desk": 40.0,
    "counter": 60.0,
    "kitchen counter": 60.0,
    "shelf": 15.0,
    "bookshelf": 30.0,
    "nightstand": 15.0,
    "cabinet": 30.0,
    "bed": 200.0,
    "sofa": 150.0,
}


@dataclass
class SurfaceCluster:
    """One detected support surface."""
    surface_type: str          # "floor", "table", "shelf", etc.
    height: float              # Y coordinate in meters
    cells: List[tuple]         # list of (ix, iy, iz) cell keys
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    area_m2: float
    mean_confidence: float
    load_capacity_kg: float
    semantic_label: int


@dataclass
class SupportResult:
    """Result of a surface support query."""
    has_support: bool
    surface_type: str          # "floor", "table", "shelf", "none"
    height: float              # surface height in meters
    confidence: float          # support confidence [0,1]
    clearance_above: float     # meters of free space above the surface at this position
    load_capacity_kg: float


class SurfaceField:
    """Detects and queries support surfaces from the cell grid.

    Usage:
        field = SurfaceField(cell_size=0.05)
        field.build(grid_data, label_names)
        result = field.query(x, y, z)
    """

    def __init__(self, cell_size: float = 0.05, normal_threshold: float = 0.85,
                 height_tolerance: float = 0.02):
        self.cell_size = cell_size
        self.normal_threshold = normal_threshold
        self.height_tolerance = height_tolerance
        self.surfaces: List[SurfaceCluster] = []
        self._floor_height: Optional[float] = None

    def build(self, grid_data: dict, label_names: Dict[int, str] = None):
        """Detect all support surfaces from the cell grid.

        Args:
            grid_data: cell grid dict keyed by (ix, iy, iz)
            label_names: optional label index → name mapping
        """
        if label_names is None:
            label_names = {}

        # Find all upward-facing surface cells
        surface_cells = []
        for key, cell in grid_data.items():
            n = cell.get("normal", np.zeros(3))
            if n[1] > self.normal_threshold and cell.get("density", 0) > 0.3:
                surface_cells.append((key, cell))

        if not surface_cells:
            return

        # Group by height (Y level)
        height_groups = {}
        for key, cell in surface_cells:
            iy = key[1]
            h = (iy + 0.5) * self.cell_size
            # Quantize height to tolerance
            h_key = round(h / self.height_tolerance) * self.height_tolerance
            if h_key not in height_groups:
                height_groups[h_key] = []
            height_groups[h_key].append((key, cell))

        # Build surface clusters
        self.surfaces = []
        for h, cells in height_groups.items():
            if len(cells) < 3:  # skip tiny clusters
                continue

            keys = [c[0] for c in cells]
            cell_data = [c[1] for c in cells]

            xs = [(k[0] + 0.5) * self.cell_size for k in keys]
            zs = [(k[2] + 0.5) * self.cell_size for k in keys]
            confs = [c.get("confidence", 0.5) for c in cell_data]

            # Determine surface type from semantic labels
            labels = [c.get("label", 0) for c in cell_data]
            most_common_label = Counter(labels).most_common(1)[0][0]
            label_name = label_names.get(most_common_label, "surface").lower()

            # Match to known surface types
            surface_type = "surface"
            load_cap = 20.0
            for known_type, cap in LOAD_CAPACITY.items():
                if known_type in label_name:
                    surface_type = known_type
                    load_cap = cap
                    break

            area = len(cells) * self.cell_size ** 2

            cluster = SurfaceCluster(
                surface_type=surface_type,
                height=float(h),
                cells=keys,
                x_min=float(min(xs)),
                x_max=float(max(xs)),
                z_min=float(min(zs)),
                z_max=float(max(zs)),
                area_m2=float(area),
                mean_confidence=float(np.average(confs)),
                load_capacity_kg=load_cap,
                semantic_label=most_common_label,
            )
            self.surfaces.append(cluster)

        # Sort by area (largest first) — floor is typically the largest
        self.surfaces.sort(key=lambda s: s.area_m2, reverse=True)

        # Identify floor as the lowest large surface
        if self.surfaces:
            floor_candidates = [s for s in self.surfaces if s.area_m2 > 1.0]
            if floor_candidates:
                self._floor_height = min(s.height for s in floor_candidates)
                for s in floor_candidates:
                    if abs(s.height - self._floor_height) < 0.05:
                        s.surface_type = "floor"
                        s.load_capacity_kg = 1000.0

    def query(self, x: float, y: float, z: float,
              tolerance: float = 0.02) -> SupportResult:
        """Query whether a surface exists to support an object at this position.

        Args:
            x, y, z: world position to query
            tolerance: height tolerance in meters (default 2cm)

        Returns:
            SupportResult with support info
        """
        best = None
        best_dist = float('inf')

        for surface in self.surfaces:
            # Check XZ bounds
            if not (surface.x_min - tolerance <= x <= surface.x_max + tolerance and
                    surface.z_min - tolerance <= z <= surface.z_max + tolerance):
                continue

            # Check height — object should be placed ON the surface (y ≈ surface.height)
            height_diff = abs(y - surface.height)
            if height_diff < tolerance + 0.1:  # within tolerance + small margin
                if height_diff < best_dist:
                    best_dist = height_diff
                    best = surface

        if best is None:
            return SupportResult(
                has_support=False, surface_type="none",
                height=0, confidence=0, clearance_above=0, load_capacity_kg=0,
            )

        return SupportResult(
            has_support=True,
            surface_type=best.surface_type,
            height=best.height,
            confidence=best.mean_confidence,
            clearance_above=self._compute_clearance(best.height, x, z),
            load_capacity_kg=best.load_capacity_kg,
        )

    def _compute_clearance(self, surface_height: float, x: float, z: float) -> float:
        """Compute free space above a surface at position (x, z)."""
        # Find the next surface above this one
        above = [s for s in self.surfaces if s.height > surface_height + 0.1]
        if not above:
            return 3.0  # assume 3m ceiling if no surface above

        for s in sorted(above, key=lambda s: s.height):
            if (s.x_min <= x <= s.x_max and s.z_min <= z <= s.z_max):
                return s.height - surface_height

        return 3.0  # no surface directly above

    @property
    def floor_height(self) -> Optional[float]:
        return self._floor_height

    def get_summary(self) -> dict:
        return {
            "num_surfaces": len(self.surfaces),
            "floor_height": self._floor_height,
            "surfaces": [
                {"type": s.surface_type, "height": s.height,
                 "area": s.area_m2, "confidence": s.mean_confidence}
                for s in self.surfaces
            ],
        }
