"""
Functional field — queries whether a position serves an object's purpose.

From unified_pipeline_theory.md §1.5:
  - Plants: proximity to window cells (closer = higher score)
  - Lamps: proximity to seating cells (1-2m optimal)
  - Storage: adjacency to wall cells
  - Rugs: centered between seating clusters
  - Tables: proximity to seating, on floor, adequate clearance

Uses cell semantic labels + positions + neighbor summaries — no language processing.

Query API:
  query_functional(grid, x, y, z, object_category) → score [0,1]
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# Semantic label categories for functional queries
WINDOW_LABELS = {"window", "glass window", "bay window"}
SEATING_LABELS = {"sofa", "chair", "armchair", "couch", "bench", "dining chair", "office chair"}
WALL_LABELS = {"wall"}
DOOR_LABELS = {"door", "doorway", "interior door", "standard door"}
BED_LABELS = {"bed", "single bed", "double bed"}
TABLE_LABELS = {"table", "dining table", "desk", "coffee table"}

# Optimal distances (meters) for functional scoring
FUNCTIONAL_RULES = {
    "plant": {
        "attract": WINDOW_LABELS,        # closer to windows = better
        "optimal_distance": 0.5,         # ideal distance to window
        "max_distance": 3.0,             # beyond this, score drops to 0
        "repel": set(),
        "wall_adjacent": False,
    },
    "lamp": {
        "attract": SEATING_LABELS,       # near seating areas
        "optimal_distance": 1.5,         # 1-2m from seating is ideal
        "max_distance": 3.0,
        "repel": set(),
        "wall_adjacent": False,
    },
    "floor_lamp": {
        "attract": SEATING_LABELS,
        "optimal_distance": 1.0,
        "max_distance": 2.5,
        "repel": set(),
        "wall_adjacent": False,
    },
    "bookshelf": {
        "attract": set(),
        "optimal_distance": 0,
        "max_distance": 0,
        "repel": set(),
        "wall_adjacent": True,           # must be near a wall
    },
    "cabinet": {
        "attract": set(),
        "optimal_distance": 0,
        "max_distance": 0,
        "repel": set(),
        "wall_adjacent": True,
    },
    "shelf": {
        "attract": set(),
        "optimal_distance": 0,
        "max_distance": 0,
        "repel": set(),
        "wall_adjacent": True,
    },
    "rug": {
        "attract": SEATING_LABELS,       # centered among seating
        "optimal_distance": 0.0,         # directly under/between seating
        "max_distance": 2.0,
        "repel": set(),
        "wall_adjacent": False,
    },
    "coffee_table": {
        "attract": SEATING_LABELS,       # near seating
        "optimal_distance": 0.8,         # arm's reach from sofa
        "max_distance": 2.0,
        "repel": DOOR_LABELS,            # not blocking doorways
        "wall_adjacent": False,
    },
    "nightstand": {
        "attract": BED_LABELS,           # next to bed
        "optimal_distance": 0.3,
        "max_distance": 1.0,
        "repel": set(),
        "wall_adjacent": False,
    },
    "tv": {
        "attract": SEATING_LABELS,       # facing seating
        "optimal_distance": 2.5,         # viewing distance
        "max_distance": 5.0,
        "repel": set(),
        "wall_adjacent": True,           # typically against a wall
    },
    "dining_chair": {
        "attract": TABLE_LABELS,         # around a table
        "optimal_distance": 0.5,
        "max_distance": 1.5,
        "repel": set(),
        "wall_adjacent": False,
    },
}

# Default rule for unknown categories
DEFAULT_RULE = {
    "attract": set(),
    "optimal_distance": 0,
    "max_distance": 0,
    "repel": set(),
    "wall_adjacent": False,
}


@dataclass
class FunctionalResult:
    """Result of a functional field query."""
    score: float                 # [0, 1] overall functional score
    attract_score: float         # proximity to desired features
    repel_score: float           # distance from undesired features
    wall_score: float            # wall adjacency score (if required)
    nearest_attract_distance: float  # meters to nearest attract cell
    nearest_attract_type: str


class FunctionalField:
    """Computes functional suitability scores for object placement.

    Usage:
        field = FunctionalField(cell_size=0.05)
        field.build(grid_data, label_names)
        result = field.query(x, y, z, "plant")
    """

    def __init__(self, cell_size: float = 0.05):
        self.cell_size = cell_size
        # Cached label clusters: label_category → list of (centroid_x, centroid_z)
        self._clusters: Dict[str, List[Tuple[float, float, float]]] = {}
        self._wall_cells: List[Tuple[float, float, float]] = []

    def build(self, grid_data: dict, label_names: Dict[int, str] = None):
        """Build functional field caches from the cell grid.

        Finds clusters of semantically meaningful cells (windows, seating, walls, etc.)
        and caches their centroid positions for fast distance queries.
        """
        if label_names is None:
            label_names = {}

        # Reverse lookup: category → set of label indices
        label_to_category = {}
        for idx, name in label_names.items():
            name_lower = name.lower().strip()
            label_to_category[idx] = name_lower

        # Collect cells by category
        category_cells: Dict[str, List[Tuple[float, float, float]]] = {}
        all_categories = (
            WINDOW_LABELS | SEATING_LABELS | WALL_LABELS |
            DOOR_LABELS | BED_LABELS | TABLE_LABELS
        )

        for key, cell in grid_data.items():
            label_idx = cell.get("label", 0)
            label_name = label_to_category.get(label_idx, "")

            pos = (
                (key[0] + 0.5) * self.cell_size,
                (key[1] + 0.5) * self.cell_size,
                (key[2] + 0.5) * self.cell_size,
            )

            for category in all_categories:
                if category in label_name or label_name in category:
                    if category not in category_cells:
                        category_cells[category] = []
                    category_cells[category].append(pos)

            # Track wall cells separately (using normal direction)
            n = cell.get("normal", np.zeros(3))
            if abs(n[1]) < 0.3 and cell.get("density", 0) > 0.5:
                self._wall_cells.append(pos)

        # Compute centroids per category cluster
        self._clusters = {}
        for category, positions in category_cells.items():
            if positions:
                centroid = np.mean(positions, axis=0)
                self._clusters[category] = [tuple(centroid)]

    def query(self, x: float, y: float, z: float,
              object_category: str) -> FunctionalResult:
        """Query functional suitability for placing an object at (x, y, z).

        Args:
            x, y, z: world position
            object_category: e.g. "plant", "lamp", "bookshelf"

        Returns:
            FunctionalResult with score [0, 1]
        """
        rule = FUNCTIONAL_RULES.get(object_category.lower(), DEFAULT_RULE)

        attract_score = 1.0
        nearest_dist = float('inf')
        nearest_type = "none"

        # Attraction scoring: proximity to desired features
        if rule["attract"]:
            best_dist = float('inf')
            for label_category in rule["attract"]:
                positions = self._clusters.get(label_category, [])
                for pos in positions:
                    dist = np.sqrt((x - pos[0])**2 + (z - pos[2])**2)
                    if dist < best_dist:
                        best_dist = dist
                        nearest_type = label_category

            nearest_dist = best_dist

            if best_dist < float('inf'):
                optimal = rule["optimal_distance"]
                max_dist = rule["max_distance"]

                if max_dist > 0:
                    # Score peaks at optimal distance, decays with Gaussian
                    deviation = abs(best_dist - optimal)
                    sigma = max_dist / 2.0
                    attract_score = float(np.exp(-(deviation / sigma)**2))
                else:
                    attract_score = 1.0
            else:
                # No attract targets found — neutral score
                attract_score = 0.5

        # Repulsion scoring: distance from undesired features
        repel_score = 1.0
        if rule["repel"]:
            for label_category in rule["repel"]:
                positions = self._clusters.get(label_category, [])
                for pos in positions:
                    dist = np.sqrt((x - pos[0])**2 + (z - pos[2])**2)
                    if dist < 1.0:  # too close to repel target
                        repel_score *= dist  # linear penalty below 1m

        # Wall adjacency scoring
        wall_score = 1.0
        if rule["wall_adjacent"]:
            if self._wall_cells:
                min_wall_dist = float('inf')
                for wpos in self._wall_cells:
                    dist = np.sqrt((x - wpos[0])**2 + (z - wpos[2])**2)
                    min_wall_dist = min(min_wall_dist, dist)

                # Score: 1.0 if within 30cm of wall, drops to 0 at 2m
                if min_wall_dist < 0.3:
                    wall_score = 1.0
                elif min_wall_dist < 2.0:
                    wall_score = 1.0 - (min_wall_dist - 0.3) / 1.7
                else:
                    wall_score = 0.0
            else:
                wall_score = 0.5  # no walls detected, neutral

        # Combined score
        score = attract_score * repel_score * wall_score

        return FunctionalResult(
            score=float(np.clip(score, 0, 1)),
            attract_score=float(attract_score),
            repel_score=float(repel_score),
            wall_score=float(wall_score),
            nearest_attract_distance=float(nearest_dist),
            nearest_attract_type=nearest_type,
        )

    def get_supported_categories(self) -> List[str]:
        """Return list of object categories with defined functional rules."""
        return list(FUNCTIONAL_RULES.keys())

    def get_summary(self) -> dict:
        return {
            "clusters": {k: len(v) for k, v in self._clusters.items()},
            "wall_cells": len(self._wall_cells),
            "supported_categories": len(FUNCTIONAL_RULES),
        }
