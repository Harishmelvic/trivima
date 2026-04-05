"""
Auto-furnishing planner — VLM identifies furniture gaps and plans placement sequence.

From vlm_architecture_theory.md Ch4:
  - Examines cell grid semantic content
  - Identifies functional gaps ("living room has sofa but no coffee table")
  - Plans placement order: anchor → dependent → accent
  - Each placement updates validation fields before the next

The VLM provides the "what" and "where roughly." The validation system provides the "where exactly."
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FurnishingItem:
    """One item to be placed during auto-furnishing."""
    category: str        # "sofa", "coffee_table", "lamp", etc.
    priority: int        # 1=anchor, 2=dependent, 3=accent
    depends_on: str      # category that must be placed first ("" if none)
    reason: str          # why this item is needed


@dataclass
class FurnishingPlan:
    """Complete plan for auto-furnishing a room."""
    room_type: str
    items: List[FurnishingItem]
    total_items: int


class AutoFurnisher:
    """Plans and executes room auto-furnishing using VLM + validation fields.

    Usage:
        furnisher = AutoFurnisher(vlm)
        plan = furnisher.detect_gaps(image, grid_data, label_names)
        for item in plan.items:
            position = furnisher.place_item(item, grid_data, ...)
    """

    def __init__(self, vlm):
        self.vlm = vlm

    def detect_gaps(
        self,
        image: np.ndarray,
        grid_data: dict,
        label_names: Dict[int, str],
    ) -> FurnishingPlan:
        """Identify missing furniture and plan placement sequence.

        Args:
            image: room photograph
            grid_data: cell grid with existing furniture
            label_names: semantic labels present in the grid

        Returns:
            FurnishingPlan with ordered items to place
        """
        # Summarize existing furniture from cell grid labels
        existing = set()
        for cell in grid_data.values():
            label_idx = cell.get("label", 0)
            name = label_names.get(label_idx, "").lower()
            if name and name != "background":
                existing.add(name)

        existing_str = ", ".join(sorted(existing)) if existing else "nothing (empty room)"

        if self.vlm is not None and self.vlm._model is not None:
            return self._vlm_gap_detection(image, existing_str)
        else:
            return self._rule_based_gaps(existing)

    def _vlm_gap_detection(self, image: np.ndarray, existing_str: str) -> FurnishingPlan:
        """Use VLM to identify gaps and plan sequence."""
        prompt = (
            f"This room currently contains: {existing_str}.\n\n"
            f"What furniture is missing? List up to 5 items in placement order "
            f"(largest/anchor pieces first, accent pieces last). "
            f"For each item, state: category, priority (1=anchor, 2=dependent, 3=accent), "
            f"and what it depends on.\n\n"
            f"Format: category | priority | depends_on | reason"
        )

        response = self.vlm.query(image, prompt, max_tokens=300)
        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> FurnishingPlan:
        """Parse VLM response into a FurnishingPlan."""
        items = []
        lines = response.strip().split("\n")

        for line in lines:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                category = parts[0].lower().strip("- ").replace(" ", "_")
                try:
                    priority = int(parts[1])
                except (ValueError, IndexError):
                    priority = 2
                depends_on = parts[2].strip() if len(parts) > 2 else ""
                reason = parts[3].strip() if len(parts) > 3 else ""

                if category and len(category) > 1:
                    items.append(FurnishingItem(
                        category=category,
                        priority=max(1, min(3, priority)),
                        depends_on=depends_on if depends_on != "none" else "",
                        reason=reason,
                    ))

        # Sort by priority
        items.sort(key=lambda x: x.priority)

        # Determine room type from response
        room_type = "room"
        for rt in ["living room", "bedroom", "kitchen", "bathroom", "office", "dining room"]:
            if rt in response.lower():
                room_type = rt
                break

        return FurnishingPlan(room_type=room_type, items=items[:8], total_items=len(items))

    def _rule_based_gaps(self, existing: set) -> FurnishingPlan:
        """Fallback: rule-based gap detection without VLM."""
        items = []

        # Living room rules
        if "sofa" not in existing and "couch" not in existing:
            items.append(FurnishingItem("sofa", 1, "", "Primary seating"))
        if ("sofa" in existing or "couch" in existing) and "coffee_table" not in existing:
            items.append(FurnishingItem("coffee_table", 2, "sofa", "Pair with sofa"))
        if "lamp" not in existing and "floor_lamp" not in existing:
            items.append(FurnishingItem("floor_lamp", 3, "", "Lighting"))
        if "rug" not in existing:
            items.append(FurnishingItem("rug", 3, "", "Ground the seating area"))

        # Bedroom rules
        if "bed" in existing and "nightstand" not in existing:
            items.append(FurnishingItem("nightstand", 2, "bed", "Bedside surface"))

        return FurnishingPlan(room_type="room", items=items[:5], total_items=len(items))

    def classify_room(self, image: np.ndarray) -> str:
        """Classify the room type from the photograph.

        Called once per scene — drives shell extension and auto-furnishing rules.
        """
        if self.vlm is None or self.vlm._model is None:
            return "room"

        prompt = (
            "What type of room is this? Reply with ONLY one of: "
            "living room, bedroom, kitchen, bathroom, office, dining room, hallway, studio, other"
        )
        response = self.vlm.query(image, prompt, max_tokens=20, temperature=0.0)
        return response.strip().lower()
