"""
Aesthetic re-ranking — VLM selects best placement from validation candidates.

From vlm_architecture_theory.md Ch4:
  - Takes top 20-50 physically valid candidates from validation fields
  - Two modes: fast (logit scoring, ~200-500ms) and full (generative, ~2-5s)
  - Re-ranks by style coherence, cultural appropriateness, visual relationships
  - Respects validation scores — high-validation candidates generally rank higher

The VLM is advisory — if it fails, system falls back to geometric aesthetic scoring.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RankedCandidate:
    """A placement candidate after VLM re-ranking."""
    x: float
    y: float
    z: float
    validation_score: float
    vlm_score: float
    vlm_rank: int
    composite_score: float
    explanation: str = ""


class AestheticRanker:
    """Re-ranks placement candidates using VLM design intelligence.

    Usage:
        ranker = AestheticRanker(vlm)
        ranked = ranker.rank(image, candidates, "floor_lamp")
    """

    def __init__(self, vlm, validation_weight: float = 0.4, vlm_weight: float = 0.6):
        self.vlm = vlm
        self.validation_weight = validation_weight
        self.vlm_weight = vlm_weight

    def rank(
        self,
        image: np.ndarray,
        candidates: List[Dict],
        category: str,
        mode: str = "fast",
    ) -> List[RankedCandidate]:
        """Re-rank candidates using VLM + validation composite score.

        Args:
            image: room photograph
            candidates: list with keys: x, y, z, validation_score, description
            category: object category
            mode: "fast" or "full"

        Returns:
            Sorted list of RankedCandidate (best first)
        """
        if self.vlm is None or self.vlm._model is None:
            # Graceful degradation: return validation-only ranking
            return self._validation_only(candidates)

        try:
            scored = self.vlm.score_candidates(image, candidates, category, mode)
        except Exception as e:
            print(f"[AestheticRanker] VLM scoring failed: {e}, falling back to validation-only")
            return self._validation_only(candidates)

        results = []
        for c in scored:
            composite = (
                self.validation_weight * c.get("validation_score", 0) +
                self.vlm_weight * c.get("vlm_score", 0)
            )
            results.append(RankedCandidate(
                x=c.get("x", 0), y=c.get("y", 0), z=c.get("z", 0),
                validation_score=c.get("validation_score", 0),
                vlm_score=c.get("vlm_score", 0),
                vlm_rank=c.get("vlm_rank", 0),
                composite_score=composite,
                explanation=c.get("vlm_explanation", ""),
            ))

        return sorted(results, key=lambda r: r.composite_score, reverse=True)

    def _validation_only(self, candidates: List[Dict]) -> List[RankedCandidate]:
        """Fallback: rank by validation score only (no VLM)."""
        results = []
        sorted_candidates = sorted(candidates, key=lambda c: c.get("validation_score", 0), reverse=True)
        for i, c in enumerate(sorted_candidates):
            results.append(RankedCandidate(
                x=c.get("x", 0), y=c.get("y", 0), z=c.get("z", 0),
                validation_score=c.get("validation_score", 0),
                vlm_score=0.0,
                vlm_rank=i,
                composite_score=c.get("validation_score", 0),
                explanation="VLM unavailable — validation-only ranking",
            ))
        return results
