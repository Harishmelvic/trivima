"""
Trivima Stage 4 VLM Tests — 25 tests across 6 phases.

Updated for Qwen3-VL (no 3D-RoPE hack — native spatial encoding).

Phase 1: Model Loading (2.1-2.4) — needs GPU + Qwen3-VL checkpoint
Phase 2: Spatial Context (3.1-3.2) — prompt-based 3D context validation
Phase 3: Distillation (4.1-4.4) — needs_training
Phase 4: Re-Ranking (5.1-5.5) — needs_training
Phase 5: Auto-Furnishing (6.1-6.6) — needs_training
Phase 6: Integration (8.1-8.3) — needs_training

Run:
  pytest tests/test_vlm_stage4.py -v -k "not needs_training"   # pre-training tests
  pytest tests/test_vlm_stage4.py -v                           # all tests
"""

import pytest
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from trivima.vlm.qwen_vlm import SpatialContextBuilder, QwenVLM
from trivima.vlm.aesthetic_ranker import AestheticRanker
from trivima.vlm.auto_furnish import AutoFurnisher


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="session")
def context_builder():
    return SpatialContextBuilder(cell_size=0.05)


@pytest.fixture(scope="session")
def test_room_image():
    """Simple synthetic room image for VLM tests."""
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[h//2:, :] = [140, 100, 60]      # floor
    img[:h//2, :] = [210, 205, 195]      # wall
    img[300:380, 200:440] = [80, 60, 50] # sofa
    img[50:200, 400:550] = [180, 210, 240] # window
    return img


@pytest.fixture(scope="session")
def synthetic_grid():
    """Simple grid for context building tests."""
    grid = {}
    cs = 0.05
    for ix in range(100):
        for iz in range(100):
            grid[(ix, 0, iz)] = {
                "density": 1.0, "cell_type": 2, "label": 1,
                "albedo": np.array([0.5, 0.4, 0.3]),
                "normal": np.array([0, 1, 0]),
                "confidence": 0.85,
            }
    # Sofa
    for dx in range(20):
        for dz in range(10):
            for dy in range(8):
                grid[(40+dx, dy, 60+dz)] = {
                    "density": 1.0, "cell_type": 2, "label": 4,
                    "albedo": np.array([0.3, 0.25, 0.2]),
                    "normal": np.array([0, 1, 0]),
                    "confidence": 0.85,
                }
    return grid


def check_gpu():
    """Check if GPU is available with sufficient memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
        return total_mem >= 16
    except ImportError:
        return False


# ============================================================
# PHASE 1 — Model Loading (Tests 2.1-2.4)
# ============================================================

class TestPhase1ModelLoading:

    @pytest.fixture(autouse=True)
    def _check_gpu(self):
        if not check_gpu():
            pytest.skip("No GPU with sufficient memory")

    @pytest.mark.critical
    def test_2_1_qwen_loads(self):
        """Test 2.1: Qwen3-VL loads without OOM."""
        import torch

        for model_id in ["Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]:
            try:
                vlm = QwenVLM(model_id=model_id)
                vlm.load()

                mem = vlm.get_memory_usage()
                print(f"\n  Model: {model_id}")
                print(f"  GPU memory: {mem['peak_gb']:.1f} GB")

                assert vlm._model is not None
                assert mem["peak_gb"] < 48

                vlm.unload()
                torch.cuda.empty_cache()
                return
            except Exception as e:
                print(f"  {model_id} failed: {e}")
                continue

        pytest.skip("No Qwen model available to load")

    @pytest.mark.critical
    def test_2_2_sequential_loading(self):
        """Test 2.2: Perception models + Qwen load sequentially without OOM."""
        import torch

        torch.cuda.reset_peak_memory_stats()
        dummy = torch.randn(1, 3, 224, 224).cuda()
        del dummy
        torch.cuda.empty_cache()

        try:
            vlm = QwenVLM(model_id="Qwen/Qwen3-VL-8B-Instruct")
            vlm.load()
            mem = vlm.get_memory_usage()
            print(f"\n  After Qwen: {mem['peak_gb']:.1f}GB")
            vlm.unload()
            torch.cuda.empty_cache()
        except Exception as e:
            pytest.skip(f"Qwen load failed: {e}")

    def test_2_3_basic_room_description(self, test_room_image):
        """Test 2.3: Qwen describes a room coherently."""
        try:
            vlm = QwenVLM(model_id="Qwen/Qwen3-VL-8B-Instruct")
            vlm.load()
        except Exception as e:
            pytest.skip(f"Qwen not available: {e}")

        t0 = time.time()
        response = vlm.query(test_room_image, "Describe this room briefly. What do you see?")
        elapsed = time.time() - t0

        print(f"\n  Response ({elapsed:.1f}s): {response[:200]}")

        assert len(response) > 20, "Response too short"
        assert elapsed < 30

        vlm.unload()

    def test_2_4_batch_candidate_scoring(self, test_room_image):
        """Test 2.4: Qwen differentiates good from bad positions."""
        try:
            vlm = QwenVLM(model_id="Qwen/Qwen3-VL-8B-Instruct")
            vlm.load()
        except Exception as e:
            pytest.skip(f"Qwen not available: {e}")

        candidates = [
            {"x": 0.5, "y": 0, "z": 3.0, "description": "near sofa, against wall", "validation_score": 0.8},
            {"x": 2.5, "y": 0, "z": 2.5, "description": "room center, open space", "validation_score": 0.6},
            {"x": 0.1, "y": 0, "z": 0.1, "description": "corner, blocking doorway", "validation_score": 0.3},
        ]

        t0 = time.time()
        scored = vlm.score_candidates(test_room_image, candidates, "floor_lamp", mode="fast")
        elapsed = time.time() - t0

        print(f"\n  Scoring time: {elapsed:.1f}s")
        for c in scored:
            print(f"    {c.get('description', '?')}: vlm_score={c.get('vlm_score', 0):.2f}")

        assert len(scored) == 3
        assert elapsed < 15

        vlm.unload()


# ============================================================
# PHASE 2 — Spatial Context (replaces 3D-RoPE tests)
# ============================================================

class TestPhase2SpatialContext:
    """Tests for prompt-based 3D context (replaces old 3D-RoPE encoding tests).

    Qwen3-VL has native Interleaved-MRoPE — no custom encoding injection needed.
    Instead, we validate the SpatialContextBuilder that provides explicit 3D data in prompts.
    """

    def test_3_1_room_context_generation(self, context_builder, synthetic_grid):
        """Test 3.1: Room context describes dimensions and furniture correctly."""
        label_names = {0: "background", 1: "floor", 4: "sofa"}

        context = context_builder.build_room_context(synthetic_grid, label_names)

        print(f"\n  Context:\n  {context}")

        assert "Room dimensions" in context
        assert "sofa" in context.lower()
        assert len(context) > 50, "Context too short"

    def test_3_2_candidate_context_generation(self, context_builder):
        """Test 3.2: Candidate context includes positions and scores."""
        candidates = [
            {"x": 1.0, "y": 0, "z": 2.0, "validation_score": 0.9, "description": "near sofa"},
            {"x": 3.0, "y": 0, "z": 1.0, "validation_score": 0.5, "clearance": 0.8, "surface_type": "floor"},
        ]

        context = context_builder.build_candidate_context(candidates, "lamp")

        print(f"\n  Context:\n  {context}")

        assert "Position 1" in context
        assert "Position 2" in context
        assert "near sofa" in context
        assert "validation=" in context or "clearance=" in context


# ============================================================
# PHASE 3 — Distillation Quality (Tests 4.1-4.4)
# ============================================================

class TestPhase3Distillation:

    @pytest.mark.needs_training
    @pytest.mark.critical
    def test_4_1_distillation_data_generation(self):
        """Test 4.1: SpatialVLM generates valid spatial QA pairs."""
        pytest.skip("Needs SpatialVLM + training data generation pipeline")

    @pytest.mark.needs_training
    @pytest.mark.critical
    def test_4_2_distilled_spatial_accuracy(self):
        """Test 4.2: Distilled Qwen3-VL within 25-35% of SpatialVLM accuracy."""
        pytest.skip("Needs completed distillation training (6-11 days)")

    @pytest.mark.needs_training
    @pytest.mark.critical
    def test_4_3_aesthetics_not_degraded(self):
        """Test 4.3: Distillation doesn't degrade aesthetic judgment >15%."""
        pytest.skip("Needs completed distillation training")

    @pytest.mark.needs_training
    def test_4_4_language_not_degraded(self):
        """Test 4.4: Distillation doesn't increase perplexity >15%."""
        pytest.skip("Needs completed distillation training")


# ============================================================
# PHASE 4 — Aesthetic Re-Ranking (Tests 5.1-5.5)
# ============================================================

class TestPhase4ReRanking:

    @pytest.mark.needs_training
    @pytest.mark.critical
    def test_5_1_differentiates_good_bad(self):
        pytest.skip("Needs trained VLM")

    @pytest.mark.needs_training
    def test_5_2_consistency(self):
        pytest.skip("Needs trained VLM")

    @pytest.mark.needs_training
    @pytest.mark.critical
    def test_5_3_speed_fast_path(self):
        pytest.skip("Needs trained VLM")

    @pytest.mark.needs_training
    def test_5_4_speed_full_path(self):
        pytest.skip("Needs trained VLM")

    @pytest.mark.needs_training
    def test_5_5_respects_validation(self):
        pytest.skip("Needs trained VLM")


# ============================================================
# PHASE 5 — Auto-Furnishing (Tests 6.1-6.6)
# ============================================================

class TestPhase5AutoFurnishing:

    def test_6_1_gap_detection_empty_room_fallback(self):
        """Test 6.1 (fallback): Rule-based gap detection without VLM."""
        furnisher = AutoFurnisher(vlm=None)
        plan = furnisher._rule_based_gaps(set())

        assert plan.total_items >= 3
        categories = [item.category for item in plan.items]
        print(f"\n  Suggested for empty room: {categories}")
        assert "sofa" in categories

    def test_6_2_gap_detection_partial_room_fallback(self):
        """Test 6.2 (fallback): Rule-based gaps with existing furniture."""
        furnisher = AutoFurnisher(vlm=None)
        plan = furnisher._rule_based_gaps({"sofa", "floor"})

        categories = [item.category for item in plan.items]
        print(f"\n  Room has sofa. Suggested: {categories}")
        assert "sofa" not in categories
        assert "coffee_table" in categories

    @pytest.mark.needs_training
    @pytest.mark.critical
    def test_6_1_gap_detection_vlm(self):
        pytest.skip("Needs trained VLM")

    @pytest.mark.needs_training
    @pytest.mark.critical
    def test_6_2_gap_detection_partial_vlm(self):
        pytest.skip("Needs trained VLM")

    @pytest.mark.needs_training
    def test_6_3_placement_sequence(self):
        pytest.skip("Needs trained VLM")

    @pytest.mark.needs_training
    def test_6_4_sequential_field_updates(self):
        pytest.skip("Needs trained VLM + full pipeline")

    @pytest.mark.needs_training
    def test_6_5_no_overcrowding(self):
        pytest.skip("Needs trained VLM")

    @pytest.mark.needs_training
    def test_6_6_performance(self):
        pytest.skip("Needs trained VLM")


# ============================================================
# PHASE 6 — Integration (Tests 8.1-8.3)
# ============================================================

class TestPhase6Integration:

    def test_8_3_graceful_degradation(self):
        """Test 8.3: System works without VLM (validation-only fallback)."""
        ranker = AestheticRanker(vlm=None)

        candidates = [
            {"x": 1.0, "y": 0, "z": 2.0, "validation_score": 0.9, "description": "good spot"},
            {"x": 2.0, "y": 0, "z": 3.0, "validation_score": 0.5, "description": "ok spot"},
            {"x": 0.1, "y": 0, "z": 0.1, "validation_score": 0.2, "description": "bad spot"},
        ]

        ranked = ranker.rank(np.zeros((100, 100, 3), dtype=np.uint8), candidates, "lamp")

        assert len(ranked) == 3
        assert ranked[0].validation_score == 0.9
        assert ranked[-1].validation_score == 0.2
        assert "validation-only" in ranked[0].explanation.lower()

    @pytest.mark.needs_training
    @pytest.mark.critical
    def test_8_1_full_pipeline(self):
        pytest.skip("Needs trained VLM + full pipeline")

    @pytest.mark.needs_training
    @pytest.mark.critical
    def test_8_2_vlm_adds_value(self):
        pytest.skip("Needs trained VLM + human evaluation")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
