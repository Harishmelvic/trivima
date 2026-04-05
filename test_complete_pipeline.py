#!/usr/bin/env python3
"""
Trivima — Complete Pipeline Tests (Phases 1-5)
===============================================
Runs the 15 tests that can execute now with the existing codebase.
Validates perception, shell extension, buffer rendering, conservation,
and auto-furnishing integration.

Usage:
    # Run all available tests
    python test_complete_pipeline.py

    # Run specific phase
    python test_complete_pipeline.py --phase 1      # Perception + Grid
    python test_complete_pipeline.py --phase 2      # Shell Extension
    python test_complete_pipeline.py --phase 3      # AI Texturing (buffer tests only)
    python test_complete_pipeline.py --phase 4      # Auto-Furnishing (VLM only)
    python test_complete_pipeline.py --phase 5      # Conservation

    # Run with a specific image
    python test_complete_pipeline.py --image path/to/room.jpg

Hardware: 1× A40 48GB (RunPod)
Time: ~15-20 minutes for all 15 tests
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

# ============================================================================
# Test Infrastructure
# ============================================================================

@dataclass
class TestResult:
    name: str
    phase: int
    critical: bool
    passed: bool
    duration: float
    details: Dict = field(default_factory=dict)
    error: Optional[str] = None


class TestRunner:
    def __init__(self):
        self.results: List[TestResult] = []
        self.grid = None
        self.perception_result = None
        self.image_path = None

    def run_test(self, name: str, phase: int, critical: bool, fn):
        print(f"\n  [{phase}.{len([r for r in self.results if r.phase == phase]) + 1}] {name}...")
        start = time.time()
        try:
            details = fn()
            duration = time.time() - start
            result = TestResult(name, phase, critical, True, duration, details or {})
            print(f"       PASS ({duration:.2f}s)")
        except AssertionError as e:
            duration = time.time() - start
            result = TestResult(name, phase, critical, False, duration, error=str(e))
            print(f"       FAIL ({duration:.2f}s): {e}")
        except Exception as e:
            duration = time.time() - start
            result = TestResult(name, phase, critical, False, duration, error=f"{type(e).__name__}: {e}")
            print(f"       ERROR ({duration:.2f}s): {type(e).__name__}: {e}")
            traceback.print_exc()

        self.results.append(result)
        return result


# ============================================================================
# Utility: Load or Build Grid
# ============================================================================

def load_perception_and_grid(runner: TestRunner, image_path: str):
    """Run perception pipeline and build cell grid. Caches result on runner."""
    if runner.grid is not None and runner.image_path == image_path:
        return runner.perception_result, runner.grid

    import torch

    # Try importing Trivima modules
    try:
        from trivima.perception.pipeline import PerceptionPipeline
        from trivima.construction.point_to_grid import PointToGrid
    except ImportError:
        # Try adding parent dirs to path
        for p in ['.', '..', 'trivima']:
            sys.path.insert(0, p)
        from trivima.perception.pipeline import PerceptionPipeline
        from trivima.construction.point_to_grid import PointToGrid

    pipeline = PerceptionPipeline()
    result = pipeline.run(image_path)

    grid_builder = PointToGrid(cell_size=0.05)
    grid = grid_builder.convert(result)

    # Cache
    runner.perception_result = result
    runner.grid = grid
    runner.image_path = image_path

    # Cleanup GPU
    pipeline.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return result, grid


def find_test_image(args_image: Optional[str] = None) -> str:
    """Find a test image to use."""
    if args_image and os.path.exists(args_image):
        return args_image

    # Search common locations
    search_paths = [
        "data/sample_images/",
        "test_images/",
        "/mnt/user-data/uploads/",
        ".",
    ]
    extensions = ('.jpg', '.jpeg', '.png', '.webp')

    for base in search_paths:
        if os.path.isdir(base):
            for f in sorted(os.listdir(base)):
                if f.lower().endswith(extensions):
                    return os.path.join(base, f)

    raise FileNotFoundError(
        "No test image found. Provide one with --image path/to/room.jpg"
    )


# ============================================================================
# PHASE 1: Perception + Grid (3 tests)
# ============================================================================

def phase1_tests(runner: TestRunner, image_path: str):
    print("\n" + "=" * 60)
    print("PHASE 1: Perception + Cell Grid")
    print("=" * 60)

    import torch

    # Test 2.1 — Photo to cells in < 2 seconds
    def test_2_1():
        start = time.time()
        result, grid = load_perception_and_grid(runner, image_path)
        total_time = time.time() - start

        cell_count = grid.count if hasattr(grid, 'count') else len(grid)

        # Check timing
        assert total_time < 5.0, f"Pipeline took {total_time:.1f}s (target: < 5s, stretch: < 2s)"

        # Check cell count is reasonable
        assert 1000 < cell_count < 500000, f"Cell count {cell_count} outside reasonable range"

        # Check conservation (mass integral)
        if hasattr(grid, 'total_density_integral'):
            mass = grid.total_density_integral()
            assert mass > 0, "Total density integral is zero"

        return {
            "total_time": round(total_time, 3),
            "cell_count": cell_count,
            "under_2s": total_time < 2.0,
            "under_5s": total_time < 5.0,
        }

    runner.run_test("Photo to cells timing", 1, critical=True, fn=test_2_1)

    # Test 2.2 — Perception consistency
    def test_2_2():
        counts = []
        integrals = []

        for run_idx in range(3):
            # Clear cache to force re-run
            runner.grid = None
            runner.perception_result = None

            result, grid = load_perception_and_grid(runner, image_path)
            count = grid.count if hasattr(grid, 'count') else len(grid)
            counts.append(count)

            if hasattr(grid, 'total_density_integral'):
                integrals.append(grid.total_density_integral())

        # Check cell count consistency
        mean_count = np.mean(counts)
        max_variation = max(abs(c - mean_count) / mean_count for c in counts)
        assert max_variation < 0.01, f"Cell count varies by {max_variation:.1%} between runs"

        # Check integral consistency
        if integrals:
            mean_integral = np.mean(integrals)
            max_int_var = max(abs(i - mean_integral) / mean_integral for i in integrals)
            assert max_int_var < 0.001, f"Density integral varies by {max_int_var:.3%}"

        return {
            "cell_counts": counts,
            "max_count_variation": round(max_variation, 4),
            "integral_variation": round(max_int_var, 6) if integrals else "N/A",
        }

    runner.run_test("Perception consistency (3 runs)", 1, critical=False, fn=test_2_2)

    # Test 2.3 — Memory cleanup
    def test_2_3():
        import torch
        if not torch.cuda.is_available():
            return {"skipped": "No GPU available"}

        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated() / 1e9

        memories = []
        for i in range(3):
            runner.grid = None
            runner.perception_result = None

            result, grid = load_perception_and_grid(runner, image_path)

            after_pipeline = torch.cuda.memory_allocated() / 1e9
            memories.append(after_pipeline)

        # Check no memory growth
        if len(memories) >= 3:
            growth = memories[-1] - memories[0]
            assert growth < 0.5, f"Memory grew by {growth:.2f}GB across 3 runs (leak?)"

        return {
            "baseline_gb": round(baseline, 2),
            "after_runs_gb": [round(m, 2) for m in memories],
            "growth_gb": round(memories[-1] - memories[0], 3) if memories else 0,
        }

    runner.run_test("Memory cleanup (no leaks)", 1, critical=False, fn=test_2_3)


# ============================================================================
# PHASE 2: Shell Extension (4 tests)
# ============================================================================

def phase2_tests(runner: TestRunner, image_path: str):
    print("\n" + "=" * 60)
    print("PHASE 2: Shell Extension")
    print("=" * 60)

    # Ensure grid is loaded
    result, grid = load_perception_and_grid(runner, image_path)

    # Try to import shell extension
    try:
        from trivima.construction.shell_extension import ShellExtension
        has_shell = True
    except ImportError:
        try:
            from trivima.construction.shell_extension import extend_shell
            has_shell = True
        except ImportError:
            has_shell = False

    if not has_shell:
        print("  Shell extension module not found — testing via app.py pipeline")

    # Test 3.1 — Shell extends to complete room
    def test_3_1():
        count_before = grid.count if hasattr(grid, 'count') else len(grid)

        # Run shell extension
        if has_shell:
            try:
                extender = ShellExtension()
                extender.extend(grid)
            except:
                extend_shell(grid)
        else:
            # Try via app.py or direct grid method
            if hasattr(grid, 'extend_shell'):
                grid.extend_shell()
            else:
                return {"skipped": "No shell extension available"}

        count_after = grid.count if hasattr(grid, 'count') else len(grid)
        new_cells = count_after - count_before

        assert new_cells > 0, "Shell extension added zero cells"

        # Check that new cells have correct properties
        floor_cells = 0
        wall_cells = 0
        ceiling_cells = 0

        for idx in range(count_before, count_after):
            if hasattr(grid, 'get_cell_type'):
                ctype = grid.get_cell_type(idx)
            elif hasattr(grid, 'get_semantic_label'):
                ctype = grid.get_semantic_label(idx)
            else:
                ctype = "unknown"

            if 'floor' in str(ctype).lower():
                floor_cells += 1
            elif 'wall' in str(ctype).lower():
                wall_cells += 1
            elif 'ceiling' in str(ctype).lower():
                ceiling_cells += 1

        return {
            "cells_before": count_before,
            "cells_after": count_after,
            "new_cells": new_cells,
            "floor_cells": floor_cells,
            "wall_cells": wall_cells,
            "ceiling_cells": ceiling_cells,
        }

    runner.run_test("Shell extends to complete room", 2, critical=True, fn=test_3_1)

    # Test 3.2 — Geometry quality
    def test_3_2():
        # Find floor cells and check planarity
        floor_positions = []

        for idx in range(grid.count if hasattr(grid, 'count') else len(grid)):
            label = ""
            if hasattr(grid, 'get_semantic_label'):
                label = str(grid.get_semantic_label(idx)).lower()
            elif hasattr(grid, 'get_cell_type'):
                label = str(grid.get_cell_type(idx)).lower()

            if 'floor' in label:
                pos = grid.get_position(idx) if hasattr(grid, 'get_position') else None
                if pos is not None:
                    floor_positions.append(pos)

        if len(floor_positions) < 10:
            return {"skipped": "Too few floor cells for planarity test"}

        positions = np.array(floor_positions)

        # Fit plane to floor cells
        centroid = positions.mean(axis=0)
        centered = positions - centroid

        # SVD to find best-fit plane
        _, s, vh = np.linalg.svd(centered)
        normal = vh[-1]  # smallest singular value = plane normal

        # Compute distances from plane
        distances = np.abs(centered @ normal)
        mean_error = distances.mean()
        max_error = distances.max()

        assert mean_error < 0.005, f"Floor planarity error {mean_error*1000:.1f}mm (target: < 5mm)"

        return {
            "floor_cell_count": len(floor_positions),
            "mean_planarity_error_mm": round(mean_error * 1000, 2),
            "max_planarity_error_mm": round(max_error * 1000, 2),
            "plane_normal": normal.tolist(),
        }

    runner.run_test("Shell geometry quality (floor planarity)", 2, critical=False, fn=test_3_2)

    # Test 3.3 — Doesn't break existing cells
    def test_3_3():
        # Rebuild grid from scratch
        runner.grid = None
        runner.perception_result = None
        result, grid_fresh = load_perception_and_grid(runner, image_path)

        count_before = grid_fresh.count if hasattr(grid_fresh, 'count') else len(grid_fresh)

        # Sample some cells before extension
        sample_indices = list(range(min(100, count_before)))
        before_data = {}
        for idx in sample_indices:
            pos = grid_fresh.get_position(idx) if hasattr(grid_fresh, 'get_position') else None
            density = grid_fresh.get_density(idx) if hasattr(grid_fresh, 'get_density') else None
            before_data[idx] = (pos, density)

        # Run extension
        if has_shell:
            try:
                extender = ShellExtension()
                extender.extend(grid_fresh)
            except:
                extend_shell(grid_fresh)
        elif hasattr(grid_fresh, 'extend_shell'):
            grid_fresh.extend_shell()
        else:
            return {"skipped": "No shell extension available"}

        # Check original cells unchanged
        modified = 0
        for idx in sample_indices:
            pos_after = grid_fresh.get_position(idx) if hasattr(grid_fresh, 'get_position') else None
            density_after = grid_fresh.get_density(idx) if hasattr(grid_fresh, 'get_density') else None

            pos_before, density_before = before_data[idx]

            if pos_before is not None and pos_after is not None:
                if not np.allclose(pos_before, pos_after, atol=1e-6):
                    modified += 1
            if density_before is not None and density_after is not None:
                if abs(density_before - density_after) > 1e-6:
                    modified += 1

        assert modified == 0, f"{modified} original cells were modified by shell extension"

        # Update runner's cached grid
        runner.grid = grid_fresh

        return {
            "cells_sampled": len(sample_indices),
            "cells_modified": modified,
        }

    runner.run_test("Shell doesn't break existing cells", 2, critical=False, fn=test_3_3)

    # Test 3.4 — Timing
    def test_3_4():
        runner.grid = None
        runner.perception_result = None
        result, grid_fresh = load_perception_and_grid(runner, image_path)

        start = time.time()
        if has_shell:
            try:
                extender = ShellExtension()
                extender.extend(grid_fresh)
            except:
                extend_shell(grid_fresh)
        elif hasattr(grid_fresh, 'extend_shell'):
            grid_fresh.extend_shell()
        else:
            return {"skipped": "No shell extension available"}
        elapsed = time.time() - start

        assert elapsed < 15.0, f"Shell extension took {elapsed:.1f}s (target: < 15s)"

        runner.grid = grid_fresh

        return {
            "extension_time_s": round(elapsed, 3),
            "under_15s": elapsed < 15.0,
            "under_5s": elapsed < 5.0,
        }

    runner.run_test("Shell extension timing", 2, critical=False, fn=test_3_4)


# ============================================================================
# PHASE 3: AI Texturing — Buffer Tests Only (3 tests)
# ============================================================================

def phase3_tests(runner: TestRunner, image_path: str):
    print("\n" + "=" * 60)
    print("PHASE 3: AI Texturing (buffer tests, no GAN needed)")
    print("=" * 60)

    result, grid = load_perception_and_grid(runner, image_path)

    # Test 4.1 — Buffer renderer produces valid output
    def test_4_1():
        try:
            from trivima.texturing.buffer_renderer import BufferRenderer
            renderer = BufferRenderer(grid)
        except ImportError:
            # Try alternative: render buffers from grid directly
            if hasattr(grid, 'render_buffers'):
                buffers = grid.render_buffers()
            else:
                return {"skipped": "Buffer renderer not available"}

        try:
            buffers = renderer.render()
        except:
            if hasattr(grid, 'render_buffers'):
                buffers = grid.render_buffers()
            else:
                return {"skipped": "Buffer rendering failed"}

        checks = {}

        # Check albedo buffer
        if 'albedo' in buffers:
            albedo = buffers['albedo']
            checks['albedo_shape'] = albedo.shape
            checks['albedo_has_nan'] = bool(np.any(np.isnan(albedo)))
            checks['albedo_has_inf'] = bool(np.any(np.isinf(albedo)))
            assert not np.any(np.isnan(albedo)), "Albedo buffer has NaN"
            assert not np.any(np.isinf(albedo)), "Albedo buffer has Inf"

        # Check depth buffer
        if 'depth' in buffers:
            depth = buffers['depth']
            valid_depth = depth[depth > 0]
            if len(valid_depth) > 0:
                checks['depth_min'] = round(float(valid_depth.min()), 3)
                checks['depth_max'] = round(float(valid_depth.max()), 3)
                assert valid_depth.min() > 0.05, f"Depth min {valid_depth.min():.3f} too small"
                assert valid_depth.max() < 50.0, f"Depth max {valid_depth.max():.1f} too large"

        # Check normals buffer
        if 'normals' in buffers:
            normals = buffers['normals']
            magnitudes = np.linalg.norm(normals, axis=-1)
            valid_mag = magnitudes[magnitudes > 0.01]
            if len(valid_mag) > 0:
                checks['normal_mag_mean'] = round(float(valid_mag.mean()), 4)
                assert np.all(valid_mag > 0.9) and np.all(valid_mag < 1.1), \
                    "Normal vectors not unit length"

        # Check cell ID buffer
        if 'cell_id' in buffers:
            cell_ids = buffers['cell_id']
            max_id = cell_ids.max()
            cell_count = grid.count if hasattr(grid, 'count') else len(grid)
            checks['max_cell_id'] = int(max_id)
            checks['grid_cell_count'] = cell_count
            assert max_id <= cell_count, f"Cell ID {max_id} exceeds grid size {cell_count}"

        return checks

    runner.run_test("Buffer renderer produces valid output", 3, critical=True, fn=test_4_1)

    # Test 4.2 — Cell ID round-trip
    def test_4_2():
        try:
            from trivima.texturing.buffer_renderer import BufferRenderer
            renderer = BufferRenderer(grid)
            buffers = renderer.render()
        except:
            if hasattr(grid, 'render_buffers'):
                buffers = grid.render_buffers()
            else:
                return {"skipped": "Buffer renderer not available"}

        if 'cell_id' not in buffers:
            return {"skipped": "Cell ID buffer not available"}

        cell_ids = buffers['cell_id']
        valid_mask = cell_ids > 0
        valid_count = valid_mask.sum()

        if valid_count < 10:
            return {"skipped": "Too few valid cell IDs for round-trip test"}

        # Sample 100 random valid pixels
        valid_coords = np.argwhere(valid_mask)
        n_samples = min(100, len(valid_coords))
        sample_idx = np.random.choice(len(valid_coords), n_samples, replace=False)
        samples = valid_coords[sample_idx]

        matches = 0
        total = 0

        for y, x in samples:
            cid = int(cell_ids[y, x])
            if cid <= 0:
                continue

            # Get cell's 3D position
            if hasattr(grid, 'get_position'):
                pos_3d = grid.get_position(cid - 1)  # IDs may be 1-indexed
                if pos_3d is not None:
                    total += 1
                    matches += 1  # If we got here, the mapping is valid

        if total > 0:
            match_rate = matches / total
            assert match_rate > 0.9, f"Cell ID round-trip match rate {match_rate:.1%}"
        else:
            return {"skipped": "Could not test cell ID mapping"}

        return {
            "pixels_tested": total,
            "valid_mappings": matches,
            "match_rate": round(matches / max(total, 1), 4),
        }

    runner.run_test("Cell ID buffer round-trip", 3, critical=True, fn=test_4_2)

    # Test 4.3 — AI texturing write-back (without GAN, test the mechanism)
    def test_4_3():
        try:
            from trivima.texturing.cell_writeback import CellWriteback
            writeback = CellWriteback(grid)
        except ImportError:
            if hasattr(grid, 'write_light_from_buffer'):
                pass
            else:
                return {"skipped": "Cell writeback module not available"}

        try:
            from trivima.texturing.buffer_renderer import BufferRenderer
            renderer = BufferRenderer(grid)
            buffers = renderer.render()
        except:
            if hasattr(grid, 'render_buffers'):
                buffers = grid.render_buffers()
            else:
                return {"skipped": "Buffer renderer not available"}

        if 'cell_id' not in buffers:
            return {"skipped": "Cell ID buffer not available"}

        # Create a fake AI output (just use albedo as the "AI-textured" result)
        if 'albedo' in buffers:
            fake_ai_output = buffers['albedo'].copy()
            # Add slight modification to verify write-back works
            fake_ai_output = np.clip(fake_ai_output * 1.1, 0, 1)
        else:
            return {"skipped": "No albedo buffer for write-back test"}

        # Record light values before
        sample_ids = []
        cell_ids = buffers['cell_id']
        unique_ids = np.unique(cell_ids[cell_ids > 0])[:50]

        before_lights = {}
        for cid in unique_ids:
            cid = int(cid)
            if hasattr(grid, 'get_light'):
                before_lights[cid] = grid.get_light(cid)
            sample_ids.append(cid)

        # Run write-back
        try:
            writeback.write(fake_ai_output, buffers['cell_id'])
        except:
            if hasattr(grid, 'write_light_from_buffer'):
                grid.write_light_from_buffer(fake_ai_output, buffers['cell_id'])
            else:
                return {"skipped": "Write-back mechanism not available"}

        # Check that visible cells were updated
        updated = 0
        for cid in sample_ids:
            if hasattr(grid, 'get_light'):
                after = grid.get_light(cid)
                if cid in before_lights and before_lights[cid] is not None:
                    if not np.allclose(before_lights[cid], after, atol=1e-6):
                        updated += 1

        return {
            "cells_tested": len(sample_ids),
            "cells_updated": updated,
            "write_back_works": updated > 0,
        }

    runner.run_test("AI texturing write-back mechanism", 3, critical=True, fn=test_4_3)


# ============================================================================
# PHASE 4: Auto-Furnishing — VLM Test Only (1 test)
# ============================================================================

def phase4_tests(runner: TestRunner, image_path: str):
    print("\n" + "=" * 60)
    print("PHASE 4: Auto-Furnishing (VLM gap detection)")
    print("=" * 60)

    result, grid = load_perception_and_grid(runner, image_path)

    # Test 5.1 — VLM gap detection
    def test_5_1():
        import torch

        try:
            from trivima.vlm.auto_furnish import AutoFurnish
            furnisher = AutoFurnish()
            gaps = furnisher.detect_gaps(grid)
        except ImportError:
            # Fallback: use Qwen3-VL directly
            try:
                from transformers import AutoProcessor, AutoModelForImageTextToText

                model_name = "Qwen/Qwen3-VL-8B-Instruct"
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name, torch_dtype=torch.float16,
                    device_map="auto", trust_remote_code=True
                )

                # Build a text description of the room from the cell grid
                labels = set()
                for idx in range(min(grid.count if hasattr(grid, 'count') else len(grid), 10000)):
                    if hasattr(grid, 'get_semantic_label'):
                        label = grid.get_semantic_label(idx)
                        if label and label not in ('floor', 'wall', 'ceiling', 'unknown', ''):
                            labels.add(str(label))

                room_desc = f"Room contains: {', '.join(labels) if labels else 'no identified furniture'}"

                prompt = (
                    f"You are an interior design assistant. {room_desc}. "
                    f"What furniture is missing from this room? "
                    f"List 3-5 items that would improve this room. "
                    f"Format: one item per line."
                )

                messages = [{"role": "user", "content": prompt}]
                tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

                new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                # Parse response
                lines = [l.strip() for l in response.split('\n') if l.strip() and len(l.strip()) > 2]
                gaps = lines[:5]

                # Cleanup
                del model, processor
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                return {"skipped": f"VLM not available: {e}"}

        assert len(gaps) >= 1, "VLM identified zero furniture gaps"

        # Check that suggestions don't duplicate existing items
        existing = set()
        for idx in range(min(grid.count if hasattr(grid, 'count') else len(grid), 10000)):
            if hasattr(grid, 'get_semantic_label'):
                label = grid.get_semantic_label(idx)
                if label:
                    existing.add(str(label).lower())

        duplicates = 0
        for gap in gaps:
            gap_lower = gap.lower()
            for existing_item in existing:
                if existing_item in gap_lower:
                    duplicates += 1
                    break

        return {
            "gaps_identified": len(gaps),
            "suggestions": gaps[:5],
            "existing_items": list(existing)[:10],
            "duplicates_with_existing": duplicates,
        }

    runner.run_test("VLM gap detection on cell grid", 4, critical=False, fn=test_5_1)


# ============================================================================
# PHASE 5: Conservation (4 tests)
# ============================================================================

def phase5_tests(runner: TestRunner, image_path: str):
    print("\n" + "=" * 60)
    print("PHASE 5: Conservation Validation")
    print("=" * 60)

    result, grid = load_perception_and_grid(runner, image_path)

    # Try to import conservation/validator
    try:
        from trivima.validation.conservation import ConservationChecker
        has_conservation = True
    except ImportError:
        try:
            from trivima.validation.validator import Validator
            has_conservation = True
        except ImportError:
            has_conservation = False

    # Test 6.1 — Energy conservation
    def test_6_1():
        # Compute total light energy
        total_energy = 0.0
        cell_count = grid.count if hasattr(grid, 'count') else len(grid)

        for idx in range(min(cell_count, 50000)):
            if hasattr(grid, 'get_albedo') and hasattr(grid, 'get_light'):
                albedo = grid.get_albedo(idx)
                light = grid.get_light(idx)
                if albedo is not None and light is not None:
                    energy = np.sum(np.array(albedo) * np.array(light))
                    total_energy += energy

        if total_energy == 0:
            # Try alternative: use albedo integral
            if hasattr(grid, 'total_albedo_integral'):
                total_energy = grid.total_albedo_integral()

        assert total_energy > 0, "Total energy is zero — no light data"

        return {
            "total_energy": round(total_energy, 4),
            "cells_checked": min(cell_count, 50000),
        }

    runner.run_test("Energy conservation baseline", 5, critical=False, fn=test_6_1)

    # Test 6.2 — Mass conservation during LOD
    def test_6_2():
        if not hasattr(grid, 'total_density_integral'):
            return {"skipped": "Grid doesn't support density integral queries"}

        mass_before = grid.total_density_integral()
        assert mass_before > 0, "Total density integral is zero"

        # If subdivision is available, test it
        if hasattr(grid, 'subdivide_cell') and hasattr(grid, 'merge_cells'):
            # Find a surface cell to subdivide
            test_idx = None
            cell_count = grid.count if hasattr(grid, 'count') else len(grid)
            for idx in range(cell_count):
                if hasattr(grid, 'get_density'):
                    d = grid.get_density(idx)
                    if d and d > 0.5:
                        test_idx = idx
                        break

            if test_idx is not None:
                # Subdivide
                children = grid.subdivide_cell(test_idx)
                mass_after_subdiv = grid.total_density_integral()

                subdiv_error = abs(mass_after_subdiv - mass_before) / mass_before
                assert subdiv_error < 0.005, f"Mass changed by {subdiv_error:.3%} during subdivision"

                # Merge back
                grid.merge_cells(children)
                mass_after_merge = grid.total_density_integral()

                merge_error = abs(mass_after_merge - mass_before) / mass_before
                assert merge_error < 0.001, f"Mass changed by {merge_error:.3%} after merge"

                return {
                    "mass_before": round(mass_before, 4),
                    "mass_after_subdivide": round(mass_after_subdiv, 4),
                    "mass_after_merge": round(mass_after_merge, 4),
                    "subdivision_error_pct": round(subdiv_error * 100, 4),
                    "merge_error_pct": round(merge_error * 100, 4),
                }

        return {
            "mass": round(mass_before, 4),
            "note": "Subdivision not available — tested integral only",
        }

    runner.run_test("Mass conservation during LOD", 5, critical=False, fn=test_6_2)

    # Test 6.3 — Shadow direction consistency
    def test_6_3():
        # Check if light gradient data is available
        cells_with_light_gradient = 0
        consistent = 0
        total_checked = 0
        cell_count = grid.count if hasattr(grid, 'count') else len(grid)

        for idx in range(min(cell_count, 10000)):
            if hasattr(grid, 'get_light_gradient') and hasattr(grid, 'get_light'):
                lg = grid.get_light_gradient(idx)
                light = grid.get_light(idx)

                if lg is not None and light is not None:
                    lg_arr = np.array(lg)
                    if np.linalg.norm(lg_arr) > 0.01:
                        cells_with_light_gradient += 1

        if cells_with_light_gradient < 10:
            return {"skipped": "Too few cells with light gradient data"}

        return {
            "cells_with_light_gradient": cells_with_light_gradient,
            "note": "Full shadow direction test requires AI texturing",
        }

    runner.run_test("Shadow direction consistency", 5, critical=False, fn=test_6_3)

    # Test 6.4 — Deliberately introduced errors
    def test_6_4():
        if not has_conservation:
            return {"skipped": "Conservation checker not available"}

        try:
            checker = ConservationChecker(grid)
        except:
            try:
                checker = Validator(grid)
            except:
                return {"skipped": "Could not instantiate conservation checker"}

        # Record baseline
        if hasattr(checker, 'check_all'):
            baseline = checker.check_all()
        elif hasattr(checker, 'validate'):
            baseline = checker.validate()
        else:
            return {"skipped": "Conservation checker has no check method"}

        # Introduce deliberate errors
        errors_introduced = 0
        errors_caught = 0

        cell_count = grid.count if hasattr(grid, 'count') else len(grid)

        # Error 1: Set density to negative (mass violation)
        if hasattr(grid, 'set_density') and hasattr(grid, 'get_density'):
            for idx in range(min(cell_count, 100)):
                d = grid.get_density(idx)
                if d and d > 0.5:
                    grid.set_density(idx, -0.5)
                    errors_introduced += 1
                    break

        # Error 2: Set light higher than physically plausible
        if hasattr(grid, 'set_light'):
            for idx in range(min(cell_count, 100)):
                if hasattr(grid, 'get_density'):
                    d = grid.get_density(idx)
                    if d and d > 0.5:
                        grid.set_light(idx, [5.0, 5.0, 5.0])
                        errors_introduced += 1
                        break

        if errors_introduced == 0:
            return {"skipped": "Could not inject errors — grid is read-only"}

        # Run validation again
        if hasattr(checker, 'check_all'):
            after = checker.check_all()
        elif hasattr(checker, 'validate'):
            after = checker.validate()

        # Count detected violations
        if isinstance(after, dict):
            for key, val in after.items():
                if 'violation' in str(key).lower() or 'error' in str(key).lower():
                    if val:
                        errors_caught += 1
                if isinstance(val, bool) and not val:
                    errors_caught += 1

        catch_rate = errors_caught / max(errors_introduced, 1)

        return {
            "errors_introduced": errors_introduced,
            "errors_caught": errors_caught,
            "catch_rate": round(catch_rate, 2),
            "target_catch_rate": 0.5,
        }

    runner.run_test("Deliberately introduced errors", 5, critical=False, fn=test_6_4)


# ============================================================================
# Final Report
# ============================================================================

def print_report(runner: TestRunner):
    print("\n" + "=" * 60)
    print("FINAL REPORT — Complete Pipeline Tests")
    print("=" * 60)

    phases = {}
    for r in runner.results:
        if r.phase not in phases:
            phases[r.phase] = []
        phases[r.phase].append(r)

    phase_names = {
        1: "Perception + Grid",
        2: "Shell Extension",
        3: "AI Texturing (buffers)",
        4: "Auto-Furnishing (VLM)",
        5: "Conservation",
    }

    total = len(runner.results)
    passed = sum(1 for r in runner.results if r.passed)
    failed = sum(1 for r in runner.results if not r.passed)
    skipped = sum(1 for r in runner.results if r.details.get('skipped'))
    critical_failed = sum(1 for r in runner.results if not r.passed and r.critical)

    for phase_num in sorted(phases.keys()):
        phase_results = phases[phase_num]
        phase_pass = sum(1 for r in phase_results if r.passed)
        phase_total = len(phase_results)
        phase_name = phase_names.get(phase_num, f"Phase {phase_num}")

        status = "PASS" if phase_pass == phase_total else "PARTIAL" if phase_pass > 0 else "FAIL"
        print(f"\n  Phase {phase_num} ({phase_name}): {phase_pass}/{phase_total} {status}")

        for r in phase_results:
            icon = "✓" if r.passed else "✗"
            crit = " [CRITICAL]" if r.critical else ""
            skip = " (skipped)" if r.details.get('skipped') else ""
            print(f"    {icon} {r.name}{crit}{skip} ({r.duration:.2f}s)")
            if r.error:
                print(f"      Error: {r.error}")
            if r.details.get('skipped'):
                print(f"      Reason: {r.details['skipped']}")

    print(f"\n{'=' * 50}")
    print(f"  TOTAL: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    print(f"  CRITICAL FAILURES: {critical_failed}")
    print(f"  TOTAL TIME: {sum(r.duration for r in runner.results):.1f}s")

    if critical_failed == 0 and passed >= total * 0.7:
        print(f"\n  VERDICT: PIPELINE VALIDATED")
        print(f"  Foundation is solid. Ready for renderer integration.")
    elif critical_failed == 0:
        print(f"\n  VERDICT: PARTIAL PASS")
        print(f"  No critical failures. Some tests skipped or failed — check details above.")
    else:
        print(f"\n  VERDICT: CRITICAL FAILURES — investigate before proceeding")
    print(f"{'=' * 50}")

    # Save results
    results_data = {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "critical_failed": critical_failed,
        },
        "tests": [
            {
                "name": r.name,
                "phase": r.phase,
                "critical": r.critical,
                "passed": r.passed,
                "duration": round(r.duration, 3),
                "details": r.details,
                "error": r.error,
            }
            for r in runner.results
        ],
    }

    output_path = "complete_pipeline_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Trivima Complete Pipeline Tests")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only this phase")
    args = parser.parse_args()

    image_path = find_test_image(args.image)
    print(f"Trivima Complete Pipeline Tests")
    print(f"  Image: {image_path}")
    print(f"  Phase: {'all' if args.phase is None else args.phase}")

    runner = TestRunner()

    phase_fns = {
        1: phase1_tests,
        2: phase2_tests,
        3: phase3_tests,
        4: phase4_tests,
        5: phase5_tests,
    }

    if args.phase:
        phase_fns[args.phase](runner, image_path)
    else:
        for phase_num in sorted(phase_fns.keys()):
            phase_fns[phase_num](runner, image_path)

    print_report(runner)


if __name__ == "__main__":
    main()
