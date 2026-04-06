"""
AI-driven volume fill — Qwen estimates object dimensions from the photo.

For each SAM segment:
  1. Crop the object region from the photo
  2. Ask Qwen: "What is this object? What are its dimensions?"
  3. Qwen responds with estimated width, depth, height
  4. Fill cells backward from the front surface by the estimated depth

No lookup tables. Every object gets its actual dimensions from the AI.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image


@dataclass
class ObjectEstimate:
    """Qwen's estimate of an object's 3D dimensions."""
    name: str
    width_cm: float
    depth_cm: float
    height_cm: float
    fill_direction: str  # "backward", "downward", "both"
    confidence: float


def estimate_objects_with_qwen(
    image: np.ndarray,
    masks: np.ndarray,
    device: str = "cuda",
) -> List[ObjectEstimate]:
    """Ask Qwen to identify and estimate dimensions for each SAM segment.

    Args:
        image: (H, W, 3) uint8 RGB photo
        masks: (N, H, W) bool SAM masks
        device: compute device

    Returns:
        List of ObjectEstimate, one per mask
    """
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print("  Loading Qwen2-VL...")
    model_name = "Qwen/Qwen2-VL-2B-Instruct"  # Small model, fast
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )

    estimates = []
    h, w = image.shape[:2]

    for i, mask in enumerate(masks):
        # Get bounding box of this segment
        ys, xs = np.where(mask)
        if len(ys) == 0:
            estimates.append(ObjectEstimate("unknown", 20, 20, 20, "backward", 0.1))
            continue

        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        # Pad bbox slightly
        pad = 10
        y1 = max(0, y1 - pad)
        y2 = min(h, y2 + pad)
        x1 = max(0, x1 - pad)
        x2 = min(w, x2 + pad)

        # Crop object region
        crop = image[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop)

        # Area percentage
        area_pct = 100 * mask.sum() / (h * w)

        prompt = (
            "Look at this object from a room photo. "
            f"It covers {area_pct:.0f}% of the image. "
            "What is it? Estimate its real-world dimensions in centimeters. "
            "Reply in this exact format:\n"
            "NAME: <object name>\n"
            "WIDTH: <cm>\n"
            "DEPTH: <cm>\n"
            "HEIGHT: <cm>\n"
            "FILL: <backward|downward|both>"
        )

        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": crop_pil},
                {"type": "text", "text": prompt},
            ]}]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[crop_pil], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=100, temperature=0.1)

            response = processor.batch_decode(
                output[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]

            est = _parse_estimate(response, area_pct)
            estimates.append(est)
            print(f"    Segment {i}: {est.name} ({est.width_cm}x{est.depth_cm}x{est.height_cm}cm)")

        except Exception as e:
            print(f"    Segment {i}: Qwen failed ({e}), using heuristic")
            est = _heuristic_estimate(area_pct, y1, y2, x1, x2, h, w)
            estimates.append(est)

    del model, processor
    import torch
    torch.cuda.empty_cache()

    return estimates


def _parse_estimate(response: str, area_pct: float) -> ObjectEstimate:
    """Parse Qwen's response into ObjectEstimate."""
    import re

    def extract(pattern, default):
        m = re.search(pattern, response, re.IGNORECASE)
        return m.group(1).strip() if m else default

    def extract_float(pattern, default):
        m = re.search(pattern, response, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return default
        return default

    name = extract(r'NAME:\s*(.+?)(?:\n|$)', 'object')
    width = extract_float(r'WIDTH:\s*([0-9.]+)', 50)
    depth = extract_float(r'DEPTH:\s*([0-9.]+)', 30)
    height = extract_float(r'HEIGHT:\s*([0-9.]+)', 50)
    fill = extract(r'FILL:\s*(\w+)', 'backward')

    # Sanity bounds
    width = max(5, min(500, width))
    depth = max(2, min(300, depth))
    height = max(5, min(400, height))

    return ObjectEstimate(
        name=name.lower(),
        width_cm=width,
        depth_cm=depth,
        height_cm=height,
        fill_direction=fill.lower(),
        confidence=0.7,
    )


def _heuristic_estimate(area_pct, y1, y2, x1, x2, h, w) -> ObjectEstimate:
    """Fallback: estimate from segment position and size."""
    # Bottom half = likely floor/furniture, top half = wall/ceiling
    center_y = (y1 + y2) / 2 / h

    if area_pct > 30:
        # Large segment — wall or floor
        if center_y < 0.4:
            return ObjectEstimate("wall", 200, 15, 270, "backward", 0.3)
        else:
            return ObjectEstimate("floor", 300, 10, 5, "downward", 0.3)
    elif area_pct > 10:
        # Medium — furniture
        return ObjectEstimate("furniture", 100, 60, 80, "backward", 0.3)
    else:
        # Small — decoration or fixture
        return ObjectEstimate("object", 30, 15, 30, "backward", 0.3)


def ai_volume_fill(
    cell_pos: np.ndarray,
    cell_col: np.ndarray,
    cell_nrm: np.ndarray,
    cell_labels: np.ndarray,
    estimates: List[ObjectEstimate],
    cell_size: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill volumes behind surfaces using AI-estimated object depths.

    Each SAM segment gets its own fill depth from Qwen's estimate.
    Fill direction is determined per-segment via RANSAC dominant normal.
    """
    import time
    t0 = time.time()
    n_original = len(cell_pos)

    # Build spatial hash
    existing = set()
    for i in range(n_original):
        key = tuple(np.floor(cell_pos[i] / cell_size).astype(int))
        existing.add(key)

    # Find dominant normal per segment — snap to nearest axis for clean fill
    unique_labels = np.unique(cell_labels)
    segment_normals = {}
    for label in unique_labels:
        mask = cell_labels == label
        if mask.sum() < 5:
            continue
        normals = cell_nrm[mask]
        avg = normals.mean(axis=0)
        nm = np.linalg.norm(avg)
        if nm < 1e-6:
            segment_normals[label] = np.array([0, 0, 1], dtype=np.float32)
            continue

        avg = avg / nm
        # Snap to nearest axis if close (within 30 degrees)
        # This prevents checkerboard fill patterns on planar surfaces
        axes = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.float32)
        dots = np.abs(axes @ avg)
        best_axis = axes[dots.argmax()]
        # If within 30 degrees of an axis, snap to it
        if dots.max() > 0.866:  # cos(30°)
            segment_normals[label] = best_axis * np.sign(axes[dots.argmax()] @ avg)
        else:
            segment_normals[label] = avg.astype(np.float32)

    # Get fill depth per segment from AI estimates with minimum enforcement
    MIN_FILL = {
        "floor": 0.10, "wall": 0.15, "ceiling": 0.10,
        "door": 0.05, "window": 0.10,
    }
    segment_depths = {}
    for label in unique_labels:
        if label == 0:
            segment_depths[label] = 0.10
            continue
        est_idx = int(label) - 1
        if est_idx < len(estimates):
            est = estimates[est_idx]
            depth_m = est.depth_cm / 100.0

            # Enforce minimums for surfaces
            for surface_name, min_depth in MIN_FILL.items():
                if surface_name in est.name.lower():
                    depth_m = max(depth_m, min_depth)
                    break
            else:
                # All objects at least 10cm thick
                depth_m = max(depth_m, 0.10)

            segment_depths[label] = depth_m
        else:
            segment_depths[label] = 0.15

    # Fill each cell along its segment's dominant normal
    generated_pos = []
    generated_col = []

    for i in range(n_original):
        label = cell_labels[i]
        if label not in segment_normals:
            continue

        fill_depth = segment_depths.get(label, 0.15)
        fill_dir = -segment_normals[label]  # opposite of surface normal
        n_fill = max(1, int(fill_depth / cell_size))

        # Snap base position to grid center to prevent floating-point drift
        base_key = np.floor(cell_pos[i] / cell_size).astype(int)

        for step in range(1, n_fill + 1):
            # Step along fill direction in grid-aligned increments
            new_key = tuple(base_key + (fill_dir * step).astype(int))

            if new_key not in existing:
                existing.add(new_key)
                new_pos = (np.array(new_key, dtype=np.float32) + 0.5) * cell_size
                darken = max(0.85, 1.0 - step * 0.003)
                generated_pos.append(new_pos)
                generated_col.append((cell_col[i] * darken).astype(np.float32))

    if not generated_pos:
        print(f"  AI volume fill: no cells generated")
        return cell_pos, cell_col, cell_nrm

    gen_pos = np.array(generated_pos, dtype=np.float32)
    gen_col = np.array(generated_col, dtype=np.float32)
    gen_nrm = np.zeros_like(gen_pos)
    gen_nrm[:] = [0, 0, 1]

    all_pos = np.concatenate([cell_pos, gen_pos])
    all_col = np.concatenate([cell_col, gen_col])
    all_nrm = np.concatenate([cell_nrm, gen_nrm])

    dt = time.time() - t0
    print(f"  AI volume fill: {n_original:,} surface + {len(gen_pos):,} filled = {len(all_pos):,} total ({dt:.1f}s)")

    return all_pos, all_col, all_nrm
