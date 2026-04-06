"""
Room dimension estimation via Qwen VLM.

Takes a room photo and asks Qwen to estimate:
  - Room type (living room, bedroom, etc.)
  - Room dimensions (width x depth in meters)
  - Ceiling height
  - Distance from camera to back wall
  - Distance behind camera to the wall

This replaces dumb heuristics like "assume square room" with
context-aware AI estimation. Qwen has seen millions of rooms
and knows what a 3x4m room looks like versus a 6x8m room.

Used by shell_extension.py to generate room shell with informed dimensions.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RoomEstimate:
    """Estimated room dimensions from VLM."""
    room_type: str           # 'living_room', 'bedroom', 'kitchen', etc.
    width_m: float           # estimated room width in meters
    depth_m: float           # estimated room depth in meters
    ceiling_height_m: float  # estimated ceiling height
    behind_camera_m: float   # estimated depth behind camera
    confidence: float        # 0-1, how confident the VLM is
    reasoning: str           # VLM's explanation


def estimate_room_dimensions(
    image_path: str,
    observed_bounds: Dict[str, float],
    focal_length: float,
    device: str = "cuda",
) -> RoomEstimate:
    """Use Qwen VLM to estimate room dimensions from a photo.

    Args:
        image_path: path to the room photo
        observed_bounds: dict with 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
            from the observed cell positions (Depth Pro output)
        focal_length: camera focal length from Depth Pro
        device: compute device

    Returns:
        RoomEstimate with informed dimensions
    """
    try:
        return _estimate_via_qwen(image_path, observed_bounds, focal_length, device)
    except Exception as e:
        print(f"  [RoomEstimator] Qwen unavailable ({e}), using heuristic fallback")
        return _estimate_heuristic(observed_bounds)


def _estimate_via_qwen(
    image_path: str,
    observed_bounds: Dict[str, float],
    focal_length: float,
    device: str,
) -> RoomEstimate:
    """Ask Qwen to estimate room dimensions."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )

    obs_width = observed_bounds['x_max'] - observed_bounds['x_min']
    obs_depth = abs(observed_bounds['z_min'])  # depth in front of camera

    prompt = f"""Look at this room photo and estimate the full room dimensions.

What I already know from depth sensing:
- The visible part of the room is approximately {obs_width:.1f}m wide and {obs_depth:.1f}m deep
- Camera focal length: {focal_length:.0f}px

Please estimate:
1. Room type (living room, bedroom, kitchen, bathroom, office, etc.)
2. Full room width in meters (not just visible part)
3. Full room depth in meters (front wall to back wall)
4. Ceiling height in meters
5. How far behind the camera the nearest wall is (meters)
6. Confidence (low/medium/high)

Consider: door sizes (~200cm tall, 80cm wide), furniture scale, architectural style,
ceiling type, room proportions. Answer in this exact format:

ROOM_TYPE: <type>
WIDTH: <meters>
DEPTH: <meters>
CEILING: <meters>
BEHIND_CAMERA: <meters>
CONFIDENCE: <low|medium|high>
REASONING: <one sentence explanation>"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image_path], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.1)

    response = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)[0]

    # Parse response
    return _parse_qwen_response(response, observed_bounds)


def _parse_qwen_response(response: str, observed_bounds: Dict[str, float]) -> RoomEstimate:
    """Parse Qwen's structured response into RoomEstimate."""
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

    room_type = extract(r'ROOM_TYPE:\s*(.+)', 'living_room')
    width = extract_float(r'WIDTH:\s*([0-9.]+)', 4.0)
    depth = extract_float(r'DEPTH:\s*([0-9.]+)', 5.0)
    ceiling = extract_float(r'CEILING:\s*([0-9.]+)', 2.7)
    behind = extract_float(r'BEHIND_CAMERA:\s*([0-9.]+)', 2.0)
    conf_str = extract(r'CONFIDENCE:\s*(\w+)', 'medium')
    reasoning = extract(r'REASONING:\s*(.+)', 'No reasoning provided')

    conf_map = {'low': 0.3, 'medium': 0.6, 'high': 0.85}
    confidence = conf_map.get(conf_str.lower(), 0.5)

    # Sanity checks: room dimensions should be reasonable
    width = max(2.0, min(15.0, width))
    depth = max(2.0, min(15.0, depth))
    ceiling = max(2.2, min(4.0, ceiling))
    behind = max(0.5, min(8.0, behind))

    return RoomEstimate(
        room_type=room_type.lower().replace(' ', '_'),
        width_m=width,
        depth_m=depth,
        ceiling_height_m=ceiling,
        behind_camera_m=behind,
        confidence=confidence,
        reasoning=reasoning,
    )


def _estimate_heuristic(observed_bounds: Dict[str, float]) -> RoomEstimate:
    """Fallback: estimate room dimensions from observed cell bounds only.

    This is the "dumb math" approach — better than nothing but worse than Qwen.
    """
    obs_width = observed_bounds['x_max'] - observed_bounds['x_min']
    obs_depth = abs(observed_bounds['z_min'])
    obs_height = observed_bounds['y_max'] - observed_bounds['y_min']

    # Assume room extends ~50% beyond observed bounds on each side
    width = obs_width * 1.5
    depth = obs_depth * 1.5
    ceiling = max(2.5, obs_height * 2.5)  # observed height is usually < half ceiling
    behind = obs_depth * 0.5  # assume camera is roughly 1/3 into the room

    return RoomEstimate(
        room_type="room",
        width_m=width,
        depth_m=depth,
        ceiling_height_m=ceiling,
        behind_camera_m=behind,
        confidence=0.2,
        reasoning="Heuristic fallback — estimated from observed cell bounds only",
    )
