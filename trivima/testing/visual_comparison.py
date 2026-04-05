"""
Visual comparison generator — side-by-side output for quality evaluation.

Layout:
  ┌──────────────┬──────────────┬──────────────┐
  │  Flat render  │  AI textured │  Ground truth │
  ├──────────────┼──────────────┼──────────────┤
  │    Depth     │   Normals    │    Labels     │
  └──────────────┴──────────────┴──────────────┘
  │ Red overlay: cells flagged for conservation violations │

Used in Week 4 testing to visually assess quality per photo.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List


def create_comparison_image(
    flat_render: np.ndarray,
    ai_textured: np.ndarray,
    ground_truth: Optional[np.ndarray],
    depth_buffer: Optional[np.ndarray] = None,
    normal_buffer: Optional[np.ndarray] = None,
    label_buffer: Optional[np.ndarray] = None,
    violation_mask: Optional[np.ndarray] = None,
    title: str = "",
) -> np.ndarray:
    """Create a side-by-side comparison image.

    Args:
        flat_render: (H, W, 3) flat cell render (albedo × basic light)
        ai_textured: (H, W, 3) AI texturing output
        ground_truth: (H, W, 3) original photograph (optional)
        depth_buffer: (H, W) depth map (optional, for bottom strip)
        normal_buffer: (H, W, 3) normal map (optional)
        label_buffer: (H, W) semantic labels (optional)
        violation_mask: (H, W) bool, cells with conservation violations
        title: text to render at the top

    Returns:
        (out_H, out_W, 3) uint8 comparison image
    """
    h, w = flat_render.shape[:2]
    pad = 4  # pixels between panels

    # Ensure all images are same size
    flat = _ensure_rgb_uint8(flat_render, h, w)
    ai = _ensure_rgb_uint8(ai_textured, h, w)

    if ground_truth is not None:
        gt = _ensure_rgb_uint8(ground_truth, h, w)
    else:
        gt = np.full((h, w, 3), 128, dtype=np.uint8)  # grey placeholder

    # Top row: flat | AI textured | ground truth
    top_row = np.concatenate([
        flat, np.full((h, pad, 3), 255, dtype=np.uint8),
        ai, np.full((h, pad, 3), 255, dtype=np.uint8),
        gt,
    ], axis=1)

    # Bottom row: depth | normals | labels
    bottom_panels = []

    if depth_buffer is not None:
        depth_vis = _colorize_depth(depth_buffer, h, w)
    else:
        depth_vis = np.full((h, w, 3), 64, dtype=np.uint8)
    bottom_panels.append(depth_vis)

    if normal_buffer is not None:
        normal_vis = _colorize_normals(normal_buffer, h, w)
    else:
        normal_vis = np.full((h, w, 3), 64, dtype=np.uint8)
    bottom_panels.append(normal_vis)

    if label_buffer is not None:
        label_vis = _colorize_labels(label_buffer, h, w)
    else:
        label_vis = np.full((h, w, 3), 64, dtype=np.uint8)
    bottom_panels.append(label_vis)

    bottom_row = np.concatenate([
        bottom_panels[0],
        np.full((h, pad, 3), 255, dtype=np.uint8),
        bottom_panels[1],
        np.full((h, pad, 3), 255, dtype=np.uint8),
        bottom_panels[2],
    ], axis=1)

    # Combine rows
    divider = np.full((pad, top_row.shape[1], 3), 255, dtype=np.uint8)
    combined = np.concatenate([top_row, divider, bottom_row], axis=0)

    # Overlay violation mask on AI textured panel
    if violation_mask is not None:
        mask_resized = _resize_mask(violation_mask, h, w)
        # Red overlay on the AI panel (offset by flat_width + pad)
        x_offset = w + pad
        for y in range(h):
            for x in range(w):
                if mask_resized[y, x]:
                    combined[y, x_offset + x] = [255, 0, 0]  # red

    return combined


def save_comparison(
    output_path: str,
    flat_render: np.ndarray,
    ai_textured: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    **kwargs,
):
    """Create and save a comparison image to disk."""
    from PIL import Image

    img = create_comparison_image(flat_render, ai_textured, ground_truth, **kwargs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(output_path)


def save_comparison_grid(
    output_path: str,
    scenes: List[dict],
    cols: int = 4,
):
    """Create a grid of comparison thumbnails across multiple scenes.

    Args:
        output_path: where to save the grid image
        scenes: list of dicts with keys: flat_render, ai_textured, ground_truth, scene_id
        cols: number of columns in the grid
    """
    from PIL import Image

    thumb_size = 256
    pad = 2
    rows = (len(scenes) + cols - 1) // cols

    grid_w = cols * (thumb_size * 3 + pad * 2) + (cols - 1) * pad
    grid_h = rows * (thumb_size + pad) + (rows - 1) * pad

    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    for i, scene in enumerate(scenes):
        row = i // cols
        col = i % cols

        comparison = create_comparison_image(
            scene["flat_render"],
            scene["ai_textured"],
            scene.get("ground_truth"),
        )

        # Resize to thumbnail
        comp_pil = Image.fromarray(comparison).resize(
            (thumb_size * 3 + pad * 2, thumb_size),
            Image.LANCZOS
        )
        thumb = np.array(comp_pil)

        y = row * (thumb_size + pad)
        x = col * (thumb_size * 3 + pad * 2 + pad)

        th, tw = thumb.shape[:2]
        if y + th <= grid_h and x + tw <= grid_w:
            grid[y:y+th, x:x+tw] = thumb

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)


# --- Helper functions ---

def _ensure_rgb_uint8(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Ensure image is (h, w, 3) uint8."""
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if img.shape[:2] != (h, w):
        from PIL import Image
        img = np.array(Image.fromarray(img).resize((w, h)))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def _colorize_depth(depth: np.ndarray, h: int, w: int) -> np.ndarray:
    """Colorize depth map using turbo colormap."""
    d = depth.copy()
    valid = d > 0
    if valid.any():
        d_min, d_max = d[valid].min(), d[valid].max()
        if d_max > d_min:
            d = np.where(valid, (d - d_min) / (d_max - d_min), 0)

    # Simple blue→red colormap
    r = np.clip(d * 2, 0, 1)
    b = np.clip(1 - d * 2, 0, 1)
    g = np.clip(1 - np.abs(d - 0.5) * 2, 0, 1)

    vis = np.stack([r, g, b], axis=-1)
    return _ensure_rgb_uint8(vis, h, w)


def _colorize_normals(normals: np.ndarray, h: int, w: int) -> np.ndarray:
    """Colorize normal map: XYZ → RGB."""
    vis = (normals + 1.0) / 2.0  # [-1,1] → [0,1]
    return _ensure_rgb_uint8(vis, h, w)


def _colorize_labels(labels: np.ndarray, h: int, w: int) -> np.ndarray:
    """Colorize semantic labels using a fixed palette."""
    # Simple hash-based coloring
    r = ((labels * 67) % 256).astype(np.uint8)
    g = ((labels * 137) % 256).astype(np.uint8)
    b = ((labels * 229) % 256).astype(np.uint8)

    vis = np.stack([r, g, b], axis=-1)
    if vis.shape[:2] != (h, w):
        from PIL import Image
        vis = np.array(Image.fromarray(vis).resize((w, h), Image.NEAREST))
    return vis


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape[:2] == (h, w):
        return mask
    from PIL import Image
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 128
