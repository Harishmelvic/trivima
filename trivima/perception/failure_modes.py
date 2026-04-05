"""
Failure mode detection and mitigation for depth estimation.

Certain scene elements produce structurally incorrect depth — not just noisy
but fundamentally wrong. This module detects these cases from SAM labels and
image analysis, then applies per-cell mitigations.

Failure modes (from single_image_precision_theory.md, Chapter 7):

  Mirror:    Depth extends behind wall into phantom room.
             → Force density=1.0, confidence=0.1, block subdivision.

  Glass:     Depth sees through glass to floor beneath.
             → Force density=1.0 (solid barrier), confidence=0.2.

  Transparent: Slight refraction distortion around vases/bottles.
             → Low confidence=0.3, expand collision margin.

  Dark scene: Universal noise, gradients meaningless everywhere.
             → All cells low confidence, rely on AI texturing.

  Sky:       Extreme depth (50-1000m), nonsense geometry.
             → Exclude from cell grid entirely, render as skybox.

  Specular:  Depth oscillates due to reflected highlights.
             → Moderate-low confidence=0.4, extra depth smoothing.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Set, Optional, Tuple


# SAM label categories that trigger failure mode mitigations
# These label strings match common SAM/Grounded-SAM output categories
MIRROR_LABELS = {"mirror", "looking glass", "vanity mirror", "wall mirror"}
GLASS_LABELS = {"glass", "glass table", "glass door", "glass panel", "glass window",
                "glass shelf", "coffee table"}  # glass coffee tables are the #1 failure
TRANSPARENT_LABELS = {"vase", "bottle", "glass bottle", "clear plastic",
                      "transparent", "crystal", "wine glass", "drinking glass"}
SKY_LABELS = {"sky", "cloud", "clouds", "blue sky"}
SPECULAR_LABELS = {"chrome", "stainless steel", "polished metal", "wet floor",
                   "glossy", "lacquered", "mirror finish"}

# Brightness threshold for dark scene detection (0-255 scale)
DARK_SCENE_THRESHOLD = 30


@dataclass
class FailureModeReport:
    """Results of failure mode detection for a scene."""
    has_mirrors: bool = False
    has_glass: bool = False
    has_transparent: bool = False
    is_dark_scene: bool = False
    has_sky: bool = False
    has_specular: bool = False

    mirror_pixel_count: int = 0
    glass_pixel_count: int = 0
    sky_pixel_count: int = 0
    mean_brightness: float = 128.0

    # Pixel masks (H, W) bool — which pixels belong to each failure category
    mirror_mask: Optional[np.ndarray] = None
    glass_mask: Optional[np.ndarray] = None
    transparent_mask: Optional[np.ndarray] = None
    sky_mask: Optional[np.ndarray] = None
    specular_mask: Optional[np.ndarray] = None

    @property
    def num_failure_types(self) -> int:
        return sum([self.has_mirrors, self.has_glass, self.has_transparent,
                    self.is_dark_scene, self.has_sky, self.has_specular])

    @property
    def predicted_success(self) -> bool:
        """Predict whether this scene will produce an acceptable cell grid."""
        # Dark scenes with no other issues are marginal
        if self.is_dark_scene:
            return False
        # Scenes with only minor failure modes (small transparent objects) are fine
        if self.num_failure_types <= 1 and not self.has_mirrors:
            return True
        # Multiple failure modes reduce likelihood of success
        return self.num_failure_types <= 2


def detect_failure_modes(
    image: np.ndarray,
    labels: np.ndarray,
    label_names: Dict[int, str],
) -> FailureModeReport:
    """Detect failure modes from image content and semantic labels.

    Args:
        image: (H, W, 3) uint8 RGB image
        labels: (H, W) int32 semantic label indices from SAM
        label_names: dict mapping label index → category name string

    Returns:
        FailureModeReport with detected failure modes and pixel masks
    """
    h, w = labels.shape[:2]
    report = FailureModeReport()

    # Build label-to-category lookup
    label_to_category = {}
    for idx, name in label_names.items():
        name_lower = name.lower().strip()
        label_to_category[idx] = name_lower

    # Detect each failure category
    mirror_mask = np.zeros((h, w), dtype=bool)
    glass_mask = np.zeros((h, w), dtype=bool)
    transparent_mask = np.zeros((h, w), dtype=bool)
    sky_mask = np.zeros((h, w), dtype=bool)
    specular_mask = np.zeros((h, w), dtype=bool)

    for idx, name in label_to_category.items():
        pixel_mask = labels == idx
        if not pixel_mask.any():
            continue

        if name in MIRROR_LABELS or "mirror" in name:
            mirror_mask |= pixel_mask
        if name in GLASS_LABELS or "glass" in name:
            glass_mask |= pixel_mask
        if name in TRANSPARENT_LABELS or "transparent" in name:
            transparent_mask |= pixel_mask
        if name in SKY_LABELS or "sky" in name:
            sky_mask |= pixel_mask
        if name in SPECULAR_LABELS:
            specular_mask |= pixel_mask

    report.mirror_mask = mirror_mask
    report.glass_mask = glass_mask
    report.transparent_mask = transparent_mask
    report.sky_mask = sky_mask
    report.specular_mask = specular_mask

    report.has_mirrors = mirror_mask.any()
    report.has_glass = glass_mask.any()
    report.has_transparent = transparent_mask.any()
    report.has_sky = sky_mask.any()
    report.has_specular = specular_mask.any()

    report.mirror_pixel_count = int(mirror_mask.sum())
    report.glass_pixel_count = int(glass_mask.sum())
    report.sky_pixel_count = int(sky_mask.sum())

    # Dark scene detection
    report.mean_brightness = float(image.mean())
    report.is_dark_scene = report.mean_brightness < DARK_SCENE_THRESHOLD

    # Specular detection from image analysis (supplement SAM labels)
    # High local brightness variance + high absolute brightness = specular
    if not report.has_specular:
        gray = image.mean(axis=2)
        local_std = _local_std(gray, kernel_size=11)
        bright_and_varying = (gray > 200) & (local_std > 40)
        if bright_and_varying.sum() > h * w * 0.02:  # >2% of image
            report.has_specular = True
            report.specular_mask = bright_and_varying

    return report


def apply_failure_mitigations(
    depth: np.ndarray,
    report: FailureModeReport,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply mitigations to the depth map based on detected failure modes.

    Args:
        depth: (H, W) float32 metric depth map from Depth Pro
        report: FailureModeReport from detect_failure_modes()

    Returns:
        (modified_depth, confidence_map) — both (H, W) float32
        confidence_map: per-pixel confidence [0, 1] to be propagated to cells
    """
    h, w = depth.shape
    modified_depth = depth.copy()
    confidence = np.ones((h, w), dtype=np.float32)

    # Sky: exclude from cell grid (set depth to 0, confidence to 0)
    if report.has_sky and report.sky_mask is not None:
        modified_depth[report.sky_mask] = 0.0
        confidence[report.sky_mask] = 0.0

    # Mirror: clamp depth to the wall surface (use median depth of surrounding non-mirror pixels)
    if report.has_mirrors and report.mirror_mask is not None:
        # Estimate wall depth from pixels around the mirror
        dilated = _dilate_mask(report.mirror_mask, kernel_size=21)
        border = dilated & ~report.mirror_mask
        if border.any():
            wall_depth = np.median(depth[border])
            modified_depth[report.mirror_mask] = wall_depth
        confidence[report.mirror_mask] = 0.1

    # Glass: force depth to a reasonable value (use surrounding context)
    if report.has_glass and report.glass_mask is not None:
        # Glass should be at roughly the same depth as surrounding furniture
        dilated = _dilate_mask(report.glass_mask, kernel_size=15)
        border = dilated & ~report.glass_mask
        if border.any():
            glass_depth = np.median(depth[border])
            # Don't replace depth entirely — blend with existing
            # (the depth "through" glass is the floor, which may be useful nearby)
            modified_depth[report.glass_mask] = glass_depth
        confidence[report.glass_mask] = 0.2

    # Transparent objects: reduce confidence, keep depth
    if report.has_transparent and report.transparent_mask is not None:
        confidence[report.transparent_mask] = 0.3

    # Specular: reduce confidence
    if report.has_specular and report.specular_mask is not None:
        confidence[report.specular_mask] = np.minimum(
            confidence[report.specular_mask], 0.4
        )

    # Dark scene: universal low confidence
    if report.is_dark_scene:
        confidence *= 0.4  # scale everything down

    return modified_depth, confidence


def _local_std(image: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """Compute local standard deviation using box filter."""
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(image.astype(np.float64), size=kernel_size)
    mean_sq = uniform_filter((image.astype(np.float64)) ** 2, size=kernel_size)
    variance = np.maximum(mean_sq - mean ** 2, 0)
    return np.sqrt(variance).astype(np.float32)


def _dilate_mask(mask: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """Dilate a binary mask."""
    from scipy.ndimage import binary_dilation
    struct = np.ones((kernel_size, kernel_size), dtype=bool)
    return binary_dilation(mask, structure=struct)
