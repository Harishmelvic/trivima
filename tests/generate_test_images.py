"""
Generate 11 synthetic test images with ground truth depth and segmentation.

Renders simple rooms using NumPy — no OpenGL, no GPU, no external deps beyond Pillow.

Test Image Set (from trivima_testing_stage2.md):
  1-5: "ScanNet-like" rooms with known geometry (quantitative comparison)
    1. living_room    — sofa, table, textured wood floor, two walls visible
    2. bedroom        — bed, nightstand, carpet floor
    3. kitchen        — counter, cabinet, tile floor with grid pattern
    4. bathroom       — small room, white tiles, toilet
    5. office         — desk, bookshelf, wood floor with grain

  6-8: "Smartphone-like" rooms (varied conditions)
    6. wide_angle     — corner view showing 3 walls + floor + ceiling
    7. cluttered      — many small objects, complex occlusion
    8. hallway        — long narrow space, strong perspective

  9-11: Failure mode images
    9. glass_table    — room with glass coffee table (invisible to depth)
    10. mirror_wall   — room with large wall mirror (phantom room)
    11. dark_room     — same as living_room but very dark (brightness < 30/255)

Each image produces:
  - RGB image (H×W×3, uint8)
  - Ground truth depth (H×W, float32, meters)
  - Ground truth segmentation (H×W, int32)
  - Ground truth label names (dict)
  - Camera intrinsics (3×3 matrix)

Usage:
  python tests/generate_test_images.py
  # Generates to data/test_images/
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class TestImage:
    """One synthetic test image with all ground truth."""
    name: str
    rgb: np.ndarray           # (H, W, 3) uint8
    depth: np.ndarray         # (H, W) float32 meters
    labels: np.ndarray        # (H, W) int32
    label_names: Dict[int, str]
    intrinsics: np.ndarray    # (3, 3) float32
    description: str

    def save(self, output_dir: str):
        d = Path(output_dir) / self.name
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.rgb).save(str(d / "rgb.png"))
        np.save(str(d / "depth.npy"), self.depth)
        np.save(str(d / "labels.npy"), self.labels)
        np.savez(str(d / "metadata.npz"),
                 intrinsics=self.intrinsics,
                 label_names=np.array(list(self.label_names.items()), dtype=object))
        # Save depth as visualization
        d_vis = self.depth.copy()
        valid = d_vis > 0
        if valid.any():
            d_min, d_max = d_vis[valid].min(), d_vis[valid].max()
            d_vis = np.where(valid, (d_vis - d_min) / (d_max - d_min + 1e-8), 0)
        Image.fromarray((d_vis * 255).astype(np.uint8)).save(str(d / "depth_vis.png"))


# ============================================================
# Room geometry primitives
# ============================================================

def make_intrinsics(h: int, w: int, fov_deg: float = 60.0) -> np.ndarray:
    """Create camera intrinsics matrix."""
    f = w / (2 * np.tan(np.radians(fov_deg / 2)))
    return np.array([
        [f, 0, w / 2],
        [0, f, h / 2],
        [0, 0, 1],
    ], dtype=np.float32)


def render_room(
    h: int, w: int,
    room_depth: float,     # Z extent of room
    room_width: float,     # X extent
    room_height: float,    # Y extent
    cam_height: float,     # camera Y position
    floor_color: Tuple[int, int, int],
    wall_color: Tuple[int, int, int],
    ceiling_color: Tuple[int, int, int] = (220, 220, 225),
    fov_deg: float = 60.0,
    objects: list = None,  # list of (label, x_min, x_max, y_min, y_max, z, color)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """Render a simple room with ray casting. Returns (rgb, depth, labels, label_names)."""
    K = make_intrinsics(h, w, fov_deg)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)
    labels = np.zeros((h, w), dtype=np.int32)
    label_names = {0: "background", 1: "floor", 2: "wall", 3: "ceiling"}

    if objects is None:
        objects = []

    # Add object labels
    for i, obj in enumerate(objects):
        label_id = 10 + i
        label_names[label_id] = obj[0]

    for py in range(h):
        for px in range(w):
            # Ray direction from camera
            rx = (px - cx) / fx
            ry = (py - cy) / fy
            rz = 1.0

            norm = np.sqrt(rx**2 + ry**2 + rz**2)
            rx, ry, rz = rx/norm, ry/norm, rz/norm

            best_t = 1e10
            best_color = (0, 0, 0)
            best_label = 0

            # Floor: y = -cam_height → t = -cam_height / ry
            if ry > 0.001:
                t = cam_height / ry
                if 0 < t < best_t:
                    hit_x = rx * t
                    hit_z = rz * t
                    if abs(hit_x) < room_width / 2 and 0 < hit_z < room_depth:
                        best_t = t
                        # Floor texture: checkerboard or wood grain
                        checker = int(hit_x * 4) % 2 ^ int(hit_z * 4) % 2
                        fc = floor_color
                        c = tuple(max(0, min(255, c + (15 if checker else -15))) for c in fc)
                        best_color = c
                        best_label = 1

            # Ceiling: y = room_height - cam_height → t = (room_height - cam_height) / (-ry)
            if ry < -0.001:
                t = (room_height - cam_height) / (-ry)
                if 0 < t < best_t:
                    hit_x = rx * t
                    hit_z = rz * t
                    if abs(hit_x) < room_width / 2 and 0 < hit_z < room_depth:
                        best_t = t
                        best_color = ceiling_color
                        best_label = 3

            # Back wall: z = room_depth → t = room_depth / rz
            if rz > 0.001:
                t = room_depth / rz
                if 0 < t < best_t:
                    hit_x = rx * t
                    hit_y = -ry * t + cam_height
                    if abs(hit_x) < room_width / 2 and 0 < hit_y < room_height:
                        best_t = t
                        best_color = wall_color
                        best_label = 2

            # Left wall: x = -room_width/2
            if rx < -0.001:
                t = (-room_width / 2) / rx
                if 0 < t < best_t:
                    hit_z = rz * t
                    hit_y = -ry * t + cam_height
                    if 0 < hit_z < room_depth and 0 < hit_y < room_height:
                        best_t = t
                        wc = tuple(max(0, min(255, c - 20)) for c in wall_color)
                        best_color = wc
                        best_label = 2

            # Right wall: x = room_width/2
            if rx > 0.001:
                t = (room_width / 2) / rx
                if 0 < t < best_t:
                    hit_z = rz * t
                    hit_y = -ry * t + cam_height
                    if 0 < hit_z < room_depth and 0 < hit_y < room_height:
                        best_t = t
                        wc = tuple(max(0, min(255, c - 10)) for c in wall_color)
                        best_color = wc
                        best_label = 2

            # Objects (flat rectangles at given Z depth)
            for i, obj in enumerate(objects):
                obj_name, ox_min, ox_max, oy_min, oy_max, oz, obj_color = obj
                if rz > 0.001:
                    t = oz / rz
                    if 0 < t < best_t:
                        hit_x = rx * t
                        hit_y = -ry * t + cam_height
                        if ox_min < hit_x < ox_max and oy_min < hit_y < oy_max:
                            best_t = t
                            best_color = obj_color
                            best_label = 10 + i

            if best_t < 1e9:
                depth[py, px] = best_t
                rgb[py, px] = best_color
                labels[py, px] = best_label

    return rgb, depth, labels, label_names


# ============================================================
# 11 Test Images
# ============================================================

def generate_all(h: int = 480, w: int = 640) -> list:
    """Generate all 11 test images."""
    images = []

    # --- 1-5: ScanNet-like rooms ---

    # 1. Living room — sofa, table, textured wood floor
    rgb, depth, labels, ln = render_room(
        h, w, room_depth=5.0, room_width=6.0, room_height=2.8, cam_height=1.6,
        floor_color=(140, 100, 60), wall_color=(210, 205, 195),
        objects=[
            ("sofa", -1.5, 1.5, 0.3, 1.0, 3.5, (80, 60, 50)),
            ("table", -0.5, 0.5, 0.3, 0.5, 2.0, (120, 80, 40)),
            ("door", 2.0, 2.8, 0.0, 2.0, 4.8, (100, 70, 45)),
        ],
    )
    images.append(TestImage("01_living_room", rgb, depth, labels, ln,
                            make_intrinsics(h, w), "Living room with sofa, table, door"))

    # 2. Bedroom
    rgb, depth, labels, ln = render_room(
        h, w, room_depth=4.5, room_width=5.0, room_height=2.7, cam_height=1.6,
        floor_color=(160, 140, 120), wall_color=(200, 195, 185),
        objects=[
            ("bed", -1.2, 1.2, 0.2, 0.8, 3.0, (180, 170, 160)),
            ("nightstand", 1.4, 1.8, 0.3, 0.7, 3.2, (100, 70, 40)),
        ],
    )
    images.append(TestImage("02_bedroom", rgb, depth, labels, ln,
                            make_intrinsics(h, w), "Bedroom with bed and nightstand"))

    # 3. Kitchen — tile floor
    rgb, depth, labels, ln = render_room(
        h, w, room_depth=4.0, room_width=4.5, room_height=2.6, cam_height=1.6,
        floor_color=(190, 185, 175), wall_color=(230, 225, 215),
        objects=[
            ("counter", -1.5, 1.5, 0.7, 0.95, 2.5, (160, 150, 140)),
            ("cabinet", -1.5, 1.5, 1.0, 2.2, 2.5, (130, 100, 60)),
        ],
    )
    images.append(TestImage("03_kitchen", rgb, depth, labels, ln,
                            make_intrinsics(h, w), "Kitchen with counter and cabinets"))

    # 4. Bathroom — small white room
    rgb, depth, labels, ln = render_room(
        h, w, room_depth=3.0, room_width=2.5, room_height=2.5, cam_height=1.6,
        floor_color=(200, 200, 195), wall_color=(240, 240, 235),
        objects=[
            ("toilet", -0.2, 0.2, 0.2, 0.5, 2.2, (230, 230, 225)),
        ],
    )
    images.append(TestImage("04_bathroom", rgb, depth, labels, ln,
                            make_intrinsics(h, w), "Small bathroom with toilet"))

    # 5. Office — desk, bookshelf
    rgb, depth, labels, ln = render_room(
        h, w, room_depth=5.0, room_width=5.0, room_height=2.8, cam_height=1.6,
        floor_color=(130, 95, 55), wall_color=(215, 210, 200),
        objects=[
            ("desk", -0.8, 0.8, 0.5, 0.8, 2.5, (110, 80, 45)),
            ("bookshelf", -2.0, -1.2, 0.0, 2.2, 4.5, (90, 65, 35)),
            ("door", 1.8, 2.5, 0.0, 2.0, 4.9, (105, 75, 45)),
        ],
    )
    images.append(TestImage("05_office", rgb, depth, labels, ln,
                            make_intrinsics(h, w), "Office with desk, bookshelf, door"))

    # --- 6-8: Smartphone-like ---

    # 6. Wide angle — corner view, 3 walls visible
    rgb, depth, labels, ln = render_room(
        h, w, room_depth=5.0, room_width=7.0, room_height=2.8, cam_height=1.6,
        floor_color=(150, 110, 70), wall_color=(205, 200, 190),
        fov_deg=90,
    )
    images.append(TestImage("06_wide_angle", rgb, depth, labels, ln,
                            make_intrinsics(h, w, 90), "Wide-angle corner view, 3 walls"))

    # 7. Cluttered — many objects
    rgb, depth, labels, ln = render_room(
        h, w, room_depth=4.0, room_width=5.0, room_height=2.7, cam_height=1.6,
        floor_color=(140, 105, 65), wall_color=(210, 205, 195),
        objects=[
            ("chair", -1.0, -0.5, 0.2, 0.8, 1.5, (90, 70, 50)),
            ("plant", 0.8, 1.1, 0.0, 1.0, 1.8, (40, 100, 40)),
            ("lamp", -0.1, 0.1, 1.0, 1.8, 2.0, (200, 180, 120)),
            ("rug", -0.8, 0.8, 0.0, 0.02, 1.2, (160, 50, 50)),
            ("shelf", 1.5, 2.0, 0.5, 2.0, 3.5, (100, 75, 45)),
        ],
    )
    images.append(TestImage("07_cluttered", rgb, depth, labels, ln,
                            make_intrinsics(h, w), "Cluttered room with many objects"))

    # 8. Hallway — long narrow, strong perspective
    rgb, depth, labels, ln = render_room(
        h, w, room_depth=10.0, room_width=2.0, room_height=2.8, cam_height=1.6,
        floor_color=(170, 160, 145), wall_color=(215, 210, 200),
        objects=[
            ("door", -0.5, 0.5, 0.0, 2.1, 9.5, (110, 80, 50)),
        ],
    )
    images.append(TestImage("08_hallway", rgb, depth, labels, ln,
                            make_intrinsics(h, w), "Long hallway with strong perspective"))

    # --- 9-11: Failure modes ---

    # 9. Glass coffee table — glass surface invisible to depth
    rgb_glass, depth_glass, labels_glass, ln_glass = render_room(
        h, w, room_depth=5.0, room_width=6.0, room_height=2.8, cam_height=1.6,
        floor_color=(140, 100, 60), wall_color=(210, 205, 195),
        objects=[
            ("sofa", -1.5, 1.5, 0.3, 1.0, 3.5, (80, 60, 50)),
            ("glass table", -1.2, 1.2, 0.5, 1.2, 1.5, (200, 220, 230)),
        ],
    )
    # Glass table: depth sees THROUGH it to floor — simulate by overwriting
    # glass table region depth with floor depth
    glass_mask = labels_glass == 11  # glass table label
    if glass_mask.any():
        # Replace glass depth with floor depth at those pixels (simulating depth failure)
        floor_depth_at_glass = depth_glass.copy()
        for py in range(h):
            for px in range(w):
                if glass_mask[py, px]:
                    floor_depth_at_glass[py, px] = depth_glass[py, px] * 1.5  # floor is behind
        depth_glass_corrupted = depth_glass.copy()
        depth_glass_corrupted[glass_mask] = floor_depth_at_glass[glass_mask]
    else:
        depth_glass_corrupted = depth_glass

    images.append(TestImage("09_glass_table", rgb_glass, depth_glass_corrupted, labels_glass, ln_glass,
                            make_intrinsics(h, w), "Room with glass coffee table (depth sees through)"))

    # 10. Mirror wall — depth shows phantom room behind mirror
    rgb_mirror, depth_mirror, labels_mirror, ln_mirror = render_room(
        h, w, room_depth=5.0, room_width=6.0, room_height=2.8, cam_height=1.6,
        floor_color=(140, 100, 60), wall_color=(210, 205, 195),
        objects=[
            ("mirror", -1.5, 1.5, 0.3, 2.2, 4.8, (180, 190, 200)),
        ],
    )
    # Mirror: depth extends behind the wall — simulate phantom room
    mirror_mask = labels_mirror == 10
    depth_mirror_corrupted = depth_mirror.copy()
    depth_mirror_corrupted[mirror_mask] = depth_mirror[mirror_mask] + 5.0  # phantom room 5m behind

    images.append(TestImage("10_mirror_wall", rgb_mirror, depth_mirror_corrupted, labels_mirror, ln_mirror,
                            make_intrinsics(h, w), "Room with large mirror (phantom room in depth)"))

    # 11. Dark room — same geometry as living room but very dark
    rgb_dark = images[0].rgb.copy()
    rgb_dark = (rgb_dark.astype(np.float32) * 0.08).astype(np.uint8)  # brightness ~20/255
    depth_dark = images[0].depth.copy()
    # Add heavy noise to simulate dark-scene depth failure
    noise = np.random.normal(0, 0.3, depth_dark.shape).astype(np.float32)
    depth_dark_noisy = np.maximum(depth_dark + noise, 0.1)

    images.append(TestImage("11_dark_room", rgb_dark, depth_dark_noisy,
                            images[0].labels.copy(), dict(images[0].label_names),
                            make_intrinsics(h, w), "Dark version of living room (brightness < 30)"))

    return images


def main():
    output_dir = Path("data/test_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating 11 synthetic test images...")
    images = generate_all()

    for img in images:
        img.save(str(output_dir))
        mean_brightness = img.rgb.mean()
        valid_depth = img.depth[img.depth > 0]
        n_labels = len(set(img.labels.flatten()))
        has_door = any("door" in v for v in img.label_names.values())
        print(f"  {img.name:25s}  {img.rgb.shape[1]}x{img.rgb.shape[0]}  "
              f"brightness={mean_brightness:.0f}  depth={valid_depth.mean():.1f}m  "
              f"labels={n_labels}  door={has_door}  {img.description}")

    print(f"\nSaved to {output_dir}/")
    print(f"Total: {len(images)} images")
    return images


if __name__ == "__main__":
    main()
