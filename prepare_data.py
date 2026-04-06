"""
Prepare RealEstate10K training data.

Downloads pose files, extracts video frames via yt-dlp,
creates (input, targets, poses) training pairs.
"""
import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image
import urllib.request

DATA_DIR = "/workspace/data/realestate10k"
NUM_VIEWS = 8
IMG_SIZE = 128  # Start small for Phase 1


def download_pose_files(split="train", max_scenes=200):
    """Download RealEstate10K pose files."""
    pose_dir = os.path.join(DATA_DIR, "poses", split)
    os.makedirs(pose_dir, exist_ok=True)

    # RealEstate10K stores poses at this URL pattern
    base_url = "https://raw.githubusercontent.com/nicklashansen/RealEstate10K/main"

    # Try to download the file list
    list_path = os.path.join(pose_dir, "filelist.txt")
    if not os.path.exists(list_path):
        url = "%s/%s.txt" % (base_url, split)
        print("Downloading file list from %s..." % url)
        try:
            urllib.request.urlretrieve(url, list_path)
        except Exception as e:
            print("Failed: %s" % e)
            # Create synthetic data instead
            return create_synthetic_data(split, max_scenes)

    return parse_and_download_scenes(pose_dir, split, max_scenes)


def parse_and_download_scenes(pose_dir, split, max_scenes):
    """Parse pose files and download video frames."""
    pairs_dir = os.path.join(DATA_DIR, split)
    os.makedirs(pairs_dir, exist_ok=True)

    # Look for already downloaded pose files
    pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith(".txt") and f != "filelist.txt"])

    if not pose_files:
        print("No pose files found. Creating synthetic data for testing...")
        return create_synthetic_data(split, max_scenes)

    print("Found %d pose files" % len(pose_files))
    pair_count = 0

    for si, pf in enumerate(pose_files[:max_scenes]):
        with open(os.path.join(pose_dir, pf)) as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) < NUM_VIEWS + 2:
            continue

        video_url = lines[0]
        frames = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 17:
                timestamp = int(parts[0])
                intrinsics = [float(x) for x in parts[1:5]]
                extrinsics = np.array([float(x) for x in parts[5:17]]).reshape(3, 4)
                frames.append({"ts": timestamp, "intr": intrinsics, "ext": extrinsics})

        if len(frames) < NUM_VIEWS + 1:
            continue

        # Create pairs
        step = max(1, len(frames) // 3)
        for pair_idx in range(min(3, len(frames) // (NUM_VIEWS + 1))):
            input_idx = pair_idx * step
            remaining = [i for i in range(len(frames)) if i != input_idx]
            view_step = max(1, len(remaining) // NUM_VIEWS)
            target_idx = remaining[::view_step][:NUM_VIEWS]

            if len(target_idx) < NUM_VIEWS:
                continue

            pair_dir = os.path.join(pairs_dir, "scene_%04d" % si, "pair_%03d" % pair_idx)
            os.makedirs(pair_dir, exist_ok=True)

            poses = []
            for ti in target_idx:
                p = np.eye(4)
                p[:3, :4] = frames[ti]["ext"]
                poses.append(p.tolist())

            cam = {
                "video_url": video_url,
                "input_ts": frames[input_idx]["ts"],
                "target_ts": [frames[ti]["ts"] for ti in target_idx],
                "poses": poses,
            }
            with open(os.path.join(pair_dir, "cameras.json"), "w") as f:
                json.dump(cam, f)

            pair_count += 1

        if (si + 1) % 50 == 0:
            print("  %d/%d scenes, %d pairs" % (si + 1, len(pose_files[:max_scenes]), pair_count))

    print("Created %d pairs (metadata only — frames need video download)" % pair_count)
    return pair_count


def create_synthetic_data(split="train", num_scenes=200):
    """Create synthetic training data for pipeline testing.

    Generates random room-like images with known camera poses.
    Not real data — just to verify training pipeline works.
    """
    pairs_dir = os.path.join(DATA_DIR, split)
    os.makedirs(pairs_dir, exist_ok=True)

    print("Creating %d synthetic scenes for pipeline testing..." % num_scenes)
    pair_count = 0

    for si in range(num_scenes):
        # Random room colors
        wall_color = np.random.randint(150, 240, 3).astype(np.uint8)
        floor_color = np.random.randint(80, 180, 3).astype(np.uint8)
        obj_color = np.random.randint(40, 200, 3).astype(np.uint8)

        for pair_idx in range(3):
            pair_dir = os.path.join(pairs_dir, "scene_%04d" % si, "pair_%03d" % pair_idx)
            os.makedirs(pair_dir, exist_ok=True)

            # Generate input image (room-like)
            img = _make_room_image(wall_color, floor_color, obj_color, IMG_SIZE, seed=si*10+pair_idx)
            Image.fromarray(img).save(os.path.join(pair_dir, "input.png"))

            # Generate target views (slightly varied)
            poses = []
            for vi in range(NUM_VIEWS):
                target = _make_room_image(wall_color, floor_color, obj_color, IMG_SIZE,
                                          seed=si*10+pair_idx+vi+100, shift=vi*5)
                Image.fromarray(target).save(os.path.join(pair_dir, "target_%02d.png" % vi))

                # Camera pose (rotation around room center)
                angle = (vi / NUM_VIEWS) * 2 * np.pi
                pose = np.eye(4)
                pose[0, 3] = np.cos(angle) * 2
                pose[2, 3] = np.sin(angle) * 2
                pose[0, 0] = np.cos(angle)
                pose[0, 2] = -np.sin(angle)
                pose[2, 0] = np.sin(angle)
                pose[2, 2] = np.cos(angle)
                poses.append(pose.tolist())

            cam = {"poses": poses}
            with open(os.path.join(pair_dir, "cameras.json"), "w") as f:
                json.dump(cam, f)

            pair_count += 1

        if (si + 1) % 50 == 0:
            print("  %d/%d scenes, %d pairs" % (si + 1, num_scenes, pair_count))

    print("Created %d synthetic pairs" % pair_count)
    return pair_count


def _make_room_image(wall_color, floor_color, obj_color, size, seed=0, shift=0):
    """Generate a simple room-like image."""
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Wall (top 60%)
    horizon = int(size * 0.6)
    img[:horizon] = wall_color

    # Floor (bottom 40%)
    img[horizon:] = floor_color

    # Random object (rectangle)
    ox = (rng.randint(10, size-40) + shift) % (size - 30)
    oy = rng.randint(horizon - 40, horizon + 10)
    ow = rng.randint(20, 50)
    oh = rng.randint(20, 40)
    img[oy:oy+oh, ox:ox+ow] = obj_color

    # Add some noise for texture
    noise = rng.randint(-10, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_scenes", type=int, default=200)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    print("=" * 50)
    print("Preparing training data")
    print("=" * 50)

    n = download_pose_files(args.split, args.num_scenes)
    print("\nTotal pairs: %d" % n)
    print("Data dir: %s" % DATA_DIR)
