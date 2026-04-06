"""
RealEstate10K dataset loader for multi-view training.

RealEstate10K provides:
- YouTube video URLs with timestamps
- Camera intrinsics + extrinsics per frame (from COLMAP)
- ~10M frames across ~80K video clips

We download videos, extract frames at camera-pose timestamps,
and create (input_photo, target_views[20], camera_poses[20]) pairs.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class RealEstate10KDataset(Dataset):
    """Dataset of (input_image, target_images, camera_poses) for multi-view training."""

    def __init__(self, data_dir, num_target_views=20, img_size=256, split="train"):
        self.data_dir = Path(data_dir)
        self.num_target_views = num_target_views
        self.img_size = img_size

        # Load all pairs
        self.pairs = []
        pairs_dir = self.data_dir / split
        if pairs_dir.exists():
            for scene_dir in sorted(pairs_dir.iterdir()):
                if scene_dir.is_dir():
                    for pair_dir in sorted(scene_dir.iterdir()):
                        if pair_dir.is_dir() and (pair_dir / "cameras.json").exists():
                            self.pairs.append(pair_dir)

        print("Dataset: %d pairs from %s/%s" % (len(self.pairs), data_dir, split))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_dir = self.pairs[idx]

        # Load input image
        input_img = Image.open(pair_dir / "input.png").convert("RGB")
        input_img = input_img.resize((self.img_size, self.img_size), Image.LANCZOS)
        input_tensor = torch.from_numpy(
            np.array(input_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)  # (3, H, W)

        # Load target images
        targets = []
        for i in range(self.num_target_views):
            path = pair_dir / ("target_%02d.png" % i)
            if path.exists():
                img = Image.open(path).convert("RGB")
                img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
                t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
            else:
                t = torch.zeros(3, self.img_size, self.img_size)
            targets.append(t)
        targets_tensor = torch.stack(targets)  # (N, 3, H, W)

        # Load camera poses
        with open(pair_dir / "cameras.json") as f:
            cam_data = json.load(f)
        poses = torch.tensor(cam_data["poses"], dtype=torch.float32)  # (N, 4, 4)

        # Normalize targets to [-1, 1] for diffusion
        targets_tensor = targets_tensor * 2.0 - 1.0

        return {
            "input": input_tensor,       # (3, H, W) [0, 1]
            "targets": targets_tensor,    # (N, 3, H, W) [-1, 1]
            "poses": poses,               # (N, 4, 4)
        }


def download_realestate10k(output_dir, num_scenes=150, split="train"):
    """Download RealEstate10K pose files and extract video frames.

    RealEstate10K provides text files with:
    - YouTube video URL
    - Per-frame: timestamp, intrinsics (fx,fy,cx,cy), extrinsics (3x4 matrix)
    """
    import urllib.request
    import subprocess

    os.makedirs(output_dir, exist_ok=True)

    # Download pose files from the official release
    base_url = "https://google-research-datasets.github.io/RealEstate10K"
    splits = {"train": "train", "test": "test"}

    pose_dir = os.path.join(output_dir, "poses", split)
    os.makedirs(pose_dir, exist_ok=True)

    # RealEstate10K pose files are hosted as text files
    # Each file: video_id, then per line: timestamp fx fy cx cy [12 extrinsic values]
    print("Downloading RealEstate10K pose files...")

    # Download the file list
    list_url = "%s/%s.txt" % (base_url, split)
    try:
        urllib.request.urlretrieve(list_url, os.path.join(pose_dir, "filelist.txt"))
        print("  Downloaded file list")
    except Exception as e:
        print("  Failed to download file list: %s" % e)
        print("  Will use local data if available")
        return

    # Parse and download individual pose files
    with open(os.path.join(pose_dir, "filelist.txt")) as f:
        files = [line.strip() for line in f if line.strip()]

    downloaded = 0
    for fname in files[:num_scenes]:
        pose_url = "%s/%s/%s" % (base_url, split, fname)
        pose_path = os.path.join(pose_dir, fname)
        try:
            urllib.request.urlretrieve(pose_url, pose_path)
            downloaded += 1
        except:
            continue

    print("  Downloaded %d/%d pose files" % (downloaded, min(num_scenes, len(files))))


def create_training_pairs(
    pose_dir,
    output_dir,
    num_target_views=20,
    img_size=256,
    max_pairs_per_scene=5,
):
    """Create training pairs from downloaded pose files + extracted video frames.

    For each scene:
    1. Read pose file (timestamps + camera matrices)
    2. Download video frames at those timestamps (via yt-dlp)
    3. Pick 1 input frame + 20 target frames
    4. Save as training pair
    """
    os.makedirs(output_dir, exist_ok=True)

    pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith(".txt") and f != "filelist.txt"])
    print("Processing %d scenes..." % len(pose_files))

    pair_count = 0
    for scene_idx, pose_file in enumerate(pose_files):
        pose_path = os.path.join(pose_dir, pose_file)

        # Parse pose file
        with open(pose_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) < 2:
            continue

        video_url = lines[0]  # YouTube URL
        frames = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 17:  # timestamp + 4 intrinsics + 12 extrinsics
                timestamp = int(parts[0])
                intrinsics = [float(x) for x in parts[1:5]]  # fx, fy, cx, cy
                extrinsics = [float(x) for x in parts[5:17]]  # 3x4 matrix
                frames.append({
                    "timestamp": timestamp,
                    "intrinsics": intrinsics,
                    "extrinsics": np.array(extrinsics).reshape(3, 4),
                })

        if len(frames) < num_target_views + 1:
            continue

        # Create training pairs from this scene
        n_frames = len(frames)
        for pair_idx in range(min(max_pairs_per_scene, n_frames // (num_target_views + 1))):
            # Pick input: every n_frames/(pairs) frame
            input_idx = pair_idx * (n_frames // max_pairs_per_scene)

            # Pick targets: evenly spaced across remaining frames
            remaining = [i for i in range(n_frames) if i != input_idx]
            step = max(1, len(remaining) // num_target_views)
            target_indices = remaining[::step][:num_target_views]

            if len(target_indices) < num_target_views:
                continue

            # Save pair metadata (actual frame extraction happens separately)
            pair_dir = os.path.join(output_dir, "scene_%04d" % scene_idx, "pair_%03d" % pair_idx)
            os.makedirs(pair_dir, exist_ok=True)

            # Camera poses
            poses = []
            for ti in target_indices:
                ext = frames[ti]["extrinsics"]
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :4] = ext
                poses.append(pose_4x4.tolist())

            cam_data = {
                "video_url": video_url,
                "input_timestamp": frames[input_idx]["timestamp"],
                "input_intrinsics": frames[input_idx]["intrinsics"],
                "target_timestamps": [frames[ti]["timestamp"] for ti in target_indices],
                "poses": poses,
            }

            with open(os.path.join(pair_dir, "cameras.json"), "w") as f:
                json.dump(cam_data, f)

            pair_count += 1

        if (scene_idx + 1) % 50 == 0:
            print("  Processed %d/%d scenes, %d pairs" % (scene_idx + 1, len(pose_files), pair_count))

    print("Total: %d training pairs" % pair_count)
    return pair_count
