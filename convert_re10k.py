"""Convert pixelSplat RE10K format to our multi-view training format."""
import torch
import numpy as np
import json
import os
import io
import glob
from PIL import Image

DATA_ROOT = "/workspace/data/re10k_extracted/re10k_subset"
OUTPUT_DIR = "/workspace/data/realestate10k/train"
NUM_VIEWS = 8
IMG_SIZE = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

pair_count = 0
total_scenes = 0

for split in ["train", "test"]:
    torch_files = sorted(glob.glob(os.path.join(DATA_ROOT, split, "*.torch")))
    for tf in torch_files:
        scenes = torch.load(tf, map_location="cpu", weights_only=False)
        for scene in scenes:
            n_frames = len(scene["images"])
            if n_frames < NUM_VIEWS + 1:
                continue

            total_scenes += 1

            # Decode images
            images = []
            for img_bytes in scene["images"]:
                img_data = img_bytes.numpy().tobytes()
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                images.append(img)

            cameras = scene["cameras"].numpy()  # (N, 18)

            # Create 3 pairs per scene
            step = max(1, n_frames // 3)
            for pair_idx in range(min(3, n_frames // (NUM_VIEWS + 1))):
                input_idx = pair_idx * step

                # Pick target views evenly spaced
                remaining = [i for i in range(n_frames) if i != input_idx]
                view_step = max(1, len(remaining) // NUM_VIEWS)
                target_idx = remaining[::view_step][:NUM_VIEWS]

                if len(target_idx) < NUM_VIEWS:
                    continue

                pair_dir = os.path.join(OUTPUT_DIR, "scene_%04d" % total_scenes, "pair_%03d" % pair_idx)
                os.makedirs(pair_dir, exist_ok=True)

                # Save input
                images[input_idx].save(os.path.join(pair_dir, "input.png"))

                # Save targets + poses
                poses = []
                for vi, ti in enumerate(target_idx):
                    images[ti].save(os.path.join(pair_dir, "target_%02d.png" % vi))

                    # Camera: 18 values = [fx, fy, cx, cy, near, far, 12 extrinsics]
                    cam = cameras[ti]
                    ext = cam[6:18].reshape(3, 4)
                    pose = np.eye(4)
                    pose[:3, :4] = ext
                    poses.append(pose.tolist())

                cam_data = {"poses": poses}
                with open(os.path.join(pair_dir, "cameras.json"), "w") as f:
                    json.dump(cam_data, f)

                pair_count += 1

    print("%s: %d scenes processed" % (split, total_scenes))

print("\nTotal: %d scenes, %d training pairs" % (total_scenes, pair_count))
print("Output: %s" % OUTPUT_DIR)
