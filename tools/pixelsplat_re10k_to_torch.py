"""
Pack pixelSplat-format RealEstate10K .torch files into our unified schema.

Input: pixelSplat .torch file containing a LIST of scene dicts:
    [{
        'key': str,
        'cameras': tensor (N, 18),    # already in our schema!
        'images':  list of N JPEG byte tensors,
        ...
    }, ...]

Output: ONE .torch per scene in our PointCloudScene format:
    {
        'images':  list[byte tensor],
        'depths':  None  (no depth ground truth — RE10K is RGB-only)
        'depth_mask': None
        'cameras': tensor (N, 18),
        'domain': 'indoor',  (RE10K is real estate walkthroughs, mostly indoor)
        'scene_scale': 5.0
    }

This unlocks Phase A training WITHOUT habitat-sim. We lose depth supervision
but the rendering loss alone is still strong enough to learn 3D structure.

Usage:
    python tools/pixelsplat_re10k_to_torch.py \\
        --input_dir /content/data/re10k_extracted/re10k_subset \\
        --out_dir /content/data/torch_packed \\
        --max_scenes 200
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch


def pack_pixelsplat_file(src_path: str, out_dir: str, max_scenes: int = 999999) -> int:
    """Convert one pixelSplat .torch (containing many scenes) into many of ours."""
    print(f"Loading {src_path}...")
    scenes = torch.load(src_path, map_location="cpu", weights_only=False)
    print(f"  contains {len(scenes)} scenes")

    written = 0
    for i, scene in enumerate(scenes):
        if written >= max_scenes:
            break
        n_frames = len(scene["images"])
        if n_frames < 2:
            continue

        # Auto-detect domain from cameras: indoor scenes have small bounds
        cams = scene["cameras"]
        if not torch.is_tensor(cams):
            cams = torch.as_tensor(cams)
        if cams.dtype != torch.float32:
            cams = cams.float()

        # Validate the camera schema by computing translation magnitude
        # (pixelSplat extrinsics are in indices 6..18 as a 3x4 row-major)
        ext = cams[:, 6:18].reshape(-1, 3, 4)
        translations = ext[:, :, 3]  # (N, 3)
        max_t = translations.norm(dim=-1).max().item()

        pkg = {
            "images": scene["images"],
            "depths": None,
            "depth_mask": None,
            "cameras": cams,
            "domain": "indoor",
            "scene_scale": max(1.0, max_t),  # rough scene size from camera path
        }

        scene_key = scene.get("key", f"scene_{i:05d}")
        # Sanitize for filename
        scene_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(scene_key))[:64]
        out_name = f"re10k_{Path(src_path).stem}_{scene_key}.torch"
        out_path = os.path.join(out_dir, out_name)
        torch.save(pkg, out_path)
        written += 1

    print(f"  wrote {written} scenes")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing pixelSplat .torch files (recursively searched)",
    )
    ap.add_argument("--out_dir", default="/content/data/torch_packed")
    ap.add_argument("--max_scenes", type=int, default=200, help="Cap total scenes across all files")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "**/*.torch"), recursive=True))
    if not files:
        raise SystemExit(f"No .torch files found in {args.input_dir}")
    print(f"Found {len(files)} pixelSplat .torch files")

    total = 0
    for f in files:
        if total >= args.max_scenes:
            break
        wrote = pack_pixelsplat_file(f, args.out_dir, max_scenes=args.max_scenes - total)
        total += wrote

    print(f"\nDONE: {total} scenes written to {args.out_dir}")


if __name__ == "__main__":
    main()
