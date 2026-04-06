#!/usr/bin/env python3
"""
Train the Voxel Diffusion Transformer.

Usage:
    # Generate training pairs from existing photos
    python train_dit.py --step data --photo_dir data/room_photos --grid_size 64

    # Train the model
    python train_dit.py --step train --epochs 100 --batch_size 4

    # Generate a voxel world from a photo
    python train_dit.py --step generate --image test_room.jpg

    # Full pipeline
    python train_dit.py --step all --epochs 50
"""

import argparse, os, sys, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Trivima Voxel DiT Training")
    parser.add_argument("--step", type=str, default="all",
                        choices=["data", "train", "generate", "all"])
    parser.add_argument("--photo_dir", type=str, default="data/room_photos")
    parser.add_argument("--data_dir", type=str, default="data/dit_pairs")
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dit_best.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  Trivima — Voxel Diffusion Transformer")
    print("=" * 60)

    if args.step in ("data", "all"):
        print("\n[Step 1] Generating training pairs...")
        from trivima.diffusion.data_pipeline import batch_generate_pairs
        n = batch_generate_pairs(
            args.photo_dir, args.data_dir,
            grid_size=args.grid_size, device=args.device,
            max_images=args.max_images)
        print(f"  Total pairs: {n}")
        torch.cuda.empty_cache()

    if args.step in ("train", "all"):
        print("\n[Step 2] Training DiT...")
        from trivima.diffusion.dit_model import VoxelDiTTrainer
        from trivima.diffusion.data_pipeline import VoxelPairDataset

        dataset = VoxelPairDataset(args.data_dir, grid_size=args.grid_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)

        trainer = VoxelDiTTrainer(grid_size=args.grid_size, device=args.device,
                                  lr=args.lr)

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("logs/dit_samples", exist_ok=True)
        best_loss = float("inf")

        for epoch in range(args.epochs):
            t0 = time.time()
            epoch_losses = []

            for batch_idx, (photos, grids) in enumerate(loader):
                photos = photos.to(args.device)
                grids = grids.to(args.device)

                losses = trainer.train_step(photos, grids)
                epoch_losses.append(losses["loss"])

                if (batch_idx + 1) % 10 == 0:
                    avg = np.mean(epoch_losses[-10:])
                    print(f"    [{epoch+1}/{args.epochs}] batch {batch_idx+1}/{len(loader)} loss={avg:.4f}")

            avg_loss = np.mean(epoch_losses)
            dt = time.time() - t0
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} ({dt:.1f}s)")

            if avg_loss < best_loss:
                best_loss = avg_loss
                trainer.save(args.checkpoint)
                print(f"  -> New best: {best_loss:.4f}")

            # Generate sample every 10 epochs
            if (epoch + 1) % 10 == 0 and len(dataset) > 0:
                photo_sample = dataset[0][0].unsqueeze(0).to(args.device)
                grid_pred = trainer.generate(photo_sample, n_steps=20)
                # Save density slice
                density = grid_pred[0, 3].cpu().numpy()
                mid = density.shape[0] // 2
                slice_img = (np.clip(density[mid], 0, 1) * 255).astype(np.uint8)
                Image.fromarray(slice_img).save(
                    f"logs/dit_samples/epoch_{epoch+1:03d}_density_slice.png")

        print(f"\n  Training complete. Best loss: {best_loss:.4f}")

    if args.step in ("generate", "all"):
        img_path = args.image or "test_room.jpg"
        if not os.path.exists(img_path):
            print(f"\n  No image at {img_path}, skipping generation")
            return

        print(f"\n[Step 3] Generating voxel world from {img_path}...")
        from trivima.diffusion.dit_model import VoxelDiTTrainer
        from trivima.diffusion.data_pipeline import cells_to_voxel_grid

        if not os.path.exists(args.checkpoint):
            print(f"  No checkpoint at {args.checkpoint}")
            return

        trainer = VoxelDiTTrainer(grid_size=args.grid_size, device=args.device)
        trainer.load(args.checkpoint)

        photo = Image.open(img_path).convert("RGB").resize((256, 256), Image.LANCZOS)
        photo_np = np.array(photo).astype(np.float32) / 255.0
        photo_tensor = torch.from_numpy(photo_np.transpose(2, 0, 1)).unsqueeze(0).to(args.device)

        t0 = time.time()
        grid = trainer.generate(photo_tensor, n_steps=50)
        dt = time.time() - t0
        print(f"  Generated {args.grid_size}^3 grid in {dt:.1f}s")

        # Convert grid to cells for rendering
        grid_np = grid[0].cpu().numpy()
        # Denormalize from [-1,1] to [0,1]
        grid_np = (grid_np + 1) / 2

        # Extract cells from dense grid
        density = grid_np[3]
        occupied = density > 0.3  # threshold

        n_occupied = occupied.sum()
        print(f"  Occupied voxels: {n_occupied} / {args.grid_size**3} ({100*n_occupied/args.grid_size**3:.1f}%)")

        # Save grid
        os.makedirs("output_dit", exist_ok=True)
        np.save("output_dit/generated_grid.npy", grid_np)

        # Visualize density slices
        for axis_name, axis in [("xy", 0), ("xz", 1), ("yz", 2)]:
            mid = args.grid_size // 2
            if axis == 0:
                s = density[mid]
            elif axis == 1:
                s = density[:, mid]
            else:
                s = density[:, :, mid]
            s_img = (np.clip(s, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(s_img).save(f"output_dit/slice_{axis_name}.png")

        print(f"  Saved to output_dit/")

    print("\nDone!")


if __name__ == "__main__":
    main()
