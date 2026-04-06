"""
Training loop for Multi-View DiT.

Usage:
    python -m trivima.multiview.train --data_dir data/realestate10k --epochs 50
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trivima.multiview.model import MultiViewDiT
from trivima.multiview.dataset import RealEstate10KDataset


class DDPMScheduler:
    def __init__(self, n_steps=1000):
        self.n_steps = n_steps
        betas = torch.linspace(1e-4, 0.02, n_steps)
        self.alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, noise, t):
        a = self.alpha_cumprod[t.long()].view(-1, 1, 1, 1, 1)
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

    def to(self, device):
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        return self


def train(args):
    device = "cuda"

    # Model
    model = MultiViewDiT(
        num_views=args.num_views,
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
    ).to(device)

    n_params = model.count_params()
    print("Model: %.1fM parameters" % (n_params / 1e6))
    print("Views: %d, img_size: %d, embed_dim: %d, depth: %d" % (
        args.num_views, args.img_size, args.embed_dim, args.depth))

    # Data
    dataset = RealEstate10KDataset(
        args.data_dir, num_target_views=args.num_views, img_size=args.img_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    if len(dataset) == 0:
        print("ERROR: No training data found in %s" % args.data_dir)
        print("Run the data preparation script first.")
        return

    # Training
    scheduler = DDPMScheduler().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")
    history = []

    print("\nTraining on %d pairs, %d batches/epoch" % (len(dataset), len(loader)))

    for epoch in range(args.epochs):
        t_epoch = time.time()
        epoch_losses = []

        model.train()
        for batch_idx, batch in enumerate(loader):
            input_img = batch["input"].to(device)       # (B, 3, H, W)
            targets = batch["targets"].to(device)         # (B, N, 3, H, W)
            poses = batch["poses"].to(device)             # (B, N, 4, 4)

            B = input_img.shape[0]

            # Random timestep
            t = torch.randint(0, scheduler.n_steps, (B,), device=device)

            # Add noise to targets
            noise = torch.randn_like(targets)
            noisy = scheduler.add_noise(targets, noise, t)

            # Predict noise
            noise_pred = model(input_img, noisy, t.float(), poses)

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % 10 == 0:
                avg = np.mean(epoch_losses[-10:])
                print("  [%d/%d] batch %d/%d loss=%.4f" % (
                    epoch+1, args.epochs, batch_idx+1, len(loader), avg))

        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        dt = time.time() - t_epoch
        history.append({"epoch": epoch+1, "loss": avg_loss, "time": dt})
        print("Epoch %d/%d: loss=%.4f (%.1fs)" % (epoch+1, args.epochs, avg_loss, dt))

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "loss": best_loss,
            }, os.path.join(args.output_dir, "best.pt"))
            print("  -> New best: %.4f" % best_loss)

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1,
            }, os.path.join(args.output_dir, "epoch_%03d.pt" % (epoch+1)))

    # Save history
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete. Best loss: %.4f" % best_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/realestate10k")
    parser.add_argument("--output_dir", type=str, default="checkpoints/multiview")
    parser.add_argument("--num_views", type=int, default=8)  # Start small
    parser.add_argument("--img_size", type=int, default=128)  # Start small
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    train(args)
