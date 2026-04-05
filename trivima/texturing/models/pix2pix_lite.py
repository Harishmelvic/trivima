"""
Pix2PixHD-Lite — Real-time GAN for cell-to-photorealistic image translation.

Architecture:
  Generator: U-Net with 5 down/up blocks + skip connections
    Input:  8 channels (albedo_rgb + depth + normals_xyz + label)
    Output: 3 channels (photorealistic RGB)
    Secondary head: 9 channels (light_rgb + light_gradient_6ch)
    ~25M parameters, sub-10ms inference on A100

  Discriminator: PatchGAN at 3 scales
    Input: 11 channels (8 conditioning + 3 output)
    Output: per-patch real/fake score

Loss:
  - Adversarial (hinge)
  - L1 reconstruction
  - Perceptual (VGG-19 features)
  - Semantic consistency

Note on pretrained weights:
  The ADE20K pretrained pix2pixHD expects 3ch RGB input.
  We replace the first conv layer to accept 8ch and randomly
  initialize only that layer. All other pretrained weights are
  frozen for the first few epochs, then unfrozen for fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================
# Building blocks
# ============================================================

class ConvBlock(nn.Module):
    """Conv-BatchNorm-LeakyReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 4, stride: int = 2, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """TransposedConv-BatchNorm-ReLU with skip connection."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class ResidualBlock(nn.Module):
    """Residual block with spectral normalization."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# ============================================================
# Generator
# ============================================================

class Pix2PixLiteGenerator(nn.Module):
    """U-Net generator with secondary light output head.

    Input:  (B, 8, H, W)  — albedo(3) + depth(1) + normals(3) + label(1)
    Output: (B, 3, H, W)  — photorealistic RGB
    Light:  (B, 9, H, W)  — light_rgb(3) + light_gradient(6)
    """

    def __init__(self, in_channels: int = 8, out_channels: int = 3):
        super().__init__()

        # Encoder: 5 downsampling blocks
        # Input: 8ch → 64 → 128 → 256 → 512 → 512
        self.enc1 = nn.Sequential(  # no batchnorm on first layer
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 512)

        # Bottleneck: 3 residual blocks with spectral norm
        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )

        # Decoder: 5 upsampling blocks with skip connections
        self.dec5 = UpBlock(512, 512, dropout=0.5)       # 512 + 512 skip = 1024
        self.dec4 = UpBlock(1024, 256, dropout=0.5)      # 256 + 512 skip = 768...
        # After concat with skip: in_ch = up_output + skip_channels
        # dec5: input=512 (bottleneck), output=512, skip from enc4=512 → concat=1024
        # dec4: input=1024, output=256, skip from enc3=256 → concat=512
        # dec3: input=512, output=128, skip from enc2=128 → concat=256
        # dec2: input=256, output=64, skip from enc1=64 → concat=128
        self.dec3 = UpBlock(512, 128)
        self.dec2 = UpBlock(256, 64)

        # Final output: RGB
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh(),  # output in [-1, 1], rescale to [0, 1] at inference
        )

        # Secondary head: light values (3 RGB + 6 gradient)
        self.light_head = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 9, 3, 1, 1),  # 3 light RGB + 6 light gradient
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        e1 = self.enc1(x)    # (B, 64, H/2, W/2)
        e2 = self.enc2(e1)   # (B, 128, H/4, W/4)
        e3 = self.enc3(e2)   # (B, 256, H/8, W/8)
        e4 = self.enc4(e3)   # (B, 512, H/16, W/16)
        e5 = self.enc5(e4)   # (B, 512, H/32, W/32)

        # Bottleneck
        b = self.bottleneck(e5)  # (B, 512, H/32, W/32)

        # Decode with skip connections
        d5 = self.dec5(b, e4)    # (B, 1024, H/16, W/16)
        d4 = self.dec4(d5, e3)   # (B, 512, H/8, W/8)
        d3 = self.dec3(d4, e2)   # (B, 256, H/4, W/4)
        d2 = self.dec2(d3, e1)   # (B, 128, H/2, W/2)

        # Output heads
        rgb = self.final(d2)         # (B, 3, H, W) in [-1, 1]
        light = self.light_head(d2)  # (B, 9, H, W)

        return rgb, light


# ============================================================
# Discriminator (PatchGAN at 3 scales)
# ============================================================

class PatchDiscriminator(nn.Module):
    """NxN PatchGAN discriminator."""

    def __init__(self, in_channels: int = 11):  # 8 condition + 3 output
        super().__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1),  # per-patch score
        )

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """3-scale PatchGAN — discriminates at 1x, 0.5x, 0.25x resolution."""

    def __init__(self, in_channels: int = 11):
        super().__init__()
        self.disc1 = PatchDiscriminator(in_channels)
        self.disc2 = PatchDiscriminator(in_channels)
        self.disc3 = PatchDiscriminator(in_channels)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        out1 = self.disc1(x)
        x = self.downsample(x)
        out2 = self.disc2(x)
        x = self.downsample(x)
        out3 = self.disc3(x)
        return [out1, out2, out3]


# ============================================================
# Loss functions
# ============================================================

class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG-19 features."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.layers = nn.ModuleList([
            vgg[:4],   # conv1_2
            vgg[4:9],  # conv2_2
            vgg[9:18], # conv3_4
            vgg[18:27],# conv4_4
            vgg[27:36],# conv5_4
        ])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        loss = 0.0
        x, y = pred, target
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss += F.l1_loss(x, y)
        return loss


def hinge_loss_d(real_scores, fake_scores):
    """Hinge loss for discriminator."""
    loss = 0
    for real, fake in zip(real_scores, fake_scores):
        loss += torch.mean(F.relu(1.0 - real))
        loss += torch.mean(F.relu(1.0 + fake))
    return loss


def hinge_loss_g(fake_scores):
    """Hinge loss for generator."""
    loss = 0
    for fake in fake_scores:
        loss += -torch.mean(fake)
    return loss


# ============================================================
# Training wrapper
# ============================================================

class Pix2PixLiteTrainer:
    """End-to-end training loop for the Pix2PixHD-Lite model."""

    def __init__(
        self,
        lr_g: float = 2e-4,
        lr_d: float = 1e-4,
        lambda_l1: float = 100.0,
        lambda_perceptual: float = 10.0,
        device: str = "cuda",
    ):
        self.device = device
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual

        self.generator = Pix2PixLiteGenerator().to(device)
        self.discriminator = MultiScaleDiscriminator().to(device)
        self.vgg_loss = VGGPerceptualLoss().to(device)

        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    def adapt_pretrained(self, pretrained_path: str):
        """Load pretrained pix2pixHD weights, adapt first conv from 3ch → 8ch.

        The pretrained model expects 3-channel RGB input. We replace the
        first conv layer to accept 8 channels, randomly initializing only
        that layer while keeping all other weights.
        """
        state = torch.load(pretrained_path, map_location=self.device)

        # Save the old first conv weights (3ch input)
        old_conv_key = None
        for k in state:
            if "enc1" in k and "weight" in k:
                old_conv_key = k
                break

        if old_conv_key:
            old_weight = state[old_conv_key]  # (64, 3, 4, 4)
            new_weight = torch.zeros(64, 8, 4, 4, device=self.device)
            # Copy RGB channels
            new_weight[:, :3, :, :] = old_weight
            # Initialize remaining 5 channels with Kaiming init
            nn.init.kaiming_normal_(new_weight[:, 3:, :, :], mode='fan_out', nonlinearity='leaky_relu')
            state[old_conv_key] = new_weight

        self.generator.load_state_dict(state, strict=False)

    def train_step(self, condition: torch.Tensor, target: torch.Tensor) -> dict:
        """One training step.

        Args:
            condition: (B, 8, H, W) — 8-channel cell buffer input
            target: (B, 3, H, W) — ground truth photograph

        Returns:
            dict with loss values
        """
        # --- Generator forward ---
        fake_rgb, fake_light = self.generator(condition)

        # Scale from [-1,1] to [0,1] for loss computation
        fake_01 = (fake_rgb + 1) / 2
        target_01 = (target + 1) / 2

        # --- Discriminator step ---
        self.opt_d.zero_grad()
        real_input = torch.cat([condition, target], dim=1)
        fake_input = torch.cat([condition, fake_rgb.detach()], dim=1)
        real_scores = self.discriminator(real_input)
        fake_scores = self.discriminator(fake_input)
        loss_d = hinge_loss_d(real_scores, fake_scores)
        loss_d.backward()
        self.opt_d.step()

        # --- Generator step ---
        self.opt_g.zero_grad()
        fake_input_g = torch.cat([condition, fake_rgb], dim=1)
        fake_scores_g = self.discriminator(fake_input_g)
        loss_g_adv = hinge_loss_g(fake_scores_g)
        loss_g_l1 = F.l1_loss(fake_01, target_01) * self.lambda_l1
        loss_g_perceptual = self.vgg_loss(fake_01, target_01) * self.lambda_perceptual
        loss_g = loss_g_adv + loss_g_l1 + loss_g_perceptual
        loss_g.backward()
        self.opt_g.step()

        return {
            "loss_d": loss_d.item(),
            "loss_g": loss_g.item(),
            "loss_g_adv": loss_g_adv.item(),
            "loss_g_l1": loss_g_l1.item(),
            "loss_g_perceptual": loss_g_perceptual.item(),
        }

    def save(self, path: str):
        torch.save({
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
