# -*- coding: utf-8 -*-
"""
GAN CAPTCHA Generator — Pure PyTorch implementation (no Lightning dependency)
"""

import os, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────
CAPTCHA_DIR = "captcha-getmean_std_labels/captcha-getmean_std_labels/class1"
OUTPUT_DIR  = "generated_samples"
BATCH_SIZE  = 128
LATENT_DIM  = 100
NUM_EPOCHS  = 100
LR          = 0.0002
BETA1       = 0.25
BETA2       = 0.999
DEVICE      = torch.device("cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ─────────────────────────────────────────────────
class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = [f for f in os.listdir(data_dir)
                       if os.path.isfile(os.path.join(data_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# ── Generator ───────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.init_size = 24 // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size * (72 // 4)))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, 72 // 4)
        return self.conv_blocks(out)

# ── Discriminator ───────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(900, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.reshape(-1, 900)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)

# ── Save image grid ────────────────────────────────────────
def save_sample_grid(generator, epoch, fixed_z, nrow=4):
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_z)
        grid = torchvision.utils.make_grid(samples, nrow=nrow, normalize=True, value_range=(-1, 1))
        path = os.path.join(OUTPUT_DIR, f"epoch_{epoch:03d}.png")
        torchvision.utils.save_image(grid, path)
    generator.train()
    return path

# ── Training ────────────────────────────────────────────────
def train():
    print(f"PyTorch {torch.__version__}  |  Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}  |  Latent dim: {LATENT_DIM}")
    print(f"LR: {LR}  beta1: {BETA1}  beta2: {BETA2}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    ])
    full_ds = CaptchaDataset(CAPTCHA_DIR, transform=transform)
    n = len(full_ds)
    train_n = int(n * 0.7)
    val_n   = int(n * 0.2)
    test_n  = n - train_n - val_n
    train_ds, temp = random_split(full_ds, [train_n, n - train_n])
    val_ds, test_ds = random_split(temp, [val_n, test_n])
    print(f"Dataset: {n} total | train={train_n} val={val_n} test={test_n}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # Models
    G = Generator(LATENT_DIM).to(DEVICE)
    D = Discriminator().to(DEVICE)
    print(f"Generator params:     {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

    loss_fn = nn.BCELoss()

    # Fixed noise for tracking progress
    fixed_z = torch.randn(8, LATENT_DIM, device=DEVICE)

    g_losses, d_losses = [], []

    print("\n=== Training ===")
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        G.train(); D.train()
        epoch_g_loss, epoch_d_loss = 0.0, 0.0
        n_batches = 0

        for real_imgs in train_loader:
            real_imgs = real_imgs.to(DEVICE)
            bs = real_imgs.size(0)
            real_label = torch.ones(bs, 1, device=DEVICE)
            fake_label = torch.zeros(bs, 1, device=DEVICE)
            z = torch.randn(bs, LATENT_DIM, device=DEVICE)

            # ── Train Generator ──
            opt_G.zero_grad()
            fake_imgs = G(z)
            g_loss = loss_fn(D(fake_imgs), real_label)
            g_loss.backward()
            opt_G.step()

            # ── Train Discriminator ──
            opt_D.zero_grad()
            real_loss = loss_fn(D(real_imgs), real_label)
            fake_loss = loss_fn(D(fake_imgs.detach()), fake_label)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_D.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            n_batches += 1

        avg_g = epoch_g_loss / n_batches
        avg_d = epoch_d_loss / n_batches
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            elapsed = time.time() - t0
            path = save_sample_grid(G, epoch, fixed_z)
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  |  G: {avg_g:.4f}  D: {avg_d:.4f}  |  {elapsed:.0f}s  |  {path}")

    # ── Final generation ──
    print("\n=== Generating final samples ===")
    G.eval()
    with torch.no_grad():
        final_z = torch.randn(16, LATENT_DIM, device=DEVICE)
        final_samples = G(final_z)
        grid = torchvision.utils.make_grid(final_samples, nrow=4, normalize=True, value_range=(-1, 1))
        final_path = os.path.join(OUTPUT_DIR, "final_generated.png")
        torchvision.utils.save_image(grid, final_path)
        print(f"Final: {final_path}")

    # ── Plot loss curves ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g_losses, label="Generator Loss", color="#e74c3c")
    ax.plot(d_losses, label="Discriminator Loss", color="#3498db")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("GAN Training Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    loss_plot_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    fig.savefig(loss_plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss plot: {loss_plot_path}")

    total_time = time.time() - t0
    print(f"\nDone. Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

if __name__ == "__main__":
    train()
