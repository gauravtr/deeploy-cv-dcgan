"""
DCGAN training loop.

Run:  python train.py

Saves:
  outputs/epoch_XXX.png        sample grid from a fixed noise vector (track progress)
  outputs/loss_curve.png       generator/discriminator loss curves
  checkpoints/gen_XXX.pth      generator weights
  checkpoints/disc_XXX.pth     discriminator weights
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dataset import get_dataloader
from models import Generator, Discriminator, weights_init


def main():
    torch.manual_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    device = config.DEVICE
    print(f"[Setup] Training on: {device}")

    loader = get_dataloader()

    gen = Generator().to(device)
    disc = Discriminator().to(device)
    gen.apply(weights_init)
    disc.apply(weights_init)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE,
                       betas=(config.BETA1, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE,
                       betas=(config.BETA1, 0.999))

    # Fixed noise -> watch the SAME 64 samples improve across epochs
    fixed_noise = torch.randn(64, config.LATENT_DIM, 1, 1, device=device)

    g_losses, d_losses = [], []

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        g_running, d_running = 0.0, 0.0

        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            b = real.size(0)
            real_labels = torch.ones(b, device=device)
            fake_labels = torch.zeros(b, device=device)

            # ---------- Train Discriminator ----------
            disc.zero_grad()
            # real batch
            out_real = disc(real)
            loss_d_real = criterion(out_real, real_labels)
            # fake batch
            noise = torch.randn(b, config.LATENT_DIM, 1, 1, device=device)
            fake = gen(noise)
            out_fake = disc(fake.detach())
            loss_d_fake = criterion(out_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # ---------- Train Generator ----------
            gen.zero_grad()
            out = disc(fake)                       # re-score fake (D was just updated)
            loss_g = criterion(out, real_labels)   # generator wants D to say "real"
            loss_g.backward()
            opt_g.step()

            g_running += loss_g.item()
            d_running += loss_d.item()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{config.EPOCHS}] Batch [{i}/{len(loader)}] "
                      f"D_loss: {loss_d.item():.4f}  G_loss: {loss_g.item():.4f}")

        g_losses.append(g_running / len(loader))
        d_losses.append(d_running / len(loader))
        print(f"[Epoch {epoch}] avg D_loss: {d_losses[-1]:.4f}  "
              f"avg G_loss: {g_losses[-1]:.4f}  ({time.time() - t0:.1f}s)")

        # Save sample grid
        if epoch % config.SAMPLE_EVERY == 0:
            gen.eval()
            with torch.no_grad():
                samples = gen(fixed_noise)
            gen.train()
            save_image(samples,
                       os.path.join(config.OUTPUT_DIR, f"epoch_{epoch:03d}.png"),
                       normalize=True, nrow=8)

        # Save checkpoints
        if epoch % config.CHECKPOINT_EVERY == 0 or epoch == config.EPOCHS:
            torch.save(gen.state_dict(),
                       os.path.join(config.CHECKPOINT_DIR, f"gen_{epoch:03d}.pth"))
            torch.save(disc.state_dict(),
                       os.path.join(config.CHECKPOINT_DIR, f"disc_{epoch:03d}.pth"))

    # Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DCGAN Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "loss_curve.png"))
    print(f"[Done] Samples + loss curve in {config.OUTPUT_DIR}/, "
          f"weights in {config.CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
