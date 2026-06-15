"""
Central configuration for the DCGAN image synthesis project.
Change values here — no other file needs editing.
"""

import torch

# ----- Data -----
DATASET = "CIFAR10"          # "CIFAR10" or "FashionMNIST"
DATA_DIR = "./data"          # auto-downloaded here
IMAGE_SIZE = 64              # images resized to 64x64 (classic DCGAN size)
CHANNELS = 3 if DATASET == "CIFAR10" else 1

# ----- Model -----
LATENT_DIM = 100             # size of random noise vector z
GEN_FEATURES = 64            # base feature maps in generator
DISC_FEATURES = 64           # base feature maps in discriminator

# ----- Training -----
BATCH_SIZE = 128
EPOCHS = 25                  # 25 is enough for visible results; 50+ for better quality
LEARNING_RATE = 2e-4
BETA1 = 0.5                  # Adam beta1 (DCGAN paper value)
NUM_WORKERS = 2

# ----- Output -----
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./outputs"
SAMPLE_EVERY = 1             # save a sample grid every N epochs
CHECKPOINT_EVERY = 5         # save model weights every N epochs

# ----- Device -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
