"""
Dataset loading and preprocessing.
Auto-downloads CIFAR-10 (or Fashion-MNIST) — no manual dataset curation needed.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config


def get_dataloader():
    if config.CHANNELS == 3:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        normalize = transforms.Normalize((0.5,), (0.5,))

    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,  # scale to [-1, 1] to match generator's Tanh output
    ])

    if config.DATASET == "CIFAR10":
        dataset = datasets.CIFAR10(config.DATA_DIR, train=True,
                                   download=True, transform=transform)
    elif config.DATASET == "FashionMNIST":
        dataset = datasets.FashionMNIST(config.DATA_DIR, train=True,
                                        download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {config.DATASET}")

    loader = DataLoader(dataset,
                        batch_size=config.BATCH_SIZE,
                        shuffle=True,
                        num_workers=config.NUM_WORKERS,
                        pin_memory=torch.cuda.is_available(),
                        drop_last=True)
    print(f"[Data] {config.DATASET}: {len(dataset)} images, "
          f"{len(loader)} batches of {config.BATCH_SIZE}")
    return loader
