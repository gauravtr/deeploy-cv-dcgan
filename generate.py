"""
Generate new synthetic images from a trained generator.

Run:  python generate.py --checkpoint checkpoints/gen_025.pth --num 64
"""

import os
import argparse
import torch
from torchvision.utils import save_image

import config
from models import Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="path to a saved generator .pth file")
    parser.add_argument("--num", type=int, default=64,
                        help="number of images to generate")
    parser.add_argument("--out", default="outputs/generated.png",
                        help="output image grid path")
    args = parser.parse_args()

    device = config.DEVICE
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(args.checkpoint, map_location=device))
    gen.eval()

    noise = torch.randn(args.num, config.LATENT_DIM, 1, 1, device=device)
    with torch.no_grad():
        images = gen(noise)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_image(images, args.out, normalize=True, nrow=8)
    print(f"[Done] {args.num} synthetic images saved to {args.out}")


if __name__ == "__main__":
    main()
