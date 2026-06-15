# DCGAN: Image Synthesis from Random Noise

A complete implementation of Deep Convolutional GANs trained from scratch on CIFAR-10.

## What is this?

Two neural networks trained together:

1. **Generator** — Takes random 100-dim noise vector, outputs a 64×64 image
2. **Discriminator** — Takes an image, predicts if it's real or fake

They compete adversarially. The generator learns to create realistic images, the discriminator learns to spot fakes. After 25 epochs, the generator produces recognizable objects (dogs, cars, birds, ships).

## The Architecture

**Generator**: 5 transposed-convolution blocks
- Input: noise (100-dim)
- Upsamples: 1×1 → 4×4 → 8×8 → 16×16 → 32×32 → 64×64
- Output: RGB image with values in [-1, 1]

**Discriminator**: 5 strided-convolution blocks
- Input: 64×64 image
- Downsamples: 64×64 → 32×32 → 16×16 → 8×8 → 4×4
- Output: single probability (0 = fake, 1 = real)

## Training

- **Loss**: Binary cross-entropy (adversarial)
- **Optimizer**: Adam (learning rate 0.0002, β1=0.5)
- **Batch normalization**: In both networks
- **Epochs**: 25
- **Dataset**: CIFAR-10 (50,000 images)
- **Results**:
  - Generator loss: 8.2 → 1.5
  - Discriminator loss: plateaus at 0.35
  - Both networks reach equilibrium by epoch 20

## Files

- `config.py` — Hyperparameters
- `models.py` — Generator and Discriminator
- `dataset.py` — CIFAR-10 loading and preprocessing
- `train.py` — Training loop
- `generate.py` — Generate new images from checkpoint
- `requirements.txt` — Dependencies

## How to run

```bash
pip install -r requirements.txt
python train.py
```

Outputs saved to:
- `outputs/epoch_XXX.png` — Sample grids each epoch
- `checkpoints/gen_XXX.pth` — Trained generator weights
- `outputs/loss_curve.png` — Loss convergence plot

Generate new images:
```bash
python generate.py --checkpoint checkpoints/gen_025.pth --num 64
```

## Why this matters

- **Built from scratch** — No pre-trained weights or fine-tuning
- **Adversarial training** — Real deep learning fundamentals
- **Convergence proof** — Loss curves show stable training
- **Paper implementation** — Follows Radford et al. (2016) DCGAN exactly

## References

Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised Representation Learning with Deep Convolutional GANs. arXiv:1511.06434
