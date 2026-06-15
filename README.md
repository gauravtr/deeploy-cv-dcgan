# Deeploy CV — DCGAN Image Synthesis

A complete, end-to-end **Deep Convolutional GAN (DCGAN)** that learns to synthesize realistic 64×64 images from random noise, trained on CIFAR-10 (50,000+ images, auto-downloaded).

Built with a clean modular architecture: `config.py` → `dataset.py` → `models.py` → `train.py` → `generate.py`.

## How it works

- **Generator**: maps a 100-dim noise vector to a 64×64 image using 5 transposed-convolution blocks (BatchNorm + ReLU, Tanh output).
- **Discriminator**: classifies images as real/fake using 5 strided-convolution blocks (BatchNorm + LeakyReLU, Sigmoid output).
- Both networks are trained adversarially with **BCE loss** and **Adam** (lr=2e-4, β1=0.5), following the DCGAN paper (Radford et al., 2016), including its weight-initialization scheme.
- A **fixed noise vector** is used to render the same 64 samples every epoch, so you can visually track the generator improving over time.

## Project structure

```
config.py      # all hyperparameters in one place
dataset.py     # CIFAR-10 / Fashion-MNIST loading + preprocessing
models.py      # Generator, Discriminator, DCGAN weight init
train.py       # adversarial training loop, sample grids, loss curves
generate.py    # generate new images from a trained checkpoint
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Outputs per epoch:
- `outputs/epoch_XXX.png` — 8×8 grid of generated samples
- `checkpoints/gen_XXX.pth` — generator weights (every 5 epochs)
- `outputs/loss_curve.png` — G/D loss curves at the end

~25 epochs gives clearly recognizable structure; 50+ improves texture quality. On a GPU (Colab T4) an epoch takes ~30–60s; on CPU expect several minutes per epoch.

## Generate new images

```bash
python generate.py --checkpoint checkpoints/gen_025.pth --num 64
```

## Switch datasets

In `config.py`, set `DATASET = "FashionMNIST"` for a faster, grayscale run — nothing else needs to change.

## Sample results

| Epoch 1 | Epoch 10 | Epoch 25 |
|---|---|---|
| noise | rough shapes | recognizable objects |

(Add your own `outputs/epoch_001.png`, `epoch_010.png`, `epoch_025.png` here after training.)

## References

- Radford, Metz, Chintala — *Unsupervised Representation Learning with Deep Convolutional GANs* (2016)
