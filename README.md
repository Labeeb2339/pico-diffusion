# 🎨 PicoDiffusion — image generation from scratch

**A diffusion model (DDPM + DDIM) implemented from first principles in PyTorch — no `diffusers`, no pretrained weights.**

A U-Net learns to denoise pure noise back into real images; DDIM turns that into a
fast deterministic sampler. Every line is written from scratch and benchmarked
honestly.

## What's inside

| Component | What it is |
|-----------|------------|
| **U-Net** (`model.py`) | Residual blocks with time conditioning, group norm + SiLU, self-attention at low resolutions, encoder-decoder with skip connections |
| **DDPM** (`diffusion.py`) | Forward (add-noise) process + cosine noise schedule + the reverse sampler (Ho et al., 2020) |
| **DDIM** (`diffusion.py`) | Deterministic `eta=0` sampler that trades steps for speed (Song et al., 2020) |
| **Training** (`train.py`) | Noise-prediction MSE loss, AdamW, EMA of weights, periodic sample + checkpoint saves |
| **Sampling** (`sample.py`) | Generate a grid of images from a checkpoint |

## How it works

1. **Forward process** — `x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε` turns any image into noise
   over `T` steps (cosine schedule).
2. **Training** — the U-Net is given a noisy image `x_t` and the timestep `t`, and
   learns to predict the noise `ε`. Loss is simple MSE against the true noise.
3. **DDPM reverse** — start from pure noise and repeatedly denoise using the
   posterior `p(x_{t-1} | x_t)`.
4. **DDIM reverse** — a non-Markovian shortcut that predicts `x_0` directly and
   re-noises, so 50 steps produce samples as good as 1000 DDPM steps.

## Results

*Trained on an NVIDIA RTX 5070 Laptop GPU. Results, loss curves, and sample grids
are filled in after each training run — see `out/` for the artifacts.*

<!-- results table + sample images go here after training -->

## Quickstart

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Train (CIFAR-10 or MNIST)
python train.py --dataset cifar10 --epochs 100
python train.py --dataset mnist --base-ch 32 --epochs 2   # fast smoke run

# Generate samples (DDIM, 50 steps)
python sample.py --ckpt out/ckpt.pt --channels 3 --n 16
```

## Files

```
model.py       # U-Net (residual blocks + attention + time embedding)
diffusion.py   # DDPM + DDIM (schedules, forward, reverse, loss)
train.py       # training loop with EMA + sampling
sample.py      # sample a grid from a checkpoint
```
