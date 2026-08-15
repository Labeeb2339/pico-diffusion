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

### MNIST (verified)

U-Net with `base_ch=64` (6.6M params), cosine schedule, 1000 timesteps, bf16 AMP,
50 epochs (23,400 steps) on an RTX 5070 Laptop GPU (~77s/epoch).

**Training loss: 1.19 → 0.028** (final 500-step moving average):

![MNIST loss curve](assets/mnist_loss.png)

**Samples** — 16 digits from pure noise (DDIM, 50 steps), zero pretrained weights:

![MNIST samples](assets/mnist_samples.png)

### CIFAR-10 (verified)

U-Net with `base_ch=64` (6.64M params), cosine schedule, 1000 timesteps, bf16 AMP,
100 epochs (39,000 steps) on an RTX 5070 Laptop GPU (~67s/epoch).

**Training loss: 1.135 → 0.056** (final 500-step moving average):

![CIFAR-10 loss curve](assets/cifar_loss.png)

**Samples** — 16 images from pure noise (DDIM, 50 steps), zero pretrained weights:

![CIFAR-10 samples](assets/cifar_samples.png)

**FID = 53.23** (2,048 generated vs 2,048 real test images, InceptionV3 features;
dependency-free Fréchet distance with a `FID(real, real) ≈ 0` sanity check).

For scale: the original DDPM paper reports ~3.17 on CIFAR-10 using a much larger
model trained far longer, so 53.23 is an honest "it learns real structure
end-to-end" number, not a state-of-the-art claim.

### Class-conditioned generation + classifier-free guidance

`model.py` gains a class embedding (`num_classes`), `train.py` trains on CIFAR-10
labels with 10% label dropout, and `ddim_sample`/`sample.py` support
classifier-free guidance: `pred = uncond + w * (cond - uncond)` at each step.

```bash
# train a conditional model
python train.py --dataset cifar10 --epochs 100 --out-dir out_cifar_cond --cfg-scale 2.0 --cfg-dropout 0.1

# sample one class (e.g. class 3) with guidance
python sample.py --ckpt out_cifar_cond/ckpt.pt --num-classes 10 --class-idx 3 --cfg-scale 3.0
```

*(conditional results + FID added once training finishes)*

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
