# 🎨 PicoDiffusion — image generation from scratch

**A diffusion model (DDPM + DDIM) implemented from first principles in PyTorch — no `diffusers`, no pretrained weights.**

A U-Net learns to denoise pure noise back into real images; DDIM turns that into a
fast deterministic sampler. Every line is written from scratch and benchmarked
honestly.

> Part of a from-scratch ML systems series: [pico-kernels](https://github.com/Labeeb2339/pico-kernels) (Triton kernels) · [PicoLM](https://github.com/Labeeb2339/picolm) (a GPT from scratch) · [pico-engine](https://github.com/Labeeb2339/pico-engine) (a GGUF inference engine).

## What's inside

| Component | What it is |
|-----------|------------|
| **U-Net** (`model.py`) | Residual blocks with time conditioning, group norm + SiLU, self-attention at low resolutions, encoder-decoder with skip connections |
| **DDPM** (`diffusion.py`) | Forward (add-noise) process + cosine noise schedule + the reverse sampler (Ho et al., 2020) |
| **DDIM** (`diffusion.py`) | Deterministic `eta=0` sampler that trades steps for speed (Song et al., 2020) |
| **DPM-Solver++ 2M** (`diffusion.py`) | 2nd-order multistep ODE solver (Lu et al., 2022); its 1st-order step is *exactly* DDIM |
| **Latent diffusion** (`vae.py`, `train_ldm.py`) | VAE compresses 3×32×32 → 4×8×8 latent, then a U-Net denoises *latents* |
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

**Samples** — 20 images, 2 per class, classifier-free guidance `w=3.0`:

![CIFAR-10 conditional samples](assets/cifar_cond_samples.png)

![CIFAR-10 conditional loss curve](assets/cifar_cond_loss.png)

**FID = 39.44** (class-conditioned, CFG `w=2.0`) — down from **53.23** for the
unconditional model. The class embedding + guidance give the model a strong
"which class am I drawing" prior, so the images sit ~26% closer to the real
CIFAR-10 distribution.

```bash
# train a conditional model
python train.py --dataset cifar10 --epochs 100 --out-dir out_cifar_cond --cfg-scale 2.0 --cfg-dropout 0.1

# sample one class (e.g. class 3) with guidance
python sample.py --ckpt out_cifar_cond/ckpt.pt --num-classes 10 --class-idx 3 --cfg-scale 3.0
```

### Sampler study: DPM-Solver++ 2M (a useful failure case)

`diffusion.py` also implements **DPM-Solver++ 2M**, a 2nd-order multistep solver
of the probability-flow ODE. Its 1st-order step is algebraically *identical* to
DDIM (`eta=0`) — proven in the docstring and unit-tested — and the 2nd-order
term reuses the previous step's `x0` prediction.

The honest result on this model is that it does **not** beat DDIM:

| sampler        | FID @ 20 steps | FID @ 50 steps |
|----------------|----------------|----------------|
| DDIM (`eta=0`) | 67.64          | **53.23**      |
| DPM-Solver++ 2M| 73.58          | 64.02          |

Why: the solver needs an `x0 = (x − σ·ε)/α` clamp for stability (`α ≈ 1e-5` at
high `t`, so an unclamped `x0` explodes). That clamp is essential on a weak
model (FID 53), but it also *corrupts* the 2nd-order correction term, which
assumes a smooth `x0` trajectory — so the multistep correction hurts instead of
helping. The 2nd-order convergence is verified in isolation (a convergence-order
test confirms the error quarters when steps double), but on a model this weak it
doesn't pay off. The lesson: higher-order solvers need a well-conditioned model
(or thresholding-aware correction) to beat DDIM.

### Latent diffusion (Stable-Diffusion-style, verified end-to-end)

`vae.py` compresses 3×32×32 → 4×8×8 (12× compression) with an L1 reconstruction
+ weak KL (`β=1e-4`) objective; `train_ldm.py` then trains a *smaller* U-Net
(4.68M params) to denoise those latents instead of pixels.

Two honest numbers, both from the same harness:

| model | FID (n=2048) |
|-------|--------------|
| latent diffusion, **before** latent normalization | 143.43 |
| latent diffusion, **after** latent normalization | **105.36** |
| pixel-space DDIM (reference) | 53.23 |

**The bug the -38 points found:** a weak-KL VAE produces latents with per-channel
std 1.4–2.4 (and near-zero posterior σ — it collapses to a deterministic encoder),
but the cosine noise schedule assumes ~N(0,1) data. That mismatch badly distorts
the diffusion's signal-to-noise ratio. The fix — standardize the latents before
diffusing, un-standardize before decoding (exactly what Stable Diffusion does) —
dropped FID 143.43 → 105.36 *and* the training loss 0.45 → 0.34.

**The honest verdict:** latent diffusion is *implemented and working* end-to-end,
but it's the wrong tool at 32×32. The remaining gap vs 53.23 is the VAE's L1 blur
(no perceptual/adversarial loss, which would need a pretrained feature net —
out of scope for a from-scratch repo) plus a 12× bottleneck on an already
low-resolution image. Latent diffusion pays off at 256×256+ (memory/compute); at
CIFAR scale, pixel-space diffusion is strictly better. Shipped as a working
architecture with a real, measured bug-fix, not a FID winner.

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
diffusion.py   # DDPM + DDIM + DPM-Solver++ (schedules, forward, reverse, loss)
train.py       # training loop with EMA + sampling
sample.py      # sample a grid from a checkpoint
vae.py         # VAE (encoder/decoder) for latent diffusion
train_vae.py   # train the VAE
train_ldm.py   # train diffusion in the VAE latent space
sample_ldm.py  # sample latent diffusion + decode
fid.py         # Fréchet Inception Distance eval (dependency-free)
```
