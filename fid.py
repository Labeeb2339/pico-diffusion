"""Fréchet Inception Distance (FID) evaluation for the diffusion model.

FID is the standard generative-quality metric: it measures the distance between
the InceptionV3 feature distributions of real and generated images. Lower is
better; a perfect generator scores ~0.

Uses torchvision's pretrained InceptionV3 (features from the final pooling
layer) and a dependency-free Frechet-distance (via eigendecomposition, no scipy).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import inception_v3

from diffusion import GaussianDiffusion
from model import UNet

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def get_inception(device) -> nn.Module:
    """InceptionV3 returning 2048-d features (the avgpool output)."""
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()
    return model.to(device).eval()


@torch.no_grad()
def compute_activations(model, images: torch.Tensor, device, batch_size: int = 64) -> np.ndarray:
    """Extract 2048-d Inception features for a batch of images in [0, 1]."""
    mean = _IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
    std = _IMAGENET_STD.to(device).view(1, 3, 1, 1)
    feats = []
    for i in range(0, images.shape[0], batch_size):
        x = images[i : i + batch_size].to(device)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        feats.append(model(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _sqrtm(mat: np.ndarray) -> np.ndarray:
    """Matrix square root of a symmetric PSD matrix (no scipy dependency)."""
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    diff = mu1 - mu2
    covmean = _sqrtm(sigma1 @ sigma2)
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def fid_from_activations(act1: np.ndarray, act2: np.ndarray) -> float:
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def load_real_cifar(n: int):
    """Load ``n`` real CIFAR-10 test images as a [0, 1] tensor."""
    # fast.ai mirror ships CIFAR-10 as ImageFolder (test/<class>/*.png)
    ds = torchvision.datasets.ImageFolder(root="./data/cifar10/test", transform=T.ToTensor())
    idx = np.random.default_rng(0).choice(len(ds), n, replace=False)
    return torch.stack([ds[i][0] for i in idx])


@torch.no_grad()
def generate_samples(ckpt_path, n, channels, image_size, device, steps=50):
    model = UNet(in_channels=channels, image_size=image_size).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    model.eval()
    diffusion = GaussianDiffusion().to(device)
    x = diffusion.ddim_sample(model, (n, channels, image_size, image_size), device, sampling_steps=steps)
    return torch.clamp((x + 1) / 2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="FID of a trained diffusion model vs CIFAR-10")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading inception (first run downloads weights) ...")
    inception = get_inception(device)

    print(f"generating {args.n} samples ...")
    fake = generate_samples(args.ckpt, args.n, args.channels, args.image_size, device, args.steps)

    print("loading real CIFAR-10 test images ...")
    real = load_real_cifar(args.n)

    print("extracting features ...")
    act_fake = compute_activations(inception, fake, device, args.batch_size)
    act_real = compute_activations(inception, real, device, args.batch_size)

    fid = fid_from_activations(act_real, act_fake)
    print(f"\nFID (n={args.n}): {fid:.2f}")


if __name__ == "__main__":
    main()
