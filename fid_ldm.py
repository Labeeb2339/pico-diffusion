"""Internal FID-style score for the latent diffusion model.

Reuses the FID machinery from ``fid.py`` (Inception features, dependency-free
    Fréchet distance, sanity check) but samples by denoising latents and decoding
    them through the trained VAE. Scores are comparable within this repository's
    fixed harness, not to canonical 50,000-sample CIFAR-10 FID from papers.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from diffusion import GaussianDiffusion
from fid import (
    compute_activations,
    fid_from_activations,
    get_inception,
    load_real_cifar,
)
from model import UNet
from vae import VAE


@torch.no_grad()
def generate_ldm(
    vae_ckpt,
    ldm_ckpt,
    n,
    latent_ch,
    device,
    steps=50,
    latent_stats=None,
    sample_batch_size=64,
    seed=0,
):
    if n < 1 or sample_batch_size < 1:
        raise ValueError("n and sample_batch_size must be at least 1")
    vae = VAE(in_channels=3, latent_channels=latent_ch).to(device)
    vae.load_state_dict(
        torch.load(vae_ckpt, map_location=device, weights_only=True)["model"]
    )
    vae.eval()
    model = UNet(
        in_channels=latent_ch,
        base_ch=64,
        image_size=8,
        ch_mults=(1, 2, 2),
        attn_res=(8, 4),
    ).to(device)
    ck = torch.load(ldm_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    model.eval()

    diffusion = GaussianDiffusion().to(device)
    stats = None
    if latent_stats:
        stats = torch.load(latent_stats, map_location=device, weights_only=True)

    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
    batches = []
    for start in range(0, n, sample_batch_size):
        batch_n = min(sample_batch_size, n - start)
        z = diffusion.ddim_sample(
            model,
            (batch_n, latent_ch, 8, 8),
            device,
            sampling_steps=steps,
        )
        if stats is not None:
            z = z * stats["std"] + stats["mean"]  # de-normalize
        x = vae.decode(z)
        batches.append(torch.clamp((x + 1) / 2, 0, 1).cpu())
    return torch.cat(batches, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-ckpt", required=True)
    ap.add_argument("--ldm-ckpt", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--latent-channels", type=int, default=4)
    ap.add_argument(
        "--latent-stats", default=None, help="latent_stats.pt for de-normalization"
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--sample-batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = get_inception(device)

    print(f"generating {args.n} latent-diffusion samples ...")
    fake = generate_ldm(
        args.vae_ckpt,
        args.ldm_ckpt,
        args.n,
        args.latent_channels,
        device,
        args.steps,
        args.latent_stats,
        args.sample_batch_size,
        args.seed,
    )

    print("loading real CIFAR-10 test images ...")
    real = load_real_cifar(args.n)

    print("extracting features ...")
    act_fake = compute_activations(inception, fake, device, args.batch_size)
    act_real = compute_activations(inception, real, device, args.batch_size)

    fid = fid_from_activations(act_real, act_fake)

    mu_r, sig_r = act_real.mean(axis=0), np.cov(act_real, rowvar=False)
    from fid import frechet_distance

    fid_self = frechet_distance(mu_r, sig_r, mu_r, sig_r)

    print(f"\nInternal LDM FID-style score (n={args.n}): {fid:.2f}")
    print(f"sanity FID(real, real) = {fid_self:.5f}  (must be ~0)")


if __name__ == "__main__":
    main()
