"""Generate samples from a trained latent diffusion model (VAE decode)."""

from __future__ import annotations

import argparse

import torch
from torchvision.utils import save_image

from diffusion import GaussianDiffusion
from model import UNet
from vae import VAE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-ckpt", required=True)
    ap.add_argument("--ldm-ckpt", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--sampler", choices=["ddim", "dpm"], default="ddim")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--latent-channels", type=int, default=4)
    ap.add_argument(
        "--latent-stats", default=None, help="latent_stats.pt for de-normalization"
    )
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ldm_samples.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = VAE(in_channels=3, latent_channels=args.latent_channels).to(device)
    vae.load_state_dict(
        torch.load(args.vae_ckpt, map_location=device, weights_only=True)["model"]
    )
    vae.eval()

    model = UNet(
        in_channels=args.latent_channels,
        base_ch=args.base_ch,
        image_size=8,
        ch_mults=(1, 2, 2),
        attn_res=(8, 4),
    ).to(device)
    ck = torch.load(args.ldm_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    model.eval()

    if args.n < 1:
        raise ValueError("--n must be at least 1")
    if not 2 <= args.steps <= 1000:
        raise ValueError("--steps must be between 2 and 1000")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    diffusion = GaussianDiffusion().to(device)
    shape = (args.n, args.latent_channels, 8, 8)
    if args.sampler == "dpm":
        z = diffusion.dpm_solver_sample(
            model, shape, device, sampling_steps=args.steps, order=args.order
        )
    else:
        z = diffusion.ddim_sample(model, shape, device, sampling_steps=args.steps)

    with torch.no_grad():
        if args.latent_stats:
            stats = torch.load(
                args.latent_stats, map_location=device, weights_only=True
            )
            z = z * stats["std"] + stats["mean"]  # de-normalize
        x = vae.decode(z)
    x = torch.clamp((x + 1) / 2, 0, 1)
    save_image(x, args.out, nrow=max(1, int(args.n**0.5)))
    print(f"saved {args.out} (device={device}, seed={args.seed})")


if __name__ == "__main__":
    main()
