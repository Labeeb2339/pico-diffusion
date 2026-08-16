"""Train the latent diffusion model (diffusion in the VAE latent space).

Images are encoded to a ``4 x 8 x 8`` latent with a frozen VAE, then the same
U-Net + DDPM/DDIM/DPM machinery from the pixel-space model denoises *latents*
instead of pixels. This is the latent-diffusion (Stable Diffusion) recipe.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from diffusion import GaussianDiffusion
from model import UNet
from train import EMA
from vae import VAE


def get_dataset():
    tf = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    return torchvision.datasets.ImageFolder(root="./data/cifar10/train", transform=tf)


@torch.no_grad()
def compute_latent_stats(vae, loader, device, max_batches=32):
    """Per-channel mean/std of the deterministic latents (latent normalization).

    The diffusion schedule assumes ~N(0,1) data, but a weak-KL VAE yields latents
    with std >> 1. Standardizing them (LDM / Stable Diffusion) fixes the mismatch.
    """
    mus = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        mus.append(vae.encode_deterministic(x.to(device)))
    all_mu = torch.cat(mus, dim=0)  # (N, C, H, W)
    return (all_mu.mean(dim=(0, 2, 3), keepdim=True),
            all_mu.std(dim=(0, 2, 3), keepdim=True))


@torch.no_grad()
def make_samples(diffusion, model, vae, device, latent_ch, latent_stats, n=16, steps=50):
    model.eval()
    vae.eval()
    z = diffusion.ddim_sample(model, (n, latent_ch, 8, 8), device, sampling_steps=steps)
    z = z * latent_stats["std"] + latent_stats["mean"]  # un-standardize
    x = vae.decode(z)
    return torch.clamp((x + 1) / 2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--latent-channels", type=int, default=4)
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--out-dir", default="out_ldm")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vae = VAE(in_channels=3, latent_channels=args.latent_channels).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    ds = get_dataset()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Latent normalization (LDM/Stable Diffusion): the diffusion schedule assumes
    # ~N(0,1) data, but a weak-KL VAE produces latents with std >> 1. Standardize.
    mean, std = compute_latent_stats(vae, loader, device)
    latent_stats = {"mean": mean, "std": std}
    torch.save(latent_stats, out / "latent_stats.pt")
    print(f"latent stats: mean={mean.flatten().tolist()} std={std.flatten().tolist()}")

    # Latent UNet: 4 x 8 x 8 -> denoise -> 4 x 8 x 8 (downsample 8->4->2).
    model = UNet(in_channels=args.latent_channels, base_ch=args.base_ch, image_size=8,
                 ch_mults=(1, 2, 2), attn_res=(8, 4)).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=0.995)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} latent-unet params={n_params/1e6:.2f}M latent={args.latent_channels}x8x8")

    losses: list[float] = []
    step = 0
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        for x, _ in loader:
            x = x.to(device)
            with torch.no_grad():
                z = (vae.encode_deterministic(x) - mean) / std  # (B, 4, 8, 8)
            t = torch.randint(0, args.timesteps, (x.shape[0],), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = diffusion.p_losses(model, z, t)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update()
            losses.append(loss.item())
            step += 1

            if step % 100 == 0:
                avg = sum(losses[-100:]) / len(losses[-100:])
                print(f"step {step} | loss {loss.item():.4f} | avg100 {avg:.4f}")

            if step % args.sample_every == 0:
                ema.apply()
                samples = make_samples(diffusion, model, vae, device, args.latent_channels, latent_stats)
                save_image(samples, out / f"sample_{step:07d}.png", nrow=4)
                torch.save({"model": model.state_dict(), "ema": ema.shadow, "step": step}, out / "ckpt.pt")
                model.train()
                print(f"    saved samples + ckpt at step {step}")

        print(f"epoch {epoch + 1}/{args.epochs} done in {time.time() - t0:.1f}s")

    ema.apply()
    samples = make_samples(diffusion, model, vae, device, args.latent_channels, latent_stats)
    save_image(samples, out / "final.png", nrow=4)
    torch.save({"model": model.state_dict(), "ema": ema.shadow, "step": step}, out / "ckpt.pt")
    json.dump({"losses": losses}, open(out / "losses.json", "w"))
    print("done.")


if __name__ == "__main__":
    main()
