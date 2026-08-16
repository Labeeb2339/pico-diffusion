"""Train the VAE (encoder + decoder) for latent diffusion on CIFAR-10.

Objective: L1 reconstruction + beta * KL. The result is a ``4 x 8 x 8`` latent
that (a) reconstructs the image and (b) is roughly unit-Gaussian — the two
properties the downstream diffusion model needs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from vae import VAE, vae_loss


def get_dataset():
    tf = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    ds = torchvision.datasets.ImageFolder(root="./data/cifar10/train", transform=tf)
    return ds


@torch.no_grad()
def make_recon_grid(vae, device, n=16):
    ds = get_dataset()
    loader = DataLoader(ds, batch_size=n, shuffle=True, num_workers=2)
    x = next(iter(loader))[0].to(device)
    vae.eval()
    recon, _, _ = vae(x)
    pairs = torch.cat([x, recon], dim=0)  # top row: real, bottom: reconstruction
    return torch.clamp((pairs + 1) / 2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--latent-channels", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--beta", type=float, default=1e-4, help="KL weight")
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--out-dir", default="out_vae")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = get_dataset()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    vae = VAE(in_channels=3, latent_channels=args.latent_channels, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = sum(p.numel() for p in vae.parameters())
    print(f"device={device} vae params={n_params/1e6:.2f}M latent={args.latent_channels}x8x8 beta={args.beta}")

    step = 0
    vae.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        for x, _ in loader:
            x = x.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                recon, mu, logvar = vae(x)
                loss, (recon_loss, kl) = vae_loss(recon, x, mu, logvar, args.beta)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % 200 == 0:
                print(f"step {step} | loss {loss.item():.4f} | recon {recon_loss.item():.4f} | kl {kl.item():.4f}")

            if step % args.sample_every == 0:
                grid = make_recon_grid(vae, device)
                save_image(grid, out / f"recon_{step:07d}.png", nrow=16)
                torch.save({"model": vae.state_dict(), "step": step}, out / "vae.pt")
                vae.train()
                print(f"    saved recon grid + vae.pt at step {step}")

        print(f"epoch {epoch + 1}/{args.epochs} done in {time.time() - t0:.1f}s")

    grid = make_recon_grid(vae, device)
    save_image(grid, out / "recon_final.png", nrow=16)
    torch.save({"model": vae.state_dict(), "step": step}, out / "vae.pt")
    print("done.")


if __name__ == "__main__":
    main()
