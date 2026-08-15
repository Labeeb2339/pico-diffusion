"""Train a from-scratch DDPM on CIFAR-10 or MNIST (optionally class-conditioned)."""

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


class EMA:
    """Exponential moving average of the model weights (standard for DDPM)."""

    def __init__(self, model: UNet, decay: float = 0.995):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self) -> None:
        for k, v in self.model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply(self) -> None:
        self.model.load_state_dict(self.shadow)


def get_dataset(name: str):
    if name == "cifar10":
        tf = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
        # fast.ai mirror ships CIFAR-10 as ImageFolder (train/<class>/*.png)
        ds = torchvision.datasets.ImageFolder(root="./data/cifar10/train", transform=tf)
        return ds, 3, 10
    tf = T.Compose([T.Pad(2), T.ToTensor(), T.Normalize([0.5], [0.5])])  # 28 -> 32
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    return ds, 1, None


@torch.no_grad()
def make_samples(diffusion, model, device, ch, image_size, num_classes, n=16, steps=50, cfg_scale=0.0):
    model.eval()
    y = None
    if num_classes is not None:
        y = torch.arange(n, device=device) % num_classes  # cycle through all classes
    x = diffusion.ddim_sample(model, (n, ch, image_size, image_size), device, sampling_steps=steps, y=y, w=cfg_scale)
    return torch.clamp((x + 1) / 2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--cfg-scale", type=float, default=2.0, help="classifier-free guidance scale (0 = off)")
    ap.add_argument("--cfg-dropout", type=float, default=0.1, help="prob of dropping the label during training (CFG)")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds, ch, num_classes = get_dataset(args.dataset)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = UNet(in_channels=ch, base_ch=args.base_ch, image_size=args.image_size, num_classes=num_classes).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=args.ema_decay)

    # AMP: bfloat16 autocast + GradScaler (uses the Blackwell tensor cores).
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_step = 0
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        ema.shadow = ck["ema"]
        start_step = ck["step"]
        print(f"resumed from {args.ckpt} at step {start_step}")

    print(f"device={device} dataset={args.dataset} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    losses: list[float] = []
    step = start_step
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device) if num_classes is not None else None
            t = torch.randint(0, args.timesteps, (x.shape[0],), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = diffusion.p_losses(model, x, t, y, p_uncond=args.cfg_dropout if num_classes else 0.0)
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
                samples = make_samples(diffusion, model, device, ch, args.image_size, num_classes, cfg_scale=args.cfg_scale)
                save_image(samples, out / f"sample_{step:07d}.png", nrow=4)
                torch.save({"model": model.state_dict(), "ema": ema.shadow, "step": step}, out / "ckpt.pt")
                model.train()
                print(f"    saved samples + ckpt at step {step}")

        print(f"epoch {epoch + 1}/{args.epochs} done in {time.time() - t0:.1f}s")

    ema.apply()
    samples = make_samples(diffusion, model, device, ch, args.image_size, num_classes, cfg_scale=args.cfg_scale)
    save_image(samples, out / "final.png", nrow=4)
    torch.save({"model": model.state_dict(), "ema": ema.shadow, "step": step}, out / "ckpt.pt")
    json.dump({"losses": losses}, open(out / "losses.json", "w"))
    print("done.")


if __name__ == "__main__":
    main()
