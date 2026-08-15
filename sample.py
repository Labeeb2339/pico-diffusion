"""Generate a grid of samples from a trained DDPM checkpoint (DDIM by default)."""

from __future__ import annotations

import argparse

import torch
from torchvision.utils import save_image

from diffusion import GaussianDiffusion
from model import UNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--num-classes", type=int, default=None, help="conditional model: number of classes")
    ap.add_argument("--class-idx", type=int, default=None, help="class to generate (default: cycle all classes)")
    ap.add_argument("--cfg-scale", type=float, default=0.0, help="classifier-free guidance scale (0 = off)")
    ap.add_argument("--out", default="samples.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=args.channels, image_size=args.image_size, num_classes=args.num_classes).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    model.eval()

    y = None
    if args.num_classes is not None:
        if args.class_idx is not None:
            y = torch.full((args.n,), args.class_idx, device=device, dtype=torch.long)
        else:
            y = torch.arange(args.n, device=device) % args.num_classes

    diffusion = GaussianDiffusion().to(device)
    x = diffusion.ddim_sample(
        model, (args.n, args.channels, args.image_size, args.image_size), device,
        sampling_steps=args.steps, eta=args.eta, y=y, w=args.cfg_scale,
    )
    x = torch.clamp((x + 1) / 2, 0, 1)
    save_image(x, args.out, nrow=int(args.n**0.5))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
