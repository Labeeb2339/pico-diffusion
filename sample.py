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
    ap.add_argument(
        "--channels", type=int, default=None, help="infer from checkpoint by default"
    )
    ap.add_argument(
        "--base-ch", type=int, default=None, help="infer from checkpoint by default"
    )
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="conditional model: number of classes",
    )
    ap.add_argument(
        "--class-idx",
        type=int,
        default=None,
        help="class to generate (default: cycle all classes)",
    )
    ap.add_argument(
        "--cfg-scale",
        type=float,
        default=0.0,
        help="classifier-free guidance scale (0 = off)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="samples.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    state = ck["ema"] if "ema" in ck else ck["model"]
    channels = args.channels or int(state["in_conv.weight"].shape[1])
    base_ch = args.base_ch or int(state["in_conv.weight"].shape[0])
    checkpoint_num_classes = (
        int(state["class_emb.weight"].shape[0]) - 1
        if "class_emb.weight" in state
        else None
    )
    if args.num_classes is not None and args.num_classes != checkpoint_num_classes:
        raise ValueError(
            f"--num-classes={args.num_classes} does not match checkpoint "
            f"({checkpoint_num_classes})"
        )
    num_classes = checkpoint_num_classes
    if args.n < 1:
        raise ValueError("--n must be at least 1")
    if not 2 <= args.steps <= 1000:
        raise ValueError("--steps must be between 2 and 1000")

    model = UNet(
        in_channels=channels,
        base_ch=base_ch,
        image_size=args.image_size,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    y = None
    if num_classes is not None:
        if args.class_idx is not None:
            if not 0 <= args.class_idx < num_classes:
                raise ValueError(f"--class-idx must be between 0 and {num_classes - 1}")
            y = torch.full((args.n,), args.class_idx, device=device, dtype=torch.long)
        else:
            y = torch.arange(args.n, device=device) % num_classes
    elif args.class_idx is not None:
        raise ValueError("--class-idx requires a conditional checkpoint")

    diffusion = GaussianDiffusion().to(device)
    x = diffusion.ddim_sample(
        model,
        (args.n, channels, args.image_size, args.image_size),
        device,
        sampling_steps=args.steps,
        eta=args.eta,
        y=y,
        w=args.cfg_scale,
    )
    x = torch.clamp((x + 1) / 2, 0, 1)
    save_image(x, args.out, nrow=max(1, int(args.n**0.5)))
    mode = "conditional" if num_classes is not None else "unconditional"
    print(
        f"saved {args.out} ({mode}, device={device}, channels={channels}, "
        f"base_ch={base_ch}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
