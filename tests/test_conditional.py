"""Tests for class-conditioned diffusion + classifier-free guidance (GPU-gated)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion import GaussianDiffusion  # noqa: E402
from model import UNet  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_conditional_forward_and_cfg() -> None:
    dev = "cuda"
    model = UNet(in_channels=3, base_ch=32, image_size=32, num_classes=10).to(dev)
    diff = GaussianDiffusion(timesteps=50).to(dev)
    x = torch.randn(2, 3, 32, 32, device=dev)
    t = torch.randint(0, 50, (2,), device=dev)
    y = torch.randint(0, 10, (2,), device=dev)

    # class conditioning must actually change the output
    out_cond = model(x, t, y)
    null = torch.full((2,), model.num_classes, device=dev, dtype=torch.long)
    out_null = model(x, t, null)
    assert not torch.allclose(out_cond, out_null), "class conditioning has no effect"

    # p_losses with CFG label dropout runs and is finite
    loss = diff.p_losses(model, x, t, y, p_uncond=0.1)
    assert torch.isfinite(loss)

    # ddim_sample with CFG produces in-range output
    samples = diff.ddim_sample(model, (2, 3, 32, 32), dev, sampling_steps=5, y=y, w=2.0)
    assert samples.shape == (2, 3, 32, 32)
    assert samples.abs().max().item() < 2.0


def test_unconditional_model_still_works() -> None:
    dev = "cuda"
    model = UNet(in_channels=1, base_ch=32, image_size=32).to(dev)  # num_classes=None
    x = torch.randn(2, 1, 32, 32, device=dev)
    t = torch.randint(0, 50, (2,), device=dev)
    out = model(x, t)  # no y argument
    assert out.shape == (2, 1, 32, 32)
    assert model.class_emb is None
