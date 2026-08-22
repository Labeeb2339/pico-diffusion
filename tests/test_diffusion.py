"""Pipeline smoke tests for the from-scratch diffusion model (GPU-gated).

These verify the forward pass, loss, gradients, and both samplers produce the
correct shapes and value ranges — the same checks as the training run, but as
tests.
"""

import pytest
import torch

from diffusion import GaussianDiffusion
from model import UNet

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


@pytest.fixture(scope="module")
def model_and_diffusion():
    device = "cuda"
    model = UNet(in_channels=3, base_ch=32).to(device)
    diffusion = GaussianDiffusion(timesteps=100).to(device)
    return model, diffusion, device


def test_forward_shape(model_and_diffusion):
    model, _, device = model_and_diffusion
    x = torch.randn(4, 3, 32, 32, device=device)
    t = torch.randint(0, 100, (4,), device=device)
    assert model(x, t).shape == x.shape


def test_loss_and_backward(model_and_diffusion):
    model, diffusion, device = model_and_diffusion
    x0 = torch.randn(4, 3, 32, 32, device=device)
    t = torch.randint(0, 100, (4,), device=device)
    loss = diffusion.p_losses(model, x0, t)
    loss.backward()
    # At init the model predicts ~0, so MSE ≈ variance of the target noise ≈ 1.
    assert 0.5 < loss.item() < 2.0
    assert model.in_conv.weight.grad is not None


def test_q_sample_limits(model_and_diffusion):
    _, diffusion, device = model_and_diffusion
    x0 = torch.randn(8, 3, 32, 32, device=device)
    tT = torch.full((8,), 99, dtype=torch.long, device=device)
    xT = diffusion.q_sample(x0, tT)
    # At t=T the image is almost entirely noise: unit variance and essentially
    # decorrelated from the clean image (cosine schedule -> alpha_cumprod ~ 0).
    assert xT.std() > 0.5
    corr = torch.mean((xT - xT.mean()) * (x0 - x0.mean())) / (xT.std() * x0.std())
    assert abs(corr.item()) < 0.3


def test_ddim_sampling(model_and_diffusion):
    model, diffusion, device = model_and_diffusion
    out = diffusion.ddim_sample(model, (4, 3, 32, 32), device, sampling_steps=20)
    assert out.shape == (4, 3, 32, 32)
    assert out.min() >= -1.5 and out.max() <= 1.5  # sane range


def test_ddpm_sampling(model_and_diffusion):
    model, diffusion, device = model_and_diffusion
    out = diffusion.p_sample_loop(model, (2, 3, 32, 32), device)
    assert out.shape == (2, 3, 32, 32)
