"""Tests for the DPM-Solver++ (2M) sampler.

Design notes:

* 1st-order DPM-Solver is algebraically identical to DDIM (eta=0) only without
  clipping. The two finite-step implementations differ where the essential
  ``x0`` clamp engages.
* The 2nd-order multistep term reduces numerical error against a fine reference
  in this synthetic test. That does not imply better image quality or FID.
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
    model = UNet(in_channels=3, base_ch=32).to(device).eval()
    diffusion = GaussianDiffusion(timesteps=100).to(device)
    return model, diffusion, device


def test_dpm_shape_and_range(model_and_diffusion):
    model, diffusion, device = model_and_diffusion
    out = diffusion.dpm_solver_sample(model, (4, 3, 32, 32), device, sampling_steps=20)
    assert out.shape == (4, 3, 32, 32)
    assert torch.isfinite(out).all()
    # x0 is clamped to [-1, 1] each step, so the final sample stays near that
    # range (small overshoot from the sigma_next/sigma_cur rescaling).
    assert out.min() >= -3.0 and out.max() <= 3.0


def test_dpm_order2_more_accurate_than_order1(model_and_diffusion):
    model, diffusion, device = model_and_diffusion
    shape = (1, 3, 8, 8)

    torch.manual_seed(0)
    ref = diffusion.dpm_solver_sample(model, shape, device, sampling_steps=256, order=1)

    torch.manual_seed(0)
    o1_16 = diffusion.dpm_solver_sample(
        model, shape, device, sampling_steps=16, order=1
    )
    torch.manual_seed(0)
    o1_32 = diffusion.dpm_solver_sample(
        model, shape, device, sampling_steps=32, order=1
    )
    torch.manual_seed(0)
    o2_16 = diffusion.dpm_solver_sample(
        model, shape, device, sampling_steps=16, order=2
    )
    torch.manual_seed(0)
    o2_32 = diffusion.dpm_solver_sample(
        model, shape, device, sampling_steps=32, order=2
    )

    e1_16 = (o1_16 - ref).abs().mean().item()
    e1_32 = (o1_32 - ref).abs().mean().item()
    e2_16 = (o2_16 - ref).abs().mean().item()
    e2_32 = (o2_32 - ref).abs().mean().item()

    # 1st-order converges as steps grow.
    assert e1_32 < e1_16
    # 2nd-order beats 1st-order at the same step count.
    assert e2_16 < e1_16
    assert e2_32 < e1_32
    # 2nd-order converges (error shrinks as steps grow).
    assert e2_32 < e2_16
