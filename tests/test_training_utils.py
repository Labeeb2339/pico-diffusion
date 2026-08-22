"""CPU-safe regression tests for training helpers."""

import pytest
import torch

from train import EMA, cuda_amp_enabled


def test_cuda_amp_detection_uses_device_type() -> None:
    assert cuda_amp_enabled(torch.device("cuda"))
    assert not cuda_amp_enabled(torch.device("cpu"))


def test_ema_context_restores_training_weights() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    ema = EMA(model, decay=0.5)
    original = model.weight.detach().clone()

    with torch.no_grad():
        model.weight.add_(2.0)
    training_weights = model.weight.detach().clone()
    ema.update()

    with ema.average_parameters():
        assert not torch.equal(model.weight, training_weights)

    assert torch.equal(model.weight, training_weights)
    assert not torch.equal(model.weight, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_cuda_amp_step_runs() -> None:
    device = torch.device("cuda")
    model = torch.nn.Linear(4, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=cuda_amp_enabled(device))

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        loss = model(torch.randn(8, 4, device=device)).square().mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert torch.isfinite(loss)
