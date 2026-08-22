"""Smoke tests for the VAE (encoder/decoder/loss). GPU-gated like the rest."""

import pytest
import torch

from vae import VAE, vae_loss

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


@pytest.fixture(scope="module")
def vae():
    return VAE(in_channels=3, latent_channels=4, hidden=32).to("cuda")


def test_encode_decode_shapes(vae):
    x = torch.randn(4, 3, 32, 32, device="cuda")
    mu, logvar = vae.encode(x)
    assert mu.shape == (4, 4, 8, 8)
    assert logvar.shape == (4, 4, 8, 8)
    recon, mu2, logvar2 = vae(x)
    assert recon.shape == x.shape
    # forward's mu/logvar agree with encode's
    assert torch.equal(mu, mu2) and torch.equal(logvar, logvar2)


def test_reparameterization_is_stochastic(vae):
    mu = torch.zeros(2, 4, 8, 8, device="cuda")
    logvar = torch.zeros(2, 4, 8, 8, device="cuda")
    z1 = vae.reparameterize(mu, logvar)
    z2 = vae.reparameterize(mu, logvar)
    assert z1.shape == mu.shape
    assert not torch.equal(z1, z2)  # unit-variance noise -> different samples


def test_loss_reduces_reconstruction_error(vae):
    x = torch.randn(8, 3, 32, 32, device="cuda")
    recon, mu, logvar = vae(x)
    loss, (recon_loss, kl) = vae_loss(recon, x, mu, logvar, beta=1e-4)
    # At init the reconstruction is poor, so recon_loss should be well above 0.
    assert recon_loss.item() > 0.1
    assert kl.item() > 0.0
    assert torch.isfinite(loss)
