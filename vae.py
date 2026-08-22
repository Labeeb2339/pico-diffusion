"""A variational autoencoder for latent diffusion, from scratch.

Compresses a ``3 x 32 x 32`` CIFAR image to a ``latent_channels x 8 x 8``
latent (a 12x compression: 3072 -> 256 dimensions), then reconstructs it. The
latent is the "data" that the diffusion model then denoises (see
``ldm.py``) — this is the latent-diffusion / Stable-Diffusion architecture,
without any pretrained weights.

The encoder predicts a diagonal Gaussian posterior ``(mu, logvar)``; training
uses the reparameterization trick and a beta-VAE objective (reconstruction +
weighted KL). With a weak KL weight, the latent is not guaranteed to be
unit-Gaussian, so the downstream training code measures and normalizes it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VAEEncoder(nn.Module):
    def __init__(
        self, in_channels: int = 3, latent_channels: int = 4, hidden: int = 64
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, stride=2, padding=1),  # hidden x16x16
            nn.SiLU(),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1),  # 2h x8x8
            nn.SiLU(),
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1),  # 2h x8x8
            nn.SiLU(),
        )
        self.mu = nn.Conv2d(hidden * 2, latent_channels, 1)
        self.logvar = nn.Conv2d(hidden * 2, latent_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class VAEDecoder(nn.Module):
    def __init__(
        self, latent_channels: int = 4, out_channels: int = 3, hidden: int = 64
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, hidden * 2, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(
                hidden * 2, hidden, 4, stride=2, padding=1
            ),  # hidden x16x16
            nn.SiLU(),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1),  # hidden x32x32
            nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    """Encoder + reparameterization + decoder, plus the beta-VAE loss."""

    def __init__(
        self, in_channels: int = 3, latent_channels: int = 4, hidden: int = 64
    ):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_channels, hidden)
        self.decoder = VAEDecoder(latent_channels, in_channels, hidden)
        self.latent_channels = latent_channels

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior mean — the stable latent used as diffusion training data."""
        return self.encoder(x)[0]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(total, (recon, kl))`` — reconstruction (L1) + weighted KL.

    ``beta`` scales the KL term; a small value keeps the latent informative
    (avoids posterior collapse) while still regularizing it toward N(0, I).
    """
    recon_loss = F.l1_loss(recon, x)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, (recon_loss, kl)
