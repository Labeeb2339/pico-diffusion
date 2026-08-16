"""DDPM + DDIM + DPM-Solver sampling, from scratch.

Implements the forward (add-noise) process and three reverse samplers:

* DDPM (Ho et al., 2020) — ancestral, stochastic.
* DDIM (Song et al., 2020) — deterministic, ``eta=0``.
* DPM-Solver++ 2M (Lu et al., 2022) — a 2nd-order multistep solver of the
  probability-flow ODE.  Its 1st-order step is *exactly* DDIM (``eta=0``); the
  2nd-order correction reuses the previous step's noise prediction, giving the
  same sample quality in far fewer steps.

The model predicts the noise ``eps`` at each step; the loss is MSE against the
true noise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, beta_schedule: str = "cosine"):
        betas = cosine_beta_schedule(timesteps) if beta_schedule == "cosine" else linear_beta_schedule(timesteps)
        self.timesteps = timesteps
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device):
        """Move all schedule tensors to ``device`` (so they can index CUDA timesteps)."""
        for name in (
            "betas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas", "posterior_variance",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward process: x_t = sqrt(ac) x0 + sqrt(1-ac) noise."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x0 + sqrt_omac * noise

    def p_losses(self, model, x0: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None, p_uncond: float = 0.0) -> torch.Tensor:
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        if y is not None and p_uncond > 0.0:
            null = getattr(model, "num_classes", None)
            if null is not None:
                # classifier-free guidance: drop labels to the null class
                drop = torch.rand(y.shape[0], device=y.device) < p_uncond
                y = torch.where(drop, torch.full_like(y, null), y)
        return F.mse_loss(model(xt, t, y), noise)

    def _eps(self, model, x: torch.Tensor, t_b: torch.Tensor, y: torch.Tensor | None, w: float) -> torch.Tensor:
        """Noise prediction, with classifier-free guidance when ``w > 0``."""
        if y is not None and w > 0.0:
            null = torch.full((x.shape[0],), model.num_classes, device=x.device, dtype=torch.long)
            return model(x, t_b, null) + w * (model(x, t_b, y) - model(x, t_b, null))
        return model(x, t_b, y)

    @torch.no_grad()
    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """One DDPM reverse step (with posterior variance when t_index > 0)."""
        betas_t = self.betas[t][:, None, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]
        sqrt_one_minus_ac_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        pred_noise = model(x, t)
        mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_ac_t * pred_noise)
        if t_index == 0:
            return mean
        noise = torch.randn_like(x)
        return mean + torch.sqrt(self.posterior_variance[t][:, None, None, None]) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape: tuple[int, ...], device) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i)
        return x

    @torch.no_grad()
    def ddim_sample(
        self, model, shape: tuple[int, ...], device, sampling_steps: int = 50, eta: float = 0.0,
        y: torch.Tensor | None = None, w: float = 0.0,
    ) -> torch.Tensor:
        """DDIM sampling (``eta=0`` deterministic, ``eta=1`` DDPM-like).

        ``y`` = class labels (conditional model), ``w`` = classifier-free
        guidance scale (0 = off).
        """
        times = torch.linspace(self.timesteps - 1, 0, sampling_steps, dtype=torch.long, device=device)
        x = torch.randn(shape, device=device)
        for i in range(len(times) - 1):
            t = times[i]
            t_prev = times[i + 1]
            t_b = torch.full((shape[0],), t, device=device, dtype=torch.long)

            pred_noise = self._eps(model, x, t_b, y, w)

            ac_t = self.alphas_cumprod[t]
            ac_prev = self.alphas_cumprod[t_prev]
            sqrt_ac_t = ac_t**0.5
            sqrt_ac_prev = ac_prev**0.5
            sqrt_1m_ac_t = (1.0 - ac_t) ** 0.5

            x0_pred = (x - sqrt_1m_ac_t * pred_noise) / sqrt_ac_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            sigma = eta * ((1.0 - ac_prev) / (1.0 - ac_t) * (1.0 - ac_t / ac_prev)) ** 0.5
            pred_dir = (1.0 - ac_prev - sigma**2) ** 0.5 * pred_noise
            x = sqrt_ac_prev * x0_pred + pred_dir
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x

    @torch.no_grad()
    def dpm_solver_sample(
        self, model, shape: tuple[int, ...], device, sampling_steps: int = 20, order: int = 2,
        y: torch.Tensor | None = None, w: float = 0.0,
    ) -> torch.Tensor:
        """DPM-Solver++ (2M) sampling of the probability-flow ODE.

        Uses the *data-prediction* form of the exponential integrator, with the
        same ``x0`` clamp as DDIM (essential for stability on a weak model:
        ``x0 = (x - sigma eps)/alpha`` is huge when ``alpha`` is tiny). With
        ``order=1`` this is *exactly* DDIM (``eta=0``); ``order=2`` adds the
        2nd-order multistep correction that reuses the previous step's ``x0``,
        so far fewer steps reach the same quality.

        ``sampling_steps`` = number of timestep points (``sampling_steps - 1``
        model evaluations), matching :meth:`ddim_sample` exactly.
        """
        times = torch.linspace(self.timesteps - 1, 0, sampling_steps, dtype=torch.long, device=device)

        # alpha_t = sqrt(alphas_cumprod), sigma_t = sqrt(1 - alphas_cumprod),
        # lambda_t = log(alpha_t / sigma_t)  (log-SNR, increases as t -> 0).
        ac = self.alphas_cumprod[times]
        alpha = ac.sqrt()
        sigma = (1.0 - ac).sqrt()
        lam = torch.log(alpha / sigma)

        x = torch.randn(shape, device=device)
        prev_x0 = None
        h_prev = None
        for i in range(len(times) - 1):
            t_cur = times[i]
            t_b = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            eps = self._eps(model, x, t_b, y, w)

            alpha_cur = alpha[i]
            alpha_next = alpha[i + 1]
            sigma_cur = sigma[i]
            sigma_next = sigma[i + 1]
            h = lam[i + 1] - lam[i]  # positive (cleaner step has higher log-SNR)

            # data prediction, clamped exactly like DDIM.
            x0 = torch.clamp((x - sigma_cur * eps) / alpha_cur, -1.0, 1.0)

            # 1st-order exponential integrator (== DDIM eta=0).
            x = (sigma_next / sigma_cur) * x - alpha_next * torch.expm1(-h) * x0

            # 2nd-order multistep correction (reuses the previous step's x0).
            if order >= 2 and prev_x0 is not None:
                r1 = h_prev / h  # ratio of previous to current step size
                x = x - (alpha_next / (2.0 * r1)) * torch.expm1(-h) * (x0 - prev_x0)

            prev_x0 = x0
            h_prev = h
        return x
