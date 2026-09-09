"""Class-conditional denoising diffusion for tabular feature vectors."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from ids_diffusion.config import DiffusionConfig


class ClassConditionalDdpm(nn.Module):
    """Predict additive noise and sample by reversing the fixed schedule."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bar: torch.Tensor
    sqrt_alpha_bar: torch.Tensor
    sqrt_one_minus_alpha_bar: torch.Tensor
    denoiser: nn.Sequential

    def __init__(self, input_dim: int, config: DiffusionConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.timesteps = config.timesteps
        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

        layers: list[nn.Module] = []
        previous = input_dim + 1
        for width in config.hidden:
            layers.extend((nn.Linear(previous, width), nn.GELU()))
            previous = width
        layers.append(nn.Linear(previous, input_dim))
        self.denoiser = nn.Sequential(*layers)

    def forward(self, clean: torch.Tensor) -> torch.Tensor:
        """Return the denoising loss at randomly sampled diffusion steps."""
        batch_size = clean.size(0)
        times = torch.randint(0, self.timesteps, (batch_size,), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = (
            self.sqrt_alpha_bar[times].unsqueeze(-1) * clean
            + self.sqrt_one_minus_alpha_bar[times].unsqueeze(-1) * noise
        )
        time_embedding = (times.float() / self.timesteps).unsqueeze(-1)
        prediction = self.denoiser(torch.cat((noisy, time_embedding), dim=-1))
        return functional.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(self, count: int, device: torch.device) -> torch.Tensor:
        """Generate bounded samples in Gaussianised feature space."""
        samples = torch.randn(count, self.input_dim, device=device)
        for step in reversed(range(self.timesteps)):
            times = torch.full((count,), step, dtype=torch.long, device=device)
            time_embedding = (times.float() / self.timesteps).unsqueeze(-1)
            predicted_noise = self.denoiser(torch.cat((samples, time_embedding), dim=-1))
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bar[step]
            random_noise = torch.randn_like(samples) if step else torch.zeros_like(samples)
            samples = (samples - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(
                alpha
            ) + torch.sqrt(beta) * random_noise
            samples = torch.nan_to_num(samples, nan=0.0, posinf=5.0, neginf=-5.0)
            samples = torch.clamp(samples, -5.0, 5.0)
        return samples


__all__ = ["ClassConditionalDdpm"]
