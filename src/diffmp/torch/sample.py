import numpy as np
import torch

import diffmp

from .model import Model


def sample(
    model: Model, n_samples: int, instance: diffmp.problems.Instance
) -> torch.Tensor:
    model.to(diffmp.utils.DEVICE)
    model.eval()
    if model.config.conditioning:
        conditioning = diffmp.utils.condition_for_sampling(
            model.config, n_samples, instance
        )
    else:
        conditioning = None

    x_t = torch.randn(
        (n_samples, model.out_size), device=diffmp.utils.DEVICE, dtype=torch.float64
    )

    alpha_bars, betas = model.noise_schedule(model.config.denoising_steps)
    alphas = np.clip(1 - betas, 1e-8, np.inf)

    if isinstance(conditioning, torch.Tensor):
        conditioning = torch.atleast_2d(conditioning)

    for t in range(len(alphas))[::-1]:
        ts = t * torch.ones(
            (n_samples, 1), device=diffmp.utils.DEVICE, dtype=torch.float64
        )
        ab_t = alpha_bars[t] * torch.ones(
            (n_samples, 1), device=diffmp.utils.DEVICE, dtype=torch.float64
        )

        z = (
            torch.randn(
                (n_samples, model.out_size),
                device=diffmp.utils.DEVICE,
                dtype=torch.float64,
            )
            if t > 1
            else torch.zeros(
                (n_samples, model.out_size),
                device=diffmp.utils.DEVICE,
                dtype=torch.float64,
            )
        )

        if isinstance(conditioning, torch.Tensor):
            model_input = torch.concat([x_t, conditioning, ts], dim=-1)
        else:
            model_input = torch.concat([x_t, ts], dim=-1)
        model_prediction = model(model_input)

        x_t = (
            1
            / alphas[t] ** 0.5
            * (
                x_t[:, : model.out_size]
                - betas[t] / (1 - ab_t) ** 0.5 * model_prediction
            )
        )
        x_t += betas[t] ** 0.5 * z

    return x_t.detach()
