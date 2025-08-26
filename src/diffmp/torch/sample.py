from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

import diffmp.utils as du

from . import Model

if TYPE_CHECKING:
    import diffmp.problems as pb


def sample(
    model: Model, n_samples: int, instance: pb.Instance, robot_idx: int = 0
) -> torch.Tensor:
    model.to(du.DEVICE)
    model.eval()
    if model.config.conditioning:
        conditioning = du.condition_for_sampling(
            model.config, n_samples, instance, robot_idx
        )
    else:
        conditioning = None
    if model.config.discretize is not None:
        tensor = torch.Tensor(
            instance.environment.discretize(model.config.discretize), device=du.DEVICE
        )
        discretize = tensor.reshape(tensor.shape[0], tensor.shape[0]).repeat(
            n_samples, 1, 1, 1
        )
    else:
        discretize = None

    x_t = torch.randn(
        (n_samples, model.out_size), device=du.DEVICE, dtype=torch.float64
    )

    alpha_bars, betas = model.noise_schedule(model.config.denoising_steps)
    alphas = np.clip(1 - betas, 1e-8, np.inf)

    if isinstance(conditioning, torch.Tensor):
        conditioning = torch.atleast_2d(conditioning)

    r_idx = robot_idx if model.config.robot_embedding else None

    for t in range(len(alphas))[::-1]:
        ts = t * torch.ones((n_samples, 1), device=du.DEVICE, dtype=torch.float64)
        ab_t = alpha_bars[t] * torch.ones(
            (n_samples, 1), device=du.DEVICE, dtype=torch.float64
        )

        z = (
            torch.randn(
                (n_samples, model.out_size),
                device=du.DEVICE,
                dtype=torch.float64,
            )
            if t > 1
            else torch.zeros(
                (n_samples, model.out_size),
                device=du.DEVICE,
                dtype=torch.float64,
            )
        )

        if isinstance(conditioning, torch.Tensor):
            model_input = torch.concat([x_t, conditioning, ts], dim=-1)
        else:
            model_input = torch.concat([x_t, ts], dim=-1)
        if discretize is not None:
            scale = (torch.ones(n_samples, dtype=torch.int) * 0,)
        else:
            scale = None

        if r_idx is not None:
            robot_id = (torch.ones(n_samples, dtype=torch.int) * r_idx,)
        else:
            robot_id = None

        model_prediction = model(
            model_input,
            discretize,
            scale,
            robot_id,
        )

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
