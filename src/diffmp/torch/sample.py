from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

import diffmp.utils as du

from . import Model
from .classifier import compute_log_posterior

if TYPE_CHECKING:
    import diffmp.problems as pb


def sample(
    model: Model,
    n_samples: int,
    instance: pb.Instance,
    robot_idx: int = 0,
    num_action_classes: int = 3,
) -> torch.Tensor:
    model.to(du.DEVICE)
    model.eval()
    if model.config.conditioning:
        conditioning = du.condition_for_sampling(
            model.config, n_samples, instance, robot_idx
        )
        conditioning = model.normalize_conditioning(conditioning)
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

    x_t_cont = torch.randn(
        (n_samples, model.noise_size), device=du.DEVICE, dtype=torch.float64
    )

    x_t_cat = torch.randint(
        0, num_action_classes, (n_samples, model.actions_dim), device=du.DEVICE
    )

    alpha_bars, betas = model.noise_schedule(model.config.denoising_steps)
    alphas = np.clip(1 - betas, 1e-8, np.inf)

    if isinstance(conditioning, torch.Tensor):
        conditioning = torch.atleast_2d(conditioning)

    r_idx = robot_idx if model.config.robot_embedding else None

    for t in reversed(range(len(alphas))):
        ts = t * torch.ones((n_samples, 1), device=du.DEVICE, dtype=torch.float64)
        ab_t = alpha_bars[t] * torch.ones(
            (n_samples, 1), device=du.DEVICE, dtype=torch.float64
        )

        z = (
            torch.randn_like(
                x_t_cont
                # (n_samples, model.noise_size),
                # device=du.DEVICE,
                # dtype=torch.float64,
            )
            # if t > 1
            if t > 0
            else torch.zeros_like(
                x_t_cont
                #     (n_samples, model.noise_size),
                #     device=du.DEVICE,
                #     dtype=torch.float64,
            )
        )

        if isinstance(conditioning, torch.Tensor):
            model_input = torch.concat([x_t_cont, conditioning, ts], dim=-1)
        else:
            model_input = torch.concat([x_t_cont, ts], dim=-1)

        if discretize is not None:
            scale = (torch.ones(n_samples, dtype=torch.int) * 0,)
        else:
            scale = None

        if r_idx is not None:
            robot_id = (torch.ones(n_samples, dtype=torch.int) * r_idx,)
        else:
            robot_id = None
        # breakpoint()
        eps_pred, cat_logits = model(
            model_input,
            x_t_cat,
            discretize,
            scale,
            robot_id,
        )
        # if model.config.classify_actions:
        #     model_prediction

        x_t_cont = (
            1
            / alphas[t] ** 0.5
            * (x_t_cont[:, : model.out_size] - betas[t] / (1 - ab_t) ** 0.5 * eps_pred)
        )
        if t > 0:
            x_t_cont += betas[t] ** 0.5 * z
        if model.config.classify_actions:
            log_p_post = compute_log_posterior(
                F.one_hot(x_t_cat, num_classes=num_action_classes)
                .float()
                .clamp(min=1e-40)
                .log(),
                F.log_softmax(cat_logits, dim=-1),
                torch.full((n_samples,), t, device=du.DEVICE, dtype=torch.long),
                alpha_bars,
                num_action_classes,
            )
            probs_post = torch.exp(log_p_post)
            x_t_cat = torch.distributions.Categorical(probs=probs_post).sample()
    x_final = x_t_cont.clone()
    x_final = model.denormalize_output(x_final)
    x_cat = x_t_cat.clone()
    if model.config.classify_actions:
        x_final[:, : x_cat.shape[1]][x_cat == 0] = -0.5
        x_final[:, : x_cat.shape[1]][x_cat == 2] = 0.5
    return x_final


def declassify_actions(
    model: Model, model_prediction: torch.Tensor, logit: torch.Tensor
):
    pass
