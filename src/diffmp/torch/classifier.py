import torch
import torch.nn as nn
import numpy as np


class ActionClassifier(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.bound_classifier = nn.Linear(
            latent_dim, 3 * action_dim, dtype=torch.float64
        )
        self.value_regressor = nn.Linear(latent_dim, action_dim, dtype=torch.float64)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bound_logits = self.bound_classifier(h)  # [B, 3 * action_dim]
        bound_logits: torch.Tensor = bound_logits.view(
            h.shape[0], -1, 3
        )  # [B, action_dim, 3]

        value_pred: torch.Tensor = self.value_regressor(h)  # [B, action_dim]

        return bound_logits, value_pred


def compute_log_posterior(log_xt, log_x0, t, alpha_bars, K):
    """
    Compute q(x_{t-1} | x_t, x0) or substitute x0_hat for model posterior.
    """
    device = log_xt.device
    # alpha_t = alpha_bar_t / alpha_bar_{t-1}
    alpha_bar = torch.tensor(alpha_bars, device=device)
    alpha_bar_t = torch.index_select(alpha_bar, 0, t)
    alpha_bar_tm1 = torch.index_select(alpha_bar, 0, torch.clamp(t - 1, min=0))
    alpha_t = alpha_bar_t / alpha_bar_tm1.clamp(min=1e-10)

    log_alpha_t = torch.log(alpha_t).unsqueeze(1).unsqueeze(2)

    # First term: alpha_t * x_t + (1-alpha_t)/K
    a = log_xt + log_alpha_t
    b = torch.log1p(-alpha_t).unsqueeze(1).unsqueeze(2) - np.log(K)
    log_first = torch.logsumexp(torch.stack([a, b.expand_as(a)], dim=-1), dim=-1)

    # Second term: alpha_bar_{t-1} * x0 + (1-alpha_bar_{t-1})/K
    log_ab_tm1 = torch.log(alpha_bar_tm1).unsqueeze(1).unsqueeze(2)
    b2 = torch.log1p(-alpha_bar_tm1).unsqueeze(1).unsqueeze(2) - np.log(K)
    log_second = torch.logsumexp(
        torch.stack([log_x0 + log_ab_tm1, b2.expand_as(log_x0)], dim=-1), dim=-1
    )

    log_theta = log_first + log_second
    log_theta_a = log_theta - torch.logsumexp(
        log_theta, dim=-1, keepdim=True
    )  # normalize
    return log_theta_a
