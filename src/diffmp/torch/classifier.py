import torch
import torch.nn as nn


class ActionClassifier(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.bound_classifier = nn.Linear(latent_dim, 3 * action_dim, dtype=torch.float64)
        self.value_regressor = nn.Linear(latent_dim, action_dim, dtype=torch.float64)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bound_logits = self.bound_classifier(h)  # [B, 3 * action_dim]
        bound_logits: torch.Tensor = bound_logits.view(
            h.shape[0], -1, 3
        )  # [B, action_dim, 3]

        value_pred: torch.Tensor = self.value_regressor(h)  # [B, action_dim]

        return bound_logits, value_pred
