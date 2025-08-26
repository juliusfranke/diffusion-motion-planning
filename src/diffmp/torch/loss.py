from geomloss import SamplesLoss
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict

_sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05)


def sinkhorn(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.Tensor(_sinkhorn(y_true.float(), y_pred.float()))


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_true - y_pred))


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_true - y_pred) ** 2)


def compute_test_loss(
    success_rate,
    baseline_success,
    duration,
    baseline_duration,
    cost,
    baseline_cost,
    alpha=5.0,
    beta=1.0,
    gamma=1.0,
    epsilon=1e-6,
    debug: bool = False,
):
    """
    Computes a test score comparing performance against a baseline.

    Parameters:
    - success_rate: Success rate of the model
    - baseline_success: Baseline success rate
    - duration: Median duration of successful runs
    - baseline_duration: Baseline median duration
    - cost: Median cost of successful runs
    - baseline_cost: Baseline median cost
    - alpha, beta, gamma: Weights for success, duration, and cost importance
    - epsilon: Small value to prevent division by zero

    Returns:
    - test_score: A higher score means better performance
    """

    if success_rate == 0:
        if debug:
            print(f"Success: {0:.4f} - Duration: {0:.4f} - Cost: {0:.4f}")
        return 0.0

    success_term = ((success_rate + epsilon) / (baseline_success + epsilon)) ** alpha
    duration_term = (baseline_duration / duration) ** beta if duration > 0 else 1.0
    cost_term = (baseline_cost / cost) ** gamma if cost > 0 else 1.0
    if debug:
        print(
            f"Success: {success_term:.4f} - Duration: {duration_term:.4f} - Cost: {cost_term:.4f}"
        )

    test_score = success_term * duration_term * cost_term
    return test_score


class ExponentialMovingAverage:
    def __init__(self, alpha=0.2):  # Alpha controls smoothing (0.1-0.3 recommended)
        self.alpha = alpha
        self.ema = None

    def update(self, new_value: float) -> float:
        if self.ema is None:
            self.ema = new_value  # Initialize with first value
        else:
            self.ema = self.alpha * new_value + (1 - self.alpha) * self.ema
        return self.ema
