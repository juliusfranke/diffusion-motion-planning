from typing import Callable, Dict, Tuple

from nptyping import NDArray
from torch import Tensor
from .loss import mae, mse, sinkhorn
from .schedules import (
    linear_noise_schedule,
    linear_noise_schedule_scaled,
    cosine_noise_schedule,
    sigmoid_noise_schedule,
)

losses: Dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "mae": mae,
    "mse": mse,
    "sinkhorn": sinkhorn,
}

schedules: Dict[str, Callable[[int], Tuple[NDArray, NDArray]]] = {
    "linear": linear_noise_schedule,
    "linear_scaled": linear_noise_schedule_scaled,
    "cosine": cosine_noise_schedule,
    "sigmoid": sigmoid_noise_schedule,
}
