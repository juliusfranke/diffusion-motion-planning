from enum import Enum
from typing import Any, Callable, Tuple

from torch import Tensor
from numpy.typing import NDArray

from .loss import mae, mse, sinkhorn
from .model import Config, Model
from .schedules import (
    cosine_noise_schedule,
    linear_noise_schedule,
    linear_noise_schedule_scaled,
    sigmoid_noise_schedule,
)


class LossFunction:
    def __init__(self, func: Callable[[Tensor, Tensor], Tensor]):
        self.func = func

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return self.func(y_true, y_pred)


class NoiseScheduleFunction:
    def __init__(self, func: Callable[[int], Tuple[NDArray, NDArray]]):
        self.func = func

    def __call__(self, N: int) -> Tuple[NDArray, NDArray]:
        return self.func(N)


class Loss(Enum):
    mae = LossFunction(mae)
    mse = LossFunction(mse)
    sinkhorn = LossFunction(sinkhorn)


class NoiseSchedule(Enum):
    linear = NoiseScheduleFunction(linear_noise_schedule)
    linear_scaled = NoiseScheduleFunction(linear_noise_schedule_scaled)
    cosine = NoiseScheduleFunction(cosine_noise_schedule)
    sigmoid = NoiseScheduleFunction(sigmoid_noise_schedule)




