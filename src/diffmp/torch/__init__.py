from enum import Enum
from typing import Callable

from torch import Tensor

from .dataset import DiffusionDataset
from .loss import mae, mse, sinkhorn
from .model import Config, Model
from .sample import sample
from .schedules import NoiseSchedule
from .train import train


class LossFunction:
    def __init__(self, func: Callable[[Tensor, Tensor], Tensor]):
        self.func = func

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return self.func(y_true, y_pred)


class Loss(Enum):
    mae = LossFunction(mae)
    mse = LossFunction(mse)
    sinkhorn = LossFunction(sinkhorn)


__all__ = [
    "Config",
    "Model",
    "NoiseSchedule",
    "LossFunction",
    "Loss",
    "DiffusionDataset",
    "train",
    "sample",
]
