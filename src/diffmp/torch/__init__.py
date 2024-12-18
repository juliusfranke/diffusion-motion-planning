from enum import Enum
from typing import Any, Callable, Tuple

from torch import Tensor
from numpy.typing import NDArray

from .loss import mae, mse, sinkhorn
from .model import Config, Model
from .schedules import NoiseSchedule


class LossFunction:
    def __init__(self, func: Callable[[Tensor, Tensor], Tensor]):
        self.func = func

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return self.func(y_true, y_pred)




class Loss(Enum):
    mae = LossFunction(mae)
    mse = LossFunction(mse)
    sinkhorn = LossFunction(sinkhorn)





