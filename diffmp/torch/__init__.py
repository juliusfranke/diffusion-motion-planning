from enum import Enum
from .loss import mae, mse, sinkhorn


class Loss(Enum):
    mae = mae
    mse = mse
    sinkhorn = sinkhorn
