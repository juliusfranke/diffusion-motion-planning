from typing import Dict
from geomloss import SamplesLoss
import torch

sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05)

def mae(real: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(real - pred))


def mse(real: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.mean((real - pred) ** 2)




