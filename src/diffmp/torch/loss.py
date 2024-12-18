from geomloss import SamplesLoss
import torch

_sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05)


def sinkhorn(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return _sinkhorn(y_true, y_pred)


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_true - y_pred))


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_true - y_pred) ** 2)
