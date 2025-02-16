from enum import Enum

import diffmp
from .base import DynamicsBase
from .unicycle1 import UnicycleFirstOrder
from .unicycle2 import UnicycleSecondOrder


class Dynamics(Enum):
    unicycle1 = UnicycleFirstOrder
    unicycle2 = UnicycleSecondOrder


def get_dynamics(name: str, timesteps: int) -> DynamicsBase:
    path = (diffmp.utils.DYN_CONFIG_PATH / name).with_suffix(".yaml")
    config = diffmp.utils.load_yaml(path)
    config["timesteps"] = timesteps
    dyn = Dynamics[config["dynamics"]].value(**config, name=name)
    return dyn


__all__ = [
    "DynamicsBase",
    "UnicycleFirstOrder",
    "UnicycleSecondOrder",
    "Dynamics",
    "get_dynamics",
]
