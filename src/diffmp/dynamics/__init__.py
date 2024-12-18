from enum import Enum

import diffmp
from .base import DynamicsBase
from .unicycle1 import UnicycleFirstOrder
from .unicycle2 import UnicycleSecondOrder


class Dynamics(Enum):
    unicycle1 = UnicycleFirstOrder
    unicycle2 = UnicycleSecondOrder


def get_dynamics(name: str) -> DynamicsBase:
    path = (diffmp.utils.DYN_CONFIG_PATH / name).with_suffix(".yaml")
    config = diffmp.utils.load_yaml(path)
    return Dynamics[config["dynamics"]].value(**config)
