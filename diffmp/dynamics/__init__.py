from enum import Enum
from typing import Dict, Type

from .base import DynamicsBase
from .unicycle1 import UnicycleFirstOrder
from .unicycle2 import UnicycleSecondOrder


class DynamicTypes(Enum):
    unicycle1 = 0
    unicycle2 = 1


dynamics: Dict[DynamicTypes, Type[DynamicsBase]] = {
    DynamicTypes.unicycle1: UnicycleFirstOrder,
    DynamicTypes.unicycle2: UnicycleSecondOrder,
}


def get_dynamics(config: Dict) -> DynamicsBase:
    return dynamics[DynamicTypes[config["dynamics"]]](**config)
