from typing import Dict, Type

from .unicycle2 import UnicycleSecondOrder
from .unicycle_1 import UnicycleFirstOrder
from .base import DynamicsBase

dynamics: Dict[str, Type[DynamicsBase]] = {
    "unicycle1": UnicycleFirstOrder,
    "unicycle2": UnicycleSecondOrder,
}


def get_dynamics(config: Dict) -> DynamicsBase:
    return dynamics[config["dynamics"]](**config)
