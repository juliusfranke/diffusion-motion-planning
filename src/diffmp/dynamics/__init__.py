from enum import Enum

import diffmp
from .base import DynamicsBase
from .unicycle1 import UnicycleFirstOrder
from .unicycle2 import UnicycleSecondOrder
from .car1 import CarWithTrailers
from .integrator2_3d import Integrator2_3D


class Dynamics(Enum):
    unicycle1 = UnicycleFirstOrder
    unicycle2 = UnicycleSecondOrder
    car_with_trailers = CarWithTrailers
    integrator2_3d = Integrator2_3D


def get_dynamics(name: str, timesteps: int, n_robots: int = 1) -> DynamicsBase:
    path = (diffmp.utils.DYN_CONFIG_PATH / name).with_suffix(".yaml")
    config = diffmp.utils.load_yaml(path)
    config["timesteps"] = timesteps
    dyn = Dynamics[config["dynamics"]].value(**config, name=name, n_robots=n_robots)
    return dyn


__all__ = [
    "DynamicsBase",
    "UnicycleFirstOrder",
    "UnicycleSecondOrder",
    "Dynamics",
    "get_dynamics",
]
