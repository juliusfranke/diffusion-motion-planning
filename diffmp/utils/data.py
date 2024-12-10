from enum import Enum
from typing import Tuple

from torch.utils.data import TensorDataset

from diffmp.torch.model import Config


class ParameterRegular(Enum):
    actions = 0
    theta_0 = 1
    delta_0 = 2


class ParameterConditioning(Enum):
    theta_start = 0
    theta_goal = 1
    area = 2
    area_blocked = 3
    area_free = 4
    env_width = 5
    env_height = 6
    n_obstacles = 7
    p_obstacles = 8
    cost = 9
    location = 10


class ParameterCalculated(Enum):
    states = 0
    theta_0_R2 = 1


def load_data(dataset: str, **kwargs) -> TensorDataset:
    return TensorDataset()


def input_output_size(config: Config) -> Tuple[int, int]:
    in_size = 0
    out_size = 0
    for regular in config.regular:
        match regular:
            case regular.actions:
                out_size += config.timesteps * len(config.dynamics.u)
            case _:
                out_size += 1

    for calculated in config.calculated:
        match calculated:
            case calculated.states:
                out_size += config.timesteps * len(config.dynamics.q)
            case calculated.theta_0_R2:
                out_size += 2

    for conditioning in config.conditioning:
        match conditioning:
            case _:
                out_size += 1

    return (in_size, out_size)
