from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


DYN_CONFIG_PATH = Path("../dynoplan/dynobench/models")
if not DYN_CONFIG_PATH.exists() or not DYN_CONFIG_PATH.is_dir():
    DYN_CONFIG_PATH = Path("data/dynamics")
    if not DYN_CONFIG_PATH.exists() or not DYN_CONFIG_PATH.is_dir():
        raise Exception("No dynamics config folder found")


class Availability(Enum):
    dataset = 0
    calculated = 1


class DynamicFactor(Enum):
    _ = 0
    q = 1
    u = 2


@dataclass
class ParameterData:
    availability: Availability
    static_size: int
    dynamic_factor: DynamicFactor


class Parameter(Enum):
    actions = ParameterData(Availability.dataset, 0, DynamicFactor.u)
    theta_0 = ParameterData(Availability.dataset, 1, DynamicFactor._)
    delta_0 = ParameterData(Availability.dataset, 1, DynamicFactor._)
    states = ParameterData(Availability.calculated, 0, DynamicFactor.q)
    theta_0_R2 = ParameterData(Availability.calculated, 2, DynamicFactor._)
    theta_start = ParameterData(Availability.calculated, 1, DynamicFactor._)
    theta_goal = ParameterData(Availability.calculated, 1, DynamicFactor._)
    area = ParameterData(Availability.dataset, 1, DynamicFactor._)
    area_blocked = ParameterData(Availability.dataset, 1, DynamicFactor._)
    area_free = ParameterData(Availability.dataset, 1, DynamicFactor._)
    env_width = ParameterData(Availability.dataset, 1, DynamicFactor._)
    env_height = ParameterData(Availability.dataset, 1, DynamicFactor._)
    n_obstacles = ParameterData(Availability.dataset, 1, DynamicFactor._)
    p_obstacles = ParameterData(Availability.dataset, 1, DynamicFactor._)
    cost = ParameterData(Availability.dataset, 1, DynamicFactor._)
    rel_l = ParameterData(Availability.dataset, 1, DynamicFactor._)
    rel_p = ParameterData(Availability.dataset, 1, DynamicFactor._)


class ParameterRegular(Enum):
    actions = Parameter.actions.value
    theta_0 = Parameter.theta_0.value
    delta_0 = Parameter.delta_0.value
    states = Parameter.states.value
    theta_0_R2 = Parameter.states.value


class ParameterConditioning(Enum):
    theta_start = Parameter.theta_start.value
    theta_goal = Parameter.theta_goal.value
    area = Parameter.area.value
    area_blocked = Parameter.area_blocked.value
    area_free = Parameter.area_free.value
    env_width = Parameter.env_width.value
    env_height = Parameter.env_height.value
    n_obstacles = Parameter.n_obstacles.value
    p_obstacles = Parameter.p_obstacles.value
    cost = Parameter.cost.value
    rel_l = Parameter.rel_l.value
    rel_p = Parameter.rel_p.value


