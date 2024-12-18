from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


DYN_CONFIG_PATH = Path("../../dynoplan/dynobench/models")
if not DYN_CONFIG_PATH.exists() or not DYN_CONFIG_PATH.is_dir():
    # print(Path().absolute())
    DYN_CONFIG_PATH = Path("data/dynamics")
    if not DYN_CONFIG_PATH.exists() or not DYN_CONFIG_PATH.is_dir():
        raise Exception("No dynamics config folder found")


class Availability(Enum):
    dataset = 0
    calculated = 1


class DynamicFactor(Enum):
    none = 0
    q = 1
    u = 2

# Not really happy with this, maybe change this later

@dataclass(frozen=True)
class ParameterData:
    availability: Availability
    static_size: int
    dynamic_factor: DynamicFactor


class Parameter(Enum):
    actions = ParameterData(Availability.dataset, 0, DynamicFactor.u)
    theta_0 = ParameterData(Availability.dataset, 1, DynamicFactor.none)
    delta_0 = ParameterData(Availability.dataset, 1, DynamicFactor.none)
    area = ParameterData(Availability.dataset, 1, DynamicFactor.none)
    area_blocked = ParameterData(Availability.dataset, 1, DynamicFactor.none)
    area_free = ParameterData(Availability.dataset, 1, DynamicFactor.none)
    env_width = ParameterData(Availability.dataset, 1, DynamicFactor.none)
    env_height = ParameterData(Availability.dataset, 1, DynamicFactor.none)
    n_obstacles = ParameterData(Availability.dataset, 1, DynamicFactor.none)
    p_obstacles = ParameterData(Availability.dataset, 1, DynamicFactor.none)
    cost = ParameterData(Availability.dataset, 1, DynamicFactor.none)
    rel_l = ParameterData(Availability.dataset, 1, DynamicFactor.none)
    rel_p = ParameterData(Availability.dataset, 1, DynamicFactor.none)

    states = ParameterData(Availability.calculated, 0, DynamicFactor.q)
    theta_0_R2 = ParameterData(Availability.calculated, 2, DynamicFactor.none)
    theta_start = ParameterData(Availability.calculated, 1, DynamicFactor.none)
    theta_goal = ParameterData(Availability.calculated, 1, DynamicFactor.none)


class ParameterRegular(Enum):
    actions = Parameter.actions
    theta_0 = Parameter.theta_0
    delta_0 = Parameter.delta_0
    states = Parameter.states
    theta_0_R2 = Parameter.theta_0_R2


class ParameterConditioning(Enum):
    theta_start = Parameter.theta_start
    theta_goal = Parameter.theta_goal
    area = Parameter.area
    area_blocked = Parameter.area_blocked
    area_free = Parameter.area_free
    env_width = Parameter.env_width
    env_height = Parameter.env_height
    n_obstacles = Parameter.n_obstacles
    p_obstacles = Parameter.p_obstacles
    cost = Parameter.cost
    rel_l = Parameter.rel_l
    rel_p = Parameter.rel_p
