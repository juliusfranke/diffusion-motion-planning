from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd


import yaml
import diffmp
from diffmp.problems import environment
from .config import (
    DynamicFactor,
    ParameterConditioning,
    ParameterData,
    Availability,
    Parameter,
    ParameterRegular,
)
import torch
from torch.utils.data import TensorDataset


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError()
    if path.is_dir():
        raise IsADirectoryError()
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data


def load_dataset(config: diffmp.torch.Config, **kwargs) -> TensorDataset:
    load: List[ParameterRegular | ParameterConditioning] = []
    calc: List[ParameterRegular | ParameterConditioning] = []

    for param in config.regular + config.conditioning:
        match param.value.value.availability:
            case Availability.dataset:
                load.append(param)
            case Availability.calculated:
                calc.append(param)
            case _:
                raise NotImplementedError(
                    f"{param.value.value.availability} not implemented for load_dataset"
                )

    load_columns = [param.name for param in load]
    dataset = pd.read_parquet(config.dataset, columns=load_columns)
    dataset = calc_param(config, calc, dataset)

    return TensorDataset()


def calc_param(
    config: diffmp.torch.Config,
    calc: List[ParameterRegular | ParameterConditioning],
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    for param in calc:
        match param.value:
            case Parameter.states:
                pass
            case Parameter.theta_0_R2:
                pass
            case Parameter.theta_start:
                pass
            case Parameter.theta_goal:
                pass
            case _:
                pass
    return dataset


def param_size(param: ParameterData, config: diffmp.torch.Config):
    size = param.static_size
    match param.dynamic_factor:
        case DynamicFactor.u:
            size += config.timesteps * len(config.dynamics.u)
        case DynamicFactor.q:
            size += config.timesteps * len(config.dynamics.q)
        case DynamicFactor.none:
            pass
        case _:
            raise NotImplementedError(
                f"{param.dynamic_factor} not implemented for param_size"
            )

    return size


def input_output_size(config: diffmp.torch.Config) -> Tuple[int, int]:
    in_size = 0
    out_size = 0
    for regular in config.regular:
        out_size += param_size(regular.value.value, config)

    for conditioning in config.conditioning:
        in_size += param_size(conditioning.value.value, config)

    in_size += out_size + 1

    return (in_size, out_size)


def condition_for_sampling(
    config: diffmp.torch.Config, n_samples: int, instance: diffmp.problems.Instance
) -> torch.Tensor:
    data: Dict[Tuple, torch.Tensor] = {}
    for condition in ParameterConditioning:
        if condition not in config.conditioning:
            continue
        match condition.value:
            case Parameter.theta_start:
                data[("q_start", "theta_0")] = (
                    torch.ones(n_samples) * instance.robots[0].start[2]
                )
            case Parameter.theta_goal:
                data[("q_goal", "theta_0")] = (
                    torch.ones(n_samples) * instance.robots[0].goal[2]
                )
            case Parameter.area_blocked:
                data[("environment", "area_blocked")] = (
                    torch.ones(n_samples) * instance.environment.area_blocked
                )
            case Parameter.area_free:
                data[("environment", "area_free")] = (
                    torch.ones(n_samples) * instance.environment.area_free
                )
            case Parameter.area:
                data[("environment", "area")] = (
                    torch.ones(n_samples) * instance.environment.area
                )
            case Parameter.env_width:
                data[("environment", "env_width")] = (
                    torch.ones(n_samples) * instance.environment.env_width
                )
            case Parameter.env_height:
                data[("environment", "env_height")] = (
                    torch.ones(n_samples) * instance.environment.env_height
                )
            case Parameter.n_obstacles:
                data[("environment", "n_obstacles")] = torch.ones(n_samples) * len(
                    instance.environment.obstacles
                )
            case Parameter.p_obstacles:
                data[("environment", "p_obstacles")] = (
                    torch.ones(n_samples)
                    * instance.environment.area_blocked
                    / instance.environment.area
                )
            case Parameter.rel_l:
                data[("misc", "rel_l")] = torch.linspace(0, 1, n_samples)
            case Parameter.rel_p:
                data[("misc", "rel_p")] = torch.linspace(0, 1, n_samples)
            case Parameter.cost:
                raise NotImplementedError
            case _:
                raise NotImplementedError(f"{condition} is not implemented")

    df = pd.DataFrame(data)

    return torch.Tensor(df.to_numpy())
