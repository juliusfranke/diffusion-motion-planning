from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

import diffmp

from .config import (
    Availability,
    DynamicFactor,
    Parameter,
    ParameterConditioning,
    ParameterData,
    ParameterRegular,
)


def load_yaml(path: Path) -> Dict[Any, Any]:
    if not path.exists():
        raise FileNotFoundError()
    if path.is_dir():
        raise IsADirectoryError()
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    assert isinstance(data, dict)
    return data


def load_dataset(
    config: diffmp.torch.Config, **kwargs
) -> diffmp.torch.DiffusionDataset:
    def param_to_col(
        params: List[ParameterRegular | ParameterConditioning],
    ) -> List[str]:
        cols: List[str] = []
        for param in params:
            cols.extend([str(col) for col in available_columns if param.name in col])
        return cols

    load_regular: List[ParameterRegular | ParameterConditioning] = []
    load_conditioning: List[ParameterRegular | ParameterConditioning] = []
    calc_regular: List[ParameterRegular | ParameterConditioning] = []
    calc_conditioning: List[ParameterRegular | ParameterConditioning] = []

    metadata = pq.read_table(config.dataset)
    available_columns: List[Tuple[str, str]] = metadata.column_names

    for param in config.regular:
        match param.value.value.availability:
            case Availability.dataset:
                load_regular.append(param)
            case Availability.calculated:
                calc_regular.append(param)

    for param in config.conditioning:
        match param.value.value.availability:
            case Availability.dataset:
                load_conditioning.append(param)
            case Availability.calculated:
                calc_conditioning.append(param)

    load_columns = param_to_col(load_regular + load_conditioning)

    dataset = pd.read_parquet(config.dataset, columns=load_columns)
    dataset = calc_param(config, calc_regular, dataset)
    dataset = dataset.sample(config.dataset_size)
    dataset = torch.tensor(dataset.to_numpy(), device=diffmp.utils.DEVICE)

    return diffmp.torch.DiffusionDataset(regular=dataset, conditioning=None)


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
    data: Dict[Tuple[str, str], torch.Tensor] = {}
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
