import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

import diffmp

from .config import CalculatedParameter, DatasetParameter


def load_yaml(path: Path) -> Dict[Any, Any]:
    if not path.exists():
        raise FileNotFoundError()
    if path.is_dir():
        raise IsADirectoryError()
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    assert isinstance(data, dict)
    return data


def param_to_col(
    parameters: List[DatasetParameter],
    available_columns: List[Tuple[str, str]],
) -> List[str]:
    cols: List[str] = []
    for param in parameters:
        col_1 = param.col_1
        col_2 = param.col_2
        cols.extend(
            [
                str(col)
                for col in available_columns
                if col_1 == col[0] and col_2 == col[1]
            ]
        )
    return cols


def load_dataset(
    config: diffmp.torch.Config, **kwargs
) -> diffmp.torch.DiffusionDataset:
    # load_regular: List[DatasetParameter | CalculatedParameter] = []
    # load_conditioning: List[DatasetParameter | CalculatedParameter] = []

    # calc_regular: List[ParameterRegular | ParameterConditioning] = []
    # calc_conditioning: List[ParameterRegular | ParameterConditioning] = []

    parameter_set = config.dynamics.parameter_set
    parameters_to_load = [
        param
        for param in parameter_set.iter_data()
        if param in config.regular + config.conditioning
    ]
    parameters_to_calc = [
        param
        for param in parameter_set.iter_calc()
        if param in config.regular + config.conditioning
    ]
    metadata = pq.read_table(config.dataset)
    available_columns: List[Tuple[str, str]] = [
        ast.literal_eval(col) for col in metadata.column_names
    ]

    load_columns = param_to_col(parameters_to_load, available_columns)

    dataset = pd.read_parquet(config.dataset, columns=load_columns)
    dataset = calc_param(config, parameters_to_calc, dataset)
    dataset = dataset.sample(config.dataset_size)
    dataset = torch.tensor(dataset.to_numpy(), device=diffmp.utils.DEVICE)

    return diffmp.torch.DiffusionDataset(regular=dataset, conditioning=None)


def calc_param(
    config: diffmp.torch.Config,
    calc: List[CalculatedParameter],
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    for param in calc:
        dataset[f"{param}"] = param.to()
    return dataset


# def param_size(param: ParameterData, config: diffmp.torch.Config):
#     size = param.static_size
#     match param.dynamic_factor:
#         case DynamicFactor.u:
#             size += config.timesteps * len(config.dynamics.u)
#         case DynamicFactor.q:
#             size += config.timesteps * len(config.dynamics.q)
#         case DynamicFactor.none:
#             pass
#         case _:
#             raise NotImplementedError(
#                 f"{param.dynamic_factor} not implemented for param_size"
#             )

#     return size


# def input_output_size(config: diffmp.torch.Config) -> Tuple[int, int]:
#     in_size = 0
#     out_size = 0
# for regular in config.regular:
#     out_size += param_size(regular.value.value, config)

# for conditioning in config.conditioning:
#     in_size += param_size(conditioning.value.value, config)

# in_size += out_size + 1

# return (in_size, out_size)


def condition_for_sampling(
    config: diffmp.torch.Config, n_samples: int, instance: diffmp.problems.Instance
) -> torch.Tensor:
    data: Dict[Tuple[str, str], torch.Tensor] = {}
    for condition in config.dynamics.parameter_set:
        if condition not in config.conditioning:
            continue
        match condition.name:
            case "theta_s":
                data[("q_start", "theta_0")] = (
                    torch.ones(n_samples, device=diffmp.utils.DEVICE)
                    * instance.robots[0].start[2]
                )
            case "theta_g":
                data[("q_goal", "theta_0")] = (
                    torch.ones(n_samples, device=diffmp.utils.DEVICE)
                    * instance.robots[0].goal[2]
                )
            case "area_blocked":
                data[("environment", "area_blocked")] = (
                    torch.ones(n_samples, device=diffmp.utils.DEVICE)
                    * instance.environment.area_blocked
                )
            case "area_free":
                data[("environment", "area_free")] = (
                    torch.ones(n_samples, device=diffmp.utils.DEVICE)
                    * instance.environment.area_free
                )
            case "area":
                data[("environment", "area")] = (
                    torch.ones(n_samples, device=diffmp.utils.DEVICE)
                    * instance.environment.area
                )
            case "env_width":
                data[("environment", "env_width")] = (
                    torch.ones(n_samples, device=diffmp.utils.DEVICE)
                    * instance.environment.env_width
                )
            case "env_height":
                data[("environment", "env_height")] = (
                    torch.ones(n_samples, device=diffmp.utils.DEVICE)
                    * instance.environment.env_height
                )
            case "n_obstacles":
                data[("environment", "n_obstacles")] = torch.ones(
                    n_samples, device=diffmp.utils.DEVICE
                ) * len(instance.environment.obstacles)
            case "p_obstacles":
                data[("environment", "p_obstacles")] = (
                    torch.ones(n_samples, device=diffmp.utils.DEVICE)
                    * instance.environment.area_blocked
                    / instance.environment.area
                )
            case "rel_l":
                data[("misc", "rel_l")] = torch.linspace(
                    0, 1, n_samples, device=diffmp.utils.DEVICE
                )
            case "rel_p":
                data[("misc", "rel_p")] = torch.linspace(
                    0, 1, n_samples, device=diffmp.utils.DEVICE
                )
            case _:
                raise NotImplementedError

    df = pd.DataFrame(data)

    return torch.Tensor(df.to_numpy())
