import ast
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

import diffmp

from .config import ParameterSet


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
    parameter_set: ParameterSet,
    available_columns: List[Tuple[str, str]],
) -> List[str]:
    cols: List[str] = []
    for param in parameter_set.iter_data():
        cols.extend([str(col) for col in param.cols])
    return cols


def load_dataset(
    config: diffmp.torch.Config, **kwargs
) -> diffmp.torch.DiffusionDataset:
    parameter_set = config.dynamics.parameter_set

    metadata = pq.read_table(config.dataset)
    available_columns: List[Tuple[str, str]] = [
        ast.literal_eval(col) for col in metadata.column_names
    ]

    load_columns = param_to_col(parameter_set, available_columns) + [
        str(("misc", "rel_c"))
    ]

    dataset = pd.read_parquet(config.dataset, columns=load_columns)
    # print(dataset.shape[0])
    dataset = calc_param(parameter_set, dataset)

    [dataset.drop(columns=p.cols, inplace=True) for p in parameter_set.required]

    dataset.drop_duplicates(inplace=True)
    # print(dataset.shape[0])
    if dataset.shape[0] > config.dataset_size:
        weights = np.ones(dataset.shape[0])
        for weight_col1, layer2 in config.weights.items():
            for weight_col2, value in layer2.items():
                weights *= dataset[(weight_col1, weight_col2)] ** value
            # weights = dataset[("misc", "weights")] = dataset.misc.rel_c**2
        dataset = dataset.sample(config.dataset_size, weights=weights)
    # dataset = dataset.sort_values(("misc", "weight"), ascending=False)
    # dataset = dataset.nlargest(config.dataset_size, ("misc", "rel_c"))
    reg_cols, cond_cols = parameter_set.get_columns()
    # print(reg_cols, "\n", cond_cols)
    regular = dataset[reg_cols]
    conditioning = dataset[cond_cols]
    regular = torch.tensor(regular.to_numpy(), device=diffmp.utils.DEVICE)
    conditioning = torch.tensor(conditioning.to_numpy(), device=diffmp.utils.DEVICE)

    return diffmp.torch.DiffusionDataset(regular=regular, conditioning=conditioning)


def calc_param(
    parameter_set: ParameterSet,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    for param in parameter_set.iter_calc():
        dataset = param.to(dataset)
    return dataset


def condition_for_sampling(
    config: diffmp.torch.Config, n_samples: int, instance: diffmp.problems.Instance
) -> torch.Tensor:
    data: Dict[Tuple[str, str], npt.NDArray] = {}
    # for condition in config.dynamics.parameter_set:
    for condition in config.dynamics.parameter_set.iter_condition():
        # if condition not in config.conditioning:
        #     continue
        match condition.name:
            case "theta_s":
                data[("env", "theta_s")] = (
                    np.ones(n_samples) * instance.robots[0].start[2]
                )
            case "theta_2_s":
                data[("env", "theta_2_s")] = (
                    np.ones(n_samples) * instance.robots[0].start[3]
                )
            case "Theta_s":
                data[("env", "Theta_s_x")] = np.ones(n_samples) * np.cos(
                    instance.robots[0].start[2]
                )
                data[("env", "Theta_s_y")] = np.ones(n_samples) * np.sin(
                    instance.robots[0].start[2]
                )
            case "Theta_2_s":
                data[("env", "Theta_2_s_x")] = np.ones(n_samples) * np.cos(
                    instance.robots[0].start[3]
                )
                data[("env", "Theta_2_s_y")] = np.ones(n_samples) * np.sin(
                    instance.robots[0].start[3]
                )
            case "Theta_g":
                data[("env", "Theta_g_x")] = np.ones(n_samples) * np.cos(
                    instance.robots[0].goal[2]
                )
                data[("env", "Theta_g_y")] = np.ones(n_samples) * np.sin(
                    instance.robots[0].goal[2]
                )
            case "Theta_2_g":
                data[("env", "Theta_2_g_x")] = np.ones(n_samples) * np.cos(
                    instance.robots[0].goal[3]
                )
                data[("env", "Theta_2_g_y")] = np.ones(n_samples) * np.sin(
                    instance.robots[0].goal[3]
                )
            case "theta_g":
                data[("env", "theta_g")] = (
                    np.ones(n_samples) * instance.robots[0].goal[2]
                )
            case "theta_2_g":
                data[("env", "theta_2_g")] = (
                    np.ones(n_samples) * instance.robots[0].goal[3]
                )
            case "s_s":
                data[("env", "s_s")] = np.ones(n_samples) * instance.robots[0].start[3]
            case "s_g":
                data[("env", "s_g")] = np.ones(n_samples) * instance.robots[0].goal[3]
            case "phi_s":
                data[("env", "phi_s")] = (
                    np.ones(n_samples) * instance.robots[0].start[4]
                )
            case "phi_g":
                data[("env", "phi_g")] = np.ones(n_samples) * instance.robots[0].goal[4]
            case "area_blocked":
                data[("environment", "area_blocked")] = (
                    np.ones(n_samples) * instance.environment.area_blocked
                )
            case "area_free":
                data[("environment", "area_free")] = (
                    np.ones(n_samples) * instance.environment.area_free
                )
            case "area":
                data[("environment", "area")] = (
                    np.ones(n_samples) * instance.environment.area
                )
            case "env_width":
                data[("environment", "env_width")] = (
                    np.ones(n_samples) * instance.environment.env_width
                )
            case "env_height":
                data[("environment", "env_height")] = (
                    np.ones(n_samples) * instance.environment.env_height
                )
            case "n_obstacles":
                data[("environment", "n_obstacles")] = np.ones(n_samples) * len(
                    instance.environment.obstacles
                )
            case "p_obstacles":
                data[("environment", "p_obstacles")] = (
                    np.ones(n_samples)
                    * instance.environment.area_blocked
                    / instance.environment.area
                )
            case "rel_l":
                data[("misc", "rel_l")] = np.linspace(0, 1, n_samples)
            case "rel_p":
                data[("misc", "rel_p")] = np.linspace(0, 1, n_samples)
            case "delta_0":
                data[("misc", "delta_0")] = np.ones(n_samples) * 0.5
            case _:
                raise NotImplementedError(
                    f"{condition.name} is not implemented as conditioning"
                )

    df = pd.DataFrame(data)

    return torch.tensor(df.to_numpy(), device=diffmp.utils.DEVICE)


def theta_to_Theta(df: pd.DataFrame, col1: str, i: str = "0") -> pd.DataFrame:
    df[(col1, f"Theta_{i}_x")] = np.cos(df[(col1, f"theta_{i}")])
    df[(col1, f"Theta_{i}_y")] = np.sin(df[(col1, f"theta_{i}")])
    return df


def Theta_to_theta(df: pd.DataFrame, col1: str, i: str = "0") -> pd.DataFrame:
    df[(col1, f"theta_{i}")] = np.atan2(
        df[(col1, f"Theta_{i}_y")], df[(col1, f"Theta_{i}_x")]
    )
    return df
