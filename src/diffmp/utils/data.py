from __future__ import annotations
import numpy as np
import ast
import h5py
import numpy.typing as npt
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
import yaml

import diffmp

import diffmp.problems as pb
import diffmp.torch as to
from .config import ParameterSet
from .h5_helpers import get_columns, get_array, get_string_array

if TYPE_CHECKING:
    from diffmp.torch import Config


def load_yaml(path: Path) -> dict[Any, Any]:
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
) -> list[str]:
    cols: list[str] = []
    for param in parameter_set.iter_data():
        cols.extend([str(col) for col in param.cols])
    return cols


def load_from_instances(
    instances: list[pb.Instance], columns: list[tuple[str, str]], n_robots: int
) -> pd.DataFrame:
    data = {}
    for column in columns:
        data[column] = []
        for instance in instances:
            match column[1]:
                case "area":
                    data[column].append(instance.environment.area)
                case "x_s":
                    robot_idx = int(column[0][6:])
                    data[column].append(instance.robots[robot_idx].start[0])
                case "x_g":
                    robot_idx = int(column[0][6:])
                    data[column].append(instance.robots[robot_idx].goal[0])
                case "y_s":
                    robot_idx = int(column[0][6:])
                    data[column].append(instance.robots[robot_idx].start[1])
                case "y_g":
                    robot_idx = int(column[0][6:])
                    data[column].append(instance.robots[robot_idx].goal[1])
                case "theta_s":
                    robot_idx = int(column[0][6:])
                    data[column].append(instance.robots[robot_idx].start[2])
                case "theta_g":
                    robot_idx = int(column[0][6:])
                    data[column].append(instance.robots[robot_idx].goal[2])
    data[("env", "idx")] = [i for i in range(len(instances))]
    df = pd.DataFrame(data)
    return df


def discretize_instances(
    instances: list[pb.Instance], discretize_config: to.DiscretizeConfig
) -> torch.Tensor:
    dim = instances[0].environment.dim
    n = discretize_config.resolution
    dis_tensor = torch.empty((len(instances), n, n))
    for i, instance in enumerate(instances):
        dis = torch.Tensor(instance.environment.discretize(discretize_config))
        if dim == pb.Dim.TWO_D:
            dis = dis.reshape(n, n)
        dis_tensor[i] = dis
    return dis_tensor


def load_dataset(config: Config, **kwargs) -> diffmp.torch.DiffusionDataset:
    parameter_set = config.dynamics.parameter_set

    load_columns = param_to_col(parameter_set)
    load_columns = [ast.literal_eval(col) for col in load_columns]
    load_columns += [("misc", "rel_c"), ("env", "idx")]
    with h5py.File(config.dataset, "r") as f:
        env_strings = get_string_array(f, "environments")

        length_group = f.get(f"length_{config.timesteps:03}")
        assert isinstance(length_group, h5py.Group)
        instance_columns = []
        scalar_columns = []
        scalar_arrays = []
        # ar = np.array([])
        for i in range(config.n_robots):
            robot_group = length_group.get(f"robot_{i:03}")
            assert isinstance(robot_group, h5py.Group)

            dataset_columns = get_columns(robot_group)
            scalar_cols_idxs: list[int] = []
            instance_columns: list[tuple[str, str]] = []
            scalar_columns: list[tuple[str, str]] = []

            for column in load_columns:
                if column == ("misc", "robot_idx"):
                    continue
                if column[0] == "env" and column[1] != "idx":
                    instance_columns.append(column)
                    continue
                if column[0][:5] == "robot":
                    instance_columns.append(column)
                    continue
                scalar_columns.append(column)
                if column not in dataset_columns:
                    raise Exception(f"Col {column} not in dataset")
                scalar_cols_idxs.append(dataset_columns.index(column))
            scalar_arrays.append(get_array(robot_group)[:, scalar_cols_idxs])

    instances = [pb.Instance.from_dict(yaml.safe_load(data)) for data in env_strings]
    instance_dataset = load_from_instances(instances, instance_columns, config.n_robots)
    pds = [
        pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(scalar_columns))
        for data in scalar_arrays
    ]
    for i in range(config.n_robots):
        pds[i][("misc", "robot_idx")] = i
    scalar_dataset = pd.concat(pds)
    # scalar_dataset = pd.DataFrame(ar, columns=pd.MultiIndex.from_tuples(scalar_columns))
    dataset = scalar_dataset.merge(instance_dataset)
    dataset = calc_param(parameter_set, dataset)

    [dataset.drop(columns=p.cols, inplace=True) for p in parameter_set.required]

    dataset.drop_duplicates(inplace=True)
    if dataset.shape[0] > config.dataset_size:
        weights = np.ones(dataset.shape[0])
        for weight_col1, layer2 in config.weights.items():
            for weight_col2, value in layer2.items():
                weights *= dataset[(weight_col1, weight_col2)] ** value
            # weights = dataset[("misc", "weights")] = dataset.misc.rel_c**2
        dataset = dataset.sample(config.dataset_size, weights=weights)
    reg_cols, cond_cols = parameter_set.get_columns()
    regular = dataset[reg_cols]
    conditioning = dataset[cond_cols]
    # print(f"{reg_cols=}")
    if config.classify_actions:
        action_cols = [col for col in regular.columns if col[0] == "actions"]
        print(f"{pd.MultiIndex.from_tuples(cond_cols)=}")
        actions_classes = torch.Tensor(
            classify_actions(regular, action_cols, -0.5, 0.5).to_numpy(),
            device=diffmp.utils.DEVICE,
        )
    else:
        actions_classes = None
    regular = torch.tensor(regular.to_numpy(), device=diffmp.utils.DEVICE)
    conditioning = torch.tensor(conditioning.to_numpy(), device=diffmp.utils.DEVICE)
    if config.discretize is not None:
        dis = discretize_instances(instances, config.discretize)
        dis = dis.reshape(dis.shape[0], 1, dis.shape[1], dis.shape[2])
        dis.to(diffmp.utils.DEVICE)
    else:
        dis = None
    if config.robot_embedding:
        row_to_id = torch.tensor(
            scalar_dataset[("misc", "robot_idx")].to_numpy(), dtype=torch.int
        )
    else:
        row_to_id = None

    return diffmp.torch.DiffusionDataset(
        regular=regular,
        conditioning=conditioning,
        discretized=dis,
        row_to_env=dataset[("env", "idx")].to_numpy(),
        row_to_id=row_to_id,
        action_classes=actions_classes,
    )


def calc_param(
    parameter_set: ParameterSet,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    for param in parameter_set.iter_calc():
        assert param.to is not None
        dataset = param.to(dataset)
    return dataset


def condition_for_sampling(
    config: diffmp.torch.Config,
    n_samples: int,
    instance: diffmp.problems.Instance,
    robot_idx: int,
) -> torch.Tensor:
    data: dict[tuple[str, str], npt.NDArray] = {}
    n_robots = len(instance.robots)
    for condition in config.dynamics.parameter_set.iter_condition():
        match condition.name:
            case "robot_id":
                data[("misc", "robot_idx")] = np.ones(n_samples) * robot_idx
            case "x_s":
                for i in range(n_robots):
                    data[(f"robot_{i:03}", "x_s")] = (
                        np.ones(n_samples) * instance.robots[i].start[0]
                    )
            case "x_g":
                for i in range(n_robots):
                    data[(f"robot_{i:03}", "x_g")] = (
                        np.ones(n_samples) * instance.robots[i].goal[0]
                    )
            case "y_s":
                for i in range(n_robots):
                    data[(f"robot_{i:03}", "y_s")] = (
                        np.ones(n_samples) * instance.robots[i].start[1]
                    )
            case "y_g":
                for i in range(n_robots):
                    data[(f"robot_{i:03}", "y_g")] = (
                        np.ones(n_samples) * instance.robots[i].goal[1]
                    )
            case "theta_s":
                for i in range(n_robots):
                    data[(f"robot_{i:03}", "theta_s")] = (
                        np.ones(n_samples) * instance.robots[i].start[2]
                    )
            case "theta_2_s":
                data[("env", "theta_2_s")] = (
                    np.ones(n_samples) * instance.robots[0].start[3]
                )
            case "Theta_s":
                for i in range(n_robots):
                    data[(f"robot_{i:03}", "Theta_s_x")] = np.ones(n_samples) * np.cos(
                        instance.robots[i].start[2]
                    )
                for i in range(n_robots):
                    data[(f"robot_{i:03}", "Theta_s_y")] = np.ones(n_samples) * np.sin(
                        instance.robots[i].start[2]
                    )
            case "Theta_2_s":
                data[("env", "Theta_2_s_x")] = np.ones(n_samples) * np.cos(
                    instance.robots[0].start[3]
                )
                data[("env", "Theta_2_s_y")] = np.ones(n_samples) * np.sin(
                    instance.robots[0].start[3]
                )
            case "Theta_g":
                for i in range(n_robots):
                    data[(f"robot_{i:03}", "Theta_g_x")] = np.ones(n_samples) * np.cos(
                        instance.robots[i].goal[2]
                    )
                for i in range(n_robots):
                    data[(f"robot_{i:03}", "Theta_g_y")] = np.ones(n_samples) * np.sin(
                        instance.robots[i].goal[2]
                    )
            case "Theta_2_g":
                data[("env", "Theta_2_g_x")] = np.ones(n_samples) * np.cos(
                    instance.robots[0].goal[3]
                )
                data[("env", "Theta_2_g_y")] = np.ones(n_samples) * np.sin(
                    instance.robots[0].goal[3]
                )
            case "theta_g":
                for i in range(n_robots):
                    data[(f"robot_{i:03}", "theta_g")] = (
                        np.ones(n_samples) * instance.robots[i].goal[2]
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
    # print(df.columns)
    # breakpoint()

    return torch.tensor(df.to_numpy(), device=diffmp.utils.DEVICE)


def theta_to_Theta(
    df: pd.DataFrame, col1: str | list[str], i: str = "0"
) -> pd.DataFrame:
    if isinstance(col1, str):
        col1 = [col1]

    for c1 in col1:
        df[(c1, f"Theta_{i}_x")] = np.cos(df[(c1, f"theta_{i}")])
        df[(c1, f"Theta_{i}_y")] = np.sin(df[(c1, f"theta_{i}")])
    return df


def Theta_to_theta(
    df: pd.DataFrame, col1: str | list[str], i: str = "0"
) -> pd.DataFrame:
    if isinstance(col1, str):
        col1 = [col1]

    for c1 in col1:
        df[(c1, f"theta_{i}")] = np.atan2(
            df[(c1, f"Theta_{i}_y")], df[(c1, f"Theta_{i}_x")]
        )
    return df


def classify_actions(
    df: pd.DataFrame | pd.Series,
    action_cols: list[tuple[str, str]],
    low: float,
    high: float,
    eps: float = 1e-2,
) -> pd.DataFrame | pd.Series:
    df = df.copy()
    class_cols = []
    for col in action_cols:
        classifier_col = (col[0], f"{col[1]}_classifier")
        class_cols.append(classifier_col)
        df[classifier_col] = np.ones(len(df))
        df.loc[df[col] <= low + eps, classifier_col] = 0
        df.loc[df[col] >= high - eps, classifier_col] = 2
    return df[class_cols]  # type:ignore


def declassify_actions(
    df: pd.DataFrame,
    action_cols: list[tuple[str, str]],
    low: float,
    high: float,
    eps: float = 1e-3,
) -> pd.DataFrame:
    for col in action_cols:
        classifier_col = (col[0], f"{col[1]}_classifier)")
        df.loc[df[classifier_col] == -1, col] = low
        df.loc[df[classifier_col] == 1, col] = high
    return df
