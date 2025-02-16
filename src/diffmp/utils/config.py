from collections.abc import Callable
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from typing import Any, List, Tuple, Optional

import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DYN_CONFIG_PATH = Path("../dynoplan/dynobench/models")
if not DYN_CONFIG_PATH.exists() or not DYN_CONFIG_PATH.is_dir():
    DYN_CONFIG_PATH = Path("data/dynamics")
    if not DYN_CONFIG_PATH.exists() or not DYN_CONFIG_PATH.is_dir():
        raise Exception("No dynamics config folder found")


class AnyStr:
    """Returns True when compared to any str"""

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return True
        return False


@dataclass
class Parameter:
    name: str
    size: int
    weight: float
    cols: List[Tuple[str, str]]


@dataclass
class DatasetParameter(Parameter):
    pass


@dataclass
class CalculatedParameter(Parameter):
    requires: List[str]
    to: Callable[[pd.DataFrame], pd.DataFrame]
    fr: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


class ParameterSet:
    def __init__(self):
        self.data_param_regular: List[DatasetParameter] = []
        self.calc_param_regular: List[CalculatedParameter] = []
        self.data_param_condition: List[DatasetParameter] = []
        self.calc_param_condition: List[CalculatedParameter] = []
        self.required: List[DatasetParameter] = []

    def add_parameter(
        self, parameter: DatasetParameter | CalculatedParameter, condition: bool
    ):
        if isinstance(parameter, DatasetParameter):
            param_list = (
                self.data_param_condition if condition else self.data_param_regular
            )
            if parameter.name in {param.name for param in param_list}:
                return None
            param_list.append(parameter)

        elif isinstance(parameter, CalculatedParameter):
            param_list = (
                self.calc_param_condition if condition else self.calc_param_regular
            )
            if parameter.name in {param.name for param in param_list}:
                return None
            if len(req := parameter.requires) > 0:
                pass
            param_list.append(parameter)

    def add_parameters(
        self, parameters: List[DatasetParameter | CalculatedParameter], condition: bool
    ):
        for param in parameters:
            self.add_parameter(param, condition)

    def set_weight(self, parameter_name: str, weight: float) -> None:
        self[parameter_name].weight = weight

    def get_columns(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        col_reg = []
        col_cond = []

        for param in self.iter_regular():
            col_reg.extend(param.cols)

        for param in self.iter_condition():
            col_cond.extend(param.cols)

        return col_reg, col_cond

    def __getitem__(self, param_name: str) -> DatasetParameter | CalculatedParameter:
        for param in self:
            if param.name == param_name:
                return param
        raise KeyError

    def iter_regular(self):
        return iter(
            sorted(self.data_param_regular, key=lambda p: p.name)
            + sorted(self.calc_param_regular, key=lambda p: p.name)
        )

    def iter_condition(self):
        return iter(
            sorted(self.data_param_condition, key=lambda p: p.name)
            + sorted(self.calc_param_condition, key=lambda p: p.name)
        )

    def iter_data(self):
        return iter(
            sorted(self.data_param_regular, key=lambda p: p.name)
            + sorted(self.data_param_condition, key=lambda p: p.name)
            + sorted(self.required, key=lambda p: p.name)
        )

    def iter_calc(self):
        return iter(
            sorted(self.calc_param_regular, key=lambda p: p.name)
            + sorted(self.calc_param_condition, key=lambda p: p.name)
        )

    def __iter__(self):
        return iter(
            sorted(self.data_param_regular, key=lambda p: p.name)
            + sorted(self.calc_param_regular, key=lambda p: p.name)
            + sorted(self.data_param_condition, key=lambda p: p.name)
            + sorted(self.calc_param_condition, key=lambda p: p.name)
        )

    def in_out_size(self) -> Tuple[int, int]:
        out_size = len(self.data_param_regular) + len(self.calc_param_regular)
        in_size = (
            out_size + len(self.data_param_condition) + len(self.calc_param_condition)
        )
        return (in_size, out_size)


def calc_rel_p(df: pd.DataFrame) -> pd.DataFrame:
    agg = {("misc", "count"): "sum"}
    if ("misc", "cost") in df.columns:
        agg[("misc", "cost")] = "mean"
    cols = df.columns
    df = pd.DataFrame(
        df.groupby(df.columns.drop([("misc", "count")]).to_list(), as_index=False).agg(
            agg
        )
    )
    df = df.reindex(columns=cols)

    env_group = [col for col in df.columns if col == ("env", AnyStr())]
    if env_group:
        df[("misc", "group_max")] = df.groupby(env_group)["count"].transform("max")
    else:
        df[("misc", "group_max")] = df[("misc", "count")].max()

    df[("misc", "rel_p")] = df[("misc", "count")] / df[("misc", "group_max")]

    df.drop(columns=[("misc", "group_max")], inplace=True)
    return df


def get_default_parameter_set() -> ParameterSet:
    param_set = ParameterSet()
    param_set.add_parameters(
        [
            DatasetParameter("area", 1, 0, [("env", "area")]),
            DatasetParameter("area_blocked", 1, 0, [("env", "area_blocked")]),
            DatasetParameter("area_free", 1, 0, [("env", "area_free")]),
            DatasetParameter("env_width", 1, 0, [("env", "env_width")]),
            DatasetParameter("env_height", 1, 0, [("env", "env_height")]),
            DatasetParameter("n_obstacles", 1, 0, [("env", "n_obstacles")]),
            DatasetParameter("p_obstacles", 1, 0, [("env", "p_obstacles")]),
            DatasetParameter("delta_0", 1, 0, [("misc", "delta_0")]),
            DatasetParameter("cost", 1, 0, [("misc", "cost")]),
            DatasetParameter("rel_l", 1, 0, [("misc", "rel_l")]),
            DatasetParameter("count", 1, 0, [("misc", "count")]),
            CalculatedParameter(
                "rel_p", 1, 1, [("misc", "rel_p")], ["count"], calc_rel_p
            ),
        ],
        condition=True,
    )

    return param_set
