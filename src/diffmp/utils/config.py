from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DYN_CONFIG_PATH = Path("../../dynoplan/dynobench/models")
if not DYN_CONFIG_PATH.exists() or not DYN_CONFIG_PATH.is_dir():
    DYN_CONFIG_PATH = Path("data/dynamics")
    if not DYN_CONFIG_PATH.exists() or not DYN_CONFIG_PATH.is_dir():
        raise Exception("No dynamics config folder found")


class AnyStr:
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return True
        return False


@dataclass(frozen=True)
class Parameter:
    name: str
    size: int


@dataclass(frozen=True)
class DatasetParameter(Parameter):
    col_1: str | AnyStr = AnyStr()
    col_2: str | AnyStr = AnyStr()


@dataclass(frozen=True)
class CalculatedParameter(Parameter):
    requires: List[str]
    to: Callable
    after_groupby: bool = False


class ParameterSet:
    def __init__(self):
        self.data_param_regular: List[DatasetParameter] = []
        self.calc_param_regular: List[CalculatedParameter] = []
        self.data_param_condition: List[DatasetParameter] = []
        self.calc_param_condition: List[CalculatedParameter] = []

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
            param_list.append(parameter)

    def add_parameters(
        self, parameters: List[DatasetParameter | CalculatedParameter], condition: bool
    ):
        for param in parameters:
            self.add_parameter(param, condition)

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


def get_default_parameter_set() -> ParameterSet:
    param_set = ParameterSet()
    param_set.add_parameters(
        [
            DatasetParameter(name="area", size=1, col_1="env", col_2="area"),
            DatasetParameter(
                name="area_blocked", size=1, col_1="env", col_2="area_blocked"
            ),
            DatasetParameter(name="area_free", size=1, col_1="env", col_2="area_free"),
            DatasetParameter(name="env_width", size=1, col_1="env", col_2="env_width"),
            DatasetParameter(
                name="env_height", size=1, col_1="env", col_2="env_height"
            ),
            DatasetParameter(
                name="n_obstacles", size=1, col_1="env", col_2="n_obstacles"
            ),
            DatasetParameter(
                name="p_obstacles", size=1, col_1="env", col_2="p_obstacles"
            ),
            DatasetParameter(name="delta_0", size=1, col_1="misc", col_2="delta_0"),
            DatasetParameter(name="cost", size=1, col_1="misc", col_2="cost"),
            DatasetParameter(name="rel_l", size=1, col_1="misc", col_2="rel_l"),
            DatasetParameter(name="count", size=1, col_1="misc", col_2="count"),
            CalculatedParameter(
                name="rel_p",
                size=1,
                requires=["count"],
                to=lambda x: x,
                after_groupby=True,
            ),
        ],
        condition=False,
    )

    return param_set
