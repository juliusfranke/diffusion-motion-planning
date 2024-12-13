from pathlib import Path
from typing import Dict, Tuple


import yaml
import diffmp
from .config import DynamicFactor, ParameterData
from torch.utils.data import TensorDataset


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError()
    if path.is_dir():
        raise IsADirectoryError()
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data


def load_data(dataset: Path, **kwargs) -> TensorDataset:
    return TensorDataset()


def param_size(param: ParameterData, config: diffmp.torch.Config):
    size = param.static_size
    match param.dynamic_factor:
        case DynamicFactor.u:
            size += config.timesteps * len(config.dynamics.u)
        case DynamicFactor.q:
            size += config.timesteps * len(config.dynamics.q)
        case DynamicFactor._:
            pass
        case _:
            raise NotImplementedError(f"{param.dynamic_factor} not implemented")

    return size


def input_output_size(config: diffmp.torch.Config) -> Tuple[int, int]:
    in_size = 0
    out_size = 0
    for regular in config.regular:
        out_size += param_size(regular.value, config)

    for conditioning in config.conditioning:
        in_size += param_size(conditioning.value, config)

    in_size += out_size + 1

    return (in_size, out_size)
