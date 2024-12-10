from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Type

from torch import Tensor, float64, save
from torch.nn import Linear, Module, ModuleList, ReLU
from torch.nn.init import kaiming_uniform_

from diffmp.dynamics.base import DynamicsBase
from diffmp.dynamics.unicycle1 import UnicycleFirstOrder
from diffmp.torch.loss import mse
from diffmp.utils.data import (
    ParameterRegular,
    ParameterConditioning,
    ParameterCalculated,
    input_output_size,
)


class Config(NamedTuple):
    dynamics: DynamicsBase
    timesteps: int
    problem: str
    n_hidden: int
    s_hidden: int
    regular: List[ParameterRegular]
    loss_fn: Callable[[Tensor, Tensor], Tensor]
    dataset: Path
    denosing_steps: int
    batch_size: int
    lr: float
    conditioning: List[ParameterConditioning] = []
    calculated: List[ParameterCalculated] = []


config = Config(
    dynamics=UnicycleFirstOrder(
        min_vel=0.1, max_vel=0.5, min_angular_vel=0.1, max_angular_vel=0.5, dt=0.1
    ),
    timesteps=1,
    problem="a",
    n_hidden=1,
    s_hidden=1,
    regular=[ParameterRegular.actions],
    loss_fn=mse,
    dataset=Path(""),
    denosing_steps=1,
    batch_size=1,
    lr=0.1,
)


class Model(Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dynamics = config.dynamics
        self.regular = config.regular
        self.conditioning = config.conditioning

        self.in_size, self.out_size = input_output_size(config)
        self.cond_size = self.in_size - self.out_size

        self.s_hidden = config.s_hidden
        self.n_hidden = config.n_hidden

        self.loss_fn = config.loss_fn

        self.path: Path | None = None

        layers: List[Linear] = [Linear(self.in_size, self.s_hidden, dtype=float64)]

        for _ in range(self.n_hidden):
            layers.append(Linear(self.s_hidden, self.n_hidden, dtype=float64))
        layers.append(Linear(self.s_hidden, self.out_size, dtype=float64))

        self.linears = ModuleList(layers)

        for layer in self.linears:
            kaiming_uniform_(layer.weight)

    def save(self):
        if isinstance(self.path, Path):
            save(self.state_dict(), self.path)
        else:
            raise Exception

    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self.linears) - 1):
            layer = self.linears[i]
            x = ReLU()(layer(x))
        return self.linears[-1](x)
