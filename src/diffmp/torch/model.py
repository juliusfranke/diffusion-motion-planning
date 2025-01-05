from __future__ import annotations

from pathlib import Path
from typing import Any, List, NamedTuple

import torch
from torch import Tensor, float64, save
from torch.nn import Linear, Module, ModuleList, ReLU
from torch.nn.init import kaiming_uniform_

import diffmp

from .schedules import NoiseSchedule


class Config(NamedTuple):
    dynamics: diffmp.dynamics.DynamicsBase
    timesteps: int
    problem: str
    n_hidden: int
    s_hidden: int
    regular: List[diffmp.utils.ParameterRegular]
    conditioning: List[diffmp.utils.ParameterConditioning]
    loss_fn: diffmp.torch.Loss
    dataset: Path
    denoising_steps: int
    batch_size: int
    lr: float
    noise_schedule: NoiseSchedule
    dataset_size: int
    optimizer: Any = torch.optim.Adam
    validation_split: float = 0.8

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        data = diffmp.utils.load_yaml(path)

        data["dynamics"] = diffmp.dynamics.get_dynamics(data["dynamics"])
        data["regular"] = [
            diffmp.utils.ParameterRegular[reg] for reg in data["regular"]
        ]
        data["loss_fn"] = diffmp.torch.Loss[data["loss_fn"]]
        data["dataset"] = Path(data["dataset"])
        if "conditioning" in data.keys():
            data["conditioning"] = [
                diffmp.utils.ParameterConditioning[cond]
                for cond in data["conditioning"]
            ]
        else:
            data["conditioning"] = []
        return cls(**data)


class Model(Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dynamics = config.dynamics
        self.regular = config.regular
        self.conditioning = config.conditioning

        self.in_size, self.out_size = diffmp.utils.input_output_size(config)
        self.cond_size = self.in_size - self.out_size

        self.s_hidden = config.s_hidden
        self.n_hidden = config.n_hidden

        self.loss_fn = config.loss_fn.value
        self.noise_schedule = config.noise_schedule.value

        self.path: Path | None = None

        layers: List[Linear] = [Linear(self.in_size, self.s_hidden, dtype=float64)]

        for _ in range(self.n_hidden):
            layers.append(Linear(self.s_hidden, self.s_hidden, dtype=float64))
        layers.append(Linear(self.s_hidden, self.out_size, dtype=float64))

        self.linears = ModuleList(layers)

        for layer in self.linears:
            kaiming_uniform_(layer.weight)

        self.optimizer = config.optimizer(self.parameters(), lr=self.config.lr)

    def save(self):
        if isinstance(self.path, Path):
            save(self.state_dict(), self.path)
        else:
            raise Exception

    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self.linears) - 1):
            layer = self.linears[i]
            x = ReLU()(layer(x))
        out = self.linears[-1](x)
        return torch.Tensor(out)
