from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple

from torch import Tensor, float64, save
import torch
from torch.nn import Linear, Module, ModuleList, ReLU
from torch.nn.init import kaiming_uniform_

import diffmp
from .schedules import NoiseSchedule
# from diffmp.dynamics import get_dynamics, DynamicsBase
# from diffmp.torch import Loss
# from diffmp.utils import (
#     ParameterCalculated,
#     ParameterConditioning,
#     ParameterRegular,
#     input_output_size,
#     load_yaml,
# )


class Config(NamedTuple):
    dynamics: diffmp.dynamics.DynamicsBase
    timesteps: int
    problem: str
    n_hidden: int
    s_hidden: int
    regular: List[diffmp.utils.ParameterRegular]
    loss_fn: diffmp.torch.Loss
    dataset: Path
    denoising_steps: int
    batch_size: int
    lr: float
    noise_schedule: NoiseSchedule
    optimizer = torch.optim.Adam
    validation_split: float = 0.8
    conditioning: List[diffmp.utils.ParameterConditioning] = []

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        data = diffmp.utils.load_yaml(path)

        data["dynam:cs"] = diffmp.dynamics.get_dynamics(data["dynamics"])
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
        return cls(**data)


def main():
    config = Config(
        dynamics=diffmp.dynamics.get_dynamics("unicycle1_v0"),
        timesteps=1,
        problem="a",
        n_hidden=1,
        s_hidden=1,
        regular=[diffmp.utils.ParameterRegular.actions],
        loss_fn=diffmp.torch.Loss.mae,
        dataset=Path(""),
        denoising_steps=1,
        batch_size=1,
        lr=0.1,
        noise_schedule=NoiseSchedule.linear_scaled,
    )
    a = config.loss_fn.value(Tensor(), Tensor())


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
        self.optimizer = config.optimizer(self.parameters(), lr=self.config.lr)

        self.path: Path | None = None

        layers: List[Linear] = [Linear(self.in_size, self.s_hidden, dtype=float64)]

        for _ in range(self.n_hidden):
            layers.append(Linear(self.s_hidden, self.s_hidden, dtype=float64))
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
