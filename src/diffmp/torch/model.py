from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, NamedTuple

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
    regular: List[diffmp.utils.DatasetParameter | diffmp.utils.CalculatedParameter]
    conditioning: List[diffmp.utils.DatasetParameter | diffmp.utils.CalculatedParameter]
    loss_fn: diffmp.torch.Loss
    dataset: Path
    denoising_steps: int
    batch_size: int
    lr: float
    noise_schedule: NoiseSchedule
    dataset_size: int
    reporters: List[diffmp.utils.Reporter]
    optimizer: Any = torch.optim.Adam
    validation_split: float = 0.8

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        data = diffmp.utils.load_yaml(path)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict) -> Config:
        required = []
        data["dynamics"] = diffmp.dynamics.get_dynamics(data["dynamics"], 5)
        reg_params = []
        for param_str in data["regular"]:
            param = data["dynamics"].parameter_set[param_str]
            reg_params.append(param)
            if not isinstance(param, diffmp.utils.CalculatedParameter):
                continue
            for req in param.requires:
                required.append(data["dynamics"].parameter_set[req])

        data["regular"] = reg_params

        if "conditioning" in data.keys():
            cond_params = []
            for param_str in data["conditioning"]:
                param = data["dynamics"].parameter_set[param_str]
                cond_params.append(param)
                if not isinstance(param, diffmp.utils.CalculatedParameter):
                    continue
                for req in param.requires:
                    required.append(data["dynamics"].parameter_set[req])

            data["conditioning"] = cond_params
        else:
            data["conditioning"] = []

        new_param_set = diffmp.utils.ParameterSet()
        new_param_set.add_parameters(data["regular"], condition=False)
        new_param_set.add_parameters(data["conditioning"], condition=True)
        new_param_set.required = [
            req for req in required if req not in data["regular"] + data["conditioning"]
        ]
        data["dynamics"].parameter_set = new_param_set

        data["reporters"] = [
            diffmp.utils.Reporters[rep].value() for rep in data["reporters"]
        ]
        data["noise_schedule"] = diffmp.torch.NoiseSchedule[data["noise_schedule"]]
        data["loss_fn"] = diffmp.torch.Loss[data["loss_fn"]]
        data["dataset"] = Path(data["dataset"])
        return cls(**data)


class Model(Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dynamics = config.dynamics
        self.regular = config.regular
        self.conditioning = config.conditioning

        self.out_size = sum([param.size for param in self.regular])
        self.cond_size = sum([param.size for param in self.conditioning])
        self.in_size = self.out_size + self.cond_size + 1

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
