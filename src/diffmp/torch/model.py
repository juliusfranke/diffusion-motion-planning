from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor, float64, save
from torch.nn import Linear, Module, ModuleList, ReLU
from torch.nn.init import kaiming_uniform_
import yaml

import diffmp

if TYPE_CHECKING:
    from diffmp.utils.reporting import OptunaReporter, Reporter

from .schedules import NoiseSchedule


@dataclass
class CompositeConfig:
    dynamics: str
    models: List[Model]
    optuna: Optional[OptunaReporter] = None
    allocation: Optional[Dict[int, int]] = None

    @classmethod
    def from_yaml(cls, path: Path, load_if_exists: bool = True) -> CompositeConfig:
        data = diffmp.utils.load_yaml(path)
        return cls.from_dict(data, load_if_exists)

    @classmethod
    def from_dict(cls, data: Dict, load_if_exists: bool = True) -> CompositeConfig:
        dynamics = diffmp.dynamics.get_dynamics(data["dynamics"], 1).name
        models: List[Model] = []
        models_path = Path("data/models")
        for model_name in data["models"]:
            config_path = models_path / (model_name + ".yaml")
            weights_path = models_path / (model_name + ".pt")
            if load_if_exists and weights_path.exists():
                model = Model.load(Path(f"data/models/{model_name}"))
            elif config_path.exists():
                config = Config.from_yaml(config_path)
                model = Model(config)
                model.path = models_path / model_name
            else:
                raise Exception
            assert model.config.dynamics.name == dynamics
            models.append(model)
        assert len(models) == len(set(m.dynamics.timesteps for m in models))
        if data.get("allocation"):
            assert isinstance(data["allocation"], dict)
            return cls(dynamics=dynamics, models=models, allocation=data["allocation"])
        return cls(dynamics=dynamics, models=models)


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
    test_instances: List[diffmp.problems.Instance]
    weights: Dict[str, Dict[str, float]]
    optimizer: Any = torch.optim.Adam
    validation_split: float = 0.8

    def to_dict(self) -> Dict:
        config = {
            "dynamics": self.dynamics.name,
            "timesteps": self.timesteps,
            "problem": self.problem,
            "n_hidden": self.n_hidden,
            "s_hidden": self.s_hidden,
            "regular": [r.name for r in self.regular],
            "conditioning": [c.name for c in self.conditioning],
            "weights": self.weights,
            "loss_fn": self.loss_fn.name,
            "dataset": str(self.dataset),
            "denoising_steps": self.denoising_steps,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "noise_schedule": self.noise_schedule.name,
            "dataset_size": self.dataset_size,
            "reporters": [diffmp.utils.Reporters(type(r)).name for r in self.reporters],
            "validation_split": self.validation_split,
        }
        return config

    def to_yaml(self, path: Path) -> None:
        with open(path, "w") as file:
            yaml.safe_dump(self.to_dict(), file)

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        data = diffmp.utils.load_yaml(path)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict, name: Optional[str] = None) -> Config:
        required = []
        data["test_instances"] = [
            diffmp.problems.Instance.from_yaml(p)
            for p in Path(f"data/test_instances/{data['dynamics']}/").glob("*.yaml")
        ]
        for instance in data["test_instances"]:
            instance.results = []
            assert isinstance(instance.baseline, diffmp.problems.Baseline)

        data["dynamics"] = diffmp.dynamics.get_dynamics(
            data["dynamics"], data["timesteps"]
        )
        # weights = {}
        if data.get("weights"):
            assert isinstance(data["weights"], dict)
            for weight_col1, layer2 in data["weights"].items():
                assert isinstance(layer2, dict)
            #     for weight_col2, value in layer2.items():
            #         weights[(weight_col1, weight_col2)] = value
        # data["weights"] = weights
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
            kaiming_uniform_(layer.weight)  # pyright: ignore

        self.optimizer = config.optimizer(self.parameters(), lr=self.config.lr)

    def save(self):
        if isinstance(self.path, Path):
            config_path = self.path.parent / (self.path.name + ".yaml")
            weights_path = self.path.parent / (self.path.name + ".pt")
            save(self.state_dict(), weights_path)
            self.config.to_yaml(config_path)
        else:
            raise Exception("Model path was not set")

    @classmethod
    def load(cls, path: Path) -> Model:
        config_path = path.parent / (path.name + ".yaml")
        weights_path = path.parent / (path.name + ".pt")
        config = Config.from_yaml(config_path)
        model = cls(config)
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.path = path
        return model

    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self.linears) - 1):
            layer = self.linears[i]
            x = ReLU()(layer(x))
        out = self.linears[-1](x)
        return torch.Tensor(out)
