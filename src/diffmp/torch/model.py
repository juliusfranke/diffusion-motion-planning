from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import torch
import yaml
from torch import Tensor, float64, save
from torch.nn import Embedding, Linear, Module, ModuleList, ReLU, Identity
from torch.nn.init import kaiming_uniform_

import diffmp.dynamics as dy
import diffmp.problems as pb
import diffmp.torch as to
from diffmp.torch.env_encoder import EnvEncoder2D, EnvEncoder3D, ScaleEmbedding
import diffmp.utils as du

if TYPE_CHECKING:
    from diffmp.utils.reporting import OptunaReporter

from .schedules import NoiseSchedule


@dataclass
class DiscretizeConfig:
    method: Literal["sd", "percent"]
    resolution: int

    def scale(self, max_env_size: float) -> float:
        return max_env_size / self.resolution

    @classmethod
    def from_dict(cls, data: dict) -> DiscretizeConfig:
        assert data["method"] in ["sd", "percent"]
        return cls(data["method"], data["resolution"])

    def to_dict(self) -> dict:
        return {"method": self.method, "resolution": self.resolution}


@dataclass
class CompositeConfig:
    dynamics: str
    models: list[Model]
    optuna: Optional[OptunaReporter] = None
    allocation: Optional[dict[int, int]] = None

    @classmethod
    def from_yaml(cls, path: Path, load_if_exists: bool = True) -> CompositeConfig:
        data = du.load_yaml(path)
        return cls.from_dict(data, load_if_exists)

    @classmethod
    def from_dict(cls, data: dict, load_if_exists: bool = True) -> CompositeConfig:
        n_robots = data["n_robots"]
        dynamics = dy.get_dynamics(data["dynamics"], 1, n_robots).name
        models: list[Model] = []
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


@dataclass
class Config:
    dynamics: dy.DynamicsBase
    timesteps: int
    n_robots: int
    problem: str
    n_hidden: int
    s_hidden: int
    regular: list[du.DatasetParameter | du.CalculatedParameter]
    conditioning: list[du.DatasetParameter | du.CalculatedParameter]
    loss_fn: to.Loss
    dataset: Path
    denoising_steps: int
    batch_size: int
    lr: float
    noise_schedule: NoiseSchedule
    dataset_size: int
    reporters: list[du.Reporter]
    test_instances: list[pb.Instance]
    weights: dict[str, dict[str, float]]
    optimizer: Any = torch.optim.Adam
    validation_split: float = 0.8
    discretize: Optional[DiscretizeConfig] = None
    robot_embedding: bool = True

    def to_dict(self) -> dict:
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
            "reporters": [du.Reporters(type(r)).name for r in self.reporters],
            "validation_split": self.validation_split,
        }
        if isinstance(self.discretize, DiscretizeConfig):
            config["discretize"] = self.discretize.to_dict()

        return config

    def to_yaml(self, path: Path) -> None:
        with open(path, "w") as file:
            yaml.safe_dump(self.to_dict(), file)

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        data = du.load_yaml(path)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict, name: Optional[str] = None) -> Config:
        required = []
        n_robots = data["n_robots"]
        if n_robots > 1:
            path = Path(f"data/test_instances/{data['dynamics']}_x{n_robots}/")
        else:
            path = Path(f"data/test_instances/{data['dynamics']}/")
        data["test_instances"] = [pb.Instance.from_yaml(p) for p in path.glob("*.yaml")]
        for instance in data["test_instances"]:
            instance.results = []
            assert isinstance(instance.baseline, pb.Baseline)

        print(n_robots)
        data["dynamics"] = dy.get_dynamics(
            data["dynamics"], data["timesteps"], n_robots
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
            if not isinstance(param, du.CalculatedParameter):
                continue
            for req in param.requires:
                required.append(data["dynamics"].parameter_set[req])

        data["regular"] = reg_params

        if "conditioning" in data.keys():
            cond_params = []
            for param_str in data["conditioning"]:
                param = data["dynamics"].parameter_set[param_str]
                cond_params.append(param)
                if not isinstance(param, du.CalculatedParameter):
                    continue
                for req in param.requires:
                    required.append(data["dynamics"].parameter_set[req])

            data["conditioning"] = cond_params
        else:
            data["conditioning"] = []

        new_param_set = du.ParameterSet()
        new_param_set.add_parameters(data["regular"], condition=False)
        new_param_set.add_parameters(data["conditioning"], condition=True)
        new_param_set.required = [
            req for req in required if req not in data["regular"] + data["conditioning"]
        ]
        data["dynamics"].parameter_set = new_param_set

        data["reporters"] = [du.Reporters[rep].value() for rep in data["reporters"]]
        data["noise_schedule"] = NoiseSchedule[data["noise_schedule"]]
        data["loss_fn"] = to.Loss[data["loss_fn"]]
        data["dataset"] = Path(data["dataset"])

        if "discretize" in data.keys():
            data["discretize"] = DiscretizeConfig.from_dict(data["discretize"])

        return cls(**data)


class Model(Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dynamics = config.dynamics
        self.regular = config.regular
        self.conditioning = config.conditioning

        self.out_size = sum([len(param.cols) for param in self.regular])
        self.cond_size = sum([len(param.cols) for param in self.conditioning])
        self.in_size = self.out_size + self.cond_size + 1

        self.s_hidden = config.s_hidden
        self.n_hidden = config.n_hidden

        self.loss_fn = config.loss_fn.value
        self.noise_schedule = config.noise_schedule.value

        self.path: Path | None = None

        if config.robot_embedding:
            emb_size = 8
            self.robot_embedding = Embedding(config.n_robots, embedding_dim=emb_size)
            self.in_size += emb_size
        else:
            self.robot_embedding = None

        if config.discretize is None:
            self.env_encoder = None
            self.scale_embedding = None
        else:
            # TODO Get number of scales from dataset!!
            scale_emb_size = 1
            res = config.discretize.resolution
            env_emb_size = res
            # self.in_size += env_emb_size + scale_emb_size
            self.in_size += env_emb_size
            self.scale_embedding = ScaleEmbedding(1, scale_emb_size)
            if self.dynamics.dim == pb.Dim.TWO_D:
                self.env_encoder = EnvEncoder2D(1, env_emb_size)
            else:
                self.env_encoder = EnvEncoder3D(1, env_emb_size)

        layers: list[Linear] = [Linear(self.in_size, self.s_hidden, dtype=float64)]

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

    def forward(
        self,
        x: Tensor,
        env_grid: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        robot_id: Optional[torch.Tensor] = None,
    ) -> Tensor:
        if self.env_encoder is not None:
            # assert self.scale_embedding is not None
            assert scale is not None
            assert env_grid is not None
            env_emb = self.env_encoder(env_grid)
            # scale_emb = self.scale_embedding(scale)
            # x = torch.concat([x, env_emb, scale_emb], dim=-1)
            x = torch.concat([x, env_emb], dim=-1)

        if self.robot_embedding is not None:
            assert robot_id is not None
            robot_emb = self.robot_embedding(robot_id)
            x = torch.concat([x, robot_emb], dim=-1)

        for i in range(len(self.linears) - 1):
            layer = self.linears[i]
            x = ReLU()(layer(x))
        out = self.linears[-1](x)
        return torch.Tensor(out)
