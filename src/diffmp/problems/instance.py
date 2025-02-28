from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import yaml

import diffmp
from diffmp.problems.obstacle import Bounds2D

from .environment import Environment
from .robots import Robot

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class Baseline:
    success: float
    duration: float
    cost: float

    def to_dict(self) -> Dict:
        return {"success": self.success, "duration": self.duration, "cost": self.cost}

    @classmethod
    def from_dict(cls, data: Dict) -> Baseline:
        return cls(data["success"], data["duration"], data["cost"])


@dataclass
class Instance:
    environment: Environment
    robots: List[Robot]
    data: Optional[Dict] = None
    name: Optional[str] = None
    baseline: Optional[Baseline] = None
    results: Optional[List[Baseline]] = None

    def plot(self, ax: Optional[Axes] = None) -> None:
        if ax is None:
            fig, ax = plt.subplots(1)
        assert isinstance(ax, Axes)
        ax.set_aspect("equal")
        self.environment.plot(ax)
        self.robots[0].plot(ax)
        ax.axis("off")

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as file:
            yaml.safe_dump(self.to_dict(), file, default_flow_style=None)

    def to_dict(self) -> Dict:
        data = {
            "environment": self.environment.to_dict(),
            "robots": [r.to_dict() for r in self.robots],
        }
        if isinstance(self.baseline, Baseline):
            data["baseline"] = self.baseline.to_dict()
        return data

    @classmethod
    def random(
        cls,
        min_size: int,
        max_size: int,
        n_obstacles_min: int,
        p_obstacles: float,
        robot_types: List[str],
    ) -> Instance:
        env = Environment.random(min_size, max_size, n_obstacles_min, p_obstacles)
        robots = [Robot.random(env, dyn) for dyn in robot_types]
        if None in robots:
            return cls.random(
                min_size, max_size, n_obstacles_min, p_obstacles, robot_types
            )
        name = str(uuid4())
        return cls(environment=env, robots=robots, name=name)

    @classmethod
    def from_dict(cls, data: Dict[Any, Any], name: Optional[str] = None) -> Instance:
        env = Environment.from_dict(data["environment"])
        robots = [Robot.from_dict(robot_data) for robot_data in data["robots"]]
        if data.get("baseline"):
            baseline = Baseline.from_dict(data["baseline"])
            return cls(env, robots, data, name, baseline)
        return cls(env, robots, data, name)

    @classmethod
    def from_yaml(cls, path: Path) -> Instance:
        data = diffmp.utils.load_yaml(path)
        name = path.stem
        return cls.from_dict(data, name=name)
