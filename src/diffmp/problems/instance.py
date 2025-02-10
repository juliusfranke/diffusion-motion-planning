from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import diffmp
from diffmp.problems.obstacle import Bounds2D

from .environment import Environment
from .robots import Robot

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class Instance:
    environment: Environment
    robots: List[Robot]
    data: Optional[Dict] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "environment": self.environment.to_dict(),
            "robots": [r.to_dict() for r in self.robots],
        }

    @classmethod
    def random(
        cls,
        min_size: int,
        max_size: int,
        n_obstacles: int,
        bounds_obstacle_size: Bounds2D,
        robot_types: List[str],
    ) -> Instance:
        env = Environment.random(min_size, max_size, n_obstacles, bounds_obstacle_size)
        robots = [Robot.random(env, dyn) for dyn in robot_types]
        return cls(environment=env, robots=robots)

    @classmethod
    def from_dict(cls, data: Dict[Any, Any], name: Optional[str] = None) -> Instance:
        env = Environment.from_dict(data["environment"])
        robots = [Robot.from_dict(robot_data) for robot_data in data["robots"]]
        return cls(env, robots, data, name)

    @classmethod
    def from_yaml(cls, path: Path) -> Instance:
        data = diffmp.utils.load_yaml(path)
        name = path.stem
        return cls.from_dict(data, name=name)
