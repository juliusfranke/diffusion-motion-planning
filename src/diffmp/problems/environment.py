from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast
import numpy as np

from .obstacle import Bounds2D, BoxObstacle, Obstacle, obstacle_from_dict


@dataclass
class Environment:
    def __init__(
        self,
        obstacles: List[Obstacle],
        min: Tuple[float, float],
        max: Tuple[float, float],
    ):
        self.obstacles = obstacles
        self.min = min
        self.max = max
        self.env_width = max[0] - min[0]
        self.env_height = max[1] - min[1]
        self.area = self.env_width * self.env_height

        # assumes non overlapping obstacles
        self.area_blocked: float = sum([o.area() for o in self.obstacles])
        self.area_free = self.area - self.area_blocked

        self.n_obstacles = len(self.obstacles)
        self.p_obstacles = self.area_blocked / self.area

    def to_dict(self) -> Dict:
        return {
            "min": self.min,
            "max": self.max,
            "obstacles": [o.to_dict() for o in self.obstacles],
        }

    def random_free(self) -> Tuple[float, float]:
        while True:
            x = np.random.random() * (self.max[0] - self.min[0]) + self.min[0]
            y = np.random.random() * (self.max[1] - self.min[1]) + self.min[1]
            is_free = True
            for obstacle in self.obstacles:
                if obstacle.is_inside(x=x, y=y):
                    is_free = False
                    break
            if not is_free:
                continue
            break

        return (x, y)

    @classmethod
    def from_dict(
        cls, data: Dict[str, List[float] | List[int | Dict[str, Any]]]
    ) -> Environment:
        assert isinstance(data["min"], list) and len(data["min"]) == 2
        assert isinstance(data["max"], list) and len(data["max"]) == 2
        min = cast(Tuple[float, float], data["min"])
        max = cast(Tuple[float, float], data["max"])

        assert isinstance(data["obstacles"], list)

        obstacles = [
            obstacle_from_dict(obstacle)
            for obstacle in data["obstacles"]
            if isinstance(obstacle, dict)
        ]

        return cls(obstacles=obstacles, min=min, max=max)

    @classmethod
    def random(
        cls,
        min_size: int,
        max_size: int,
        n_obstacles: int,
        bounds_obstacle_size: Bounds2D,
    ) -> Environment:
        x_max = int(np.random.random() * (max_size - min_size) + min_size)
        y_max = int(np.random.random() * (max_size - min_size) + min_size)
        bounds_environment = Bounds2D(0, x_max, 0, y_max)
        obstacles = [
            BoxObstacle.random(bounds_environment, bounds_obstacle_size)
            for _ in range(n_obstacles)
        ]
        return cls(obstacles=obstacles, min=(0, 0), max=(x_max, y_max))
