from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast, Optional
import numpy as np
from shapely import union_all, difference, intersection, Polygon

from .obstacle import Bounds2D, BoxObstacle, Obstacle, obstacle_from_dict


def area_blocked(
    min: Tuple[float, float], max: Tuple[float, float], obstacles: List[Obstacle]
) -> float:
    geom_env = Polygon(
        (
            (min[0], min[1]),
            (max[0], min[1]),
            (max[0], max[1]),
            (min[0], max[1]),
        )
    )
    geom_obstacles = union_all([o.geometry() for o in obstacles])
    geom_obstacles_in_env = intersection(geom_obstacles, geom_env)
    return geom_obstacles_in_env.area


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

        self.area_blocked = area_blocked(self.min, self.max, self.obstacles)
        self.area_free = self.area - self.area_blocked

        self.n_obstacles = len(self.obstacles)
        self.p_obstacles = self.area_blocked / self.area

    def to_dict(self) -> Dict:
        return {
            "min": list(self.min),
            "max": list(self.max),
            "obstacles": [o.to_dict() for o in self.obstacles],
        }

    def random_free(self, clearance: float = 0.0) -> Optional[Tuple[float, float]]:
        max_tries = 1000
        for _ in range(max_tries):
            x = np.random.random() * (self.max[0] - self.min[0]) + self.min[0]
            y = np.random.random() * (self.max[1] - self.min[1]) + self.min[1]
            is_free = True
            for obstacle in self.obstacles:
                if obstacle.is_inside(x=x, y=y, clearance=clearance):
                    is_free = False
                    break
            if not is_free:
                continue
            break
        else:
            return None

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
        n_obstacles_min: int,
        p_obstacles: float,
    ) -> Environment:
        assert 0 <= p_obstacles <= 1
        x_max = int(np.random.random() * (max_size - min_size) + min_size)
        y_max = int(np.random.random() * (max_size - min_size) + min_size)
        bounds_environment = Bounds2D(0, x_max, 0, y_max)
        env_area = x_max * y_max
        obstacle_max_area = env_area / n_obstacles_min
        blocked_goal = env_area * p_obstacles
        obstacles = []
        count = 0
        max_tries = n_obstacles_min * 10
        while True:
            if count >= max_tries:
                return cls.random(min_size, max_size, n_obstacles_min, p_obstacles)
            count += 1
            obstacle = BoxObstacle.random(
                bounds_environment=bounds_environment, max_area=obstacle_max_area
            )
            obstacles.append(obstacle)
            area_b = area_blocked((0, 0), (x_max, y_max), obstacles)
            if area_b > blocked_goal or np.isclose(area_b, blocked_goal, atol=0.1):
                break
            obstacle_max_area = min(blocked_goal - area_b, obstacle_max_area)
        return cls(obstacles=obstacles, min=(0, 0), max=(x_max, y_max))
