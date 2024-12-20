from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .obstacle import Obstacle, obstacle_from_dict


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

    @classmethod
    def from_dict(
        cls, data: Dict[str, Tuple[float, float] | List[int | Dict[str, Any]]]
    ) -> Environment:
        assert isinstance(data["min"], tuple)
        assert isinstance(data["max"], tuple)
        assert isinstance(data["obstacles"], list)
        min = data["min"]
        max = data["max"]
        obstacles = [
            obstacle_from_dict(obstacle)
            for obstacle in data["obstacles"]
            if isinstance(obstacle, dict)
        ]

        return cls(obstacles=obstacles, min=min, max=max)
