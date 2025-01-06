from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

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
