from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math
from typing import Tuple, Dict, cast


@dataclass
class Obstacle(ABC):
    center: Tuple[float, float]

    @abstractmethod
    def area(self) -> float: ...


@dataclass
class BoxObstacle(Obstacle):
    size: Tuple[float, float]

    def area(self) -> float:
        return self.size[0] * self.size[1]


@dataclass
class CylinderObstacle(Obstacle):
    radius: float

    def area(self) -> float:
        return math.pi * self.radius**2


class ObstacleType(Enum):
    box = BoxObstacle
    cylinder = CylinderObstacle


def obstacle_from_dict(
    data: Dict[str, float | int | str | Tuple[int, int]],
) -> Obstacle:
    assert isinstance(data["type"], str)
    assert isinstance(data["center"], list) and len(data["center"]) == 2
    center = cast(Tuple[float, float], data["center"])

    obstacle_type = ObstacleType[data["type"]]
    match obstacle_type:
        case ObstacleType.box:
            assert isinstance(data["size"], list) and len(data["size"]) == 2
            size = cast(Tuple[float, float], data["size"])
            return ObstacleType.box.value(center, size)
        case ObstacleType.cylinder:
            assert isinstance(data["radius"], float | int)
            return ObstacleType.cylinder.value(center, data["radius"])
        case _:
            raise TypeError(f"Obstacle type {obstacle_type} does not exist")
