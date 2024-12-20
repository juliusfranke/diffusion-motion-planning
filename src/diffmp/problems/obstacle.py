from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math
from typing import Tuple, Dict


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
    assert isinstance(data["center"], tuple)
    obstacle_type = ObstacleType[data["type"]]
    match obstacle_type:
        case ObstacleType.box:
            assert isinstance(data["size"], tuple)
            return ObstacleType.box.value(data["center"], data["size"])
        case ObstacleType.cylinder:
            assert isinstance(data["radius"], float | int)
            return ObstacleType.cylinder.value(data["center"], data["radius"])
        case _:
            raise TypeError(f"Obstacle type {obstacle_type} does not exist")
